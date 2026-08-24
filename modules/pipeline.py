from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum

from modules.conflict import (
    Conflict,
    ConflictStatus,
    detect_conflicts,
)
from modules.evidence import (
    EvidenceResult,
    build_reverse_reference_index,
    classify_candidates,
    evidence_results_to_clauses,
    expand_for_conflict_check,
    is_sufficient,
    normalize_for_comparison,
)
from modules.evidence_model import EvidenceModel
from modules.generation import generate_answer
from modules.grounding import GroundingValidationResult, validate_grounding
from modules.retriever import load_knowledge_base, semantic_retrieve, expand_cross_references
from modules.policy_versioning import (
    PolicyVersion,
    TemporalContext,
    build_effective_kb,
    changed_clause_ids,
    resolve_temporal_context,
)

#: Canonical evaluation order for temporal branches.
BRANCH_ORDER: tuple[PolicyVersion, ...] = (
    PolicyVersion.PRE_AMENDMENT,
    PolicyVersion.POST_AMENDMENT,
)


class PipelineOutcome(str, Enum):
    ANSWER = "ANSWER"
    NO_EVIDENCE = "NO_EVIDENCE"
    CONFLICT = "CONFLICT"
    TEMPORAL_AMBIGUITY = "TEMPORAL_AMBIGUITY"


@dataclass
class PipelineResult:
    """
    Complete structured result of the ClauseGuard safety pipeline.
    """

    outcome: PipelineOutcome
    question: str
    answer: str = ""
    citations: list[str] = field(default_factory=list)
    refusal_reason: str = ""
    evidence_results: list[EvidenceResult] = field(default_factory=list)
    retained_evidence: list[EvidenceResult] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)
    structural_clauses: list[dict] = field(default_factory=list)
    grounding_validation: GroundingValidationResult | None = None

    # Populated only when outcome is TEMPORAL_AMBIGUITY. Each branch owns
    # its own evidence, conflicts and grounding; nothing is merged.
    branch_results: dict[PolicyVersion, PipelineResult] = field(
        default_factory=dict,
    )


def _run_branch(
    question: str,
    model: EvidenceModel,
    knowledge_base: list[dict],
    candidates: list[dict]
) -> PipelineResult:
    """
    Runs the existing ClauseGuard downstream pipeline against ONE
    internally consistent effective policy state.

    This function is temporal-agnostic: it simply receives the effective
    knowledge base for a single policy version plus candidate clauses
    retrieved from THAT SAME knowledge base.
    """
    reverse_index = build_reverse_reference_index(knowledge_base)

    # 2. Evidence Classification
    evidence_results = classify_candidates(
        question=question,
        candidates=candidates,
        model=model,
    )

    # 3. Sufficiency Check
    sufficient, retained_evidence = is_sufficient(evidence_results)

    if not sufficient:
        return PipelineResult(
            outcome=PipelineOutcome.NO_EVIDENCE,
            question=question,
            refusal_reason=(
                "The policy manual does not contain sufficient verified evidence "
                "to answer this question. Please direct the query to a supervisor "
                "or policy specialist."
            ),
            evidence_results=evidence_results,
            retained_evidence=[],
        )

    # 4. Restore KB clauses & Structural expansion for conflict
    retained_clauses = evidence_results_to_clauses(
        retained_evidence,
        knowledge_base,
    )

    structural_clauses = expand_for_conflict_check(
        retained_clauses,
        knowledge_base,
        reverse_index,
    )

    # 5. Conflict Analysis
    conflicts = detect_conflicts(
        evidence=retained_evidence,
        structural_clauses=structural_clauses,
        model=model,
    )

    active_conflicts = [
        c for c in conflicts
        if c.status in (ConflictStatus.CONFIRMED, ConflictStatus.UNRESOLVED)
    ]

    if active_conflicts:
        primary = active_conflicts[0]
        refusal_msg = (
            f"A policy contradiction was detected between clause {primary.clause_a} "
            f"and clause {primary.clause_b}. {primary.reasoning} "
            "This question cannot be settled from the manual alone; "
            "please direct the query to a supervisor."
        )

        return PipelineResult(
            outcome=PipelineOutcome.CONFLICT,
            question=question,
            refusal_reason=refusal_msg,
            evidence_results=evidence_results,
            retained_evidence=retained_evidence,
            conflicts=active_conflicts,
            structural_clauses=structural_clauses,
        )

    # 6. Answer Generation
    gen_result = generate_answer(
        question=question,
        evidence=retained_evidence,
        model=model,
    )

    # 7. Grounding Validation
    val_result = validate_grounding(
        answer=gen_result.answer,
        citations=gen_result.citations,
        retained_evidence=retained_evidence,
    )

    if not val_result.is_valid:
        return PipelineResult(
            outcome=PipelineOutcome.NO_EVIDENCE,
            question=question,
            refusal_reason=(
                f"Generated answer failed grounding validation: {val_result.reasoning}"
            ),
            evidence_results=evidence_results,
            retained_evidence=retained_evidence,
            conflicts=[],
            structural_clauses=structural_clauses,
            grounding_validation=val_result,
        )

    return PipelineResult(
        outcome=PipelineOutcome.ANSWER,
        question=question,
        answer=gen_result.answer,
        citations=gen_result.citations,
        evidence_results=evidence_results,
        retained_evidence=retained_evidence,
        conflicts=[],
        structural_clauses=structural_clauses,
        grounding_validation=val_result,
    )


# ---------------------------------------------------------------------------
# Temporal plumbing
#
# The temporal layer sits ABOVE the existing pipeline. It decides which
# effective policy state(s) to evaluate; each state is then processed by
# the unchanged ClauseGuard pipeline with its own retrieval, evidence,
# conflict analysis, generation and grounding. Evidence never crosses
# between branches.
# ---------------------------------------------------------------------------


def _retrieve_for_view(
    question: str,
    view: list[dict],
    top_k: int,
) -> list[dict]:
    """Retrieve candidates against ONE effective policy view."""

    semantic_candidates = semantic_retrieve(
        question,
        view,
        top_k=top_k,
    )

    return expand_cross_references(
        semantic_candidates,
        view,
    )


def _candidates_from_pool(
    pool: list[dict],
    view: list[dict],
) -> list[dict]:
    """
    Map caller-supplied candidates onto an effective policy view.

    Clauses that do not exist in the view (e.g. a clause inserted by the
    amendment, evaluated against the historical state) are dropped rather
    than invented. Effective clause text always comes from the view.
    """

    by_id = {clause["id"]: clause for clause in view}

    mapped: list[dict] = []

    for candidate in pool:
        clause = by_id.get(candidate["id"])

        if clause is None:
            continue

        merged = {**clause}

        for key in ("similarity", "retrieval_reason"):
            if key in candidate:
                merged[key] = candidate[key]

        mapped.append(merged)

    return mapped


def _answers_agree(first: PipelineResult, second: PipelineResult) -> bool:
    """
    Deterministically decide whether two ANSWER results are materially
    identical (same normalized answer text and same citation set).
    """

    return (
        normalize_for_comparison(first.answer)
        == normalize_for_comparison(second.answer)
        and set(first.citations) == set(second.citations)
    )


def _aggregate_branches(
    question: str,
    branch_results: dict[PolicyVersion, PipelineResult],
) -> PipelineResult:
    """
    Combine independently evaluated policy-version branches.

    TEMPORAL_AMBIGUITY is deliberately narrow: it is returned only when
    the branches do NOT already agree on a single safe outcome. When both
    versions produce the same outcome (and, for answers, materially the
    same grounded content), that shared outcome is returned directly — a
    difference between policy VERSIONS is never treated as a CONFLICT.

    The pre-amendment branch is used as the canonical representative when
    branches agree, because the base manual is the canonical corpus.
    """

    pre = branch_results.get(PolicyVersion.PRE_AMENDMENT)
    post = branch_results.get(PolicyVersion.POST_AMENDMENT)

    if pre is not None and post is not None:
        if pre.outcome == post.outcome:

            if pre.outcome == PipelineOutcome.ANSWER:
                if _answers_agree(pre, post):
                    return pre
            else:
                # Both NO_EVIDENCE or both CONFLICT: the refusal holds
                # regardless of which policy version applies.
                return pre

    return PipelineResult(
        outcome=PipelineOutcome.TEMPORAL_AMBIGUITY,
        question=question,
        refusal_reason=(
            "The applicable policy depends on temporal context that was "
            "not provided, and the relevant policy versions lead to "
            "different outcomes. Provide the relevant date for a "
            "definitive answer."
        ),
        branch_results=branch_results,
    )


def run_pipeline(
    question: str,
    model: EvidenceModel,
    knowledge_base: list[dict] | None = None,
    top_k: int = 8,
    pre_retrieved_candidates: list[dict] | None = None,
    explicit_date: date | None = None,
) -> PipelineResult:
    """
    Run the end-to-end ClauseGuard query pipeline.

    Temporal resolution happens FIRST and deterministically (no LLM):

        1. explicit_date wins when supplied;
        2. otherwise dates extracted from the question;
        3. otherwise the question may be evaluated against both the
           pre- and post-amendment policy states when it touches
           amendment-changed provisions (undated questions never guess
           today's policy).

    Contradictory temporal inputs raise ValueError; callers (e.g. the
    CLI) are expected to pre-validate with resolve_temporal_context().
    """

    if knowledge_base is None:
        raw_knowledge_base = load_knowledge_base()
    else:
        raw_knowledge_base = knowledge_base

    temporal_resolution = resolve_temporal_context(
        question,
        explicit_date=explicit_date,
    )

    if temporal_resolution.context == TemporalContext.INPUT_CONFLICT:
        raise ValueError(
            "Conflicting temporal input: "
            + " ".join(temporal_resolution.notes)
        )

    # ------------------------------------------------------------------
    # EXPLICIT: exactly one effective policy state is selected.
    # ------------------------------------------------------------------
    if temporal_resolution.context == TemporalContext.EXPLICIT:
        version = temporal_resolution.versions[0]

        effective_kb = build_effective_kb(raw_knowledge_base, version)

        if pre_retrieved_candidates is not None:
            candidates = _candidates_from_pool(
                pre_retrieved_candidates,
                effective_kb,
            )
        else:
            candidates = _retrieve_for_view(
                question,
                effective_kb,
                top_k=top_k,
            )

        return _run_branch(
            question=question,
            model=model,
            knowledge_base=effective_kb,
            candidates=candidates,
        )

    # ------------------------------------------------------------------
    # ABSENT: no explicit temporal information.
    #
    # Probe deterministically (BM25 is local and free) whether the
    # question even reaches amendment-changed provisions. Ordinary
    # questions keep flowing through the normal single-version
    # pipeline with zero temporal complexity.
    #
    # Amendment-sensitive questions NEVER silently pick today's policy:
    # both effective states are evaluated fully and independently —
    # each with its OWN effective KB, retrieval, evidence, sufficiency,
    # structural expansion, conflict analysis, generation and grounding.
    # ------------------------------------------------------------------

    probe_view = build_effective_kb(
        raw_knowledge_base,
        PolicyVersion.PRE_AMENDMENT,
    )

    probe_candidates = (
        _candidates_from_pool(pre_retrieved_candidates, probe_view)
        if pre_retrieved_candidates is not None
        else _retrieve_for_view(question, probe_view, top_k=top_k)
    )

    amendment_sensitive = bool(
        {clause["id"] for clause in probe_candidates}
        & changed_clause_ids(raw_knowledge_base)
    )

    if not amendment_sensitive:
        return _run_branch(
            question=question,
            model=model,
            knowledge_base=probe_view,
            candidates=probe_candidates,
        )

    branch_results: dict[PolicyVersion, PipelineResult] = {}

    for version in BRANCH_ORDER:
        effective_kb = build_effective_kb(raw_knowledge_base, version)

        candidates = (
            _candidates_from_pool(pre_retrieved_candidates, effective_kb)
            if pre_retrieved_candidates is not None
            else _retrieve_for_view(question, effective_kb, top_k=top_k)
        )

        branch_results[version] = _run_branch(
            question=question,
            model=model,
            knowledge_base=effective_kb,
            candidates=candidates,
        )

    return _aggregate_branches(question, branch_results)

from __future__ import annotations

from dataclasses import dataclass, field
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
)
from modules.evidence_model import EvidenceModel
from modules.generation import generate_answer
from modules.grounding import GroundingValidationResult, validate_grounding
from modules.retriever import load_knowledge_base, semantic_retrieve, expand_cross_references


class PipelineOutcome(str, Enum):
    ANSWER = "ANSWER"
    NO_EVIDENCE = "NO_EVIDENCE"
    CONFLICT = "CONFLICT"


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


def run_pipeline(
    question: str,
    model: EvidenceModel,
    knowledge_base: list[dict] | None = None,
    top_k: int = 15,
    pre_retrieved_candidates: list[dict] | None = None,
) -> PipelineResult:
    """
    Run the end-to-end ClauseGuard query pipeline.

    Flow:
        question
           ↓
        retrieve (semantic top_k + cross-ref expansion)
           ↓
        evidence classification
           ↓
        sufficiency check
           ├── false → NO_EVIDENCE refusal
           ↓
        structural conflict expansion
           ↓
        conflict detection
           ├── confirmed/unresolved conflict → CONFLICT refusal
           ↓
        grounded answer generation
           ↓
        grounding validation
           ├── invalid → safe refusal
           ↓
        ANSWER

    Parameters
    ----------
    pre_retrieved_candidates:
        If supplied, skip the embedding/retrieval step and use these
        clause dicts directly as the candidate pool.  Each dict must
        have at least ``id``, ``text``, and ``cross_references`` keys.
        This is intended for offline testing only.
    """

    if knowledge_base is None:
        knowledge_base = load_knowledge_base()

    reverse_index = build_reverse_reference_index(knowledge_base)

    # 1. Retrieval (or use pre-computed candidates for offline testing)
    if pre_retrieved_candidates is not None:
        candidates = pre_retrieved_candidates
    else:
        semantic_candidates = semantic_retrieve(
            question,
            knowledge_base,
            top_k=top_k,
        )
        candidates = expand_cross_references(
            semantic_candidates,
            knowledge_base,
        )

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

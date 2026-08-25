"""
Offline integration tests for the temporal policy-version layer inside
the pipeline.

All LLM calls are replaced by a recording fake model. All retrieval is
bypassed via pre_retrieved_candidates. Zero network access.

What these tests protect:

- explicit historical date  -> existing pipeline -> CONFLICT
- explicit amended date     -> existing pipeline -> grounded ANSWER
- undated sensitive query   -> TEMPORAL_AMBIGUITY with independently
                               evaluated branches
- undated ordinary query    -> unchanged single-version pipeline
- branch isolation          -> no evidence/prompts cross between views
- conflict isolation        -> PRE 10-vs-30 conflicts; POST 14-vs-14 does
                               not even reach the conflict model
- grounding isolation       -> a failed branch is never rescued by the other
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from modules.conflict import ConflictStatus
from modules.evidence_model import EvidenceModel
from modules.pipeline import PipelineOutcome, run_pipeline
from modules.policy_versioning import (
    PolicyVersion,
    build_effective_kb,
)


# ---------------------------------------------------------------------------
# Recording fake model
# ---------------------------------------------------------------------------


class RecordingFakeModel(EvidenceModel):
    """
    Returns queued responses in order (cycling when exhausted) and
    records every prompt so tests can assert exactly what each stage
    was allowed to see.
    """

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._index = 0
        self.calls: list[tuple[str, str, bool]] = []

    @property
    def prompts(self) -> list[str]:
        return [user for (_, user, _) in self.calls]

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = True,
    ) -> str:
        self.calls.append((system_prompt, user_prompt, json_mode))
        response = self._responses[self._index % len(self._responses)]
        self._index += 1
        return response


# ---------------------------------------------------------------------------
# Synthetic knowledge base mirroring the real reporting-period fixture
# ---------------------------------------------------------------------------


def _clause(clause_id: str, part: str, section: str, text: str,
            references: list[str]) -> dict:
    return {
        "id": clause_id,
        "part": part,
        "section": section,
        "text": text,
        "cross_references": references,
        "embedding": [],
    }


MINI_KB = [
    _clause(
        "§4.3.2", "Part 4", "4.3 Reporting",
        # Mirrors the real manual's Markdown shape: the operative value is
        # already wrapped in **bold**, exactly the shape that exposed the
        # nested-emphasis substitution bug.
        "A recipient must report any change in household composition, "
        "income, address, or the circumstances of any household member "
        "within **10 calendar days** of the change occurring.",
        [],
    ),
    _clause(
        "§9.1.4", "Part 9", "9.1 Overpayments",
        "Where an overpayment has arisen from a change of circumstances, "
        "and the recipient reported the change within the **30 calendar "
        "days** required under §4.3, no overpayment shall be established.",
        ["§4.3"],
    ),
]

UNRELATED_CLAUSE = _clause(
    "§8.2.1", "Part 8", "8.2 Applications",
    "An application must be made on the approved form and delivered to "
    "the office of the Department.",
    [],
)

POOL_ALL = [
    {**clause, "similarity": 0.9, "retrieval_reason": "semantic"}
    for clause in MINI_KB
]


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------


def _evidence(status: str, covers: str, quote: str) -> str:
    return json.dumps({
        "status": status,
        "covers": covers,
        "evidence_quote": quote,
        "reasoning": "test reasoning",
    })


def _conflict_confirmed() -> str:
    return json.dumps({
        "same_scope": True,
        "conflict": True,
        "conflict_type": "citation_mismatch",
        "reasoning": "§9.1.4 attributes 30 days to §4.3 but §4.3.2 states 10 days.",
    })


def _answer(answer: str, citations: list[str]) -> str:
    return json.dumps({"answer": answer, "citations": citations})


QUOTE_10 = (
    "report any change in household composition, income, address, or the "
    "circumstances of any household member within 10 calendar days of the "
    "change occurring"
)

QUOTE_30 = (
    "reported the change within the 30 calendar days required under "
    "§4.3, no overpayment shall be established"
)

QUOTE_14 = (
    "report any change in household composition, income, address, or the "
    "circumstances of any household member within 14 calendar days of the "
    "change occurring"
)

HISTORICAL_DATE = date(2026, 2, 20)
AMENDED_DATE = date(2026, 4, 10)


# ---------------------------------------------------------------------------
# Explicit-date queries exercise the EXISTING pipeline unchanged
# ---------------------------------------------------------------------------


def test_historical_explicit_date_produces_conflict():
    model = RecordingFakeModel([
        _evidence("SUPPORTED", "reporting deadline", QUOTE_10),
        _evidence("SUPPORTED", "overpayment reporting reference", QUOTE_30),
        _conflict_confirmed(),
    ])

    result = run_pipeline(
        question=(
            "I started a new job on February 20, 2026. How long do I have "
            "to tell the office?"
        ),
        model=model,
        knowledge_base=MINI_KB,
        pre_retrieved_candidates=POOL_ALL,
    )

    assert result.outcome == PipelineOutcome.CONFLICT
    assert len(result.conflicts) >= 1
    assert result.conflicts[0].status == ConflictStatus.CONFIRMED

    pair_ids = {
        result.conflicts[0].clause_a,
        result.conflicts[0].clause_b,
    }
    assert pair_ids == {"§4.3.2", "§9.1.4"}

    # Refused answers are never generated.
    assert result.answer == ""


def test_post_amendment_explicit_date_produces_grounded_answer():
    model = RecordingFakeModel([
        # PRE-view texts must never be classified: only POST texts exist here.
        _evidence("SUPPORTED", "reporting deadline", QUOTE_14),
        _evidence("IRRELEVANT", "", ""),
        _answer(
            "You must tell the office within 14 calendar days (§4.3.2).",
            ["§4.3.2"],
        ),
    ])

    result = run_pipeline(
        question=(
            "I started a new job on April 10, 2026. How long do I have to "
            "tell the office?"
        ),
        model=model,
        knowledge_base=MINI_KB,
        pre_retrieved_candidates=POOL_ALL,
    )

    assert result.outcome == PipelineOutcome.ANSWER
    assert result.citations == ["§4.3.2"]
    assert result.grounding_validation is not None
    assert result.grounding_validation.is_valid
    assert result.branch_results == {}

    joined_prompts = "\n".join(model.prompts)

    # Branch isolation: only the effective (amended) state reached the model.
    assert "14 calendar days" in joined_prompts
    assert "10 calendar days" not in joined_prompts
    assert "30 calendar days" not in joined_prompts

    # Conflict isolation: 14 vs 14 never reaches the conflict model.
    assert not any(
        "policy conflict detector" in prompt for prompt in model.prompts
    )


def test_grounding_failure_in_a_branch_stays_failed():
    model = RecordingFakeModel([
        _evidence("SUPPORTED", "reporting deadline", QUOTE_14),
        _evidence("IRRELEVANT", "", ""),
        _answer("Report within 14 days per §99.99.", ["§99.99"]),
    ])

    result = run_pipeline(
        question="How long do I have to report?",
        model=model,
        knowledge_base=MINI_KB,
        pre_retrieved_candidates=POOL_ALL,
        explicit_date=AMENDED_DATE,
    )

    assert result.outcome == PipelineOutcome.NO_EVIDENCE
    assert result.answer == ""
    assert result.grounding_validation is not None
    assert not result.grounding_validation.is_valid
    assert result.grounding_validation.invalid_citations == ["§99.99"]


def test_clauses_absent_from_a_view_are_dropped_not_invented():
    # §10.5.3A-style inserted clause: exists only in the amended state.
    post_kb = build_effective_kb(MINI_KB, PolicyVersion.POST_AMENDMENT)

    # Force-insertion semantics differ (mini KB has no §10.5.3 host), so
    # simulate an amendment-inserted candidate directly.
    inserted_candidate = [{
        "id": "§10.5.3A",
        "part": "Part 10",
        "section": "10.5 Sanctions",
        "text": "A sanction must not be imposed where the change would "
                "have increased the award.",
        "cross_references": [],
        "embedding": [],
        "similarity": 0.8,
        "retrieval_reason": "semantic",
    }]

    model = RecordingFakeModel([])

    result = run_pipeline(
        question="What happens if I did not report?",
        model=model,
        knowledge_base=MINI_KB,
        pre_retrieved_candidates=inserted_candidate,
        explicit_date=HISTORICAL_DATE,
    )

    # The clause does not exist in the historical state: it is dropped,
    # no evidence is classified, and no clause is invented.
    assert result.outcome == PipelineOutcome.NO_EVIDENCE
    assert model.calls == []
    assert result.evidence_results == []


# ---------------------------------------------------------------------------
# Undated queries
# ---------------------------------------------------------------------------


def test_undated_sensitive_query_yields_temporal_ambiguity():
    model = RecordingFakeModel([
        # ---- PRE branch ----
        _evidence("SUPPORTED", "reporting deadline", QUOTE_10),
        _evidence("SUPPORTED", "overpayment reporting reference", QUOTE_30),
        _conflict_confirmed(),
        # ---- POST branch ----
        _evidence("SUPPORTED", "reporting deadline", QUOTE_14),
        _evidence("IRRELEVANT", "", ""),
        _answer(
            "You must tell the office within 14 calendar days (§4.3.2).",
            ["§4.3.2"],
        ),
    ])

    result = run_pipeline(
        question=(
            "I started a new job. How long do I have to tell the office?"
        ),
        model=model,
        knowledge_base=MINI_KB,
        pre_retrieved_candidates=POOL_ALL,
    )

    assert result.outcome == PipelineOutcome.TEMPORAL_AMBIGUITY
    assert result.answer == ""
    assert set(result.branch_results) == {
        PolicyVersion.PRE_AMENDMENT,
        PolicyVersion.POST_AMENDMENT,
    }

    pre = result.branch_results[PolicyVersion.PRE_AMENDMENT]
    post = result.branch_results[PolicyVersion.POST_AMENDMENT]

    # Each branch concluded independently through the full pipeline.
    assert pre.outcome == PipelineOutcome.CONFLICT
    assert post.outcome == PipelineOutcome.ANSWER
    assert post.citations == ["§4.3.2"]

    # Branch isolation: distinct evidence objects, distinct verified quotes.
    assert pre.retained_evidence is not post.retained_evidence
    assert pre.evidence_results is not post.evidence_results
    pre_quotes = {r.evidence_quote for r in pre.retained_evidence}
    post_quotes = {r.evidence_quote for r in post.retained_evidence}
    assert pre_quotes and post_quotes
    assert pre_quotes.isdisjoint(post_quotes)
    assert any("10 calendar days" in q for q in pre_quotes)
    assert any("14 calendar days" in q for q in post_quotes)

    # Both branches saw the SAME question; only their policy states differed.
    assert pre.question == post.question == result.question


def test_undated_insensitive_query_keeps_single_version_pipeline():
    kb = MINI_KB + [UNRELATED_CLAUSE]

    model = RecordingFakeModel([
        _evidence("SUPPORTED", "application procedure",
                  "An application must be made on the approved form and "
                  "delivered to the office of the Department."),
        _answer(
            "You must apply using the approved form (§8.2.1).",
            ["§8.2.1"],
        ),
    ])

    result = run_pipeline(
        question="How do I apply for support?",
        model=model,
        knowledge_base=kb,
        pre_retrieved_candidates=[
            {**UNRELATED_CLAUSE, "similarity": 0.9,
             "retrieval_reason": "semantic"},
        ],
    )

    # Ordinary question: normal ANSWER, no temporal complexity.
    assert result.outcome == PipelineOutcome.ANSWER
    assert result.citations == ["§8.2.1"]
    assert result.branch_results == {}

    # Exactly ONE classification pass and ONE generation: the pipeline
    # ran once against a single policy version.
    assert len(model.calls) == 2


def test_undated_branches_agreeing_on_no_evidence_collapse():
    model = RecordingFakeModel([
        _evidence("IRRELEVANT", "", ""),   # PRE §4.3.2
        _evidence("IRRELEVANT", "", ""),   # POST §4.3.2
    ])

    result = run_pipeline(
        question="How long do I have to report?",
        model=model,
        knowledge_base=MINI_KB,
        pre_retrieved_candidates=[{**MINI_KB[0],
                                   "similarity": 0.9,
                                   "retrieval_reason": "semantic"}],
    )

    # Both versions lack evidence: refusing holds regardless of version,
    # so this is plain NO_EVIDENCE, not a temporal ambiguity.
    assert result.outcome == PipelineOutcome.NO_EVIDENCE
    assert result.branch_results == {}


def test_undated_branches_agreeing_on_identical_answers_collapse():
    kb = MINI_KB + [UNRELATED_CLAUSE]

    pool = [
        {**MINI_KB[0], "similarity": 0.9, "retrieval_reason": "semantic"},
        {**UNRELATED_CLAUSE, "similarity": 0.8, "retrieval_reason": "semantic"},
    ]

    same_answer = _answer(
        "You must apply using the approved form (§8.2.1).",
        ["§8.2.1"],
    )
    same_support = _evidence(
        "SUPPORTED",
        "application procedure",
        "An application must be made on the approved form and delivered "
        "to the office of the Department.",
    )

    model = RecordingFakeModel([
        # PRE branch: amended clause IRRELEVANT, application rule SUPPORTED
        _evidence("IRRELEVANT", "", ""),
        same_support,
        same_answer,
        # POST branch: identical treatment
        _evidence("IRRELEVANT", "", ""),
        same_support,
        same_answer,
    ])

    result = run_pipeline(
        question="How do I apply for support?",
        model=model,
        knowledge_base=kb,
        pre_retrieved_candidates=pool,
    )

    # The question touched an amended clause (sensitive), but both policy
    # versions produced materially the same grounded answer, so the shared
    # ANSWER is returned rather than a spurious temporal ambiguity.
    assert result.outcome == PipelineOutcome.ANSWER
    assert result.citations == ["§8.2.1"]


def test_failed_branch_is_not_rescued_by_the_other_branch():
    # Variant without the §9.1.4 -> §4.3 cross-reference, so the PRE
    # branch answers cleanly instead of hitting the planted 10-vs-30
    # structural conflict. This isolates the grounding property under
    # test: a branch that fails grounding must stay failed.
    kb_no_xref = [
        MINI_KB[0],
        _clause(
            "§9.1.4", "Part 9", "9.1 Overpayments",
            "Where an overpayment has arisen from a change of circumstances, "
            "and the recipient reported the change within the 30 calendar "
            "days required under §4.3, no overpayment shall be established.",
            [],
        ),
    ]

    pool = [
        {**clause, "similarity": 0.9, "retrieval_reason": "semantic"}
        for clause in kb_no_xref
    ]

    model = RecordingFakeModel([
        # PRE branch: answers cleanly from the historical rule
        _evidence("SUPPORTED", "reporting deadline", QUOTE_10),
        _evidence("IRRELEVANT", "", ""),
        _answer("Report within 10 calendar days (§4.3.2).", ["§4.3.2"]),
        # POST branch: generator hallucinates a citation -> fails grounding
        _evidence("SUPPORTED", "reporting deadline", QUOTE_14),
        _evidence("IRRELEVANT", "", ""),
        _answer("Report within 14 days per §99.99.", ["§99.99"]),
    ])

    result = run_pipeline(
        question="How long do I have to report a change?",
        model=model,
        knowledge_base=kb_no_xref,
        pre_retrieved_candidates=pool,
    )

    assert result.outcome == PipelineOutcome.TEMPORAL_AMBIGUITY

    pre = result.branch_results[PolicyVersion.PRE_AMENDMENT]
    post = result.branch_results[PolicyVersion.POST_AMENDMENT]

    assert pre.outcome == PipelineOutcome.ANSWER
    # The POST branch failed grounding and remains failed; it is NOT
    # rescued by the healthy PRE branch.
    assert post.outcome == PipelineOutcome.NO_EVIDENCE
    assert post.grounding_validation is not None
    assert not post.grounding_validation.is_valid
    assert post.answer == ""


# ---------------------------------------------------------------------------
# Temporal input discipline
# ---------------------------------------------------------------------------


def test_contradicting_explicit_and_question_dates_raise():
    model = RecordingFakeModel([])

    with pytest.raises(ValueError):
        run_pipeline(
            question=(
                "I started a new job on February 20, 2026. How long do I "
                "have to tell the office?"
            ),
            model=model,
            knowledge_base=MINI_KB,
            pre_retrieved_candidates=POOL_ALL,
            explicit_date=date(2026, 4, 10),
        )

    # Refusal happens before ANY model call.
    assert model.calls == []


def test_branch_evidence_never_leaks_between_versions():
    model = RecordingFakeModel([
        # PRE branch
        _evidence("SUPPORTED", "reporting deadline", QUOTE_10),
        _evidence("SUPPORTED", "overpayment reporting reference", QUOTE_30),
        _conflict_confirmed(),
        # POST branch
        _evidence("SUPPORTED", "reporting deadline", QUOTE_14),
        _evidence("IRRELEVANT", "", ""),
        _answer(
            "You must tell the office within 14 calendar days (§4.3.2).",
            ["§4.3.2"],
        ),
    ])

    result = run_pipeline(
        question=(
            "I started a new job. How long do I have to tell the office?"
        ),
        model=model,
        knowledge_base=MINI_KB,
        pre_retrieved_candidates=POOL_ALL,
    )

    assert result.outcome == PipelineOutcome.TEMPORAL_AMBIGUITY

    pre = result.branch_results[PolicyVersion.PRE_AMENDMENT]
    post = result.branch_results[PolicyVersion.POST_AMENDMENT]

    # The POST branch's generation input must contain only POST evidence:
    # its generator prompt cannot contain historical figures.
    # Only the POST branch reaches generation (the PRE branch refuses on
    # conflict before any answer exists), so exactly one generation prompt
    # exists and it must contain ONLY amended-state figures.
    generation_prompts = [
        prompt for prompt in model.prompts
        if "VERIFIED POLICY EVIDENCE" in prompt
    ]
    assert len(generation_prompts) == 1

    assert "14 calendar days" in generation_prompts[0]
    assert "10 calendar days" not in generation_prompts[0]
    assert "30 calendar days" not in generation_prompts[0]

    # Conflict analysis in the PRE branch operated on full PRE clause text.
    conflict_prompt = next(
        prompt for prompt in model.prompts
        if "policy conflict detector" in prompt
    )
    assert "10 calendar days" in conflict_prompt
    assert "30 calendar days" in conflict_prompt

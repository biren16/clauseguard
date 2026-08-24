"""
Pipeline integration tests.

All LLM calls are replaced with FakeModel instances.
All retrieval (embedding API) is bypassed via pre_retrieved_candidates.
This suite runs fully offline and does not consume any API quota.
"""

from __future__ import annotations

import json

from modules.conflict import ConflictStatus
from modules.evidence_model import EvidenceModel
from modules.pipeline import PipelineOutcome, run_pipeline

# These tests verify single-version pipeline ROUTING (answer / conflict /
# no-evidence / grounding-failure). Since Amendment No. 2026-01, an UNDATED
# reporting question is amendment-sensitive and routes to TEMPORAL_AMBIGUITY
# (see tests/test_temporal_pipeline.py). Each test therefore pins the
# historical policy state explicitly via run_pipeline's temporal input —
# the assertions and fixture semantics are otherwise unchanged.
from datetime import date as _date

_EXPLICIT_HISTORICAL_DATE = _date(2025, 1, 1)


# ---------------------------------------------------------------------------
# Fake model
# ---------------------------------------------------------------------------


class SequentialFakeModel(EvidenceModel):
    """
    Returns responses in sequence.
    Cycles through the list when exhausted.
    """

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._index = 0

    def generate(self, *, system_prompt: str, user_prompt: str, json_mode: bool = True) -> str:
        response = self._responses[self._index % len(self._responses)]
        self._index += 1
        return response


# ---------------------------------------------------------------------------
# Synthetic knowledge base (no real embeddings needed)
# ---------------------------------------------------------------------------

_KB = [
    {
        "id": "§4.3.2",
        "part": "Part 4",
        "section": "4.3 Reporting",
        "text": (
            "A recipient must report any change in household composition, income, "
            "address, or the circumstances of any household member within "
            "10 calendar days of the change occurring."
        ),
        "cross_references": [],
        "embedding": [0.0] * 768,
    },
    {
        "id": "§9.1.4",
        "part": "Part 9",
        "section": "9.1 Overpayments",
        "text": (
            "Where an overpayment has arisen from a change of circumstances, and the "
            "recipient reported the change within the 30 calendar days required under "
            "§4.3, no overpayment shall be established."
        ),
        "cross_references": ["§4.3"],
        "embedding": [0.0] * 768,
    },
    {
        "id": "§7.1.3",
        "part": "Part 7",
        "section": "7.1 Needs",
        "text": (
            "The needs figure is calculated by reference to household size and "
            "composition, except in the case of full-time students (see §5.4), "
            "and is subject to the adjustments in §7.3."
        ),
        "cross_references": ["§5.4", "§7.3"],
        "embedding": [0.0] * 768,
    },
]

# Pre-retrieved candidates list (bypasses embedding API)
_CANDIDATES = [
    {**clause, "similarity": 0.9, "retrieval_reason": "semantic"}
    for clause in _KB
]


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------


def _evidence_response(status: str, covers: str, quote: str) -> str:
    return json.dumps({
        "status": status,
        "covers": covers,
        "evidence_quote": quote,
        "reasoning": "test reasoning",
    })


def _conflict_response() -> str:
    return json.dumps({
        "same_scope": True,
        "conflict": True,
        "conflict_type": "citation_mismatch",
        "reasoning": "§9.1.4 attributes 30 days to §4.3 but §4.3.2 states 10 days.",
    })


def _answer_response(answer: str, citations: list[str]) -> str:
    return json.dumps({"answer": answer, "citations": citations})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_pipeline_returns_answer_when_evidence_sufficient():
    """
    Scenario: §4.3.2 is SUPPORTED; others are IRRELEVANT; no numeric conflict.
    Expected: ANSWER outcome with §4.3.2 citation.
    """
    model = SequentialFakeModel([
        # classify_candidates — one call per candidate (3 entries)
        _evidence_response(
            "SUPPORTED", "reporting deadline",
            "report any change in household composition, income, address, or the circumstances of any household member within 10 calendar days of the change occurring",
        ),
        _evidence_response("IRRELEVANT", "", ""),
        _evidence_response("IRRELEVANT", "", ""),
        # Conflict check: §4.3.2 <-> §9.1.4 — no actual conflict
        json.dumps({
            "same_scope": False,
            "conflict": False,
            "conflict_type": "none",
            "reasoning": "Different scopes.",
        }),
        # generate_answer
        _answer_response(
            "You must report within 10 calendar days per §4.3.2.",
            ["§4.3.2"],
        ),
    ])

    result = run_pipeline(
        question="How long do I have to report a change?",
        model=model,
        knowledge_base=_KB,
        pre_retrieved_candidates=_CANDIDATES,
        explicit_date=_EXPLICIT_HISTORICAL_DATE,
    )

    assert result.outcome == PipelineOutcome.ANSWER
    assert "§4.3.2" in result.citations
    assert result.answer != ""


def test_pipeline_returns_no_evidence_when_all_irrelevant():
    """
    Scenario: all candidates are classified IRRELEVANT.
    Expected: NO_EVIDENCE outcome.
    """
    model = SequentialFakeModel([
        _evidence_response("IRRELEVANT", "", ""),
        _evidence_response("IRRELEVANT", "", ""),
        _evidence_response("IRRELEVANT", "", ""),
    ])

    result = run_pipeline(
        question="What is the capital of France?",
        model=model,
        knowledge_base=_KB,
        pre_retrieved_candidates=_CANDIDATES,
        explicit_date=_EXPLICIT_HISTORICAL_DATE,
    )

    assert result.outcome == PipelineOutcome.NO_EVIDENCE
    assert result.answer == ""


def test_pipeline_returns_conflict_when_contradiction_detected():
    """
    Scenario: §4.3.2 (10 days) and §9.1.4 (30 days) both SUPPORTED;
    conflict model confirms citation mismatch.
    Expected: CONFLICT outcome.
    """
    model = SequentialFakeModel([
        # Evidence classification (3 candidates)
        _evidence_response(
            "SUPPORTED", "reporting deadline",
            "report any change in household composition, income, address, or the circumstances of any household member within 10 calendar days of the change occurring",
        ),
        _evidence_response(
            "SUPPORTED", "change reporting deadline",
            "reported the change within the 30 calendar days required under §4.3, no overpayment shall be established",
        ),
        _evidence_response("IRRELEVANT", "", ""),
        # Conflict analysis — §4.3.2 <-> §9.1.4 pair
        _conflict_response(),
    ])

    result = run_pipeline(
        question="How long do I have to report a change of circumstances?",
        model=model,
        knowledge_base=_KB,
        pre_retrieved_candidates=_CANDIDATES,
        explicit_date=_EXPLICIT_HISTORICAL_DATE,
    )

    assert result.outcome == PipelineOutcome.CONFLICT
    assert len(result.conflicts) >= 1
    assert result.conflicts[0].status == ConflictStatus.CONFIRMED


def test_pipeline_grounding_failure_routes_to_no_evidence():
    """
    Scenario: generation returns a citation (§99.99) not in retained evidence.
    Expected: NO_EVIDENCE with grounding_validation.is_valid == False.
    """
    model = SequentialFakeModel([
        # classify_candidates (3 entries)
        _evidence_response(
            "SUPPORTED", "reporting deadline",
            "report any change in household composition, income, address, or the circumstances of any household member within 10 calendar days of the change occurring",
        ),
        _evidence_response("IRRELEVANT", "", ""),
        _evidence_response("IRRELEVANT", "", ""),
        # Conflict check fires: §4.3.2 <-> §9.1.4 — no conflict returned
        json.dumps({
            "same_scope": False,
            "conflict": False,
            "conflict_type": "none",
            "reasoning": "Different scopes.",
        }),
        # generate_answer — hallucinated clause ID
        _answer_response(
            "Report within 10 days per §99.99.",
            ["§99.99"],
        ),
    ])

    result = run_pipeline(
        question="How long to report a change?",
        model=model,
        knowledge_base=_KB,
        pre_retrieved_candidates=_CANDIDATES,
        explicit_date=_EXPLICIT_HISTORICAL_DATE,
    )

    assert result.outcome == PipelineOutcome.NO_EVIDENCE
    assert result.grounding_validation is not None
    assert not result.grounding_validation.is_valid

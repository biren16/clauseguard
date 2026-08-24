from __future__ import annotations

from dataclasses import dataclass
from modules.evidence import EvidenceResult


@dataclass
class GroundingValidationResult:
    """
    Result of deterministic grounding and citation validation.
    """

    is_valid: bool
    invalid_citations: list[str]
    reasoning: str


def validate_grounding(
    answer: str,
    citations: list[str],
    retained_evidence: list[EvidenceResult],
) -> GroundingValidationResult:
    """
    Deterministically validate that all citations belong to the retained evidence set.

    Rules for Day 1:
    1. Answer must be non-empty.
    2. Citations list must be non-empty.
    3. Every cited clause ID must match a clause ID present in retained_evidence.
    """

    if not answer.strip():
        return GroundingValidationResult(
            is_valid=False,
            invalid_citations=[],
            reasoning="Generated answer is empty.",
        )

    if not citations:
        return GroundingValidationResult(
            is_valid=False,
            invalid_citations=[],
            reasoning="Generated answer contains no clause citations.",
        )

    valid_clause_ids = {
        result.clause_id
        for result in retained_evidence
    }

    invalid_citations = [
        citation
        for citation in citations
        if citation not in valid_clause_ids
    ]

    if invalid_citations:
        return GroundingValidationResult(
            is_valid=False,
            invalid_citations=invalid_citations,
            reasoning=(
                "Answer cited clause(s) not present in the verified evidence set: "
                + ", ".join(invalid_citations)
            ),
        )

    return GroundingValidationResult(
        is_valid=True,
        invalid_citations=[],
        reasoning="All citations verified against retained evidence set.",
    )

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json

from google import genai


class EvidenceStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIAL = "PARTIAL"
    IRRELEVANT = "IRRELEVANT"


@dataclass
class EvidenceResult:
    clause_id: str
    status: EvidenceStatus
    covers: str
    evidence_quote: str
    reasoning: str


def is_sufficient(
    results: list[EvidenceResult],
) -> tuple[bool, list[EvidenceResult]]:
    """
    Apply the Day-1 deterministic sufficiency rule.

    Sufficient if:
    1. At least one SUPPORTED clause exists, or
    2. At least two PARTIAL clauses cover distinct aspects.

    When a SUPPORTED clause exists, qualifying PARTIAL clauses are
    retained as well because they may provide additional parts of
    the answer.

    This function is intentionally isolated so the Day-1 heuristic
    can later be replaced with claim-coverage logic.
    """

    supported = [
        result
        for result in results
        if result.status == EvidenceStatus.SUPPORTED
    ]

    partials_with_coverage = [
        result
        for result in results
        if (
            result.status == EvidenceStatus.PARTIAL
            and result.covers.strip()
        )
    ]

    if supported:
        return True, supported + partials_with_coverage

    distinct_covers = {
        result.covers.strip().lower()
        for result in partials_with_coverage
    }

    if len(distinct_covers) >= 2:
        return True, partials_with_coverage

    return False, []


def build_reverse_reference_index(
    clauses: list[dict],
) -> dict[str, list[str]]:
    """
    Build a reverse-reference index.

    Maps a referenced clause/section to the clauses that point to it.

    Example:

        §9.1.4 -> references §4.3

    produces an entry allowing us to discover §9.1.4 when starting
    from §4.3 or one of its child clauses.
    """

    reverse_index: dict[str, list[str]] = {}

    for clause in clauses:
        clause_id = clause["id"]

        for reference in clause.get("cross_references", []):
            reverse_index.setdefault(reference, []).append(clause_id)

    return reverse_index


def _is_related_reference(
    clause_id: str,
    reference: str,
) -> bool:
    """
    Determine whether two clause identifiers are hierarchically related.

    Examples:

        §4.3.2 is related to §4.3
        §4.3 is related to §4.3.2
        §4.3.2 is related to §4.3.2
    """

    return (
        clause_id == reference
        or clause_id.startswith(reference + ".")
        or reference.startswith(clause_id + ".")
    )


def expand_for_conflict_check(
    evidence: list[dict],
    clauses: list[dict],
    reverse_index: dict[str, list[str]],
) -> list[dict]:
    """
    Build the conflict-check set.

    Start with the retained evidence clauses and add structurally
    connected clauses in BOTH directions:

    1. Forward references:
       evidence clause -> referenced clause

    2. Reverse references:
       another clause -> evidence clause/section

    Structurally connected clauses are included for conflict analysis
    even if they were classified IRRELEVANT for directly answering
    the user's question.

    Each expanded clause receives provenance metadata explaining why
    it entered the conflict-check set.
    """

    by_id = {
        clause["id"]: clause
        for clause in clauses
    }

    conflict_set: dict[str, dict] = {}

    # Retained evidence is always part of the conflict-check set.
    for clause in evidence:
        conflict_set[clause["id"]] = {
            **clause,
            "conflict_reason": "evidence",
            "linked_to": None,
            "reference": None,
        }

    for evidence_clause in evidence:
        evidence_id = evidence_clause["id"]

        # Forward references:
        # evidence clause -> referenced clause(s)
        for reference in evidence_clause.get(
            "cross_references",
            [],
        ):
            for clause_id, clause in by_id.items():
                if _is_related_reference(
                    clause_id,
                    reference,
                ):
                    if clause_id not in conflict_set:
                        conflict_set[clause_id] = {
                            **clause,
                            "conflict_reason": "forward_reference",
                            "linked_to": evidence_id,
                            "reference": reference,
                        }

        # Reverse references:
        # another clause -> evidence clause/section
        for reference, referencing_ids in reverse_index.items():
            if _is_related_reference(
                evidence_id,
                reference,
            ):
                for clause_id in referencing_ids:
                    clause = by_id.get(clause_id)

                    if clause and clause_id not in conflict_set:
                        conflict_set[clause_id] = {
                            **clause,
                            "conflict_reason": "reverse_reference",
                            "linked_to": evidence_id,
                            "reference": reference,
                        }

    return list(conflict_set.values())


SYSTEM_PROMPT = """You are an evidence classifier for a policy Q&A system.

Your job is ONLY to evaluate whether the supplied policy clause contains
evidence that contributes to answering the user's question.

You may use ONLY:
- the user's question
- the supplied clause text

Do not use your pretrained knowledge.
Do not infer rules that are not stated in the clause.
Do not use cross-references or assume what another clause might say.

Classify the clause as exactly one of:

SUPPORTED:
The clause directly establishes a rule needed to answer the question.

PARTIAL:
The clause establishes one useful part of the answer, but does not
provide the whole answer.

IRRELEVANT:
The clause does not provide evidence needed to answer the question.

For SUPPORTED or PARTIAL, provide an exact quote copied from the clause
that demonstrates the evidence.

The evidence_quote MUST be an exact substring of the supplied clause text.

Respond with ONLY valid JSON:

{
  "status": "SUPPORTED" | "PARTIAL" | "IRRELEVANT",
  "covers": "short description of what part of the question this covers",
  "evidence_quote": "exact quote from the clause, or empty string",
  "reasoning": "brief explanation"
}

For IRRELEVANT:
- covers must be ""
- evidence_quote must be ""
"""


def classify_clause(
    question: str,
    clause: dict,
    client: genai.Client,
    model: str = "gemini-3.6-flash",
) -> EvidenceResult:
    """
    Classify one clause for evidence relevance.

    Fails closed if Gemini returns malformed output or an evidence quote
    that does not literally occur in the clause.
    """

    clause_text = clause["text"]

    # Deliberately exclude cross_references, embeddings, similarity,
    # and other structural metadata from the classifier prompt.
    prompt = (
        f"Question:\n{question}\n\n"
        f"Candidate clause {clause['id']}:\n"
        f"{clause_text}"
    )

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={
                "system_instruction": SYSTEM_PROMPT,
                "response_mime_type": "application/json",
            },
        )

        raw = response.text.strip()
        parsed = json.loads(raw)

        status = EvidenceStatus(parsed["status"])

        covers = parsed.get("covers", "")
        evidence_quote = parsed.get("evidence_quote", "")
        reasoning = parsed.get("reasoning", "")

    except (
        json.JSONDecodeError,
        KeyError,
        ValueError,
        TypeError,
    ) as exc:
        return EvidenceResult(
            clause_id=clause["id"],
            status=EvidenceStatus.IRRELEVANT,
            covers="",
            evidence_quote="",
            reasoning=f"PARSE_ERROR: {exc}",
        )

    # IRRELEVANT clauses cannot manufacture evidence.
    if status == EvidenceStatus.IRRELEVANT:
        return EvidenceResult(
            clause_id=clause["id"],
            status=status,
            covers="",
            evidence_quote="",
            reasoning=reasoning,
        )

    # A supporting classification without an exact quote fails closed.
    if not evidence_quote:
        return EvidenceResult(
            clause_id=clause["id"],
            status=EvidenceStatus.IRRELEVANT,
            covers="",
            evidence_quote="",
            reasoning=(
                "INVALID_EVIDENCE: model returned no evidence quote"
            ),
        )

    # The quote must literally exist in the source clause.
    if evidence_quote not in clause_text:
        return EvidenceResult(
            clause_id=clause["id"],
            status=EvidenceStatus.IRRELEVANT,
            covers="",
            evidence_quote="",
            reasoning=(
                "INVALID_EVIDENCE: evidence quote does not occur "
                "verbatim in source clause"
            ),
        )

    # A supporting classification without a coverage description
    # cannot participate in the Day-1 sufficiency rule.
    if not covers.strip():
        return EvidenceResult(
            clause_id=clause["id"],
            status=EvidenceStatus.IRRELEVANT,
            covers="",
            evidence_quote="",
            reasoning=(
                "INVALID_EVIDENCE: supporting classification "
                "has no coverage description"
            ),
        )

    return EvidenceResult(
        clause_id=clause["id"],
        status=status,
        covers=covers.strip(),
        evidence_quote=evidence_quote,
        reasoning=reasoning,
    )


def classify_candidates(
    question: str,
    candidates: list[dict],
    client: genai.Client,
    model: str = "gemini-3.6-flash",
) -> list[EvidenceResult]:
    """Classify each candidate independently."""

    results = []

    for clause in candidates:
        result = classify_clause(
            question=question,
            clause=clause,
            client=client,
            model=model,
        )

        results.append(result)

        print(
            f"{result.clause_id}: "
            f"{result.status.value} | "
            f"{result.covers or '-'}"
        )

    return results
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import re

from google import genai


class EvidenceStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIAL = "PARTIAL"
    IRRELEVANT = "IRRELEVANT"


@dataclass
class EvidenceResult:
    """
    Result of classifying one policy clause against a user question.

    Important:
    - This is classification/provenance data.
    - It is NOT the full knowledge-base clause.
    - Use evidence_results_to_clauses() before passing retained evidence
      into structural/conflict expansion.
    """

    clause_id: str
    status: EvidenceStatus
    covers: str
    evidence_quote: str
    reasoning: str


def normalize_for_comparison(text: str) -> str:
    """
    Normalize harmless formatting differences for quote verification.

    This intentionally does NOT rewrite the actual words. It only removes
    formatting differences that can be introduced by Markdown rendering or
    model output.
    """

    # Remove Markdown bold markers while preserving their contents.
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)

    # Collapse spaces, tabs, and line breaks.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def verify_quote(
    evidence_quote: str,
    clause_text: str,
) -> bool:
    """
    Verify that the model's evidence quote is grounded in the source clause.

    Markdown bold markers and whitespace differences are ignored during
    comparison.

    The words themselves must still occur in the supplied source clause.

    Fail closed:
    - empty quote -> False
    - quote not found -> False
    """

    if not evidence_quote.strip():
        return False

    normalized_quote = normalize_for_comparison(evidence_quote)
    normalized_clause = normalize_for_comparison(clause_text)

    return normalized_quote in normalized_clause


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

    IMPORTANT DAY-1 LIMITATION:

    SUPPORTED means that a clause establishes a directly relevant rule.
    It does not prove that the clause answers every aspect of a
    multi-part question.

    This heuristic is intentionally isolated so it can later be replaced
    with proper claim-coverage logic without rewriting the rest of the
    pipeline.
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

    # A SUPPORTED clause is enough under the Day-1 rule, but retain
    # qualifying PARTIAL clauses because they may provide additional
    # aspects of the answer.
    if supported:
        return True, supported + partials_with_coverage

    # Without a SUPPORTED clause, at least two distinct PARTIAL
    # coverage descriptions are required.
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

    produces:

        {
            "§4.3": ["§9.1.4"]
        }

    This allows conflict expansion to discover clauses that reference
    retained evidence even when the retained evidence does not itself
    reference those clauses.
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

    This allows a reference to a parent section such as §4.3 to connect
    to child clauses such as §4.3.2.
    """

    return (
        clause_id == reference
        or clause_id.startswith(reference + ".")
        or reference.startswith(clause_id + ".")
    )


def evidence_results_to_clauses(
    results: list[EvidenceResult],
    clauses: list[dict],
) -> list[dict]:
    """
    Convert validated EvidenceResults back into full knowledge-base clauses.

    EvidenceResult intentionally contains only:
        - classification
        - coverage
        - verified quote
        - reasoning

    Conflict expansion requires the original knowledge-base metadata,
    including:
        - id
        - text
        - cross_references
        - part
        - section
        - etc.

    Results whose clause_id does not exist in the knowledge base are
    silently omitted rather than inventing a clause.
    """

    by_id = {
        clause["id"]: clause
        for clause in clauses
    }

    return [
        by_id[result.clause_id]
        for result in results
        if result.clause_id in by_id
    ]


def expand_for_conflict_check(
    evidence: list[dict],
    clauses: list[dict],
    reverse_index: dict[str, list[str]],
) -> list[dict]:
    """
    Build the conflict-check set.

    Start with retained evidence clauses and add structurally connected
    clauses in BOTH directions:

    1. Forward references:
       evidence clause -> referenced clause

    2. Reverse references:
       another clause -> evidence clause/section

    Structurally connected clauses are included for conflict analysis
    even if they were classified IRRELEVANT for directly answering
    the user's question.

    The evidence classifier is deliberately NOT called again here.

    Each structurally expanded clause receives provenance metadata:

        conflict_reason
        linked_to
        reference

    so the conflict module and debugging output can explain why the
    clause entered the conflict-check set.
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

        # ---------------------------------------------------------
        # Forward references:
        #
        # evidence clause -> referenced clause/section
        # ---------------------------------------------------------
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

        # ---------------------------------------------------------
        # Reverse references:
        #
        # another clause -> evidence clause/section
        # ---------------------------------------------------------
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

Your job is to evaluate whether the supplied policy clause contains
evidence that contributes to answering the user's question.

You may use:
- the user's question
- the supplied clause text

You may use ordinary language understanding to map a user's wording
to terminology used in the clause.

For example:
A user saying "I started a new job" may correspond to a clause discussing
a "change in income" or "change in circumstances."

This kind of language matching is allowed.

However, DO NOT:
- use external policy knowledge
- rely on assumptions about the program
- invent rules not stated in the supplied clause
- infer facts from other clauses
- use cross-reference metadata to decide relevance
- assume what a referenced section contains

The supplied clause text is the only policy evidence available to you.

Classify the clause as exactly one of:

SUPPORTED:
The clause directly establishes a rule needed to answer the question.

PARTIAL:
The clause establishes one useful part of the answer, but does not
provide the whole answer.

IRRELEVANT:
The clause does not provide evidence needed to answer the question.

For SUPPORTED or PARTIAL:

- "covers" must describe the specific part of the user's question
  that the clause supports.
- "evidence_quote" must be copied from the supplied clause text.
- The evidence quote must be traceable to the supplied clause.
- Do not invent or synthesize a quote.

For IRRELEVANT:

- "covers" must be ""
- "evidence_quote" must be ""

Respond with ONLY valid JSON:

{
  "status": "SUPPORTED" | "PARTIAL" | "IRRELEVANT",
  "covers": "short description of what part of the question this covers",
  "evidence_quote": "exact quote from the clause, or empty string",
  "reasoning": "brief explanation of the judgement"
}
"""


def classify_clause(
    question: str,
    clause: dict,
    client: genai.Client,
    model: str = "gemini-3.6-flash",
) -> EvidenceResult:
    """
    Classify one clause for evidence relevance.

    Any malformed or insufficiently grounded model response fails closed
    to IRRELEVANT.

    The classifier does NOT receive structural metadata such as:

        - cross_references
        - similarity scores
        - embeddings
        - retrieval rank
        - conflict provenance

    This prevents structural metadata from biasing the relevance decision.
    """

    clause_text = clause["text"]

    # Deliberately expose only:
    #   1. the user question
    #   2. the candidate clause text
    #
    # Do NOT include cross_references or retrieval metadata.
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

        # Strict enum parsing.
        #
        # Anything outside the three allowed statuses fails closed.
        status = EvidenceStatus(parsed["status"])

        covers = parsed.get("covers", "")
        evidence_quote = parsed.get("evidence_quote", "")
        reasoning = parsed.get("reasoning", "")

        if not isinstance(covers, str):
            raise ValueError("covers must be a string")

        if not isinstance(evidence_quote, str):
            raise ValueError("evidence_quote must be a string")

        if not isinstance(reasoning, str):
            raise ValueError("reasoning must be a string")

    except (
        json.JSONDecodeError,
        KeyError,
        ValueError,
        TypeError,
        AttributeError,
    ) as exc:
        return EvidenceResult(
            clause_id=clause["id"],
            status=EvidenceStatus.IRRELEVANT,
            covers="",
            evidence_quote="",
            reasoning=f"PARSE_ERROR: {exc}",
        )

    # ---------------------------------------------------------
    # IRRELEVANT
    #
    # An irrelevant clause cannot carry evidence into downstream
    # generation or sufficiency decisions.
    # ---------------------------------------------------------
    if status == EvidenceStatus.IRRELEVANT:
        return EvidenceResult(
            clause_id=clause["id"],
            status=status,
            covers="",
            evidence_quote="",
            reasoning=reasoning,
        )

    # ---------------------------------------------------------
    # SUPPORTED / PARTIAL
    #
    # Both require an explicit coverage description.
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # SUPPORTED / PARTIAL
    #
    # Both require a quote that can be verified against the original
    # source clause.
    #
    # Markdown and whitespace differences are tolerated.
    # Actual unsupported words are not.
    # ---------------------------------------------------------
    if not verify_quote(
        evidence_quote=evidence_quote,
        clause_text=clause_text,
    ):
        return EvidenceResult(
            clause_id=clause["id"],
            status=EvidenceStatus.IRRELEVANT,
            covers="",
            evidence_quote="",
            reasoning=(
                "INVALID_EVIDENCE: evidence quote could not be "
                "verified against source clause"
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
    """
    Classify each candidate independently.

    Every clause receives an independent model judgement.

    Validation happens inside classify_clause(), so this function only
    returns EvidenceResults that have passed the evidence integrity checks
    or have explicitly failed closed.
    """

    results: list[EvidenceResult] = []

    for clause in candidates:
        result = classify_clause(
            question=question,
            clause=clause,
            client=client,
            model=model,
        )

        results.append(result)

        # Make validation failures visible during development.
        tag = ""

        if result.reasoning.startswith(
            ("INVALID_EVIDENCE", "PARSE_ERROR")
        ):
            tag = " [DOWNGRADED]"

        print(
            f"{result.clause_id}: "
            f"{result.status.value}{tag} | "
            f"{result.covers or '-'}"
        )

    return results
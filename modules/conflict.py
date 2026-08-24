from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import re

from modules.evidence import EvidenceResult
from modules.evidence_model import (
    EvidenceModel,
    EvidenceModelProviderError,
    EvidenceModelRateLimitError,
)


# ---------------------------------------------------------------------------
# Conflict result types
# ---------------------------------------------------------------------------


class ConflictStatus(str, Enum):
    CONFIRMED = "CONFIRMED"
    UNRESOLVED = "UNRESOLVED"


class ConflictType(str, Enum):
    NUMERIC_SCOPE_CONFLICT = "ordinary"
    ORDINARY = "ordinary"
    CITATION_MISMATCH = "citation_mismatch"
    NONE = "none"


@dataclass
class ConflictCandidate:
    """
    One clause participating in conflict analysis.

    Evidence candidates use their already-verified evidence quote.

    Structural candidates use the original KB clause text.
    """

    clause_id: str
    text: str
    source: str
    references: tuple[str, ...] = ()


@dataclass
class Conflict:
    """
    Result of conflict analysis.

    CONFIRMED:
        The model determined that the clauses contain a genuine
        contradiction.

    UNRESOLVED:
        The pair reached conflict analysis, but the model response
        could not be trusted. This is deliberately escalated rather
        than silently cleared.
    """

    clause_a: str
    clause_b: str
    quote_a: str
    quote_b: str
    reasoning: str
    status: ConflictStatus
    conflict_type: ConflictType | None = None


# ---------------------------------------------------------------------------
# Numeric extraction
# ---------------------------------------------------------------------------


SECTION_REFERENCE_RE = re.compile(
    r"§\s*\d+(?:\.\d+)*"
)

NUMBER_RE = re.compile(
    r"\b\d+(?:\.\d+)?\b"
)


def _extract_numbers(text: str) -> set[str]:
    """
    Extract explicit numeric values while ignoring section identifiers.

    Example:

        "See §4.3.2. Report within 30 days."

    becomes:

        {"30"}
    """

    without_sections = SECTION_REFERENCE_RE.sub("", text)

    return set(NUMBER_RE.findall(without_sections))


# ---------------------------------------------------------------------------
# Numeric requirement extraction
# ---------------------------------------------------------------------------


def _extract_numeric_requirements(text: str) -> set[str]:
    """
    Extract numeric statements that look like actual policy requirements.

    This is intentionally conservative.

    A bare number or a section reference is not enough.

    Examples that qualify:

        "within 10 calendar days"
        "30 calendar days required"
        "must report within 14 days"

    This helps prevent structural clauses containing references or
    unrelated numbers from generating unnecessary LLM calls.
    """

    cleaned = SECTION_REFERENCE_RE.sub("", text)

    patterns = [
        r"\bwithin\s+\d+(?:\.\d+)?\s+(?:calendar\s+)?days?\b",
        r"\b\d+(?:\.\d+)?\s+(?:calendar\s+)?days?\s+(?:required|allowed|permitted)\b",
        r"\bmust\b[^.]{0,100}\b\d+(?:\.\d+)?\s+(?:calendar\s+)?days?\b",
        r"\bshall\b[^.]{0,100}\b\d+(?:\.\d+)?\s+(?:calendar\s+)?days?\b",
    ]

    matches: list[str] = []

    for pattern in patterns:
        matches.extend(
            re.findall(
                pattern,
                cleaned,
                flags=re.IGNORECASE,
            )
        )

    return set(matches)


# ---------------------------------------------------------------------------
# Candidate construction
# ---------------------------------------------------------------------------


def build_candidates(
    evidence: list[EvidenceResult],
    structural_clauses: list[dict],
) -> list[ConflictCandidate]:
    """
    Merge retained evidence and structural conflict clauses.

    Retained evidence:
        use verified evidence_quote.

    Structural clauses:
        use original KB text.

    Duplicate clause IDs are removed.
    """

    candidates: list[ConflictCandidate] = []

    for result in evidence:
        quote = result.evidence_quote.strip()

        if not quote:
            continue

        candidates.append(
            ConflictCandidate(
                clause_id=result.clause_id,
                text=quote,
                source="evidence",
            )
        )

    seen_ids = {
        candidate.clause_id
        for candidate in candidates
    }

    for clause in structural_clauses:
        clause_id = clause["id"]

        if clause_id in seen_ids:
            continue

        references = tuple(
            str(reference)
            for reference in clause.get(
                "cross_references",
                [],
            )
        )

        candidates.append(
            ConflictCandidate(
                clause_id=clause_id,
                text=clause["text"],
                source=clause.get(
                    "conflict_reason",
                    "structural",
                ),
                references=references,
            )
        )

        seen_ids.add(clause_id)

    return candidates


# ---------------------------------------------------------------------------
# Numeric candidate generation
# ---------------------------------------------------------------------------


def find_numeric_disagreement_pairs(
    candidates: list[ConflictCandidate],
) -> list[
    tuple[
        ConflictCandidate,
        ConflictCandidate,
    ]
]:
    """
    Cheap deterministic pre-filter.

    A pair reaches the LLM only when:

    1. both clauses contain actual numeric requirements;
    2. both contain explicit numbers;
    3. the numeric sets differ.

    This function does NOT decide whether a conflict exists.
    """

    pairs = []

    for i, first in enumerate(candidates):
        numbers_first = _extract_numbers(first.text)

        if not numbers_first:
            continue

        requirements_first = _extract_numeric_requirements(
            first.text
        )

        if not requirements_first:
            continue

        for second in candidates[i + 1:]:
            numbers_second = _extract_numbers(second.text)

            if not numbers_second:
                continue

            requirements_second = _extract_numeric_requirements(
                second.text
            )

            if not requirements_second:
                continue

            if numbers_first == numbers_second:
                continue

            pairs.append(
                (first, second)
            )

    return pairs


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------


SCOPE_CONFLICT_PROMPT = """
You are a policy conflict detector.

You are given two policy clauses containing different explicit numeric
requirements.

Determine whether they represent a genuine contradiction.

There are THREE possible situations.

1. ORDINARY CONFLICT

Both clauses apply to the same factual requirement and impose
incompatible values.

Example:

Clause A:
A recipient must report within 10 calendar days.

Clause B:
A recipient must report within 30 calendar days.

If both apply to the same situation, this is a conflict.

2. DIFFERENT-SCOPE RULE

The clauses apply to genuinely different factual situations.

Example:

Clause A:
Ordinary applicants must report within 10 calendar days.

Clause B:
Applicants in a special emergency category have 30 days.

This is NOT a conflict.

3. CITATION / REFERENCE MISMATCH

One clause explicitly states, summarizes, incorporates, or attributes
a requirement to another section, but the stated numeric requirement
does not match the actual requirement in the referenced clause.

THIS IS A CONFLICT.

Do NOT dismiss this merely because the clauses have different
operational purposes.

For example:

Clause A (§4.3.2):
A recipient must report within 10 calendar days.

Clause B (§9.1.4):
30 calendar days required under §4.3.

Even if §9.1.4 is discussing overpayment consequences rather than
the reporting procedure, its statement about what §4.3 requires
contradicts the operative 10-day requirement.

When the supplied text supports this citation mismatch:

same_scope = true
conflict = true
conflict_type = "citation_mismatch"

Do not use outside knowledge.

Reason ONLY from the supplied text.

Clause A ({clause_a_id}):
{clause_a_text}

Clause B ({clause_b_id}):
{clause_b_text}

References associated with Clause A:
{clause_a_references}

References associated with Clause B:
{clause_b_references}

Return ONLY valid JSON:

{{
  "same_scope": true or false,
  "conflict": true or false,
  "conflict_type": "ordinary" or "citation_mismatch" or "none",
  "reasoning": "one or two concise sentences"
}}

Rules:

- conflict=true requires conflict_type to be ordinary or citation_mismatch.
- conflict=false requires conflict_type to be none.
- A citation mismatch can be a conflict even when operational purposes differ.
- Do not dismiss citation mismatches because one clause concerns
  overpayments, enforcement, eligibility, or another downstream effect.
"""


# ---------------------------------------------------------------------------
# Strict model-output parsing
# ---------------------------------------------------------------------------


def _parse_bool(value: object) -> bool:
    """
    Accept ONLY actual JSON booleans.

    Never use bool(value), because:

        bool("false") == True
    """

    if isinstance(value, bool):
        return value

    raise ValueError(
        "Expected JSON boolean, "
        f"got {type(value).__name__}"
    )


def _parse_conflict_type(value: object) -> str:
    if value is None:
        return "none"

    if not isinstance(value, str):
        raise ValueError(
            "conflict_type must be a string"
        )

    allowed = {
        "ordinary",
        "citation_mismatch",
        "none",
        "numeric_scope_conflict",
    }

    if value not in allowed:
        raise ValueError(
            f"invalid conflict_type: {value!r}"
        )

    if value == "numeric_scope_conflict":
        return "ordinary"

    return value


# ---------------------------------------------------------------------------
# Unresolved result
# ---------------------------------------------------------------------------


def _unresolved_conflict(
    a: ConflictCandidate,
    b: ConflictCandidate,
    reason: str,
) -> Conflict:
    return Conflict(
        clause_a=a.clause_id,
        clause_b=b.clause_id,
        quote_a=a.text,
        quote_b=b.text,
        reasoning=(
            "UNRESOLVED: "
            f"{reason} "
            "Flagged conservatively for manual review."
        ),
        status=ConflictStatus.UNRESOLVED,
    )


# ---------------------------------------------------------------------------
# Single-pair conflict analysis
# ---------------------------------------------------------------------------


def check_pair_for_conflict(
    a: ConflictCandidate,
    b: ConflictCandidate,
    model: EvidenceModel,
) -> Conflict | None:
    """
    Analyze one candidate pair.

    IMPORTANT:

    Provider/API failures are NOT swallowed.

    Only malformed model output becomes UNRESOLVED.

    This prevents interface bugs, network failures, and provider errors
    from being misrepresented as ordinary model uncertainty.
    """

    prompt = SCOPE_CONFLICT_PROMPT.format(
        clause_a_id=a.clause_id,
        clause_a_text=a.text,
        clause_a_references=(
            ", ".join(a.references)
            if a.references
            else "none"
        ),
        clause_b_id=b.clause_id,
        clause_b_text=b.text,
        clause_b_references=(
            ", ".join(b.references)
            if b.references
            else "none"
        ),
    )

    # ---------------------------------------------------------
    # Provider call
    # ---------------------------------------------------------

    raw = model.generate(
        system_prompt="",
        user_prompt=prompt,
        json_mode=True,
    ).strip()

    # ---------------------------------------------------------
    # Parse ONLY the model response here
    # ---------------------------------------------------------

    try:
        parsed = json.loads(raw)

        if not isinstance(parsed, dict):
            raise ValueError(
                "model response must be a JSON object"
            )

        same_scope = _parse_bool(
            parsed["same_scope"]
        )

        conflict = _parse_bool(
            parsed["conflict"]
        )

        conflict_type = _parse_conflict_type(
            parsed.get(
                "conflict_type",
                "none",
            )
        )

        reasoning = parsed.get(
            "reasoning",
            "",
        )

        if not isinstance(reasoning, str):
            raise ValueError(
                "reasoning must be a string"
            )

        reasoning = reasoning.strip()

    except (
        json.JSONDecodeError,
        KeyError,
        ValueError,
        TypeError,
    ) as exc:
        return _unresolved_conflict(
            a,
            b,
            f"invalid model judgement ({exc}).",
        )

    # ---------------------------------------------------------
    # Cross-field validation
    # ---------------------------------------------------------

    if conflict:
        if not same_scope and conflict_type != "citation_mismatch":
            return _unresolved_conflict(
                a,
                b,
                "model returned conflict=true with "
                "same_scope=false for non-citation mismatch.",
            )

        if conflict_type == "none":
            return _unresolved_conflict(
                a,
                b,
                "model returned conflict=true with "
                "conflict_type='none'.",
            )

        if not reasoning:
            return _unresolved_conflict(
                a,
                b,
                "model returned conflict=true "
                "without reasoning.",
            )

    else:
        if conflict_type != "none":
            return _unresolved_conflict(
                a,
                b,
                "model returned conflict=false with "
                f"conflict_type={conflict_type!r}.",
            )

    # ---------------------------------------------------------
    # No conflict
    # ---------------------------------------------------------

    if not conflict:
        return None

    # ---------------------------------------------------------
    # Confirmed conflict
    # ---------------------------------------------------------

    return Conflict(
        clause_a=a.clause_id,
        clause_b=b.clause_id,
        quote_a=a.text,
        quote_b=b.text,
        reasoning=(
            f"CONFIRMED [{conflict_type}]: "
            f"{reasoning}"
        ),
        status=ConflictStatus.CONFIRMED,
        conflict_type=ConflictType(conflict_type),
    )


# ---------------------------------------------------------------------------
# Full conflict detector
# ---------------------------------------------------------------------------


def detect_conflicts(
    evidence: list[EvidenceResult],
    structural_clauses: list[dict],
    model: EvidenceModel,
    max_model_calls: int = 8,
) -> list[Conflict]:
    """
    Complete conflict-analysis pipeline.

    1. Build candidates.
    2. Remove section-reference numbers.
    3. Find genuine numeric disagreement candidates.
    4. Ask the LLM to classify them.
    5. Stop after max_model_calls.

    The call budget protects live Groq usage.
    """

    if max_model_calls < 1:
        raise ValueError(
            "max_model_calls must be >= 1"
        )

    candidates = build_candidates(
        evidence=evidence,
        structural_clauses=structural_clauses,
    )

    numeric_pairs = find_numeric_disagreement_pairs(
        candidates
    )

    print(
        f"\nCONFLICT CANDIDATES: "
        f"{len(numeric_pairs)} pair(s)",
        flush=True,
    )

    if not numeric_pairs:
        return []

    conflicts: list[Conflict] = []

    for index, (first, second) in enumerate(
        numeric_pairs
    ):
        if index >= max_model_calls:
            print(
                "CONFLICT ANALYSIS STOPPED: "
                f"max_model_calls={max_model_calls}",
                flush=True,
            )
            break

        print(
            f"  MODEL CHECK: "
            f"{first.clause_id} <-> "
            f"{second.clause_id}",
            flush=True,
        )

        result = check_pair_for_conflict(
            a=first,
            b=second,
            model=model,
        )

        if result is None:
            print(
                "  -> no conflict",
                flush=True,
            )
            continue

        conflicts.append(result)

        print(
            f"  -> {result.status.value}: "
            f"{result.reasoning}",
            flush=True,
        )

    return conflicts
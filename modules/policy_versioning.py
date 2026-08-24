"""
Deterministic temporal policy resolution for Amendment No. 2026-01.

This module owns ONE question and only one question:

    Which effective policy state(s) should be evaluated?

It is completely deterministic and never involves an LLM.

Design (see DECISIONS.md):

    BASE KB + structured amendment operations
        -> deterministic resolver
        -> PRE-AMENDMENT view / POST-AMENDMENT view

Each view is internally consistent. The existing ClauseGuard pipeline
(retrieval, evidence, sufficiency, structural expansion, conflict,
generation, grounding) runs independently on whichever view(s) the
resolver selects. This module deliberately knows nothing about
evidence, conflicts, answers, or grounding, and those modules know
nothing about time.

Temporal anchors (Amendment No. 2026-01, paragraph 5):

    - Paragraph 2 amendments (reporting periods, §4.3.2 / §9.1.4)
      apply to changes of circumstances OCCURRING on or after
      1 March 2026                     -> anchor: CHANGE_OCCURRED

    - Paragraphs 1, 3 and 4 amendments (earnings disregard §6.4.1,
      income thresholds §6.6.1, sanctions §10.5.2 / §10.5.3A) apply to
      determinations MADE on or after 1 March 2026
                                       -> anchor: DETERMINATION_DATE

Both anchor classes currently share the same threshold date
(AMENDMENT_EFFECTIVE_DATE), so a user-supplied date resolves to a
policy version deterministically without deciding WHICH kind of date
it is. The anchor is still represented explicitly because it is part
of the amendment's semantics and is surfaced in ambiguity responses.

The base knowledge base is NEVER mutated: amended views are built
in-memory from deep copies. Historical and amended clause text are
never merged or synthesized into conditional clauses.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from datetime import date
from enum import Enum


# ---------------------------------------------------------------------------
# Core temporal vocabulary
# ---------------------------------------------------------------------------


class PolicyVersion(str, Enum):
    """An effective policy state."""

    PRE_AMENDMENT = "PRE_AMENDMENT"
    POST_AMENDMENT = "POST_AMENDMENT"


class TemporalAnchor(str, Enum):
    """
    Which real-world event a provision's date must be compared against.

    Derived from the transitional provisions of Amendment No. 2026-01
    (paragraph 5). See module docstring.
    """

    CHANGE_OCCURRED = "CHANGE_OCCURRED"
    DETERMINATION_DATE = "DETERMINATION_DATE"


ANCHOR_DESCRIPTIONS: dict[TemporalAnchor, str] = {
    TemporalAnchor.CHANGE_OCCURRED: (
        "the date the change of circumstances occurred"
    ),
    TemporalAnchor.DETERMINATION_DATE: (
        "the date the determination was made"
    ),
}


# Amendment metadata (data/Amendment No. 2026-01.md).
AMENDMENT_NUMBER = "2026-01"
AMENDMENT_ISSUE_DATE = date(2026, 2, 12)
AMENDMENT_EFFECTIVE_DATE = date(2026, 3, 1)


class TemporalContext(str, Enum):
    """
    Outcome of temporal input resolution.

    EXPLICIT:
        Exactly one effective policy version is determinately selected.

    ABSENT:
        No usable temporal information was supplied. The question may
        need to be evaluated against multiple policy versions.

    INPUT_CONFLICT:
        The supplied temporal inputs are mutually inconsistent (for
        example --date says one thing and the question text implies
        another). The system refuses to silently choose one.
    """

    EXPLICIT = "EXPLICIT"
    ABSENT = "ABSENT"
    INPUT_CONFLICT = "INPUT_CONFLICT"


@dataclass(frozen=True)
class TemporalResolution:
    """
    Deterministic result of temporal input resolution.

    versions:
        The policy version(s) that must be evaluated. Exactly one for
        EXPLICIT; potentially both for ABSENT when the question is
        amendment-sensitive; empty for INPUT_CONFLICT.
    """

    context: TemporalContext
    versions: tuple[PolicyVersion, ...] = ()
    target_date: date | None = None
    notes: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Deterministic date extraction
#
# Supported explicit forms (all validated as real calendar dates):
#
#   2026-02-20            ISO
#   20 February 2026      day-first month name (abbreviations allowed)
#   February 20, 2026     month-first month name (comma optional)
#   02/20/2026            numeric month/day/year
#   20/02/2026            numeric day/month/year
#
# A numeric form where BOTH day-like components are <= 12 (e.g.
# 05/06/2026) is genuinely ambiguous between the two conventions and is
# deliberately NOT interpreted. Uninterpretable dates are ignored rather
# than guessed; the question then resolves as having no usable date.
# ---------------------------------------------------------------------------


_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

_MONTH_NAME_RE = (
    r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)"
)

_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_DAY_FIRST_RE = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+" + _MONTH_NAME_RE + r"\.?,?\s+(\d{4})\b",
    re.IGNORECASE,
)
_MONTH_FIRST_RE = re.compile(
    r"\b" + _MONTH_NAME_RE + r"\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b",
    re.IGNORECASE,
)
_NUMERIC_DATE_RE = re.compile(r"\b(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})\b")

_DATE_PATTERNS = (
    _ISO_DATE_RE,
    _DAY_FIRST_RE,
    _MONTH_FIRST_RE,
    _NUMERIC_DATE_RE,
)


def _make_date(year: int, month: int, day: int) -> date | None:
    """Construct a date, returning None for impossible calendar dates."""
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _month_number(name: str) -> int:
    return _MONTHS[name[:3].lower()]


def extract_dates(text: str) -> list[date]:
    """
    Extract explicit calendar dates from free text.

    Only unambiguous, valid dates are returned. Ambiguous numeric
    forms and impossible calendar dates are skipped (never guessed).

    Duplicate dates (expressed in different formats) are collapsed.
    Results are ordered by their position in the text.
    """

    found: list[tuple[int, date]] = []

    for pattern in _DATE_PATTERNS:
        for match in pattern.finditer(text):
            parsed = _parse_match(pattern, match)
            if parsed is not None:
                found.append((match.start(), parsed))

    # Order by position and collapse duplicates.
    found.sort(key=lambda item: item[0])

    seen: set[date] = set()
    ordered: list[date] = []
    for _, parsed in found:
        if parsed not in seen:
            seen.add(parsed)
            ordered.append(parsed)

    return ordered


def _parse_match(pattern: re.Pattern, match: re.Match) -> date | None:
    if pattern is _ISO_DATE_RE:
        year, month, day = (int(g) for g in match.groups())

    elif pattern is _DAY_FIRST_RE:
        day = int(match.group(1))
        month = _month_number(match.group(2))
        year = int(match.group(3))

    elif pattern is _MONTH_FIRST_RE:
        month = _month_number(match.group(1))
        day = int(match.group(2))
        year = int(match.group(3))

    else:  # _NUMERIC_DATE_RE
        first, second, year = (int(g) for g in match.groups())

        if first > 12 >= second:
            # e.g. 20/02/2026 -> day/month/year
            day, month = first, second
        elif second > 12 >= first:
            # e.g. 02/20/2026 -> month/day/year
            month, day = first, second
        else:
            # Both <= 12 (genuinely ambiguous) or both > 12
            # (impossible): do not guess.
            return None

    return _make_date(year, month, day)


def parse_date_string(value: str) -> date:
    """
    Strictly parse a standalone date string (e.g. the CLI --date value).

    The whole trimmed string must be exactly one recognized, unambiguous
    date expression.

    Raises ValueError otherwise.
    """

    trimmed = value.strip()

    matches: list[tuple[int, str, str]] = [
        (match.start(), match.group(0), pattern.pattern)
        for pattern in _DATE_PATTERNS
        for match in pattern.finditer(trimmed)
    ]

    if len(matches) == 1:
        start, matched_text, _ = matches[0]
        candidate = None

        for pattern in _DATE_PATTERNS:
            match = pattern.search(matched_text)
            if match:
                candidate = _parse_match(pattern, match)
                break

        remainder = trimmed.replace(matched_text, " ", 1).strip()

        if candidate is not None and start == 0 and remainder == "":
            return candidate

    raise ValueError(
        f"Unrecognized or ambiguous date: {value!r}. "
        "Supported forms include 2026-04-10, 20 February 2026, "
        "February 20, 2026, 20/02/2026 and 02/20/2026."
    )


def format_date(target: date) -> str:
    """Human-readable date used in user-facing explanations."""

    return target.strftime("%d %B %Y").lstrip("0")


# ---------------------------------------------------------------------------
# Input resolution
# ---------------------------------------------------------------------------


def _version_for_date(target: date) -> PolicyVersion:
    """
    Map a concrete date to the applicable policy version.

    The amendment takes effect ON 1 March 2026 ("on or after"), which
    holds for every anchor class in this amendment.
    """

    if target >= AMENDMENT_EFFECTIVE_DATE:
        return PolicyVersion.POST_AMENDMENT

    return PolicyVersion.PRE_AMENDMENT


def resolve_temporal_context(
    question: str,
    explicit_date: date | None = None,
) -> TemporalResolution:
    """
    Deterministically resolve which policy version(s) apply.

    Precedence:

        1. explicit_date (e.g. CLI --date)
        2. dates extracted from the question text
        3. no explicit temporal information

    If explicit temporal inputs exist AND disagree, the resolution
    fails closed as INPUT_CONFLICT instead of silently choosing one.
    """

    question_dates = extract_dates(question)

    if explicit_date is not None:
        explicit_version = _version_for_date(explicit_date)

        disagreeing = sorted(
            {
                _version_for_date(question_date).value
                for question_date in question_dates
                if _version_for_date(question_date) != explicit_version
            }
        )

        if disagreeing:
            return TemporalResolution(
                context=TemporalContext.INPUT_CONFLICT,
                versions=(),
                target_date=explicit_date,
                notes=(
                    f"--date {format_date(explicit_date)} selects "
                    f"{explicit_version.value}, but the question text "
                    "also contains date(s) selecting a different "
                    f"policy version ({', '.join(disagreeing)}).",
                ),
            )

        return TemporalResolution(
            context=TemporalContext.EXPLICIT,
            versions=(explicit_version,),
            target_date=explicit_date,
        )

    question_versions = {
        _version_for_date(question_date)
        for question_date in question_dates
    }

    if len(question_versions) == 1:
        version = next(iter(question_versions))
        return TemporalResolution(
            context=TemporalContext.EXPLICIT,
            versions=(version,),
            target_date=question_dates[0],
        )

    if len(question_versions) > 1:
        return TemporalResolution(
            context=TemporalContext.INPUT_CONFLICT,
            versions=(),
            notes=(
                "The question contains multiple dates that select "
                "different policy versions.",
            ),
        )

    return TemporalResolution(
        context=TemporalContext.ABSENT,
        versions=(
            PolicyVersion.PRE_AMENDMENT,
            PolicyVersion.POST_AMENDMENT,
        ),
    )


# ---------------------------------------------------------------------------
# Structured amendment operations
#
# Each operation is a deterministic transformation applied to the base
# knowledge base IN MEMORY. Text substitutions are silent no-ops when
# the target pattern is absent (which keeps synthetic test KBs
# working); tests/test_policy_versioning.py verifies every operation
# against the REAL corpus so silent drift cannot hide there.
#
# Anchors are taken from the transitional provisions (paragraph 5):
# paragraph 2 -> CHANGE_OCCURRED; paragraphs 1/3/4 -> DETERMINATION_DATE.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AmendmentTextSubstitution:
    """Replace one exact substring within a single clause."""

    paragraph: str
    target_id: str
    old: str
    new: str
    anchor: TemporalAnchor
    description: str

    def apply_to(self, kb: list[dict]) -> bool:
        clause = next(
            (c for c in kb if c["id"] == self.target_id),
            None,
        )

        if clause is None:
            return False

        clause["text"] = clause["text"].replace(self.old, self.new)
        return True


@dataclass(frozen=True)
class AmendmentClauseInsertion:
    """Insert a new canonical clause after an existing clause."""

    paragraph: str
    after_id: str
    anchor: TemporalAnchor
    description: str
    clause_id: str
    text: str
    fallback_part: str = ""
    fallback_section: str = ""

    def apply_to(self, kb: list[dict]) -> bool:
        if any(clause["id"] == self.clause_id for clause in kb):
            return False

        host = next(
            (clause for clause in kb if clause["id"] == self.after_id),
            None,
        )

        new_clause = {
            "id": self.clause_id,
            "part": (
                host["part"]
                if host and host.get("part")
                else self.fallback_part
            ),
            "section": (
                host["section"]
                if host and host.get("section")
                else self.fallback_section
            ),
            "cross_references": [],
            "embedding": [],
            "text": self.text,
        }

        if host is not None:
            kb.insert(kb.index(host) + 1, new_clause)
        else:
            kb.append(new_clause)

        return True


_INCOME_THRESHOLD_TABLE_PRE = (
    "| Household size | Monthly threshold |\n"
    "|:--|:--|\n"
    "| 1 | $1,180 |\n"
    "| 2 | $1,590 |\n"
    "| 3 | $2,000 |\n"
    "| 4 | $2,410 |\n"
    "| 5 | $2,820 |\n"
    "| each additional member | + $410 |"
)

_INCOME_THRESHOLD_TABLE_POST = (
    "| Household size | Monthly threshold |\n"
    "|:--|:--|\n"
    "| 1 | $1,225 |\n"
    "| 2 | $1,650 |\n"
    "| 3 | $2,075 |\n"
    "| 4 | $2,500 |\n"
    "| 5 | $2,925 |\n"
    "| each additional member | + $425 |"
)


#: Amendment No. 2026-01 operations, in amendment order.
AMENDMENT_OPERATIONS: tuple = (
    AmendmentTextSubstitution(
        paragraph="1.1",
        target_id="§6.4.1",
        old="$120 per month",
        new="**$175 per month**",
        anchor=TemporalAnchor.DETERMINATION_DATE,
        description="Earnings disregard increased to $175 per month.",
    ),
    AmendmentTextSubstitution(
        paragraph="2.1",
        target_id="§4.3.2",
        old="10 calendar days",
        new="**14 calendar days**",
        anchor=TemporalAnchor.CHANGE_OCCURRED,
        description="Reporting period extended to 14 calendar days.",
    ),
    AmendmentTextSubstitution(
        paragraph="2.2",
        target_id="§9.1.4",
        old="30 calendar days",
        new="**14 calendar days**",
        anchor=TemporalAnchor.CHANGE_OCCURRED,
        description=(
            "Overpayment reporting reference aligned to 14 calendar days."
        ),
    ),
    AmendmentTextSubstitution(
        paragraph="3.1",
        target_id="§6.6.1",
        old=_INCOME_THRESHOLD_TABLE_PRE,
        new=_INCOME_THRESHOLD_TABLE_POST,
        anchor=TemporalAnchor.DETERMINATION_DATE,
        description="Income threshold table substituted.",
    ),
    AmendmentTextSubstitution(
        paragraph="4.1",
        target_id="§10.5.2",
        old="20 per cent",
        new="**15 per cent**",
        anchor=TemporalAnchor.DETERMINATION_DATE,
        description="Sanction reduction set to 15 per cent.",
    ),
    AmendmentClauseInsertion(
        paragraph="4.2",
        after_id="§10.5.3",
        anchor=TemporalAnchor.DETERMINATION_DATE,
        description=(
            "No sanction for failure to report an award-increasing change."
        ),
        clause_id="§10.5.3A",
        text=(
            "A sanction must not be imposed in respect of a failure to "
            "report where the change of circumstances in question would "
            "have increased the award."
        ),
        fallback_part="Part 10 — Sanctions",
        fallback_section="10.5 Sanctions",
    ),
)


# ---------------------------------------------------------------------------
# Effective knowledge-base views
# ---------------------------------------------------------------------------


def build_effective_kb(
    knowledge_base: list[dict],
    version: PolicyVersion,
) -> list[dict]:
    """
    Build an in-memory effective policy view WITHOUT mutating the base KB.

    PRE_AMENDMENT:
        The historical/canonical manual state (shallow per-clause copies;
        clause texts are immutable strings, so the base cannot change).

    POST_AMENDMENT:
        A deep copy with the structured amendment operations applied.
        New clauses keep their canonical IDs (e.g. §10.5.3A) and do not
        exist at all in the pre-amendment view.
    """

    if version is PolicyVersion.PRE_AMENDMENT:
        return [dict(clause) for clause in knowledge_base]

    amended_kb = copy.deepcopy(knowledge_base)

    for operation in AMENDMENT_OPERATIONS:
        operation.apply_to(amended_kb)

    return amended_kb


def changed_clause_ids(knowledge_base: list[dict]) -> frozenset[str]:
    """
    Deterministically identify clauses whose effective content differs
    between the pre- and post-amendment views.

    This includes substituted clauses and clauses inserted by the
    amendment (which exist only in the post-amendment view).
    """

    pre_kb = build_effective_kb(knowledge_base, PolicyVersion.PRE_AMENDMENT)
    post_kb = build_effective_kb(knowledge_base, PolicyVersion.POST_AMENDMENT)

    pre_by_id = {clause["id"]: clause["text"] for clause in pre_kb}
    post_by_id = {clause["id"]: clause["text"] for clause in post_kb}

    changed = {
        clause_id
        for clause_id, text in pre_by_id.items()
        if post_by_id.get(clause_id) != text
    }

    changed.update(set(post_by_id) - set(pre_by_id))

    return frozenset(changed)


def version_label(version: PolicyVersion) -> str:
    """
    Human-readable label derived from the amendment's effective date,
    used by the temporal ambiguity renderer.
    """

    effective = format_date(AMENDMENT_EFFECTIVE_DATE)

    if version is PolicyVersion.POST_AMENDMENT:
        return f"ON OR AFTER {effective}"

    return f"BEFORE {effective}"

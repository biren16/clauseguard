"""
Deterministic unit tests for the temporal policy-version layer.

Zero LLM calls. The REAL knowledge base is used to guard against
silent corpus drift in the amendment operations.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from modules.policy_versioning import (
    AMENDMENT_EFFECTIVE_DATE,
    AMENDMENT_OPERATIONS,
    PolicyVersion,
    TemporalAnchor,
    TemporalContext,
    build_effective_kb,
    changed_clause_ids,
    extract_dates,
    format_date,
    parse_date_string,
    resolve_temporal_context,
    version_label,
)


KB_PATH = Path("data/knowledge_base.json")


@pytest.fixture(scope="module")
def real_kb() -> list[dict]:
    return json.loads(KB_PATH.read_text(encoding="utf-8"))


def clause(kb: list[dict], clause_id: str) -> dict:
    return next(c for c in kb if c["id"] == clause_id)


# ---------------------------------------------------------------------------
# Date extraction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("I started on 2026-02-20.", [date(2026, 2, 20)]),
        ("Started 20 February 2026", [date(2026, 2, 20)]),
        ("Started 20 Feb 2026", [date(2026, 2, 20)]),
        ("Started February 20, 2026", [date(2026, 2, 20)]),
        ("Started Feb 20 2026", [date(2026, 2, 20)]),
        ("Effective 1 March 2026", [date(2026, 3, 1)]),
        ("Report by 02/20/2026 please", [date(2026, 2, 20)]),
        ("Report by 20/02/2026 please", [date(2026, 2, 20)]),
        ("Change on 2026-04-10, decided 11 April 2026",
         [date(2026, 4, 10), date(2026, 4, 11)]),
        # Same date expressed twice collapses.
        ("2026-02-20 and 20 February 2026", [date(2026, 2, 20)]),
    ],
)
def test_extract_dates_supported_forms(text, expected):
    assert extract_dates(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "I started a new job",                       # no date at all
        "within 10 calendar days of the change",     # durations are not dates
        "see §4.3.2 and §9.1.4",                     # section references
        "on 31 February 2026",                       # impossible calendar date
        "on 2026-02-30",                             # impossible ISO date
        "filed on 05/06/2026",                       # ambiguous day/month order
        "on 32/13/2026",                             # both components invalid
        "sometime last week",                        # relative dates not guessed
    ],
)
def test_extract_dates_does_not_guess(text):
    assert extract_dates(text) == []


def test_extract_dates_boundary_is_inclusive():
    # Amendment says changes ON or after 1 March 2026 are post-amendment;
    # extraction itself must still recognize the boundary date.
    assert extract_dates("on 2026-03-01") == [AMENDMENT_EFFECTIVE_DATE]


# ---------------------------------------------------------------------------
# Strict standalone date parsing (CLI --date)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("2026-04-10", date(2026, 4, 10)),
        (" 2026-04-10 ", date(2026, 4, 10)),
        ("20 February 2026", date(2026, 2, 20)),
        ("February 20, 2026", date(2026, 2, 20)),
        ("20/02/2026", date(2026, 2, 20)),
        ("02/20/2026", date(2026, 2, 20)),
    ],
)
def test_parse_date_string_valid(value, expected):
    assert parse_date_string(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "",
        "not a date",
        "05/06/2026",          # ambiguous: must refuse, not guess
        "31 February 2026",    # impossible date
        "meeting on 2026-04-10 soon",   # extra words -> not a pure date
        "2026-04-10 and 2026-05-11",    # two dates
    ],
)
def test_parse_date_string_invalid(value):
    with pytest.raises(ValueError):
        parse_date_string(value)


# ---------------------------------------------------------------------------
# Temporal resolution
# ---------------------------------------------------------------------------


def _versions(resolution):
    return resolution.versions


def test_resolution_absent_when_no_temporal_information():
    r = resolve_temporal_context("I started a new job. How long do I have?")
    assert r.context == TemporalContext.ABSENT
    assert set(r.versions) == {
        PolicyVersion.PRE_AMENDMENT,
        PolicyVersion.POST_AMENDMENT,
    }


def test_resolution_explicit_pre_from_question_text():
    r = resolve_temporal_context(
        "I started a new job on February 20, 2026. How long do I have "
        "to tell the office?"
    )
    assert r.context == TemporalContext.EXPLICIT
    assert _versions(r) == (PolicyVersion.PRE_AMENDMENT,)
    assert r.target_date == date(2026, 2, 20)


def test_resolution_explicit_post_from_question_text():
    r = resolve_temporal_context(
        "I started a new job on April 10, 2026. How long do I have to "
        "tell the office?"
    )
    assert r.context == TemporalContext.EXPLICIT
    assert _versions(r) == (PolicyVersion.POST_AMENDMENT,)


def test_resolution_boundary_on_effective_date_is_post():
    r = resolve_temporal_context("The change occurred on 2026-03-01.")
    assert r.context == TemporalContext.EXPLICIT
    assert _versions(r) == (PolicyVersion.POST_AMENDMENT,)


def test_resolution_day_before_effective_date_is_pre():
    r = resolve_temporal_context("The change occurred on 2026-02-28.")
    assert r.context == TemporalContext.EXPLICIT
    assert _versions(r) == (PolicyVersion.PRE_AMENDMENT,)


def test_explicit_date_takes_precedence_over_question_date():
    # --date agrees with the question's date: fine.
    r = resolve_temporal_context(
        "change on 2026-04-10", explicit_date=date(2026, 5, 5)
    )
    assert r.context == TemporalContext.EXPLICIT
    assert _versions(r) == (PolicyVersion.POST_AMENDMENT,)
    assert r.target_date == date(2026, 5, 5)


def test_explicit_date_conflicting_with_question_date_fails_closed():
    r = resolve_temporal_context(
        "change on February 20, 2026", explicit_date=date(2026, 4, 10)
    )
    assert r.context == TemporalContext.INPUT_CONFLICT
    assert r.versions == ()
    assert r.notes


def test_two_question_dates_selecting_different_versions_fail_closed():
    r = resolve_temporal_context(
        "change on 2026-02-20 but determined on 2026-04-10"
    )
    assert r.context == TemporalContext.INPUT_CONFLICT
    assert r.versions == ()


# ---------------------------------------------------------------------------
# Effective KB views against the REAL corpus
# ---------------------------------------------------------------------------


def test_pre_view_preserves_historical_values(real_kb):
    pre = build_effective_kb(real_kb, PolicyVersion.PRE_AMENDMENT)

    assert "10 calendar days" in clause(pre, "§4.3.2")["text"]
    assert "14 calendar days" not in clause(pre, "§4.3.2")["text"]
    assert "30 calendar days" in clause(pre, "§9.1.4")["text"]
    assert "$120 per month" in clause(pre, "§6.4.1")["text"]
    assert "$1,180" in clause(pre, "§6.6.1")["text"]
    assert "20 per cent" in clause(pre, "§10.5.2")["text"]

    # §10.5.3A does not exist in the historical state.
    assert all(c["id"] != "§10.5.3A" for c in pre)


def test_post_view_applies_all_amendments(real_kb):
    post = build_effective_kb(real_kb, PolicyVersion.POST_AMENDMENT)

    text_432 = clause(post, "§4.3.2")["text"]
    assert "14 calendar days" in text_432
    assert "10 calendar days" not in text_432
    # The amendment replaces the figure "in both places where it occurs".
    assert text_432.count("**14 calendar days**") == 2

    assert "**14 calendar days**" in clause(post, "§9.1.4")["text"]
    assert "30 calendar days" not in clause(post, "§9.1.4")["text"]

    assert "**$175 per month**" in clause(post, "§6.4.1")["text"]
    assert "$120 per month" not in clause(post, "§6.4.1")["text"]

    assert "$1,225" in clause(post, "§6.6.1")["text"]
    assert "$1,180" not in clause(post, "§6.6.1")["text"]
    assert "+ $425 |" in clause(post, "§6.6.1")["text"]

    assert "**15 per cent**" in clause(post, "§10.5.2")["text"]
    assert "20 per cent" not in clause(post, "§10.5.2")["text"]


def test_post_view_inserts_canonical_10503a_after_10503(real_kb):
    post = build_effective_kb(real_kb, PolicyVersion.POST_AMENDMENT)

    inserted = clause(post, "§10.5.3A")

    # Canonical ID; no synthetic -PRE/-POST suffixes.
    assert inserted["id"] == "§10.5.3A"
    assert "would have increased the award" in inserted["text"]

    # Inserted immediately after its host clause, inheriting placement.
    host_index = next(
        i for i, c in enumerate(post) if c["id"] == "§10.5.3"
    )
    assert post.index(inserted) == host_index + 1

    # Structural metadata mirrors the surrounding section.
    assert inserted["part"] == clause(post, "§10.5.3")["part"]
    assert inserted["section"] == clause(post, "§10.5.3")["section"]


def test_amendment_operations_match_real_corpus(real_kb):
    """
    Corpus-drift guard: every amendment operation must actually apply to
    the real knowledge base. Substitution ops are silent no-ops on
    synthetic fixtures by design, so THIS test is what makes silent
    drift impossible in production.
    """

    post = build_effective_kb(real_kb, PolicyVersion.POST_AMENDMENT)
    pre_ids = {c["id"]: c["text"] for c in real_kb}
    post_texts = {c["id"]: c["text"] for c in post}

    for operation in AMENDMENT_OPERATIONS:
        if hasattr(operation, "target_id"):
            assert operation.target_id in pre_ids, operation.paragraph
            assert post_texts[operation.target_id] != pre_ids[operation.target_id]
        else:
            assert operation.clause_id in post_texts
            assert operation.clause_id not in pre_ids


def test_base_kb_is_never_mutated(real_kb):
    snapshot = json.dumps(real_kb)

    build_effective_kb(real_kb, PolicyVersion.PRE_AMENDMENT)
    build_effective_kb(real_kb, PolicyVersion.POST_AMENDMENT)
    changed_clause_ids(real_kb)

    assert json.dumps(real_kb) == snapshot


def test_kb_file_on_disk_is_never_mutated(real_kb):
    before = KB_PATH.read_bytes()

    build_effective_kb(real_kb, PolicyVersion.POST_AMENDMENT)

    assert KB_PATH.read_bytes() == before


def test_changed_clause_ids_on_real_corpus(real_kb):
    assert changed_clause_ids(real_kb) == frozenset({
        "§6.4.1",
        "§4.3.2",
        "§9.1.4",
        "§6.6.1",
        "§10.5.2",
        "§10.5.3A",
    })


# ---------------------------------------------------------------------------
# Anchors and labels
# ---------------------------------------------------------------------------


def test_reporting_amendments_use_change_occurred_anchor():
    by_paragraph = {op.paragraph: op for op in AMENDMENT_OPERATIONS}

    assert (
        by_paragraph["2.1"].anchor is TemporalAnchor.CHANGE_OCCURRED
    )
    assert (
        by_paragraph["2.2"].anchor is TemporalAnchor.CHANGE_OCCURRED
    )


def test_determination_amendments_use_determination_anchor():
    by_paragraph = {op.paragraph: op for op in AMENDMENT_OPERATIONS}

    for paragraph in ("1.1", "3.1", "4.1", "4.2"):
        assert (
            by_paragraph[paragraph].anchor
            is TemporalAnchor.DETERMINATION_DATE
        )


def test_version_labels_derive_from_effective_date():
    assert format_date(AMENDMENT_EFFECTIVE_DATE) == "1 March 2026"

    assert version_label(PolicyVersion.PRE_AMENDMENT) == "BEFORE 1 March 2026"
    assert (
        version_label(PolicyVersion.POST_AMENDMENT)
        == "ON OR AFTER 1 March 2026"
    )

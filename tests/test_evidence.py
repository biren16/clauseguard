from modules.evidence import (
    EvidenceResult,
    EvidenceStatus,
    build_reverse_reference_index,
    evidence_results_to_clauses,
    expand_for_conflict_check,
    is_sufficient,
    verify_quote,
    normalize_for_comparison,
)


def make_result(
    clause_id: str,
    status: str,
    covers: str = "",
    evidence_quote: str = "",
) -> EvidenceResult:
    return EvidenceResult(
        clause_id=clause_id,
        status=EvidenceStatus(status),
        covers=covers,
        evidence_quote=evidence_quote,
        reasoning="test",
    )


# ---------------------------------------------------------------------------
# Sufficiency
# ---------------------------------------------------------------------------


def test_one_supported_clause_is_sufficient():
    results = [
        make_result(
            "§4.3.2",
            "SUPPORTED",
            "reporting deadline",
        ),
    ]

    sufficient, evidence = is_sufficient(results)

    assert sufficient is True
    assert [r.clause_id for r in evidence] == ["§4.3.2"]


def test_two_distinct_partial_clauses_are_sufficient():
    results = [
        make_result(
            "§4.3.2",
            "PARTIAL",
            "reporting deadline",
        ),
        make_result(
            "§4.3.3",
            "PARTIAL",
            "reporting method",
        ),
    ]

    sufficient, evidence = is_sufficient(results)

    assert sufficient is True
    assert {
        r.clause_id for r in evidence
    } == {"§4.3.2", "§4.3.3"}


def test_one_partial_clause_is_insufficient():
    results = [
        make_result(
            "§7.1.3",
            "PARTIAL",
            "student treatment",
        ),
    ]

    sufficient, evidence = is_sufficient(results)

    assert sufficient is False
    assert evidence == []


def test_same_partial_coverage_is_insufficient():
    results = [
        make_result(
            "§4.3.2",
            "PARTIAL",
            "reporting deadline",
        ),
        make_result(
            "§4.3.4",
            "PARTIAL",
            "reporting deadline",
        ),
    ]

    sufficient, evidence = is_sufficient(results)

    assert sufficient is False
    assert evidence == []


def test_supported_clause_keeps_qualifying_partial_alongside_it():
    results = [
        make_result(
            "§4.3.2",
            "SUPPORTED",
            "reporting deadline",
        ),
        make_result(
            "§4.3.3",
            "PARTIAL",
            "reporting method",
        ),
    ]

    sufficient, evidence = is_sufficient(results)

    assert sufficient is True
    assert {
        r.clause_id for r in evidence
    } == {"§4.3.2", "§4.3.3"}


def test_supported_clause_drops_partial_with_empty_covers():
    results = [
        make_result(
            "§4.3.2",
            "SUPPORTED",
            "reporting deadline",
        ),
        make_result(
            "§9.1.4",
            "PARTIAL",
            "",
        ),
    ]

    sufficient, evidence = is_sufficient(results)

    assert sufficient is True
    assert {
        r.clause_id for r in evidence
    } == {"§4.3.2"}


def test_irrelevant_result_is_not_sufficient():
    result = make_result(
        "§9.1.4",
        "IRRELEVANT",
        "reporting deadline",
        "30 calendar days",
    )

    sufficient, evidence = is_sufficient([result])

    assert sufficient is False
    assert evidence == []


# ---------------------------------------------------------------------------
# Quote verification
# ---------------------------------------------------------------------------


def test_normalize_for_comparison_collapses_whitespace():
    assert normalize_for_comparison(
        "hello   world\nthis\tis"
    ) == "hello world this is"


def test_normalize_for_comparison_removes_markdown_bold():
    assert normalize_for_comparison(
        "within **10 calendar days**"
    ) == "within 10 calendar days"


def test_verify_quote_allows_whitespace_differences():
    clause = (
        "A recipient must report a change within "
        "10 calendar days of the change."
    )

    quote = (
        "A recipient must report a change within\n"
        "10 calendar days of the change."
    )

    assert verify_quote(quote, clause) is True


def test_verify_quote_allows_markdown_bold_difference():
    clause = (
        "A recipient must report a change within "
        "**10 calendar days** of the change."
    )

    quote = (
        "A recipient must report a change within "
        "10 calendar days of the change."
    )

    assert verify_quote(quote, clause) is True


def test_verify_quote_accepts_markdown_bold_in_quote():
    clause = (
        "A recipient must report a change within "
        "10 calendar days of the change."
    )

    quote = (
        "A recipient must report a change within "
        "**10 calendar days** of the change."
    )

    assert verify_quote(quote, clause) is True


def test_verify_quote_rejects_text_not_in_clause():
    clause = (
        "A recipient must report a change within 10 days."
    )

    quote = (
        "A recipient must report a change within 30 days."
    )

    assert verify_quote(quote, clause) is False


def test_verify_quote_rejects_empty_quote():
    assert verify_quote(
        "",
        "Some clause text.",
    ) is False


# ---------------------------------------------------------------------------
# EvidenceResult -> knowledge-base clause adapter
# ---------------------------------------------------------------------------


def test_evidence_results_to_clauses_maps_by_clause_id():
    results = [
        make_result(
            "§4.3.2",
            "SUPPORTED",
            "reporting deadline",
            "10 calendar days",
        ),
    ]

    clauses = [
        {
            "id": "§4.3.2",
            "part": "Part 4",
            "section": "4.3",
            "text": "Report within 10 calendar days.",
            "cross_references": [],
        },
    ]

    mapped = evidence_results_to_clauses(
        results,
        clauses,
    )

    assert len(mapped) == 1
    assert mapped[0]["id"] == "§4.3.2"
    assert mapped[0]["cross_references"] == []


def test_evidence_results_to_clauses_ignores_missing_clause_ids():
    results = [
        make_result(
            "§does.not.exist",
            "SUPPORTED",
            "something",
            "something",
        ),
    ]

    clauses = [
        {
            "id": "§4.3.2",
            "text": "Some clause.",
            "cross_references": [],
        },
    ]

    mapped = evidence_results_to_clauses(
        results,
        clauses,
    )

    assert mapped == []


# ---------------------------------------------------------------------------
# Reverse-reference index
# ---------------------------------------------------------------------------


def test_build_reverse_reference_index():
    clauses = [
        {
            "id": "§9.1.4",
            "cross_references": ["§4.3"],
        },
        {
            "id": "§4.3.2",
            "cross_references": [],
        },
        {
            "id": "§4.3.3",
            "cross_references": ["§8.5"],
        },
    ]

    reverse_index = build_reverse_reference_index(clauses)

    assert reverse_index == {
        "§4.3": ["§9.1.4"],
        "§8.5": ["§4.3.3"],
    }


# ---------------------------------------------------------------------------
# Conflict expansion
# ---------------------------------------------------------------------------


def test_conflict_expansion_keeps_structurally_linked_clause():
    clauses = [
        {
            "id": "§4.3.2",
            "text": (
                "A recipient must report a change "
                "within 10 calendar days."
            ),
            "cross_references": [],
        },
        {
            "id": "§9.1.4",
            "text": (
                "Where an overpayment has arisen from "
                "a change of circumstances..."
            ),
            "cross_references": ["§4.3"],
        },
    ]

    reverse_index = build_reverse_reference_index(clauses)

    evidence = [
        clauses[0],
    ]

    expanded = expand_for_conflict_check(
        evidence=evidence,
        clauses=clauses,
        reverse_index=reverse_index,
    )

    ids = {clause["id"] for clause in expanded}

    assert "§4.3.2" in ids
    assert "§9.1.4" in ids


def test_conflict_expansion_adds_reverse_reference_provenance():
    clauses = [
        {
            "id": "§4.3.2",
            "text": "Report within 10 calendar days.",
            "cross_references": [],
        },
        {
            "id": "§9.1.4",
            "text": "30 calendar days required under §4.3.",
            "cross_references": ["§4.3"],
        },
    ]

    reverse_index = build_reverse_reference_index(clauses)

    expanded = expand_for_conflict_check(
        evidence=[clauses[0]],
        clauses=clauses,
        reverse_index=reverse_index,
    )

    by_id = {
        clause["id"]: clause
        for clause in expanded
    }

    assert by_id["§9.1.4"]["conflict_reason"] == "reverse_reference"
    assert by_id["§9.1.4"]["linked_to"] == "§4.3.2"
    assert by_id["§9.1.4"]["reference"] == "§4.3"


def test_conflict_expansion_marks_original_evidence():
    clauses = [
        {
            "id": "§4.3.2",
            "text": "Report within 10 calendar days.",
            "cross_references": [],
        },
    ]

    reverse_index = build_reverse_reference_index(clauses)

    expanded = expand_for_conflict_check(
        evidence=[clauses[0]],
        clauses=clauses,
        reverse_index=reverse_index,
    )

    assert len(expanded) == 1
    assert expanded[0]["conflict_reason"] == "evidence"
    assert expanded[0]["linked_to"] is None
    assert expanded[0]["reference"] is None


def test_conflict_expansion_adds_forward_reference():
    clauses = [
        {
            "id": "§7.1.3",
            "text": (
                "Full-time students are treated under §5.4."
            ),
            "cross_references": ["§5.4"],
        },
        {
            "id": "§5.4.1",
            "text": "Care allowance is disregarded.",
            "cross_references": [],
        },
        {
            "id": "§5.4.2",
            "text": (
                "Care allowance does not alter household composition."
            ),
            "cross_references": [],
        },
    ]

    reverse_index = build_reverse_reference_index(clauses)

    expanded = expand_for_conflict_check(
        evidence=[clauses[0]],
        clauses=clauses,
        reverse_index=reverse_index,
    )

    by_id = {
        clause["id"]: clause
        for clause in expanded
    }

    assert "§7.1.3" in by_id
    assert "§5.4.1" in by_id
    assert "§5.4.2" in by_id

    assert by_id["§5.4.1"]["conflict_reason"] == "forward_reference"
    assert by_id["§5.4.1"]["linked_to"] == "§7.1.3"
    assert by_id["§5.4.1"]["reference"] == "§5.4"
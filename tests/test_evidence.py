from modules.evidence import (
    EvidenceResult,
    build_reverse_reference_index,
    expand_for_conflict_check,
    is_sufficient,
)


def make_result(
    clause_id: str,
    status: str,
    covers: str = "",
    evidence_quote: str = "",
) -> EvidenceResult:
    return EvidenceResult(
        clause_id=clause_id,
        status=status,
        covers=covers,
        evidence_quote=evidence_quote,
        reasoning="test",
    )

def test_one_supported_clause_is_sufficient():
    results = [
        make_result("§4.3.2", "SUPPORTED", "reporting deadline"),
    ]

    sufficient, evidence = is_sufficient(results)

    assert sufficient is True
    assert [r.clause_id for r in evidence] == ["§4.3.2"]


def test_two_distinct_partial_clauses_are_sufficient():
    results = [
        make_result("§4.3.2", "PARTIAL", "reporting deadline"),
        make_result("§4.3.3", "PARTIAL", "reporting method"),
    ]

    sufficient, evidence = is_sufficient(results)

    assert sufficient is True
    assert len(evidence) == 2


def test_one_partial_clause_is_insufficient():
    results = [
        make_result("§7.1.3", "PARTIAL", "student treatment"),
    ]

    sufficient, evidence = is_sufficient(results)

    assert sufficient is False
    assert evidence == []


def test_same_partial_coverage_is_insufficient():
    results = [
        make_result("§4.3.2", "PARTIAL", "reporting deadline"),
        make_result("§4.3.4", "PARTIAL", "reporting deadline"),
    ]

    sufficient, evidence = is_sufficient(results)

    assert sufficient is False
    assert evidence == []


def test_reverse_reference_index_finds_incoming_reference():
    clauses = [
        {
            "id": "§4.3.2",
            "cross_references": [],
        },
        {
            "id": "§9.1.4",
            "cross_references": ["§4.3"],
        },
    ]

    reverse_index = build_reverse_reference_index(clauses)

    assert "§9.1.4" in reverse_index["§4.3"]


def test_conflict_expansion_keeps_structurally_linked_clause():
    evidence = [
        {
            "id": "§4.3.2",
            "cross_references": [],
        }
    ]

    clauses = [
        {
            "id": "§4.3.2",
            "cross_references": [],
        },
        {
            "id": "§9.1.4",
            "cross_references": ["§4.3"],
        },
    ]

    reverse_index = build_reverse_reference_index(clauses)

    conflict_set = expand_for_conflict_check(
        evidence,
        clauses,
        reverse_index,
    )

    ids = {clause["id"] for clause in conflict_set}

    assert "§4.3.2" in ids
    assert "§9.1.4" in ids


def test_supported_clause_keeps_qualifying_partial_alongside_it():
    results = [
        make_result("§4.3.2", "SUPPORTED", "reporting deadline"),
        make_result("§4.3.3", "PARTIAL", "reporting method"),
    ]

    sufficient, evidence = is_sufficient(results)

    assert sufficient is True
    assert {r.clause_id for r in evidence} == {
        "§4.3.2",
        "§4.3.3",
    }


def test_supported_clause_drops_partial_with_empty_covers():
    results = [
        make_result("§4.3.2", "SUPPORTED", "reporting deadline"),
        make_result("§9.1.4", "PARTIAL", ""),
    ]

    sufficient, evidence = is_sufficient(results)

    assert sufficient is True
    assert {r.clause_id for r in evidence} == {"§4.3.2"}
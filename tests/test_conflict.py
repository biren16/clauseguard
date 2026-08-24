import json

from modules.conflict import (
    ConflictCandidate,
    ConflictStatus,
    ConflictType,
    _extract_numbers,
    _extract_numeric_requirements,
    _parse_bool,
    check_pair_for_conflict,
    detect_conflicts,
    find_numeric_disagreement_pairs,
)
from modules.evidence import (
    EvidenceResult,
    EvidenceStatus,
)


# ---------------------------------------------------------------------------
# Test model
# ---------------------------------------------------------------------------

class FakeModel:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def generate(
        self,
        *,
        system_prompt,
        user_prompt,
        json_mode=True,
    ):
        self.calls += 1
        return self.response


# ---------------------------------------------------------------------------
# Numeric extraction
# ---------------------------------------------------------------------------

def test_section_references_are_not_numbers():
    text = (
        "See §4.3.2 and §9.1.4 for the relevant provisions."
    )

    assert _extract_numbers(text) == set()


def test_actual_deadline_numbers_are_detected():
    text = (
        "The recipient must report "
        "within 10 calendar days."
    )

    assert _extract_numbers(text) == {"10"}

    requirements = _extract_numeric_requirements(text)

    assert requirements


def test_section_reference_and_real_number_are_separated():
    text = (
        "The recipient must report within 30 calendar days "
        "under §4.3."
    )

    assert _extract_numbers(text) == {"30"}

    requirements = _extract_numeric_requirements(text)

    assert requirements


def test_multiple_real_numbers_are_detected():
    text = (
        "The recipient must report within 10 days "
        "and provide supporting documents within 20 days."
    )

    numbers = _extract_numbers(text)

    assert numbers == {"10", "20"}


# ---------------------------------------------------------------------------
# Numeric requirement filtering
# ---------------------------------------------------------------------------

def test_non_normative_numeric_clause_is_filtered():
    candidates = [
        ConflictCandidate(
            clause_id="§1.2.2",
            text=(
                "Cross-references in this manual are "
                "given in the form §4.3.2."
            ),
            source="structural",
        ),
        ConflictCandidate(
            clause_id="§9.1.4",
            text=(
                "The recipient must report within "
                "30 calendar days."
            ),
            source="evidence",
        ),
    ]

    assert find_numeric_disagreement_pairs(
        candidates
    ) == []


def test_submission_method_does_not_become_conflict_candidate():
    candidates = [
        ConflictCandidate(
            clause_id="§4.3.2",
            text=(
                "A recipient must report the change "
                "within 10 calendar days."
            ),
            source="evidence",
        ),
        ConflictCandidate(
            clause_id="§4.3.3",
            text=(
                "A report under §4.3.2 may be submitted "
                "online or by post."
            ),
            source="structural",
        ),
    ]

    assert find_numeric_disagreement_pairs(
        candidates
    ) == []


def test_clause_with_only_section_numbers_has_no_requirements():
    text = (
        "This provision refers to §4.3 and §7.1.3."
    )

    assert _extract_numeric_requirements(text) == set()


# ---------------------------------------------------------------------------
# Candidate-pair generation
# ---------------------------------------------------------------------------

def test_known_10_vs_30_conflict_reaches_candidate_stage():
    candidates = [
        ConflictCandidate(
            clause_id="§4.3.2",
            text=(
                "A recipient must report any change "
                "within 10 calendar days."
            ),
            source="evidence",
        ),
        ConflictCandidate(
            clause_id="§9.1.4",
            text=(
                "A recipient must report the change "
                "within 30 calendar days."
            ),
            source="evidence",
        ),
    ]

    pairs = find_numeric_disagreement_pairs(
        candidates
    )

    assert len(pairs) == 1

    first, second = pairs[0]

    assert {
        first.clause_id,
        second.clause_id,
    } == {
        "§4.3.2",
        "§9.1.4",
    }


def test_same_numeric_requirement_is_not_candidate():
    candidates = [
        ConflictCandidate(
            clause_id="§1",
            text="Report within 10 calendar days.",
            source="evidence",
        ),
        ConflictCandidate(
            clause_id="§2",
            text="Notify the office within 10 calendar days.",
            source="evidence",
        ),
    ]

    assert find_numeric_disagreement_pairs(
        candidates
    ) == []


def test_clause_without_number_is_not_candidate():
    candidates = [
        ConflictCandidate(
            clause_id="§1",
            text="The recipient must report the change promptly.",
            source="evidence",
        ),
        ConflictCandidate(
            clause_id="§2",
            text="The recipient must report within 30 calendar days.",
            source="evidence",
        ),
    ]

    assert find_numeric_disagreement_pairs(
        candidates
    ) == []


# ---------------------------------------------------------------------------
# Strict boolean parsing
# ---------------------------------------------------------------------------

def test_parse_bool_accepts_real_true():
    assert _parse_bool(True) is True


def test_parse_bool_accepts_real_false():
    assert _parse_bool(False) is False


def test_parse_bool_rejects_string_true():
    try:
        _parse_bool("true")
    except ValueError:
        pass
    else:
        raise AssertionError(
            "_parse_bool must reject string 'true'"
        )


def test_parse_bool_rejects_string_false():
    try:
        _parse_bool("false")
    except ValueError:
        pass
    else:
        raise AssertionError(
            "_parse_bool must reject string 'false'"
        )


def test_parse_bool_rejects_integer():
    try:
        _parse_bool(1)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "_parse_bool must reject integer values"
        )


# ---------------------------------------------------------------------------
# Pair-level conflict detection
# ---------------------------------------------------------------------------

def test_same_scope_numeric_conflict_is_confirmed():
    a = ConflictCandidate(
        clause_id="§4.3.2",
        text="Report within 10 calendar days.",
        source="evidence",
    )

    b = ConflictCandidate(
        clause_id="§9.1.4",
        text="Report within 30 calendar days.",
        source="evidence",
    )

    model = FakeModel(
        json.dumps(
            {
                "same_scope": True,
                "conflict": True,
                "conflict_type": "numeric_scope_conflict",
                "reasoning": (
                    "Both clauses impose different reporting "
                    "deadlines for the same change."
                ),
            }
        )
    )

    result = check_pair_for_conflict(
        a=a,
        b=b,
        model=model,
    )

    assert result is not None
    assert result.status == ConflictStatus.CONFIRMED
    assert (
        result.conflict_type
        == ConflictType.NUMERIC_SCOPE_CONFLICT
    )

    assert {
        result.clause_a,
        result.clause_b,
    } == {
        "§4.3.2",
        "§9.1.4",
    }


def test_different_scope_numeric_difference_is_not_conflict():
    a = ConflictCandidate(
        clause_id="§1",
        text=(
            "General applicants must report "
            "within 10 calendar days."
        ),
        source="evidence",
    )

    b = ConflictCandidate(
        clause_id="§2",
        text=(
            "Applicants in the special emergency "
            "category have 30 days."
        ),
        source="evidence",
    )

    model = FakeModel(
        json.dumps(
            {
                "same_scope": False,
                "conflict": False,
                "conflict_type": None,
                "reasoning": (
                    "The clauses apply to different "
                    "recipient categories."
                ),
            }
        )
    )

    result = check_pair_for_conflict(
        a=a,
        b=b,
        model=model,
    )

    assert result is None


def test_same_scope_without_conflict_returns_none():
    a = ConflictCandidate(
        clause_id="§1",
        text="Report within 10 calendar days.",
        source="evidence",
    )

    b = ConflictCandidate(
        clause_id="§2",
        text=(
            "The office may review the report "
            "within 30 calendar days."
        ),
        source="evidence",
    )

    model = FakeModel(
        json.dumps(
            {
                "same_scope": True,
                "conflict": False,
                "conflict_type": None,
                "reasoning": (
                    "The numbers refer to different "
                    "requirements."
                ),
            }
        )
    )

    result = check_pair_for_conflict(
        a=a,
        b=b,
        model=model,
    )

    assert result is None


# ---------------------------------------------------------------------------
# Citation mismatch
# ---------------------------------------------------------------------------

def test_citation_mismatch_is_confirmed():
    """
    This is the planted real-world fixture.

    §4.3.2 establishes the operative 10-day requirement.

    §9.1.4 attributes a 30-day requirement to §4.3.

    The clauses have different immediate purposes, but the second clause
    makes a numerically inconsistent claim about the cited provision.
    """

    a = ConflictCandidate(
        clause_id="§4.3.2",
        text=(
            "A recipient must report any change in "
            "household composition, income, address, "
            "or circumstances within 10 calendar days."
        ),
        source="evidence",
    )

    b = ConflictCandidate(
        clause_id="§9.1.4",
        text=(
            "Where an overpayment has arisen from a "
            "change of circumstances, the recipient "
            "reported the change within the 30 calendar "
            "days required under §4.3."
        ),
        source="reverse_reference",
    )

    model = FakeModel(
        json.dumps(
            {
                "same_scope": False,
                "conflict": True,
                "conflict_type": "citation_mismatch",
                "reasoning": (
                    "§9.1.4 attributes a 30-day reporting "
                    "requirement to §4.3, while §4.3.2 "
                    "states a 10-day reporting requirement."
                ),
            }
        )
    )

    result = check_pair_for_conflict(
        a=a,
        b=b,
        model=model,
    )

    assert result is not None
    assert result.status == ConflictStatus.CONFIRMED
    assert (
        result.conflict_type
        == ConflictType.CITATION_MISMATCH
    )


# ---------------------------------------------------------------------------
# Fail-closed parsing
# ---------------------------------------------------------------------------

def test_missing_conflict_key_is_unresolved():
    a = ConflictCandidate(
        clause_id="§4.3.2",
        text="Report within 10 calendar days.",
        source="evidence",
    )

    b = ConflictCandidate(
        clause_id="§9.1.4",
        text="Report within 30 calendar days.",
        source="evidence",
    )

    model = FakeModel(
        json.dumps(
            {
                "same_scope": True,
                "reasoning": (
                    "Both clauses concern reporting."
                ),
            }
        )
    )

    result = check_pair_for_conflict(
        a=a,
        b=b,
        model=model,
    )

    assert result is not None
    assert result.status == ConflictStatus.UNRESOLVED
    assert result.reasoning.startswith("UNRESOLVED:")


def test_string_boolean_response_is_unresolved():
    a = ConflictCandidate(
        clause_id="§4.3.2",
        text="Report within 10 calendar days.",
        source="evidence",
    )

    b = ConflictCandidate(
        clause_id="§9.1.4",
        text="Report within 30 calendar days.",
        source="evidence",
    )

    model = FakeModel(
        json.dumps(
            {
                "same_scope": "false",
                "conflict": "false",
                "conflict_type": None,
                "reasoning": (
                    "Invalid string booleans."
                ),
            }
        )
    )

    result = check_pair_for_conflict(
        a=a,
        b=b,
        model=model,
    )

    assert result is not None
    assert result.status == ConflictStatus.UNRESOLVED
    assert result.reasoning.startswith("UNRESOLVED:")


def test_invalid_conflict_type_is_unresolved():
    a = ConflictCandidate(
        clause_id="§1",
        text="Report within 10 calendar days.",
        source="evidence",
    )

    b = ConflictCandidate(
        clause_id="§2",
        text="Report within 30 calendar days.",
        source="evidence",
    )

    model = FakeModel(
        json.dumps(
            {
                "same_scope": True,
                "conflict": True,
                "conflict_type": "made_up_type",
                "reasoning": "Invalid type.",
            }
        )
    )

    result = check_pair_for_conflict(
        a=a,
        b=b,
        model=model,
    )

    assert result is not None
    assert result.status == ConflictStatus.UNRESOLVED


# ---------------------------------------------------------------------------
# End-to-end deterministic conflict detection
# ---------------------------------------------------------------------------

def test_detect_conflict_with_realistic_evidence():
    evidence = [
        EvidenceResult(
            clause_id="§4.3.2",
            status=EvidenceStatus.SUPPORTED,
            covers="reporting deadline",
            evidence_quote=(
                "A recipient must report any change in "
                "income within 10 calendar days."
            ),
            reasoning="Supported.",
        ),
        EvidenceResult(
            clause_id="§9.1.4",
            status=EvidenceStatus.SUPPORTED,
            covers="change reporting deadline",
            evidence_quote=(
                "reported the change within the "
                "30 calendar days required under §4.3"
            ),
            reasoning="Supported.",
        ),
    ]

    model = FakeModel(
        json.dumps(
            {
                "same_scope": False,
                "conflict": True,
                "conflict_type": "citation_mismatch",
                "reasoning": (
                    "§9.1.4 attributes 30 days to §4.3, "
                    "while §4.3.2 states 10 days."
                ),
            }
        )
    )

    conflicts = detect_conflicts(
        evidence=evidence,
        structural_clauses=[],
        model=model,
        max_model_calls=3,
    )

    assert len(conflicts) == 1

    conflict = conflicts[0]

    assert conflict.status == ConflictStatus.CONFIRMED

    assert (
        conflict.conflict_type
        == ConflictType.CITATION_MISMATCH
    )

    assert {
        conflict.clause_a,
        conflict.clause_b,
    } == {
        "§4.3.2",
        "§9.1.4",
    }

    assert "10" in conflict.quote_a
    assert "30" in conflict.quote_b


# ---------------------------------------------------------------------------
# Model-call budget
# ---------------------------------------------------------------------------

def test_detect_conflicts_respects_model_call_budget():
    evidence = [
        EvidenceResult(
            clause_id="§1",
            status=EvidenceStatus.SUPPORTED,
            covers="deadline",
            evidence_quote="Report within 10 days.",
            reasoning="Supported.",
        ),
        EvidenceResult(
            clause_id="§2",
            status=EvidenceStatus.SUPPORTED,
            covers="deadline",
            evidence_quote="Report within 20 days.",
            reasoning="Supported.",
        ),
        EvidenceResult(
            clause_id="§3",
            status=EvidenceStatus.SUPPORTED,
            covers="deadline",
            evidence_quote="Report within 30 days.",
            reasoning="Supported.",
        ),
    ]

    model = FakeModel(
        json.dumps(
            {
                "same_scope": True,
                "conflict": True,
                "conflict_type": "numeric_scope_conflict",
                "reasoning": "Different deadlines.",
            }
        )
    )

    detect_conflicts(
        evidence=evidence,
        structural_clauses=[],
        model=model,
        max_model_calls=1,
    )

    assert model.calls == 1


# ---------------------------------------------------------------------------
# Regression: markdown bold (**) in KB clause text must not break extraction
# ---------------------------------------------------------------------------

def test_markdown_bold_does_not_break_number_extraction():
    """
    §9.1.4's KB text contains '**30 calendar days**'.

    The ** markers were previously placed immediately adjacent to the
    digit, breaking the \\b word-boundary anchor in the regex patterns
    and causing _extract_numbers / _extract_numeric_requirements to
    return empty sets.

    After the fix, the ** markers are stripped before matching.
    """
    # Exact text from the KB for §9.1.4
    text = (
        "Where an overpayment has arisen from a change of circumstances, "
        "and the recipient reported the change within the "
        "**30 calendar days** required under §4.3, no overpayment "
        "shall be established in respect of any period before the date "
        "on which the Department was in a position to act on the report."
    )

    assert "30" in _extract_numbers(text)
    assert _extract_numeric_requirements(text)  # must be non-empty


def test_markdown_bold_clause_reaches_conflict_candidate_stage():
    """
    When §9.1.4's KB text is used as the conflict candidate text, the
    pair with §4.3.2 must survive the numeric pre-filter and reach the
    LLM call stage.

    Before the fix this produced 0 pairs because ** broke the patterns.
    """
    # §4.3.2 full KB text (no markdown bold around the number here,
    # but let's use the actual form with bold to be realistic)
    clause_432_text = (
        "A recipient must report any change in household composition, "
        "income, address, or the circumstances of any household member "
        "within **10 calendar days** of the change occurring, or within "
        "10 calendar days of the recipient becoming aware of the change, "
        "whichever is later."
    )
    # §9.1.4 full KB text with bold
    clause_914_text = (
        "Where an overpayment has arisen from a change of circumstances, "
        "and the recipient reported the change within the "
        "**30 calendar days** required under §4.3, no overpayment "
        "shall be established in respect of any period before the date "
        "on which the Department was in a position to act on the report."
    )

    candidates = [
        ConflictCandidate(
            clause_id="§4.3.2",
            text=clause_432_text,
            source="evidence",
        ),
        ConflictCandidate(
            clause_id="§9.1.4",
            text=clause_914_text,
            source="evidence",
        ),
    ]

    pairs = find_numeric_disagreement_pairs(candidates)

    assert len(pairs) == 1
    ids = {pairs[0][0].clause_id, pairs[0][1].clause_id}
    assert ids == {"§4.3.2", "§9.1.4"}


# ---------------------------------------------------------------------------
# Regression: evidence_quote snippet must not replace full KB text in conflict
# ---------------------------------------------------------------------------

def test_build_candidates_uses_full_kb_text_not_evidence_quote():
    """
    build_candidates must look up full KB clause text from structural_clauses
    for evidence candidates, NOT use the truncated evidence_quote snippet.

    The evidence_quote is a short LLM-generated snippet for answer generation.
    It may not contain the keyword context (within/must/required) that the
    numeric pre-filter patterns need.

    Example: the LLM might return:
        "time limit to report a change of circumstances (30 calendar days)"
    This passes no numeric requirement pattern — so without the fix, the pair
    is silently dropped before the LLM conflict call.
    """
    from modules.conflict import build_candidates

    # Simulate the short evidence_quote the LLM actually generated
    short_quote_914 = "time limit to report a change of circumstances (30 calendar days)"

    # Simulate what structural_clauses contains (full KB texts)
    structural_clauses = [
        {
            "id": "§4.3.2",
            "text": (
                "A recipient must report any change in household composition, "
                "income, address, or circumstances within 10 calendar days."
            ),
            "cross_references": [],
            "conflict_reason": "evidence",
        },
        {
            "id": "§9.1.4",
            "text": (
                "Where an overpayment has arisen from a change of "
                "circumstances, and the recipient reported the change within "
                "the **30 calendar days** required under §4.3, no overpayment "
                "shall be established."
            ),
            "cross_references": ["§4.3"],
            "conflict_reason": "evidence",
        },
    ]

    evidence = [
        EvidenceResult(
            clause_id="§4.3.2",
            status=EvidenceStatus.SUPPORTED,
            covers="reporting deadline",
            evidence_quote="reporting a change in income within 10 calendar days",
            reasoning="",
        ),
        EvidenceResult(
            clause_id="§9.1.4",
            status=EvidenceStatus.SUPPORTED,
            covers="30 day limit",
            evidence_quote=short_quote_914,
            reasoning="",
        ),
    ]

    candidates = build_candidates(
        evidence=evidence,
        structural_clauses=structural_clauses,
    )

    # Candidates should use full KB text, not the snippets
    by_id = {c.clause_id: c for c in candidates}
    assert "§4.3.2" in by_id
    assert "§9.1.4" in by_id

    # The §9.1.4 candidate text must be the full KB text, not the short quote
    assert short_quote_914 not in by_id["§9.1.4"].text
    assert "30 calendar days" in by_id["§9.1.4"].text
    assert "overpayment" in by_id["§9.1.4"].text

    # Most importantly: the pair must now survive the pre-filter
    pairs = find_numeric_disagreement_pairs(candidates)
    assert len(pairs) == 1
    ids = {pairs[0][0].clause_id, pairs[0][1].clause_id}
    assert ids == {"§4.3.2", "§9.1.4"}
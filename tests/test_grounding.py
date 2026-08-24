from modules.evidence import EvidenceResult, EvidenceStatus
from modules.grounding import validate_grounding


def make_retained(clause_id: str) -> EvidenceResult:
    return EvidenceResult(
        clause_id=clause_id,
        status=EvidenceStatus.SUPPORTED,
        covers="coverage",
        evidence_quote="some quote",
        reasoning="reason",
    )


def test_valid_citations_pass():
    retained = [make_retained("§4.3.2")]
    result = validate_grounding(
        answer="Report within 10 days under §4.3.2.",
        citations=["§4.3.2"],
        retained_evidence=retained,
    )

    assert result.is_valid is True
    assert result.invalid_citations == []


def test_unretained_citation_fails():
    retained = [make_retained("§4.3.2")]
    result = validate_grounding(
        answer="Report within 30 days under §9.1.4.",
        citations=["§9.1.4"],
        retained_evidence=retained,
    )

    assert result.is_valid is False
    assert result.invalid_citations == ["§9.1.4"]


def test_empty_citations_fails():
    retained = [make_retained("§4.3.2")]
    result = validate_grounding(
        answer="Report within 10 days.",
        citations=[],
        retained_evidence=retained,
    )

    assert result.is_valid is False


def test_empty_answer_fails():
    retained = [make_retained("§4.3.2")]
    result = validate_grounding(
        answer="",
        citations=["§4.3.2"],
        retained_evidence=retained,
    )

    assert result.is_valid is False

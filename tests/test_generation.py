import json
from modules.evidence import EvidenceResult, EvidenceStatus
from modules.generation import (
    GenerationResult,
    format_evidence_for_generation,
    generate_answer,
)


class FakeModel:
    def __init__(self, response: str):
        self.response = response

    def generate(self, *, system_prompt: str, user_prompt: str, json_mode: bool = True) -> str:
        return self.response


def test_format_evidence_for_generation():
    evidence = [
        EvidenceResult(
            clause_id="§4.3.2",
            status=EvidenceStatus.SUPPORTED,
            covers="reporting deadline",
            evidence_quote="must report within 10 calendar days",
            reasoning="Supported.",
        )
    ]

    formatted = format_evidence_for_generation(evidence)
    assert "§4.3.2" in formatted
    assert "must report within 10 calendar days" in formatted


def test_generate_answer_success():
    evidence = [
        EvidenceResult(
            clause_id="§4.3.2",
            status=EvidenceStatus.SUPPORTED,
            covers="reporting deadline",
            evidence_quote="must report within 10 calendar days",
            reasoning="Supported.",
        )
    ]

    model = FakeModel(
        json.dumps(
            {
                "answer": "You must report changes within 10 calendar days pursuant to §4.3.2.",
                "citations": ["§4.3.2"],
            }
        )
    )

    result = generate_answer("How long to report?", evidence, model)

    assert result.answer == "You must report changes within 10 calendar days pursuant to §4.3.2."
    assert result.citations == ["§4.3.2"]


def test_generate_answer_malformed_json():
    evidence = [
        EvidenceResult(
            clause_id="§4.3.2",
            status=EvidenceStatus.SUPPORTED,
            covers="reporting deadline",
            evidence_quote="must report within 10 calendar days",
            reasoning="Supported.",
        )
    ]

    model = FakeModel("Not valid JSON")

    result = generate_answer("How long to report?", evidence, model)

    assert result.answer == ""
    assert result.citations == []


def test_generate_answer_empty_evidence():
    model = FakeModel("{}")
    result = generate_answer("How long to report?", [], model)

    assert result.answer == ""
    assert result.citations == []

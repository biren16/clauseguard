from __future__ import annotations

from dataclasses import dataclass
import json

from modules.evidence import EvidenceResult
from modules.evidence_model import (
    EvidenceModel,
    EvidenceModelProviderError,
    EvidenceModelRateLimitError,
)


@dataclass
class GenerationResult:
    """
    Structured result of answer generation.
    """

    answer: str
    citations: list[str]
    raw_response: str = ""


GENERATION_PROMPT = """
You are a policy assistant for the Household Support Program.

Your task is to answer the user's question using ONLY the provided verified policy evidence quotes.

CRITICAL RULES:
1. Base your answer STRICTLY and SOLELY on the provided evidence quotes below.
2. Do NOT use any outside knowledge, assumptions, or inferences about policy.
3. Every factual claim in your answer must be directly supported by the provided text.
4. Include exact clause ID citations (e.g. §4.3.2) in your answer text where appropriate.
5. In the "citations" array, list all clause IDs that were directly cited or relied upon to construct the answer.

USER QUESTION:
{question}

VERIFIED POLICY EVIDENCE:
{evidence_text}

Return ONLY valid JSON matching this schema:
{{
  "answer": "your detailed grounded answer here with inline citations like §4.3.2",
  "citations": ["§4.3.2"]
}}
"""


def format_evidence_for_generation(evidence: list[EvidenceResult]) -> str:
    """
    Format retained evidence results for inclusion in the generation prompt.

    Only verified quotes from retained evidence results are included.
    No candidate pool text, unretained clauses, or reasoning strings are passed.
    """

    blocks = []
    for result in evidence:
        quote = result.evidence_quote.strip()
        if quote:
            blocks.append(f"Clause {result.clause_id}:\n{quote}")

    return "\n\n".join(blocks)


def generate_answer(
    question: str,
    evidence: list[EvidenceResult],
    model: EvidenceModel,
) -> GenerationResult:
    """
    Generate a grounded answer and citations from verified retained evidence.

    Raises provider errors if infrastructure calls fail.
    Returns empty/invalid GenerationResult if model output is malformed.
    """

    if not evidence:
        return GenerationResult(
            answer="",
            citations=[],
            raw_response="No evidence provided to generation module.",
        )

    evidence_text = format_evidence_for_generation(evidence)

    prompt = GENERATION_PROMPT.format(
        question=question,
        evidence_text=evidence_text,
    )

    try:
        raw_response = model.generate(
            system_prompt="",
            user_prompt=prompt,
            json_mode=True,
        ).strip()
    except (EvidenceModelRateLimitError, EvidenceModelProviderError):
        raise

    try:
        parsed = json.loads(raw_response)

        if not isinstance(parsed, dict):
            return GenerationResult(
                answer="",
                citations=[],
                raw_response=raw_response,
            )

        answer = parsed.get("answer", "")
        citations_raw = parsed.get("citations", [])

        if not isinstance(answer, str):
            answer = ""

        if not isinstance(citations_raw, list):
            citations_raw = []

        citations = [
            str(c).strip()
            for c in citations_raw
            if isinstance(c, (str, int, float)) and str(c).strip()
        ]

        return GenerationResult(
            answer=answer.strip(),
            citations=citations,
            raw_response=raw_response,
        )

    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return GenerationResult(
            answer="",
            citations=[],
            raw_response=raw_response,
        )

from __future__ import annotations

from abc import ABC, abstractmethod
import os

from groq import Groq


class EvidenceModel(ABC):
    """
    Provider-independent interface for the evidence classifier.

    The evidence pipeline only knows that it can provide:
        system_prompt + user_prompt

    and receive:
        raw model text

    All safety-critical decisions remain outside this class:
      - JSON validation
      - status validation
      - evidence quote verification
      - sufficiency
      - structural expansion
      - conflict detection
    """

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        raise NotImplementedError


class GroqEvidenceModel(EvidenceModel):
    """
    Groq implementation of the provider-independent evidence model.

    This class is deliberately thin. It only handles communication
    with Groq and returns the raw response.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "openai/gpt-oss-20b",
    ):
        self.api_key = api_key or os.environ["GROQ_API_KEY"]

        self.client = Groq(
            api_key=self.api_key,
        )

        self.model = model

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=0,
            max_tokens=500,
            response_format={
                "type": "json_object",
            },
        )

        content = response.choices[0].message.content

        if not content:
            raise RuntimeError(
                "Groq returned an empty response."
            )

        return content
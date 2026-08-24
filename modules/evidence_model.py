from __future__ import annotations

from abc import ABC, abstractmethod
import os

from groq import (
    APIConnectionError,
    APIStatusError,
    Groq,
    RateLimitError,
)


class EvidenceModelError(RuntimeError):
    """Base error for model/provider failures."""


class EvidenceModelRateLimitError(EvidenceModelError):
    """Groq rate or token limit was reached."""


class EvidenceModelConnectionError(EvidenceModelError):
    """The provider could not be reached."""


class EvidenceModelProviderError(EvidenceModelError):
    """The provider returned another API-level failure."""


class EvidenceModel(ABC):
    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = True,
    ) -> str:
        raise NotImplementedError


class GroqEvidenceModel(EvidenceModel):
    """
    Thin Groq adapter.

    The rest of the application does not know about Groq's API.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_retries: int = 0,
        timeout: float = 30.0,
    ):
        self.api_key = api_key or os.environ["GROQ_API_KEY"]

        self.client = Groq(
            api_key=self.api_key,
            max_retries=max_retries,
            timeout=timeout,
        )

        self.model = (
            model
            or os.environ.get(
                "GROQ_MODEL",
                "openai/gpt-oss-20b",
            )
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = True,
    ) -> str:
        request = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "temperature": 0,
            "max_tokens": 500,
        }

        if json_mode:
            request["response_format"] = {
                "type": "json_object",
            }

        try:
            response = self.client.chat.completions.create(
                **request,
            )

        except RateLimitError as exc:
            raise EvidenceModelRateLimitError(
                "Groq rate/token limit reached."
            ) from exc

        except APIConnectionError as exc:
            raise EvidenceModelConnectionError(
                "Could not connect to Groq API."
            ) from exc

        except APIStatusError as exc:
            raise EvidenceModelProviderError(
                f"Groq API error: {exc}"
            ) from exc

        content = response.choices[0].message.content

        if not content:
            raise EvidenceModelProviderError(
                "Groq returned an empty response."
            )

        return content
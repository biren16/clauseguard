"""
Optional offline utility for generating Gemini embeddings during manual ingestion.

Not used during runtime Q&A pipeline or CLI execution.
"""

from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "gemini-embedding-001"


def create_embedding(text: str, client=None) -> list[float]:
    """Create an embedding vector for a piece of text (offline KB ingestion only)."""
    if client is None:
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")

        from google import genai
        client = genai.Client(api_key=api_key)

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )

    return response.embeddings[0].values
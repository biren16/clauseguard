import json
from pathlib import Path

from modules.embeddings import create_embedding


KNOWLEDGE_BASE_PATH = Path("data/knowledge_base.json")


def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:
    """Calculate cosine similarity between two vectors."""

    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = sum(a * a for a in vector_a) ** 0.5
    magnitude_b = sum(b * b for b in vector_b) ** 0.5

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


def retrieve(
    question: str,
    top_k: int = 10,
) -> list[dict]:
    """Retrieve the most semantically similar policy clauses."""

    knowledge_base = json.loads(
        KNOWLEDGE_BASE_PATH.read_text(encoding="utf-8")
    )

    question_embedding = create_embedding(question)

    scored_clauses = []

    for clause in knowledge_base:
        score = cosine_similarity(
            question_embedding,
            clause["embedding"],
        )

        scored_clauses.append(
            {
                **clause,
                "similarity": score,
            }
        )

    scored_clauses.sort(
        key=lambda clause: clause["similarity"],
        reverse=True,
    )

    return scored_clauses[:top_k]
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


def load_knowledge_base() -> list[dict]:
    """Load the locally generated knowledge base."""

    return json.loads(
        KNOWLEDGE_BASE_PATH.read_text(encoding="utf-8")
    )


def semantic_retrieve(
    question: str,
    knowledge_base: list[dict],
    top_k: int = 10,
) -> list[dict]:
    """Retrieve clauses by semantic similarity."""

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
                "retrieval_reason": "semantic",
            }
        )

    scored_clauses.sort(
        key=lambda clause: clause["similarity"],
        reverse=True,
    )

    return scored_clauses[:top_k]


def is_related_reference(
    candidate_id: str,
    reference: str,
) -> bool:
    """
    Determine whether a clause belongs to a referenced section.

    Example:
        §4.3.2 is related to §4.3
    """

    return (
        candidate_id == reference
        or candidate_id.startswith(reference + ".")
    )

def expand_cross_references(
    candidates: list[dict],
    knowledge_base: list[dict],
) -> list[dict]:
    """
    Expand candidates using explicit cross-references.

    If a candidate references another clause, include that clause.
    Also include clauses that reference the candidate itself.
    """

    by_id = {
        clause["id"]: clause
        for clause in knowledge_base
    }

    expanded: dict[str, dict] = {
        clause["id"]: clause
        for clause in candidates
    }

    for candidate in candidates:
        candidate_id = candidate["id"]

        # Follow references made by this clause.
        for reference in candidate["cross_references"]:
            referenced_clause = by_id.get(reference)

            if referenced_clause:
                expanded.setdefault(
                    reference,
                    {
                        **referenced_clause,
                        "similarity": None,
                        "retrieval_reason": "cross_reference",
                    },
                )

        # Find clauses that reference this candidate.
        # or its parent section.
        for clause in knowledge_base:
            if any(
                is_related_reference(candidate_id, reference)
                or is_related_reference(reference, candidate_id)
                for reference in clause["cross_references"]
            ):
                expanded.setdefault(
                    clause["id"],
                    {
                        **clause,
                        "similarity": None,
                        "retrieval_reason": "cross_reference",
                    },
                )

    return list(expanded.values())


def retrieve(
    question: str,
    top_k: int = 10,
) -> list[dict]:
    """
    Retrieve a candidate pool using semantic search
    followed by cross-reference expansion.
    """

    knowledge_base = load_knowledge_base()

    semantic_candidates = semantic_retrieve(
        question,
        knowledge_base,
        top_k=top_k,
    )

    return expand_cross_references(
        semantic_candidates,
        knowledge_base,
    )
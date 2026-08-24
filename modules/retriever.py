import json
import math
import re
from collections import Counter
from pathlib import Path


KNOWLEDGE_BASE_PATH = Path("data/knowledge_base.json")

# Standard minimal English stopwords to filter non-informative query tokens
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below",
    "to", "from", "up", "down", "out", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "s", "t", "can", "will", "just", "don",
    "should", "now", "i", "my", "me", "we", "our", "you", "your", "do",
    "does", "did", "have", "has", "had", "would", "could", "shall",
}


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric and section reference tokens."""
    return re.findall(r"[a-z0-9§\.]+", text.lower())


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


class BM25Retriever:
    """Deterministic, local BM25 ranking over policy clauses."""

    def __init__(self, corpus: list[dict], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.docs: list[Counter] = []
        self.doc_len: list[int] = []
        self.doc_freqs: Counter = Counter()
        self.N = len(corpus)

        for clause in corpus:
            text_to_index = (
                f"{clause.get('text', '')} "
                f"{clause.get('section', '')} "
                f"{clause.get('part', '')} "
                f"{clause.get('id', '')}"
            )
            tokens = _tokenize(text_to_index)
            self.doc_len.append(len(tokens))
            term_counts = Counter(tokens)
            self.docs.append(term_counts)
            for term in set(tokens):
                self.doc_freqs[term] += 1

        self.avgdl = (sum(self.doc_len) / self.N) if self.N > 0 else 1.0

    def score(self, query: str) -> list[tuple[float, int]]:
        query_tokens = [
            t for t in _tokenize(query)
            if t not in _STOPWORDS
        ]

        # If query only contains stopwords, fall back to all tokens
        if not query_tokens:
            query_tokens = _tokenize(query)

        scored: list[tuple[float, int]] = []

        for i, doc in enumerate(self.docs):
            score = 0.0
            dl = self.doc_len[i]
            for t in query_tokens:
                if t not in doc:
                    continue
                df = self.doc_freqs[t]
                idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
                tf = doc[t]
                num = tf * (self.k1 + 1)
                denom = tf + self.k1 * (1 - self.b + self.b * (dl / self.avgdl))
                score += idf * (num / denom)
            scored.append((score, i))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored


def bm25_retrieve(
    question: str,
    knowledge_base: list[dict],
    top_k: int = 15,
) -> list[dict]:
    """Retrieve clauses using local deterministic BM25 ranking."""

    retriever = BM25Retriever(knowledge_base)
    scored_indices = retriever.score(question)

    scored_clauses = []
    for score, idx in scored_indices[:top_k]:
        clause = knowledge_base[idx]
        scored_clauses.append(
            {
                **clause,
                "similarity": score,
                "retrieval_reason": "lexical",
            }
        )

    return scored_clauses


def semantic_retrieve(
    question: str,
    knowledge_base: list[dict],
    top_k: int = 15,
) -> list[dict]:
    """
    Standard candidate retrieval interface.

    Uses deterministic local BM25 ranking over knowledge base clauses,
    requiring zero external API calls or credentials.
    """
    return bm25_retrieve(
        question=question,
        knowledge_base=knowledge_base,
        top_k=top_k,
    )


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
        for reference in candidate.get("cross_references", []):
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

        # Find clauses that reference this candidate or its parent section.
        for clause in knowledge_base:
            if any(
                is_related_reference(candidate_id, reference)
                or is_related_reference(reference, candidate_id)
                for reference in clause.get("cross_references", [])
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
    top_k: int = 15,
) -> list[dict]:
    """
    Retrieve a candidate pool using local deterministic retrieval
    followed by cross-reference expansion.
    """

    knowledge_base = load_knowledge_base()

    candidates = semantic_retrieve(
        question,
        knowledge_base,
        top_k=top_k,
    )

    return expand_cross_references(
        candidates,
        knowledge_base,
    )
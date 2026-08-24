from modules.retriever import (
    BM25Retriever,
    bm25_retrieve,
    cosine_similarity,
    expand_cross_references,
    semantic_retrieve,
)


def test_cross_reference_matches_parent_section():
    candidates = [
        {
            "id": "§4.3.2",
            "cross_references": [],
            "similarity": 0.9,
        }
    ]

    knowledge_base = [
        {
            "id": "§4.3.2",
            "cross_references": [],
        },
        {
            "id": "§9.1.4",
            "cross_references": ["§4.3"],
        },
    ]

    expanded = expand_cross_references(
        candidates,
        knowledge_base,
    )

    ids = {clause["id"] for clause in expanded}

    assert "§9.1.4" in ids


def test_bm25_retriever_ranking():
    kb = [
        {
            "id": "§1.1.1",
            "part": "Part 1",
            "section": "Eligibility",
            "text": "A person is eligible for support if household income is below threshold.",
            "cross_references": [],
        },
        {
            "id": "§4.3.2",
            "part": "Part 4",
            "section": "Reporting Deadlines",
            "text": "Must report change of circumstances within 10 calendar days.",
            "cross_references": [],
        },
        {
            "id": "§9.1.4",
            "part": "Part 9",
            "section": "Overpayments",
            "text": "Overpayment rules apply if reporting occurs after 30 days under §4.3.",
            "cross_references": ["§4.3"],
        },
    ]

    results = bm25_retrieve("how long do I have to report a change?", kb, top_k=2)

    assert len(results) == 2
    assert results[0]["id"] == "§4.3.2"
    assert results[0]["retrieval_reason"] == "lexical"
    assert results[0]["similarity"] > 0


def test_cosine_similarity():
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    v3 = [0.0, 1.0, 0.0]

    assert abs(cosine_similarity(v1, v2) - 1.0) < 1e-6
    assert abs(cosine_similarity(v1, v3) - 0.0) < 1e-6
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_semantic_retrieve_requires_no_credentials(monkeypatch):
    # Ensure GEMINI_API_KEY is not set
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    kb = [
        {
            "id": "§7.1.3",
            "part": "Part 7",
            "section": "Needs",
            "text": "Needs calculation for full-time students.",
            "cross_references": [],
        }
    ]

    results = semantic_retrieve("full-time student", kb, top_k=1)
    assert len(results) == 1
    assert results[0]["id"] == "§7.1.3"
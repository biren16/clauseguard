from modules.retriever import expand_cross_references


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
import json

from dotenv import load_dotenv

from modules.evidence import classify_candidates
from modules.evidence_model import GroqEvidenceModel


load_dotenv(override=True)

model = GroqEvidenceModel()


with open("data/knowledge_base.json", "r", encoding="utf-8") as f:
    knowledge_base = json.load(f)


by_id = {
    clause["id"]: clause
    for clause in knowledge_base
}


TEST_CASES = [
    (
        "I started a new job. How long do I have to tell the office?",
        [
            "§4.3.2",
            "§4.3.3",
            "§9.1.4",
        ],
    ),
    (
        "How is the needs figure calculated for a household with a full-time student?",
        [
            "§7.1.3",
            "§5.4.1",
            "§5.4.2",
        ],
    ),
]


for question, clause_ids in TEST_CASES:
    print("\n" + "=" * 80)
    print(question)
    print("=" * 80)

    candidates = [
        by_id[clause_id]
        for clause_id in clause_ids
        if clause_id in by_id
    ]

    results = classify_candidates(
        question=question,
        candidates=candidates,
        model=model,
    )

    print("\nRESULTS:\n")

    for result in results:
        print(result.clause_id)
        print(f"  status: {result.status.value}")
        print(f"  covers: {result.covers or '-'}")
        print(f"  quote:  {result.evidence_quote or '-'}")
        print(f"  why:    {result.reasoning}")
        print()
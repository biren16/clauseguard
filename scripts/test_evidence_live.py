import json
import os

from google import genai

from modules.evidence import classify_candidates


with open("data/knowledge_base.json", encoding="utf-8") as f:
    clauses = json.load(f)


client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)


TEST_CASES = [
    (
        "I started a new job. How long do I have to tell the office?",
        ["§4.3.2", "§9.1.4", "§4.3.3"],
    ),
    (
        "How is the needs figure calculated for a household with a full-time student?",
        ["§7.1.3", "§5.4"],
    ),
]


for question, clause_ids in TEST_CASES:
    print("\n" + "=" * 80)
    print(question)
    print("=" * 80)

    candidates = [
        clause
        for clause in clauses
        if clause["id"] in clause_ids
    ]

    results = classify_candidates(
        question=question,
        candidates=candidates,
        client=client,
    )

    print("\nRESULTS:")

    for result in results:
        print(f"\n{result.clause_id}")
        print(f"  status: {result.status.value}")
        print(f"  covers: {result.covers}")
        print(f"  quote:  {result.evidence_quote}")
        print(f"  why:    {result.reasoning}")
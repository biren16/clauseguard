import json
from pathlib import Path

from modules.embeddings import create_embedding
from modules.parser import parse_manual


MANUAL_PATH = Path("data/policy-manual.md")
KNOWLEDGE_BASE_PATH = Path("data/knowledge_base.json")


def build_knowledge_base() -> None:
    """Parse the policy manual, embed each clause, and save the result."""

    markdown = MANUAL_PATH.read_text(encoding="utf-8")
    clauses = parse_manual(markdown)

    records = []

    for index, clause in enumerate(clauses, start=1):
        print(f"Embedding clause {index}/{len(clauses)}: {clause.id}")

        embedding = create_embedding(clause.text)

        records.append(
            {
                "id": clause.id,
                "part": clause.part,
                "section": clause.section,
                "text": clause.text,
                "cross_references": clause.cross_references,
                "embedding": embedding,
            }
        )

    KNOWLEDGE_BASE_PATH.write_text(
        json.dumps(records, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved {len(records)} clauses to {KNOWLEDGE_BASE_PATH}")


if __name__ == "__main__":
    build_knowledge_base()
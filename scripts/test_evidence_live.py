import json

from dotenv import load_dotenv

from modules.conflict import (
    ConflictStatus,
    detect_conflicts,
)
from modules.evidence import (
    build_reverse_reference_index,
    classify_candidates,
    evidence_results_to_clauses,
    expand_for_conflict_check,
    is_sufficient,
)
from modules.evidence_model import GroqEvidenceModel


load_dotenv(override=True)

model = GroqEvidenceModel(
    max_retries=0,
)


# ------------------------------------------------------------------
# Load knowledge base
# ------------------------------------------------------------------

with open(
    "data/knowledge_base.json",
    "r",
    encoding="utf-8",
) as f:
    knowledge_base = json.load(f)


by_id = {
    clause["id"]: clause
    for clause in knowledge_base
}


reverse_index = build_reverse_reference_index(
    knowledge_base
)


# ------------------------------------------------------------------
# Test cases
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# Run
# ------------------------------------------------------------------

for question, clause_ids in TEST_CASES:

    print("\n" + "=" * 80)
    print(question)
    print("=" * 80)

    candidates = [
        by_id[clause_id]
        for clause_id in clause_ids
        if clause_id in by_id
    ]

    print(
        f"\nCLASSIFYING {len(candidates)} evidence candidates...",
        flush=True,
    )

    for candidate in candidates:
        print(
            f"  → {candidate['id']}",
            flush=True,
        )

    # --------------------------------------------------------------
    # Evidence
    # --------------------------------------------------------------

    results = classify_candidates(
        question=question,
        candidates=candidates,
        model=model,
    )

    print(
        "EVIDENCE CLASSIFICATION COMPLETE",
        flush=True,
    )

    print("\nEVIDENCE:\n")

    for result in results:
        print(result.clause_id)
        print(
            f"  status: {result.status.value}"
        )
        print(
            f"  covers: {result.covers or '-'}"
        )
        print(
            f"  quote:  {result.evidence_quote or '-'}"
        )
        print(
            f"  why:    {result.reasoning}"
        )
        print()

    # --------------------------------------------------------------
    # Sufficiency
    # --------------------------------------------------------------

    sufficient, retained_results = is_sufficient(
        results
    )

    print("SUFFICIENCY:")
    print(
        f"  sufficient: {sufficient}"
    )
    print(
        f"  retained:   {len(retained_results)}"
    )
    print()

    if not sufficient:
        print(
            "Evidence is insufficient; "
            "skipping conflict analysis."
        )
        continue

    # --------------------------------------------------------------
    # Restore KB clauses
    # --------------------------------------------------------------

    retained_clauses = evidence_results_to_clauses(
        retained_results,
        knowledge_base,
    )

    print("RETAINED CLAUSES:")

    for clause in retained_clauses:
        print(
            f"  {clause['id']}"
        )

    print()

    # --------------------------------------------------------------
    # Structural expansion
    # --------------------------------------------------------------

    structural_clauses = expand_for_conflict_check(
        retained_clauses,
        knowledge_base,
        reverse_index,
    )

    print("STRUCTURAL CONFLICT SET:\n")

    if not structural_clauses:
        print("  none")
    else:
        for clause in structural_clauses:
            print(
                f"  {clause['id']}: "
                f"{clause.get('conflict_reason', 'structural')}"
            )

    print()

    # --------------------------------------------------------------
    # Conflict analysis
    # --------------------------------------------------------------

    conflicts = detect_conflicts(
        evidence=retained_results,
        structural_clauses=structural_clauses,
        model=model,
        max_model_calls=3,
    )

    print("\nCONFLICT RESULTS:\n")

    if not conflicts:
        print("  none")
        continue

    confirmed = [
        conflict
        for conflict in conflicts
        if conflict.status == ConflictStatus.CONFIRMED
    ]

    unresolved = [
        conflict
        for conflict in conflicts
        if conflict.status == ConflictStatus.UNRESOLVED
    ]

    if confirmed:
        print("CONFIRMED CONFLICTS:\n")

        for conflict in confirmed:

            conflict_type = (
                conflict.conflict_type.value
                if conflict.conflict_type
                else "unknown"
            )

            print(
                f"  {conflict.clause_a} <-> "
                f"{conflict.clause_b}"
            )
            print(
                f"    type:    {conflict_type}"
            )
            print(
                f"    quote A: {conflict.quote_a}"
            )
            print(
                f"    quote B: {conflict.quote_b}"
            )
            print(
                f"    why:     {conflict.reasoning}"
            )
            print()

    if unresolved:
        print("UNRESOLVED CHECKS:\n")

        for conflict in unresolved:
            print(
                f"  {conflict.clause_a} <-> "
                f"{conflict.clause_b}"
            )
            print(
                f"    reason: {conflict.reasoning}"
            )
            print()
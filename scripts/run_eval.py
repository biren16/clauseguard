#!/usr/bin/env python
"""
ClauseGuard — 10-Question Evaluation Harness

Runs a representative set of questions through the live pipeline and reports
PASS/FAIL against expected outcomes.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/run_eval.py
"""

from __future__ import annotations

import os
import sys
import time

# Make sure repo root is on PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(override=True)

from modules.evidence_model import GroqEvidenceModel
from modules.pipeline import PipelineOutcome, run_pipeline
from modules.retriever import load_knowledge_base


# ---------------------------------------------------------------------------
# 10-question eval suite
# ---------------------------------------------------------------------------
#
# Format: (question, expected_outcome, notes)
#
# The two mandatory corpus fixtures from the spec are:
#   Fixture 5.1 — the 10-day vs 30-day contradiction  → CONFLICT
#   Fixture 5.2 — student needs calculation gap        → NO_EVIDENCE

EVAL_CASES: list[tuple[str, PipelineOutcome, str]] = [
    # 1. Direct factual query — reporting deadline (Fixture 5.1 question variant)
    (
        "I started a new job. How long do I have to tell the office?",
        PipelineOutcome.CONFLICT,
        "Fixture 5.1: §4.3.2 (10 days) vs §9.1.4 (30 days) contradiction",
    ),
    # 2. Reporting method — §4.3.3
    (
        "How can I submit a change-of-circumstances report to the Department?",
        PipelineOutcome.ANSWER,
        "§4.3.3 covers reporting methods",
    ),
    # 3. Student needs calculation gap (Fixture 5.2)
    (
        "How is the needs figure calculated for a household with a full-time student?",
        PipelineOutcome.NO_EVIDENCE,
        "Fixture 5.2: §7.1.3 defers to §5.4 for student rules; §5.4 is absent from KB",
    ),
    # 4. Eligibility — household composition
    (
        "Who counts as a household member for benefit purposes?",
        PipelineOutcome.ANSWER,
        "Part 2 / §2.1 series covers household membership",
    ),
    # 5. Income disregard — earnings
    (
        "Is any part of employment earnings ignored when calculating my benefit?",
        PipelineOutcome.ANSWER,
        "Part 6 covers disregards including earnings",
    ),
    # 6. Sanction consequence — failure to report
    (
        "What happens if I don't report a change within the required period?",
        PipelineOutcome.ANSWER,
        "§4.3.4 references §10.5 sanctions and overpayment in Part 9",
    ),
    # 7. Overpayment recovery — method
    (
        "How does the Department recover an overpayment from me?",
        PipelineOutcome.ANSWER,
        "Part 9 covers overpayment recovery",
    ),
    # 8. Appeal process
    (
        "How do I appeal a decision about my benefit?",
        PipelineOutcome.ANSWER,
        "Part 11 covers appeals",
    ),
    # 9. Out-of-scope — federal/tax
    (
        "How do I file my federal income tax return?",
        PipelineOutcome.NO_EVIDENCE,
        "Completely outside the scope of the policy manual",
    ),
    # 10. Program start — initial application
    (
        "What documents do I need to apply for the Household Support Program?",
        PipelineOutcome.ANSWER,
        "Part 3 covers applications and required documentation",
    ),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def main() -> None:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY is not set.")
        sys.exit(1)

    print()
    print("=" * 72)
    print("  ClauseGuard — Evaluation Harness")
    print(f"  {len(EVAL_CASES)} questions")
    print("=" * 72)

    model = GroqEvidenceModel(max_retries=0)

    print("  Loading knowledge base...", end="", flush=True)
    knowledge_base = load_knowledge_base()
    print(f" {len(knowledge_base)} clauses.")
    print()

    results = []

    for idx, (question, expected, notes) in enumerate(EVAL_CASES, start=1):
        label = f"Q{idx:02d}"
        print(f"  {label}  {question}")
        print(f"       Expected: {expected.value}  — {notes}")

        t0 = time.time()
        try:
            result = run_pipeline(
                question=question,
                model=model,
                knowledge_base=knowledge_base,
            )
            actual = result.outcome
            error_msg = None
        except Exception as exc:
            actual = None
            error_msg = str(exc)
        elapsed = time.time() - t0

        passed = (actual == expected)
        results.append((label, question, expected, actual, passed, error_msg, elapsed))

        status = "PASS ✔" if passed else "FAIL ✖"
        actual_str = actual.value if actual else f"ERROR: {error_msg}"
        print(f"       Actual:   {actual_str}   [{elapsed:.1f}s]   {status}")
        print()

    # Summary table
    print("=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    total = len(results)
    passed_count = sum(1 for r in results if r[4])
    print(f"  {passed_count}/{total} questions passed")
    print()
    print(f"  {'Q':>4}  {'Expected':>14}  {'Actual':>14}  {'Time':>6}  Result")
    print(f"  {'-'*4}  {'-'*14}  {'-'*14}  {'-'*6}  ------")
    for label, _, expected, actual, passed, error_msg, elapsed in results:
        actual_str = actual.value if actual else "ERROR"
        status = "PASS" if passed else "FAIL"
        print(f"  {label:>4}  {expected.value:>14}  {actual_str:>14}  {elapsed:5.1f}s  {status}")
    print()
    print(f"  Accuracy: {passed_count}/{total} = {100*passed_count//total}%")
    print("=" * 72)
    print()

    sys.exit(0 if passed_count == total else 1)


if __name__ == "__main__":
    main()

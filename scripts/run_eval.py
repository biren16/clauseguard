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
# Format: (question, acceptable_outcomes, notes)
#
# The two mandatory corpus fixtures from the spec are:
#   Fixture 5.1 — the 10-day vs 30-day contradiction  → CONFLICT (historical)
#   Fixture 5.2 — student needs calculation gap        → NO_EVIDENCE
#
# Since Amendment No. 2026-01 (effective 1 March 2026), UNDATED questions
# that reach amendment-changed provisions are evaluated against BOTH policy
# states. When the versions lead to materially different outcomes the result
# is TEMPORAL_AMBIGUITY; when every version produces materially the same
# grounded answer, that shared ANSWER is returned directly.

EVAL_CASES: list[tuple[str, set[PipelineOutcome], str]] = [
    # 1. Direct factual query — reporting deadline (Fixture 5.1 question variant)
    (
        "I started a new job. How long do I have to tell the office?",
        {PipelineOutcome.TEMPORAL_AMBIGUITY},
        "Fixture 5.1, undated: historical state conflicts (§4.3.2 10d vs "
        "§9.1.4 30d); amended state aligns at 14d — date required",
    ),
    # 2. Reporting method — §4.3.3
    (
        "How can I submit a change-of-circumstances report to the Department?",
        {PipelineOutcome.ANSWER},
        "§4.3.3 covers reporting methods (unchanged by the amendment)",
    ),
    # 3. Student needs calculation gap (Fixture 5.2)
    (
        "How is the needs figure calculated for a household with a full-time student?",
        {PipelineOutcome.NO_EVIDENCE},
        "Fixture 5.2: §7.1.3 defers to §5.4 for student rules; §5.4 is absent from KB",
    ),
    # 4. Eligibility — household composition
    (
        "Who counts as a household member for benefit purposes?",
        {PipelineOutcome.ANSWER},
        "Part 2 / §2.1 series covers household membership",
    ),
    # 5. Income disregard — earnings
    (
        "Is any part of employment earnings ignored when calculating my benefit?",
        {PipelineOutcome.TEMPORAL_AMBIGUITY},
        "§6.4.1 disregard changed $120 -> $175 on 1 March 2026; the undated "
        "question sees both figures — date required",
    ),
    # 6. Sanction consequence — failure to report
    (
        "What happens if I don't report a change within the required period?",
        {PipelineOutcome.TEMPORAL_AMBIGUITY},
        "Touches §4.3.2/§9.1.4 reporting periods, which differ per version; "
        "the historical state contains the planted contradiction",
    ),
    # 7. Overpayment recovery — method
    (
        "How does the Department recover an overpayment from me?",
        {PipelineOutcome.ANSWER, PipelineOutcome.TEMPORAL_AMBIGUITY},
        "Part 9 covers recovery methods; §9.1.4's text was amended, so "
        "branch answers may coincide or diverge depending on what the "
        "generator cites",
    ),
    # 8. Appeal process
    (
        "How do I appeal a decision about my benefit?",
        {PipelineOutcome.ANSWER},
        "Part 11 covers appeals (unchanged by the amendment)",
    ),
    # 9. Out-of-scope — federal/tax
    (
        "How do I file my federal income tax return?",
        {PipelineOutcome.NO_EVIDENCE},
        "Completely outside the scope of the policy manual",
    ),
    # 10. Program start — initial application
    (
        "What documents do I need to apply for the Household Support Program?",
        {PipelineOutcome.ANSWER},
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

    model = GroqEvidenceModel()

    print("  Loading knowledge base...", end="", flush=True)
    knowledge_base = load_knowledge_base()
    print(f" {len(knowledge_base)} clauses.")
    print()

    results = []

    for idx, (question, expected, notes) in enumerate(EVAL_CASES, start=1):
        label = f"Q{idx:02d}"
        print(f"  {label}  {question}")
        expected_str = "/".join(o.value for o in sorted(expected, key=lambda o: o.value))
        print(f"       Expected: {expected_str}  — {notes}")

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

        passed = (actual in expected)
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
        expected_str = "/".join(o.value for o in sorted(expected, key=lambda o: o.value))
        status = "PASS" if passed else "FAIL"
        print(f"  {label:>4}  {expected_str:>32}  {actual_str:>20}  {elapsed:5.1f}s  {status}")
    print()
    print(f"  Accuracy: {passed_count}/{total} = {100*passed_count//total}%")
    print("=" * 72)
    print()

    sys.exit(0 if passed_count == total else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
ClauseGuard — Grounded Policy Question Answering CLI

Usage:
    python main.py "Your question here?"
    python main.py --date 2026-04-10 "Your question here?"
    python main.py          # interactive mode
"""

from __future__ import annotations

import os
import sys
import textwrap

from dotenv import load_dotenv

load_dotenv(override=True)

from modules.evidence_model import GroqEvidenceModel
from modules.pipeline import PipelineOutcome, run_pipeline
from modules.retriever import load_knowledge_base
from modules.policy_versioning import (
    AMENDMENT_EFFECTIVE_DATE,
    AMENDMENT_NUMBER,
    AMENDMENT_OPERATIONS,
    ANCHOR_DESCRIPTIONS,
    TemporalContext,
    format_date,
    parse_date_string,
    resolve_temporal_context,
    version_label,
)
from modules.pipeline import BRANCH_ORDER


# ---------------------------------------------------------------------------
# ANSI colour helpers (degrade gracefully when not a tty)
# ---------------------------------------------------------------------------


def _is_tty() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


_USE_COLOUR = _is_tty()


def _col(code: str, text: str) -> str:
    if not _USE_COLOUR:
        return text
    return f"\033[{code}m{text}\033[0m"


def green(t: str) -> str:  return _col("32;1", t)
def red(t: str)   -> str:  return _col("31;1", t)
def yellow(t: str)-> str:  return _col("33;1", t)
def cyan(t: str)  -> str:  return _col("36;1", t)
def bold(t: str)  -> str:  return _col("1", t)
def dim(t: str)   -> str:  return _col("2", t)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _hr(char: str = "─", width: int = 72) -> str:
    return dim(char * width)


def _wrap(text: str, indent: int = 2, width: int = 70) -> str:
    prefix = " " * indent
    return textwrap.fill(text, width=width, initial_indent=prefix,
                         subsequent_indent=prefix)


def _render_branch_outcome(result) -> None:
    """Render one branch's outcome using the standard iconography."""

    if result.outcome == PipelineOutcome.ANSWER:
        print(green("    ✔  ANSWER"))
        print(_wrap(result.answer, indent=6))
        if result.citations:
            print(bold("    Citations:"))
            for cid in result.citations:
                print(f"      {cyan(cid)}")

    elif result.outcome == PipelineOutcome.CONFLICT:
        print(red("    ✖  POLICY CONFLICT DETECTED — ANSWER WITHHELD"))
        print(_wrap(result.refusal_reason, indent=6))
        for conflict in result.conflicts:
            status_label = conflict.status.value
            print(
                f"      {yellow(conflict.clause_a)} ↔ "
                f"{yellow(conflict.clause_b)}  [{status_label}]"
            )
            print(_wrap(conflict.quote_a, indent=8))
            print(_wrap(conflict.quote_b, indent=8))

    else:
        print(yellow("    ⚠  INSUFFICIENT EVIDENCE — ANSWER WITHHELD"))
        print(_wrap(result.refusal_reason, indent=6))


def render_temporal_ambiguity(result) -> None:
    """
    Deterministically render a TEMPORAL_AMBIGUITY result.

    Every statement shown here is derived from either the amendment
    constants (modules.policy_versioning) or the independently evaluated
    branch results. No LLM composes this response and no facts are
    invented: each branch's answer, citations, conflicts and refusals are
    exactly that branch's own verified pipeline output.
    """

    effective = format_date(AMENDMENT_EFFECTIVE_DATE)

    print(yellow("  ⚠  TEMPORAL AMBIGUITY — DATE REQUIRED FOR A DEFINITIVE ANSWER"))
    print(_hr())
    print()
    print(_wrap(
        f"Your question touches provisions changed by Amendment No. "
        f"{AMENDMENT_NUMBER}, which took effect on {effective}. The "
        f"applicable policy therefore depends on when the relevant event "
        f"happened, and no date was provided."
    ))

    anchors = {
        operation.anchor
        for operation in AMENDMENT_OPERATIONS
    }
    anchor_text = " or ".join(
        ANCHOR_DESCRIPTIONS[anchor]
        for anchor in sorted(anchors, key=lambda a: a.value)
    )
    print()
    print(_wrap(
        f"For the purposes of this amendment, the relevant date is "
        f"{anchor_text}, depending on the provision involved."
    ))
    print()

    for version in BRANCH_ORDER:
        branch = result.branch_results.get(version)

        if branch is None:
            continue

        print(dim(f"  If the relevant date was {version_label(version)}:"))
        _render_branch_outcome(branch)
        print()

    print(bold("  Next step:"))
    print(_wrap(
        "Please provide the relevant date (for example: "
        '--date 2026-04-10) so a definitive answer can be given '
        "under the correct policy version."
    ))
    print()


def render_result(result) -> None:
    print()
    print(_hr())

    if result.outcome == PipelineOutcome.ANSWER:
        print(green("  ✔  ANSWER"))
        print(_hr())
        print()
        print(_wrap(result.answer, indent=2))
        print()
        if result.citations:
            print(bold("  Citations:"))
            for cid in result.citations:
                print(f"    {cyan(cid)}")
        print()

    elif result.outcome == PipelineOutcome.CONFLICT:
        print(red("  ✖  POLICY CONFLICT DETECTED — ANSWER WITHHELD"))
        print(_hr())
        print()
        print(_wrap(result.refusal_reason, indent=2))
        print()
        if result.conflicts:
            print(bold("  Conflicting provisions:"))
            for conflict in result.conflicts:
                status_label = conflict.status.value
                print(f"    {yellow(conflict.clause_a)} ↔ {yellow(conflict.clause_b)}  [{status_label}]")
                print(_wrap(conflict.quote_a, indent=6))
                print(_wrap(conflict.quote_b, indent=6))
                print(_wrap(conflict.reasoning, indent=6))
                print()

    elif result.outcome == PipelineOutcome.NO_EVIDENCE:
        print(yellow("  ⚠  INSUFFICIENT EVIDENCE — ANSWER WITHHELD"))
        print(_hr())
        print()
        print(_wrap(result.refusal_reason, indent=2))
        print()

    elif result.outcome == PipelineOutcome.TEMPORAL_AMBIGUITY:
        render_temporal_ambiguity(result)

    print(_hr())
    print()


def render_banner() -> None:
    print()
    print(bold("  ClauseGuard") + dim("  — Grounded Policy Q&A"))
    print(dim("  Household Support Program Policy Manual"))
    print()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _run_question(question: str, model, knowledge_base: list,
                  explicit_date=None) -> None:
    print()
    print(bold("  Question:") + f"  {question}")
    try:
        result = run_pipeline(
            question=question,
            model=model,
            knowledge_base=knowledge_base,
            explicit_date=explicit_date,
        )
    except ValueError as exc:
        # Defensive: conflicting temporal inputs should already have been
        # rejected before the pipeline was invoked.
        print(red(f"  INPUT ERROR: {exc}"))
        sys.exit(2)

    render_result(result)


def main() -> None:
    render_banner()

    import argparse
    parser = argparse.ArgumentParser(description="ClauseGuard - Grounded Policy Q&A")
    parser.add_argument(
        "--date",
        type=str,
        help=(
            "Explicit temporal context used to select the applicable policy "
            "version (e.g. 2026-04-10, 20/02/2026, 'February 20, 2026')."
        ),
    )
    parser.add_argument("question", type=str, nargs="*", help="The question to ask")
    args = parser.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print(red("  ERROR: GROQ_API_KEY is not set. Please check your .env file."))
        sys.exit(1)

    explicit_date = None

    if args.date:
        try:
            explicit_date = parse_date_string(args.date)
        except ValueError as exc:
            print(red(f"  INPUT ERROR: {exc}"))
            sys.exit(2)

    model = GroqEvidenceModel()

    print(dim("  Loading knowledge base..."), end="", flush=True)
    knowledge_base = load_knowledge_base()
    print(dim(f" {len(knowledge_base)} clauses loaded."))

    # Single-question mode: python main.py "question"
    if args.question:
        question = " ".join(args.question).strip()
        if not question:
            print(red("  ERROR: Empty question."))
            sys.exit(1)

        # Deterministic temporal precedence: --date wins, but if the
        # question text also carries dates that select a DIFFERENT
        # policy version, refuse instead of silently choosing.
        resolution = resolve_temporal_context(
            question,
            explicit_date=explicit_date,
        )

        if resolution.context == TemporalContext.INPUT_CONFLICT:
            print(red("  INPUT ERROR: Conflicting temporal information."))
            for note in resolution.notes:
                print(_wrap(note, indent=4))
            print(_wrap(
                "Remove or correct the dates so exactly one policy "
                "version is selected.",
                indent=4,
            ))
            sys.exit(2)

        _run_question(question, model, knowledge_base, explicit_date)
        return

    # Interactive mode
    print(dim("  Type your question and press Enter.  Type 'quit' or Ctrl-C to exit."))
    print()

    while True:
        try:
            question = input(bold("  Question: ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            print(dim("  Goodbye."))
            break

        if not question:
            continue

        if question.lower() in {"quit", "exit", "q"}:
            print(dim("  Goodbye."))
            break

        _run_question(question, model, knowledge_base, explicit_date)


if __name__ == "__main__":
    main()

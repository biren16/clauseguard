#!/usr/bin/env python
"""
ClauseGuard — Grounded Policy Question Answering CLI

Usage:
    python main.py "Your question here?"
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


def _run_question(question: str, model, knowledge_base: list) -> None:
    print()
    print(bold("  Question:") + f"  {question}")
    result = run_pipeline(
        question=question,
        model=model,
        knowledge_base=knowledge_base,
    )
    render_result(result)


def main() -> None:
    render_banner()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print(red("  ERROR: GROQ_API_KEY is not set. Please check your .env file."))
        sys.exit(1)

    model = GroqEvidenceModel()

    print(dim("  Loading knowledge base..."), end="", flush=True)
    knowledge_base = load_knowledge_base()
    print(dim(f" {len(knowledge_base)} clauses loaded."))

    # Single-question mode: python main.py "question"
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:]).strip()
        if not question:
            print(red("  ERROR: Empty question."))
            sys.exit(1)
        _run_question(question, model, knowledge_base)
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

        _run_question(question, model, knowledge_base)


if __name__ == "__main__":
    main()

# ClauseGuard — Grounded Policy Question Answering

ClauseGuard answers natural-language questions about the **Calder County
Household Support Program** policy manual — and is designed to refuse when it
cannot support an answer.

## Overview

A policy manual is a domain where a fluent but wrong answer is worse than no
answer. A general-purpose LLM asked *"How long do I have to report a change of
circumstances?"* produces confident prose from pretrained priors, without
knowing whether the governing clause was retrieved, whether another provision
contradicts it, or which policy version applies on a given date.

ClauseGuard puts deterministic safety machinery around the LLM. The model is
used only where language understanding is genuinely needed — classifying
candidate clauses and expressing an answer — while every decision that
determines **what the system is allowed to say** is made by verifiable code:

```text
verified evidence → grounded generation → deterministic citation validation
```

The system is intentionally conservative:

> **Answer when the evidence supports it. Refuse when it does not.**

---

## Safety Philosophy

- **Evidence before generation.** Candidate clauses are retrieved, classified,
  and quote-verified *before* any answer is attempted. The generator never
  sees unverified material.
- **Fail-closed behavior.** Malformed model output, unverifiable quotes, and
  provider anomalies reduce the available evidence or block the answer — they
  are never silently promoted.
- **No unsupported policy claims.** The generator receives only verified
  evidence quotes and must cite the clauses it relied on; unsupported claims
  have no path into an accepted answer.
- **Contradiction → refusal.** When two provisions impose incompatible
  requirements on the same rule — including one clause misstating what another
  requires — the question cannot be settled from the manual alone and the
  answer is withheld.
- **Insufficient evidence → refusal.** If classification retains nothing that
  establishes the answer, the system says so instead of improvising.
- **Grounding validation.** Every generated answer passes a deterministic
  check that its cited clause IDs belong to the retained evidence set;
  failures are refused.

---

## Quick Start

### Requirements

- Python 3 (developed and tested on CPython 3.14)
- A Groq API key — the only runtime credential

### Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### API Configuration

Copy `.env.example` to `.env` and set your Groq key:

```bash
GROQ_API_KEY=gsk_...
GROQ_MODEL=openai/gpt-oss-20b      # optional, this is the default
```

No Google/Gemini key is needed at runtime. The derived knowledge base
(`data/knowledge_base.json`) ships with the repository, so no ingestion or
embedding step is required to run the tests or the CLI. Its `embedding` fields
are unused legacy data — runtime retrieval is local deterministic BM25 over
clause text (`modules/retriever.py`).

### Run the CLI

Single question:

```bash
PYTHONPATH=. python main.py "I started a new job. How long do I have to tell the office?"
```

### Explicit Temporal Date

To select the applicable policy version deterministically:

```bash
PYTHONPATH=. python main.py --date 2026-04-10 "I started a new job. How long do I have to tell the office?"
```

Supported date forms include `2026-04-10`, `20 February 2026`,
`February 20, 2026`, `20/02/2026` and `02/20/2026`. If the question text also
contains dates that select a *different* policy version, the CLI refuses with
a clear input error instead of silently choosing one.

### Interactive Mode

```bash
PYTHONPATH=. python main.py
```

Type questions at the prompt; `quit`, `exit`, or Ctrl-C ends the session.

### Run Tests

```bash
pytest -q
```

All tests run fully offline — no API calls, no network, no credentials.

### Run Evaluation

A live evaluation harness runs ten representative questions against the real
pipeline (requires a Groq API key):

```bash
PYTHONPATH=. .venv/bin/python scripts/run_eval.py
```

---

## Example Outcomes

All four outcomes come from the same pipeline. The sketches below reflect
actual observed behavior, abridged for readability.

### ANSWER — verified evidence, grounded response

Sufficient verified evidence exists, no contradiction is detected, and every
cited clause belongs to the retained evidence set:

```text
✔ ANSWER
You must notify the Department within 14 calendar days of the change
occurring … (§4.3.2).
Citations: §4.3.2
```

If grounding validation fails — for example, the model cites a clause outside
its evidence set — the answer is refused instead.

### NO_EVIDENCE — insufficient evidence, escalation

Classification retains nothing sufficient, so the question is refused rather
than improvised:

```text
⚠ INSUFFICIENT EVIDENCE — ANSWER WITHHELD
The policy manual does not contain sufficient verified evidence to answer
this question. Please direct the query to a supervisor or policy specialist.
```

### CONFLICT — contradictory provisions, answer withheld

Two provisions disagree about the same rule (here: the historical reporting
period versus a clause citing it). The conflict pair, quotes, and reasoning
are surfaced for human review:

```text
✖ POLICY CONFLICT DETECTED — ANSWER WITHHELD
Conflicting provisions:
  §4.3.2 ↔ §9.1.4  [CONFIRMED]
```

### TEMPORAL_AMBIGUITY — undated question, both policy states evaluated

An undated question that reaches amendment-changed provisions is evaluated
against **both** policy versions independently. Because they lead to different
outcomes here, the response shows each branch's verified conclusion side by
side and asks for the relevant date:

```text
⚠ TEMPORAL_AMBIGUITY — DATE REQUIRED FOR A DEFINITIVE ANSWER
If the relevant date was BEFORE 1 March 2026:
  ✖ POLICY CONFLICT DETECTED (§4.3.2 ↔ §9.1.4)
If the relevant date was ON OR AFTER 1 March 2026:
  ✔ ANSWER (14 calendar days, §4.3.2)
Next step: please provide the relevant date …
```

This response is composed deterministically from branch results: the LLM does
not decide that the question is temporally ambiguous, and the system never
silently assumes today's policy.

---

## Architecture

```text
User question (+ optional explicit date)
      │
      ▼
┌──────────────────────────┐
│ Temporal resolution      │  deterministic, no LLM involved:
│ modules/policy_versioning│  which policy version(s) apply?
└──────┬───────────────────┘  Amendment No. 2026-01 → PRE / POST / both
       │  effective policy state(s)
       ▼
┌──────────────────────────┐
│ Retrieval                │  local deterministic BM25 ranking
│ modules/retriever        │  + forward cross-reference expansion
└──────┬───────────────────┘
       │  candidate clauses
       ▼
┌──────────────────────────┐
│ Evidence classification  │  LLM: SUPPORTED / PARTIAL / IRRELEVANT?
│ modules/evidence         │  + quote verification (normalized
└──────┬───────────────────┘    substring check against source)
       │  EvidenceResult list
       ▼
┌──────────────────────────┐
│ Sufficiency              │  fail-closed rules:
│ modules/evidence         │  1 SUPPORTED, or 2+ distinct PARTIALs
└──────┬───────────────────┘
       │  retained evidence ─────────────────────┐
       ▼                                         │
┌──────────────────────────┐                     │
│ Structural expansion     │  forward + reverse  │
│ for conflict check       │  references         │
└──────┬───────────────────┘                     │
       │  structural clauses                     │
       ▼                                         │
┌──────────────────────────┐                     │
│ Conflict detection       │  numeric pre-filter │
│ modules/conflict         │  + LLM pair analysis│
└──────┬───────────────────┘  citation-mismatch   │
       │  no conflicts                aware       │
       ▼                                         │
┌──────────────────────────┐                     │
│ Answer generation        │  LLM: grounded JSON │
│ modules/generation       │  {answer, citations}│
└──────┬───────────────────┘ ◄───────────────────┘
       │  receives only verified retained evidence
       ▼
┌──────────────────────────┐
│ Grounding validation     │  deterministic: every cited clause ID
│ modules/grounding        │  ∈ branch's retained evidence set?
└──────┬───────────────────┘
       │
       ▼
   ANSWER / CONFLICT / NO_EVIDENCE / TEMPORAL_AMBIGUITY
```

Key invariants of this flow:

- **Temporal resolution is deterministic.** Date parsing, version selection,
  and amendment application involve no model calls; the LLM never chooses
  which policy version applies.
- **Conflict analysis sees one internally consistent policy state per run.**
  It operates on full knowledge-base clause text within that state and is
  itself temporal-agnostic.
- **Structural conflict context is not automatically answer evidence.**
  Forward/reverse reference expansion widens the *conflict-check* set only;
  those clauses reach the generator solely if independently classified as
  sufficient evidence.
- **Generation receives only verified retained evidence** from its own
  branch — quotes, scores, rejection reasoning, and other-version text are
  withheld.
- **Grounding is deterministic** — citation membership in the branch's
  retained evidence set — so an unsupported or mis-cited answer cannot ship.

---

## Pipeline Outcomes

| Outcome | Meaning |
|---|---|
| `ANSWER` | Grounded answer produced from verified evidence, with exact clause citations validated against that evidence |
| `CONFLICT` | A policy contradiction was detected within the evaluated policy state — the answer is withheld and the conflicting provisions are surfaced |
| `NO_EVIDENCE` | Verified evidence is insufficient to establish an answer — the question is refused and escalated to a human |
| `TEMPORAL_AMBIGUITY` | The applicable policy version depends on a date that was not provided, and the independently evaluated versions lead to different outcomes — both verified branch results are shown and the relevant date is requested |

---

## Temporal Policy Handling

[Amendment No. 2026-01](data/Amendment%20No.%202026-01.md) (issued 12 February
2026, effective **1 March 2026**) amends several provisions of the manual:
the reporting period (§4.3.2, §9.1.4), the earnings disregard (§6.4.1),
income thresholds (§6.6.1), and sanctions (§10.5.2, plus new clause §10.5.3A).

`modules/policy_versioning.py` applies the amendment as structured
deterministic operations over in-memory copies of the knowledge base — the
file on disk is never mutated.

### Explicit date

When a date is supplied (`--date`, or unambiguously present in the question),
exactly one effective policy state is selected and passed through the normal
pipeline:

- change on **20 February 2026** → historical state: §4.3.2 = 10 days vs
  §9.1.4 = 30 days → citation mismatch → `CONFLICT`;
- change on **10 April 2026** → amended state: §4.3.2 = 14 days,
  §9.1.4 = 14 days → no same-state conflict → grounded `ANSWER`.

These are different policy *versions*, not contradictory policies: a version
difference alone is never reported as a conflict.

### No date

The system does not silently assume today's policy. If an undated question
touches amendment-changed provisions, **both** states are evaluated fully and
independently — separate retrieval, evidence classification, sufficiency,
structural expansion, conflict analysis, generation, and grounding per branch.
If the branches agree on a single safe result, that result is returned; if
they materially differ, the outcome is `TEMPORAL_AMBIGUITY` with each branch's
verified conclusion exposed side by side.

### Branch isolation

Evidence, citations, conflict results, and grounding are never merged across
branches. Each branch's generator sees only its own state's evidence; a branch
that fails grounding stays failed and is never rescued by its sibling.

### Temporal anchors

Per the amendment's transitional provisions, different amendments attach to
different events:

| Provisions | Anchor event |
|---|---|
| §4.3.2 / §9.1.4 reporting periods | the date the change of circumstances **occurred** |
| §6.4.1 disregard, §6.6.1 thresholds, §10.5.2 / §10.5.3A sanctions | the date the **determination** was made |

Both anchor classes share the 1 March 2026 threshold, so any explicit date
resolves deterministically without guessing which kind of date it is; the
anchors remain part of the modelled amendment semantics.

### Important limitation

Whether an undated question "touches" changed provisions is decided by a
bounded deterministic probe: BM25 top-k retrieval plus cross-reference
expansion over the historical view, intersected with the amendment-changed
clause set. This probe is a performance heuristic, **not** a complete
detector. Clauses that exist only in the amended state (e.g. §10.5.3A) or
that rank beyond the retrieval window without structural links can be missed,
in which case the question is evaluated against the historical state alone.
Such misses degrade toward refusal or a single-state grounded answer — they
never bypass evidence verification, conflict detection, or grounding checks.

See [DECISIONS.md](DECISIONS.md) §33 for the full temporal design rationale.

---

## Module Reference

| Module | Purpose |
|---|---|
| `modules/parser.py` | Parse `data/policy-manual.md` into `Clause` objects |
| `modules/ingest.py` | Offline utility that built `knowledge_base.json` (embedding-based; not needed at runtime) |
| `modules/embeddings.py` | Offline embedding adapter (not used during runtime Q&A) |
| `modules/retriever.py` | Local deterministic BM25 ranking + cross-reference expansion |
| `modules/evidence_model.py` | Abstract `EvidenceModel` interface + `GroqEvidenceModel` adapter |
| `modules/evidence.py` | Evidence classification, quote verification, sufficiency rules, structural conflict expansion |
| `modules/conflict.py` | Conflict detection: numeric pre-filter + LLM pair analysis, citation-mismatch aware (temporal-agnostic) |
| `modules/generation.py` | Grounded answer generation from retained evidence only |
| `modules/grounding.py` | Deterministic citation validation |
| `modules/policy_versioning.py` | Deterministic temporal resolution and PRE/POST effective policy views for Amendment No. 2026-01 |
| `modules/pipeline.py` | End-to-end orchestration, including temporal branching and aggregation |
| `main.py` | CLI entry point |
| `scripts/run_eval.py` | Ten-question live evaluation harness |

---

## Test Suite

```text
tests/
├── test_conflict.py            27 tests  — numeric extraction, conflict analysis
├── test_embeddings.py           1 test   — embedding adapter
├── test_evidence.py            21 tests  — classification, quote verify, sufficiency
├── test_generation.py           4 tests  — generation module (offline / mocked)
├── test_grounding.py            4 tests  — grounding validation (offline)
├── test_parser.py               3 tests  — manual parsing
├── test_pipeline.py             4 tests  — end-to-end routing (offline / mocked)
├── test_policy_versioning.py   49 tests  — temporal resolution, date parsing, effective KB views
├── test_retriever.py            4 tests  — BM25 retrieval, cross-ref expansion, no credentials
└── test_temporal_pipeline.py   11 tests  — temporal branch isolation, ambiguity routing (offline)
```

All 128 tests run fully offline — no API calls, no network, no credentials
required. The suite includes regression guards for every previously observed
failure mode (unverifiable quotes, string booleans, malformed conflict output,
provider-error separation, citation hallucination) as well as the temporal
properties (boundary dates, input conflicts, branch and grounding isolation).

---

## Key Design Decisions

The full architectural rationale — including why each safety gate exists and
which limitations are accepted — is documented in
[DECISIONS.md](DECISIONS.md). Highlights:

- Fail-closed safety architecture throughout the pipeline
- Local deterministic BM25 retrieval (zero runtime external embedding dependencies)
- Citation-mismatch detection as a distinct conflict pattern
- Reverse-reference index for directional blind spots
- Deterministic temporal policy resolution with independent PRE/POST views (§33)
- Explicitly documented Day-1 scope limitations (§28)

---

## AI Usage

ClauseGuard's development used AI assistance for architecture review, prompt
design, debugging, and implementation under human direction. A complete
disclosure log — what each tool did, where AI proposals were wrong and had to
be corrected, and how claims were checked against the source corpus — is in
[AI-USAGE.md](AI-USAGE.md).

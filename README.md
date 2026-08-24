# ClauseGuard — Grounded Policy Question Answering

ClauseGuard answers natural-language questions about the **Calder County
Household Support Program** policy manual with:

- **Zero hallucination** — every claim is a verbatim quote from a verified clause.
- **Explicit refusals** — when the evidence is insufficient or contradictory,
  the system refuses to answer rather than guessing.
- **Citation-mismatch detection** — the system catches cases where two clauses
  cite contradictory numeric requirements for the same rule.

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API keys

Copy `.env.example` to `.env` and fill in your keys:

```
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=AI...
GROQ_MODEL=openai/gpt-oss-20b      # optional, this is the default
```

### 3. Run the CLI

**Single question:**
```bash
PYTHONPATH=. python main.py "I started a new job. How long do I have to tell the office?"
```

**Interactive mode:**
```bash
PYTHONPATH=. python main.py
```

### 4. Run the test suite (offline, no API keys needed)

```bash
PYTHONPATH=. .venv/bin/python -m pytest -v tests/
```

### 5. Run the live evaluation harness (requires API keys)

```bash
PYTHONPATH=. .venv/bin/python scripts/run_eval.py
```

---

## Architecture

```
User question
      │
      ▼
┌─────────────┐
│  Retrieval  │  cosine similarity over pre-embedded KB + cross-ref expansion
└──────┬──────┘
       │  candidate clauses
       ▼
┌──────────────────────┐
│  Evidence classifier │  LLM: is each clause SUPPORTED / PARTIAL / IRRELEVANT?
│                      │  + quote verification (substring check)
└──────┬───────────────┘
       │  EvidenceResult list
       ▼
┌──────────────┐
│  Sufficiency │  fail-closed rules (1 SUPPORTED or 2+ distinct PARTIALs)
└──────┬───────┘
       │  retained evidence  ──────────────────────┐
       ▼                                           │
┌──────────────────────┐                           │
│  Conflict expansion  │  forward + reverse refs   │
└──────┬───────────────┘                           │
       │  structural clauses                       │
       ▼                                           │
┌──────────────────┐                               │
│  Conflict        │  numeric pre-filter + LLM     │
│  detector        │  citation-mismatch aware      │
└──────┬───────────┘                               │
       │  no conflicts                             │
       ▼                                           │
┌──────────────────────┐   ◄───────────────────────┘
│  Answer generation   │  LLM: grounded JSON {answer, citations}
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Grounding validator │  all citations ∈ retained evidence set?
└──────┬───────────────┘
       │
       ▼
   ANSWER / CONFLICT / NO_EVIDENCE
```

### Pipeline outcomes

| Outcome | Meaning |
|---|---|
| `ANSWER` | Grounded answer with exact clause citations |
| `CONFLICT` | Detected policy contradiction — answer withheld, conflict surfaced |
| `NO_EVIDENCE` | Insufficient verified evidence — answer withheld, supervisor directed |

---

## Module Reference

| Module | Purpose |
|---|---|
| `modules/parser.py` | Parse `data/policy-manual.md` into `Clause` objects |
| `modules/ingest.py` | Embed each clause with `gemini-embedding-001` and write `knowledge_base.json` |
| `modules/embeddings.py` | Google AI embedding adapter |
| `modules/retriever.py` | Cosine-similarity semantic search + cross-reference expansion |
| `modules/evidence_model.py` | Abstract `EvidenceModel` interface + `GroqEvidenceModel` adapter |
| `modules/evidence.py` | Evidence classification, quote verification, sufficiency, conflict expansion |
| `modules/conflict.py` | Conflict detection: numeric pre-filter + LLM pair analysis |
| `modules/generation.py` | Grounded answer generation |
| `modules/grounding.py` | Deterministic citation validation |
| `modules/pipeline.py` | End-to-end orchestration |
| `main.py` | CLI entry point |
| `scripts/run_eval.py` | 10-question live evaluation harness |

---

## Test Suite

```
tests/
├── test_conflict.py     24 tests  — numeric extraction, conflict analysis
├── test_embeddings.py    1 test   — embedding adapter
├── test_evidence.py     21 tests  — classification, quote verify, sufficiency
├── test_generation.py    4 tests  — generation module (offline / mocked)
├── test_grounding.py     4 tests  — grounding validation (offline)
├── test_pipeline.py      4 tests  — end-to-end routing (offline / mocked)
├── test_parser.py        3 tests  — manual parsing
└── test_retriever.py     1 test   — cross-reference expansion
```

All 62 tests run fully offline (no API calls, no network).

---

## Key Design Decisions

See [DECISIONS.md](DECISIONS.md) for the full design rationale, including:

- Fail-closed safety architecture
- Citation-mismatch detection
- Reverse-reference index
- Day-1 scope limitations

---

## AI Usage

See [AI-USAGE.md](AI-USAGE.md) for a complete log of AI tool assistance.

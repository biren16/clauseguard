# AI Usage Log — ClauseGuard

This document records every significant use of AI assistance during the
development of ClauseGuard, per the hackathon submission requirements.

---

## 1. Architecture Design

**Tool**: Antigravity (Google DeepMind) coding assistant  
**Task**: Designing the overall pipeline structure  
**Contribution**: The AI proposed the seven-stage pipeline:
retrieve → classify → sufficiency → structural expansion → conflict detection
→ generation → grounding validation.  The fail-closed safety philosophy and
the decision to use a separate conflict module were AI-suggested.

**Human review**: The architecture was reviewed, adjusted (e.g. adding the
reverse-reference index after the first test runs revealed the §9.1.4 / §4.3.2
conflict was being missed), and approved before implementation.

---

## 2. Prompt Engineering

**Tool**: Antigravity  
**Tasks**:
- Writing the evidence classification prompt (`EVIDENCE_PROMPT` in
  `modules/evidence.py`).
- Writing the conflict detection prompt (`SCOPE_CONFLICT_PROMPT` in
  `modules/conflict.py`).
- Writing the answer generation prompt (`GENERATION_PROMPT` in
  `modules/generation.py`).

**Contribution**: The AI drafted initial prompt text.  All three prompts were
significantly iterated by hand, in particular:
- The evidence prompt was tightened to forbid paraphrasing in quotes.
- The conflict prompt was extended to handle the citation-mismatch case
  (`same_scope=false` but still a conflict) after initial tests showed the
  model was incorrectly clearing §9.1.4 / §4.3.2 as out-of-scope.
- The generation prompt was constrained to produce only JSON with explicit
  citation lists.

---

## 3. Test Suite Design

**Tool**: Antigravity  
**Task**: Writing unit tests for all modules  
**Contribution**: The AI generated test scaffolding and initial test cases
for `test_evidence.py`, `test_conflict.py`, `test_parser.py`,
`test_retriever.py`, `test_embeddings.py`, `test_generation.py`,
`test_grounding.py`, and `test_pipeline.py`.

All unit and integration tests use `FakeModel` / `SequentialFakeModel`
instances that return pre-scripted JSON responses.  No live Groq API calls
are made during `pytest`, which keeps the suite offline, deterministic, and
within API budget constraints.

**Human review**: All tests were reviewed.  Several were modified:
- The `§4.3.2` / `§9.1.4` fixture in `test_conflict.py` was hand-crafted from
  the actual manual text after the citation-mismatch design decision.
- The `pre_retrieved_candidates` parameter was added to `run_pipeline()` after
  the initial pipeline tests tried to call the live Google embedding API.

---

## 4. Bug Fixing

**Tool**: Antigravity  
**Tasks**:
- Fixing `ConflictType` enum missing from `modules/conflict.py` (tests were
  importing it; it existed only in `modules/evidence.py`).
- Fixing `_extract_numeric_requirements` return type mismatch (was `list`,
  tests expected `set`).
- Fixing cross-field validation that incorrectly rejected `conflict=true,
  same_scope=false` for citation mismatches.
- Fixing `check_pair_for_conflict` return condition that was gating on
  `same_scope` instead of just `conflict`.

---

## 5. Documentation

**Tool**: Antigravity  
**Tasks**: Initial drafts of `DECISIONS.md`, `README.md`, and this file.  
**Human review**: All documentation was read, revised, and supplemented with
project-specific detail before submission.

---

## 6. Model Selections

| Component | Model | Reason |
|---|---|---|
| Evidence classification | Groq `openai/gpt-oss-20b` (configurable via `GROQ_MODEL`) | Fast, JSON-mode support, low cost per classification call |
| Conflict detection | Same | Same model used for consistency |
| Answer generation | Same | Keeps the inference stack uniform |
| Embedding | `gemini-embedding-001` via Google AI | High-quality semantic embeddings for retrieval |

All models were selected for their combination of speed and reliability within
the API constraints of a 24-hour hackathon.

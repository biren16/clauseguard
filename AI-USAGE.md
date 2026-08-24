# AI Usage Log — ClauseGuard

This document records how AI assistance was actually used during
development, as accurately as I can reconstruct it. Per the hackathon
rules, this is disclosure, not a sales pitch — it names what each tool did
and did not do, including where AI-proposed designs were wrong and had to be
corrected.

Multiple AI tools were used at different stages, not one tool throughout:

- **Claude** — architecture review, prompt design, debugging, and code for
  `modules/evidence.py` and `modules/conflict.py`, across many iterative
  rounds, before any coding agent was involved.
- **ChatGPT / OpenAI GPT models** — used in parallel during the design and
  debugging work. Proposals were cross-checked against Claude's and against
  actual repository behavior rather than being trusted by default. The exact
  model version used during those early design rounds was not consistently
  recorded, so it is not claimed here.
- **Antigravity** — used later to finish implementation
  (generation, grounding, pipeline wiring, CLI, evaluation harness, and
  related integration work) from a written handoff specification, after the
  evidence/conflict core had already been designed and partially verified.

The human developer was directly involved throughout the first two stages
and monitored the coding-agent work in the final stage. The AI tools were
used as engineering and reasoning aids; their proposals and generated code
were not treated as automatically correct.

---

## 1. Architecture Design — Claude + GPT, Iterative, Pre-Agent

The initial ClauseGuard pipeline was manually designed and implemented
before substantial coding-agent assistance was introduced.

The original system focused on:

- policy-manual parsing,
- knowledge-base construction,
- semantic retrieval,
- evidence classification,
- evidence sufficiency,
- and grounded answering.

As testing exposed deeper problems in the evidence and sufficiency stages,
AI assistance was introduced to help reason about the required architectural
refinements.

The resulting advanced pipeline became:

retrieve → classify → sufficiency → structural expansion →
conflict → generate → validate grounding → answer/refuse

The architecture went through several real revisions:

- An early version only retrieved candidates semantically. Claude identified
  that a genuinely contradicting clause, §9.1.4, might never be retrieved for
  a deadline question because it concerns overpayment recoverability rather
  than the reporting deadline. This led to cross-reference-based retrieval
  expansion.

- A later version followed forward cross-references but still missed the
  direction of the §9.1.4 relationship: §9.1.4 references §4.3 while
  §4.3.2 does not reference §9.1.4. This required a reverse-reference index,
  which was built and confirmed against the actual manual.

- GPT proposed the three-way `SUPPORTED` / `PARTIAL` / `IRRELEVANT`
  classification, improving on an earlier binary design. Claude identified
  that the accompanying "no material gap" sufficiency rule was not
  deterministically implementable as proposed. The design was therefore
  reconciled into the shipped `is_sufficient()` rule.

- The architecture was revised again after discovering that a clause could
  be `IRRELEVANT` for directly answering the user's question while still
  being important to conflict analysis. This is why evidence classification
  and structural conflict expansion remain separate stages.

The final architecture therefore represents the original human-designed
pipeline plus later AI-assisted architectural refinement, with the final
decisions and acceptance criteria remaining under human review.

---

## 2. Prompt Engineering — Claude, GPT, Then Coding Agent

Prompts for evidence classification and conflict analysis went through
multiple revisions.

The most important iterations were driven by actual failures:

- A live Groq run using `gpt-oss-20b` surfaced that evidence quote
  verification rejected a substantively correct quote from §7.1.3 because
  the model used a Unicode hyphen while the manual contained an ASCII
  hyphen. This was confirmed by inspecting the actual manual. The fix was
  Unicode punctuation normalization in quote comparison rather than
  weakening source verification.

- An earlier conflict design used keyword/aspect overlap on `covers`
  descriptions. Manual tracing showed that the real §4.3.2 and §9.1.4
  conflict descriptions shared only one meaningful token, so a lexical
  similarity gate could miss the contradiction. The aspect-matching gate
  was removed in favor of a numeric pre-filter followed by semantic
  conflict analysis.

- A later live run showed the model correctly recognized that §9.1.4 dealt
  with overpayment consequences and therefore returned no ordinary
  same-scope conflict. Human review identified that this was actually the
  central citation-mismatch pattern: §9.1.4 attributes a 30-calendar-day
  requirement to §4.3 even though the operative §4.3.2 provision states
  10 calendar days.

  The conflict prompt was therefore explicitly extended to recognize
  citation-mismatch contradictions rather than assuming every conflict
  must be two independently competing rules with identical immediate
  purpose.

The final live verification confirmed the §4.3.2 / §9.1.4 case as a
conflict rather than rationalizing the earlier false negative away.

---

## 3. Real Bugs Found and Fixed — Claude, GPT, Pre-Agent

Several important bugs were identified before the final coding-agent phase.

### 3.1 PARTIAL evidence retention

`is_sufficient()` initially discarded qualifying `PARTIAL` clauses whenever
a `SUPPORTED` clause existed.

This could silently produce incomplete answers to multi-part questions.

The issue was found through code review and fixed with a regression test
covering the intersection of the `SUPPORTED` and `PARTIAL` branches.

### 3.2 Unsafe boolean coercion

The conflict parser initially risked relying on Python truthiness.

This is unsafe because:

```python
bool("false") == True
```

The implementation was changed to use strict `_parse_bool()` validation,
which accepts only actual JSON booleans.

Tests explicitly verify rejection of:

```json
"true"
```

and:

```json
"false"
```

as string values.

### 3.3 Provider-call errors being masked as parse errors

An early version of `check_pair_for_conflict()` wrapped the provider call
inside the same exception handler used for JSON parsing.

This meant a programming/interface error, such as an unsupported
`json_mode` argument, could become a misleading "invalid model judgement"
rather than surfacing as the actual interface failure.

The provider call was separated from the JSON parsing/validation block.

This remains a regression requirement in the final implementation.

### 3.4 Reverse-reference direction

The reverse-reference index was necessary because the actual corpus contains
a one-way relationship:

* §9.1.4 references §4.3.
* §4.3.2 does not reference §9.1.4.

An outward-only reference walk from §4.3.2 could therefore never discover
§9.1.4.

This was confirmed against the actual manual rather than inferred from
clause IDs.

### 3.5 Numeric section-reference false positives

The conflict pre-filter initially risked treating identifiers such as
`§4.3.2` as substantive numeric requirements.

The implementation was revised so that section references are excluded
from numeric requirement extraction.

This prevents structural metadata from creating false conflict candidates.

---

## 4. Implementation Completion — Antigravity

Once the evidence and conflict modules were designed and partially verified
against live model behavior, the remaining implementation was handed to a
coding agent through a written specification.

The handoff included:

* the established architecture,
* previously discovered bugs,
* mandatory safety fixtures,
* explicit fail-closed requirements,
* and instructions not to silently modify or dismiss safety-relevant
  fixtures merely because an initial model result disagreed with them.

The coding agent was responsible for completing and integrating the
remaining implementation, including:

* `modules/generation.py` — answer generation using validated retained
  evidence only,
* `modules/grounding.py` — deterministic citation-ID validation,
* `modules/pipeline.py` — end-to-end orchestration with
  `ANSWER`, `NO_EVIDENCE`, and `CONFLICT` outcomes,
* `main.py` — CLI,
* `scripts/run_eval.py` — evaluation harness,
* `modules/evidence_model.py` — provider abstraction and typed provider
  failures.

The final implementation was verified to preserve the important constraints:

* The `SUPPORTED` + `PARTIAL` retention regression remains covered.
* `_parse_bool()` rejects string `"true"` and `"false"`.
* Reverse-reference expansion is covered.
* The grounding validator is scoped to deterministic citation-ID
  membership rather than pretending to provide full semantic claim
  verification.
* Provider failures remain distinct from malformed model responses.
* The conflict provider call remains separated from the JSON-parsing
  exception handler.
* The §4.3.2 / §9.1.4 fixture was verified as a conflict in the completed
  implementation.

---

## 5. Corpus Fixtures — Verified, Not Assumed

The two central safety/evaluation fixtures were checked directly against
the actual policy manual rather than inferred from clause IDs.

### Fixture A — §4.3.2 / §9.1.4

§4.3.2 states a 10-calendar-day reporting deadline.

§9.1.4 states that reporting within the "30 calendar days required under
§4.3" prevents an overpayment determination.

This conflicts with the operative §4.3.2 requirement and represents the
citation-mismatch contradiction that the conflict pipeline was specifically
designed to detect.

The completed system identifies this fixture as a conflict.

### Fixture B — Full-time student

§7.1.3 refers to §5.4 for full-time-student treatment.

The inspected §5.4.1 and §5.4.2 clauses concern care-allowance disregard
and household composition and do not provide the missing student
needs-calculation rule.

The completed system therefore routes this case to `NO_EVIDENCE` rather
than inventing an answer.

These fixtures were checked against the actual knowledge-base/manual text.

The final evaluation harness was also checked against the actual knowledge
base rather than relying solely on AI-generated assumptions about clause
contents.

---

## 6. A Caught Instance of AI Fabrication

During drafting, one AI-generated version of the project plan stated a
specific policy amendment number and effective date as if it were a known
upcoming requirement.

That amendment was not present in the manual, and the hackathon's future
requirement change was not known in advance.

The claim was caught by checking it against the actual corpus rather than
accepting a plausible-sounding specific detail.

This reinforced a project-wide rule:

> AI-generated claims about the policy manual must not be treated as
> authoritative without checking the source material.

No fabricated policy detail was knowingly carried into the final system.

---

## 7. Model Selection

| Component                           | Model / Tool                               | Notes                                                                          |
| ----------------------------------- | ------------------------------------------ | ------------------------------------------------------------------------------ |
| Architecture / reasoning assistance | Claude + ChatGPT / OpenAI GPT models       | Used during iterative design and debugging before the final coding-agent phase |
| Evidence classification             | Groq `openai/gpt-oss-20b` via `GROQ_MODEL` | Structured JSON classification                                                 |
| Conflict detection                  | Groq `openai/gpt-oss-20b`                  | Semantic conflict and citation-mismatch analysis                               |
| Answer generation                   | Groq `openai/gpt-oss-20b`                  | Grounded answer generation from retained evidence                              |
| Embeddings                          | `gemini-embedding-001` via Google AI       | Semantic retrieval                                                             |
| Implementation assistance           | Antigravity                                | Final implementation, integration, testing, CLI, and evaluation harness        |

The Groq model is configurable through `GROQ_MODEL`.

The final system therefore uses two external model providers:

* Groq for language-model inference.
* Google AI for embeddings.

Model choices were driven by the requirements of the 24-hour hackathon,
including structured-output support, speed, availability, retrieval
quality, and API quota constraints.

---

## 8. Testing and Verification

The automated test suite was deliberately designed to avoid unnecessary
live API calls.

Unit and deterministic integration tests use mocked model responses such
as `FakeModel` / `SequentialFakeModel`.

This keeps the test suite:

* deterministic,
* reproducible,
* fast,
* and independent of live provider quota.

Live Groq calls were reserved for targeted integration testing.

The final verification reported:

```text
62 passed, 1 warning
```

The final repository was also verified as:

* working tree clean,
* `main` synchronized with `origin/main`,
* all implementation commits pushed,
* no unpushed commits remaining.

The final evaluation harness contained 10 structured evaluation cases and
was checked against the actual knowledge base.

---

## 9. Scope and Honesty Note

AI proposals were wrong or incomplete at multiple points during this
project.

Examples included:

* a sufficiency rule that silently dropped valid evidence,
* an aspect-match heuristic that could miss the project's central
  contradiction,
* unsafe boolean coercion,
* an exception handler capable of masking a real interface bug,
* and an outright fabricated policy detail.

These failures were caught by comparing AI proposals against:

* actual repository code,
* deterministic test results,
* live model output,
* and actual policy-manual text.

The project therefore does not treat AI coherence as evidence of correctness.

A passing test is not automatically proof that the test itself is correct.

Likewise, a plausible model explanation is not evidence that the underlying
policy interpretation is correct.

The safety approach used throughout the project was:

> AI proposes → human reviews → implementation runs → tests challenge the
> implementation → failures are investigated → the source is checked →
> only then is the behavior accepted.

This checking process is a central part of the engineering story of
ClauseGuard.

---

## 10. Human Oversight

The initial pipeline was human-designed and manually implemented.

AI assistance became substantially more important later, when the evidence
and sufficiency stages exposed architectural issues that were difficult to
resolve through isolated changes.

The human developer remained responsible for:

* deciding which architectural proposals were acceptable,
* checking AI claims against the actual corpus,
* identifying mandatory safety fixtures,
* reviewing generated code,
* writing and reviewing regression tests,
* deciding when an apparent model failure represented a real system bug,
* and accepting the final implementation.

Antigravity was therefore used as a coding and implementation
agent, not as an autonomous authority over the project's architecture or
policy interpretation.

The final ClauseGuard system represents a human-directed development
process assisted by multiple AI tools at different stages.
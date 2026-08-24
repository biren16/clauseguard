# Architectural & Design Decisions — ClauseGuard

This document records the key design choices made during development, the
reasons for each, and explicit statements of Day-1 scope limitations.

---

## 1. Fail-Closed Safety Architecture

**Decision**: every gate in the pipeline is fail-closed.

- Evidence classification: a model response that cannot be parsed or whose
  quote cannot be verified against the source clause is downgraded to
  `IRRELEVANT`, never to `SUPPORTED`.

- Boolean parsing: the code never uses `bool(value)` on model output.
  `bool("false") == True` in Python, which would silently turn a refusal
  into a confirmation. All JSON booleans are validated with `isinstance(v,
  bool)`.

- Conflict analysis: a model response that is malformed, self-contradictory,
  or missing required fields is escalated to `UNRESOLVED` rather than cleared.
  An `UNRESOLVED` pair blocks the answer just as a `CONFIRMED` conflict does.

- Grounding validation: if the model returns a citation whose clause ID does
  not appear in the verified retained evidence set, the answer is refused.

**Rationale**: A safety-critical policy assistant should never provide an
ungrounded answer.  Incorrect answers in a benefits context cause real harm.

---

## 2. Evidence Quote Verification

**Decision**: after the LLM returns an evidence quote, the system checks that
every non-stopword in the quote also appears in the source clause text.
Markdown bold markers (`**`) and Unicode hyphens (em-dash, en-dash) are
normalised before comparison.

**Rationale**: prevents the model from inventing or rephrasing text and
presenting it as a direct quotation.

---

## 3. Reverse-Reference Index for Conflict Expansion

**Decision**: before conflict analysis, we build a reverse-reference index:
for each clause in the knowledge base, we record every other clause that cites
it.  Retained evidence clauses are then structurally expanded to include their
forward references *and* every clause in the KB that cites them.

**Rationale**: §4.3.2 establishes the 10-day rule.  §9.1.4 mentions a 30-day
rule *attributed to* §4.3, creating a citation mismatch.  Without the reverse
index, §9.1.4 would never appear in the candidate set for the question "How
long do I have to report a change?" and the contradiction would be silently
missed.

---

## 4. Citation-Mismatch Detection

**Decision**: the conflict-detection LLM is explicitly instructed that a
citation mismatch (one clause makes a numerically inconsistent claim about
what another clause says) constitutes a conflict even when `same_scope` is
`false`.  The cross-field validation code permits `same_scope=false` when
`conflict_type="citation_mismatch"`.

**Rationale**: §9.1.4 and §4.3.2 have different *operational* purposes
(overpayment consequences vs. reporting procedure) so a naive scope check
would clear the pair.  But §9.1.4's statement about §4.3's requirement is
factually inconsistent with §4.3.2.

---

## 5. Numeric Pre-Filter (`_extract_numeric_requirements`)

**Decision**: before calling the LLM for conflict analysis, we filter pairs
to only those where *both* clauses contain patterns like "within N calendar
days" or "N days required".  A clause containing only section references
(§4.3.2) but no normative numeric requirement is excluded.

**Rationale**: without this filter, structural clauses containing only
cross-references would generate spurious LLM calls and high API costs.

---

## 6. Sufficiency Rules

**Decision**:
- One `SUPPORTED` clause with a non-empty `covers` string is sufficient.
- Two or more `PARTIAL` clauses with *distinct* `covers` strings are
  sufficient together.
- A single `PARTIAL` is insufficient.
- Two `PARTIAL` clauses with the *same* `covers` string are insufficient
  (they must address distinct aspects).

**Rationale**: the rule mirrors the real-world principle that a partial policy
answer needs complementary clauses to be complete.

---

## 7. Day-1 Scope Limitations

The following are explicitly deferred and are **not bugs**:

| Limitation | Reason deferred |
|---|---|
| `covers` string comparison is literal token matching, not semantic | Semantic deduplication requires an embedding comparison per result pair, which adds latency and cost |
| Conflict detection is limited to numeric disagreements | Qualitative contradictions (e.g. two clauses giving contradictory eligibility criteria with no numbers) are not in scope |
| Knowledge base uses offline pre-computed embeddings | Re-embedding at query time would add ~2 s per request |
| max_model_calls=8 default in detect_conflicts | Protects live Groq budget; structural expansion can produce O(n²) pairs |

---

## 8. conftest.py / PYTHONPATH

The test suite requires `PYTHONPATH=.` to resolve `modules.*`.  There is no
`conftest.py` or `setup.py` because the project is intentionally kept as a
flat script-style codebase for hackathon submission.  A production system
would package `modules/` as an installable package.

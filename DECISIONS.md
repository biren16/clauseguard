# Architectural & Design Decisions — ClauseGuard

This document records the principal architectural and engineering decisions
made during the development of ClauseGuard, the reasoning behind them, and
the limitations that remain intentionally within the Day-1 scope.

ClauseGuard is designed as a fail-closed policy question-answering system.
Its objective is not to maximize the number of questions answered, but to
prevent the system from presenting unsupported, contradictory, or
ungrounded policy interpretations as authoritative answers.

---

## 1. Initial Architecture and Later Refinement

**Decision**: The project began with a manually designed and implemented core
pipeline covering:

- policy-manual parsing,
- knowledge-base construction,
- semantic retrieval,
- evidence classification,
- evidence sufficiency,
- and grounded answering.

As testing exposed deeper problems in the evidence and sufficiency stages,
AI assistance was used to reason about architectural refinements.

The resulting advanced pipeline is:

```text
retrieve
   ↓
classify evidence
   ↓
check sufficiency
   ↓
structural/reference expansion
   ↓
detect conflicts
   ↓
generate answer
   ↓
validate grounding
   ↓
answer OR refuse
```

The architectural refinements were reviewed against the actual policy corpus,
repository behavior, deterministic tests, and live model behavior before being
accepted.

**Rationale**: The original architecture was sufficient for straightforward
grounded retrieval, but difficult cases demonstrated that semantic similarity
alone could not guarantee that all relevant or contradictory provisions would
be discovered.

---

## 2. Evidence Classification Is Separate From Conflict Detection

**Decision**: Evidence relevance and conflict detection are separate stages.

Evidence classification answers:

> Does this clause provide usable evidence for the user's question?

Conflict detection answers:

> Do retained or structurally connected clauses contain an unresolved
> contradiction?

A clause can therefore be:

* `IRRELEVANT` for directly answering the user's question,
* while still being important to conflict analysis.

**Rationale**: The §4.3.2 / §9.1.4 fixture demonstrated this distinction.
§9.1.4 concerns overpayment consequences rather than directly establishing
the reporting procedure, but it contains a statement about the reporting
deadline attributed to §4.3. That statement can be important evidence of a
contradiction even though the clause is not the primary answer to the user's
question.

Combining these two decisions into one relevance score risks discarding
important contradiction evidence.

---

## 3. Three-Way Evidence Classification

**Decision**: Candidate clauses are classified as:

* `SUPPORTED`
* `PARTIAL`
* `IRRELEVANT`

A `SUPPORTED` result indicates that the clause directly supports a relevant
part of the question.

A `PARTIAL` result indicates that the clause provides relevant evidence but
does not independently establish the full answer.

An `IRRELEVANT` result does not contribute direct answer evidence.

**Rationale**: A binary relevant/irrelevant classification was insufficient
for questions where multiple clauses contribute different parts of an answer.

The three-way classification allows the sufficiency stage to distinguish
between a complete supporting provision and a provision that only contributes
one aspect of a multi-part answer.

---

## 4. Evidence Quotes Must Be Grounded in the Source Clause

**Decision**: A `SUPPORTED` or `PARTIAL` result is accepted only when its
evidence quote can be verified against the original clause text.

If the quote cannot be verified, the result is downgraded to `IRRELEVANT`.

The comparison tolerates harmless formatting differences such as:

* whitespace,
* line breaks,
* Markdown formatting,
* Unicode punctuation variants.

It does not permit arbitrary unsupported wording to become a quotation.

**Rationale**: The language model must not be allowed to manufacture a
quotation while still receiving the authority of a source citation.

A live run exposed a concrete case where the model used a Unicode hyphen in
`full-time` while the source contained an ASCII hyphen. The substantive quote
was correct, so the correct fix was normalization of harmless punctuation
differences rather than weakening quote verification.

---

## 5. Fail-Closed Evidence Parsing

**Decision**: Malformed evidence-model responses fail closed.

Examples include:

* invalid JSON,
* missing `status`,
* invalid status values,
* non-string evidence fields,
* missing coverage for a supporting result,
* unverifiable evidence quotes.

Such results are not promoted to supporting evidence.

They are downgraded to `IRRELEVANT`.

**Rationale**: A malformed model response should never increase the amount of
evidence available to the answer-generation stage.

---

## 6. Deterministic Sufficiency Rules

**Decision**: Day-1 sufficiency is deterministic.

The current rule is:

1. At least one `SUPPORTED` clause is sufficient.
2. If there is no `SUPPORTED` clause, at least two `PARTIAL` clauses with
   distinct coverage descriptions are required.
3. A single `PARTIAL` clause is insufficient.
4. When a `SUPPORTED` clause exists, qualifying `PARTIAL` clauses are retained
   as well.

**Rationale**: Sufficiency must not depend on an unconstrained model
judgement.

An earlier implementation could discard valid `PARTIAL` evidence when a
`SUPPORTED` clause existed. That could result in incomplete evidence being
passed to later stages. The retention behavior was therefore made explicit
and regression-tested.

**Day-1 limitation**: `covers` descriptions are currently compared using
literal normalized strings rather than semantic similarity. This is
intentionally deferred.

---

## 7. Structural Expansion Is Required for Conflict Analysis

**Decision**: Conflict analysis does not operate only on semantically
retrieved candidates.

After evidence is retained, the system expands the conflict-check set using
structural references in the knowledge base.

The expansion considers:

1. forward references from retained clauses;
2. reverse references from other clauses pointing toward retained clauses;
3. hierarchical relationships between parent and child sections where
   applicable.

**Rationale**: Semantic retrieval answers the question:

> Which clauses look relevant to this question?

Conflict analysis requires a different question:

> Which clauses could materially interact with these provisions?

Those are not necessarily the same set.

---

## 8. Reverse-Reference Index

**Decision**: The knowledge base receives a reverse-reference index before
conflict expansion.

For example, if:

```text
§9.1.4 → references §4.3
```

the reverse index allows the system to discover:

```text
§4.3 → §9.1.4
```

when §4.3 becomes relevant.

**Rationale**: The actual corpus contains the critical relationship:

* §9.1.4 references §4.3;
* §4.3.2 does not reference §9.1.4.

An outward-only traversal beginning at §4.3.2 therefore cannot discover
§9.1.4.

The reverse-reference index was introduced specifically to prevent this
directional blind spot.

---

## 9. Citation-Mismatch Contradictions Are a Distinct Conflict Pattern

**Decision**: Conflict detection explicitly recognizes citation mismatches.

A contradiction does not require two clauses to be independently imposing
the same operational rule.

A clause may instead make a factual claim about what another clause requires,
and that claim may contradict the actual referenced provision.

This is treated as a conflict pattern even when the two clauses have different
immediate purposes.

**Canonical fixture**:

* §4.3.2 states a 10-calendar-day reporting requirement.
* §9.1.4 refers to a "30 calendar days required under §4.3."

The second clause concerns overpayment consequences, but its statement about
the requirement attributed to §4.3 is inconsistent with the operative
provision.

**Rationale**: A naive same-scope test can incorrectly clear this pair because
the clauses appear to serve different purposes.

This was a real false-negative encountered during development. The solution
was to explicitly teach the conflict analysis stage to recognize citation
mismatch rather than redefining the fixture as non-conflicting.

---

## 10. Numeric Conflict Pre-Filtering

**Decision**: Candidate conflict pairs are first filtered using deterministic
numeric requirement extraction before an LLM conflict judgement is requested.

The pre-filter looks for substantive normative requirements such as:

```text
within 10 calendar days
30 days required
```

Section identifiers such as:

```text
§4.3.2
§9.1.4
```

are not treated as substantive numeric requirements.

**Rationale**:

The structural conflict set can contain clauses that merely contain section
references. Sending every possible pair to the model would:

* increase API usage,
* increase latency,
* create unnecessary opportunities for model errors,
* and produce false conflict candidates.

The deterministic pre-filter narrows the expensive semantic analysis to pairs
that contain a plausible numeric disagreement.

**Limitation**: The current conflict detector is primarily designed around
numeric disagreement patterns. Qualitative contradictions without numeric
requirements remain outside the full Day-1 conflict-detection guarantee.

---

## 11. Strict Boolean Validation

**Decision**: Model-generated booleans are accepted only when they are actual
JSON booleans.

The implementation never uses Python truthiness for model output.

For example:

```python
bool("false") == True
```

Therefore strings such as:

```json
"true"
"false"
```

are rejected rather than interpreted as booleans.

**Rationale**: A string/boolean coercion bug could convert a model response
intended to say "no conflict" into a truthy Python value and silently alter a
safety decision.

This behavior is protected by explicit tests.

---

## 12. Malformed Conflict Judgements Fail Closed

**Decision**: A malformed conflict-model response does not clear the pair.

If the model response is:

* malformed JSON,
* missing required fields,
* contains invalid boolean types,
* or otherwise cannot be safely interpreted,

the pair becomes `UNRESOLVED`.

`UNRESOLVED` is treated as blocking the answer just like a confirmed conflict.

**Rationale**: In a safety-critical policy assistant:

```text
unknown
```

must not become:

```text
safe
```

The system therefore prefers escalation over an unsupported clearance.

---

## 13. Provider Errors Are Separate From Model-Judgement Errors

**Decision**: Provider/network failures are kept separate from malformed model
content.

The provider abstraction exposes typed errors for situations such as:

* rate limits,
* connection failures,
* provider/API failures.

The conflict parser does not catch provider-call failures and relabel them as
JSON parsing problems.

**Rationale**: During development, the provider call was temporarily inside
the same exception block as JSON parsing. This could turn a genuine programming
or interface error, such as an unsupported method argument, into a misleading
"invalid model judgement."

The provider call and response parsing are therefore deliberately separated.

---

## 14. Provider Abstraction

**Decision**: Evidence, conflict, and generation logic depend on the
provider-independent `EvidenceModel` interface rather than directly on the
Groq client.

The interface provides:

```text
system_prompt
user_prompt
json_mode
```

and returns raw model text.

`GroqEvidenceModel` is the current implementation.

**Rationale**: Keeping provider communication separate from evidence and
conflict logic makes the safety-critical pipeline easier to test and allows
mock models to be used without contacting the provider.

The same abstraction also makes provider failures explicit.

---

## 15. Live API Calls Are Not Used for Deterministic Tests

**Decision**: Unit and deterministic integration tests use fake models such
as `FakeModel` and `SequentialFakeModel`.

Live Groq calls are reserved for:

* targeted provider tests,
* live evidence checks,
* and manual CLI/integration verification.

**Rationale**:

Live model calls introduce:

* quota consumption,
* rate limits,
* nondeterminism,
* latency,
* and external availability dependencies.

The core test suite therefore remains offline and deterministic.

---

## 16. API Budget Protection

**Decision**: Conflict detection uses a model-call budget.

The conflict stage does not blindly send every O(n²) pair to the provider.

A configurable `max_model_calls` limit prevents structural expansion from
turning into an uncontrolled number of API requests.

**Rationale**: Reverse-reference expansion increases the number of clauses
that may need consideration. This is necessary for safety, but without a
budget it could create excessive provider calls.

Budget protection is therefore treated as part of the architecture rather
than merely an operational convenience.

---

## 17. Generation Receives Only Validated Evidence

**Decision**: Answer generation receives only evidence that survived:

1. evidence classification,
2. quote verification,
3. sufficiency filtering.

Generation does not receive:

* the complete candidate pool,
* rejected clauses,
* unverified quotes,
* retrieval scores,
* embeddings,
* or internal conflict-analysis reasoning.

**Rationale**: Giving the generator access to untrusted or rejected material
would undermine the earlier safety gates.

The generator's job is to express validated evidence, not decide which
unverified policy material is trustworthy.

---

## 18. Structured Answer Generation

**Decision**: The generation stage requests a structured response containing
an answer and explicit clause citations.

Conceptually:

```json
{
  "answer": "...",
  "citations": ["§X.Y.Z"]
}
```

The generated response is parsed before it reaches the user.

**Rationale**: Separating answer text from citation identifiers allows the
grounding stage to validate citations deterministically.

---

## 19. Grounding Validation Is Deliberately Narrow

**Decision**: Day-1 grounding validation verifies that every clause ID cited
by the generated answer belongs to the retained, verified evidence set.

It does **not** claim to perform full semantic proof that every sentence of a
paraphrased answer is logically entailed by the source.

**Rationale**: Full claim-level semantic grounding is a substantially harder
problem. A naive substring check would reject legitimate paraphrases, while a
looser semantic check would require additional modelling and validation.

The project therefore uses a precise, deterministic guarantee rather than
claiming a stronger guarantee that has not been implemented.

**Deferred improvement**: claim-level semantic grounding can be added in a
future version.

---

## 20. Refusal Is a First-Class Outcome

**Decision**: The pipeline has explicit refusal outcomes rather than treating
every query as something that must produce an answer.

The primary outcomes are:

```text
ANSWER
NO_EVIDENCE
CONFLICT
```

An unresolved conflict is handled as a blocking safety condition.

**Rationale**: A policy assistant should be allowed to say:

> "The available evidence is insufficient."

or:

> "The policy corpus contains conflicting provisions."

rather than inventing a resolution.

---

## 21. Full Pipeline Separation

**Decision**: The final pipeline separates the major responsibilities into
independent modules:

```text
retriever
    ↓
evidence
    ↓
sufficiency
    ↓
structural expansion
    ↓
conflict
    ↓
generation
    ↓
grounding
    ↓
pipeline outcome
```

The CLI is kept separate from the pipeline itself.

**Rationale**: This makes individual safety gates independently testable and
prevents presentation logic from becoming entangled with policy reasoning.

It also allows the same pipeline to be exercised by:

* unit tests,
* evaluation scripts,
* CLI execution,
* and future interfaces.

---

## 22. Offline Knowledge Base and Retrieval

**Decision**: The system uses the pre-computed knowledge base and embeddings
rather than generating embeddings for every query.

The knowledge base contains parsed policy clauses with structural metadata
and pre-computed retrieval information.

**Rationale**:

Query-time embedding generation adds latency and external dependency
requirements. Pre-computation is sufficient for the hackathon's corpus and
makes retrieval predictable.

---

## 23. Knowledge-Base Structure Is Preserved During Expansion

**Decision**: Evidence classification results are not treated as complete
knowledge-base clauses.

The system explicitly maps retained `EvidenceResult` objects back to their
original knowledge-base records before structural/conflict expansion.

**Rationale**: `EvidenceResult` contains classification/provenance fields,
while structural analysis requires metadata such as:

* clause ID,
* clause text,
* cross-references,
* section information,
* and related structural metadata.

Keeping these representations separate prevents accidental loss of source
metadata and avoids treating model-generated classification data as the
authoritative knowledge base.

---

## 24. No Hard-Coded Safety Fixture

**Decision**: The conflict detector must detect the §4.3.2 / §9.1.4 conflict
through general structural and semantic rules.

The implementation does not contain a special rule equivalent to:

```text
if clause_id == "§4.3.2" and clause_id == "§9.1.4":
    conflict
```

**Rationale**: Hard-coding the known fixture would make the demonstration
pass without demonstrating that the architecture can detect the underlying
pattern.

The desired behavior is detection of the general citation-mismatch pattern.

---

## 25. Central Safety Fixtures Are Verified Against the Corpus

Two cases are treated as mandatory regression fixtures.

### Fixture A — §4.3.2 / §9.1.4

The corpus was checked directly.

§4.3.2 establishes a 10-calendar-day reporting requirement.

§9.1.4 refers to "30 calendar days required under §4.3."

The system is expected to detect this as a citation-mismatch conflict and
refuse to provide a confident answer.

### Fixture B — Full-Time Student

§7.1.3 refers to §5.4 for the full-time-student treatment.

The inspected candidate clauses do not provide the missing student
needs-calculation rule.

The system is therefore expected to route the query to `NO_EVIDENCE` rather
than inventing the missing rule.

**Rationale**: These fixtures test two different safety properties:

1. detecting a subtle contradiction;
2. refusing to answer when evidence is missing.

---

## 26. AI-Generated Claims Are Not Treated as Policy Authority

**Decision**: Claims about the policy manual must be checked against the
actual corpus.

During development, an AI-generated project-plan version introduced a
specific policy amendment number and effective date that was not supported by
the manual.

The claim was rejected after checking the source.

**Rationale**: Plausible specificity is not evidence.

This became a project-wide rule:

```text
AI proposes → human reviews → source/tests challenge the proposal
→ only verified behavior is accepted.
```

---

## 27. Human Oversight

**Decision**: AI assistance is treated as engineering and reasoning support,
not as autonomous authority over policy interpretation.

The human developer remained responsible for:

* deciding which architectural proposals were accepted;
* checking claims against the actual policy corpus;
* reviewing generated code;
* reviewing regression tests;
* identifying mandatory safety fixtures;
* investigating apparent model failures;
* and accepting the final behavior.

Antigravity was used as a coding and implementation agent during the later
completion phase, after the evidence/conflict core had already been designed
and partially verified.

**Rationale**: The central engineering discipline of ClauseGuard is
verification rather than trust in model confidence.

---

## 28. Day-1 Scope Limitations

The following limitations are intentional and documented rather than hidden.

| Limitation                 | Current behavior                                                          |
| -------------------------- | ------------------------------------------------------------------------- |
| `covers` comparison        | Literal normalized string comparison rather than semantic deduplication   |
| Conflict scope             | Primarily numeric/citation-mismatch conflict detection                    |
| Full semantic grounding    | Deferred; grounding currently validates citation-ID membership            |
| Knowledge-base embeddings  | Pre-computed rather than generated at query time                          |
| Conflict model calls       | Limited by `max_model_calls` to protect API budget                        |
| Qualitative contradictions | Not guaranteed to be detected when no numeric/citation pattern is present |

These limitations should not be presented as capabilities the system does
not actually provide.

---

## 29. Testing Philosophy

**Decision**: A green test suite is necessary but not sufficient evidence of
correctness.

The project uses:

* deterministic unit tests,
* mocked provider responses,
* targeted live model checks,
* corpus verification,
* and end-to-end evaluation.

Tests specifically protect previously discovered failure modes, including:

* `SUPPORTED` + `PARTIAL` evidence retention;
* rejection of string booleans;
* reverse-reference expansion;
* numeric section-reference filtering;
* citation-mismatch conflict detection;
* provider-error separation;
* grounding citation membership;
* and pipeline outcome routing.

**Rationale**: A test can itself encode the wrong assumption. Therefore,
important fixtures were checked against the actual policy manual rather than
being trusted merely because the test passed.

---

## 30. Final Engineering Principle

The central design principle of ClauseGuard is:

> **When the system cannot establish a safe answer from verified evidence,
> it should refuse rather than guess.**

This principle appears throughout the architecture:

* invalid evidence is discarded;
* insufficient evidence produces `NO_EVIDENCE`;
* confirmed conflicts produce `CONFLICT`;
* unresolved conflict checks block confident answers;
* unsupported citations fail grounding;
* provider failures remain visible;
* and AI-generated claims are checked against the source corpus.

The goal is therefore not:

> "Always answer."

The goal is:

> **"Answer when the evidence supports it, and explicitly refuse when it
> does not."**

# PoC Design: Semantic Concept Detection on Claim Descriptions

**Status:** proposal for PoC
**Owner:** (you)
**Date:** 2026-09-23
**Context:** PO requirement, detect whether stored keyword-concepts are *semantically* present in
free-text `damage description` / `accident description` fields, tolerating paraphrase
("fluid leaking" ≈ "liquid coming out") and negation ("no fire", "brakes worked fine").

---

## 1. What this actually is (naming)

The PO calls it "semantic search." The shipped shape is **per-concept presence detection with an
evidence span**, not a search index. For each stored concept the system returns present/absent + a
confidence + the sentence that triggered it. Fix this vocabulary in review so nobody builds a
retrieval/search index that isn't needed.

**In scope (PoC):** given claim text + a managed set of concepts, return which concepts are present,
with confidence and evidence, in one API call.
**Out of scope (PoC):** model fine-tuning, multi-language, high-cardinality (>~100) concept sets,
adjuster-feedback learning loop (noted as future).

---

## 2. Constraints (these decided the architecture)

| Constraint | Consequence |
|---|---|
| Microsoft-only shop | Azure / AKS deployment; MS-aligned runtimes preferred |
| Backend = Java, Frontend = TypeScript/React | no Python service anywhere |
| **No unauthorised languages, no Python, no Rust** | rules out HF `tokenizers` (Rust) and any Python inference |
| Data governance (claim text ≈ PII) | **no data egress**; no external embedding/LLM API |
| Production runs on CPU (no GPU in prod) | model must be small + CPU-fast |

**Rejected because of the above:**
- *Azure OpenAI embeddings*: managed, but sends claim text out of the app boundary; extra DPA/egress review. (Tenant-scoped, so not impossible, held as a fallback only.)
- *TypeSafe Jev / any decision-model API*: external SaaS, data egress. Out.
- *Laya (self-host decision model)*: ModernBERT-large is impractical on CPU (measured: ~0.4 to 1.2 s/inference single, minutes for batches without a GPU). Reserve for GPU-only toy/research.
- *HF `tokenizers`*: Rust. Out. Use a pure-Java WordPiece tokenizer instead.

---

## 3. Language & dependency policy (explicit, because it was called out)

| Layer | Choice | Language |
|---|---|---|
| Tokenizer | DJL `BertFullTokenizer` + `vocab.txt` | **pure Java** |
| Embedding math (pool, normalize, cosine) | in-house | **pure Java** |
| Concept matching / API / service | Spring Boot | **Java** |
| Negation scoping | Stanford CoreNLP or Apache OpenNLP | **Java** |
| Frontend | React | **TypeScript** |
| Inference runtime | **ONNX Runtime** (Microsoft), via JNI | pre-built **C++ native binary** |

**The one native dependency to ratify:** ONNX Runtime is a compiled C++ library (Microsoft's own),
loaded through JNI, the same category as the JVM itself and any BLAS. Performant transformer
inference on the JVM cannot avoid native numeric code; there is no pure-Java option at usable speed.
It is **not Rust, not Python, not source we maintain**: it is a binary dependency. If the team
rejects even Microsoft-owned native binaries, there is no viable on-prem embedding path and the
requirement forces a downgrade to lexical/GloVe-average matching (materially worse, see §10).

---

## 4. Architecture

```
 React / TypeScript                     Java (Spring Boot) on AKS
 ┌───────────────────┐   POST /analyze  ┌─────────────────────────────────────────┐
 │ damage / accident │ ───{ text }────► │ 1. split text into sentences            │
 │ description fields│                  │ 2. tokenize (pure-Java WordPiece)       │
 │ (debounced)       │                  │ 3. embed each sentence ──► ONNX Runtime │
 │                   │ ◄──{ matches }── │    (bge-small .onnx, in-process, CPU)   │
 │ concept chips +   │                  │ 4. cosine vs concept vectors (in-mem)   │
 │ confidence +      │                  │ 5. per-concept threshold                │
 │ evidence sentence │                  │ 6. negation guard on matched sentence   │
 └───────────────────┘                  └─────────────────────────────────────────┘
                                        concept vectors embedded ONCE at startup
```

All model weights (`model.onnx`, `vocab.txt`) baked into the container image. Nothing leaves the pod.

---

## 5. Model

- **`bge-small-en-v1.5`**: 33M params, 384-dim, MIT licence. Pre-exported ONNX available on HF (download the `.onnx` + `vocab.txt` once; no Python conversion). Pooling = **CLS**.
- Alternative: **`all-MiniLM-L6-v2`**: 22M, 384-dim, Apache. Pooling = **mean**. Smaller/faster, slightly lower quality.
- Do **not** use bge-large: the accuracy gain is eaten by negation noise (see §10) and it costs CPU.

**Pooling mode must match the model** (CLS for bge, mean for MiniLM). Wrong mode = silently garbage
vectors, no error. Pin in config; assert first concept embedding against a known-good vector at startup.

---

## 6. Concept store (data model)

Managed by PO/adjusters; embedded once at startup or on edit. **No model retraining ever**: adding a
keyword = embed one row.

```
concept(
  id           uuid,
  key          text,        -- e.g. "fluid_leak"
  label        text,        -- "Fluid leak" (UI)
  definition   text,        -- "fluid leaking from the vehicle: oil, coolant, brake fluid, fuel"
  threshold    real,        -- per-concept cosine cutoff (tuned, see §9)
  trigger_terms text[],     -- lexical stems for the negation guard (§8)
  vector       real[384]    -- precomputed embedding of `definition`
)
```

---

## 7. API contract

```
POST /api/claims/analyze
Content-Type: application/json
{ "damageDescription": "...", "accidentDescription": "..." }

200 OK
{
  "matches": [
    { "key": "fluid_leak", "label": "Fluid leak", "present": true,
      "score": 0.71, "evidence": "liquid was pooling under the car" },
    { "key": "fire_damage", "label": "Fire damage", "present": false, "score": 0.12, "evidence": null }
  ]
}
```

`evidence` = the sentence with the highest similarity to that concept → drives UI highlighting and
gives the adjuster something concrete to confirm/override.

---

## 8. Sentence-level + negation

**Embed per sentence, match concept to its best sentence.** Two reasons:
1. Negation stays local, "brakes worked fine" is one sentence, far easier to scope than a 500-word blob.
2. You get the evidence span for free.

**Negation guard (Java, no Python/Rust):** on the *matched sentence only*, use CoreNLP/OpenNLP to
detect whether the concept's `trigger_terms` fall inside a negation scope; if every mention is
negated, suppress the match. Runs on one short sentence, so cost is negligible.

> **Measured warning:** a naïve character-window negation rule *hurt* overall accuracy in the
> prototype (macro AUROC 0.837 → 0.782; `brake_problem` 0.890 → 0.667) by over-suppressing true
> positives, long claim narratives scatter "no"/"not" everywhere. **Use proper clause scoping, not a
> window.** For the first PoC cut you may ship without negation and flag it as the known refinement,
> but budget for it: negation false-positives are what discredit the feature with adjusters.

---

## 9. Validation: reuse today's labeled set

A 200-claim benchmark already exists (public-domain **NHTSA** vehicle complaints), labeled by
two-annotator consensus (independent models, 95.5% agreement) + government flags for fire/crash/injury.
Port the labels CSV as the PoC's acceptance test.

- Metric: **AUROC** (threshold-free ranking) + **best-F1** + **negation-subset accuracy** (concept
  word present but concept absent, the case that matters).
- Tune each concept's `threshold` on this set.
- Acceptance bar: agree with PO on a per-concept precision/recall target before building the loop.

**Prototype findings that transfer (Python throwaway, embeddings = bge-small):**

| | result |
|---|---|
| macro AUROC (6 PO concepts) | **0.837** |
| strong concepts | brake_problem 0.89, transmission 0.90, electrical 0.84 |
| weak spot | rare concepts (fluid_leak F1 0.19, only 4 positives) + negation |
| negation-subset acc (no guard) | 0.75 (embeddings alone false-positive on negation) |

Read: embeddings are a solid, deployable baseline; **rare concepts and negation are the two risks**
to watch, both measurable on this set.

---

## 10. Why not the cheaper alternatives

- **Lexical / keyword match:** no paraphrase ("liquid coming out" ≠ "fluid leak"). Fails the core requirement.
- **GloVe / word2vec averaging (pure JVM, no native):** loses word order and negation; quality well
  below transformer embeddings. Only the fallback-of-the-fallback if native binaries are refused (§3).
- **Cross-encoder / NLI per keyword:** O(N) full model passes per claim (re-encodes claim+keyword for
  every concept), does not scale with the keyword count. Rejected in prototyping.

---

## 11. Build sequence (PoC milestones)

1. **Loop skeleton**: Spring Boot `/analyze` returns stub matches; React renders concept chips. Proves the round-trip.
2. **Embedding provider**: `EmbeddingProvider` interface + `DjlOnnxEmbeddingProvider` (ONNX Runtime + pure-Java tokenizer). Embed concept store at startup; assert known-good vector.
3. **Matching**: sentence split, cosine, per-concept threshold. Tune thresholds on the 200-claim set.
4. **Evidence**: return the best-matching sentence per concept; highlight in UI.
5. **Negation guard**: CoreNLP clause scope on matched sentences. Re-measure against §9.
6. **UI polish + override**: adjuster can accept/reject a detected concept; captures data for future threshold tuning.

---

## 12. Risks & open questions

- **Native-binary policy** (§3): needs a ruling on ONNX Runtime before build. Blocking.
- **Concept count**: architecture assumes tens to low hundreds of concepts (in-memory cosine loop). If
  the PO wants thousands, add a JVM vector index (Lucene HNSW, pure Java) rather than looping.
- **Negation quality**: the measured risk; proper clause scoping required, validate on the set.
- **Threshold drift**: per-concept thresholds tuned on NHTSA vehicle text; re-tune on real production
  claim text before production. NHTSA is a proxy, not the live distribution.
- **Definition wording**: the concept `definition` string is the query vector; sloppy definitions
  (esp. vague ones like "electrical problem") hurt every match. Treat definitions as tunable config.

---

## 13. Future (not PoC)

- Adjuster-feedback loop → periodic threshold re-tuning or a small fine-tuned classifier head.
- GPU path (Laya / cross-encoder) only if the organisation provisions GPU inference, keep on the 3090 for
  research until then.
- Vector index (Lucene HNSW) if the concept set grows large.

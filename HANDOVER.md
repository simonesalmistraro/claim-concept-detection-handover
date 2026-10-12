# Handover: semantic concept detection on claim descriptions

**Purpose of this file.** Opening context for an AI assistant working inside the target environment. It carries conclusions, decisions and open items from a prototyping spike done outside that environment. Read this before the design doc. No customer data, names or internal identifiers appear here or in the companion files.

**Companion files (the only assets that transfer):**

| file | what it is |
|---|---|
| `PoC-semantic-concept-detection.md` | design doc, 13 sections, dated 2026-09-23 |
| `claims.json` | 200 public-domain NHTSA vehicle complaint narratives (id, text, make, model) |
| `labeling_sheet.csv` | 200 rows x 6 concept columns, 0/1 labels, 38 cells blank |
| `LABELING_README.md` | labeling rules, concept definitions, negation rule |

Nothing else transfers. Python harness, model checkpoints and session logs stay outside.

---

## 1. The requirement, in the product owner's words

> User is going to type some text in damage description and accident description fields. We need to do semantic search to identify some keywords. The system will store some keywords and we need to read what the user types and see whether those keywords are present or not. For example a stored keyword can be "fluid leaking" and the user might type "liquid is coming out". The system should understand both mean the same.

**Reframe agreed during the spike.** This is not search. It is per-concept presence detection with an evidence span: for each stored concept, return present/absent, a confidence, and the sentence that triggered it. Paraphrase must match. Negation ("checked for leaks, found none") must not match. Fix this vocabulary early so nobody builds a retrieval index.

## 2. Hard constraints (these decided everything)

| constraint | consequence |
|---|---|
| Microsoft-aligned stack, Azure and AKS | prefer MS-owned runtimes and services |
| Backend Java, frontend TypeScript/React | no Python service anywhere |
| No unauthorised languages: no Python, no Rust, in source or build | rules out Hugging Face `tokenizers` (Rust) and any Python inference |
| Claim text is personal data, no egress | no third-party embedding or LLM API |
| Production is CPU only | model must be small and CPU-fast |

## 3. What was measured

Dataset: 200 public NHTSA complaints. Six product-owner concepts: `fluid_leak`, `brake_problem`, `warning_light`, `transmission_problem`, `electrical_problem`, `engine_stall`. Plus three government-flag concepts (fire, crash, injured) that come with official labels.

**Approach A: stock embedding model** (bge-small-en-v1.5, 33M params, sentence-level cosine against concept definition vectors).

| concept | AUROC | best F1 | negation-subset acc | n negation | positives |
|---|---|---|---|---|---|
| fluid_leak | 0.820 | 0.194 | 0.182 | 11 | 4 |
| brake_problem | 0.890 | 0.700 | 0.926 | 54 | 19 |
| warning_light | 0.800 | 0.615 | 0.718 | 39 | 38 |
| transmission_problem | 0.905 | 0.500 | 1.000 | 2 | 12 |
| electrical_problem | 0.839 | 0.596 | 0.667 | 3 | 29 |
| engine_stall | 0.770 | 0.521 | 1.000 | 3 | 29 |
| **macro** | **0.837** | | **0.749** | | |

**Approach B: Laya** (open 421M ModernBERT decision model, Apache 2.0, run locally).

| test | A embeddings | B Laya |
|---|---|---|
| 3 gov concepts, macro AUROC | 0.833 | 0.837 |
| 3 gov concepts, negation-subset accuracy | 0.566 | 0.697 |
| 6 PO concepts, macro AUROC | 0.837 | run never completed on CPU |

**Latency, CPU, one 607-character claim, warm, p50:** Laya 670 ms per concept, about 4 s for six concepts. Embeddings tens of ms for all six. On a consumer GPU Laya extrapolates to 30 to 50 ms for six concepts.

**Negation experiment:** a naive character-window negation rule on top of embeddings *hurt* macro AUROC, 0.837 to 0.782, by over-suppressing true positives. Proper clause scoping is required, not a window.

**Approach C: CLM-v0.1-8B** (8B frozen encoder plus heads, needs GPU). Not run. Ruled out for production by the CPU constraint before any measurement.

## 4. Honesty flags on the numbers above

Read these before quoting any figure.

1. **Labels are model-generated.** Ground truth for the six PO concepts is a consensus of two independent models (95.5% agreement), not human labels. Two models agreeing is not correctness. One annotator was a decision-model API, so the Laya comparison is partly circular. Hand-labeling was planned and not done. 38 cells are still blank.
2. **Laya never finished on the six PO concepts.** Rejected for production on speed, which stands. Its accuracy on the concepts that matter is unknown.
3. **Negation subsets are tiny for three concepts.** n=2 and n=3. The 1.000 figures there mean nothing. Do not quote them.
4. **fluid_leak has 4 positives.** F1 0.19 is a data problem as much as a model problem.
5. **NHTSA text is a proxy.** Thresholds tuned here will drift on real claim text. Re-tune before production.

## 5. Decisions taken

**Chosen for the PoC:** Java, Spring Boot, DJL `BertFullTokenizer` (pure Java WordPiece) plus ONNX Runtime (Microsoft, C++ binary via JNI) running bge-small on CPU. Concept vectors embedded once at startup. Sentence split, cosine, per-concept threshold, evidence sentence returned. Negation via CoreNLP or OpenNLP clause scoping on the matched sentence only. Full detail in the design doc, sections 3 to 8.

**Rejected and why:**

| option | reason |
|---|---|
| Lexical keyword match | no paraphrase, fails the core requirement |
| Third-party embedding or LLM API (OpenRouter, Hugging Face, decision-model SaaS) | data egress |
| Azure OpenAI embeddings | egress outside the app boundary, held as fallback only |
| Laya self-hosted | too slow on CPU, Python reference inference, custom head not a stock ONNX export |
| CLM | needs GPU, same objection as Laya but larger |
| HF `tokenizers` | Rust |
| GloVe averaging | pure JVM but materially worse, fallback of last resort |
| Cross-encoder per keyword | one full model pass per concept, does not scale |

**Reopened after the doc was written (2026-09-23, later the same day):**

- An internal AI gateway team liked Laya. That gateway is a proxy to external LLM APIs. It cannot host Laya. Routing through it to an external model is the egress being avoided. This mismatch must be reconciled before anyone builds on "Laya via the gateway".
- Azure AI Foundry, or Azure AI Language custom multi-label classification, sits inside the tenant trust boundary. That is a materially different posture from a third-party API and may clear governance where OpenRouter could not. Trade-off: a trained classifier learns negation from examples but every new concept means retrain and redeploy, and needs roughly 50 labeled examples per concept. Embeddings add a concept by embedding one string. If the concept list churns, that retraining tax decides it.
- Auth for any Azure service: Managed Identity via `DefaultAzureCredential`, never an API key in config, image or repo. If a key is unavoidable, Key Vault at runtime, fetched with a managed identity.

## 6. Blocking question for the team

ONNX Runtime is a Microsoft-owned pre-built C++ binary loaded through JNI. Performant transformer inference on the JVM cannot avoid native numeric code. The proposed line is: no Rust, no Python, no non-JVM source in our codebase or build, but Microsoft-owned native binaries are acceptable, same category as the JVM itself. **If the team rejects even that, the only on-prem path is GloVe averaging, which is materially worse, and the tenant-hosted Azure classifier becomes the primary option instead.** Get this ruling before writing code.

## 7. Open items, in priority order

1. **Hand-label the negation subset.** About 110 cells across `brake_problem`, `warning_light`, `fluid_leak`. Rules in `LABELING_README.md`. Until this is done every accuracy number is a model's opinion.
2. **Fill the 38 blank cells** in `labeling_sheet.csv`.
3. **Add an Options section** to the design doc: (A) self-hosted DJL/ONNX embeddings, (B) tenant-hosted Azure Foundry or Azure Language classifier, (C) Laya or CLM on GPU. Columns: trust boundary, ops ownership, add-a-concept cost, measured accuracy, latency on prod hardware.
4. **Measure embeddings CPU latency** on the same single claim used for the Laya timing, so both sides are measured rather than one measured and one estimated.
5. **Agree acceptance targets** with the product owner per concept, precision and recall, before building the negation loop.
6. Only if GPU budget appears: finish Laya and run CLM on the six PO concepts as a ceiling row.

## 8. Suggested first actions for the assistant in this environment

- Read `PoC-semantic-concept-detection.md` sections 1 to 4 and 12.
- Confirm the language and dependency policy in section 3 against the actual repo and build rules here. Report any mismatch before anything else.
- Check whether ONNX Runtime Java (`com.microsoft.onnxruntime:onnxruntime`) and DJL are already approved dependencies in this environment. If not, that is the first conversation to have.
- Check whether an Azure AI Language or Foundry resource already exists in the tenant. If yes, option B gets cheaper and the Options section in item 3 above becomes the priority.
- Do not start the build sequence in section 11 until the blocking question in section 6 has a ruling.

# Matching Methods

This document is the menu of methods to test for task→icon matching, written paper-style: clear
baselines, a small set of strong core methods, and a broad "advanced / novel" section for
experimentation and as a data-science showcase. The experimental setup (data, validation, metrics,
abstention rationale) is in [experimentation-strategy.md](experimentation-strategy.md).

Every method below produces a **ranking** of icons for a query, then passes through a shared
**abstention gate** (§Abstention) that returns the top-1 icon **or `None`**.

Notation: a query `q` is a normalized task (title + tags). An icon `I` has a description document
`d(I)` built from `visual_concepts` / `task_intents` / `example_tasks` / `reasoning`, a quality
score `u(I) = icon_usefulness/10`, and `poor_matches(I)`.

---

## Baselines

### B0 — current-approach replica
Reproduces today's matcher as the reference point: a single concatenated description per icon, the
English `BAAI/bge-small-en-v1.5` bi-encoder, cosine similarity, top-1, fixed threshold 0.50. Lets
us quantify improvement against the status quo on identical inputs.

### B1 — lexical
**BM25 / TF-IDF** over the concatenated icon documents. Fast, interpretable, no model download,
strong on literal keyword overlap. Also the engine behind the manual/iPad keyboard search.

---

## Core methods (the POC ships these)

### M1 — field-weighted dense bi-encoder + quality prior
Encode `q` and `d(I)` with a **multilingual** sentence encoder (default
`intfloat/multilingual-e5-small`; alternatives `bge-m3`, `gte-multilingual-base`). Score =
weighted combination of per-field cosine similarities (`visual_concepts`, `task_intents`,
`example_tasks`) plus a **quality prior** in `u(I)`:

```
score(q, I) = Σ_field w_field · cos(enc(q), enc(field(I)))  +  λ · u(I)
```

The multilingual encoder is the **default** (tasks may be ES/RU); `bge-small-en` is kept only as
the English-only baseline (B0). Instruction prefixes (E5/BGE style) applied to the query.

### M2 — nearest-example voting (asymmetric)
Match `q` against each icon's individual `example_tasks` (task↔task similarity), then aggregate
per-icon (max / mean / softmax-vote). Exploits that `example_tasks` are *literally tasks*, so the
query and the index live in the same distribution — often the strongest single signal.

### M3 — hybrid (recommended POC engine)
**Reciprocal Rank Fusion (RRF)** of B1 (lexical) + M1 and/or M2 (dense). Combines literal overlap
with semantic generalization. **Recommendation: ship M3 + quality prior + calibrated abstention
gate** as the POC's default; B0/B1 as comparison baselines.

---

## Abstention gate (wraps every method)

Turns a ranking into "top-1 or None". Approaches compared:

- **Calibrated score threshold** — calibrate `score → P(acceptable)` (Platt / isotonic /
  temperature), then threshold the calibrated probability. (Raw cosine isn't comparable across
  queries; calibration is what makes a global threshold valid.)
- **Margin / entropy** — top1−top2 gap, or entropy of the softmax over candidate scores.
- **OOD / density** — Mahalanobis distance of `enc(q)` to the icon-embedding cloud, or kNN density;
  flags queries with no good icon nearby.
- **Conformal prediction** — distribution-free coverage guarantee on the "shown" set at a chosen
  risk level.
- **Learned gate** — logistic regression / small MLP over features (top-k scores, margins, lexical
  overlap, `u(I)`, OOD distance) predicting "acceptable@1".

Trained/calibrated on the "no good icon" verdicts, deliberately-included negative tasks, and
`poor_matches`. Evaluated by the precision–coverage curve and gate AUROC/AUPRC.

---

## Advanced / novel methods (the showcase)

Grouped and described. Not all ship at once — this is the experimentation surface and a deliberate
breadth of modern techniques to practice and present.

### Representation & embeddings
- **Multi-vector / multi-aspect** — keep one embedding per field; score by best/weighted field
  match rather than collapsing to one vector.
- **ColBERT-style late interaction** — token-level embeddings with max-sim; captures fine-grained
  term alignment between task words and concept words.
- **SPLADE** — learned sparse representations (term-weighted), bridging lexical and dense; runs on
  an inverted index for speed.
- **Instruction-tuned embedding prompts** — task-specific prompts for E5/BGE/GTE ("Represent this
  task for retrieving a matching icon concept").
- **HyDE (Hypothetical Document Embeddings)** — have an LLM write a hypothetical *icon description*
  for the task, embed that, and retrieve — aligns query and corpus modalities.
- **query2doc / doc2query expansion** — expand the query (or pre-expand each icon document) with
  generated text to improve lexical/dense recall.
- **Pseudo-relevance feedback (Rocchio)** — re-rank using the top initial hits to refine the query
  vector.
- **Test-time augmentation** — average embeddings of several LLM paraphrases of the query.
- **LLM-as-encoder** — last-token embeddings from an instruct LLM (`e5-mistral-7b-instruct`,
  `gte-Qwen2`) for stronger semantics (heavier; candidate for offline/oracle or distillation
  source).
- **Matryoshka embeddings** — train/representation that allows truncating dimensions for a
  speed/quality dial.

### Set / distance matching
- **Word Mover's Distance / Optimal Transport (Sinkhorn)** — treat the query as a bag of word
  embeddings and the icon as a bag of `visual_concepts`/`task_intents` embeddings; match by minimal
  transport cost. Naturally handles "several concepts partially match".
- **Energy-based matching** — a learned compatibility energy between query and concept sets.

### Multimodal (distinctive — we have the rendered PNGs)
- **CLIP / SigLIP text→image retrieval** — embed the actual rendered icon glyph image and match the
  task text against it directly, *without* descriptions. A genuinely different signal and a strong
  story (the icons are literally images).
- **Fusion** — combine the image embedding with the description-text embedding (late fusion or a
  learned gate).
- **Vision-language LoRA fine-tune** — adapt CLIP on our (task, icon-image) pairs once we have
  labels.

### Training / metric learning
- **Contrastive fine-tuning** — InfoNCE/triplet on a small bi-encoder, in-batch negatives +
  `poor_matches` as **hard negatives** (see experimentation-strategy §4).
- **LoRA / PEFT** — parameter-efficient fine-tune of an embedding model on our labels.
- **Embedding adapters** — a small linear/MLP projection head on top of frozen embeddings, trained
  on labels; cheap, fast, surprisingly effective.
- **Cross-encoder reranker** — jointly encode (query, candidate) for the top-N; high accuracy,
  higher latency → used offline or distilled (below).
- **monoT5 / RankT5** — sequence-to-sequence rerankers.
- **Distillation** — distill cross-encoder or LLM-listwise rankings into the fast bi-encoder
  (margin-MSE / KL); buys reranker-quality at bi-encoder latency.
- **Learning-to-rank (LambdaMART)** and a small **MLP reranker** over cheap features (per-field
  sims, popularity, `u(I)`, lexical overlap, OOD distance).
- **SetFit** — few-shot contrastive fine-tuning when labels are scarce.
- **Multi-task** — jointly predict relevance + `icon_usefulness` (shared representation).
- **Curriculum + ANCE** — iterative hard-negative refresh during training.
- **Self-training / pseudo-labeling** — bootstrap from silver labels, retrain, repeat.

### LLM-augmented
- **Two-stage retrieve → LLM listwise rerank (RankGPT-style)** — the quality oracle / upper bound,
  and the basis of the silver judge. Too slow for real time, used for data/eval.
- **LLM query expansion** — synonyms/paraphrase to help embedding recall.
- **RAG explanation** — generate a short rationale for *why* an icon was chosen (useful for the
  labelling UI and debugging).

### Graph / structure
- **Icon–icon similarity graph + label propagation** — spread sparse labels across visually/
  semantically similar icons.
- **Small GNN** over the icon graph for representation refinement.
- **WordNet / ConceptNet expansion** of `visual_concepts` to widen concept coverage.

### Calibration & abstention (also core — see the gate above)
- Platt / isotonic / temperature calibration; **conformal prediction**; **OOD detection**
  (Mahalanobis, kNN density); entropy/margin gates; **MMR** for diverse candidate sets in
  labelling.

### Uncertainty / Bayesian
- **Gaussian / probabilistic embeddings** (embed as a distribution, not a point).
- **MC-dropout** confidence; **deep ensembles** — both also feed active learning (BALD).

### Personalization / online (forward-looking)
- **Contextual bandits / Thompson sampling** — learn the *user's* icon preferences from agenda
  feedback over time.
- **Off-policy / counterfactual evaluation** — evaluate new matchers from logged user choices
  without a live A/B.

### Efficiency / latency
- **int8 / binary quantization** of embeddings; **ONNX / OpenVINO** runtime for the encoder.
- **FAISS / HNSW** approximate nearest-neighbour over the 4k icons (likely unnecessary at 4k, but
  part of the latency study).
- **LRU query cache**; **quantization-aware distillation**.

### Active-learning strategies
- Uncertainty, **query-by-committee**, **core-set**, **BALD**, expected-model-change — see
  experimentation-strategy §6.5.

---

## Latency strategy

Icon embeddings are **precomputed offline** and cached to `.npz` (mirroring the existing matcher's
cache pattern). At query time: encode the query **once** with a small/quantized model, compute
cosine over the cached matrix with numpy (4k icons is trivial; FAISS optional), apply the quality
prior and gate, and **LRU-cache** results. Target: low-single-digit milliseconds per query after
warm-up, well within the real-time agenda-render budget.

## Generality / anti-overfit

Train/tune on generic + synthetic data; validate on held-out personal + multilingual; **report per
split**. The danger is overfitting to the user's personal task style — separate in-domain vs
generic numbers make that visible. Fine-tuned models are validated on the generic and multilingual
splits specifically to confirm they generalize.

## Recommendation for the POC

Ship **M3 (RRF of B1 + M1/M2) + quality prior + a calibrated abstention gate**, with a multilingual
default encoder, and B0/B1 as baselines in the leaderboard. Everything in "Advanced / novel" is
queued as subsequent experiments.

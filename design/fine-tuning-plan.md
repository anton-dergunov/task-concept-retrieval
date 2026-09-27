# Fine-tuning plan: data splits, training data, architectures, LLM judge

How the human labels are used, what a model is trained on, and which architectures to try.
Constraints from CLAUDE.md apply throughout:

- matching runs on the agenda's render path, so it must be fast and local;
- LLMs are for building data and judging only;
- the matcher must be able to abstain.

## 1. Roles of the datasets

| Dataset | Role |
|---|---|
| **Personal** (private) | The target distribution, and the primary result |
| **Personal-synth** (public) | The publishable stand-in. Its `origin == "skeleton"` subset follows the private topic mix; the `oversample` / `life_area` rows add everyday life and career |
| **Realistic**, **Public-short**, **Public-expanded** | Generality: short vs. long tasks, EN/ES/RU |

**No eval dataset is ever used for training.** Training data comes from separate pools
(§4), made disjoint from the eval sets by source id.

## 2. Dev / test split

Every labelled task is assigned to **dev** or **test** by a hash of its id
(`tcr.datasets.split_of`, 50/50). This is not decided by labelling order. The labelling
tool alternates test and dev tasks, so any labelled prefix is split exactly in half.

- **test**: the uncontaminated half. It is touched only for final numbers, once per
  reported result, and never inspected while developing.
- **dev**: everything else. That includes hyperparameters, calibrating the abstention gate
  (the "no good icon" labels), calibrating and validating the LLM judge, model selection,
  and error analysis.

## 3. Is the labelled data enough for evaluation?

Yes, with these targets.

- **One method's top-1 accuracy**, as a 95% interval half-width:

  | test tasks | 250 | 500 | 1,000 |
  |---|---|---|---|
  | ± points | 6.2 | 4.4 | 3.1 |

- **Two methods compared on the same tasks** (McNemar, 80% power, about 25% of tasks where
  they disagree): a 5-point difference needs about 800 test tasks; a 3-point difference
  needs about 1,700.
- **Suggested labelling targets:**
  - about 1,600 Personal tasks (≈ 800 test + 800 dev);
  - about 500 Personal-synth;
  - about 300 each of Public-short and Public-expanded.
- **Beyond that:** label by active learning (matcher disagreement, low gate confidence),
  not at random.
- **Skipped and "no good icon":** "skip" lowers the effective n. "No good icon" labels are
  useful: they are the negatives the abstention gate needs.
- **Graded rankings, not just top-1:** labels are ranked lists, so nDCG@k and MRR are
  available and more sample-efficient than top-1 accuracy.

## 4. Training data: generated labels, not human labels

Human labels are too few to train on: a few thousand tasks against 3,486 icons, most of
which would never appear. They are the yardstick that shows whether training helped.
Training data comes from:

1. **Icon `example_tasks`**: 10–20 per icon, about 50k (task, icon) positives for free.
   `poor_matches` provide about 25k hard negatives. These are "training-only" signals,
   never used for eval.
2. **LLM-judge labels (silver) on unlabelled task pools**, disjoint from every eval set:
   - the ~8.5k MS-LaTTE titles not used by Public-short or Public-expanded, short and
     expanded;
   - the MASSIVE remainder;
   - a fresh synthetic pool generated from the public life-area skeletons;
   - developer TODO corpora.
3. **In-batch and mined hard negatives**: icons the current matchers rank highly but the
   judge rejects.

## 5. Model formulation: score (task, icon) pairs; do not classify over 3,486 classes

A classification head over a fixed icon vocabulary cannot learn icons it has not seen, and
most icons will have few or no positives. Scoring a *pair* — the task against the icon's
description and/or image — generalizes to unseen icons, and to icons added later.

Two stages, matching the latency budget:

- **Retriever (the hot path):** a dual encoder. Task text goes to a vector; icon vectors
  are precomputed offline (3.5k of them). One text encode plus a dot product, a few ms on
  CPU. Candidates: multilingual-e5 or Qwen3-Embedding-0.6B for text. Fine-tuned with a
  contrastive (InfoNCE) loss on §4 data with hard negatives.
- **Reranker (optional, over the top 30–100):** a small model that scores each candidate,
  batched. A model under 2B parameters, LoRA fine-tuned, with a pointwise relevance
  logit, or a listwise head. It is the quality ceiling for local inference.
- **Hybrid:** the "icon as a learned vector" idea also fits the dual encoder. Initialise
  each icon's vector from its description embedding and let fine-tuning adjust it. That is
  a classification head that still generalizes, because it starts from descriptions.

### 5.1 Using the icon images

Two families. Model names are examples to verify at implementation time.

**(A) Two towers (CLIP-style, cheap).**

- Icon side: fuse a *text* embedding of the description with an *image* embedding of the
  glyph, from SigLIP 2 (multilingual text tower) or jina-clip-v2 (one model with text and
  image towers in a shared space, 89 languages). Task side: a text embedding.
- Start with late fusion: a learned weighted sum of description and image similarity.
  Then fine-tune a small projection on top.
- Icon vectors are precomputed, so the image model costs nothing at query time.
- Caveat: CLIP-family models were trained on photos. On abstract monochrome glyphs,
  zero-shot quality is modest, so fine-tuning or the description text carries most of the
  signal. The descriptions are already an image → text distillation made by Gemini, so
  measure what the image adds on top of them before investing.

**(B) One vision-language model (a single network, expensive).**

- The task text, the icon image, and optionally its description, in one prompt. Examples:
  a 2–3B Qwen-VL-family model, SmolVLM2, or a multimodal embedding model built on a VLM
  (GME-style), which gives a single network that embeds both sides.
- **As a reranker**, score each (task, icon image) pair, or one prompt with a numbered
  contact sheet. Image cost is small: at 28 px per visual token, a 128 px icon is about
  20 tokens, and a 30-icon sheet at about 900 px is about 1k tokens. Context length is not
  the constraint; latency is: 30–100 forward passes, or one long listwise pass, per task.
- **As an embedder** (GME-style), icon vectors can again be precomputed, so it fits the
  retriever slot. This is the closest thing to "one network for image, description and
  task".
- Adding vision to a text-only 2B LLM means training a projector (LLaVA-style) and is a
  project in itself. Start from a model that already has vision.

**Recommended order:**

1. text-only dual encoder, fine-tuned;
2. add the image tower (A) as a fused feature;
3. try a VLM reranker (B) on the top 30;
4. distil (3) back into (1) if it wins.

Each step is kept only if it beats the previous one on **dev**.

## 6. The LLM judge

**Input format.** One request per task: the task (title + body) and the top-30 candidate
icons pooled from several matchers (as in the labelling tool). Two ways to identify
images:

- interleave a text label ("Icon 7:") before each image part;
- one **contact-sheet image** with large printed numbers on each cell. This is cheaper, is
  a single image, and the ids are unambiguous.

The judge returns graded relevance (0–3) per icon, plus "no good icon".

**How many icons per request.** APIs accept far more images than needed. The limit that
matters is accuracy: small glyphs, numbering errors, and attention spread thin.
**20–30 icons per request** is the practical range; above about 50, quality should be
expected to drop. Candidates come from the retrievers, so the judge never needs the whole
set.

**Bias control.**

- Shuffle the order and ask 2–3 times with different permutations; average, and keep only
  stable judgments.
- Report position-vs-choice statistics, as the labelling tool does for humans.
- Compare an image judge against a text-only judge that reads the descriptions.

**Calibration on dev before any large run.**

- Agreement with human dev labels: top-1 agreement, nDCG of the human ranking under the
  judge's grades, and κ on the "no good icon" decision.
- With 300–500 dev tasks these are estimated to about ±4–5 points, which is enough to pick
  a judge (model, prompt, image vs. text) and set grade thresholds.
- Gemini vs. Claude vs. a third model on the same dev tasks gives the committee setup from
  experimentation-strategy §6.3.

**How much judge data to generate.**

- **Minimum useful:** about 3k tasks × 30 graded candidates, about 90k graded pairs. Enough
  for a LoRA or projection fine-tune of a small dual encoder to show whether it helps.
- **Comfortable:** about 10k tasks, about 300k pairs, plus the ~50k free example-task
  positives. At this scale small rerankers and embedders are typically distilled from LLM
  judgments.
- **Measure, don't guess:** train at 1k, 3k and 10k judged tasks and plot dev nDCG. Stop
  when the curve flattens. The learning curve is itself a reportable result.
- **Cost and time:** one request per task, with about 1k image tokens plus text. Ten
  thousand requests are a few days of free-tier quota, or small spend on Vertex.
- **Icon coverage:** at about 3 positives per task, 10k tasks give about 30k positives
  (~9 per icon on average, very skewed). Description-based scoring covers the tail.

## 7. Experiment order

1. Label: about 400 Personal (200 dev / 200 test) first, then continue toward the targets
   in §3.
2. Baselines on dev: B1, M1, M2, M3, plus zero-shot image towers (SigLIP 2 / jina-clip-v2).
3. Judge calibration on dev (§6); pick the judge.
4. Judge about 3k training-pool tasks; fine-tune the dual encoder; learning curve;
   then 10k if it keeps improving.
5. Add image features; try the VLM reranker; calibrate the abstention gate on dev.
6. One final run on **test** for each dataset.
7. Report **eval-set fidelity**: does the method ranking on Personal-synth (skeleton
   subset) match the ranking on Personal? (design/anonymization.md §4)

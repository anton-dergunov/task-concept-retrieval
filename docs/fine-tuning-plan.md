# Fine-tuning plan: data splits, training data, architectures, LLM judge

How the human labels are used, what a model is trained on, and which architectures to try.
Constraints from CLAUDE.md apply throughout:

- matching runs on the agenda's render path, so it must be fast and local;
- LLMs are for building data and judging only;
- the matcher must be able to abstain.

**Target behaviour.** A good icon is the glanceable reminder of what a task is about:
central object → context/activity → meaningful action, never properties. Abstain if
nothing fits. This is defined once in [icon-selection-principles.md](icon-selection-principles.md)
and used verbatim by the labeller, the judge, the description prompt and the model's
instruction prompt.

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
- **judge-dev**: the first 51 labelled Personal tasks (2026-11-13), on which the LLM judge's
  prompt was developed (§6.1): 5 are examples in its prompt, 46 were read for error
  analysis. Both their test and dev halves were looked at, so they count for neither.
  `split_of` returns `"judge-dev"` for the ids in `data/private/judge/judge_dev_ids.txt`.
  Labels made after that keep alternating test and dev.

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

**Model input.** The query is the task title plus its **nearest parent heading**
("Music catalog support: Evaluate bids against my catalog rules"). The body is optional
(first ~400 characters): the title says *what* the task is, the body mostly says *how*.
Datasets carry `parents` where the source has them: org outline ancestors (Personal,
Realistic) and informative MS-LaTTE list names (Public-short). "Title vs. title +
parent vs. + body" is an ablation on dev. For an LLM-based reranker, the instruction
prompt is the selection principles in short form, so fine-tuning starts from the intended
behaviour rather than learning it from labels alone.

**Graded targets.** Human ranks, judge grades, and v2 `example_tasks` (rank 1) vs.
`context_tasks` (rank 2) are all graded, so train with a graded or listwise loss and
evaluate with nDCG, which rewards getting the central subject first.

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

The prompt includes the parent headings and the selection principles
(icon-selection-principles.md): central subject first, context second, never properties,
siblings distinguishable. The judge returns graded relevance (0–3) per icon: 3 = shows the
central subject, 2 = shows the context or activity, 1 = partial, 0 = no. It also returns
"no good icon".

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

**Model choice and cost** (checked 2026-09-27 against Google's published Gemini API
prices; Vertex normally matches, but check the console before a large run).

Per task:

- **input** about 1,800 tokens: the contact sheet is about 1,100 tokens at high media
  resolution (Gemini 3 counts about 280 / 560 / 1,120 tokens per image at low / medium /
  high), plus about 700 tokens of instructions and task text;
- **output** about 1,500 tokens: about 400 of JSON grades plus low-level thinking.

Output is billed at 6–8× the input rate, so it dominates the cost.

| Model (Vertex) | $/1M tokens in / out | 1,000 tasks, one pass | batch (−50%) |
|---|---|---|---|
| Gemini 3.1 Pro (preview) | 2.00 / 12.00 | ~$22 (up to ~$40 with long thinking) | ~$11–20 |
| **Gemini 3.5 Flash** | 1.50 / 9.00 | **~$16** | **~$8** |
| Gemini 3 Flash (preview) | 0.50 / 3.00 | ~$5 | ~$3 |
| Gemini 3.1 Flash-Lite | 0.25 / 1.50 | ~$3 | ~$1.40 |

- **Selection:** run 3.1 Pro, 3.5 Flash and 3 Flash on about 200 dev tasks (a few dollars
  in total) and measure agreement with the human labels. Use the cheapest model within
  about 2–3 points of the best for bulk judging; keep Pro for tasks where the permuted
  runs disagree.
- **Permutations:** three permuted passes triple the cost. With 3.5 Flash that is about $50
  for 1,000 tasks, or about $25 in batch.
- **Cheaper output:** ask only for the icons that fit (top 5 with grades, plus "no good
  icon") instead of grading all 30, and use the lowest thinking level. Together these cut
  output about 2–3×.
- **Batch mode:** Vertex batch prediction returns within hours, which is fine for dataset
  building.

### 6.1 As built (2026-11-13)

The judge was built and tuned before many human labels existed, because Gemini access
may end at any time. It works the way the author labels
([icon-selection-principles.md](icon-selection-principles.md), "How to choose"), in
single-request stages (`tcr/judge.py`):

- **A, scan and plan:** the task (title, parent headings, body start, up to 10 sibling
  titles), 5 of the author's labelled tasks as worked examples (in words), and the same
  ~100 candidates the author saw in the labeller, as 4 numbered contact sheets with one
  line each from the v2 description (never names). Output: the analysis (central subject,
  what distinguishes it, associations, qualifiers), every somewhat relevant candidate, and
  up to 4 description searches.
- **Searches** run locally (M3 on v2, 10 new icons per query).
- **B, decide:** the analysis, the collected icons and the search results → a ranking
  graded 3 / 2 / 1, or "no good icon"; B may search once more.

**Tuning.** On the judge-dev set (§2): 46 validation tasks, two runs per version, error
analysis between versions, rules kept general (never a hint for one task). Each version
fixed a pattern of misses: the action or a side noun instead of the subject, and puns (j2);
the parent heading's domain (j3); the plain object before clever aspects (j4, neutral;
frozen). Run-to-run noise: hit@1 differs on 3–6 of 45 tasks.

| Frozen prompt j4 (2 runs each) | hit@1 | hit@1, icons the author saw | exact@1 | nDCG@5 | $ / task, batch |
|---|---|---|---|---|---|
| matcher M3.v2 (no LLM) | 0.33 | — | 0.09 | 0.21 | 0 |
| Gemini 3.8 Flash, low thinking | 0.35 | 0.42 | 0.10 | 0.28 | 0.007 |
| Gemini 3.1 Pro | 0.27 | 0.34 | 0.12 | 0.26 | 0.024 |
| **Gemini 4 Argon** | **0.50** | **0.58** | **0.26** | **0.38** | 0.019 (promo) |

hit@1: the judge's first icon is among the author's picks. Argon against Flash, per task:
hit@1 better on 10, worse on 1; nDCG@5 better on 26, worse on 7. Medium thinking did not
help Flash. **Argon is the judge** for Personal, the Personal-style pool, Personal-synth and
Realistic; Flash for the easy public sets. The bias is known: the prompt was tuned on these
46 tasks, so the unbiased check is the author's labels made after 2026-11-13.

Development: `scripts/judge_dev.py` (run / score), the private log
`data/private/judge/iterations.md` and report `data/private/judge/report.html`
(`scripts/judge_report.py`); metrics only in `results/judge_dev/summary.json`. Bulk runs:
`scripts/judge_batch.py`, hand-off in [judge-bulk-run.md](judge-bulk-run.md).

## 7. Icon description quality (v2 descriptions)

The descriptions are the only matching signal, and v1 had measurable problems: glyph
misreadings, short example tasks reused across icons, and generic repeated negatives.
v2 (done 2026-11-06) fixes these with a new schema and prompt (depiction → readings →
examples → hard near-misses → anchored scores), a contrast pass over look-alike glyphs,
a blind QA pass and a contamination guard. In a blind check of 200 random glyphs, v1 was
wrong or partly right on 27% and v2 on 0.5%. Whether v2 also matches better is decided on
dev labels. Details: [icon-descriptions.md](icon-descriptions.md).

## 8. Experiment order

Human dev labels are the bottleneck: almost every decision waits on them. So labelling
starts first, and work that needs no labels runs alongside it.

1. ~~Regenerate the icon descriptions (v2, §7)~~ (done 2026-11-06). The labelling pools
   were rebuilt over v1 + v2 the same day.
2. **Label** Personal tasks, alternating test and dev, toward the targets in §3. The first
   51 became the judge-dev set (§2); the labels after them give the final numbers and the
   unbiased check of the judge.
3. ~~The LLM judge (§6.1)~~ (done 2026-11-13) and the training pools: the private
   Personal-style pool (~5,000, `synth_personal.py pool`) and 3,000 MS-LaTTE titles
   (`data/train_pools/`). **Bulk judging** of every dataset and pool with Vertex batch, in
   priority order, while access lasts: [judge-bulk-run.md](judge-bulk-run.md).
4. **Still label-free** ([next-label-free-work.md](next-label-free-work.md)): a first
   fine-tune of the multilingual dual encoder on v2's free signal; zero-shot image-tower
   baselines (SigLIP 2 / jina-clip-v2) and their fusion with M1.
5. **With ~200 dev labels:** v1 vs v2 (icon-descriptions.md §7); baselines B1, M1, M2, M3,
   the image towers and the fine-tune on dev; the judge against the new labels.
   **Fine-tune the dual encoder on judge labels** of the pools + v2's free signal;
   learning curve at 1k / 3k / 8k pool tasks. This is the step most likely to produce the
   best fast model.
6. **Add image features; try the VLM reranker** and distil it into the dual encoder if
   it wins; fit the usefulness prior and calibrate the abstention gate on dev ("no good
   icon" labels).
7. One final run on **test** for each dataset; export the winner as a bundle for the
   agenda ([deployment.md](deployment.md)).
8. Report **eval-set fidelity**: does the method ranking on Personal-synth (skeleton
   subset) match the ranking on Personal? (docs/anonymization.md §4)

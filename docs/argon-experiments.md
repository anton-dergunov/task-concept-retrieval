# Experiments that need Argon labels (menu, from 2026-11-21)

For the author and for a fresh Claude Code session. Six experiments that need paid Argon
judging (Gemini 4 Argon on Vertex), in the order the author chose to do them, each with a
rough cost. They come after Kaggle round 5
([next-data-and-distillation.md](next-data-and-distillation.md) §6), whose act1k vs
rand1k answer decides how #1 is shaped. Argon's promotional price ends after 2026-11-30;
after that every cost below doubles.

**Budget (author, 2026-11-21):** ~$50 beyond round 3's cap, stretching to ~$100;
the rest of the Google credit is kept for other projects. Raise the cap by deleting
`data/private/augment/cap-round3.json` and rerunning a stage with `BUDGET=<new money>`:
the new cap is the spend log at that moment plus that amount.

## What a dollar buys

- **Judged tasks:** ~$19 per 1,000, all in (v2, 2026-11-21: Argon judging, ES/RU
  translations, invented parents, four rewrites each, at the promotional price). So
  $50 ≈ 2,500 tasks.
- **Quality:** the learning curve gives about **+0.03 nDCG@5 per tripling** of the judged
  pool (0 / 1k / 3k / 8.9k tasks → 0.330 / 0.364 / 0.393 / 0.428, round 2). From
  ~11.9k judged tasks (after round 5):

  | spend | judged tasks | random tasks | if active is ~2× as label-efficient |
  |---|---:|---:|---:|
  | first $50 | 11.9k → 14.4k | ~+0.005 | ~+0.01 |
  | second $50 | → 16.9k | ~+0.004 | ~+0.008 |
  | after 11-30 (price ×2) | half the tasks per $ | ~+0.002–0.003 | ~+0.005 |

  So the money buys roughly +0.005 per $50 with random selection, more only if round 5
  shows that active selection is worth more per label. Each experiment below is also
  worth doing for the technique it teaches, which is half the point of this project.

## How the active tasks were chosen (round 3, for reference)

`scripts/select_active.py`, on 16.7k unlabelled candidates scored by three fine-tuned
models (e5-large teacher, e5-base second teacher, e5-small student):

- priority = (1 − the teacher's softmax probability of its top-1): **uncertainty**
  + 0.5 if the teacher's top-1 is not in the student's top 5: where the deployed model is
  probably wrong
  + 0.5 if the two teachers disagree on the top-1: **query-by-committee**.
- Guards against active learning's known failure (the most "uncertain" items are junk
  that no icon fits): the lowest 15% by teacher top-1 score are never chosen; at most 4
  tasks per teacher top-1 icon; near-duplicate topics skipped (top-10 overlap ≥ 7);
  ≤ 33% MS-LaTTE. The teacher's top 15 join the judge's candidates.
- Result: 2,000 active + 1,000 random. The active ones are harder: on them the teacher's
  hit@1 is 0.559 against 0.761 on the random ones (e5-small: 0.506 vs 0.702).

**Repeating it** (a second loop on the actively trained model) is the standard pool-based
loop. Three rules: select with the newest models (round 5's start scores the candidates
with round 4's; for selection by round 5's own models, add a ~1 h Kaggle scoring
session); keep a random third in every batch, to keep the training distribution anchored
and to measure whether selection still beats random; watch label noise, which rises with
hardness (compare no_good_icon and the grade spread of active vs random labels).
Gains usually shrink from one loop to the next.

## The menu

### 1. Active-learning strategy study (~$30–50; can be round 6's batch)

**Technique:** pool-based active learning, comparing acquisition functions.
**Design:** one batch split into equal arms, e.g. 4 × 600 tasks (~$46): uncertainty
only (1 − p₁), committee only (teacher disagreement and student-not-in-top-5),
diversity-first (cluster the teacher's text embeddings and take uncertain tasks near
cluster centres, as in core-set / BADGE), and random. Each arm also gets the teacher's top
15 as judge candidates, so arms differ only in which tasks they hold.
**Measure:** one e5-small per arm on Kaggle (free), two ways: on top of everything
(the realistic gain, small, probably within the intervals) and alone with the free text
(F1@1k's recipe, where ~600 tasks still move nDCG@5 by several points). Plus each arm's
hardness on its own labels (teacher hit@1, no_good_icon rate).
**Code:** `select_active.py --strategy …` (arms; leave out near-duplicates of the 3,000
judged tasks, since round 4's models have not seen their labels), the arms' `sel`
values in train_f1.py presets, a v3 stage in `vertex_round3_jobs.sh`.
**Shape by round 5:** if F1@1k-act clearly beats F1@1k-rand, weight the arms toward
active strategies; if they tie, the study is the question itself (which signal, if any,
helps), at a smaller size.

**Round 5's answer (2026-11-21, f1-dual-encoder.md §13):** F1@1k-act is *worse* than
F1@1k-rand (ΔnDCG@5 −0.007 [−0.010, −0.003]) and they tie on top of the pool. The combined
uncertainty + committee selection hurts when its tasks are most of the data. If #1 runs, the
diversity-first arm (hard tasks spread over clusters, not piled on the hardest) is the one
worth testing against random; uncertainty-only and committee-only would likely repeat this
result. Decide together with round 5's all3k point (next-data-and-distillation.md §6).

### 2. Model-in-the-loop adversarial tasks (~$15–25)

**Technique:** adversarial data collection (Dynabench style): create hard examples rather
than find them.
**Design:** show Flash (cheap) the current model's errors on judged pool tasks (task,
the model's wrong top-1, the judge's right icon) and ask for new tasks of the same kind:
the same confusion, other subjects. ~1,000 tasks generated (~$3 Flash), filtered against
the eval sets and near-duplicates, judged by Argon (~$19), with translations and rewrites.
**Measure:** the fooling rate (how often the current model's top-1 misses on the new
tasks, against ~30% on random pool tasks), then a student trained with them vs without.
**Risk:** generated text drifts from real tasks; keep the Personal-style prompt rules of
`synth_personal.py`.

### 3. Minimal pairs / counterfactuals (~$10–20)

**Technique:** counterfactual data augmentation, contrast sets.
**Design:** Flash edits a judged task so that the right icon should change ("Call the
plumber about the sink" → "Email the plumber about the sink"), ~1,000 pairs; Argon judges
the edited task (the original is judged already). Pairs where the judge's icon did not
change are dropped or kept as invariance examples.
**Measure:** a contrast-set accuracy (does the model's top-1 follow the edit?) on held-out
pairs, before and after training on the rest; silver as usual.
**Code:** an `augment_pools.py minpair` kind, a `pair` loss term in train_f1.py (the edit's
icon above the original's, and the reverse).

### 4. Pairwise preference labels (~$10–20)

**Technique:** preference data and ranking losses (the retrieval cousin of RLHF/DPO).
**Design:** for ~2,000 pool tasks where two models disagree, Argon answers "which of
these two icons fits the task better, or neither?" (short prompts, cheaper than full
judging). Train with a pairwise margin loss on top of the graded InfoNCE.
**Measure:** agreement of pairwise answers with the graded labels where both exist (how
reliable are absolute grades?), then the student with vs without the pairs.

### 5. VLM reranker as a teacher (~$20–40)

**Technique:** multimodal teacher, cross-modal distillation (backlog of
next-data-and-distillation.md, "Images"; next-label-free-work.md step 5).
**Design:** Argon sees the task and the teacher's top 20 as glyph images (no names),
returns a ranking, for ~2,000–3,000 tasks. Its rankings become soft targets for the
e5-small student: image knowledge at no query cost.
**Measure:** the reranker alone on silver dev (is it better than F1@all-large-e8?), then
the student distilled from it vs from the text teacher.

### 6. Judge-noise ceiling (~$5)

**Technique:** label-noise estimation, inter-annotator agreement.
**Design:** re-judge ~300 already judged tasks (eval dev and pool) in a fresh run, with
another seed of the candidate order. Agreement of top-1 and grades, Cohen's κ; with the
author's own labels when they exist.
**Measure:** the silver ceiling (how much of the gap to 1.0 is judge noise), which tells
how far silver differences of 0.005 can be trusted.

## Totals

| # | experiment | ~$ (promotional) |
|---|---|---:|
| 1 | active-learning strategy study | 30–50 |
| 2 | adversarial tasks | 15–25 |
| 3 | minimal pairs | 10–20 |
| 4 | pairwise preferences | 10–20 |
| 5 | VLM reranker teacher | 20–40 |
| 6 | judge-noise ceiling | 5 |
| | all six | ~90–160 |

With ~$50: #1. With ~$100: #1 and #2 or #3. #6 is cheap enough to add to any batch.
After 2026-11-30 the same list costs about twice as much.

**Not on the list:** labels for the abstention gate (it can be calibrated on the labels
that exist). And a reminder that every item optimises toward Argon's taste: the author's
own labels (labeller, Personal) are the check that Argon's taste is the author's, and
100–200 of them are worth more confidence than any of the above.

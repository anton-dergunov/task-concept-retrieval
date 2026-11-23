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
**all3k (2026-11-22):** +3k judged tasks gave +0.008 (title), exactly the curve's rate, so
random labels still pay ~+0.005 per $50. #1 drops to the bottom of round 6's ranking
(below).

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
After 2026-11-30 the same list costs about twice as much. (Superseded by round 6's choice
below.)

## Round 6's choice (2026-11-22, after round 5)

**How it is ranked (author, 2026-11-22, CLAUDE.md):** first what an item teaches about ML,
or whether it makes a blog post (~60%), then what it gains for the **big model**, the one
that runs on the Mac (~40%; deployment.md "Two models"). Scores 1–5, judged here.

| # | experiment | teaches / post (60%) | big-model gain (40%) | score | ~$ |
|---|---|---:|---:|---:|---:|
| 5 | VLM reranker as a teacher | 5: listwise multimodal reranking, cross-modal distillation | 3–4: the one teacher that can be better than e5-large, and the image signal reaches the big model through distillation | **4.4** | 20–40 |
| 3 | minimal pairs | 5: contrast sets, counterfactual augmentation, a pairwise loss term, a contrast-set metric | 3: aims at what the model reads wrongly in a title | **4.2** | 10–20 |
| 2 | adversarial tasks | 4: model-in-the-loop data collection, fooling rate | 3: hard examples of the big model's own confusions | **3.6** | 15–25 |
| 4 | pairwise preferences | 4: preference labels, margin loss, how reliable absolute grades are | 2 | 3.2 | 10–20 |
| 6 | judge-noise ceiling | 3: inter-annotator agreement, κ, the silver ceiling | 1: no gain, but says which Δs of 0.005 to trust | 2.2 | 5 |
| 1 | active-learning arms | 2: round 5 has already taught its main lesson | 1: active ≤ random (f1-dual-encoder.md §13) | 1.6 | 30–50 |
| — | more random tasks | 1 | 2: ~+0.005 per $50 on the curve (round 5's all3k landed on it) | 1.4 | any |

**Recommendation: #5 + #3 + #2, with #6 added** (~$50–90 in all, within the ~$100 the
author allows). #4 follows if money is left. #1 and plain random tasks are dropped. Round 5
showed that random labels still pay at the curve's rate, but the rate is small (+0.005 per
$50) and teaches nothing new. The four chosen items fit in **one v3 sitting**, and every
Argon call must finish by 2026-11-30 (the price doubles after that):

| when | where | step |
|---|---|---|
| 11-23/24 | Mac | code: `augment_pools.py minpair` (#3) and `adversarial` (#2) with Flash; the VLM-rerank request (#5: task + the big model's top 20 as glyph images, no names; reuse `tcr/describe_v2.py`'s image render); re-judge set (#6); a 30–50-task pilot of each, read by hand |
| 11-25 | Vertex (Flash, ~$5) | generate ~1,000 minimal pairs and ~1,000 adversarial tasks; contamination and near-duplicate filters |
| 11-26 | Vertex (Argon, ~$45–80) | one batch: judge minimal pairs + adversarial tasks (+ translations, rewrites), VLM rerank of ~2,000 tasks, re-judge 300 |
| 11-27/29 | Mac | buffer for failed or retried requests; import; `f1_data.py build` |
| from 11-28 | Kaggle (free) | round 6: the big model with each new kind of data, the VLM-distilled big model, and the small model last |

**Approved (author, 2026-11-22):** all of it, up to **$100**, on the contact account (the
Vertex project showed £123.57 of credit left). Round 6 has its own cap, so round 3's stays
as it was: `CAP_TAG=round6 BUDGET=100` fixes `data/private/augment/cap-round6.json` at the
spend log + $100 ($487.82) on its first run.

**Batch 1 = `vertex_round3_jobs.sh v3a`** (~$30): #6 re-judges 300 eval dev tasks
(Personal 150, Personal-synth 75, Realistic 75; `noise_ceiling.py select`), same judge and
prompt, the ~100 candidates in another order (`judge_batch.py prepare --ids …
--shuffle-seed 1`, the whole set kept as context for sibling titles); `noise_ceiling.py
report` → results/noise_ceiling.md. #3: Flash edits 1,000 judged tasks with a clear top
icon and ≥ 3 words (pool-synth-j4 600, pool-synth2-j4 150, pool-mslatte-j4 250; 20% from
validation tasks, the contrast set, never trained on) by 1–3 words so the best icon should
change (`augment_pools.py minpair` → `<dataset>.mp.jsonl`). Argon then judges the edits with
the original's top 5 among the candidates (`minpair-pool`, runs mp-synth-j4 and
mp-mslatte-j4). Batch 2 (v3b: #5, #2) follows.

### Batch 1 results (run 2026-11-22, $23.89)

The spend log went from $387.82 to $411.71, so **$76.11** of round 6's cap ($487.82) is left
for batch 2. Every request succeeded.

**#6 judge-noise ceiling** ([results/noise_ceiling.md](../results/noise_ceiling.md), 300
dev tasks):
- **Argon's top-1 is reliable, its grades are not.** The two runs name the same top-1 on 73%
  of tasks, but each run's top-1 is a hit under the other's grades on 96%. Grade agreement
  is low: quadratic κ 0.35. No-good-icon agrees on every task but one.
- **The ceiling is far away.** The re-judge, scored as a method against the original gold,
  reaches nDCG@5 0.888 and hit@1 0.96. The best model (F1@all-large-all3k+parent) reaches
  0.502 / 0.63 on the same tasks. Judge noise explains little of the gap to 1.0; the model
  explains most of it.
- **Which Δs to trust.** All 75 stored methods were rescored against the re-judge's gold.
  The method order hardly moves (Spearman ρ 0.994). Judge noise is a median 11% of the
  per-task variance of a paired ΔnDCG@5. A bootstrap over tasks already includes it.
  - On the pooled dev set (3,230 tasks), judge noise alone gives a paired Δ an SD of
    ~0.0015. So a pooled Δ of 0.005 is ~3 SD of judge noise.
  - The limit is which tasks are in the set, not the judge. The bootstrap intervals in
    the silver tables stay the test.
  - Per-set Δs on 75–150 tasks below ~0.01 can flip sign between runs of the judge.
- **A hint of winner's curse.** The leaders lose a little under the re-judge, and the
  baselines do not. F1@all-large-all3k+parent: −0.010, 95% [−0.027, +0.008]; M3.v2 −0.002.
  The gap between the big and the small model shrinks from +0.040 to +0.029, still clearly
  positive. Neither change is significant. They fit leaders chosen on this gold.
- **Caveat:** this measures the noise of the same judge (candidate order, sampling), not
  Argon's systematic taste. The author's own labels are still the only check of that.

**#3 minimal pairs:**
- **Pairs judged:** 988 (Personal-synth 747, MS-LaTTE 241, after the contamination filter).
- **How many edits changed the icon:** an edit *flips* when the original's top icon is no
  longer a hit on it. 85% of the MS-LaTTE edits flip, but only 50% of the Personal-synth
  edits. Flash often edited a Personal-style title without changing what it is about; the
  rest are kept as invariance pairs.
- **The contrast set:** 198 pairs of validation tasks (118 flips, 80 keeps), scored by
  `scripts/contrast_eval.py`
  ([results/contrast/2026-11-22.md](../results/contrast/2026-11-22.md)). These are the
  "before" numbers.
  - **The models do follow edits.** On flips the F1 models change their top-1 90–95% of
    the time. They keep a wrong icon ("stuck") 0–8% of the time; M3 does 8–12%.
  - **Edits are not harder than originals.** hit@1 on the edit is mostly above the
    original's: F1@all-large-all3k+parent 0.81 vs 0.67. Flash's edits name a concrete object
    ("flight", "office keys"), so there is no contrast-set drop in Gardner et al.'s sense.
  - **Pair accuracy** (both sides a hit): F1 0.53–0.59, M3.v2 0.39. F1@all-kd-final (0.59)
    is level with the big models (0.54–0.55): ±0.09 on 118 flips, so not a difference.
  - **On keeps**, the models name the same top-1 on 31–44% of pairs, and the judge itself
    on 39%. They are no more jumpy than the judge.
- **The training side:** the 790 pairs of training tasks are `mp-synth` / `mp-mslatte` rows
  in the next `f1_data.py build` (tcr/finetune.py `MP_RUNS`). Each row carries its
  `orig_task_id`. The original's top icons that the judge left ungraded are hard
  negatives.
  - train_f1.py uses them with `--mp rows` (plain graded rows) or `--mp pair
    --pair-weight w`. The second also adds a pairwise term on the 492 flips: each side's
    best icon must score above the other side's. The arms are preset in the `round6b` plan
    (F1@all-lr-para-all3k-mp, -mpp), with round 5's F1@all-lr-para-all3k as the control.
  - Since the models already follow edits, the question for round 6b is whether the pairs
    raise pair accuracy and silver, not whether they make the model notice the edit.

**Add to batch 2:** grade the unjudged icons in the leaders' top 10 on dev (`g5`,
`grade_unjudged.py --depth 10`, ~8,000 icons, ~$10–25). A reranker can only be scored
fairly with them ([reranker-plan.md](reranker-plan.md) §4).

### Batch 2 (v3b): design and pilots (2026-11-23)

**Budget (author, 2026-11-23):** ~$130 of credit left, **$100 the safe limit** for batch 2;
cut nothing if possible, the adversarial count first if needed. Its own cap: `CAP_TAG=round6b`,
spend log $411.71 + $100 = **$511.71**. The run: vertex-round3-run.md (`v3b`).

| item | n | ~$ (measured rates) |
|---|---:|---:|
| g5: the two leaders' unjudged top-10 icons on dev (F1@all-large-all3k, F1@all-kd-final, +parent) | 12,792 icons on 2,843 tasks | 14 |
| #5 VLM rerank, dev (the measurement) | 3,230 | 19 |
| #5 VLM rerank, judged training tasks (non-val, random) | 1,000 | 6 |
| #5 VLM rerank, unjudged candidates (75% Personal-style, 25% MS-LaTTE, random) | 3,000 | 18 |
| #2 Flash generation (4 tasks per seed) | 248 seeds | 0.3 |
| #2 Argon judging, ES/RU translations, rewrites | ~910 tasks | 20 |
| g6: the VLM's unjudged top 3 on dev (after the VLM's dev run) | ~1,000 tasks | 5 |
| **total** (`vlm_rerank.py budget`) | | **~83** |

**#5 VLM reranker** (`tcr/vlm.py`, `scripts/vlm_rerank.py`). One request per task: the
judge's system prompt and few-shot block, the task as the judge saw it (parents,
siblings), and the big model's served top 20 on one numbered contact sheet. The order is
hash-shuffled, so the VLM cannot copy the first stage's, and names are never shown. It
returns the central subject, all 20 numbers best first with a grade 0–3 on the judge's
scale, reasons for the top 3, and no_good_icon. Numbers it leaves out go to the end (more
than 3 missing: retried once).
- **Which tasks, and why** (the author asked for the best final quality and the most ML
  learned):
  - Dev measures the reranker and its agreement with the judge on 3,230 tasks.
  - The 3,000 unjudged candidates are realistic first-stage lists (the model never trained
    on them), the lists a reranker sees in use. They are new labels at a third of judging's
    price, which asks whether a listwise VLM reranker works as a cheap labeller.
  - The 1,000 judged training tasks give round 6 the clean A/B: the same texts trained
    with the judge's grades vs the VLM's.
- **Pilot** (online, 40 tasks: 30 dev over the five sets, 10 training; two arms; gold +
  g1–g4):

  | 30 dev tasks | nDCG@5 | hit@1 | ΔnDCG@5 vs first stage [95%] | numbers left out |
  |---|---:|---:|---|---:|
  | first stage (F1@all-large-all3k+parent) | 0.578 | 0.77 | | |
  | VLM, images + one-line descriptions | **0.742** | **0.93** | +0.164 [+0.095, +0.238] | 0 |
  | VLM, images only | 0.692 | 0.87 | +0.113 [+0.034, +0.200] | 17 |

  On the 10 training tasks: 0.593 → 0.778 (img+desc). The descriptions help, and without
  them the VLM drops more numbers, so **the full run uses img+desc**.
  - 9 of the 40 VLM top-1s come from the first stage's ranks 11–20. The VLM finds what the
    first stage buried: "Data residency, sovereign deployment…" gets the globe-lock at rank
    14, the judge's #1; "Annotate" gets the highlighter at rank 18; "Pt session" gets
    exercise, not bones. Those icons are often outside every grade set, hence g6.
  - It grades more generously than the judge: many 3s where the judge gave 2. This matters
    when its grades become training labels.
  - Caveat: the judge and the reranker are the same model (Argon), so part of the gain is
    agreement with its own taste. The noise ceiling (re-judge 0.888) bounds that, and the
    author's labels stay the check.
  - Cost: ~4.0k input and ~450 output tokens, **$0.0063 per request in batch**.
- **After the run:** `vlm_rerank.py report` → results/vlm/<date>.md (VLM vs first stage per
  set with paired bootstrap, top-1 agreement with the judge, how often the top-1 comes from
  ranks 11–20, position bias, rank correlation with the first stage). The dev rankings are
  a stored silver method (`VLM@all-large-all3k.v2+parent`), scored with
  `silver_eval.py --extra-grades g1,…,g6`. The cand/train labels are in
  `data/private/vlm/labels/`, for round 6's training code (not written yet: VLM-labelled
  rows, distillation from the VLM's order).

**#2 adversarial tasks** (`augment_pools.py adversarial|adversarial-pool|adversarial-report`).
- **Seeds:** F1's validation tasks of the three round-2 pools (983; the big model never
  trained on them), where gold has a good icon and the big model's top-1 is not it.
  - Only **139 are judged misses** (the judge saw the top-1 and graded it < 2). The big
    model misses only 13–20% of its judged validation top-1s (synth 12.7%, synth2 16.4%,
    MS-LaTTE 20.3%); that is the base rate the fooling rate must beat.
  - Another **109 have a top-1 the judge never saw**, kept as a second kind (in g1, ~70% of
    such picks were misses). The report splits the two kinds: do verified errors make
    better seeds than probable ones?
- **Generation:** per seed, Flash gets the task and the right and wrong icons in words. It
  names the confusion as a general pattern, then writes 4 new tasks with the same trap on
  other subjects, in the seed's style, each with `right_icon` and `lure` (248 × 4 ≈ 990).
- **Filters:** eval titles (contamination guard); near-duplicates (e5 cosine > 0.95) of any
  judged pool task or of another generated task.
- **Judging:** the big model's top 10 and the seed's judged icons join the judge's ~100
  candidates, so the fooling rate needs no extra grading (runs adv-synth-j4, adv-mslatte-j4).
- **Pilot** (12 seeds online → 48 tasks, 45 kept after 3 near-duplicates; Argon judged them
  online):
  - **Fooling rate 14/45 = 31%**, against the big model's 13–20% miss rate on the validation
    tasks the seeds came from. By seed kind: 5/18 from judged misses, 9/27 from unjudged
    top-1s, too few to tell apart. No task got no_good_icon, so the generated text is
    realistic enough to have a right icon.
  - Flash names real, general confusions: "matched the common consumer meaning of a
    domain-specific ML term" (temperature scaling → thermometer, attention → eye), "the
    action verb instead of the listing object" (Remove eBay listings), "the parent category
    instead of the document" (Review syllabus → library). The tasks keep the seed's style
    (MS-LaTTE seeds get terse two-word titles).
  - Some "fools" are near-misses (a battery glyph next to the judge's battery glyphs), so
    the fooling rate is an upper bound on real errors.
  - The model's top-1 is never the seed's own wrong icon: the trap transfers as a pattern,
    not as a glyph.
- **Pilot spend in all:** $2.58 (VLM $1.0, adversarial $1.6).
- **Examples:** `data/private/vlm/pilot.html` (private: Personal tasks; `vlm_rerank.py
  pilot-report`).

**Not on the list:** labels for the abstention gate (it can be calibrated on the labels
that exist). And a reminder that every item optimises toward Argon's taste: the author's
own labels (labeller, Personal) are the check that Argon's taste is the author's, and
100–200 of them are worth more confidence than any of the above.

### Batch 2 results (run 2026-11-23, $83.60)

The spend log went from $411.71 to $495.31: **$83.60**, at the estimate. Round 6 in all:
$107.49. The author's console showed £41.36 (~$54) of credit left on 2026-11-23 (it lags by
about a day). Every request succeeded. Reports: [results/vlm/2026-11-23.md](../results/vlm/2026-11-23.md),
[results/adversarial/2026-11-23.md](../results/adversarial/2026-11-23.md).

**g5, g6:** 12,792 unjudged icons in the two leaders' top 10 on dev, and the VLM's unjudged
top 3, are graded. Scores below use gold + g1–g6.

**#5 VLM reranker, dev (3,230 tasks, img+desc):**

| set | first stage nDCG@5 | VLM nDCG@5 | Δ [95%] | first hit@1 | VLM hit@1 |
|---|---:|---:|---|---:|---:|
| Personal | 0.577 | 0.798 | +0.220 [+0.209, +0.233] | 0.677 | 0.942 |
| Personal-synth | 0.626 | 0.822 | +0.196 [+0.180, +0.212] | 0.716 | 0.946 |
| Public-expanded | 0.616 | 0.806 | +0.191 [+0.163, +0.218] | 0.714 | 0.946 |
| Public-short | 0.713 | 0.819 | +0.106 [+0.088, +0.124] | 0.841 | 0.925 |
| Realistic | 0.595 | 0.805 | +0.210 [+0.185, +0.235] | 0.698 | 0.927 |
| pooled | 0.612 | 0.808 | **+0.196** [+0.188, +0.204] | 0.714 | **0.939** |

(First stage: F1@all-large-all3k+parent, the deployed big model. Against round 6a's
F1@all-large-para-all3k the VLM leads by +0.188 [+0.180, +0.196], hit@1 723 / 73.)

- **The reranker almost reaches the judge's own ceiling.** hit@1 0.939, where the re-judge
  of batch 1 reached 0.96 against the first run's gold. It gains most where the first
  stage is weakest (Personal +0.22) and least on terse public titles (+0.11).
- **It does not copy the first stage.** Rank correlation with the first stage's order is
  0.40, with the shown position 0.03 (no position bias). Its top-1 is the first stage's
  top-1 on 27% of tasks, and comes from ranks 11–20 on 20%: the top 20 holds the answer
  much more often than the top 1 does.
- **It is not a copy of the judge either.** Its top-1 equals the judge's top-1 on 52% of
  tasks (the re-judge: 73%) and is a hit on 94%: it picks another good icon.
- **Same-model caveat.** Judge, reranker and g6's grader read the same descriptions, and
  the first two are the same model. Part of the gain is Argon agreeing with itself. The
  author's labels are the check.
- **On the 1,000 judged training tasks** the gain is +0.059 (Personal-style +0.042,
  MS-LaTTE +0.101). The first stage trained on those labels, so its own numbers there are
  inflated; the dev figures are the measurement.
- **How its grades compare with the judge's** (the 20 shown icons of the 1,000 training
  tasks; rows the VLM's grade, columns the judge's, "unseen" = outside the judge's
  candidates):

  | VLM \ judge | 0 (seen) | 1 | 2 | 3 | unseen |
  |---|---:|---:|---:|---:|---:|
  | 0 | 1,905 | 29 | 45 | 17 | 1,660 |
  | 1 | 3,772 | 453 | 653 | 72 | 2,279 |
  | 2 | 1,707 | 284 | 2,196 | 683 | 1,442 |
  | 3 | 209 | 9 | 422 | 1,806 | 357 |

  It grades generously: 9.1 icons per task at ≥ 2 against the judge's 7.1, and 7.2 at 1.
  Its 3s are the judge's 2s and 3s (91% of the known ones); its 1s are mostly the judge's
  0s (76%). So as training labels its grades are **calibrated**: 3 and 2 stay, 1 is masked,
  0 is a hard negative (train_f1.py `vlm_grade_row`).
- **Price:** $0.0063 per task in batch, a third of a full Argon judgment ($0.019).

**#2 adversarial tasks (971 judged: 679 Personal-style, 292 MS-LaTTE):**

| set | tasks | fooled [95%] | base rate (validation tasks of the seeds' pool) |
|---|---:|---|---|
| Personal-style | 679 | **0.274** [0.242, 0.309] | 0.127 (pool-synth), 0.164 (pool-synth2) |
| MS-LaTTE | 292 | **0.277** [0.229, 0.331] | 0.203 |

- **The trap transfers, roughly doubling the error rate on Personal-style text** (×1.4 on
  MS-LaTTE). Fooled = the big model's top-1 is no hit under the judge, who saw the model's
  top 10 (judged@1 0.99), so no extra grading is needed.
- **Verified errors make no better seeds than probable ones:** 0.288 vs 0.262 (Personal-style),
  0.271 vs 0.296 (MS-LaTTE); the intervals overlap.
- **The model rarely falls for the seed's own wrong icon** (top-1 = the seed's wrong icon on
  3–5%): the confusion carries over as a pattern, not as a glyph.
- No-good-icon on 1 task of 971: the generated text is realistic enough to have an answer.

**What it means for the rest of the money.** The VLM labels cost a third of judging, and
they are the strongest labels the project has: they reach icons the judge's candidate pool
missed. Whether they also **train** well is round 6b's first question (f1-kaggle-run.md
"Round 6b"): the same 1,000 tasks with the judge's grades vs the VLM's, 1,000 / 3,000
VLM-labelled candidates alone, and the 3,000 on top of the pool. If they train as well as
the judge's grades, a last Argon batch (by 2026-11-29) reranks more unlabelled candidates:
~10.7k remain (9.0k Personal-style, 4.7k MS-LaTTE, minus the 3,000), $19 per 3,000.

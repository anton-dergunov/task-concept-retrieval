# F1: the fine-tuned dual encoder

Design of method **F1** ([next-label-free-work.md](next-label-free-work.md) step 2,
[fine-tuning-plan.md](fine-tuning-plan.md) §5): `intfloat/multilingual-e5-small`
fine-tuned on the LLM judge's labels of the training pools and on the free signal in the
v2 icon descriptions. This file explains every choice and why it was made; the run guide
is [f1-kaggle-run.md](f1-kaggle-run.md) (round 1: [f1-colab-run.md](f1-colab-run.md)), the
results and rounds 2–3 are at the end; what comes next is in
[next-data-and-distillation.md](next-data-and-distillation.md).

**Names.** *F1* is a method name, like B0, M1, M3 ("fine-tuned, first version"); it has
nothing to do with the F1 score. A training run is named `F1@<pool tasks>`: `F1@0`
trains on the free signal only, `F1@1k` and `F1@3k` on that many judged pool tasks,
`F1@all` on all of them. Those four hold out 5% of the icons (below); `F1`, the method
itself, is trained on everything.

## 1. What is trained

A dual encoder: one network turns a text into a 384-dimensional unit vector, and the
score of (task, icon) is the cosine of the task's vector and the icon's vector.

- **Task side:** `query: ` + the normalized title, sometimes with its nearest parent
  heading (`query: Family: Call nagi`), built by `tcr.org_tasks.query_text`, exactly as
  the other methods build their queries.
- **Icon side:** `passage: ` + a text made from the icon's v2 description (§3).
- **One network for both sides** (e5 is trained that way; the prefixes tell it which
  side it is encoding). Icon vectors are computed once; at query time only the task is
  encoded, so F1 costs the same as M1 on the agenda's render path.

The base model already scores pairs (that is M1's encoder). Fine-tuning moves the vectors
so that a task lands next to the icons the judge picked for tasks like it.

## 2. Training data

| data | judge run | tasks | role |
|---|---|---|---|
| Personal-style pool, first generation | `pool-synth-j4` | 4,986 | 90% train, 10% validation |
| Personal-style pool, step 0c extension | `pool-synth2-j4` | 1,890 | 90% train, 10% validation |
| MS-LaTTE pool | `pool-mslatte-j4` | 3,000 | 90% train, 10% validation |
| ES / RU translations of the pool tasks | (step 0a) | ×2 | same split as their task |
| v2 free signal | — | ~75k queries | train only |
| the five **eval sets** (Personal, Personal-synth, Realistic, Public-short, Public-expanded) | `*-j4` | — | **never** trained on; their dev halves are the silver leaderboard |

**Judge labels of the pools.** For each pool task the judge looked at its ~100-icon
candidate pool plus its own searches (~113 icons on average), marked ~22 as relevant in
its first stage and graded ~9 in its final ranking: grade 3 (the icon *is* the task's
idea), 2 (good), 1 (acceptable). The task text is read from each judge run's snapshot
(`tcr.silver.run_tasks`), never from the pool files, which were rewritten after judging.
Icon names go through `tcr.silver.canonical` (aliases count once), and glyphs rejected
in the manual icon review are dropped (`tcr.silver.load_gold`). F1 uses the v2 icon set
(2,862 glyphs after discards and aliases); a grade on a glyph outside it (one that only
v1 describes) is dropped.

**Translations.** Every pool task is English; Public-short is 18% Spanish or Russian.
Step 0a translated title and parent headings. A translation is skipped when step 0a
flagged that language as too close to an eval title (`contaminated`), and when it equals
the English title (brand names: "Path of exile"), since it would only duplicate the
English row. A translated task keeps the English task's grades: the icon depends on the
idea, not the language.

**Parents.** The Personal-style pool has generated parent headings (step 0b), MS-LaTTE
tasks have their list name where it is informative (27%), translations have translated
parents. Each time a task with a parent is drawn, the parent is used with probability
0.5, so the model learns both the bare title and the "Parent: title" form; the eval sets
contain both, and on silver the parent did not help M3, so neither form is favoured.

**The v2 free signal**: texts that are already in the v2 descriptions, written by Gemini
in the v2 passes, so they cost nothing now. Counted by summing list lengths over the
2,862 icons:

- *example tasks*: ~16 English (some with a parent: "Client Scheduling: Send
  availability link") + 3 Spanish + 3 Russian per icon = 63,630. A query whose positive
  is that icon, grade 3.
- *context tasks*: ~4 per icon = 11,401. Tasks where the icon fits only as context
  ("Move Tuesday's 1:1 with Maya" for the calendar: the central object is the meeting),
  grade 2.
- *near-misses*: ~6 per icon = 17,605. Tasks that look as if they belong to the icon but
  do not (a look-alike glyph, a shared word: "Buy fresh Medjool dates" for the calendar).
  Used as hard negatives (§4).

The free signal outnumbers the judged pool rows about 3 to 1, and every free-signal
query knows only one positive. So each epoch draws a fresh sample of 25,000 free-signal
queries rather than all of them; the judge labels keep the larger weight, and the
learning curve (§6) varies only the judge data.

**What never goes in.** No eval-set task and no eval-set judge label. The data build
checks that no eval task id is in a pool, and drops any training text equal to an eval
title after normalization (`tcr.describe_v2.contamination_hits`, exact). The near-duplicate
check by embedding ran upstream: the v2 guard for the free signal, step 0a's
`contaminated` flags for the translations.

## 3. The icon document

```
passage: Shows: <primary object; modifiers>. Means: <readings>. For: <task intents>.
<distinguishing> <reasoning>
```

- **Shows** (the depiction) lets the model connect "router" or "envelope" in a title to
  the drawn object; **Means** (readings, strongest first) and **For** (task intents)
  carry the ideas; **distinguishing** (from the contrast pass, on 951 glyphs) says how a
  glyph differs from its look-alikes, which is what separates the calendar variants;
  **reasoning** last, so truncation at 256 tokens cuts it first (median length 216
  tokens, 90th percentile 268).
- **Left out:** example tasks, context tasks and near-misses. They are query-side text
  and they are the training queries: an icon document containing its own example tasks
  would let the model match a training query to a copy of itself, which teaches nothing
  that carries over to new tasks. Icon names, tags and categories are never used (hard
  rule 1).
- The field labels ("Shows:", "Means:") are plain words, so the encoder can tell the
  drawn object from the meaning.

## 4. The loss: masked, graded, multi-positive InfoNCE

**A batch** holds 128 queries. For each query one positive icon is drawn, with
probability proportional to grade² among its graded icons (so a grade-3 icon is drawn
2.25 times as often as a grade-2 one, 9 times a grade-1 one), and one hard negative. The
batch's icon passages are the union of these, deduplicated: up to 256 icons, each encoded
once.

**Scores.** s(i, j) = 20 · cos(query i, icon j) for every query i and every icon j in
the batch (scale 20 = temperature 0.05, the sentence-transformers default).

**Target for query i:** a distribution over the batch's icons, proportional to the
judge's grade for every icon graded for i that happens to be in the batch, not just the
one drawn for it. **Loss:** cross-entropy between that target and softmax(s(i, ·)).

**Masks.** Some in-batch icons must count neither as positive nor as negative for a
query; they are removed from its softmax:

- icons the judge marked relevant in its first stage but did not grade: it considered
  them and set them aside, which is not the same as "wrong" (they include second-best
  readings and near look-alikes);
- for a free-signal query, the look-alikes of its icon (`_meta.lookalikes` in the v2
  files): "Buy 2026 wall calendar" is an example of one calendar glyph but fits all of
  them, and nobody graded the others.

Everything else in the batch is a negative: other queries' positives (in-batch
negatives) and the hard negatives.

**Hard negatives** for a pool task are icons the judge saw and did not mark relevant:
seen − relevant − graded. They come from the candidate pool, i.e. they are what B1, M1
and M3 rank highly for the task, so they are exactly the mistakes the current methods
make.

**Worked example** (scale omitted). Queries: q1 "Call grandma" (graded: `call` 3,
`phone_in_talk` 3, `elderly_woman` 2; relevant but ungraded: `wifi_calling`), q2
"Renew passport" (graded: `badge` 3), q3 "Buy 2026 wall calendar" (free signal,
`calendar_month` 3; look-alikes: `calendar_today`). Drawn: q1 → `call` + hard negative
`agriculture`; q2 → `badge` + hard negative `flight`; q3 → `calendar_month`. Batch icons:
`call`, `agriculture`, `badge`, `flight`, `calendar_month`. If `phone_in_talk` had been
drawn by another query it would also be in q1's target, at weight 3/(3+3).

| | call | agriculture | badge | flight | calendar_month |
|---|---|---|---|---|---|
| q1 target | 1 | 0 | 0 | 0 | 0 |
| q2 target | 0 | 0 | 1 | 0 | 0 |
| q3 target | 0 | 0 | 0 | 0 | 1 |

Had `wifi_calling` or `calendar_today` been in the batch, it would be masked for q1 or
q3 respectively.

**The icon→query direction** (weight 0.3). Each icon that is some query's positive also
gets a row: its scores against the batch's queries plus one of its own near-misses, with
the target spread over the queries that graded it. This is the only place where the
near-misses enter: "Buy fresh Medjool dates" is a negative for the calendar it was
written for, but it may well be a positive for a fruit icon, so it competes only in its
own icon's row.

**Why not sentence-transformers' MultipleNegativesRankingLoss**, the usual start: it
takes one positive per query and treats every other in-batch passage as a negative.
Here a query has ~9 graded icons; with only 2,862 icons and popular ones (phone, calendar,
document) the positive of one query is often a positive of another in the same batch,
and MNRL would push the model away from correct icons. Graded targets also match how the
leaderboard scores (nDCG with the grades as gains).

**Other settings.** AdamW, learning rate 2e-5, 5% warmup then linear decay, fp16 on a
GPU, query length ≤ 64 tokens, icon text ≤ 256. The word-embedding matrix (96M of the
model's 118M parameters, one row per token of a 250k multilingual vocabulary) is frozen:
training sees mostly English, and updating only the rows of tokens it sees would drift
those away from the untouched Spanish and Russian ones; it also cuts optimizer memory by
80%. The 12 transformer layers (21M parameters) are trained.

## 5. Validation

- **Held-out tasks:** 10% of the pool tasks by a hash of the task id, with all their
  languages and parent forms. Every half epoch: nDCG@5 and hit@1 against the judge's
  grades over all 2,862 icons, reported for English and for ES/RU apart. Training stops
  when pooled nDCG@5 has not improved for three evaluations (1.5 epochs), at most six
  epochs, and the best evaluation's weights are kept. The base model is evaluated at
  step 0 as the reference.
- **Held-out icons:** 143 icons (5%, by a hash of the name) never appear in training,
  neither as a positive, a negative nor a free-signal target. *Unseen-icon recall@10*:
  over held-out tasks whose grade-3 icons include a held-out one, how often that icon is
  in the top 10, against the base model. Fine-tuning learns a mapping from task
  language to description language; this checks that the mapping carries over to icons
  it never saw, i.e. that F1 still works for icons added later or seldom used. The method
  `F1` itself is trained with all icons.
- **The silver leaderboard** (step 1) is the comparison with the other methods. Its tasks
  are the eval sets' dev halves, which training never sees.

## 6. The learning curve

`F1@0`, `F1@1k`, `F1@3k`, `F1@all` differ only in how many judged pool tasks they train
on (drawn by hash, proportionally from the two pools); the free signal and every setting
are the same. Plotted as silver nDCG@5 against pool tasks, it says (a) what the judge
labels add over the free signal (`F1@0` vs the rest), and (b) whether more judged tasks
would still help: a curve still rising at `all` means another judged pool would pay
off; a flat one means the money goes elsewhere.

## 7. Comparing F1 with M3.v2 fairly

The judge graded only icons in its candidate pool, which B1, M1, M2 and M3 built. F1 is
not a pool builder: when its top-1 is an icon the judge never saw, silver counts a miss
whether the icon is good or not. B0 shows the size of the problem: 31% of its top-1s were
unjudged. So F1's hit@1 is reported as an interval:

- **lower bound:** unjudged top-1s counted as misses (the usual hit@1);
- **upper bound:** unjudged top-1s counted as hits (`hit1_ub`).

Decision: lower bound above M3.v2 with McNemar p < 0.05 → F1 wins; upper bound below
M3.v2 → F1 loses; in between, the unjudged top-1s are graded. Before that, count them and
estimate the cost. The grading is an add-on, not a rerun of the judge: the task, the
judge's own graded icons as anchors (glyph and one-line description) and the new icon,
graded 0–3 on the same rubric, with ~50 already-graded icons mixed in blind to measure
agreement with the judge. Public tasks go to the free `gemini` tier; Personal tasks to
Claude (an Opus subagent), never to the free tier.

nDCG@5 has the same bias. The judged@1 column shows how large it is.

## 8. Alternatives set aside (later experiments)

- **Frozen icon tower** (train only the query side against fixed description vectors):
  every step becomes cheap and unseen icons work by construction, but the icon side
  cannot adapt; a good fallback if the shared model overfits.
- **Learned per-icon vectors** initialised from the description embedding (§5 "hybrid"):
  a classification head that still starts from the descriptions; held-out icons would
  keep their initial vectors. Natural next step if F1 wins.
- **A memory bank of icon vectors** (all 2,862 in every softmax, refreshed every few
  hundred steps): a full softmax instead of in-batch negatives. Worth it only if
  in-batch negatives turn out too easy.
- **Bigger encoders** (e5-base/large, bge-m3, Qwen3-Embedding-0.6B) as teachers to distil
  into e5-small: step 5 of next-label-free-work.md.

## 9. Where it runs

A probe on the M1 Mac (MPS, 16 GB shared memory) did not finish a single timing step of
e5-small training in 10 minutes and made the laptop unusable, most likely from memory
pressure. So the Mac only builds the data (`scripts/f1_data.py`), runs a tiny CPU smoke
test, and scores checkpoints on silver (which needs the private gold labels). Every real
run is on Colab: `scripts/train_f1.py` needs only torch and transformers, is shipped
inside the data tarball, picks the GPU itself and resumes after a lost session. The
private Personal-style pool goes into the tarball, approved by the author (2026-11-14);
no eval data does.

## Code

| file | what |
|---|---|
| `tcr/finetune.py` | icon document, training rows, splits, `FineTunedMatcher` |
| `scripts/f1_data.py` | builds `data/private/f1/f1_data.tar.gz` |
| `scripts/train_f1.py` | training (standalone; runs on Kaggle or Colab) |
| `scripts/f1_report.py` | learning-curve plot and tables from `results/f1/` |
| `scripts/f1_review.py` | private HTML page: F1's picks on 50 held-out pool tasks, next to the judge's and M3.v2's |
| `tcr/methods/__init__.py` | `F1` → `.cache/models/F1/`, `F1@<tag>` → `.cache/models/F1-<tag>/` |
| `scripts/silver_eval.py` | scores F1; merges with stored rankings of earlier runs |

## Results

Round 1 ran on Kaggle (one T4; Colab had ended the job after 3 hours), 2026-11-15. Silver
leaderboard: [results/silver/2026-11-15.md](../results/silver/2026-11-15.md); training logs
and the learning curve: [results/f1/](../results/f1/) (`scripts/f1_report.py` →
`learning_curve.png`, `learning_curve.md`).

**F1 wins on silver** (decision rule of §7). Pooled over the 3,230 dev tasks:

| method | nDCG@5 | hit@1 [lower, upper] | judged@1 | Δ nDCG@5 vs M3.v2 | hit@1 wins / losses |
|---|---:|---:|---:|---:|---:|
| F1@all.v2+parent | 0.436 | 0.557 [0.557, 0.683] | 0.875 | +0.100 [+0.090, +0.109] | 769 / 380, p < 0.001 |
| F1.v2+parent | 0.433 | 0.548 [0.548, 0.674] | 0.874 | +0.097 | 754 / 395, p < 0.001 |
| F1@all.v2 | 0.428 | 0.544 | 0.864 | +0.092 | 726 / 381 |
| M3.v2 | 0.337 | 0.437 | 0.998 | ref | |

- The lower bound alone beats M3.v2 (p < 0.001), so the unjudged top-1s need no grading
  for this decision. F1 leads on every eval set; the gain is largest on Personal (+0.13
  nDCG@5 with the parent) and smallest on Public-expanded (+0.04).
- 12.5% of F1's top-1s were never shown to the judge, so its absolute level lies between
  the bounds. Telling F1 variants apart will need these graded (§7), since they all share
  the bias.
- **F1 ≈ F1@all**: holding out 5% of the icons costs nothing measurable. Unseen-icon
  recall@10 rises from 0.27 (base) to 0.59, so F1 still works for icons it never trained on.
- **The parent heading** helps F1 on Personal (+0.024 nDCG@5) and hurts on Realistic
  (−0.03); pooled +0.008. For M3 it did nothing.

**Learning curve (§6):**

| pool tasks | 0 | 1k | 3k | 8.9k |
|---|---:|---:|---:|---:|
| silver nDCG@5 (pooled) | 0.330 | 0.364 | 0.393 | 0.428 |
| held-out pool nDCG@5 | 0.283 | 0.348 | 0.391 | 0.450 |

- The free signal alone (F1@0) only reaches M3.v2 (0.330 vs 0.337). The judge labels add
  the +0.10.
- **No plateau in the data:** roughly +0.03 silver nDCG@5 per tripling of judged tasks,
  still rising at 8.9k. More training signal would pay off. Since the Gemini budget is
  spent, that signal has to come from distillation (next-label-free-work.md, "Later").
- **Within a run:** three quarters of the gain arrive in the first half epoch
  (0.15 → 0.37 held-out), then +0.08 over the next 5.5 epochs. F1@all's best evaluation is
  its last one: the linear LR decay to 0 ended it, not early stopping, and validation
  never dropped (no overfitting). So round 1 was cut short by its LR schedule, not
  limited by data. F1@1k and F1@3k did converge (best near epoch 5).
- **Speed:** 1.48 s/step on a T4 with gradient checkpointing, three times the estimate.
  F1 / F1@all took ~60 min each, the five runs ~3.5 hours on one of Kaggle's two GPUs.

## 10. Round 2 (ran 2026-11-16 on Kaggle's two T4s)

Changes in `scripts/train_f1.py` (hand-off: [f1-kaggle-run.md](f1-kaggle-run.md)):

- **Longer, faster learning** (`F1@all-lr`): LR 5e-5, 10 epochs.
- **Full-catalogue negatives** (`F1@all-bank`, `--bank-negs`): in round 1 each query's
  softmax covered only the ~184 icons in its batch, while evaluation ranks it against all
  2,862. Now every other icon joins as a negative from a bank of stale vectors,
  re-encoded every 100 steps, with no gradient through them. A bank icon is masked when it
  is already in the batch, held out, graded for the query or set aside by the judge, the
  same masks as §4.
- Both together (`F1@all-lrbank`).
- **A bigger encoder** (`F1@all-base`, `--model intfloat/multilingual-e5-base`): if it wins
  clearly, it becomes the teacher to distil into e5-small.
- **Two GPUs:** one process per GPU, each with a queue (`--run A,B,C`).
- **Speed:** `--checkpointing auto` runs one worst-case step (passages padded to 256)
  without gradient checkpointing and keeps it off if the step fits in 85% of the GPU's
  memory. Inputs are padded to a multiple of 8, and peak memory is logged.
- **Lighter scoring:** each best checkpoint carries its icon vectors (`icon_matrix.npz`,
  with a hash of each icon's document), which `FineTunedMatcher` uses when the documents
  are unchanged, so the Mac encodes only the queries.

### Round 2 results

Silver leaderboard: [results/silver/2026-11-16.md](../results/silver/2026-11-16.md); logs in
[results/f1/](../results/f1/). Each run against the one it changes, on the same 3,230 pooled
dev tasks, all with the parent heading (ΔnDCG@5 with its bootstrap 95% interval; hit@1:
tasks only the first gets right / only the second, McNemar p):

| comparison | ΔnDCG@5 | Personal | hit@1 | held-out nDCG@5 |
|---|---:|---:|---:|---:|
| F1@all-lr vs F1@all | **+0.023** [+0.018, +0.028] | +0.021 | 265 / 199, p = 0.003 | 0.494 vs 0.450 |
| F1@all-base vs F1@all-lr | **+0.014** [+0.008, +0.022] | +0.018 | 426 / 343, p = 0.003 | 0.505 vs 0.494 |
| F1@all-base vs F1@all | +0.037 [+0.030, +0.044] | +0.038 | 473 / 324, p < 0.001 | 0.505 vs 0.450 |
| F1@all-bank vs F1@all | −0.022 [−0.028, −0.017] | −0.040 | 271 / 318, p = 0.058 | 0.414 vs 0.450 |
| F1@all-lrbank vs F1@all-lr | −0.018 [−0.024, −0.013] | −0.024 | 244 / 281, p = 0.12 | 0.480 vs 0.494 |
| F1@all + I1-sb (w 0.2) vs F1@all | +0.012 [+0.009, +0.016] | +0.001 | 212 / 143, p < 0.001 | – |

Best pooled: **F1@all-base.v2+parent**, nDCG@5 0.474, hit@1 0.603 [upper bound 0.751],
judged@1 0.852; F1@all-lr.v2+parent 0.459 / 0.578; M3.v2 0.337 / 0.437.

- **The higher LR is the better recipe.** At 5e-5 with 10 epochs, F1@all-lr peaked at
  epoch 5 (held-out 0.494) and early stopping ended it at epoch 6; at 2e-5 the schedule
  had decayed to 0 while validation still rose. It also ran without gradient
  checkpointing (12.7 GB peak), at 0.98 s/step instead of 1.48: 41 minutes.
- **e5-base wins, by less than the LR did.** Same old recipe (2e-5, 6 epochs) as F1@all,
  so e5-base with the new LR is untested (round 3). It is ahead of F1@all-lr on every set
  except Personal-synth (−0.004). Its judged@1 is lower
  (0.852 vs 0.877) and its upper bound higher (0.751 vs 0.700): it picks more icons the
  judge never saw, so the gap is if anything understated. Speed: 3.35 s/step with
  checkpointing (the worst case did not fit in 85% of the T4), 138 minutes.
- **Size and speed of e5-base** (measured on the M1 Mac, PyTorch CPU, 2 threads, one query
  at a time): 278M parameters (192M in the frozen word embeddings, 86M in the 12
  transformer layers; e5-small 118M = 96M + 22M), 768-d vectors, 34 ms p50 / 69 ms p95 per
  query against 9.6 / 14.5 ms, checkpoint 1.1 GB against 470 MB. Decision (author,
  2026-11-17): the deployed model stays e5-small; bigger models are teachers, with the
  option of shipping a bigger one (or several sizes) if it is much better.
- **Full-catalogue negatives hurt**, most on Personal, and also on held-out pool tasks.
  The likely cause is false negatives: every icon outside the judge's ~113 seen ones
  becomes a negative, including good ones it never saw (12–15% of top-1s are such icons).
  In-batch negatives are mostly icons graded for *other* tasks, a safer sample. Soft
  targets from a teacher give those icons mass instead (round 3).
- **Image fusion** helps a little pooled, nothing on Personal; I1-sb alone is weak (0.168,
  but judged@1 0.44, so heavily underestimated). No jina-clip-v2 vectors came back (the run
  guide expected its remote-code load to be the fragile one); SigLIP 2 so400m glyph
  vectors exist, not scored yet.
- The parent heading helps every run on Personal (+0.017 for e5-base) and hurts on
  Realistic (−0.012), as in round 1.

## 11. Round 3: teachers and distillation (built 2026-11-17)

Plan, budget and schedule: [next-data-and-distillation.md](next-data-and-distillation.md);
run guide: [f1-kaggle-run.md](f1-kaggle-run.md). Changes in `scripts/train_f1.py`:

- **Teachers:** `F1@all-base-lr` (e5-base, LR 3e-5, 10 epochs) and `F1@all-large`
  (multilingual-e5-large, 560M parameters, same prefixes and pooling; 2e-5, 5 epochs).
  On a T4, e5-large fits only with `--chunk-tokens 2048` (in its preset). The batch's texts
  are encoded in chunks of about 2k tokens, each under an outer activation checkpoint that
  keeps only the output vectors and reruns the chunk in the backward pass. The loss still
  sees the whole batch, so the gradients are those of batch 128 (the effect of GradCache,
  checked on CPU to 3e-7 relative), and activation memory no longer grows with the number
  of icons in the batch.
- **Teacher scores** (`--score`, or automatically before a student run): the best finished
  teacher by held-out nDCG@5 encodes every training text, title and parent forms, and keeps
  its top 64 icons with scaled scores (`<out>/teacher/<run>.npz`, rows of queries.jsonl).
- **Distillation loss** (`--kd-weight`, `--kd-temp`): KL(teacher ‖ student) over the whole
  catalogue, fresh vectors for the batch's icons and the stale bank (§10) for the rest,
  held-out icons excluded on both sides; added to the graded loss of §4, which stays.
  Unlabelled rows count only here. This is the bank of §10 with soft targets: an icon the
  judge never saw is no longer a plain negative.
- **New rows** (`tcr/finetune.py`, `scripts/f1_data.py`): `distill-mslatte`, the 5,729
  MS-LaTTE tasks in no pool or eval set; later `distill-cand` (generated Personal-style
  tasks) and `para-*` rewrites of judged tasks, used with the original's grades where the
  teacher still ranks a grade-3 icon in its top 10 (`--para graded`) or with teacher
  targets only (`--para kd`); and tasks of the actively selected pools (`--sel`). None of
  them changes the rows earlier runs train on (checked: the rebuilt queries.jsonl has
  round 2's rows byte for byte, plus the new ones).
- **Students:** `F1@all-kd` (e5-small, the LR recipe, KD on the existing texts) and
  `F1@all-kd-ms` (+ MS-LaTTE). Rounds 4–5 add the rewrites, the candidates and the new
  Argon labels (presets `round4`, `round5`).

### Round 3 results (ran 2026-11-18/19; extra grades 2026-11-20)

Silver leaderboards: [results/silver/2026-11-19.md](../results/silver/2026-11-19.md) (plain)
and [results/silver/2026-11-20-extra.md](../results/silver/2026-11-20-extra.md) (with the
grades of every unjudged top-1 of the five leaders, below); logs in
[results/f1/](../results/f1/). Both students learned from F1@all-base-lr: e5-large finished
after them. Same format as round 2, all with the parent heading, on the extra-graded gold:

| comparison | ΔnDCG@5 | Personal | hit@1 | held-out nDCG@5 |
|---|---:|---:|---:|---:|
| F1@all-base-lr vs F1@all-base | +0.012 [+0.007, +0.016] | +0.022 | 243 / 213, p = 0.17 | 0.543 vs 0.505 |
| F1@all-large vs F1@all-base-lr | **+0.028** [+0.021, +0.035] | +0.019 | 424 / 314, p < 0.001 | 0.565 vs 0.543 |
| F1@all-kd vs F1@all-lr | +0.005 [+0.000, +0.009] | +0.013 | 173 / 195, p = 0.27 | 0.509 vs 0.494 |
| F1@all-kd-ms vs F1@all-lr | **+0.009** [+0.004, +0.014] | +0.016 | 252 / 210, p = 0.06 | 0.516 vs 0.494 |
| F1@all-kd-ms vs F1@all-kd | +0.004 [−0.000, +0.008] | +0.003 | 191 / 127, p < 0.001 | 0.516 vs 0.509 |
| F1@all-base-lr vs F1@all-kd-ms | +0.023 [+0.016, +0.030] | +0.031 | 426 / 308, p < 0.001 | 0.543 vs 0.516 |

Best pooled (extra-graded): **F1@all-large.v2+parent**, nDCG@5 0.543, hit@1 0.697; the best
e5-small, F1@all-kd-ms.v2+parent, 0.493 / 0.626; F1@all-lr.v2+parent 0.484 / 0.613; M3.v2
0.337 / 0.437. On the plain gold the same order: 0.513 / 0.467 / 0.459.

- **e5-large is the teacher.** It beats e5-base-lr on every set (most on Realistic and
  Public-expanded, +0.05; least on Personal and Public-short, +0.02) in 5 epochs and 6.3 h on a T4 with chunked encoding
  (§11; no layer checkpointing needed). Its held-out curve was still rising at the last
  evaluation (+0.003 per evaluation), partly the LR decaying to 0: round 4 runs it for 8
  epochs (`F1@all-large-e8`).
- **e5-base gains from the LR recipe**, less than e5-small did (+0.012 vs +0.023), more on
  Personal (+0.022).
- **Distillation keeps only part of the teacher's lead.** The teacher (e5-base-lr) was
  0.032 ahead of F1@all-lr on silver; F1@all-kd-ms recovers 0.009 of it (28%; 35% on
  Personal), short of the ≥ half the plan aimed at. On held-out pool tasks it recovers
  45% (0.022 of 0.049), so part of what the student learns is the teacher's view of the
  pool's own labels. Round 4 repeats both students with e5-large (`-L`), whose lead over
  F1@all-lr is 0.059.
- **The 5.7k unlabelled MS-LaTTE tasks add a little** (+0.004, interval touching 0; hit@1
  191 / 127), nothing on Personal. Cheap text helps only through the teacher, so its value
  is bounded by how much the student can absorb.
- **Grading the unjudged picks** (`grade_unjudged.py`, g1: 1,421 picks of the five leaders on
  971 dev tasks, each graded by its eval set's own judge model: Argon 860 tasks, Flash 111; $4.7): on 50 blind
  calibration icons the graders gave the same hit (grade ≥ 2) as the original judge on 48
  and the same grade on 46, so the grades are usable. Only **30% of the unjudged top-1s are
  hits** (Argon 30%, Flash 32%), against ~72% of the judged ones: *hit@1 ub* overstates a
  method by far. Per method, the share of its unjudged top-1s that are hits rises with
  model size (e5-small 0.29, kd-ms 0.32, e5-base 0.34, e5-base-lr 0.35, e5-large 0.38),
  so bigger models were underrated a little on the plain gold, but the order holds. Every
  graded method gains ~0.03 nDCG@5; differences between them barely move (≤ 0.004). From
  now on, compare graded methods on the `-extra` table and ungraded ones on the plain
  table; a new method's unjudged picks are best graded before close calls.
- **Other languages** (the translated eval set, DE FR PT IT ZH JA PL, 1,010 dev tasks each;
  [2026-11-20-xling](../results/silver/2026-11-20-xling.md)): F1@all-kd-ms.v2+parent scores
  0.456–0.482 against 0.516 on the original text, and gains more over the untuned
  e5-small in every language (+0.13 to +0.17) than on the original (+0.09). No pool
  translations are needed (next-data-and-distillation.md step 2).

## 12. Round 4: the e5-large teacher, rewrites, generated tasks (ran 2026-11-20)

Queues and questions: [f1-kaggle-run.md](f1-kaggle-run.md) "Round 4". Every student learned
from F1@all-large (pinned); the e5-large teacher scored every training row first (input of
`select_active.py`). One session, both GPUs; the e8 run took 8.1 h.

### Round 4 results

Silver leaderboard: [results/silver/2026-11-20.md](../results/silver/2026-11-20.md) (plain
gold: the round-4 runs' unjudged picks are not graded yet); logs in
[results/f1/](../results/f1/). Paired, all with the parent heading:

| comparison | ΔnDCG@5 [95%] | Personal | Realistic | hit@1 | held-out nDCG@5 |
|---|---:|---:|---:|---:|---:|
| F1@all-large-e8 vs F1@all-large | **+0.006** [+0.002, +0.009] | −0.001 | −0.002 | 172 / 141, p = 0.09 | 0.576 vs 0.565 |
| F1@all-kd-L vs F1@all-kd | +0.003 [−0.000, +0.006] | +0.002 | +0.007 | 139 / 106, p = 0.04 | 0.517 vs 0.509 |
| F1@all-kd-ms-L vs F1@all-kd-ms | +0.002 [−0.001, +0.006] | +0.004 | +0.008 | 122 / 143, p = 0.22 | 0.516 vs 0.516 |
| F1@all-kd-gen vs F1@all-kd-ms-L | +0.004 [+0.000, +0.007] | +0.002 | +0.001 | 176 / 131, p = 0.01 | 0.527 vs 0.516 |
| F1@all-kd-para vs F1@all-kd-L | +0.004 [−0.000, +0.008] | +0.003 | +0.005 | 193 / 169, p = 0.23 | 0.530 vs 0.517 |
| F1@all-lr-para vs F1@all-lr | **+0.019** [+0.014, +0.025] | +0.023 | +0.038 | 284 / 237, p = 0.04 | 0.530 vs 0.494 |
| F1@all-lr-para vs F1@all-kd-L | **+0.012** [+0.006, +0.017] | +0.007 | +0.031 | 259 / 227, p = 0.16 | 0.530 vs 0.517 |
| F1@all-kd-para vs F1@all-lr-para | −0.007 [−0.013, −0.002] | −0.004 | −0.026 | 233 / 241, p = 0.75 | 0.530 vs 0.530 |
| F1@all-large-e8 vs F1@all-lr-para | +0.040 | +0.035 | +0.066 | | 0.576 vs 0.530 |

Best pooled (plain gold): **F1@all-large-e8.v2+parent**, nDCG@5 0.519, hit@1 0.653
(F1@all-large 0.513 / 0.644); the best e5-small, **F1@all-lr-para.v2+parent**, 0.479 / 0.592
(Personal 0.437, Realistic 0.453); F1@all-kd-gen 0.473; F1@all-lr 0.459; M3.v2 0.337 / 0.437.

- **Rewrites with the original's grades are the best cheap text so far.** Four rewrites per
  judged task (terse, reworded, casual ES, casual RU), each kept only where the teacher still
  ranks one of the original's grade-3 icons in its top 10 (5,604 of 34,650 dropped), lift
  e5-small by +0.019, most on Realistic (+0.038) and Personal (+0.023): the register
  variation the eval sets have and the pool lacked. The same rewrites with teacher targets
  only (`-para`, kd) give a third of that (+0.004 over kd-L): the original's grades carry
  more than the teacher's view of them.
- **A better teacher hardly moves the student.** With e5-large instead of e5-base-lr, the
  distilled students gain +0.002–0.003. F1@all-large is 0.054 ahead of F1@all-lr, and
  F1@all-kd-L keeps 0.008 of that (15%; round 3's e5-base-lr teacher: 28%). Distillation as
  built (KL to the teacher's scores over the catalogue, top 64 kept per text) is not where
  the next gains are; graded text is.
- **The 11k generated Personal-style tasks add a little** (+0.004, hit@1 176 / 131,
  p = 0.01), not on Personal (+0.002): unlabelled text through the teacher again.
- **The longer schedule helps e5-large a little.** The 8-epoch run peaked at epoch 5 (its LR
  decays more slowly than the 5-epoch run's) and stopped at epoch 6 by patience; +0.006 on
  silver, from the public sets and Personal-synth, not Personal or Realistic. Round 5's
  teacher keeps the 8-epoch schedule.
- **A bug in the evaluation schedule** (found here, fixed for round 5): with an odd epoch
  (401 steps, 2 evaluations per epoch) the training loop evaluated at in-epoch step 400 and
  again at the epoch's end one step later. The second evaluation could only tie or lose,
  so it spent one of the 3 patience slots: runs stopped after ~1 epoch without gain instead
  of ~1.5. It changes when runs stop, not their best checkpoints, so earlier results stand.
- **Other languages** ([2026-11-20-xling](../results/silver/2026-11-20-xling.md), nDCG@5
  with the parent heading): F1@all-lr-para is the best e5-small in all 7 languages (DE
  0.471, FR 0.478, PT 0.485, IT 0.469, ZH 0.495, JA 0.484, PL 0.464), +0.008 to +0.032
  over F1@all-lr, although its rewrites add only casual ES and RU: varied phrasing
  carries across languages. Still no pool translations needed.

**With every leader's top-1 graded** (g1–g3, 2026-11-21,
[2026-11-21-extra](../results/silver/2026-11-21-extra.md); judged@1 = 1.000 for all five
below, paired with the parent heading):

| comparison | ΔnDCG@5 [95%] | Personal | Realistic | hit@1 |
|---|---:|---:|---:|---:|
| F1@all-large-e8 vs F1@all-large | +0.005 [+0.002, +0.009] | −0.001 | −0.001 | 174 / 129, p = 0.01 |
| F1@all-lr-para vs F1@all-lr | **+0.021** [+0.015, +0.026] | +0.025 | +0.037 | 306 / 233, p = 0.002 |
| F1@all-kd-gen vs F1@all-kd-ms | +0.006 [+0.002, +0.010] | +0.006 | +0.009 | 190 / 166, p = 0.22 |
| F1@all-lr-para vs F1@all-kd-ms | +0.012 [+0.006, +0.017] | +0.009 | +0.041 | 260 / 229, p = 0.17 |
| F1@all-kd-gen vs F1@all-lr-para | −0.005 [−0.010, +0.000] | −0.003 | −0.032 | 246 / 253, p = 0.79 |
| F1@all-large-e8 vs F1@all-lr-para | +0.045 [+0.037, +0.053] | +0.040 | +0.070 | 550 / 308, p < 0.001 |

Pooled: F1@all-large-e8 0.552 / hit@1 0.711, F1@all-large 0.547, F1@all-lr-para 0.507 /
0.636, F1@all-kd-gen 0.502, F1@all-lr 0.487. Grading moves nothing: graded rewrites stay
the best e5-small text (now with p = 0.002 on hit@1), and the rest of round 4's answers stand.

### Round 5 inputs (2026-11-21)

Vertex v2 judged the 3,000 selected tasks (next-data-and-distillation.md §4). No model has
trained on them, so round 4's scoring pass (`--score`, data/private/f1/teacher/) gives
a fresh eval of its models on new Personal-style and MS-LaTTE text (title, parent where
present; nDCG@5 / hit@1):

| model | all (2,999) | random (1,000) | active (1,999) | random synth / ms | active synth / ms |
|---|---:|---:|---:|---:|---:|
| F1@all-large | 0.554 / 0.626 | 0.637 / 0.761 | 0.512 / 0.559 | 0.741 / 0.799 | 0.563 / 0.550 (hit@1) |
| F1@all-base-lr | 0.522 / 0.618 | 0.607 / 0.732 | 0.480 / 0.560 | 0.731 / 0.735 | 0.565 / 0.551 (hit@1) |
| F1@all-lr | 0.465 / 0.571 | 0.568 / 0.702 | 0.414 / 0.506 | 0.676 / 0.752 | 0.515 / 0.486 (hit@1) |

- **Selection found hard tasks:** −0.20 hit@1 for the teacher, −0.20 for e5-small,
  on Personal-style and MS-LaTTE alike. Whether hard tasks teach more is round 5's
  question (rand1k vs act1k; 1k-rand vs 1k-act).
- The judge's candidates included F1@all-large's top 15, so its top-1 is always judged
  (judged@1 0.99, the others 0.93–0.97) and its numbers are a little favoured.
- On random fresh tasks the e5-small student is 0.06 hit@1 behind the teacher, as on
  silver (0.07 for F1@all-lr vs F1@all-large).

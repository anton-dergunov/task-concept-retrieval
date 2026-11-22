# Plan: more training signal after F1 round 2 (from 2026-11-17; updated 2026-11-22, after round 5)

For the author and for a fresh Claude Code session. Read CLAUDE.md first (hard rules), then
[f1-dual-encoder.md](f1-dual-encoder.md) §12–13 (round 4 and round 5 results) and
[deployment.md](deployment.md) "Two models". Run
guides: [f1-kaggle-run.md](f1-kaggle-run.md) (training) and
[vertex-round3-run.md](vertex-round3-run.md) (Gemini jobs).

## Where things stand (2026-11-22)

Done: Kaggle rounds 3–5, Vertex v1, grade, v2 and g2–g4 (every leader's top-1 judged).
Round 5 is fully scored (f1-dual-encoder.md §13). The plan now has **two models**: a big
one on the Mac, which comes first, and a small one for phones and tablets ("Decisions";
deployment.md "Two models"). Next: round 6's Argon batch (argon-experiments.md "Round 6's
choice", to be approved; all Argon work done by 2026-11-30) and the free Kaggle runs of §7.

- **Best on silver** (g1–g4, +parent, [2026-11-22-extra](../results/silver/2026-11-22-extra.md)):
  **F1@all-large-all3k** (the big model's candidate) nDCG@5 0.567, hit@1 0.714;
  **F1@all-kd-final** (the small model's candidate: graded rewrites + distillation)
  0.534 / 0.659; F1@all-lr-para-all3k 0.521; F1@all-lr-para 0.512; M3.v2 0.336 / 0.437.
  On the translated set, e5-large scores 0.568–0.594 in all 7 languages against 0.597 on
  the original text; e5-small loses 0.035–0.076.
- **Round 5's answers:** active selection did not beat random (a tie on top of the pool,
  −0.007 alone). +3k judged tasks gave e5-small +0.008 (title only), as the learning curve
  predicts, and e5-large +0.010. Graded rewrites add +0.019 on the larger pool, and
  distillation now adds +0.013 on top of them: the two stack.
- **The big model fits the Mac:** e5-large runs at a median 98 ms per task (4 CPU threads,
  fp32) with 1.9 GB peak RSS, against a 200–400 ms budget.
- **v2 (2026-11-21, $58):** 2,999 of 3,000 selected tasks judged by Argon (1,998 active,
  999 random with a label in the training data), all with ES/RU translations, the
  Personal-style ones with invented parents, all with four rewrites. **The active tasks
  are harder:** round 4's teacher has hit@1 0.559 on them against 0.761 on the random
  ones (e5-small F1@all-lr: 0.506 vs 0.702; f1-dual-encoder.md "Round 5 inputs").
- **Round 4's answers:** rewrites with grades +0.019 for e5-small (Personal +0.023,
  Realistic +0.038); with teacher targets only +0.004. The e5-large teacher gives its
  students +0.002–0.003 over e5-base-lr: distillation keeps 15% of the teacher's lead.
  Generated tasks +0.004 (not on Personal). e5-large for 8 epochs +0.006.
- **Unjudged top-1s are mostly misses:** graded blind, 30% of them are hits (calibration
  agreement with the judge 48/50), so *hit@1 ub* is no guide. Every graded method gained
  ~0.03; the order did not change.
- **The learning curve has not flattened:** about +0.03 nDCG@5 per tripling of judged pool
  tasks (0 / 1k / 3k / 8.9k → 0.330 / 0.364 / 0.393 / 0.428). The 11.9k point (round 5,
  LR recipe) is on it: +0.008 over 8.9k, title only.
- **Cost of judged data** (spend log): Argon ≈ $19 per 1,000 tasks in batch at the promotional
  price, which doubles after 2026-11-30; generating Personal-style tasks with Flash ≈ $2.6 per
  1,000 (measured in v1). Tripling the pool at random (+20k tasks) would cost ~$400 for +0.03.
- **Budget:** $96.7 of round 3's $115 cap spent (spend log $387.82, cap $406.14), $18.3
  left after g4 ($1.85). The author allows ~$50 more (up to ~$100) for
  [argon-experiments.md](argon-experiments.md). The Vertex project showed £123.57 of credit
  left on 2026-11-22 (102 of 226 GBP used; shared with other projects).

## Decisions (author, 2026-11-17; updated 2026-11-22)

- ~~The deployed model stays e5-small~~ **Replaced 2026-11-22 by two models**
  (deployment.md "Two models", CLAUDE.md goals). A *big* model runs on the author's
  MacBook Air M1 for the Emacs agenda (~200 ms per task, up to 300–400 ms; it must not
  burden a busy Mac) and comes first. A *small* model runs on Android / iPad as a rare
  fallback, limited first by memory. e5-large fits the big slot: 98 ms median per task on
  4 CPU threads, 1.9 GB peak RSS in fp32 (measured 2026-11-22). e5-small (~9 ms, 0.9 GB)
  stays the small one. So e5-large is no longer only a teacher, and distillation now serves
  the small model only.
- **How experiments are ranked** (author, 2026-11-22): what it teaches about ML or whether
  it makes a blog post (~60%), then the gain for the big model (~40%).
- **Round 6's Argon money (author, 2026-11-22):** up to $100 on the contact account, for
  argon-experiments.md "Round 6's choice"; its own cap (`CAP_TAG=round6`, $487.82 on the
  spend log).
- **Kaggle: 30 GPU-hours per week on one account** (the binding limit for e5-large runs,
  ~10–12 h each); Colab's free tier is not an option (it terminated an F1 run after ~3 h).
  The project is not in a hurry for final results, so Kaggle work follows the weekly reset
  while the Argon work, which must finish by 2026-11-30, goes first.
- **Budget ~$100** from a new $300 Gemini credit, including actively selected Argon judging.
  The Vertex jobs run in ~3 days (another Vertex job is running until then) and must finish
  before 2026-11-30. The author runs them (hand-off: vertex-round3-run.md).
- **Personal eval text never goes to Kaggle.** Kaggle gets the training data (the
  Personal-style pool is approved) and public text; scoring on Personal stays on the Mac.
- **Moderate or heavy compute runs on Kaggle** (CLAUDE.md "Compute").
- **The project will be shared, so the model must work in many languages.** First measure
  with a translated eval set in DE, FR, PT, IT, ZH, JA, PL; translate the training pools into
  a language only if it turns out weak.

## Why distillation first, and which augmentations

The judge's labels are what lifted F1 (F1@0, the v2 free signal alone, only reaches M3), so
"more data" means more *task-first* text with good labels. Options considered:

| idea | verdict |
|---|---|
| (a) translate tasks into other languages | done for ES/RU (step 0a). More languages only where the translated eval set shows a gap (step 2 below) |
| (b) rephrase tasks keeping every detail | worth it in a stronger form: realistic *registers* (terse note, reworded, casual ES/RU), not synonyms; kept only where the teacher agrees with the original's labels |
| (c) generate tasks from the icon image | least promising: this is the v2 free signal already (example tasks written while looking at the glyph), and F1@0 shows icon-first text is weak — too on-the-nose, one positive each |
| **distillation** | the main bet: the fine-tuned e5-base teacher scores every icon for any text, for free, including the icons the judge never saw. It gives dense soft labels to the existing training text, labels unlabelled text (MS-LaTTE leftovers, generated tasks), and fixes why the full-catalogue negatives failed |
| **active selection for Argon** | the paid step: judge the tasks where the models are unsure or disagree, plus a random control to measure whether selection beat random |

## Schedule

| when | where | step | state |
|---|---|---|---|
| 11-18/19 | Kaggle round 3 | 1. teachers e5-base-lr, e5-large; students F1@all-kd, F1@all-kd-ms | done |
| 11-19 | Vertex v1 ($31) | 2. translated eval set; rewrites; 11k candidate tasks | done |
| 11-20 | Vertex grade ($4.7) | 5. grades of the leaders' unjudged top-1s | done |
| 11-20 | Kaggle round 4 | 3. score candidates; rewrite and candidate students; e5-large 8 epochs | done |
| 11-20 | Mac | 4a. select 2k active + 1k random | done |
| 11-21 | Vertex v2 ($58) | 4b. Argon judges them, + translations, parents, rewrites; g2 grading | done |
| 11-21 | Vertex g3 ($1.1) | grades of the round-4 picks g2 missed; rewrites of 290 more selected tasks | done |
| 11-21/22 | Kaggle round 5 | 6. retrain with the new labels; e5-large teacher and final student; scores for round 6 | done (active ≤ random; all3k on the curve; kd-final and large-all3k lead) |
| 11-22 | Vertex g4 ($1.85) | grades of round 5's leaders' unjudged top-1s | done |
| 11-23/29 | Mac + Vertex v3 (~$50–90) | round 6's Argon batch: #5 VLM reranker, #3 minimal pairs, #2 adversarial tasks, #6 noise ceiling (argon-experiments.md "Round 6's choice"); waits for the author's go | Argon done by 11-30 |
| any time | Kaggle round 6a (free) | §7: the big model with graded rewrites (+ self-distillation) | to prepare |

## 1. Round 3 (Kaggle) — done 2026-11-19

Answers (f1-dual-encoder.md "Round 3 results"): e5-base gains +0.012 from the LR recipe;
e5-large beats it by +0.028 and is the teacher from now on; the distilled e5-small keeps
28% of its teacher's lead (target was ≥ half); the MS-LaTTE text adds +0.004.

Design, as run:

- GPU 0: `F1@all-base-lr` (e5-base, LR 3e-5, 10 epochs; ~2.5–3.5 h) → `F1@all-kd`,
  `F1@all-kd-ms` (e5-small students of the best finished teacher; ~1 h each).
- GPU 1: `F1@all-large` (e5-large, 5 epochs; ~6–8 h by the speed of e5-base ×3.5).
- Questions it answers: does e5-base gain from the LR recipe as e5-small did? How much of
  the teacher's lead does a distilled e5-small keep (target: ≥ half of the 0.015–0.03 gap)?
  Do 5.7k unlabelled MS-LaTTE tasks add anything on top?

## 2. Vertex v1 — done 2026-11-19 ($31)

Outputs: `public_xling.jsonl` (1,010 of the 1,014 dev tasks × 7 languages; 4 tasks got no
translation), rewrites of 8,892 judged tasks (2,686 MS-LaTTE, public; 6,206 Personal-style,
private; 34,650 rows in the tarball after 172 contaminated, empty or unchanged ones were
dropped), 11,002 candidates ($28.8, ~$2.6 per 1,000). Reading 25 public rewrites: most keep
the subject; a few drift ("Training session" → "Attend workout session") or lose the action
in the terse form ("Weigh baby" → "baby weight"), which the teacher filter of
`F1@all-lr-para` is for. The author's spot-check of ~30 is still open.

**Translated eval set, scored** ([2026-11-20-xling](../results/silver/2026-11-20-xling.md),
1,010 tasks per language, nDCG@5 with the parent heading):

| | src | de | fr | pt | it | zh | ja | pl |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M1.v2 (untuned e5-small) | 0.423 | 0.335 | 0.303 | 0.303 | 0.303 | 0.342 | 0.340 | 0.303 |
| F1@all-lr | 0.516 | 0.444 | 0.458 | 0.462 | 0.437 | 0.477 | 0.466 | 0.451 |
| F1@all-kd-ms | 0.516 | 0.461 | 0.467 | 0.471 | 0.458 | 0.476 | 0.482 | 0.456 |
| gain of F1@all-kd-ms over M1 | +0.093 | +0.126 | +0.164 | +0.169 | +0.155 | +0.133 | +0.142 | +0.153 |

**Rule outcome: no language gets pool translations.** In every language F1's gain over M1
is larger than on the original text (+0.11 to +0.17 against +0.09): fine-tuning on EN/ES/RU
carries over, and F1 loses less to translation than M1 does (−0.03 to −0.06 against
−0.08 to −0.12; part of the loss is the translation itself, scored against the original's
gold). M3 collapses in French, Portuguese, Italian and Polish (0.19–0.23), the fixed
description vectors' weakness. The distilled student is ahead of F1@all-lr in 6 of 7
languages (+0.01 to +0.02 in DE, IT, JA): the teacher's soft targets travel across
languages. The money stays for v2 and grading. Re-score this set for each round's
leaders (cheap for e5-small: ~4 min per method).

Design, as run:

- **Translated eval set** (`augment_pools.py translate --eval`, `xling-build`): the 1,014
  dev tasks of Public-short, Public-expanded and Realistic, titles and parents, into 7
  languages → `data/datasets/public_xling.jsonl` (public, commit; eval only, never
  training; its titles join the contamination check). Each translation keeps its
  original's gold. Scored with `silver_eval.py --datasets xling`: one table per language
  plus `xling-src`, the same tasks in their original language. ~$1.
  **Rule:** a language whose F1 gain over M1.v2 (the untuned e5-small) is clearly below the
  gain on xling-src gets pool translations in v2 (`augment_pools.py translate --judge-run
  <pool run> --langs …`, ~$2–3 per language, then `train_f1.py --pool-langs`).
- **Rewrites** (`augment_pools.py rewrite`): four per judged training task — terse,
  reworded, casual Spanish, casual Russian — keeping the central subject, its context and
  a meaningful action → `<dataset>.rw.jsonl`, about 9k tasks × 4. ~$4.
- **Candidates** (`synth_personal.py pool`, `augment_pools.py extra`): ~10k new
  Personal-style tasks, unlabelled, minus near-duplicates of judged ones →
  `personal_synth_cand.jsonl`. ~$31. They are distillation text and the pool that
  active selection picks from.

## 3. Round 4 (Kaggle) — done 2026-11-20

Answers (f1-dual-encoder.md §12, paired on silver with the parent heading):
- e5-large teacher vs e5-base-lr for the student: +0.003 (kd-L vs kd), +0.002 (kd-ms-L vs
  kd-ms). Distillation is not where the gains are.
- Rewrites: **with the original's grades +0.019** over F1@all-lr (`F1@all-lr-para`, the best
  e5-small, +0.012 over kd-L); with teacher targets only +0.004 over kd-L. Round 5 takes
  graded rewrites, v2 writes them for the new tasks too.
- 11k generated Personal-style tasks: +0.004 over kd-ms-L, +0.002 on Personal.
- `F1@all-large-e8`: +0.006 [+0.002, +0.009] over F1@all-large; peaked at epoch 5 of the
  slower schedule, stopped at epoch 6. Round 5's teacher keeps 8 epochs.
- Found: a double evaluation at each odd-length epoch's end spent a patience slot; fixed in
  train_f1.py for round 5.

Design, as run:

Tarball rebuilt 2026-11-20 (19.7 MB): round 3's rows unchanged and in the same relative
order, plus 34,650 rewrite rows and 11,002 candidates; the teacher scores are recomputed,
since they are keyed by the hash of queries.jsonl. Every round-4 student is pinned to
F1@all-large. Queues (f1-kaggle-run.md):

- GPU 0: `--score F1@all-large,F1@all-base-lr,F1@all-lr` (input of step 4), then
  `F1@all-kd-L`, `F1@all-kd-ms-L` (round 3's students with e5-large: the controls),
  `F1@all-kd-gen` (+ candidates), `F1@all-lr-para` (rewrites with inherited grades,
  teacher-filtered: the author's idea (b)), `F1@all-kd-para` (rewrites with teacher
  targets). ~6–8 h.
- GPU 1: `F1@all-large-e8`, e5-large for 8 epochs (~10 h): was round 3's teacher
  undertrained? Decides round 5's teacher epochs.
- Questions: does the e5-large teacher give the student more than e5-base-lr did (kd-L vs
  kd, against round 3's +0.005)? Do rewrites help, and which way (-para vs kd-L)? Do 11k
  generated Personal-style tasks help (-gen vs kd-ms-L), on Personal above all?

## 4. Active selection and Vertex v2 — done 2026-11-21 ($57.7)

Selection done 2026-11-20 (`select_active.py --teacher F1-all-large --student F1-all-lr
--teacher2 F1-all-base-lr`): 1,997 Personal-style + 1,003 MS-LaTTE tasks, 2,000 active (mean
priority 1.72 against 1.15 over all 16,731 candidates; 1,120 distinct teacher top-1s) and
1,000 random. v2 now also writes rewrites of the new tasks (~$0.5), since graded rewrites
were round 4's best text. Grade round 4's leaders' unjudged top-1s in the same sitting
(`GRADE_NAME=g2 GRADE_METHODS=F1@all-large-e8.v2+parent,F1@all-lr-para.v2+parent,F1@all-kd-gen.v2+parent`, ~$3–5).

- `scripts/select_active.py`: priority = 1 − teacher top-1 probability + 0.5 if the
  teacher's top-1 is not in the student's top 5 + 0.5 if two teachers disagree; no junk
  (lowest 15% teacher score), ≤ 4 tasks per teacher top-1, no near-duplicate topics,
  ≤ 33% MS-LaTTE. **2,000 active + 1,000 random** (the control). Each task's teacher top 15
  joins the judge's candidate pool (`judge_batch.py prepare --extra-candidates`), so the
  icons the models favour get judged.
- `vertex_round3_jobs.sh v2`: Argon judges both pool files (`pool-synth3-j4`,
  `pool-mslatte2-j4`), then their ES/RU translations and parent headings. ~$63 + ~$2.

Result: Argon $55.85 for 3,000 tasks ($18.6 per 1,000), translations, parents and
rewrites $1.87. 2,999 labelled (1 failed; 2 without a graded v2 icon), ES/RU for all but 8
MS-LaTTE tasks. Two fixes before round 5 (2026-11-21): the 290 selected tasks in the
validation bucket now train (they never validate, so the bucket only dropped paid labels;
their rewrites were written in the g3 sitting), and the 3,000 tasks are no longer also
distillation text (distill-cand 9,005, distill-mslatte 4,726). How round 4's models do on
these fresh tasks: f1-dual-encoder.md "Round 5 inputs".

## 5. Grading unjudged top-1s — done 2026-11-20 ($4.7)

Result (f1-dual-encoder.md "Round 3 results"): 1,421 picks of the five round-2/3 leaders;
48/50 calibration icons agree with the judge on hit; only 30% of the unjudged top-1s are
hits; every graded method gains ~0.03 and the order holds. Grade the round-4 and round-5
leaders the same way (`GRADE_NAME=g2`, ~$3–5 each time) before deciding close calls.

Round 4's leaders (2026-11-21): g2 ($0.54) graded only F1@all-large-e8's 113 picks:
`grade_unjudged.py` took each method's newest stored ranking, which for the two e5-small
leaders was the translated-set run (no gold, so no picks). Fixed (it skips -xling
rankings); g3 ($1.03) graded their 236 picks. Calibration 49/50 on hit both times; 39%
(g2) and 26% (g3) of the picks are hits. With g1–g3 every leader's top-1 is judged, and
round 4's answers hold (f1-dual-encoder.md §12). Round 5's leaders: `GRADE_NAME=g4`.

Close variants now differ by 0.01–0.02 nDCG@5 while 12–15% of their top-1s were never shown
to the judge (f1-dual-encoder.md §7). `scripts/grade_unjudged.py` asks each eval set's own
judge model (Argon; Flash for the public sets) to grade them, with its earlier analysis and
graded ranking as the scale and 50 blind calibration icons. `silver_eval.py
--extra-grades g1` then scores with them (written to `<date>-extra`, so the plain table
stays comparable). Grade the leaders of rounds 2–3 together: about 1,000 picks.

## 6. Round 5 (Kaggle, after v2; data built 2026-11-21)

Code done (presets `round5`; `tcr.silver.NEW_POOL_RUNS` joins the new runs once their labels
exist; their tasks carry `sel`, all train and none validates, so held-out numbers stay
comparable). Found while checking the data: the 1k subsets were drawn from the rewrite rows
as well as the tasks, so rand1k and act1k held ~230 tasks each; fixed (999 and 1,000).

- `F1@all-lr-rand1k` vs `F1@all-lr-act1k`: **did active selection beat random** at equal size?
  On top of 8.9k tasks, 1k add only ~+0.004, so the difference may stay inside the interval:
- `F1@1k-rand` vs `F1@1k-act`: the same question with the selected tasks alone (F1@1k's
  recipe; F1@1k, 1k random tasks of the earlier pools, is the third point), where 1k
  tasks are worth ~+0.04. These two decide round 6's batch (argon-experiments.md #1).
- `F1@all-lr-all3k`: the learning-curve point at ~12.9k judged tasks.
- `F1@all-large-all3k`: the teacher retrained, 8 epochs (round 4: the e8 schedule won).
- `F1@all-kd-final`, the student on everything: graded rewrites (round 4's best text) +
  distillation + both unlabelled sources, pinned to `F1@all-large-e8` so it runs on GPU 0
  in the same session as the teacher's retraining (the teacher matters little: round 4).
  The candidate for the next export (deployment.md).
- `F1@all-lr-para-all3k`: its control without distillation: does the teacher add anything
  on top of graded rewrites?
- `F1@all-base-lr-all3k`: e5-base retrained, the mid-size bundle option; preset kept, out
  of the queues (e5-large and e5-small bracket it, and the deployed model stays e5-small).
- Queues (f1-kaggle-run.md): GPU 0 first scores every row with F1@all-large-e8,
  F1@all-lr-para and F1@all-base-lr (round 6's selection input, ~0.5–1 h), then rand1k,
  act1k, 1k-rand, 1k-act, all3k, lr-para-all3k, kd-final (~9–10 h); GPU 1 large-all3k
  (~9–11 h; resumes in a second session if it times out).
- After import: score on silver, grade the leaders (g4), and if round 6 goes ahead, run
  `select_active.py` on **this** build's queries.jsonl (the scores are keyed by its hash:
  do not rebuild the data before selecting).

**Answers so far (2026-11-21; f1-dual-encoder.md §13).** The first session ran out of disk
after `--score` and the four 1k runs; the rest run in a retry commit.

- **Active vs random, on top:** a tie, ΔnDCG@5 +0.000 [−0.004, +0.004] with the parent
  heading (−0.002 title only). **Alone:** active is worse, −0.007 [−0.010, −0.003], hit@1
  112 / 148 (p = 0.03). Uncertainty and committee selection found hard tasks (−0.20 hit@1
  for the teacher), and those teach less than typical ones when they are most of the data.
- **+1k judged tasks on 8.9k: +0.003** either way, as the curve predicts; all on Personal
  (+0.012), from the new tasks' style. At ~$19 per 1k that is the price to beat; graded
  rewrites gave +0.021 for a few dollars.
- **Consequence for round 6:** argon-experiments.md #1 as planned (arms of acquisition
  functions vs random) would measure differences smaller than this round's intervals; by
  its own rule (a tie, or worse) it shrinks to the question itself or gives way to #2/#3,
  which change the kind of data rather than its amount. Decide after all3k: if +3k tasks
  give the predicted ~+0.008, more random labels still pay at the curve's rate; if less,
  the pool is past what more of the same can buy.

**Answers, second half (2026-11-22; f1-dual-encoder.md §13, g1–g4, +parent unless noted).**

- **all3k is on the curve:** +0.008 [+0.003, +0.013] title only (predicted +0.008), +0.013
  with the parent heading (Personal +0.022). So more random labels still pay at the
  curve's rate, ~+0.005 per $50 for e5-small, and e5-large gains a little more from the same
  labels (+0.010).
- **Graded rewrites on 11.9k tasks:** +0.019 over all3k (Realistic +0.062), the same as
  round 4.
- **Distillation on top of graded rewrites:** kd-final vs para-all3k +0.013 [+0.008,
  +0.018] (title +0.017). The teacher's soft targets and the rewrites' grades stack, where
  round 4's teacher targets *instead of* grades lost. F1@all-kd-final, 0.534 / 0.659, is
  the best e5-small (+0.022 over F1@all-lr-para) and the small model's candidate.
- **The retrained teacher:** +0.010 over e8 (F1@all-large-all3k, 0.567 / 0.714), the best
  model and the big model's candidate. The student's gap shrank from 0.045 to 0.033.
- **Languages:** e5-large loses 0.003–0.029 to translation, e5-small 0.035–0.076.
- **Round 6's decision.** By the plan's rule, plain random labels still pay, but only
  ~+0.005 per $50. By the author's ranking (what it teaches first, then the big model's
  gain), they come last. Round 6's Argon batch is **#5 VLM reranker teacher, #3 minimal
  pairs, #2 adversarial tasks, plus #6 the noise ceiling** (~$50–90).
  argon-experiments.md "Round 6's choice" has the scores and an 8-day schedule. It waits
  for the author's go and a raised cap. The teacher scores in `data/private/f1/teacher/`
  are still F1@all-large-e8's (round 5's GPU 0 did not score with F1@all-large-all3k): any
  step that needs the new teacher's scores first runs `--score F1@all-large-all3k` on
  Kaggle.

## 7. Next: round 6 (from 2026-11-22)

**Argon (by 2026-11-30):** argon-experiments.md "Round 6's choice". Code and pilots
11-23/24, Flash generation 11-25, one Argon batch 11-26, buffer to 11-29.

**Kaggle round 6a (free; can start before the Argon data exists).** The big model is now
the first priority, and it has never had the recipe that made the small one better:

- `F1@all-large-para-all3k`: e5-large with graded rewrites (round 4/5's best text, +0.02
  for e5-small, never given to e5-large). Rewrites raise steps per epoch from 471 to ~772,
  so 8 epochs would take ~18 h at 10.8 s/step. Either use 5 epochs (~11.6 h, and check the
  LR decay against e8's lesson) or `--max-minutes` with a resume in a second session (§4 of
  f1-kaggle-run.md).
- `F1@all-large-kd-self`: born-again self-distillation, an e5-large student of
  F1@all-large-all3k with graded rewrites + the unlabelled text (the kd-final recipe at the
  big size). This is a new technique for the project. It needs `--score F1@all-large-all3k`
  first (~1 h).
- The small model last: kd-final re-pinned to the best big model once round 6a has one.

**Export (only when asked; deployment.md):** fp16 ONNX bundles of the big and the small
candidates, each with latency and RSS measured on the exported encoder.

## Budget

| item | planned $ | spent $ |
|---|---:|---:|
| translated eval set | 1 | 0.50 |
| rewrites | 4 | 1.52 |
| ~10k candidates (11,002) | 31 | 28.83 |
| unjudged grading (g1) | 10 | 4.68 |
| Argon, 3k selected tasks | 63 | 55.85 |
| their translations, parents and rewrites | 2.5 | 1.87 |
| unjudged grading of round 4's leaders (g2, g3) | 3–5 | 1.57 |
| unjudged grading of round 5's leaders (g4) | 3–5 | 1.85 |
| **total** | **~115** | **96.68** |

The cap is fixed at $406.14 on the spend log (`data/private/augment/cap-round3.json`,
`BUDGET=115`); the log stands at $387.82, so $18.3 is left. No language needed pool
translations. Beyond the cap, the author allows ~$50
(up to ~$100) for [argon-experiments.md](argon-experiments.md). The Argon price doubles
after 2026-11-30.

## Backlog (after rounds 3–5)

Argon-paid experiments (active-learning strategies, adversarial tasks, minimal pairs,
pairwise preferences, a VLM reranker teacher, the judge-noise ceiling), with costs:
[argon-experiments.md](argon-experiments.md).

Data and labels:
- **Minimal pairs:** an LLM edits a task so the right icon changes ("Call the plumber about
  the sink" → "Email the plumber …"), labelled by the judge or the teacher; teaches what
  matters in a title.
- **Parent-heading noise:** drop, swap or add a misleading parent; Realistic loses with the
  parent in every run.
- **Query noise on the fly** (no LLM): typos, lowercase, dropped diacritics, token dropout.
- **Pool translations into more languages** where step 2's rule says so.
- ~~More active rounds if round 5 shows selection beats random~~: it did not (§6).

Models and training:
- **Positive-aware hard negatives** (NV-Retriever): mine with the teacher, drop negatives
  scoring near the positive.
- **Learned per-icon vectors** on top of the best F1 (f1-dual-encoder.md §8); **frozen
  icon tower** as a fallback.
- **Other teachers:** Qwen3-Embedding-0.6B or bge-m3 (need their own pooling and prompts);
  GradCache for large batches on big models.
- **Teacher ensembles**, e.g. e5-base + e5-large + I1: image signal through distillation at
  no query cost.

Images:
- Score **I1-so** (SigLIP 2 so400m vectors exist; the Personal queries on the Mac).
- **Glyph vector as an icon-side feature:** a learned projection of the SigLIP image vector
  added to the icon's text vector, trained with F1 (query side unchanged).
- **Fine-tuned text→glyph tower**; **VLM reranker** as a teacher (next-label-free-work.md
  step 5).

Deployment:
- **Vocabulary trimming** (optional bundles per language set): e5's 250k-token embedding
  table is most of its size (e5-base 278M → ~115M with EN/ES/RU tokens, e5-small 118M →
  ~35M). Same speed per query, smaller download and memory, faster load. The default
  bundle keeps the full vocabulary, since the model must work in many languages.
- **Several model sizes** as bundles, if a bigger one is much better.

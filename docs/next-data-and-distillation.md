# Plan: more training signal after F1 round 2 (from 2026-11-17; updated 2026-11-20, round 4)

For the author and for a fresh Claude Code session. Read CLAUDE.md first (hard rules), then
[f1-dual-encoder.md](f1-dual-encoder.md) §10–11 (round 2 and round 3 results). Run
guides: [f1-kaggle-run.md](f1-kaggle-run.md) (training) and
[vertex-round3-run.md](vertex-round3-run.md) (Gemini jobs).

## Where things stand (2026-11-20)

Done: Kaggle rounds 3 and 4, Vertex v1 and grade, the active selection (3,000 tasks). Next:
Vertex v2 (the author, before 2026-11-30), then Kaggle round 5.

- **Best on silver** (plain gold, [2026-11-20](../results/silver/2026-11-20.md);
  f1-dual-encoder.md §12): F1@all-large-e8.v2+parent nDCG@5 0.519, hit@1 0.653; the best
  e5-small, **F1@all-lr-para.v2+parent** (rewrites with the original's grades), 0.479 /
  0.592; F1@all-lr 0.459; M3.v2 0.337 / 0.437. Extra-graded (round 3's leaders only,
  [2026-11-20-extra](../results/silver/2026-11-20-extra.md)): F1@all-large 0.543.
- **Round 4's answers:** rewrites with grades +0.019 for e5-small (Personal +0.023,
  Realistic +0.038); with teacher targets only +0.004. The e5-large teacher gives its
  students +0.002–0.003 over e5-base-lr: distillation keeps 15% of the teacher's lead.
  Generated tasks +0.004 (not on Personal). e5-large for 8 epochs +0.006.
- **Unjudged top-1s are mostly misses:** graded blind, 30% of them are hits (calibration
  agreement with the judge 48/50), so *hit@1 ub* is no guide. Every graded method gained
  ~0.03; the order did not change.
- **The learning curve has not flattened:** about +0.03 nDCG@5 per tripling of judged pool
  tasks (0 / 1k / 3k / 8.9k → 0.330 / 0.364 / 0.393 / 0.428).
- **Cost of judged data** (spend log): Argon ≈ $19 per 1,000 tasks in batch at the promotional
  price, which doubles after 2026-11-30; generating Personal-style tasks with Flash ≈ $2.6 per
  1,000 (measured in v1). Tripling the pool at random (+20k tasks) would cost ~$400 for +0.03.
- **Budget:** $35.5 of round 3's $115 cap spent (spend log $326.67, cap $406.14), $79.5
  left for v2's ~$65.

## Decisions (author, 2026-11-17)

- **The deployed model stays e5-small** (~10 ms per query on the Mac). e5-base (~35 ms,
  1.1 GB) and e5-large are teachers; shipping a bigger model, or several sizes, stays an
  option if it is much better. In the author's own use the model loads per process, so load
  time dominates there; other uses need size and speed.
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
| next | Vertex v2 (~$66) | 4b. Argon judges them, + translations, parents, rewrites; g2 grading | the author, before 11-30 |
| after v2 | Kaggle round 5 | 6. retrain with the new labels; e5-large teacher and final student | |

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

## 4. Active selection and Vertex v2 (by ~11-27, ~$65)

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

## 5. Grading unjudged top-1s — done 2026-11-20 ($4.7)

Result (f1-dual-encoder.md "Round 3 results"): 1,421 picks of the five round-2/3 leaders;
48/50 calibration icons agree with the judge on hit; only 30% of the unjudged top-1s are
hits; every graded method gains ~0.03 and the order holds. Grade the round-4 and round-5
leaders the same way (`GRADE_NAME=g2`, ~$3–5 each time) before deciding close calls.

Close variants now differ by 0.01–0.02 nDCG@5 while 12–15% of their top-1s were never shown
to the judge (f1-dual-encoder.md §7). `scripts/grade_unjudged.py` asks each eval set's own
judge model (Argon; Flash for the public sets) to grade them, with its earlier analysis and
graded ranking as the scale and 50 blind calibration icons. `silver_eval.py
--extra-grades g1` then scores with them (written to `<date>-extra`, so the plain table
stays comparable). Grade the leaders of rounds 2–3 together: about 1,000 picks.

## 6. Round 5 (Kaggle, after v2)

Code done (presets `round5`; `tcr.silver.NEW_POOL_RUNS` joins the new runs once their labels
exist; their tasks carry `sel` and never join validation, so held-out numbers stay
comparable):

- `F1@all-lr-rand1k` vs `F1@all-lr-act1k`: **did active selection beat random** at equal size?
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
- Queues (f1-kaggle-run.md): GPU 0 rand1k, act1k, all3k, lr-para-all3k, kd-final (~8–9 h);
  GPU 1 large-all3k (~9–11 h; resumes in a second session if it times out).

## Budget

| item | planned $ | spent $ |
|---|---:|---:|
| translated eval set | 1 | 0.50 |
| rewrites | 4 | 1.52 |
| ~10k candidates (11,002) | 31 | 28.83 |
| unjudged grading (g1) | 10 | 4.68 |
| Argon, 3k selected tasks | 63 | |
| their translations and parents | 2 | |
| **total** | **~111** | **35.52 so far** |

The cap is fixed at $406.14 on the spend log (`data/private/augment/cap-round3.json`,
`BUDGET=115`); the log stands at $326.67, so $79.5 is left: v2 (~$65), then ~$14 for
pool translations into weak languages (step 2's rule, ~$2–3 each) and a g2 grading of
round 4's leaders (~$3–5). The Argon price doubles after 2026-11-30.

## Backlog (after rounds 3–5)

Data and labels:
- **Minimal pairs:** an LLM edits a task so the right icon changes ("Call the plumber about
  the sink" → "Email the plumber …"), labelled by the judge or the teacher; teaches what
  matters in a title.
- **Parent-heading noise:** drop, swap or add a misleading parent; Realistic loses with the
  parent in every run.
- **Query noise on the fly** (no LLM): typos, lowercase, dropped diacritics, token dropout.
- **Pool translations into more languages** where step 2's rule says so.
- **More active rounds** if round 5 shows selection beats random.

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

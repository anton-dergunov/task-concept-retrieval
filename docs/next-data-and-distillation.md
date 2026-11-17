# Plan: more training signal after F1 round 2 (from 2026-11-17)

For the author and for a fresh Claude Code session. Read CLAUDE.md first (hard rules), then
[f1-dual-encoder.md](f1-dual-encoder.md) §10–11 (round 2 results, round 3 design). Run
guides: [f1-kaggle-run.md](f1-kaggle-run.md) (training) and
[vertex-round3-run.md](vertex-round3-run.md) (Gemini jobs).

## Where things stand

- **Best on silver:** F1@all-base.v2+parent (e5-base fine-tuned), nDCG@5 0.474, hit@1 0.603;
  the best e5-small is F1@all-lr.v2+parent, 0.459 / 0.578; M3.v2 0.337 / 0.437. The higher
  LR gave +0.023 over round 1, e5-base +0.014 more, negatives from the whole catalogue −0.02
  (f1-dual-encoder.md "Round 2 results").
- **The learning curve has not flattened:** about +0.03 nDCG@5 per tripling of judged pool
  tasks (0 / 1k / 3k / 8.9k → 0.330 / 0.364 / 0.393 / 0.428).
- **Cost of judged data** (spend log): Argon ≈ $19 per 1,000 tasks in batch at the promotional
  price, which doubles after 2026-11-30; generating Personal-style tasks with Flash ≈ $3 per
  1,000. Tripling the pool at random (+20k tasks) would cost ~$400 for +0.03.

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

| when | where | step |
|---|---|---|
| now | Kaggle round 3 | 1. teachers e5-base-lr, e5-large; students F1@all-kd, F1@all-kd-ms |
| ~11-20 | Vertex v1 (~$40) | 2. translated eval set; rewrites; ~10k candidate tasks |
| after round 3 | Vertex grade (~$10) | 5. grades of the leaders' unjudged top-1s |
| after v1 | Kaggle round 4 | 3. score candidates; rewrite and candidate students |
| after round 4 | Mac + Vertex v2 (~$65) | 4. select 2k + 1k; Argon judges them, by ~11-27 |
| after v2 | Kaggle round 5 | 6. retrain with the new labels; final teacher and student |

## 1. Round 3 (Kaggle, now)

Code done (f1-dual-encoder.md §11); tarball rebuilt; queues in f1-kaggle-run.md.

- GPU 0: `F1@all-base-lr` (e5-base, LR 3e-5, 10 epochs; ~2.5–3.5 h) → `F1@all-kd`,
  `F1@all-kd-ms` (e5-small students of the best finished teacher; ~1 h each).
- GPU 1: `F1@all-large` (e5-large, 5 epochs; ~6–8 h by the speed of e5-base ×3.5).
- Questions it answers: does e5-base gain from the LR recipe as e5-small did? How much of
  the teacher's lead does a distilled e5-small keep (target: ≥ half of the 0.015–0.03 gap)?
  Do 5.7k unlabelled MS-LaTTE tasks add anything on top?

## 2. Vertex v1 (~11-20, ~$40)

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

## 3. Round 4 (Kaggle, after v1)

Rebuild the tarball (it picks up the rewrites and candidates). Queues:

- `train_f1.py --score <best teacher>,F1@all-lr[,the other teacher]`: teacher and student
  scores of every row, input of step 4 (minutes).
- Students `F1@all-lr-para` (rewrites with inherited grades, teacher-filtered: the
  author's idea (b)), `F1@all-kd-para` (rewrites with teacher targets), `F1@all-kd-gen`
  (+ candidates). The teacher names in the presets may need the round-3 winner first.

## 4. Active selection and Vertex v2 (by ~11-27, ~$65)

- `scripts/select_active.py`: priority = 1 − teacher top-1 probability + 0.5 if the
  teacher's top-1 is not in the student's top 5 + 0.5 if two teachers disagree; no junk
  (lowest 15% teacher score), ≤ 4 tasks per teacher top-1, no near-duplicate topics,
  ≤ 33% MS-LaTTE. **2,000 active + 1,000 random** (the control). Each task's teacher top 15
  joins the judge's candidate pool (`judge_batch.py prepare --extra-candidates`), so the
  icons the models favour get judged.
- `vertex_round3_jobs.sh v2`: Argon judges both pool files (`pool-synth3-j4`,
  `pool-mslatte2-j4`), then their ES/RU translations and parent headings. ~$63 + ~$2.

## 5. Grading unjudged top-1s (~$10, any time after round 3 is scored)

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
- `F1@all-base-lr-all3k` (the teacher retrained) → `F1@all-kd-final`, the student on
  everything: the candidate for the next export (deployment.md).

## Budget

| item | $ |
|---|---:|
| translated eval set | 1 |
| rewrites | 4 |
| ~10k candidates | 31 |
| unjudged grading | 10 |
| Argon, 3k selected tasks | 63 |
| their translations and parents | 2 |
| **total** | **~111** |

The cap (`augment_pools.py cap --budget 100`) stops a job that would cross it; set
`BUDGET=115` to allow all of it, or drop candidates to ~7k (`N_CAND_MAX`).

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

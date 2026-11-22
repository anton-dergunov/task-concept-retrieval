# Running round 3's Gemini jobs on Vertex (hand-off)

For the author, and for a fresh Claude Code session afterwards. What the jobs are for:
[next-data-and-distillation.md](next-data-and-distillation.md) steps 2, 4 and 5. Everything
runs from the Mac (light: batch submissions and polling), unattended, through one script:
`scripts/vertex_round3_jobs.sh`. Every stage is safe to rerun after an interruption: done
work is skipped and a submitted batch job is never resubmitted.

**When:** after the other Vertex job ends (~2026-11-20), and v2 before 2026-11-30, when
Argon's promotional price ends (it doubles).

**State (2026-11-21):** all stages done: v1 2026-11-19 ($30.9), grade 2026-11-20 (g1,
$4.7), v2 2026-11-21 ($57.7: Argon $55.85 for 3,000 tasks, translations, parents and
rewrites $1.87), round 4's leaders graded (g2 $0.54, g3 $1.03; g2 missed two methods, see
next-data-and-distillation.md §5), round 5's leaders graded 2026-11-22 (g4 $1.85;
calibration 50/50 on hit). The spend log is at $387.82 against the cap of $406.14, so $18.3
is left. Open: the author's rewrite spot-check. Next on Vertex: round 6's Argon batch
(argon-experiments.md "Round 6's choice": #5, #3, #2, #6, ~$50–90 beyond this cap; the
author's go first; done by 2026-11-30).

## Account (pinned)

Round 3 runs on the *contact* profile only (gcloud configuration `acervo-contact`, its
project and ADC file), as pinned in `.env` (`VERTEX_ACCOUNT`, `VERTEX_PROJECT`,
`GOOGLE_APPLICATION_CREDENTIALS`, `CLOUDSDK_ACTIVE_CONFIG_NAME`). `tcr/gcp.py` checks the pin
before any Vertex call and stops otherwise: a shell variable that differs from `.env`, an
ADC file of another account (asked from Google) or billing another project, a gcloud
configuration of another account, a `gs://` path outside `gs://<project>-tcr-batch`, a
batch job of another project. Every gcloud call gets `--account` and `--project`. The jobs
script runs the preflight first:

```bash
.venv/bin/python scripts/gcp_check.py --ping    # account, project, billing, API, bucket, one Flash call
```

Once per project, enable the Vertex AI API (2026-11-19: off in the contact project):
`gcloud services enable aiplatform.googleapis.com --configuration=acervo-contact
--project=<VERTEX_PROJECT>`. The bucket is created by the first batch job.

## Budget

The new $300 credit is a fresh trial; the spend log still holds the first one ($291.14).
The first call fixes this round's cap at log + min(credit left − reserve, `BUDGET`), in
`data/private/augment/cap-round3.json` (delete it to set it again). Every job refuses a
round that would cross it.

| stage | what | model | ~$ |
|---|---|---|---:|
| v1 | translated eval set (1,014 dev tasks × 7 languages; eval only) | Flash batch | 1 |
| v1 | rewrites of ~8.9k judged training tasks, 4 each | Flash batch | 4 |
| v1 | ~10k new Personal-style candidate tasks | Flash online | 31 |
| grade | the leaders' unjudged dev top-1s (~1,000) + 50 calibration icons | Argon / Flash batch | 10 |
| v2 | 3,000 selected tasks judged (2k active + 1k random) | Argon batch | 63 |
| v2 | their ES/RU translations and parent headings | Flash batch | 2 |

Total ~$111: run with `BUDGET=115`, or keep `BUDGET=100` and set `N_CAND_MAX=7000`.
v1 keeps `V2_RESERVE` ($75) free for grade and v2 when it sizes the candidates.

## Stage v1 (any time; a few hours)

```bash
CREDIT_LEFT=<USD under Billing → Credits> BUDGET=115 scripts/vertex_round3_jobs.sh v1
```

1. `augment_pools.py translate --eval` on the three public eval runs, then `xling-build`
   → `data/datasets/public_xling.jsonl` (and `data/datasets/xling/*.xl.jsonl`). Public:
   commit them.
2. `augment_pools.py rewrite` on `pool-synth-j4`, `pool-synth2-j4`, `pool-mslatte-j4` →
   `data/private/train_pools/personal_synth_pool.rw.jsonl` (private) and
   `data/train_pools/train_mslatte.rw.jsonl` (public: commit).
3. `synth_personal.py pool --n …` then `augment_pools.py extra` →
   `data/private/train_pools/personal_synth_cand.jsonl`. The pool file is backed up first
   (`personal_synth_pool.before-cand.jsonl`).

A trial first, if wanted (a 5-task batch job each, cents; a batch job can take minutes to
an hour): `augment_pools.py rewrite --judge-run pool-mslatte-j4 --limit 5` and
`augment_pools.py translate --eval --judge-run realistic-j4 --limit 5`, then read the
output lines. The full run continues from there (the 5 count as done).

Watch: `augment_pools.py status --judge-run <run> --kind rw|xleval`; the spend with
`.venv/bin/python -c "from tcr import judge; print(judge.spent())"`.

**After v1 (Mac):**
- Spot-check ~30 rewrites in `train_mslatte.rw.jsonl` (the author): same central subject?
- Score the translated eval set (light):
  `nice .venv/bin/python scripts/silver_eval.py --datasets xling --methods M1,M3,F1@all-lr --versions v2`
  → `results/silver/<date>-xling.md`. A language whose F1 gain over M1.v2 is clearly below
  the gain on `xling-src` gets pool translations (`augment_pools.py translate --judge-run
  <pool run> --langs <l>`, ~$2–3 per language) before v2.
- `f1_data.py build`, upload, Kaggle round 4 ([f1-kaggle-run.md](f1-kaggle-run.md)).

## Stage grade (after Kaggle round 3 is scored; ~1 h)

```bash
CREDIT_LEFT=… BUDGET=115 GRADE_METHODS=F1@all-large.v2+parent,F1@all-base-lr.v2+parent,F1@all-kd-ms.v2+parent,F1@all-lr.v2+parent,F1@all-base.v2+parent \
  scripts/vertex_round3_jobs.sh grade
```

`grade_unjudged.py prepare` collects the methods' unjudged dev top-1s from the stored
rankings (prints how many per method and the estimate), `run` submits one batch job per
judge model and writes `data/private/judge/extra_grades/g1.jsonl`. It prints the agreement
on the calibration icons (same grade / within 1 / same hit); below ~70% same hit, read the
reasons before using the grades. Then:
`silver_eval.py --methods … --versions v2 --extra-grades g1` → `results/silver/<date>-extra.md`.

g1 result: 48/50 calibration icons agree on hit, 46/50 on the grade; 30% of the unjudged
top-1s are hits; $4.7 instead of the ~$10 estimate (f1-dual-encoder.md "Round 3 results").
For later rounds, grade only the new leaders under a new name, e.g. `GRADE_NAME=g2
GRADE_METHODS=F1@all-kd-L.v2+parent,… scripts/vertex_round3_jobs.sh grade`, then score with
`--extra-grades g1,g2`.

## Stage v2 (after Kaggle round 4 and the selection; by ~2026-11-27)

```bash
.venv/bin/python scripts/f1_data.py import ~/Downloads/f1_runs.zip      # round 4, with teacher scores
.venv/bin/python scripts/select_active.py --teacher F1-all-large --student F1-all-lr --teacher2 F1-all-base-lr
CREDIT_LEFT=… BUDGET=115 scripts/vertex_round3_jobs.sh v2
```

`select_active.py` writes `data/private/train_pools/personal_synth_pool3.jsonl` (private)
and `data/train_pools/mslatte2.jsonl` (public: commit), each with `<file>.extra.json`, the
teacher's top 15 per task, which join the judge's candidate pool (`F1-all-large` is the teacher: round 3's best, and
the one round 4 scores every row with). v2 judges them with Argon
(runs `pool-synth3-j4`, `pool-mslatte2-j4`; `judge_batch.py status --run <run>` to
watch), exports the labels, then translates them and invents the parents of the
Personal-style ones.

**After v2:** the labels exist, so `tcr.silver.NEW_POOL_RUNS` adds both runs to the training
pools; their rows carry `sel` (active / random), all are in the train split, and only round
5's presets train on them. `f1_data.py build`, upload, Kaggle round 5 (done 2026-11-21 up
to the upload).

## Continue in a new Claude Code session

> Read CLAUDE.md, docs/next-data-and-distillation.md and docs/vertex-round3-run.md. Vertex
> stage <v1|grade|v2> finished (spend log $…). Check its outputs as the doc says, score
> what it enables, and prepare the next Kaggle round.

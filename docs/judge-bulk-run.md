# Hand-off: the LLM judge's bulk batch runs

For the author, in their own terminal, or for a fresh session. The judge is built and
tuned (fine-tuning-plan §6.1); this runs it over every dataset with Vertex batch jobs while
Gemini access lasts. Read CLAUDE.md first.

## Setup (once)

- `.env` points the project at the **public** gcp profile, per process only
  (`VERTEX_PROJECT`, `GOOGLE_APPLICATION_CREDENTIALS`, `CLOUDSDK_ACTIVE_CONFIG_NAME`). The
  globally active gcloud config is not touched, so other jobs on other profiles keep running.
- A budget alert in the billing console (Billing → Budgets) at $150 / $250.
- Every call adds its cost to `data/private/judge/spend.jsonl`; `--max-usd` refuses to
  submit a round whose estimate would cross the cap. Check the total any time:
  `python -c "from tcr import judge; print(judge.spent())"`.

## Runs, in priority order

Access may end at any time, so the most valuable labels come first. Each line is one run;
`run` submits a round, waits, collects, runs the searches locally, and submits the next
round, until every task is done (2–3 rounds; each round takes minutes to a few hours).

```bash
cd ~/projects/research/task-concept-retrieval
B=".venv/bin/python scripts/judge_batch.py"

# 1. Personal (2,941 tasks, Argon, ≈ $57): the real distribution. Prepared and round 1
#    submitted on 2026-11-13, so only `run` is needed.
$B run --run personal-j4 --max-usd 90
# 2. The Personal-style training pool (~5,000 tasks; private; Argon, ≈ $95).
$B prepare --run pool-synth-j4 --pool data/private/train_pools/personal_synth_pool.jsonl \
   --name personal_synth_pool --private --model gemini-4-argon && $B run --run pool-synth-j4 --max-usd 190
# 3. Personal-synth (≈ $29) and Realistic (≈ $12), Argon.
$B prepare --run synth-j4 --dataset personal_synth --model gemini-4-argon && $B run --run synth-j4 --max-usd 220
$B prepare --run realistic-j4 --dataset realistic --model gemini-4-argon && $B run --run realistic-j4 --max-usd 235
# 4. The easy public sets and the MS-LaTTE pool, Flash (≈ $30 together).
$B prepare --run public-short-j4 --dataset public_short && $B run --run public-short-j4 --max-usd 245
$B prepare --run public-expanded-j4 --dataset public_expanded && $B run --run public-expanded-j4 --max-usd 250
$B prepare --run pool-mslatte-j4 --pool data/train_pools/mslatte.jsonl --name train_mslatte \
   && $B run --run pool-mslatte-j4 --max-usd 280
```

Models and costs come from the model comparison (fine-tuning-plan §6.1): Argon ≈ $0.019 and
Flash ≈ $0.007 per task in batch, measured. **Argon's promotional price ends 2026-11-30**
(then it doubles): submit runs 1–3 before about 2026-11-26. Together with the ≈ $17 spent
on development, everything comes to ≈ $250 of the $300. The `--max-usd` caps are cumulative
over all runs (they compare against the spend log), so they stop the tail first.

**Watch:** `$B status --run <run>` (job states and counts); `$B estimate --run <run>`.
**Resume** after a crash, a closed terminal or a new session: run the same `run` line
again. The state lives in `data/private/judge/batch/<run>/`, so a submitted job is never
resubmitted. **Restart one round** only by hand: delete its `rounds/r<k>_c<j>/state.json`
(the job is then resubmitted).
**Export** when a run is finished: `$B export --run <run>` → private datasets to
`data/private/judge/labels/<run>.jsonl`, public ones to `results/judge_labels/<run>.jsonl`
(commit those; they hold ids, grades, reasons and the analysis of public tasks only).

**Do not change `tcr/judge.py`'s prompt during a run**: a run records its prompt version,
and `judge_batch.py` refuses to build the next round under another one.

## After the runs

- Judge vs. the author on the labels made after 2026-11-13 (never seen while the prompt
  was developed): an unbiased check of the judge (`scripts/judge_dev.py` scores the
  judge-dev set only; score the new labels against the exported Personal labels).
- Training (fine-tuning-plan §8, step 5) uses the pool runs only; the eval-set runs serve
  evaluation (judge-vs-human, prediction-powered inference).

## Continuing in a new Claude session

No session is needed while the runs go: `run` does every round by itself, including the
searches between rounds. Start a session afterwards, with this prompt:

> Read CLAUDE.md, docs/judge-bulk-run.md and docs/fine-tuning-plan.md §6.1 and §8. The
> judge's batch runs are finished or in progress: check each with `judge_batch.py status`,
> export the finished ones, and summarise coverage (tasks done / failed / "no good icon"),
> spend (`tcr.judge.spent()`) and the grade distribution. Then score the exported Personal
> labels against my labels made after 2026-11-13 (not the judge-dev set): the unbiased
> check of the judge. Then continue with fine-tuning-plan §8, step 4–5.

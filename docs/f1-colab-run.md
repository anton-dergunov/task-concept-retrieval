# Running F1's training on Colab (hand-off)

For the author, and for a fresh Claude Code session that continues afterwards. The
design is in [f1-dual-encoder.md](f1-dual-encoder.md); this file is only how to run it.
Training does not run on the Mac (f1-dual-encoder.md §9); data building, import and
scoring do.

**State (2026-11-15):** data built (`data/private/f1/f1_data.tar.gz`, 8.1 MB; private:
holds the Personal-style pool text, approved for the author's Drive); the scripts passed
a CPU smoke test (training, pause/resume, checkpoint, silver scoring with merging). No
real run yet.

## 1. On the Mac: build the data (done; rerun after any change to pools, labels or v2)

```bash
.venv/bin/python scripts/f1_data.py build
```

Prints the counts per source, split and language. The tarball also carries
`train_f1.py`, so it is the only file Colab needs.

## 2. Upload

Put `data/private/f1/f1_data.tar.gz` into Google Drive at `MyDrive/tcr-f1/`.

## 3. On Colab

New notebook → *Runtime → Change runtime type → T4 GPU* (free tier is enough). Three cells:

```python
# 1. Drive + data (local disk is faster than Drive for reading)
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/f1 && cp /content/drive/MyDrive/tcr-f1/f1_data.tar.gz /content/f1/ \
  && cd /content/f1 && tar xzf f1_data.tar.gz && ls f1_data
```

```python
# 2. Check the GPU
!nvidia-smi --query-gpu=name,memory.total --format=csv
```

```python
# 3. All five runs, in order F1, F1@all, F1@0, F1@1k, F1@3k; checkpoints go to Drive
!cd /content/f1 && python f1_data/train_f1.py --data f1_data \
    --out /content/drive/MyDrive/tcr-f1/runs --plan 2>&1 | grep -v -i warning
```

**What to expect.** Each run starts with a `base` evaluation (the untrained model), then
logs a line every 20 steps (`loss …, s/step, ≤ N min left`) and evaluates every half
epoch (held-out nDCG@5, EN and ES/RU, unseen-icon recall). About 400 steps per epoch for
F1 / F1@all, 200–260 for the others; at most 6 epochs, usually fewer (early stopping).
Expected ~0.5 s/step on a T4, not yet measured: F1 and F1@all ≤ 35 min each, the other
three ~15 min each, **about 1.5–2 hours in all**. If the first lines show more than
~1.5 s/step, the GPU is not in use (check cell 2). Out of GPU memory: add `--batch 64`.

**Watch** from the notebook output, or from another cell:

```python
!tail -n 3 /content/drive/MyDrive/tcr-f1/runs/*/train_log.jsonl
```

Keep the browser tab open: the free tier ends idle sessions.

## 4. If the session ends

Reconnect and run cells 1 and 3 again. Finished runs (with `done.json`) are skipped; an
interrupted one resumes from its last evaluation (`last/state.pt`, written at every
evaluation). One run alone: `--run F1@3k` instead of `--plan`.

## 5. Bring the results back

In Colab, without the resume states:

```python
!cd /content/drive/MyDrive/tcr-f1 && zip -qr f1_runs.zip runs -x '*/last/*' && ls -lh f1_runs.zip
```

Download `f1_runs.zip` from Drive (about 2.3 GB: five checkpoints of 470 MB), then on the
Mac:

```bash
.venv/bin/python scripts/f1_data.py import ~/Downloads/f1_runs.zip
```

It copies each run's best checkpoint to `.cache/models/F1`, `F1-all`, `F1-0`, `F1-1k`,
`F1-3k` (gitignored) and its `done.json` + `train_log.jsonl` to `results/f1/` (metrics
only: commit them), and prints each run's best held-out nDCG@5 against the base model.

## 6. Score on the silver leaderboard (Mac, ~5 min per checkpoint)

```bash
.venv/bin/python scripts/silver_eval.py --methods F1,F1@all,F1@0,F1@1k,F1@3k --versions v2
```

The earlier methods are merged in from the 2026-11-14 rankings (rescored on the current
gold), so the table in `results/silver/<date>.md` has F1 next to M3.v2 with the Δ column.
Each F1 row also has a `+parent` variant.

## 7. Continue in a new Claude Code session

> Read CLAUDE.md, docs/f1-dual-encoder.md and docs/f1-colab-run.md. The F1 runs are
> imported and scored (results/silver/<date>.md, results/f1/). Next: apply the decision
> rule of f1-dual-encoder.md §7 (hit@1 [lower, upper] vs M3.v2); if the bounds straddle
> M3.v2, count F1's unjudged dev top-1s and estimate the grading cost before running
> anything; plot the learning curve; write the results into f1-dual-encoder.md and
> next-label-free-work.md step 2.

# Running F1 round 2 and the I1 glyph vectors on Kaggle (hand-off)

For the author, and for a fresh Claude Code session that continues afterwards. The
design is in [f1-dual-encoder.md](f1-dual-encoder.md) (§10 for round 2). Kaggle replaced
Colab: Colab ended the round-1 job after 3 hours, while Kaggle runs a notebook headless for
up to 12 hours and gives two T4 GPUs. Data building, import and scoring stay on the Mac.

**State (2026-11-16):** `data/private/f1/f1_data.tar.gz` rebuilt (17.5 MB). Its training
data is byte-for-byte the same as round 1's; new in it: the round-2 `train_f1.py`,
`i1_embed.py` and `glyphs/` (2,862 glyph PNGs, 256 px). It is private: it holds the
Personal-style pool text, approved for the author's Kaggle account. No eval data is in it.

## What runs

| GPU | queue | what it tests | estimate |
|---|---|---|---|
| 0 | `F1@all-lr` | learning rate 5e-5 (from 2e-5), 10 epochs (from 6): round 1 was still improving when the LR decayed to 0 | 1.2–1.7 h |
| 0 | `F1@all-bank` | every other icon of the catalogue as a negative (stale vectors refreshed every 100 steps), not only the ~184 in the batch | 0.8–1.1 h |
| 0 | `F1@all-lrbank` | both | 1.2–1.8 h |
| 1 | `i1_embed.py` | glyph vectors for I1: SigLIP 2 base and so400m, jina-clip-v2 | ~10 min |
| 1 | `F1@all-base` | `multilingual-e5-base` (278M parameters; teacher candidate), F1@all's recipe | 2–3 h |

All four runs are comparable with round 1's F1@all: same data, same 139 held-out icons.
Checkpointing is chosen per run (`--checkpointing auto`): the first step is a worst-case
memory check, and checkpointing stays off when it fits (faster). Round 1 ran at 1.48 s/step
with it on. **In all about 4–5 hours**, within Kaggle's 12-hour limit.

## 1. Upload the data

Kaggle → *Datasets* → your F1 dataset → *New Version* → upload
`data/private/f1/f1_data.tar.gz` (Kaggle unpacks it). Keep the dataset private.

## 2. Notebook settings

The round-1 notebook, or a new one: *Settings* → Accelerator **GPU T4 ×2**, Internet **on**
(models download from Hugging Face), Persistence: Files. Attach the dataset version from
step 1.

## 3. Cells

```python
# 1. Find the data (as in round 1)
from pathlib import Path
INPUT_ROOT = Path("/kaggle/input")
scripts = list(INPUT_ROOT.rglob("train_f1.py"))
assert len(scripts) == 1, f"Expected exactly one train_f1.py, found {len(scripts)}: {scripts}"
TRAIN_SCRIPT = scripts[0]
DATA_DIR = TRAIN_SCRIPT.parent
I1_SCRIPT = DATA_DIR / "i1_embed.py"
WORK = Path("/kaggle/working")
RUNS, I1_OUT = WORK / "runs", WORK / "i1"
print(TRAIN_SCRIPT, DATA_DIR, sorted(p.name for p in DATA_DIR.iterdir()))
!nvidia-smi --query-gpu=index,name,memory.total --format=csv
```

```python
# 2. Resume: if an earlier version's output is attached as an input, start from it
#    (finished runs are skipped, an interrupted one resumes from its last evaluation).
import shutil
for d in INPUT_ROOT.rglob("*"):
    if d.is_dir() and d.name in ("runs", "i1") and DATA_DIR not in d.parents:
        shutil.copytree(d, WORK / d.name, dirs_exist_ok=True)
        print("resumed from", d)
```

```python
# 3. Two queues, one per GPU; logs in /kaggle/working/gpu0.log and gpu1.log
import os, subprocess, time

def launch(gpu, cmd, log):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    return subprocess.Popen(cmd, shell=True, env=env, stdout=open(WORK / log, "w"),
                            stderr=subprocess.STDOUT)

train = f"python {TRAIN_SCRIPT} --data {DATA_DIR} --out {RUNS}"
jobs = [
    launch(0, f"{train} --run F1@all-lr,F1@all-bank,F1@all-lrbank", "gpu0.log"),
    launch(1, f"python {I1_SCRIPT} --data {DATA_DIR} --out {I1_OUT}; "
              f"{train} --run F1@all-base", "gpu1.log"),
]
while any(j.poll() is None for j in jobs):      # progress in the notebook log every 10 min
    time.sleep(600)
    for log in ("gpu0.log", "gpu1.log"):
        lines = [l for l in (WORK / log).read_text().splitlines() if "warn" not in l.lower()]
        print(log, *lines[-2:], sep="\n  ", flush=True)
print("exit codes:", [j.returncode for j in jobs])
```

```python
# 4. Package (without the resume states)
!cd /kaggle/working && zip -qr f1_runs.zip runs i1 gpu0.log gpu1.log -x '*/last/*' && ls -lh f1_runs.zip
```

Then **Save Version → Save & Run All (Commit)**. It runs headless; you can close the
browser. Watch it under the notebook's *Versions* → the running version → *Logs*: cell 3
prints the last lines of both queues every 10 minutes. Each run logs `max_mem_gb` and
`sec_per_step` every 20 steps in `runs/<run>/train_log.jsonl`.

**Quick check before committing (optional, ~5 min, interactive session):**
`!python {TRAIN_SCRIPT} --data {DATA_DIR} --out /tmp/p --probe` times 30 steps with and
without gradient checkpointing for e5-small; add `--model intfloat/multilingual-e5-base`
for the base model.

## 4. If a version fails or times out

Open the failed version's *Output*. If it has `runs/` or `i1/`, attach that version as an
input (*Add Input* → *Your Work* → the notebook) and commit again: cell 2 copies it back,
finished runs are skipped, an interrupted one resumes from `last/state.pt`. A tower that
fails in `i1_embed.py` (most likely jina-clip-v2, which needs remote code) is reported in
`gpu1.log` and skipped; training still runs.

## 5. Bring the results back

Download `f1_runs.zip` from the version's *Output* (about 3 GB: three e5-small
checkpoints of 470 MB and e5-base's ~1.1 GB), then on the Mac:

```bash
.venv/bin/python scripts/f1_data.py import ~/Downloads/f1_runs.zip
```

It copies each run's best checkpoint to `.cache/models/F1-all-lr`, … (gitignored), its
`done.json` + `train_log.jsonl` to `results/f1/` (metrics only: commit them), and the glyph
vectors to `.cache/i1/`. Each checkpoint now carries its icon vectors
(`icon_matrix.npz`), so the Mac encodes only the short queries.

## 6. Score on the silver leaderboard (Mac)

```bash
# round 2, ~2-4 min per checkpoint
nice .venv/bin/python scripts/silver_eval.py --methods F1@all-lr,F1@all-bank,F1@all-lrbank,F1@all-base --versions v2
# I1, image only: encodes the ~3.2k dev queries once per tower (siglip2-base ~5-10 min on
# 2 threads; so400m several times longer, so only if base shows signal)
nice .venv/bin/python scripts/silver_eval.py --methods I1-sb --versions v2
# fusion with the best F1 so far, weight sweep (the query vectors are cached by then)
nice .venv/bin/python scripts/silver_eval.py --methods F1@all+I1-sb~0.1,F1@all+I1-sb~0.2,F1@all+I1-sb~0.35 --versions v2
```

Each call merges the earlier methods from the stored rankings, so `results/silver/<date>.md`
grows into one table. I1 only works with `--versions v2` (the glyph vectors cover the v2
icon set).

## 7. Continue in a new Claude Code session

> Read CLAUDE.md, docs/f1-dual-encoder.md (§10) and docs/f1-kaggle-run.md. Round 2 and the
> I1 glyph vectors are imported and scored (results/silver/<date>.md, results/f1/). Next:
> compare each round-2 run with F1@all (ΔnDCG@5 with its interval, hit@1 bounds, held-out
> curves via scripts/f1_report.py), decide the recipe for the next F1, judge whether e5-base
> is worth distilling (next-label-free-work.md "Later"), and write the results into
> f1-dual-encoder.md and next-label-free-work.md steps 2–3.

# Running F1 on Kaggle (hand-off): round 3, then rounds 4–5

For the author, and for a fresh Claude Code session that continues afterwards. The design
is in [f1-dual-encoder.md](f1-dual-encoder.md) (§11 for round 3), the plan in
[next-data-and-distillation.md](next-data-and-distillation.md). Kaggle runs a notebook
headless for up to 12 hours on two T4 GPUs; data building, import and scoring stay on the
Mac (CLAUDE.md "Compute"). Round 2's guide is this file's git history (2026-11-16).

**State (2026-11-17):** `data/private/f1/f1_data.tar.gz` rebuilt (17.7 MB). Its training rows
are round 2's byte for byte; new: 5,729 unlabelled MS-LaTTE rows (`distill-mslatte`) and the
round-3 `train_f1.py`. Private (Personal-style pool text, approved for the author's Kaggle
account); no eval data. Repacked 2026-11-18 with only train_f1.py changed (memory fixes for
e5-large, below); upload it as a new version.

## What runs (round 3)

| GPU | queue | what it tests | estimate |
|---|---|---|---|
| 0 | `F1@all-base-lr` | e5-base with the LR recipe that won round 2 (3e-5, 10 epochs) | 2.5–4 h |
| 0 | `F1@all-kd` | e5-small distilled from the best finished teacher (F1@all-base-lr or round 2's F1@all-base), plus the usual loss | 0.8–1.3 h |
| 0 | `F1@all-kd-ms` | the same + 5.7k unlabelled MS-LaTTE tasks | 0.9–1.4 h |
| 1 | `F1@all-large` | multilingual-e5-large (560M) as a teacher, 2e-5, 5 epochs | 6–8 h |

Before `F1@all-kd`, the queue scores every training text with the teacher (a few minutes,
`runs/teacher/<run>.npz`). All runs are comparable with round 2: same rows, same 139
held-out icons, same validation tasks.

## 1. Upload the data

Kaggle → *Datasets* → your F1 dataset → *New Version* → upload
`data/private/f1/f1_data.tar.gz` (Kaggle unpacks it). Keep the dataset private.

## 2. Notebook settings

The round-2 notebook: *Settings* → Accelerator **GPU T4 ×2**, Internet **on**, Persistence:
Files. Attach the new dataset version, and **attach round 2's version output** as an input
(*Add Input* → *Your Work* → the notebook → the round-2 version): cell 2 copies its runs, so
F1@all-base can be the teacher and F1@all-lr is there for round 4's scoring.

## 3. Cells

```python
# 1. Find the data
from pathlib import Path
INPUT_ROOT = Path("/kaggle/input")
scripts = [p for p in INPUT_ROOT.rglob("train_f1.py") if "runs" not in p.parts]
assert len(scripts) == 1, f"Expected exactly one train_f1.py, found {len(scripts)}: {scripts}"
TRAIN_SCRIPT = scripts[0]
DATA_DIR = TRAIN_SCRIPT.parent
WORK = Path("/kaggle/working")
RUNS = WORK / "runs"
print(TRAIN_SCRIPT, DATA_DIR, sorted(p.name for p in DATA_DIR.iterdir()))
!nvidia-smi --query-gpu=index,name,memory.total --format=csv
```

```python
# 2. Earlier versions' runs (attached outputs): copied in, so finished runs are skipped,
#    an interrupted one resumes, and finished ones can be teachers.
import shutil, zipfile
for z in INPUT_ROOT.rglob("f1_runs.zip"):
    with zipfile.ZipFile(z) as f:
        f.extractall(WORK)
    print("unpacked", z)
for d in INPUT_ROOT.rglob("runs"):
    if d.is_dir() and DATA_DIR not in d.parents:
        shutil.copytree(d, RUNS, dirs_exist_ok=True)
        print("copied", d)
OLD = {p.parent.name for p in RUNS.glob("*/done.json")} if RUNS.exists() else set()
print("finished before this version:", sorted(OLD))
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
    launch(0, f"{train} --run F1@all-base-lr,F1@all-kd,F1@all-kd-ms", "gpu0.log"),
    launch(1, f"{train} --run F1@all-large", "gpu1.log"),
]
while any(j.poll() is None for j in jobs):      # progress in the notebook log every 10 min
    time.sleep(600)
    for log in ("gpu0.log", "gpu1.log"):
        lines = [l for l in (WORK / log).read_text().splitlines() if "warn" not in l.lower()]
        print(log, *lines[-2:], sep="\n  ", flush=True)
print("exit codes:", [j.returncode for j in jobs])
```

```python
# 4. Package this version's new runs (not the copied ones, not the resume states)
new = [p.parent.name for p in RUNS.glob("*/done.json") if p.parent.name not in OLD]
items = [f"runs/{n}" for n in new] + ["runs/teacher", "gpu0.log", "gpu1.log"]
items = [i for i in items if (WORK / i).exists()]
print("packaging", items)
!cd /kaggle/working && zip -qr f1_runs.zip {" ".join(items)} -x '*/last/*' && ls -lh f1_runs.zip
```

Then **Save Version → Save & Run All (Commit)**. It runs headless; you can close the
browser. Watch it under the notebook's *Versions* → the running version → *Logs*: cell 3
prints the last lines of both queues every 10 minutes. Each run logs `max_mem_gb`,
`sec_per_step` and, for students, `loss_kd` every 20 steps in `runs/<run>/train_log.jsonl`.

**e5-large's memory.** The first try (2026-11-18) ran out of GPU memory: 10.8 of the
T4's 14.6 GB were in use and 3.3 GB more were reserved but fragmented. train_f1.py now
sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` itself, and the `F1@all-large`
preset encodes each batch in checkpointed chunks of 2,048 tokens (`--chunk-tokens`). That
gives the same gradients as the full batch of 128, so its in-batch negatives stay as in
the other runs, while activation memory stays at one chunk's worth. The run's first lines
say whether the memory check also turned on layer checkpointing. If it did, steps are
about 25% slower (7.5–10 h, still under the 12 h limit, and resumable, §4). If it still
runs out of memory, set GPU 1's queue to `F1@all-large --chunk-tokens 1024` and commit
again.

## 4. If a version fails or times out

Attach the failed version's output as an input too and commit again: cell 2 copies its
runs, finished runs are skipped, an interrupted one resumes from `last/state.pt`.

## 5. Bring the results back

Download `f1_runs.zip` from the version's *Output* (about 3 GB: e5-large's checkpoint is
~2.2 GB), then on the Mac:

```bash
.venv/bin/python scripts/f1_data.py import ~/Downloads/f1_runs.zip
```

It copies each run's best checkpoint to `.cache/models/F1-<tag>` (gitignored), its
`done.json` + `train_log.jsonl` to `results/f1/` (commit them) and the teacher scores to
`data/private/f1/teacher/`.

## 6. Score on the silver leaderboard (Mac)

```bash
nice .venv/bin/python scripts/silver_eval.py --methods F1@all-base-lr,F1@all-kd,F1@all-kd-ms,F1@all-large --versions v2
```

Each checkpoint carries its icon vectors, so the Mac encodes only the ~6.5k dev queries:
minutes for e5-small and e5-base, ~15 min for e5-large under `nice`. The table in
`results/silver/<date>.md` grows by merging the stored rankings.

## Rounds 4 and 5

Same notebook, same cells; only the dataset version (rebuilt after the Vertex jobs,
[vertex-round3-run.md](vertex-round3-run.md)) and cell 3's queues change. Attach the
previous round's output each time.

- **Round 4** (after Vertex v1). First the scores for active selection, then the students
  (the presets name the teacher candidates; check `TEACHERS` in train_f1.py if a new
  winner came out of round 3):
  - GPU 0: `{train} --score F1@all-base-lr,F1@all-lr && {train} --run F1@all-kd-para,F1@all-kd-gen`
  - GPU 1: `{train} --score F1@all-large; {train} --run F1@all-lr-para`
  Then `select_active.py --teacher <best> --student F1-all-lr [--teacher2 <other>]` on the Mac.
- **Round 5** (after Vertex v2 and `f1_data.py build`, which then includes the new labels):
  - GPU 0: `--run F1@all-lr-rand1k,F1@all-lr-act1k,F1@all-lr-all3k`
  - GPU 1: `--run F1@all-base-lr-all3k,F1@all-kd-final`

## 7. Continue in a new Claude Code session

> Read CLAUDE.md, docs/next-data-and-distillation.md and docs/f1-kaggle-run.md. Round <n> is
> imported and scored (results/silver/<date>.md, results/f1/). Compare each new run with
> its reference (paired ΔnDCG@5 with interval, hit@1 wins/losses, Personal separately,
> held-out curves), decide the next step by the plan's questions, and write the results
> into f1-dual-encoder.md and next-data-and-distillation.md.

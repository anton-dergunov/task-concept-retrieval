# Running F1 on Kaggle (hand-off): round 4, then round 5

For the author, and for a fresh Claude Code session that continues afterwards. The design
is in [f1-dual-encoder.md](f1-dual-encoder.md) (§11 for round 3), the plan in
[next-data-and-distillation.md](next-data-and-distillation.md). Kaggle runs a notebook
headless for up to 12 hours on two T4 GPUs; data building, import and scoring stay on the
Mac (CLAUDE.md "Compute"). Round 2's guide is this file's git history (2026-11-16).

**State (2026-11-20):** `data/private/f1/f1_data.tar.gz` rebuilt for round 4 (19.7 MB; round
3's is kept as `f1_data.round3.tar.gz`). Round 3's rows are all there, unchanged and in the
same relative order; new: 34,650 rewrite rows (`para-synth`, `para-mslatte`), 11,002
generated Personal-style tasks (`distill-cand`) and the round-4 `train_f1.py`. Private
(Personal-style pool text, approved for the author's Kaggle account); no eval data. Upload
it as a new version.

## Round 4 (after Vertex v1; prepared 2026-11-19, queues revised 2026-11-20)

Round 3 is imported (silver 2026-11-19): F1@all-large is the best teacher (held-out nDCG@5
0.565 vs e5-base-lr 0.543), so every round-4 student is pinned to it. Before the session:

1. **Round 3's runs on Kaggle.** If round 3 has a committed version, attach its output. If it
   ran as a Draft Session (its outputs are gone with the VM), upload
   `~/Downloads/f1_runs.zip` as a new private dataset (Kaggle unpacks it; cell 2 copies its
   `runs/`). Attach round 2's version output as before (F1@all-lr). Delete the zip on the
   Mac only after this.
2. Done 2026-11-20: `f1_data.py build` after v1. Upload `f1_data.tar.gz` as a new version.

| GPU | queue | what it tests | estimate |
|---|---|---|---|
| 0 | `--score F1@all-large,F1@all-base-lr,F1@all-lr` | teacher and student scores of every row, candidates included: input of select_active.py | 0.5–1 h |
| 0 | `F1@all-kd-L` | round 3's F1@all-kd with the e5-large teacher: the control for -para | 0.8–1.3 h |
| 0 | `F1@all-kd-ms-L` | F1@all-kd-ms with e5-large: the control for -gen | 0.9–1.4 h |
| 0 | `F1@all-kd-gen` | + 11k unlabelled generated Personal-style tasks | 1–1.5 h |
| 0 | `F1@all-lr-para` | rewrites with the original's grades, kept where the teacher agrees | 1–1.5 h |
| 0 | `F1@all-kd-para` | rewrites with teacher targets | 1–1.5 h |
| 1 | `F1@all-large-e8` | e5-large for 8 epochs instead of 5: round 3's run was still improving at its last step (+0.003 per evaluation). Sets round 5's teacher epochs | ~10 h |

GPU 0's queue is ~6–8 h. The e8 run is the longest single job so far (5 epochs took 6.3 h);
if the session times out, it resumes from `last/state.pt` in round 5's session (§4), first
in GPU 1's queue. The students are pinned to `F1@all-large`, so e8 finishing early cannot
change their teacher.

Cell 3's two launches for round 4:

```python
jobs = [
    launch(0, f"python {TRAIN_SCRIPT} --data {DATA_DIR} --out {RUNS} --score F1@all-large,F1@all-base-lr,F1@all-lr"
              f" && {train} --run F1@all-kd-L,F1@all-kd-ms-L,F1@all-kd-gen,F1@all-lr-para,F1@all-kd-para", "gpu0.log"),
    launch(1, f"{train} --run F1@all-large-e8", "gpu1.log"),
]
```

After import, score on the Mac:
`nice .venv/bin/python scripts/silver_eval.py --methods F1@all-kd-L,F1@all-kd-ms-L,F1@all-kd-gen,F1@all-lr-para,F1@all-kd-para --versions v2`
(e5-small: minutes each), and F1@all-large-e8 separately (~15 min under `nice`). Then grade
the new leaders' unjudged top-1s (`GRADE_NAME=g2 … vertex_round3_jobs.sh grade`) and
rescore with `--extra-grades g1,g2`.

## What ran (round 3)

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
import shutil, subprocess, zipfile
for z in INPUT_ROOT.rglob("f1_runs.zip"):          # rounds 1–3
    with zipfile.ZipFile(z) as f:
        f.extractall(WORK)
    print("unpacked", z)
for z in INPUT_ROOT.rglob("f1_runs.tar.zst"):      # round 4 on
    subprocess.run(f"zstd -dc '{z}' | tar -xf - -C {WORK}", shell=True, check=True)
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
import shutil, subprocess
new = [p.parent.name for p in RUNS.glob("*/done.json") if p.parent.name not in OLD]
items = [f"runs/{n}" for n in new] + ["runs/teacher", "gpu0.log", "gpu1.log"]
items = [i for i in items if (WORK / i).exists()]
print("packaging", items)
if not shutil.which("zstd"):
    subprocess.run("apt-get update -qq && apt-get install -y -qq zstd", shell=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
if shutil.which("zstd"):
    cmd = f"tar -cf - --exclude='*/last/*' {' '.join(items)} | zstd -3 -T0 -q -o f1_runs.tar.zst"
else:                                   # no zstd: store without compressing (fast, ~1.4x larger)
    cmd = f"zip -0 -qr f1_runs.zip {' '.join(items)} -x '*/last/*'"
print(cmd)
subprocess.run(cmd, shell=True, check=True, cwd=WORK)
!ls -lh /kaggle/working/f1_runs.*
```

(Before round 4, cell 4 used `zip -qr f1_runs.zip …`: ~10 minutes for round 3's 3 GB;
see "Faster packaging" below.) zstd is not preinstalled on Kaggle; `apt-get update` then
`apt-get install zstd` works (checked 2026-11-20, Internet on). If the install ever fails,
the cell falls back to `zip -0` (no compression: fast, ~4.4 GB) on its own, so a headless
commit never loses its runs; `f1_data.py import` reads both.

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

Download `f1_runs.tar.zst` (rounds 1–3: `f1_runs.zip`) from the version's *Output* (about 3 GB: e5-large's checkpoint is
~2.2 GB), then on the Mac:

```bash
.venv/bin/python scripts/f1_data.py import ~/Downloads/f1_runs.tar.zst   # or f1_runs.zip
```

It copies each run's best checkpoint to `.cache/models/F1-<tag>` (gitignored), its
`done.json` + `train_log.jsonl` to `results/f1/` (commit them) and the teacher scores to
`data/private/f1/teacher/`.

### Faster packaging and download (chosen and built 2026-11-19)

**Why cell 4 takes ~10 minutes.** `zip -r` compresses with deflate on one core. Round 3's
zip shrank 4.42 GB to 3.17 GB (28% saved; the fp32 weights compress 23–35%, the `.npz`
files not at all). Timed on the Mac on F1-all-kd's 470 MB checkpoint:

| compressor | time | size |
|---|---|---|
| deflate (`gzip -6`, what zip uses) | 57 s | 65.0% |
| `zstd -3`, one core | 2.9 s | 65.2% |
| `zstd -3 -T0`, all cores | 0.8 s | 65.2% |

So the time goes into deflate, not into the archive format. zstd gives the same size about
20× faster on one core.

**Choice: `tar` + `zstd -3 -T0` → `f1_runs.tar.zst`.** This should take under a minute on
Kaggle, limited by the disk, and the download stays the size it is now. Three changes:

- cell 4: `tar -cf - --exclude='*/last/*' {items} | zstd -3 -T0 -q -o f1_runs.tar.zst`
  (zstd is not preinstalled on Kaggle: the cell runs `apt-get update && apt-get install
  zstd`, and falls back to `zip -0` if that fails);
- cell 2: also unpack `f1_runs.tar.zst` from attached outputs, and still read the older
  `f1_runs.zip`;
- `f1_data.py import`: accept `.tar.zst`. macOS tar is built without zstd, and Python
  3.11's tarfile can't read it, so decompress with the Homebrew `zstd` binary (or
  `zstandard`) first.

If a run is needed before this is built, `zip -0 -qr` (store without compressing) needs no
other change. It packs in about a minute, but the download grows to 4.4 GB.

**The slow download.** Round 3's zip was ready at 01:34 and was in `~/Downloads` at 01:43,
so the click did register. Nothing appeared for the first ~2 minutes; the cause is
unknown. The browser's download list (⌥⌘L in Safari, ⇧⌘J in Chrome) shows whether a
download has started. That session was a *Draft Session*, so the file came from the live VM
and is lost when the session ends. A committed version's output stays.

**Later, if downloading stays awkward: upload to the project's bucket.** Kaggle's *Add-ons →
Google Cloud Services* links a Google account. Then cell 4 runs
`UserSecretsClient().set_gcloud_credentials(project=…)` and
`gcloud storage cp f1_runs.tar.zst gs://<VERTEX_PROJECT>-tcr-batch/f1/round<n>/`. On the
Mac, `gcloud storage cp gs://…/f1_runs.tar.zst ~/Downloads/` shows progress and retries on
its own. The bucket is private and already holds the Vertex batch files, and gcloud is set
up on the Mac (`tcr/vertex_batch.py`). Cost: about $0.12/GB of egress, so ~$0.40 per
round; delete the file after import. Still to check: does the add-on's login also work
in a committed (headless) run? A private Hugging Face repo would be free, but it puts
weights trained on Personal-style text with a new third party, so it is not the choice.

**Disk vs output.** The VM's disk is 57.6 GiB, but only `/kaggle/working` is saved, and that
is capped at 19.5 GiB. It held 7.1 GiB after round 3: `runs/`, with the resume states and
the copied earlier runs, plus the 3 GB archive. It grows each round as copied runs pile
up. With the bucket upload, the archive can be written to `/kaggle/tmp` (not saved), which
frees its 3 GB of output. A smaller lever: the `best/` checkpoints are fp32, and fp16 would
halve them. But check [deployment.md](deployment.md) on encoder precision first.

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

- **Round 4:** the queues above. Then `select_active.py --teacher F1-all-large --student
  F1-all-lr --teacher2 F1-all-base-lr` on the Mac.
- **Round 5** (after Vertex v2 and `f1_data.py build`, which then includes the new labels).
  First set `F1@all-large-all3k`'s epochs in train_f1.py (8 if F1@all-large-e8 beat
  F1@all-large, else 5) and adjust `F1@all-kd-final`'s sources to round 4's answers:
  - GPU 0: `--run F1@all-lr-rand1k,F1@all-lr-act1k,F1@all-lr-all3k,F1@all-base-lr-all3k`
  - GPU 1: `--run F1@all-large-e8,F1@all-large-all3k,F1@all-kd-final` (e8 only if it timed
    out in round 4; finished runs are skipped)

## 7. Continue in a new Claude Code session

> Read CLAUDE.md, docs/next-data-and-distillation.md and docs/f1-kaggle-run.md. Round <n> is
> imported and scored (results/silver/<date>.md, results/f1/). Compare each new run with
> its reference (paired ΔnDCG@5 with interval, hit@1 wins/losses, Personal separately,
> held-out curves), decide the next step by the plan's questions, and write the results
> into f1-dual-encoder.md and next-data-and-distillation.md.

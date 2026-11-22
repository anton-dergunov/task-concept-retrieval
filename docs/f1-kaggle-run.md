# Running F1 on Kaggle (hand-off): round 5 next

For the author, and for a fresh Claude Code session that continues afterwards. The design
is in [f1-dual-encoder.md](f1-dual-encoder.md) (§11 for round 3), the plan in
[next-data-and-distillation.md](next-data-and-distillation.md). Kaggle runs a notebook
headless for up to 12 hours on two T4 GPUs; data building, import and scoring stay on the
Mac (CLAUDE.md "Compute"). Round 2's guide is this file's git history (2026-11-16).

**State (2026-11-21, evening):** round 5's first session (an editor session) ran out of
disk at 17:21: GPU 0 finished `--score`, rand1k, act1k, 1k-rand and 1k-act, then
`F1@all-large-all3k` (GPU 1) and `F1@all-lr-all3k` (GPU 0) died writing checkpoints. Those
four runs and `runs/teacher` were packaged by hand, imported and scored (f1-dual-encoder.md
§13). The retry ([below](#round-5-retry-after-the-disk-ran-out-2026-11-21)) for the other
four runs was first started in the editor too, then stopped and committed: **an editor
session (running cells by hand) loses `/kaggle/working` when it ends; only Save Version →
Save & Run All (Commit) keeps the output.** Next: download the commit's `f1_runs.tar.zst`,
import, score. `data/private/f1/f1_data.tar.gz` is round 5's build (21.3 MB; round 4's kept
as `f1_data.round4.tar.gz`, round 3's as `f1_data.round3.tar.gz`): **do not rebuild it** until
round 6's selection is done.

The round-5 data: 173,381 rows. Round 4's rows are all there, unchanged and in the same
relative order, except the 3,000 distillation rows of the tasks Argon has judged since.
New: the 2,997 selected tasks with labels (`sel` active 1,998 / random 999; EN, ES, RU:
~8.9k rows), their 11,720 rewrite rows, and the round-5 `train_f1.py`. distill-cand 9,005,
distill-mslatte 4,726. Private (Personal-style pool text, approved for the author's Kaggle
account); no eval data. Checked on the Mac: each preset's pool (F1@all-lr 8,887 tasks;
rand1k 9,886; act1k 9,887; all3k 11,884; F1@1k-rand 999 and F1@1k-act 1,000 alone), CPU
smoke runs of F1@all-lr-act1k and F1@1k-act.

## Round 5 retry (after the disk ran out, 2026-11-21)

**What happened.** `/kaggle/working` (the saved output, 19.5 GiB) filled up. Cell 2 used to copy
every earlier run into `runs/` (11 runs, ~10 GB, two of them e5-large at 2.2 GB), and an
e5-large run needs ~11 GB of its own while it trains: `best/` 2.2 GB, `last/state.pt` (weights
+ Adam's two moments, fp32) ~6.6 GB, and `save_best` writes `best.tmp` before it removes the
old `best/` (+2.2 GB for a moment). GPU 1 died at its 4th evaluation (step ~942), in
`save_best`. GPU 0's queue died when its next checkpoint hit the full disk, and cell 4's archive
was cut off at 1.08 GB. Round 4 fit only because it had fewer earlier runs to copy.

**Fix (cells 2 and 4 below).** Cell 2 unpacks earlier runs to `/tmp/prev` (the VM's root disk,
not saved) and copies into `runs/` only `NEED` (this round's teachers) and interrupted runs
(those with `last/state.pt`), then checks that ≥ 15 GB is free. Cell 4 writes with `zstd -f`.
A run that finished in an earlier version but was not copied is not skipped, so the queues list
only the runs still to do. Peak use: ~16 GB.

**Time.** All3k has 471 steps per epoch (e8: 401). At 10.8 s/step, 8 full epochs are 11.3 h
plus 16 evaluations, over the limit. But e8 stopped by patience after epoch 6 (best at epoch 5,
489 min); all3k doing the same is ~9.3 h. The epochs stay at 8: a shorter schedule changes the
LR decay, which is what made e8 beat the 5-epoch run. `--max-minutes 630` pauses it resumably
(§4) if it runs long, early enough for cell 4.

Inputs: the round-5 data version and **round 4's version output** (F1-all-large-e8, the
teacher of all three remaining students). Round 3's runs are not needed. Cell 2's check:

```python
NEED = {"F1-all-large-e8"}   # round 5 retry: the students' teacher
```

| GPU | queue | estimate |
|---|---|---|
| 0 | `F1@all-lr-all3k`, `F1@all-lr-para-all3k`, `F1@all-kd-final` (the first student scores e8's teacher file first, ~20–30 min) | ~6.5 h |
| 1 | `F1@all-large-all3k --max-minutes 630` | ~9.3 h, at most ~11 h |

```python
jobs = [
    launch(0, f"{train} --run F1@all-lr-all3k,F1@all-lr-para-all3k,F1@all-kd-final", "gpu0.log"),
    launch(1, f"{train} --run F1@all-large-all3k --max-minutes 630", "gpu1.log"),
]
```

Run it as **Save Version → Save & Run All (Commit)**, not in the editor: a commit keeps its
output after a timeout, so a paused GPU 1 can resume from it. After import, score all eight
round-5 runs as below.

## Round 5 (after Vertex v2; prepared 2026-11-21)

Questions (next-data-and-distillation.md §6): does active selection beat random at equal
size, on top of the pool (rand1k vs act1k) and alone, where 1k tasks still matter (1k-rand
vs 1k-act)? Where is the learning curve at ~11.9k tasks? The retrained teacher, and the
final student (graded rewrites + distillation) against its control without the teacher.
GPU 0 first writes the scores round 6's selection needs (argon-experiments.md #1), so no
second session is needed before selecting.

Inputs to attach: the new data version; **round 4's version output** (`f1_runs.tar.zst`:
F1-all-large-e8, F1-all-lr-para, the other round-4 runs); and **round 3's runs dataset**
(the `f1_runs.zip` uploaded for round 4: F1-all-base-lr, F1-all-large, F1-all-lr). Cell 2's
check becomes:

```python
NEED = {"F1-all-large-e8", "F1-all-lr-para", "F1-all-base-lr"}   # round 5: scored; the students' teacher
```

| GPU | queue | what it tests | estimate |
|---|---|---|---|
| 0 | `--score F1@all-large-e8,F1@all-lr-para,F1@all-base-lr` | every row, candidates included: round 6's selection input; large-e8's file is also the pinned students' teacher scores | 0.5–1 h |
| 0 | `F1@all-lr-rand1k`, `F1@all-lr-act1k` | +1k random vs +1k active tasks on top of the 8.9k | ~1 h each |
| 0 | `F1@1k-rand`, `F1@1k-act` | the same 1k alone (F1@1k's recipe) | ~0.5 h each |
| 0 | `F1@all-lr-all3k` | all 2,997 new tasks: the learning-curve point | ~1.2 h |
| 0 | `F1@all-lr-para-all3k` | + graded rewrites: kd-final's control | ~1.5–2 h |
| 0 | `F1@all-kd-final` | + distillation from F1@all-large-e8 and both unlabelled sources: the export candidate | ~2 h |
| 1 | `F1@all-large-all3k` | the teacher retrained on everything, 8 epochs | ~9–11 h |

GPU 0's queue is ~9–10 h, ordered so that what decides round 6's spend comes first; if
the session times out, kd-final (last) resumes in the next one (§4), as does GPU 1's run.

```python
jobs = [
    launch(0, f"python {TRAIN_SCRIPT} --data {DATA_DIR} --out {RUNS} --score F1@all-large-e8,F1@all-lr-para,F1@all-base-lr"
              f" && {train} --run F1@all-lr-rand1k,F1@all-lr-act1k,F1@1k-rand,F1@1k-act,F1@all-lr-all3k,F1@all-lr-para-all3k,F1@all-kd-final", "gpu0.log"),
    launch(1, f"{train} --run F1@all-large-all3k", "gpu1.log"),
]
```

After import (`f1_data.py import`), on the Mac:

```bash
nice .venv/bin/python scripts/silver_eval.py --methods F1@all-lr-rand1k,F1@all-lr-act1k,F1@1k-rand,F1@1k-act,F1@all-lr-all3k,F1@all-lr-para-all3k,F1@all-kd-final,F1@all-large-all3k --versions v2
```

then grade the leaders' unjudged top-1s (`GRADE_NAME=g4 GRADE_METHODS=… vertex_round3_jobs.sh
grade`, skips g1–g3), rescore with `--extra-grades g1,g2,g3,g4`, and decide round 6 from
1k-act vs 1k-rand. Do not rebuild the data before `select_active.py`: the new teacher
scores are keyed by this build's queries.jsonl.

## Round 4 (after Vertex v1; prepared 2026-11-19, queues revised 2026-11-20)

Round 3 is imported (silver 2026-11-19): F1@all-large is the best teacher (held-out nDCG@5
0.565 vs e5-base-lr 0.543), so every round-4 student is pinned to it. Before the session:

1. **Round 3's runs on Kaggle.** If round 3 has a committed version, attach its output. If it
   ran as a Draft Session (its outputs are gone with the VM), upload
   `~/Downloads/f1_runs.zip` as a new private dataset (Kaggle unpacks it; cell 2 copies its
   `runs/`). Attach round 2's version output as before (F1@all-lr). Delete the zip on the
   Mac only after this. (Done 2026-11-20: round 4 runs in a new notebook without round 2's
   output, so `runs/F1-all-lr` was added to that zip from the Mac's imported copy,
   `.cache/models/F1-all-lr` + `results/f1/F1-all-lr.*`; the zip, 3.66 GB, is the only
   input besides the data.)
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

After import, score on the Mac (done 2026-11-20):
`nice .venv/bin/python scripts/silver_eval.py --methods F1@all-kd-L,F1@all-kd-ms-L,F1@all-kd-gen,F1@all-lr-para,F1@all-kd-para,F1@all-large-e8 --versions v2`
(e5-small: minutes each, e5-large ~15 min under `nice`). Then grade
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
# 2. Earlier versions' runs (attached outputs), unpacked to /tmp/prev (not saved). Only this
#    round's teachers (NEED) and interrupted runs are copied into runs/: /kaggle/working is
#    capped at 19.5 GiB and an e5-large run needs ~11 GB of it while training.
import shutil, subprocess, zipfile
PREV = Path("/tmp/prev")
PREV.mkdir(parents=True, exist_ok=True)
for z in INPUT_ROOT.rglob("f1_runs.zip"):          # rounds 1–3
    with zipfile.ZipFile(z) as f:
        f.extractall(PREV)
    print("unpacked", z)
zsts = list(INPUT_ROOT.rglob("f1_runs.tar.zst"))   # round 4 on
if zsts and not shutil.which("zstd"):              # not preinstalled on Kaggle (Internet on)
    subprocess.run("apt-get update -qq && apt-get install -y -qq zstd", shell=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
if zsts and not shutil.which("zstd"):              # fallback: the Python decoder
    subprocess.run("pip install -q zstandard", shell=True, check=True)
for z in zsts:
    if shutil.which("zstd"):
        subprocess.run(f"zstd -dc '{z}' | tar -xf - -C {PREV}", shell=True, check=True)
    else:
        import tarfile, zstandard
        with open(z, "rb") as fh, zstandard.ZstdDecompressor().stream_reader(fh) as r, \
                tarfile.open(fileobj=r, mode="r|") as t:
            t.extractall(PREV)
    print("unpacked", z)
P = PREV / "runs"
for d in INPUT_ROOT.rglob("runs"):                 # an attached version's runs/ (resume states too)
    if d.is_dir() and DATA_DIR not in d.parents:
        shutil.copytree(d, P, dirs_exist_ok=True)
        print("found", d)
for done in INPUT_ROOT.rglob("done.json"):         # finished runs wherever an upload put them
    d = done.parent
    if DATA_DIR not in d.parents and not (P / d.name / "done.json").exists():
        shutil.copytree(d, P / d.name, dirs_exist_ok=True)
        print("found run", d)
NEED = {"F1-all-large-e8"}   # round 5 retry: the students' teacher
RUNS.mkdir(parents=True, exist_ok=True)
for d in sorted(P.glob("*")) if P.exists() else []:
    resume = (d / "last" / "state.pt").exists() and not (d / "done.json").exists()
    if (d.name in NEED or resume) and not (RUNS / d.name).exists():
        shutil.copytree(d, RUNS / d.name)
        print("copied", d.name, "(resumes)" if resume else "")
OLD = {p.parent.name for p in RUNS.glob("*/done.json")}
print("finished before this version, copied:", sorted(OLD))
assert NEED <= OLD, (f"missing earlier runs {sorted(NEED - OLD)}; inputs: "
                     + "; ".join(str(p) for p in INPUT_ROOT.glob("*/*/*")))
free = shutil.disk_usage(WORK).free / 1e9
print(f"/kaggle/working free: {free:.1f} GB")
assert free >= 15, "an e5-large run needs ~11 GB in /kaggle/working, the small runs ~3 GB more"
```

The `NEED` check stops the version within a minute when an earlier round's runs did not
come in, instead of failing GPU 0's queue after launch while GPU 1 trains on (round 4's
first commit, 2026-11-20: `--score` found no `runs/F1-all-large/best`). `NEED` lists every
earlier run the queues read: `--score` targets and pinned teachers. Earlier runs not copied
are not skipped, so list only runs still to do. The free-space check is round 5's lesson
(see the retry section).

```python
# 3. Two queues, one per GPU; logs in /kaggle/working/gpu0.log and gpu1.log
import os, subprocess, time

def launch(gpu, cmd, log):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    return subprocess.Popen(cmd, shell=True, env=env, stdout=open(WORK / log, "w"),
                            stderr=subprocess.STDOUT)

train = f"python {TRAIN_SCRIPT} --data {DATA_DIR} --out {RUNS}"
# Round 5 retry's queues (each round's are in its own section above)
jobs = [
    launch(0, f"{train} --run F1@all-lr-all3k,F1@all-lr-para-all3k,F1@all-kd-final", "gpu0.log"),
    launch(1, f"{train} --run F1@all-large-all3k --max-minutes 630", "gpu1.log"),
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
    cmd = f"tar -cf - --exclude='*/last/*' {' '.join(items)} | zstd -3 -T0 -q -f -o f1_runs.tar.zst"
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
interrupted runs, which resume from `last/state.pt`. Take the runs it finished out of the
queues: cell 2 does not copy finished runs, so they would not be skipped. Cell 4 packages
them; import them from that version's archive. If it ran out of disk, read the retry section
of round 5 first.

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

**A committed version keeps its output; there is no hurry.** The output of a *Save & Run
All (Commit)* version (round 4's included) stays with that version: close the tab or stop
the editor session, download when convenient, and download again if the file is damaged.
The next round attaches that version's output as an input (cell 2 unpacks
`f1_runs.tar.zst`), so the archive never goes back up from the Mac. Only a Draft
Session's `/kaggle/working` disappears with its VM. Round 4's archive (2026-11-20) took
four clicks on *Output* → download before the browser started it.

**Easiest command-line route: the Kaggle CLI** (free; nothing is stored with a third
party). Once: `.venv/bin/pip install kaggle`, then Kaggle → *Settings* → *API* → *Create New
Token*, and move the downloaded `kaggle.json` to `~/.kaggle/` (`chmod 600`). Per round:

```bash
.venv/bin/kaggle kernels output <user>/<notebook-slug> -p ~/Downloads --file-pattern '^f1_runs\.tar\.zst$'
```

It fetches the latest version's output; `--file-pattern` (a regex on the file name) keeps
it from also downloading `runs/` (~7 GB, resume states included). Try it first with
`'^gpu0\.log$'`. Not yet tried here; the bucket below is the fallback.

**If both stay awkward: upload to the project's bucket** (Google Cloud Storage, not
Drive). Kaggle's *Add-ons →
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
- **Round 5:** the queues and inputs in "Round 5" at the top. Presets set by round 4's
  answers (f1-dual-encoder.md §12): the teacher retrains for 8 epochs, the final student
  takes graded rewrites and is pinned to `F1@all-large-e8`, so both fit one session.

## 7. Continue in a new Claude Code session

> Read CLAUDE.md, docs/next-data-and-distillation.md and docs/f1-kaggle-run.md. Round <n> is
> imported and scored (results/silver/<date>.md, results/f1/). Compare each new run with
> its reference (paired ΔnDCG@5 with interval, hit@1 wins/losses, Personal separately,
> held-out curves), decide the next step by the plan's questions, and write the results
> into f1-dual-encoder.md and next-data-and-distillation.md.

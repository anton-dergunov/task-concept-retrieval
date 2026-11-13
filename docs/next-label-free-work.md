# Hand-off: work that needs no labels (while the author labels)

For a fresh Claude Code session. Read CLAUDE.md first (hard rules), then this file, then
[fine-tuning-plan.md](fine-tuning-plan.md) §4–6 and §8. Delete this file when its tasks
are done, and move anything still open into fine-tuning-plan §8.

## Where things stand (2026-11-06)

- Two icon description sets: v1 and v2 ([icon-descriptions.md](icon-descriptions.md)).
  `tcr.data.load_icons(version="v1"|"v2")`, `TCR_DESC=v2`, `python -m tcr.eval --desc v2`.
  v2 reads glyphs far better (blind QA: 0.5% vs 27% wrong or partly on 200 random glyphs)
  and does better on the bootstrap eval; v1 stays the default until dev labels decide.
- The labelling pools were rebuilt over v1 + v2 and deployed to the NAS. The author is
  labelling Personal first (about 400 tasks: 200 dev, 200 test). **No human task labels
  exist yet**, so nothing here can be scored on human labels until they come back
  (`rsync -a nas:~/labeller/labels/ labeller/labels/ && python scripts/import_labels.py`).
- Existing matchers: B0, B1 (BM25), M1 (field-weighted e5), M2 (nearest example), M3
  (RRF of B1 + M1), M3x (exported bundle); `tcr/methods/`. The bootstrap eval is
  `tcr/eval.py`.

## Rules that matter here

- **No eval dataset is ever used for training or tuning** (Personal, Realistic,
  Personal-synth, Public-short, Public-expanded). Training text must also pass the
  contamination check: `tcr.describe_v2.contamination_hits(texts)` (exact or e5
  cosine ≥ 0.95 against every eval title). The **test** split
  (`tcr.datasets.split_of`) is never looked at.
- Never match on icon names, tags or categories; the name is only an ID.
- Private text never goes to the free `gemini` backend (`private=True` in
  `tcr.llm.complete_json`). Anything derived from `data/private/` stays out of git.
- LLMs build data and judge; they are never on the query hot path. Matchers must stay
  fast and local, and must be able to abstain.
- The author commits; suggest a commit message after each change. Any job that may run
  longer than an hour: per the author's global CLAUDE.md, either check in at least every
  30 minutes or hand the command over.
- Installing better packages is fine; say what was added and why.

## Tasks, in order

### 1. Training task pools (fine-tuning-plan §4, item 2)

**Done 2026-11-13.** `data/train_pools/mslatte.jsonl` (3,000 public MS-LaTTE titles,
`scripts/datasets/build_train_pools.py`) and the private Personal-style pool
`data/private/train_pools/personal_synth_pool.jsonl` (~5,000,
`scripts/datasets/synth_personal.py pool`). The MASSIVE leftovers and developer TODO
corpora were skipped: the public pool is already the easy part of the mix. The original
brief, kept for reference:

Unlabelled task titles for the LLM judge to label later (step 5 of §8), disjoint from
every eval set by source id **and** by `contamination_hits`.

- **MS-LaTTE remainder:** the titles no public dataset uses. `tcr.describe_v2.reference_pool()`
  already computes this set (2–8 words, deduplicated, eval titles removed); widen its
  length filter if needed. Some of them served as style references in the v2 prompts,
  which does not matter for training.
- **MASSIVE remainder** (EN/ES/RU), the items Public-short does not use, rewritten as
  tasks the same way `scripts/datasets/build_public_short.py` does.
- **A fresh synthetic pool** from the generic **life-area skeletons** in
  `scripts/datasets/synth_personal.py` (written from scratch; `AREA_BASE`). Do not
  use the skeletons derived from private clusters for public training data. Aim for
  long, author-style titles with parent headings, EN/ES/RU.
- Optional: developer TODO corpora (experimentation-strategy §5.1); check licences.
- Output: `data/train_pools/<source>.jsonl` in the shared dataset schema
  (`tcr/datasets.py`), with a `NOTICE.md` for the licences. Commit if public and a few MB.
  Report counts, length mix and the contamination drops.
- Done when: about 10k titles across sources, each line traceable to its source id, the
  contamination check rerun clean.

### 2. First fine-tune of the dual encoder on v2's free signal (§4, item 1; §5)

Tests the training pipeline before judge labels exist.

- **Pairs** from `load_icons(version="v2")`: each example task (EN as `"Parent: task"`,
  ES, RU) → its icon as a rank-1 positive; `context_tasks` as rank-2 (graded or weighted);
  `near_misses` as hard negatives for that icon; in-batch negatives besides.
- **Icon side:** the icon's description **without its example tasks** (depiction,
  readings, intents, reasoning), or a learned per-icon vector initialised from that
  embedding (§5 "hybrid"). The icon side must not contain the query text, or training
  is trivial.
- **Model:** `intfloat/multilingual-e5-small` (the default encoder; fast on CPU), with
  sentence-transformers' MultipleNegativesRankingLoss (or its cached variant) plus the
  hard negatives; MPS on the Mac. Keep the e5 `query:` / `passage:` prefixes.
- **Validation without labels:** hold out about 10% of example tasks per icon, and
  separately about 5% of icons entirely (generalization to unseen icons). Report
  nDCG@10 / P@1 on both against M1 and M3 on the same held-out queries. This is still
  only suggestive (same generator); the verdict comes from dev labels.
- **Deliverable:** a method key (e.g. `F1`) in `tcr/methods/__init__.py` that loads the
  checkpoint from `.cache/models/` (gitignored), usable by `tcr.eval` and
  `scripts/build_label_bundle.py`, and exportable by `tcr/bundle.py` (check
  [deployment.md](deployment.md): encoder precision, ONNX). Results JSON in `results/`.

### 3. Zero-shot image towers (§5.1 A)

- Text → glyph-image retrieval with SigLIP 2 (multilingual text tower) and jina-clip-v2,
  over `data/icons/*.png` or the crisp 256 px render (`tcr.describe_v2.glyph_png`).
  Precompute icon vectors; only the task text is encoded at query time.
- Late fusion with M1 (a weighted sum of description and image similarity), weight left
  to tune on dev later.
- Methods `I1` (image only) and `I1+M1` (fusion) in the registry. Expect modest zero-shot
  quality on abstract glyphs; the point is to measure it.

### 4. LLM-judge harness (§6), ready to calibrate

**Done 2026-11-13** (fine-tuning-plan §6.1): `tcr/judge.py`, tuned on the author's first
51 labels with `scripts/judge_dev.py`; bulk runs with `scripts/judge_batch.py`. The
original brief, kept for reference:

- One request per task: title (+ parent heading), the top-30 candidates pooled as in the
  labeller, shown as one contact sheet with large printed numbers
  (`tcr.judging.contact_sheet` is a starting point), the selection principles in short
  form, and graded output 0–3 per icon plus "no good icon".
- Order shuffled per call; 2–3 permutations; keep stable judgments.
- Batch path via `tcr/vertex_batch.py` (as in `scripts/describe_v2_pass1.py`).
- Smoke-test on about 20 **public training-pool** tasks (not eval tasks). Calibration
  against the author's dev labels waits for those labels (§8 step 4).

## Decisions already taken

- Score (task, icon) pairs; no classifier over a fixed icon vocabulary (§5).
- Default encoder: multilingual e5-small; `bge-small-en` only as the English baseline.
- v2's usefulness prior is a placeholder: (recognizability + task_fit) / 10.
- The labelling grids are drawn from both description versions; provenance labels look
  like `M3.v2@1`.

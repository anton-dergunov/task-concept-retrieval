# Plan: work while the author labels (from 2026-11-14)

For the author and for a fresh Claude Code session. Read CLAUDE.md first (hard rules),
then this file, then [fine-tuning-plan.md](fine-tuning-plan.md) §4–6 and §8. Move
anything still open into fine-tuning-plan §8 when this plan is done.

The author's next labels will take about two weeks. Little has to wait for them: the
judge's labels already cover every eval set and both training pools, so models can be
built, trained and compared now. The human labels then give the verdict and calibrate
the abstention gate. The one deadline is Gemini access on Google Cloud, which may end
(free trial), so everything that needs Gemini comes first.

## Where things stand

- **Judge labels, all runs finished** (2026-11-14, prompt j4; details in fine-tuning-plan
  §6.1): Personal 2,941, Personal-synth 1,506, Realistic 623, Public-short 825,
  Public-expanded 500 (eval sets: evaluation only, never training); the Personal-style pool
  4,986 and the MS-LaTTE pool 3,000 (training). One task failed in 15,381.
  Public runs: `results/judge_labels/<run>.jsonl`; private: `data/private/judge/labels/`.
- **Spend:** $248.50 of the $300 trial by the spend log (`tcr.judge.spent()`), including
  the backfilled first pool generation. The log is an estimate: the billing console has
  the real credit and the trial's end date.
- **Human labels:** 51 Personal (all judge-dev, so they count for nothing else) and 10
  Public-short. The judge's hit@1 against the author is 0.49 on the judge-dev tasks (95%
  interval 0.35–0.63); the batch run reproduces the online runs.
- **Matchers:** B0, B1 (BM25), M1 (field-weighted e5), M2 (nearest example), M3 (RRF of
  B1 + M1), M3x (exported bundle), each on v1 or v2 descriptions. No fine-tuned or image
  method yet.

Three facts about the judge labels that every step below must handle:

1. **Rejected icons.** The 280 icons rejected in the manual review
   (`data/labels/icon_curation.jsonl`, `tcr.data.icon_labels()`) are missing from v2 but
   not filtered from v1, and the judge's candidates come from both. They appear in 1,262
   ranking entries and are the top-1 on 0.2–1.1% of tasks. Drop them from training
   targets and from silver gold.
2. **The judge almost never abstains**: "no good icon" on 0.1% of tasks, against 2 of 51
   for the author. The top-1 gets grade 3 on about 96% of tasks. Judge labels can train
   the ranking but cannot calibrate the abstention gate; that needs the author's "none"
   labels and hit/miss on dev.
3. **Pool bias.** The judge graded only icons from its candidate pool (the ~100 the
   labeller shows, built from B1/M1/M3 over v1 + v2) and its own searches. An icon
   outside that set is *unjudged*, not wrong. Methods that built the pool are favoured
   in silver scores.

## Rules that matter here

- **Eval datasets are never trained on**, their judge labels included. Training uses the
  two pools and v2's free signal (example tasks, context tasks, near-misses). New
  training text passes `tcr.describe_v2.contamination_hits` against every eval title.
- **Test is not looked at**: silver scores are computed on `split_of(id) == "dev"` only
  (judge-dev tasks excluded too).
- Never match on icon names, tags or categories; LLMs never on the query path; matchers
  can abstain. Private text never goes to the free `gemini` backend (`private=True`).
  Nothing derived from `data/private/` goes into git.
- Jobs over an hour: check in every 30 minutes or hand the command over (global
  CLAUDE.md). The author commits.

## 0. Gemini jobs to run now, while access lasts

Everything after this step runs locally or on Colab. These are the only planned jobs that
need Gemini, cheapest first. Argon's promotional price ends 2026-11-30.

| Job | What it gives | Model | Est. cost |
|---|---|---|---|
| **0a. Translate the labelled pools to ES and RU** | Multilingual training pairs with the same labels (the icon does not depend on the language); every pool task is English now, Public-short is 18% ES/RU | Flash, Vertex batch | ~$3–5 |
| **0b. Parent headings for the Personal-style pool** | Personal has a parent heading on every task, the pool on none; without them the model never sees the `"Parent: task"` input it is evaluated on | Flash, Vertex batch | ~$1–2 |
| **0c. More Personal-style pool tasks, judged with Argon** | The pool from ~8k toward ~10k labelled tasks, the "comfortable" size of fine-tuning-plan §6; the learning curve (step 2) shows afterwards whether it mattered | generation Flash, judge Argon | what is left, $0.025 per task |

**All three in one command** (unattended, a few hours; safe to rerun after any
interruption):

```bash
CREDIT_LEFT=<USD under Billing → Credits> scripts/gemini_last_jobs.sh
```

- **Budget.** The first run fixes a cap on the spend log:
  min(log + `CREDIT_LEFT`, $300) − $8 reserve, kept in `data/private/augment/cap.json`
  (delete it to set the cap again). Every job refuses a round that would cross it, and
  the pool extension is sized to what is left: about 1,700 tasks if the whole $300 is
  credit. The log missed the first pool generation's Vertex calls ($10.32, backfilled
  2026-11-14); pool generation now logs its calls. Billing lags by up to a day, and v2
  description work may have been billed to the same account, which the log does not
  show. So `CREDIT_LEFT` from the console is the real limit, and the $300 term only
  guards against a lagging console.
- **0a + 0b:** `scripts/augment_pools.py run --judge-run <run>` translates every task of a
  finished judge run (title, parents, body) and, with `--gen-parents`, invents the
  parent headings, all in one Flash batch job per round. Output, one line per task:
  `data/private/train_pools/personal_synth_pool.aug.jsonl` (private) and
  `data/train_pools/train_mslatte.aug.jsonl` (public: commit). Translated titles too close
  to an eval title are flagged in `contaminated` and must not be trained on.
- **0c:** the original pool is saved as `personal_synth_pool.before-extend.jsonl`.
  `synth_personal.py pool --n … --rounds 4` then rewrites `personal_synth_pool.jsonl` with
  the old tasks plus new ones, keeping the old ids. `augment_pools.py extra` keeps the new
  tasks, minus near-duplicates of judged ones, in `personal_synth_pool_extra.jsonl`. The
  judge run `pool-synth2-j4` (Argon, prompt j4) labels as many of them as the budget allows
  and exports to `data/private/judge/labels/pool-synth2-j4.jsonl`.
- **Watch:** `augment_pools.py status --judge-run <run>`;
  `judge_batch.py status --run pool-synth2-j4`; the spend with
  `python -c "from tcr import judge; print(judge.spent())"`.
- Without Gemini later: `claude-cli` (the author's subscription) covers small private
  jobs, the free `gemini` tier covers public text.

## 1. Silver leaderboard

The first comparison on real task distributions; the bootstrap eval (`tcr/eval.py`)
only tests on the generator's own example tasks.

- Gold per task: the judge's ranking with grades 3/2/1 as gains, rejected icons removed.
- Metrics on the dev split of every eval set: nDCG@5, hit@1 (top-1 graded ≥ 2), and
  **judged@1**, the share of top-1s the judge graded at all. Report hit@1 also on judged
  top-1s only, as `judge_dev.py` does for the author.
- Methods: B0, B1, M1, M2, M3, each on v1 and v2; then every new method below.
- Output: `results/silver/<date>.json` and a markdown table; a script
  `scripts/silver_eval.py`.
- What it can and cannot say: a relative ranking of methods, biased toward pool builders
  and limited by judge accuracy (~0.5 hit@1 against the author). The verdict stays with
  human dev labels.

## 2. Fine-tune the dual encoder (the main bet)

Fine-tuning-plan §5 and §8 step 5: the step most likely to give the best fast model.

- **Model:** `intfloat/multilingual-e5-small` (fast on CPU; the deployed model must stay
  small), `query:` / `passage:` prefixes kept. Query `"Parent: title"` where a parent
  exists.
- **Icon side:** the v2 description **without its example tasks** (depiction, readings,
  intents, reasoning), so the icon side never contains query text. Option: a learned
  per-icon vector initialised from that embedding (§5 "hybrid").
- **Training pairs:**
  - judge labels of both pools: grade-3 icons as positives, grades 2/1 as weaker
    positives (weighted or a graded loss), plus 0a's translations of the same tasks;
  - hard negatives: candidates the judge saw and did not grade, and the top M3 hits it
    rejected;
  - v2's free signal: example tasks → rank-1, `context_tasks` → rank-2, `near_misses` as
    hard negatives;
  - in-batch negatives, with sentence-transformers' (cached) MultipleNegativesRankingLoss
    as the start.
- **Validation:** early stopping on a held-out 10% of pool tasks; about 5% of icons held
  out entirely (unseen-icon generalization); the silver leaderboard (step 1) to compare
  with M1/M3.
- **Learning curve:** train on 1k / 3k / 8k (and ~10.5k after 0c) pool tasks; plot silver
  nDCG@5. It is a reportable result, and it says whether more judging is worth paying for.
- **Deliverable:** method `F1` in `tcr/methods/__init__.py`, loading the checkpoint from
  `.cache/models/` (gitignored); usable by `tcr.eval`, the label bundle and
  `tcr/bundle.py` export ([deployment.md](deployment.md): encoder precision, ONNX).
  Results JSON in `results/`.
- **Hardware:** the Mac (M1, 16 GB, PyTorch MPS) should be enough: ~120M parameters,
  short texts, tens of thousands of pairs; expected tens of minutes per epoch. Write the
  script so it runs unchanged on Colab (device auto-detected, data from one tarball)
  for sweeps and bigger models.

## 3. Zero-shot image towers and fusion (§5.1 A)

- Text → glyph-image retrieval with SigLIP 2 (multilingual text tower) and jina-clip-v2,
  over the crisp 256 px render (`tcr.describe_v2.glyph_png`). Icon vectors precomputed;
  only the task is encoded at query time.
- Methods `I1` (image only) and `I1+M1` / `I1+F1` (weighted sum of similarities).
  Weight chosen on the silver leaderboard for now; re-checked on human dev labels.
- Expect modest zero-shot quality on abstract glyphs; the point is to measure what the
  image adds on top of the descriptions. Runs on the Mac; Colab is faster.

## 4. Abstention gate, built now and calibrated later

- Features per task: top-1 score, margin to top-2, agreement between methods (F1, M3,
  I1), the icon's usefulness prior, query length.
- Prototype the gate on silver: "would the judge grade this top-1 ≥ 2?" gives a
  precision–coverage curve per method.
- The real threshold is set on human dev labels (their "no good icon" and misses), since
  the judge almost never abstains. Until then the curve shows which features carry signal.

## 5. GPU work (Colab)

The steps where the Mac is too slow:

- **Bigger encoders as teachers:** e5-base/large, bge-m3 or Qwen3-Embedding-0.6B
  fine-tuned the same way; distil into e5-small if they win.
- **VLM reranker** (§5.1 B) over the top 30, e.g. a 2–3B Qwen-VL-family model with LoRA,
  trained on the same judge labels; then distilled into the dual encoder (§5, order 3–4).
- Colab free (T4) is enough for e5-base and LoRA on a 2B VLM at small batch; an A100
  (Colab Pro) for anything larger or for sweeps.

## What waits for the human labels

Each of these is quick once ~200 dev labels exist:

- the unbiased judge check (judge vs. labels made after 2026-11-13);
- v1 vs. v2 on human labels (icon-descriptions.md §7);
- the method ranking on human dev labels against the silver ranking: does the silver
  leaderboard predict the human one?;
- the abstention threshold;
- prediction-powered inference: the judge's eval-set labels plus the human labels give
  tighter estimates than the human labels alone;
- final test numbers (fine-tuning-plan §8 step 7).

## Decisions already taken

- Score (task, icon) pairs; no classifier over a fixed icon vocabulary (§5).
- Default encoder: multilingual e5-small; `bge-small-en` only as the English baseline.
- The judge's labels are good enough to train on (the author read them on the 51
  judge-dev tasks); no second judging pass of the eval sets.
- v2's usefulness prior is a placeholder: (recognizability + task_fit) / 10.

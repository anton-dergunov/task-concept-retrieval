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
  4,986 + 1,890 (step 0c, run `pool-synth2-j4`) and the MS-LaTTE pool 3,000 (training).
  One task failed in 16,271. Public runs: `results/judge_labels/<run>.jsonl`; private:
  `data/private/judge/labels/`. Step 0 is done (validation below).
- **Spend:** $291.14 by the spend log after step 0 (cap $295), including the backfilled
  first pool generation. Gemini is no longer needed for the planned work.
- **Human labels:** 51 Personal (all judge-dev, so they count for nothing else) and 10
  Public-short. The judge's hit@1 against the author is 0.49 on the judge-dev tasks (95%
  interval 0.35–0.63); the batch run reproduces the online runs.
- **Matchers:** B0, B1 (BM25), M1 (field-weighted e5), M2 (nearest example), M3 (RRF of
  B1 + M1), M3x (exported bundle), each on v1 or v2 descriptions; on the silver
  leaderboard (step 1, done) M3.v2 led until F1, the fine-tuned dual encoder (step 2,
  rounds 1–2 done 2026-11-16), beat it by +0.10, then +0.14 nDCG@5 (e5-base). The image
  method I1 (step 3) is scored: weak alone, +0.012 in fusion.
- **Next:** [next-data-and-distillation.md](next-data-and-distillation.md) (from
  2026-11-17): teachers, distillation into e5-small, cheap text, active Argon labels.

Three facts about the judge labels that every step below must handle:

1. **Rejected icons.** The 280 icons rejected in the manual review
   (`data/labels/icon_curation.jsonl`, `tcr.data.rejected_names()`) are missing from v2
   but not filtered from v1, and the judge's candidates come from both. They are the
   judge's top-1 on 0.78% of tasks (127 of 16,271): payment 23, table 14, view_real_size
   12, format_quote 7, and ~20 speed_* numbers picked by the literal number in the title
   ("1.75 cues floating point"). **Decision (2026-11-14):** they are dropped from silver
   gold and training targets; methods are *not* filtered, so evaluation can show which
   methods show them (`rej@1`) and what the production filter changes ("served"
   metrics). The hard rule applies only to serving: `scripts/export_bundle.py` leaves them
   out (`--keep-rejected` to keep). More rules can be added the same way later.
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
  2026-11-14); pool generation now logs its calls. On 2026-11-14 the console showed
  £185.36 of the credit used (≈ $246.60 at the credit's own rate, $300 = £225.52) and
  £40.16 (≈ $53) left, so the log matches the bill. The run was started with
  `CREDIT_LEFT=53 RESERVE=5`.
- **The account is a paid one** (upgraded before 2026-11-16 so the credit still covers
  Gemini). Use beyond the credit is billed to the card afterwards, not refused, so the
  cap is the only guard.
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

### Step 0 results (validated 2026-11-14)

| Output | Check |
|---|---|
| `pool-synth2-j4` labels, 1,890 | ids match the run's snapshot; no failed task, unknown icon, bad grade or duplicate; top-1 grade 3 on 97.9% (first pool: 97.1%), rejected top-1 0.79% |
| Pool extension | 1,908 new tasks after near-duplicate removal, 1,890 judged (budget), 18 not; no overlap with the first run. The rewritten `personal_synth_pool.jsonl` lost 2 judged tasks (`pool-33d0a3101c`, `pool-790c87ab8f`); the judge's snapshot keeps them |
| `personal_synth_pool.aug.jsonl`, 6,876 | every judged task; RU titles all Cyrillic, no ES title equal to the EN one; 11 tasks with a translation flagged as too close to an eval title; generated parents on all (depth 1: 18%, 2: 81%, 3: 0.3%; Personal ~46 / 50 / 4%, but the query uses only the nearest parent) |
| `train_mslatte.aug.jsonl`, 2,978 of 3,000 (public) | the 22 left out are brand names ("Path of exile") refused by the Cyrillic check; 59 RU titles equal the EN brand name; 59 tasks with a flagged translation; no private text or paths |

Rules for training that follow: read task text from each judge run's snapshot
(`data/private/judge/batch/<run>/tasks.jsonl`, `tcr.silver.run_tasks`), never from the pool
file; skip a translation equal to the EN title and the languages listed in `contaminated`.

### For the author's next icon review (~1 week)

The judge also matches literal symbols with icons that are not reviewed yet. Two groups,
by the judge's top-1 count over all 16,271 labelled tasks:

- **v2 discarded them, v1 still describes them** (2.6% of top-1s): compress 40,
  file_json 28, format_letter_spacing_wider 19 + _standard 17 (|A| read as a matrix
  norm), text_compare 19, sql 18, vertical_align_center 14, format_line_spacing 12,
  language_spanish 12, http 11, raw_on 10, match_word 9, decimal_increase 8.
- **Kept by v2, the glyph is letters or numerals:** superscript 56 (x² for "scaling
  laws", "second-order"), translate 29, generating_tokens 26, spellcheck 19,
  handwriting_recognition 17, glyphs 13, markdown 10, language_japanese_kana 8,
  picture_as_pdf 5, copyright 5. (format_list_numbered, 127, is used for ranking tasks
  and looks fair.)

Rejects take effect on the next `scripts/silver_eval.py` run with no code change.

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

**Done 2026-11-14:** `scripts/silver_eval.py` (gold and metrics in `tcr/silver.py`),
results in [results/silver/2026-11-14.md](../results/silver/2026-11-14.md); 3,230 dev
tasks, ~15 min on the Mac. Pooled over the five sets:

| method | nDCG@5 | hit@1 | judged@1 | rej@1 |
|---|---:|---:|---:|---:|
| M3.v2 | 0.337 | 0.437 | 1.00 | 0 |
| M1.v2 | 0.292 | 0.377 | 0.97 | 0 |
| B0.v2 | 0.262 | 0.354 | **0.69** | 0 |
| B1.v2 | 0.242 | 0.322 | 0.92 | 0 |
| M1.v1 | 0.206 | 0.262 | 0.97 | 0.031 |
| M3.v1 | 0.200 | 0.259 | 1.00 | 0.035 |
| B0.v1 | 0.174 | 0.222 | 0.64 | 0.059 |
| M2.v2 / v1 | 0.096 | 0.13 | 0.64–0.78 | 0–0.024 |

- **M3.v2 leads** on every set except Public-short, where M1.v2 has a higher nDCG@5
  (+0.022, 95% CI +0.002 to +0.043) at the same hit@1. Personal is the hardest set
  (M3.v2 hit@1 0.34; Public-short 0.66).
- **v2 beats v1 for every method**, by 0.10–0.14 nDCG@5. The judge's pool drew from both
  versions equally, so pool bias does not explain it, but the judge and v2 come from the
  same model family. icon-descriptions.md §7 decides on human labels.
- **The nearest parent adds nothing on silver** (M3.v2+parent: ΔnDCG@5 +0.000, 184 hit@1
  wins and 184 losses).
- **Rejected icons:** v1 methods put one at top-1 on 2–6% of tasks (B0 and B1 most); the
  serving filter adds 0.5–0.9 points of hit@1. v2 shows none.
- **The silver score cannot rank methods outside the judge's pool.** B0 was not a pool
  source: 31% of its top-1s are unjudged and count as misses, while its hit@1 on judged
  top-1s is the highest (0.51; partly selection, since the judged ones are where it
  agrees with the pool). The same will hold for F1 and I1. Before comparing them, judge
  their unjudged dev top-1s (public sets: free `gemini`; Personal: `claude-cli`), or
  report the bounds (all unjudged wrong … all right).
- M2 (nearest example on the title alone) is weak; in the pool it ran on title + body.

## 2. Fine-tune the dual encoder (the main bet)

Fine-tuning-plan §5 and §8 step 5: the step most likely to give the best fast model.

**Status 2026-11-17: round 2 done** (f1-dual-encoder.md "Round 2 results"): the higher LR
+0.023 nDCG@5 over F1@all, e5-base +0.014 more (F1@all-base.v2+parent 0.474 / hit@1 0.603),
full-catalogue negatives −0.02. The deployed model stays e5-small; e5-base and e5-large
become teachers to distil from: [next-data-and-distillation.md](next-data-and-distillation.md).

**Status 2026-11-16: round 1 done, F1 wins on silver; round 2 built, waiting for Kaggle.**
Results in [f1-dual-encoder.md](f1-dual-encoder.md) "Results"; plot in
[results/f1/learning_curve.png](../results/f1/learning_curve.png):

- F1@all.v2+parent: nDCG@5 0.436, hit@1 0.557 (upper bound 0.683) against M3.v2's
  0.337 / 0.437 (ΔnDCG@5 +0.100 [+0.090, +0.109], McNemar p < 0.001). The lower bound wins,
  so no grading was needed for the decision. 12.5% of F1's top-1s are unjudged.
- The learning curve is still rising: about +0.03 nDCG@5 per tripling of judged tasks
  (0 / 1k / 3k / 8.9k → 0.330 / 0.364 / 0.393 / 0.428). The free signal alone only
  matches M3.v2.
- Within a run, the gain comes early, but F1@all's best was its last evaluation: the LR
  decayed to 0 while validation still rose. Round 2 (f1-dual-encoder.md §10, hand-off
  [f1-kaggle-run.md](f1-kaggle-run.md)) tests a higher LR with more epochs, negatives
  from the whole catalogue, and e5-base as a teacher, on Kaggle's two T4s.
- Review page (private): `scripts/f1_review.py` → `data/private/f1/review.html`, F1@all's
  picks on 50 held-out pool tasks, chosen so they cannot anchor the author's labels of
  eval tasks.

**Status 2026-11-15: built, waiting for the Colab runs.** Design (data, loss, validation,
the fair comparison with M3.v2) in [f1-dual-encoder.md](f1-dual-encoder.md); the hand-off
for the runs in [f1-colab-run.md](f1-colab-run.md). Training moved to Colab: an MPS probe on
the Mac did not finish one step in 10 minutes and made the laptop unusable. Data built
(9,870 judged pool tasks → 26,423 training rows in EN/ES/RU + 2,928 validation rows, 75k
free-signal rows, 139 icons held out); the scripts passed a CPU smoke test. Where the list
below and that doc differ, the doc is what was built (e.g. a masked graded loss instead of
MNRL, the judge's relevant-but-ungraded icons masked rather than used as negatives).

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
- **Hardware:** expected the Mac to be enough; it was not (an MPS probe made it unusable),
  so all training runs on Kaggle (CLAUDE.md "Compute"); the script runs unchanged there
  (device auto-detected, data from one tarball).

## 3. Zero-shot image towers and fusion (§5.1 A)

- Text → glyph-image retrieval with SigLIP 2 (multilingual text tower) and jina-clip-v2,
  over the crisp 256 px render (`tcr.describe_v2.glyph_png`). Icon vectors precomputed;
  only the task is encoded at query time.
- Methods `I1` (image only) and `I1+M1` / `I1+F1` (weighted sum of similarities).
  Weight chosen on the silver leaderboard for now; re-checked on human dev labels.
- Expect modest zero-shot quality on abstract glyphs; the point is to measure what the
  image adds on top of the descriptions.

**Status 2026-11-17: SigLIP 2 base scored.** I1-sb alone: nDCG@5 0.168 (judged@1 only
0.44, so underestimated); fused with F1@all at weight 0.2: +0.012 pooled
[+0.009, +0.016], +0.001 on Personal. so400m glyph vectors exist, not scored; no
jina-clip-v2 vectors came back. Further image ideas: next-data-and-distillation.md backlog.

**Status 2026-11-16: built, glyph vectors waiting for the Kaggle run.** The work is split
by privacy:
- Kaggle encodes the 2,862 public glyph renders (`scripts/i1_embed.py`, shipped in the F1
  tarball, ~10 min on a T4; the Mac's CPU would need ~30 min under load).
- The Mac encodes only the dev queries, once per tower, cached in `.cache/i1/`.

Methods `I1-sb` (SigLIP 2 base), `I1-so` (SigLIP 2 so400m), `I1-jc` (jina-clip-v2) and
fusions `<text>+<I1>~<w>` (standardized scores, weight w), all in `tcr/methods/image.py`.
A 20-glyph test with SigLIP 2 base on the Mac picked sensible glyphs, including for
Russian and Spanish titles ("Позвонить маме" → call). Gemini Flash-Lite's free tier is not
needed here (no LLM reads the images). Its use would be grading unjudged top-1s on the
public dev sets.

## 4. Abstention gate, built now and calibrated later

- Features per task: top-1 score, margin to top-2, agreement between methods (F1, M3,
  I1), the icon's usefulness prior, query length.
- Prototype the gate on silver: "would the judge grade this top-1 ≥ 2?" gives a
  precision–coverage curve per method.
- The real threshold is set on human dev labels (their "no good icon" and misses), since
  the judge almost never abstains. Until then the curve shows which features carry signal.

## 5. GPU work (Kaggle; Colab ends long jobs)

The steps where the Mac is too slow:

- **Bigger encoders as teachers:** e5-base/large, bge-m3 or Qwen3-Embedding-0.6B
  fine-tuned the same way; distil into e5-small if they win.
- **VLM reranker** (§5.1 B) over the top 30, e.g. a 2–3B Qwen-VL-family model with LoRA,
  trained on the same judge labels; then distilled into the dual encoder (§5, order 3–4).
- Kaggle's free T4 ×2 (12 h per session, headless) is enough for e5-base and LoRA on a
  2B VLM at small batch: one queue per GPU ([f1-kaggle-run.md](f1-kaggle-run.md)). An A100
  is needed only for anything larger.
- **Distillation from the best teacher** (round 2's e5-base, a larger encoder or the
  reranker) into e5-small. The student learns the teacher's scores over all 2,862 icons on
  *unlabelled* text: the MS-LaTTE tasks not in any pool or eval set
  (`data/raw/MS-LaTTE.json`, public; eval titles removed with `contamination_hits`) and all
  75k free-signal queries. The learning curve says more training signal still helps, and
  this gets it without paying for more judging.
- **Learned per-icon vectors** (§5 "hybrid") on top of the best F1.
- **Grading unjudged top-1s** once F1 variants have to be told apart on silver (they share
  the ~12% unjudged gap). Public sets on the free `gemini` tier, Personal on `claude-cli`.

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

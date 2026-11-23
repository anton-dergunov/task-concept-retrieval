# A second stage for the big model: cross-encoder and late interaction (plan)

For the author and for the session that builds it. Status: **plan only (2026-11-23)**. It
builds on the fine-tuned dual encoder (F1, [f1-dual-encoder.md](f1-dual-encoder.md)), which
stays as the first stage. The run guide is [f1-kaggle-run.md](f1-kaggle-run.md). Deployment
limits are in [deployment.md](deployment.md) "Two models".

## 1. Why

The big model usually finds a right icon, and then often puts another one first. Silver dev
split, pooled over 3,230 tasks, plain gold (results/silver/2026-11-22):

| method | hit@1 | hit@3 | hit@5 | hit@10 | judge's #1 in top-10 |
|---|---:|---:|---:|---:|---:|
| F1@all-large-all3k.v2+parent | 0.650 | 0.840 | 0.895 | **0.945** | 0.683 |
| F1@all-kd-final.v2+parent | 0.608 | 0.802 | 0.864 | 0.930 | 0.648 |
| M3.v2 | 0.437 | 0.635 | 0.715 | 0.807 | 0.452 |

A perfect reorder of the big model's top 10 would lift hit@1 from 0.65 to 0.945. The model
is a dual encoder: it turns the task and each icon into one vector each, before it sees
the other. It cannot read the task and the icon together, nor weigh icons against each
other, and the judge does both. The noise ceiling (argon-experiments.md "Batch 1 results")
puts the judge itself at nDCG@5 0.888 against the model's 0.502. That number is optimistic,
since the judge is compared with itself, but the gap is real.

The second stage reads the task together with each of the first stage's top K icons and
reorders them. Two designs follow. The cross-encoder is the strong, costly one. Late
interaction is the cheap one, which could also replace the first stage.

## 2. Design A: cross-encoder reranker

- **Input:** `query: <task text, with its parent heading>` and the icon document
  (`tcr.finetune.icon_doc`, the same text F1 encodes), one pair per candidate. It outputs
  one relevance score per pair.
- **Candidates:** the big model's top K, with K = 10 at first. K = 20 is the experiment: it
  is slower, and hit@20 is still unmeasured, since the stored rankings keep a top 10.
- **Base model:** it must be multilingual (EN/ES/RU, plus the other languages of the
  translated set). Candidates, smallest first:
  - an mMiniLM cross-encoder (~118M parameters);
  - multilingual-e5-small or -base, trained as a cross-encoder from its encoder weights;
  - bge-reranker-v2-m3 (568M).

  Check each one's licence before use. Some multilingual rerankers are licensed for
  non-commercial use only.
- **Training:** one list per training text. The candidates are the first stage's top K
  from the teacher scores that `train_f1.py --score` already writes (`teacher/<run>.npz`,
  top 64 per text), plus the icons the judge graded.
  - **Targets** are the judge's grades: 3, 2, 1, and 0 for icons it saw and left out.
    Icons it never saw are masked out of the loss, or given the first stage's score as a
    soft target. Which is better is one of the experiments.
  - **Loss:** a listwise softmax cross-entropy over the list with graded targets (ListNet).
    A pointwise head (binary, grade ≥ 2) gives a calibrated probability for the gate.
  - The first stage is frozen; only the reranker trains. Its validation is F1's validation
    split (`split == "val"`).
- **Final score:** the reranker's score, optionally blended with the first stage's (one
  weight, fitted on dev). The abstention gate is refitted on the reranker's probability.
- **Latency:** this is the risk. The big model's encoder already takes ~120–200 ms per task
  on the M1, and the budget is ~200 ms, up to 300–400 ms. K pairs of ~100–250 tokens each
  must fit in what is left. Measure first (§5, step 0): an untrained model of each size, in
  ONNX, on the M1, K = 10 and 20, with the icon document cut to 128 and 256 tokens.
- **Where it can fail:** it reorders only what the first stage found (5.5% of tasks have
  no good icon in the top 10). A reranker trained on one first stage's candidates may need
  retraining when that first stage changes.

## 3. Design B: late interaction (ColBERT-style)

- **Model:** the F1 encoder keeps one vector per token instead of mean-pooling them, through
  a linear projection to ~128 dimensions. The score of a pair is the sum over query tokens
  of the best-matching icon token (MaxSim). The icon's token vectors are computed once, at
  export, as the icon vectors are now.
- **Cost:** one query encoder pass, as now, plus MaxSim over the candidates. That is
  milliseconds for a top 50, or for all 4,257 icons. Memory: 4,257 icons × ≤ 256 tokens ×
  128 dimensions in fp16 is ≤ 280 MB, less after trimming padding and stop tokens. That is
  acceptable on the Mac, too much for the small model on Android or iPad.
- **Training:** start from an F1 checkpoint, with the same rows, contrastive loss and hard
  negatives as train_f1.py, plus the projection. Optionally distil the first stage's or the
  cross-encoder's scores. PyLate (sentence-transformers based) has ColBERT training and
  MaxSim. Check its licence and its fit with train_f1.py's data first; a MaxSim head in
  train_f1.py may be simpler.
- **Use:** either rerank the dual encoder's top 50, or replace the dual encoder outright
  (one model, retrieval by MaxSim over every icon).
- **Expected gain:** between the dual encoder and the cross-encoder. Token-level matching
  helps with the central object ("keys", "flight") but still does not let icons compete
  with each other.

## 4. Data: no new Argon labels needed to train, some to measure

**Training** needs nothing new. Every training row in `queries.jsonl` already has graded
icons. The judge-seen icons give graded negatives, and `--score` gives each text its
first-stage candidates. Rewrites, translations and the minimal-pair rows (`mp-*`) carry
grades too.

Argon data that would help but is not required:
- **#5 (VLM reranker, batch 2):** Argon ranks the big model's top 20 for ~2,000 training
  tasks. That is complete listwise labels on exactly the lists a reranker sees, so it is
  the best training data this plan could get. It is also a teacher to distil.
- **#3's contrast set** (scripts/contrast_eval.py) tests whether reading the pair together
  helps where a word changes the icon.

**Validation** has a hole, and it matters more for a reranker than it did for F1. Gold
grades only the icons the judge saw. After g1–g4, the big model's top 10 still holds **2.5
ungraded icons per dev task on average**: 2,622 of 3,230 tasks hold at least one, ~8,000
distinct (task, icon) pairs. A reranker that moves one of them to first place scores a miss
on silver, even when it is right, so silver would understate the second stage.

The fix is to grade them, as g1–g4 did for top-1s, with the same tool at depth 10:

```bash
python scripts/grade_unjudged.py prepare --name g5 --depth 10 --skip g1,g2,g3,g4 \
    --methods F1@all-large-all3k.v2+parent,F1@all-kd-final.v2+parent
```

That is ~8,000 icons over ~2,600 tasks. g1 cost ~$0.003 per icon, so **~$10–25**, and it
needs Argon, so **before 2026-11-30**: add it to batch 2 (v3b), within round 6's $76.11.
Afterwards, score with `silver_eval.py --extra-grades g1,g2,g3,g4,g5`. Without it, report
`hit@1 judged` and `hit@1 ub` next to hit@1 and treat small gains as unproven. The test
split stays untouched until the final numbers.

## 5. Steps

0. **Latency on the M1 (Mac, ~half a day, no training).** Cross-encoders of each candidate
   size in ONNX fp16/int8, K = 10/20, two document lengths. MaxSim with a random
   projection over the top 50 and over all icons. This decides which base model is
   allowed at all.
1. **Data (Mac).** `f1_data.py build` adds, per training row, the first stage's top K
   candidates from the teacher scores. Then measure hit@20 of the big model on dev (a
   20-deep ranking via `silver_eval.py`'s TOP_K).
2. **Cross-encoder (Kaggle).** The base model chosen in step 0, ListNet + a pointwise head,
   unjudged icons masked; a variant with soft first-stage targets; then K = 20.
3. **Late interaction (Kaggle).** From F1@all-large-all3k: rerank the top 50, then full
   retrieval.
4. **Score (Mac).** Silver with g5, the contrast set, latency and memory per method. The
   gate refitted on silver dev.
5. **If it wins:** the export path in deployment.md (two ONNX graphs, or one with MaxSim),
   only when the author asks.

## 6. Open questions

- How much of the gap is reordering (this plan) and how much is retrieval (the 5.5% with
  no good icon in the top 10)? hit@20 answers part of it.
- Does a reranker learn Argon's taste better than the author's? The author's own labels
  stay the check (argon-experiments.md "Not on the list").
- Can the small model use any of this? Probably not: the phone's limit is memory, and both
  designs add to it. The small model keeps the dual encoder.

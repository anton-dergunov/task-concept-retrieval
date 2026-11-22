# Deploying a matcher to the agenda

How a method from this repository reaches the agenda in
[agentic-org-planner](https://github.com/anton-dergunov/agentic-org-planner) without taking this repository, torch or
sentence-transformers with it.

## The bundle

`scripts/export_bundle.py` writes a **bundle**: a directory of precomputed icon data plus
a manifest. The agenda's runner (`scripts/org_task_icon_matcher.py` in agentic-org-planner)
needs only `onnxruntime`, `tokenizers` and `numpy` to score with it.

| File | Holds |
|---|---|
| `manifest.json` | format, `tag` (the agenda's cache identity), the ONNX encoder and tokenizer (URL + sha256, downloaded on first use), which task fields the model reads (`query.inputs`) and how they are normalized, fusion and gate settings |
| `names.txt` | icon names; row *i* of every matrix is icon *i* |
| `dense.npy` | M1 collapsed into one matrix (fp16): Σ_f w_f·cos(q, e_f)/Σw is linear in the normalized query, so it equals `D @ q` |
| `prior.npy` | the additive quality prior |
| `bm25.npz` | B1 as a sparse vocab × icons matrix: Okapi's per-(term, icon) weight does not depend on the query, so a query's score is the sum of its tokens' rows |

The M3 bundle is about 4 MB. The encoder is downloaded separately (235 MB).

## Exporting

```bash
.venv/bin/python scripts/export_bundle.py --method M3 \
    --out ../../products/agentic-org-planner/icons/task-matcher --check
```

`--check` compares the bundle with the in-repo method on the Realistic tasks. `M3x` in
`tcr.cli` / the web UI scores with the bundle in `TCR_BUNDLE` (default `.cache/bundle`).
It is not a method for `tcr.eval`: the bundle indexes `example_tasks`, which the bootstrap
eval holds out as queries.

## What the export has to get right

- **Encoder precision.** The icon side is encoded with the same ONNX encoder the runner
  uses. With the int8 export the bundle agreed with M3 on only 80% of top-1 picks. e5
  packs icon scores tightly (median top-1/top-2 margin 0.003), and int8's noise (query
  cosine 0.996 to fp32) reorders those near-ties. fp16 and fp32 both agree on 99.7%, so
  the bundle uses fp16.
- **No batching.** With dynamic int8 quantization, a text's vector depends on its batch
  mates: activations are scaled over the whole input tensor. Both sides encode one text
  per call, so this cannot arise even if a quantized encoder is used again.
- **Graph optimizations.** onnxruntime 1.27's layer-norm fusion fails to load the fp16
  encoder. Both sides use `ORT_ENABLE_BASIC`, so they also run the identical graph. The
  runner (onnxruntime 1.27) and `BundleMatcher` (1.30) give identical answers on all 623
  Realistic tasks.
- **Query text.** The agenda sends the task title with its org markup (keyword, priority
  and tags removed), plus the parents and body when `query.inputs` lists them.
  `compose_query` / `normalize` build the matcher's text from those, and the runner mirrors
  them. A model trained on markup would declare `"normalize": "none"`.

## Two models (author, 2026-11-22)

Until 2026-11-22 the plan was to ship one model, e5-small. The plan now has two models with
different budgets. Both should be as good as possible; the big one comes first.

| | big model | small model |
|---|---|---|
| where | the author's MacBook Air M1 16 GB, Emacs agenda | Android phone / iPad |
| use | background prediction while the agenda renders: ~20 tasks per view, usually only a couple new (the rest cached) | rare fallback, when the big model has not given a task an icon yet |
| budget | ~200 ms per task, up to 300–400 ms; load time is offset; must not burden a busy Mac | memory first, then size and speed |
| candidate now | **F1@all-large-all3k** (e5-large, 560M) | **F1@all-kd-final** (e5-small, 118M) |
| silver nDCG@5 / hit@1 (g1–g4, +parent; f1-dual-encoder.md §13) | 0.567 / 0.714 | 0.534 / 0.659 |
| translated set, nDCG@5 in 7 languages | 0.568–0.594 | 0.470–0.511 |

**Measured on the Mac** (2026-11-22; PyTorch fp32 on CPU, one title + parent per call after
warm-up, 60 Realistic tasks, the checkpoint's icon vectors precomputed; the Mac was busy,
load average ~15):

| model | threads | median ms | p95 ms | load s | peak RSS |
|---|---:|---:|---:|---:|---:|
| e5-large (F1@all-large-all3k) | 4 | 98 | 114 | 3.7 | 1.9 GB |
| e5-large | 2 | 103 | 348 | 4.3 | 2.0 GB |
| e5-small (F1@all-kd-final) | 4 | 9 | 11 | 4.7 | 0.9 GB |
| e5-small | 2 | 9 | 10 | 4.1 | 0.9 GB |

e5-large fits the budget about 2–3× over. On 2 threads its median stays ~100 ms, but under
load the tail grows (p95 348 ms), so the runner should keep 4 threads, or encode new tasks
one at a time at low priority. The remaining ~100–300 ms could pay for a second stage,
e.g. a small cross-encoder reranking the top-k (fine-tuning-plan.md "Two stages").

What each export still has to check:
- **Big model:** fp16 ONNX ("Encoder precision" above: int8 reorders near-ties) roughly
  halves its 2.2 GB of weights; measure latency and RSS of the exported encoder, not
  PyTorch's. Load time is ~4 s from a warm disk cache; the agenda's runner loads once per
  process, so a long-lived helper process would hide it.
- **Small model:** memory and download size first. Vocabulary trimming (backlog of
  next-data-and-distillation.md: e5-small 118M → ~35M with EN/ES/RU tokens) conflicts with
  the many-languages goal, so a mobile bundle would keep the full vocabulary or offer
  per-language-set variants. The on-device runtime (ONNX Runtime Mobile, Core ML) is still
  to choose.

## A new model

Any method that fits the same shape (text encoder + precomputed icon matrix, optionally
BM25 fused by RRF) exports as a new bundle. That includes a fine-tuned dual encoder,
exported to ONNX and hosted where the manifest's URL points. Copying it into the agenda
changes the `tag`, which discards the agenda's cached icons.

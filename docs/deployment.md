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

## A new model

Any method that fits the same shape (text encoder + precomputed icon matrix, optionally
BM25 fused by RRF) exports as a new bundle. That includes a fine-tuned dual encoder,
exported to ONNX and hosted where the manifest's URL points. Copying it into the agenda
changes the `tag`, which discards the agenda's cached icons.

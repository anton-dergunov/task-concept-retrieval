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

The M3 bundle is about 4 MB. The encoder is downloaded separately (235 MB). An F1 bundle
(fine-tuned dual encoder, "F1 bundles" below) is dense-only: no `bm25.npz`, a zero prior.

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
| candidate now | **F1@all-large-all3k** (e5-large, 560M; deployed 2026-11-22) | **F1@all-kd-final** (e5-small, 118M; marked for Rumbo) |
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

### Deployed (2026-11-22)

**Big model: F1@all-large-all3k is the agenda's matcher**, bundle tag `F1-df69a05fab6d`, in
agentic-org-planner's `icons/task-matcher/` (5.9 MB). The encoder is fp16 ONNX (1.1 GB) and
is published as an interim model (below). On the Mac, under load ~19:

| check | result |
|---|---|
| bundle vs stored PyTorch rankings, Realistic dev (315) | top-1 100.0%, \|Δz\| median 0.001; silver unchanged (0.562 / 0.698) |
| the agenda's runner (Homebrew python3, onnxruntime 1.27) vs `BundleMatcher` (1.30), all 623 Realistic tasks | 623 / 623 identical |
| `BundleMatcher`, one text per call | p50 119 ms, p95 201 ms |
| the runner, one task (load + encode) | 4.9 s, max RSS 1.8 GB |
| the runner, 623 tasks in one process | 132 s (≈ 210 ms per task), max RSS 2.6 GB |
| the export itself | ~1.5 min, peak memory 7.4 GB |

The agenda sends a couple of new tasks per render to a short-lived process, so the
load time dominates and the RSS is held only for those seconds. The gate is deliberately
low (the author wants to see what F1 picks): it shows the top-1 when P(acceptable) ≥ 0.2,
which on silver dev is 100% of tasks (precision 71.4%); see "F1 bundles".

**Small model: F1@all-kd-final, marked for Rumbo** (plan-viewer `docs/plans/NEXT.md`,
session 7): the fallback for tasks Emacs has not given an icon yet. Exported to
`.cache/bundles/F1-all-kd-final-{fp16,int8}/`, the encoder in `.cache/onnx/F1-all-kd-final/`:

| precision | size | top-1 vs PyTorch (2,208 dev tasks) | silver nDCG@5 / hit@1 | p50 on the Mac |
|---|---:|---:|---:|---:|
| PyTorch fp32 (reference) | 470 MB | — | 0.526 / 0.653 | |
| fp16 | 235 MB | 99.9% | 0.526 / 0.653 | |
| int8 (dynamic) | 118 MB | 70.1% | 0.498 / 0.615 | 4.7 ms |

int8's silver drop is overstated a little, since its new picks are more often unjudged (counted
as misses), but it is real: dynamic int8 reorders e5's near-ties ("Encoder precision" above).
Still to settle for the small model: the on-device runtime and its fp16 support, and
vocabulary trimming (backlog of next-data-and-distillation.md: 118M → ~35M with EN/ES/RU
tokens, which conflicts with the many-languages goal).

## F1 bundles

```bash
nice .venv/bin/python scripts/export_bundle.py --method F1 --run all-large-all3k \
    --out ../../products/agentic-org-planner/icons/task-matcher --check
nice .venv/bin/python scripts/export_bundle.py --method F1 --run all-kd-final --precision int8 \
    --out .cache/bundles/F1-all-kd-final-int8 --check --dataset realistic,personal,public_short
```

- **Encoder:** the checkpoint's transformer is exported to ONNX (`last_hidden_state`, as
  Xenova's export, so the runner is unchanged), fp32 first, then converted to fp16 or int8.
  It goes in `.cache/onnx/F1-<run>/`, the folder that is uploaded to Hugging Face.
  The files are hard-linked where `fetch` and the agenda runner look for them
  (`<sha256[:16]>-<file>`), so neither downloads a file that is not uploaded yet.
- **Tokenizer:** the checkpoint's `tokenizer.json` keeps training's padding (to a multiple
  of 8). The runner feeds every id with attention mask 1, so pads were mean-pooled into the
  vector, and top-1 agreement fell to 78% even with fp32. The export ships it with padding
  and truncation off, edited as JSON, because `tokenizers`' own save is not byte-stable.
- **Icon side:** FineTunedMatcher's matrix (fp32 PyTorch vectors, mostly from the GPU), as
  fp16. F1 has no prior and no BM25, so the bundle is dense-only.
- **Query:** title + nearest parent (`query.inputs`), with `max_length` 64, as in training.
- **Gate:** silver's hit@1 of the shown top-1 (judge grade ≥ 2, extra grades g1–g4) is
  fitted against the stored z-signal of `F1@<run>.v2+parent` on the dev split (Platt). The
  threshold is the z where P(acceptable) = `--min-prob` (default 0.2). The export prints
  coverage and precision at other values:

  | P ≥ | e5-large z ≥ | shown | precision | e5-small z ≥ | shown | precision |
  |---:|---:|---:|---:|---:|---:|---:|
  | 0.2 | 3.16 | 100.0% | 71.4% | 3.00 | 99.9% | 66.0% |
  | 0.3 | 3.64 | 99.7% | 71.5% | 3.49 | 99.5% | 66.1% |
  | 0.5 | 4.40 | 90.2% | 74.6% | 4.26 | 83.0% | 71.6% |
  | 0.7 | 5.15 | 56.5% | 82.2% | 5.03 | 40.9% | 80.3% |

  The real threshold waits for human labels (next-label-free-work.md §4).
- **Tag:** the encoder files' URLs are not part of the tag, because their sha256 already names
  them, so pinning a Hugging Face revision keeps the agenda's cached icons.

## Publishing the encoder (Hugging Face)

The author uploads the files. The repos are public and marked interim in their model cards
(`README.md` in each upload folder).

```bash
pip install -U huggingface_hub            # the `hf` CLI
hf auth login                             # a *write* token from https://huggingface.co/settings/tokens
hf repos create AntonDergunov/task-icon-encoder-e5-large-interim --public
hf upload AntonDergunov/task-icon-encoder-e5-large-interim .cache/onnx/F1-all-large-all3k . \
    --commit-message "Interim F1@all-large-all3k encoder (fp16 ONNX) and bundle"
hf repos create AntonDergunov/task-icon-encoder-e5-small-interim --public
hf upload AntonDergunov/task-icon-encoder-e5-small-interim .cache/onnx/F1-all-kd-final . \
    --commit-message "Interim F1@all-kd-final encoder (fp16 and int8 ONNX) and bundles"
python3 -c "from huggingface_hub import HfApi; a = HfApi(); \
  [print(r, a.model_info(r).sha) for r in ('AntonDergunov/task-icon-encoder-e5-large-interim', \
   'AntonDergunov/task-icon-encoder-e5-small-interim')]"
```

The manifests' URLs then pin the revision: re-export with `--revision <sha>` (the ONNX is
reused, the tag stays the same). A HEAD request on each URL checks the file:
`curl -sIL <url> | grep -i x-linked-etag` shows its sha256. A different HF user name
needs `--hf-repo`.

## A new model

Any method that fits the same shape (text encoder + precomputed icon matrix, optionally
BM25 fused by RRF) exports as a new bundle. That includes a fine-tuned dual encoder,
exported to ONNX and hosted where the manifest's URL points. Copying it into the agenda
changes the `tag`, which discards the agenda's cached icons.

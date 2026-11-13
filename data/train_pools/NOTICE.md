# Training pools: sources and licenses

Unlabelled tasks for the LLM judge to label, used only for training
(docs/fine-tuning-plan.md §4). None of them is in an eval dataset: they are disjoint by
source id, and every title passed the contamination check (`tcr.describe_v2.contamination_hits`:
no exact or e5 cosine ≥ 0.95 match with any eval title). Same JSONL schema as
`data/datasets/` (see its NOTICE.md).

| File | Contents | Source and license |
|---|---|---|
| `mslatte.jsonl` | 3,000 MS-LaTTE task titles that no public dataset uses, cleaned as in Public-short; the list name as parent heading when informative | MS-LaTTE (Microsoft), MIT License, Copyright (c) Microsoft Corporation; license text in `data/datasets/LICENSE-MS-LaTTE.txt`. `meta.source_id` is the original `ID` |

Built by `scripts/datasets/build_train_pools.py`. The Personal-style pool
(`scripts/datasets/synth_personal.py pool`) is generated from the private task list's
abstracted topics and stays in `data/private/train_pools/` until it is audited like
Personal-synth.

# Task Concept Retrieval

Given a short task description, retrieve the one icon that conveys what the task is about, or
return nothing when no icon is good enough.

```
$ echo '["Save money for vacation", "Walk at least 8k steps daily", "Caminar 8000 pasos al día",
         "Renew passport before the trip", "asdf qwerty zxcv"]' | python -m tcr.cli
{
  "Save money for vacation": "savings",
  "Walk at least 8k steps daily": "run_circle",
  "Caminar 8000 pasos al día": "run_circle",
  "Renew passport before the trip": "passport",
  "asdf qwerty zxcv": null
}
```

The icons are the 4,257 [Material Symbols](https://fonts.google.com/icons). The matcher exists to
put a glanceable icon next to each task in a planner's agenda
([agentic-org-planner](https://github.com/anton-dergunov/agentic-org-planner)), and this
repository is the test bed where the matching is built and measured.

## What makes it a retrieval problem worth studying

- **The icon names are not used.** Material Symbols' names, tags and categories were written for
  interface designers, not for tasks. Every icon is matched on a generated description instead:
  a vision model looked at each rendered glyph, without its name, and wrote what it shows, which
  task intents it suits, example tasks, and tasks it should *not* be used for.
- **No LLM on the query path.** The agenda shows icons while it renders, so matching must be
  local and fast. Language models build the data and judge results; they never answer a query.
- **Abstaining is a correct answer.** The agenda shows the top icon automatically, so a wrong
  icon is worse than none. Every method is wrapped in a gate that returns the top icon or nothing.
- **Tasks are multilingual.** English, Spanish and Russian, so the default encoder is
  multilingual.

## Methods

| Method | What it does |
|---|---|
| B0 | The matcher this project started from: one concatenated description per icon, an English bi-encoder, a fixed threshold |
| B1 | Lexical: BM25 over the icon descriptions |
| M1 | Dense bi-encoder with per-field weights (visual concepts, task intents, example tasks) and a prior on how useful the icon is |
| M2 | Nearest-example voting: the query is matched against each icon's example tasks, then scores are aggregated per icon |
| M3 | Hybrid: reciprocal rank fusion of the lexical and dense methods. The default |

The abstention gate, the other methods under consideration and the reasoning behind them are in
[docs/matching-methods.md](docs/matching-methods.md).

## Data

The icon descriptions (`data/icon_descriptions/`, one JSON per icon) are the only matching
signal. How the first generation was produced, what is measurably wrong with it, and the design
of the second generation are in [docs/icon-descriptions-v2.md](docs/icon-descriptions-v2.md).

Task datasets, all in one JSONL schema (`data/datasets/`, licences in
[NOTICE.md](data/datasets/NOTICE.md)):

| Dataset | Contents |
|---|---|
| Realistic | 623 tasks from the planner's sample files |
| Personal-synth | 1,506 tasks generated from abstracted topics of my own task list and audited for leakage ([docs/anonymization.md](docs/anonymization.md)) |
| Public-short | 600 real to-do titles from MS-LaTTE and 225 MASSIVE items rewritten as tasks, in English, Spanish and Russian |
| Public-expanded | 500 MS-LaTTE concepts expanded into longer tasks with a body, in the same three languages |

My own task list is the primary evaluation set and is not in the repository.

## Status

Working: the icon catalogue and rendered glyphs, the first-generation descriptions, the five
methods above with the abstention gate, a command-line and web search, a labelling tool that
runs on a phone or tablet ([labeller/](labeller/README.md)), and the export of a method as a
standalone bundle for the planner ([docs/deployment.md](docs/deployment.md)).

There are no headline numbers yet, on purpose. The current evaluation (`python -m tcr.eval`)
is a bootstrap that holds out generated example tasks as queries; it is useful for comparing
methods and wiring the metrics, not for reporting quality. Reported results wait for human
labels.

Next:

1. Gold labels on the public datasets and on my own tasks, collected with the labelling tool.
2. The second generation of icon descriptions, compared against the first on those labels.
3. A benchmark of the methods on the gold set, then a fine-tuned matcher
   ([docs/fine-tuning-plan.md](docs/fine-tuning-plan.md)).

The experimental design, including the LLM judge, calibration and the dev/test split, is in
[docs/experimentation-strategy.md](docs/experimentation-strategy.md).

## Running it

Python 3.11 or later.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt

# one query, with the ranked shortlist
.venv/bin/python -m tcr.cli search "save money for vacation"

# a JSON array of tasks on stdin -> {task: icon or null}
echo '["Walk at least 8k steps daily", "Caminar 8000 pasos al día"]' | .venv/bin/python -m tcr.cli

# web interface with the rendered icons
.venv/bin/python -m tcr.cli serve

# compare methods on the bootstrap evaluation
.venv/bin/python -m tcr.eval
```

`--method {B0,B1,M1,M2,M3}` selects the method. The first query downloads the encoder.

## Layout

```
tcr/        the matcher: methods, abstention gate, search, evaluation, bundle export
scripts/    the data pipeline: catalogue, glyph rendering, icon descriptions, datasets
labeller/   the labelling tool
data/       icons, their descriptions, task datasets
docs/       design documents
results/    evaluation runs
```

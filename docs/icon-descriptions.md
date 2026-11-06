# Icon descriptions: v1 and v2

The icon descriptions are the **only matching signal**. Matchers never read an icon's
name, tags or categories (CLAUDE.md hard rule 1); they read text that a vision model wrote
while looking at the glyph alone. The descriptions feed every matcher, the labelling
tool's search, and the training data: example tasks are positives, near-misses are
negatives.

Two versions exist side by side, so description quality is an experimental variable:

| | v1 | v2 |
|---|---|---|
| Files | `data/icon_descriptions/` (4,257) | `data/icon_descriptions_v2/` (3,958: 3,619 glyphs + alias copies) |
| Made by | `scripts/describe_icons.py`, mostly Gemini 3.1 Flash-Lite, free tier | Gemini 4 Argon in three passes (§5) |
| Kept for matching | 3,486 glyphs | 2,862 glyphs |
| Load with | `load_icons(version="v1")` (the default) | `load_icons(version="v2")`, or `TCR_DESC=v2` |

`tcr/data.py` maps both onto the same field views (§3.2), so every matcher, the eval
(`python -m tcr.eval --desc v1|v2`) and the bundle export run on either. v1 stays the
default until v2 wins on human labels (§7).

## 1. v1 and what was wrong with it

`scripts/describe_icons.py` sent one rendered 128 px glyph per call, with no name, and
asked for this JSON:

```jsonc
{
  "icon_usefulness": 8,       // 0–10 fitness for task labelling
  "discard": false,           // true → unusable
  "visual_concepts": [...],   // 5–15 visual ideas
  "task_intents": [...],      // 5–15 intents the icon could mark
  "example_tasks": [...],     // 10–20 task titles
  "poor_matches": [...],      // 5–10 tasks that should not use this icon
  "reasoning": "…"
}
```

About 80% (3,422) came from Gemini 3.1 Flash-Lite on the free tier, 494 from untagged
early runs, and the rest from 2.5 Flash-Lite, 2.5 Flash and 3.5 Flash. Every font alias
was described separately. Measured over the 3,798 non-discarded files:

| Problem | Measurement | Consequence |
|---|---|---|
| **Glyph misreadings** | Spot check of 10 Flash-Lite icons, re-described by 3.1 Pro: **2 misread** (a bathroom scale read as a chat bubble; settings sliders read as "comparison"), 2 weaker | A misread icon cannot be found by matching or by search |
| **Short, generic examples** | Median **4 words** (p90 6); the author's real titles have median 9 (p90 13) | Trains on a regime unlike real tasks |
| **Examples reused across icons** | 6,438 strings appear for more than one icon ("optimize database query performance": 75 icons) | One task is a positive for dozens of icons |
| **Generic negatives** | The 20 most common strings are **35% of all negatives** ("buy groceries" for 1,353 icons); most were copied from the prompt's own sample titles | They teach nothing a random negative would not |
| **Prompt anchoring** | The prompt's sample titles recur hundreds of times | Topic skew toward knowledge work |
| **One-word visual concepts** | Median 1 word ("focus", "document", "alert") | Mixes what is drawn with what it means; modifiers (a plus, a slash, a badge) are not tied to a meaning |
| **Terse intents** | Median 2 words ("measure", "configure") | Too coarse to separate neighbouring icons |
| **Usefulness poorly spread** | 1,985 icons at 8, 851 at 7, 435 at 4 | Effectively three levels; `discard` hard to audit |
| **No notion of look-alikes** | Each icon described in isolation | Families (alarm variants, signal bars) get near-identical text |
| **English only** | — | ES/RU tasks rely entirely on the multilingual encoder |

The `reasoning` field was the most reliable part (median 189 characters, usually sensible).

## 2. v2 design principles

0. **State the purpose and use the selection principles.** The prompt opens with what the
   icon is for, a glanceable reminder of the one thing a task is about, and the salience
   order from [icon-selection-principles.md](icon-selection-principles.md): central
   object → context or activity → meaningful action → never properties. Examples are
   tasks whose **central subject** the icon shows.
1. **Describe before interpreting.** First what is literally drawn, then what it means:
   chain-of-thought for vision, aimed at the misreadings.
2. **Separate the glyph from its meanings.** A base object plus modifiers, each tied to
   a meaning (plus = add, slash = off or cancelled, clock = scheduled, check = done).
3. **General first, personal lightly.** The descriptions are meant for release, so
   examples span ordinary life areas. Personal relevance enters only through which areas
   and title styles are represented, never through private text (§4).
4. **Titles, not bodies.** A title says what a task is; icons represent the what.
5. **Hard negatives must be hard.** A near-miss shares a keyword or theme with the icon
   but deserves a different icon, and names the concept that fits better.
6. **Contrast with look-alikes.** Near-identical glyphs need text that tells them apart.
7. **Calibrated scores** from anchored rubrics, split into separate factors.
8. **No evaluation contamination** (§4).

## 3. The v2 schema

### 3.1 Fields

```jsonc
{
  "schema_version": 2,
  "depiction": {
    "primary_object": "bathroom scale with a small display",   // literal, 3–10 words
    "modifiers": ["three dots on the display"],                // badges, overlays, arrows, slashes
    "composition": "single object, front view"
  },
  "readings": [                               // what a viewer would take it to mean, strongest first
    {"meaning": "weighing yourself / body weight", "strength": "primary"},
    {"meaning": "weighing an object, measuring mass", "strength": "secondary"}
  ],
  "ambiguity": "At small sizes the display dots can make it look like a chat bubble.",
  "modifier_semantics": [],                   // [{"modifier": "plus badge", "meaning": "add / new"}]
  "domains": ["health_fitness", "home", "shopping"],
  "task_intents": ["track body weight over time", "weigh luggage before a flight"],
  "example_tasks": [                          // rank-1 positives: the icon shows the CENTRAL subject
    {"task": "Weigh myself every Monday and log it", "parent": "", "domain": "health_fitness"},
    {"task": "Compare Withings vs Garmin", "parent": "Smart Scale", "domain": "shopping"}
  ],
  "example_tasks_es": ["…"], "example_tasks_ru": ["…"],   // 3 each, written natively
  "context_tasks": [                          // rank-2 positives: the icon fits as the CONTEXT
    {"task": "Log calories for the cutting phase", "central": "food / nutrition"}
  ],
  "near_misses": [                            // hard negatives
    {"task": "Reply to the group chat about Saturday", "better": "chat bubble", "trap": "lookalike_glyph"}
  ],
  "distinguishing": "…",                      // pass 2 only: what sets it apart from its look-alikes
  "scores": {"recognizability": 4, "task_fit": 4, "distinctiveness": 3},   // 1–5, anchored
  "discard": false,
  "reasoning": "Clear scale metaphor; …",
  "_meta": {…}                                // provenance, see §5.5
}
```

- **Sizes asked for:** readings 1–4; task_intents 6–12; exactly 16 example tasks (5 of
  2–5 words, 8 of 6–12, 3 of 13–16; 4 with a parent heading that supplies the subject a
  generic title lacks; at least 3 domains); 3 ES + 3 RU examples; context_tasks 3–6;
  near_misses 5–8 using at least 3 traps. Vertex rejects array and integer bounds in the
  response schema, so the counts live in the prompt and are checked afterwards
  (`soft_problems()`); departures are recorded, not retried.
- **Near-miss traps**, following the selection principles: the concept appears only as a
  **property** (a rain icon for "waterproof headlamp"), as a **generic verb** (a check
  mark for "verify git sync"), as **shared vocabulary** with another subject, as a
  **look-alike glyph**, or as the **parent's topic** when a specific title points
  elsewhere.
- **Domains** (fixed): work, career, learning, tech, health_fitness, finance, home,
  shopping, food, social_family, travel, hobbies, admin, communication, creative,
  outdoors. They are facets for balanced generation, per-domain evaluation and
  stratified sampling.
- **Score anchors (1–5):** recognizability: 1 = most people would guess wrong, 3 = clear
  in context, 5 = universally obvious. task_fit: 1 = almost no to-do tasks, 3 = one narrow
  family, 5 = many common tasks across domains. distinctiveness: 1 = interchangeable with
  a look-alike, 5 = unique.

### 3.2 How the matchers read v2

`tcr.data._v2_doc` maps v2 onto the `IconDoc` views that every matcher uses:

| View | v1 | v2 |
|---|---|---|
| `visual_concepts` | visual_concepts | depiction: primary object + modifiers |
| `readings` | (empty) | reading meanings |
| `task_intents` | task_intents | task_intents |
| `example_tasks` | example_tasks | EN tasks (`"Parent: task"` when nested, the same format as a query with its parent) + ES + RU |
| `poor_matches` | poor_matches | near-miss tasks |
| `reasoning` | reasoning | reasoning |
| usefulness prior | icon_usefulness / 10 | (recognizability + task_fit) / 10, a placeholder until fitted on labels |
| discard | discard | discard, unless the manual review calls the glyph useful or somewhat |

M1's field weights are per version (`config.field_weights`): v2 adds `readings` at 1.0,
because v2 splits what is drawn (weight 0.5) from what it means. `context_tasks`,
`scores`, `ambiguity` and `distinguishing` are loaded but not matched on yet; they are
meant for graded training labels and for the abstention gate (an ambiguous icon should
need more confidence). Embedding caches are keyed on the encoded texts, so v1 and v2
never share vectors.

## 4. Conditioning and the contamination guard

**Style references.** Each request carries 8 rotating reference titles, seeded by the
glyph's hash, so no fixed list anchors the run (v1's main source of reuse). They come from
a training-only pool: 5 short MS-LaTTE titles that no public dataset uses, and 3 long
style exemplars. Anything equal to an eval title is excluded.

**Never in a prompt:** task bodies, private tasks, and any title from an eval dataset
(Personal, Realistic, Personal-synth, Public-short, Public-expanded).

**Guard** (`scripts/v2_guard.py`). After every pass, every generated task string
(example, ES/RU, context and near-miss tasks) that equals an eval title after
normalization, or lies within multilingual-e5 cosine 0.95 of one, is dropped. Eval tasks
in the training text would inflate every later result. The eval sets include the private
Personal list, compared locally only, so the files record only how many strings were
dropped (`_meta.guard`), never the text or the matched title. Over all passes it dropped
0.4% of the strings (8 exact matches), mostly common errands such as "Unload the
dishwasher" or "Renew passport before it expires". The guard is idempotent; rerun it
whenever the v2 files or an eval set change.

## 5. How v2 was made (2026-11-06)

All passes used Gemini 4 Argon through Vertex AI (`tcr/llm.py` for single calls,
`tcr/vertex_batch.py` for batch jobs). Requests never carry an icon's name: the glyph,
re-rendered crisply at 256 px and centred on its ink, is the only input, and batch
responses are matched to glyphs by an image hash.

### 5.1 Manual review of the icon set

Before generation, every distinct glyph was shown blind on the labeller's `/curate` page
([labeller/README.md](../labeller/README.md)) and marked *reject*, *somewhat useful* or
*useful* (`data/labels/icon_curation.jsonl`): 832 useful, 13 somewhat, 280 reject, and
2,776 seen but left undecided. Rejects are glyphs that can never be a task's subject,
such as `1.5x` or a megapixel count, which v1 kept with usefulness 8 and 7. v2 skips them.
The review is also the human reference for `discard`.

### 5.2 Model pilot

50 glyphs, stratified by the review (including 4 of v1's known misreads), × 5 configs
with the same prompt, references and glyph (`scripts/v2_pilot.py`, `results/v2_pilot/`).
A Claude Opus agent judged the readings blind, with models hidden and slots shuffled.

| Config | Wrong | Partly | Text realism / fit / near-miss (1–5) | Rejects discarded | Pass-1 cost, batch |
|---|---|---|---|---|---|
| **Gemini 4 Argon** | 0 | 0 | 4.7 / 3.6 / 4.2 | 5 of 6 | ~$44 (promo price) |
| Gemini 3.1 Pro | 0 | 1 | 3.5 / 3.3 / 3.3 | 3 of 6 | ~$127 |
| Gemini 3.8 Flash, low thinking | 1 | 2 | 4.1 / 3.6 / 3.9 | 3 of 6 | ~$14 |
| Gemini 3.8 Flash | 1 | 3 | 4.1 / 3.2 / 3.7 | 3 of 6 | ~$24 |
| Gemini 3 Flash | 2 | 4 | 3.5 / 2.5 / 2.6 | 1 of 6 | ~$24 |

Misreadings concentrated on small corner badges, frames identified by one detail, and
glyphs with two plausible readings. Argon led on every measure and was used for all
passes; one model for every glyph keeps comparisons clean.

### 5.3 Pass 1: describe

`scripts/describe_v2_pass1.py`, one Vertex batch job (half price, no rate limits):
3,619 requests, one per distinct glyph except the review's rejects, v1's discards
included. 337 font aliases got copies, as did 2 glyphs that render identically once
centred. All 3,619 replies were valid after one retry. $43.95. 18% of the glyphs have soft
problems, mostly 17–20 examples instead of 16, or the wrong number of parent headings.

### 5.4 Pass 2: contrast among look-alikes

`scripts/describe_v2_pass2.py`. Look-alike groups come from image embeddings of the
glyphs only (`tcr/glyphs.py` `lookalike_groups`): CLIP ViT-B/32 cosine ≥ 0.975,
average-linkage clusters, split with Ward when larger than 8. CLIP sees every thin black
glyph as similar (0.95 would have grouped 71% of them), so the threshold is high. It
gave 350 groups of 2–8 glyphs (951 glyphs, 33%), and contact sheets of random groups
showed real look-alikes: alarm / timer / clock, bed sizes, signal-bar counts, filled vs
outlined radio buttons.

Each request shows one group's glyphs side by side, each under a letter with its pass-1
description, and rewrites per glyph: `readings`, `ambiguity`, `near_misses` (at least 2
look-alike traps pointing at a sibling's concept), `scores.distinctiveness`, and a new
`distinguishing` sentence ("A diagonal slash cuts across the snowflake, reversing the
meaning from cooling to turning cooling off"). One batch job, $4.15. Distinctiveness of
these glyphs went from mostly 4 to an even spread (229 kept glyphs now score 1). The
replaced pass-1 fields stay in `_meta.pass1`.

### 5.5 Pass 3: QA

`scripts/v2_qa.py`. 485 glyphs judged blind by four Claude Opus subagents (§6.1),
v1 and v2 side by side. The 18 glyphs whose v2 reading was judged wrong or partly were
described again, with the judge's correction as a reviewer note ("follow the reviewer
only where the image confirms it"), and their look-alike groups contrasted again.

### 5.6 What a v2 file records

`_meta` holds the provenance: model, prompt version (`v2.0`; `v2.1-contrast` under
`_meta.contrast`; `v2.0-qa` under `_meta.qa`), `passes`, the glyph's pixel hash, token
usage, soft problems, `alias_of` (for copies), `lookalikes` (the group's other members,
by name as IDs only), the replaced pass-1 fields, and the guard's drop counts.

**Cost:** about $48 for passes 1–2 at Argon's promotional batch price (valid until
2026-11-30, double afterwards); pass 3's online calls were a few cents; the Opus judging
ran on the Claude Code subscription.

## 6. Results

Full tables: `results/v2_stats/report.md` (`scripts/v2_report.py`) and
`results/v2_qa/report.md`.

### 6.1 Glyph reading (blind QA)

Each glyph was shown on a contact sheet with its v1 and v2 descriptions as slots A/B in
an order shuffled per glyph, in one common format ("shows" / "means" / "note"). The
judges gave correct / partly / wrong with a correction. Random 200 glyphs kept by v1 or
v2 (v2 judged after pass 2):

| | wrong (95% CI) | wrong or partly (95% CI) |
|---|---|---|
| v1 | 2.5% (1.1–5.7%) | 27.0% (21.3–33.5%) |
| v2 | 0.0% (0–1.9%) | 0.5% (0.1–2.8%) |

On 302 doubtful glyphs (low recognizability, the review's overrides, and the largest
v1/v2 disagreement) v2 was wrong or partly right on 5.6% and v1 on 65.9%. v1's "partly"
is mostly a missed modifier: a slash read as a generic "off", a badge ignored, a floppy
disk read as a document.

**Caveat:** the blinding is imperfect. v1's keyword lists look different from v2's
sentences, and v2's several hedged readings are easier to accept as fair. The gap is
probably overstated, but it agrees with the pilot.

### 6.2 Text diversity (kept glyphs)

| Measure | v1 | v2 | Target |
|---|---|---|---|
| Words per example, median (p90) | 4 (6) | 9 (14) | the author's titles: 9 (13) |
| Length mix 2–5 / 6–12 / 13–16 words | 0.85 / 0.15 / 0.00 | 0.37 / 0.46 / 0.17 | 0.31 / 0.50 / 0.19 |
| Example strings used by more than one glyph | 5,671 | 1,087 | — |
| Most reused example | 70 glyphs | 14 glyphs | — |
| Distinct negatives / all negatives | 3,231 / 24,874 | 16,517 / 17,425 | — |
| Share of negatives taken by the top 20 strings | 36% | 0.9% | — |
| Languages | EN | EN + 3 ES + 3 RU per glyph | — |

The example tasks' domains have an entropy of 0.92 of the maximum. Home, tech and work
lead; food, career and communication are thin.

### 6.3 Discriminability (bootstrap eval)

Hold out one example task per icon as the query and retrieve among all icons
(`python -m tcr.eval --desc v1|v2`, 150 queries, seed 0):

| nDCG@10 | B1 (BM25) | M1 (dense) | M2 (nearest example) | M3 (B1 + M1) |
|---|---|---|---|---|
| v1 | 0.139 | 0.092 | 0.032 | 0.106 |
| v2 | 0.201 | 0.246 | 0.064 | 0.261 |

Only suggestive: the held-out queries come from the same generator as the descriptions,
and v2 has fewer icons (2,862 vs 3,486).

## 7. What decides between v1 and v2

The decision metric is extrinsic: human labels on the **dev** split.

- The same matchers (B1, M1, M2, M3) on v1 vs v2: nDCG@10, P@1, MRR, and precision at
  50/70/90% coverage for the abstention gate, per dataset and for short vs long tasks.
- **Pool fairness.** The labelling grids are drawn from every matcher on *both* versions,
  plus random icons and search (`scripts/build_label_bundle.py`). Neither version's
  candidates are privileged, and each label's provenance (`M3.v2@1`) records which
  version proposed the chosen icon, which gives recall-of-pool per version directly.
- **Labelling efficiency** (free, from the tool's logs): how often the chosen icon came
  from search (a pool miss), median time per label, and the "no good icon" rate.

**Decision rule:** adopt v2 as the default if it improves dev nDCG@10 for the primary
matcher on Personal and does not hurt Public-short. Report the test split once, at the
end, for both versions: the v1 vs v2 ablation is a result in itself.

Also open: fit the usefulness prior (§3.2) and the discard rule on the labels and the
manual review, instead of the placeholder formula.

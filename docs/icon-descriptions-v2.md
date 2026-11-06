# Icon descriptions v2: what is wrong with v1, and a better format

The icon descriptions (`data/icon_descriptions/`) are the **only matching signal**. They
feed every matcher, the labelling tool's search, and the fine-tuning data (`example_tasks`
as positives, `poor_matches` as negatives). This document records the problems with v1,
proposes a v2 schema and generation process, and defines how to measure whether v2 is
better. v1 stays in place; v2 goes to `data/icon_descriptions_v2/`, so description quality
is an experimental variable.

## 1. Evidence: problems with v1

v1 was produced by `scripts/describe_icons.py`: one image per call, with no icon name.
About 80% (3,422) came from Gemini 3.1 Flash-Lite on the free tier; 494 are untagged early
runs; the rest came from 2.5 Flash-Lite, 2.5 Flash and 3.5 Flash. Statistics are over the
3,798 non-discarded icons unless stated.

| Problem | Measurement | Consequence |
|---|---|---|
| **Glyph misreadings** | Spot check of 10 random Flash-Lite icons against the glyph, re-described by 3.1 Pro: **2 misread** (a bathroom scale read as a "speech bubble / chat"; settings sliders read as "comparison / alternatives"), 2 weaker (an electric meter read as a battery; bones read as archaeology rather than a medical appointment). Pro was right on all 10 | A misread icon is unreachable by matching *and* by search |
| **Example tasks are short and generic** | Median **4 words** (p90 6); the author's real titles have median 9 (p90 13). Top first words: review, update, organize, research, set | Trains on a regime unlike real tasks; weak lexical and semantic overlap with real titles |
| **Example tasks are reused across icons** | 57,944 examples, 42,914 distinct; **6,438 strings appear for more than one icon** ("optimize database query performance": 75 icons; "draft project proposal": 72) | Contradictory positives: one task labelled positive for dozens of icons |
| **Negatives are generic, not hard** | 27,102 `poor_matches`, 3,412 distinct; **the 20 most common strings are 35% of all negatives**. "buy groceries" appears for 1,353 icons, "schedule team meeting" for 972. Most of the top ones are copied from the example titles *in the prompt itself* | They teach nothing a random negative would not; hard negatives are missing |
| **Prompt anchoring** | The prompt's sample titles ("Review quarterly budget", "Research vector database alternatives", "Organize photography equipment") recur as examples and negatives hundreds of times | Topic skew toward knowledge work; low diversity |
| **Visual concepts are one abstract word** | Median 1 word; top entries "focus", "document", "person", "alert", "structure", "disabled", "growth" | Mixes *what is drawn* with *what it means*; modifiers (a plus, a slash, a clock badge) are not tied to their meaning |
| **Task intents are terse** | Median 2 words ("measure", "configure") | Too coarse to separate neighbouring icons |
| **Usefulness is poorly spread** | 0–10 scale with 1,985 icons at 8, 851 at 7, 435 at 4; effectively three levels | A weak prior; `discard` decisions are hard to audit |
| **No notion of look-alikes** | Each icon is described in isolation | Families (alarm variants, badge variants, the numbered "mp" camera series) get near-identical descriptions |
| **English only** | — | Search and lexical matching fail for ES/RU tasks; the multilingual encoder has to do all the bridging |

The reasoning field is the most reliable part (median 189 characters, usually sensible). It
stays, but becomes more structured (§3).

## 2. Design principles

0. **State the purpose, and use the selection principles.** The prompt opens with what the
   icon is for: a glanceable reminder of the one thing a task is about. It then gives the
   salience order from [icon-selection-principles.md](icon-selection-principles.md):
   central object → context/activity → meaningful action → never properties.
   Everything below is generated *through* that lens: examples are tasks whose **central
   subject** this icon shows, not tasks that merely mention something related.
1. **Describe before interpreting.** First record what is literally drawn, then what it
   means. This is chain-of-thought for vision; it is where the misreadings come from.
2. **Separate the glyph from its meanings.** A base object plus modifiers (`+` = add or new,
   slash = disabled or cancelled, clock = scheduled or history, check = done or verified,
   `!` = alert, magnifier = search). Modifier semantics are reusable across the whole icon
   set.
3. **General first, personal lightly.** The descriptions and a future model are meant for
   release (GitHub, Hugging Face), so examples must span ordinary life areas, not one
   person's interests. Personal relevance enters only through *which areas and title
   styles are represented*, never through private text (§4).
4. **Titles, not bodies.** A task's *title* says what the task is; its body says how the
   author will do it. Icons represent the what, so example tasks are titles, written the
   way real people write them.
5. **Hard negatives must be hard.** A near-miss shares a keyword or theme with the icon but
   deserves a different icon, and names the concept that would fit better.
6. **Contrast with look-alikes.** Near-identical glyphs need text that tells them apart.
7. **Calibrated scores** from anchored rubrics, split into separate factors instead of one
   global number.
8. **No evaluation contamination.** Nothing from an eval dataset may appear in a prompt or
   in the generated examples (§4).

## 3. Proposed v2 schema

```jsonc
{
  "schema_version": 2,
  "depiction": {
    "primary_object": "bathroom scale with a small display",   // literal, 3–10 words
    "modifiers": ["three dots on the display"],                // badges, overlays, arrows, slashes
    "composition": "single object, front view"
  },
  "readings": [                               // what a viewer would take it to mean, ranked
    {"meaning": "weighing yourself / body weight", "strength": "primary"},
    {"meaning": "weighing an object, measuring mass", "strength": "secondary"}
  ],
  "ambiguity": "At small sizes the display dots can make it look like a chat bubble.",
  "modifier_semantics": [],                   // e.g. [{"modifier": "plus badge", "meaning": "add / new"}]
  "domains": ["health_fitness", "home", "shopping"],   // fixed taxonomy (below)
  "task_intents": [                           // verb + object, 3–7 words
    "track body weight over time",
    "weigh luggage before a flight",
    "measure ingredients for baking"
  ],
  "example_tasks": [                          // this icon shows the task's CENTRAL subject (rank-1 fit)
    "Weigh myself every Monday and log it in the spreadsheet",
    "Check the suitcase is under 23 kg before the airport run",
    "Buy a kitchen scale that can do grams precisely",
    {"parent": "Home baking", "task": "Measure flour by weight instead of cups"}
  ],
  "context_tasks": [                          // this icon fits as the CONTEXT (a good rank-2 answer)
    {"task": "Log calories for the cutting phase", "central": "food / nutrition"}
  ],
  "near_misses": [                            // hard negatives: the concept is present, but not what the task is about
    {"task": "Track the package delivery for the new scale", "better": "parcel / delivery truck", "trap": "shared object, other subject"},
    {"task": "Buy a heavy-duty waterproof jacket", "better": "jacket / clothing", "trap": "property (heavy)"},
    {"task": "Reply to the group chat about Saturday", "better": "chat bubble", "trap": "look-alike glyph"}
  ],
  "scores": {                                 // anchored 1–5 rubrics (below)
    "recognizability": 4,  // would most people read it the same way?
    "task_fit": 4,         // how many realistic tasks would it label well?
    "distinctiveness": 3   // how different is it from look-alike icons?
  },
  "discard": false,
  "reasoning": "Clear scale metaphor; useful for health and packing tasks; can be confused with chat icons at small size.",
  "_meta": {"model": "…", "prompt_version": "v2.0", "pass": "describe|contrast|translate"}
}
```

- **Field sizes:** `readings` 1–4; `task_intents` 6–12; `example_tasks` 12–20 (about a
  quarter with a `parent` heading, where the parent supplies the subject a generic title
  lacks); `context_tasks` 3–6; `near_misses` 5–8.
- **Near-miss traps**, which follow the selection principles: the icon's concept appears
  only as a **property** (a rain icon for "waterproof headlamp"); as a **generic verb**
  (a check mark for "verify git sync"); as **shared vocabulary** with another subject; as a
  **look-alike glyph**; or as the **parent's topic** when the title is specific and points
  elsewhere.
- **Graded positives:** `example_tasks` are rank-1 positives and `context_tasks` rank-2,
  which gives graded training labels (fine-tuning-plan §4).
- **Domain taxonomy (fixed):** work, career, learning, tech, health_fitness, finance,
  home, shopping, food, social_family, travel, hobbies, admin, communication, creative,
  outdoors. It gives facets for balanced example generation, per-domain evaluation and
  stratified sampling.
- **Score anchors (1–5):**
  - recognizability: 1 = most people would guess wrong; 3 = clear in context; 5 = universally obvious.
  - task_fit: 1 = almost no to-do tasks; 3 = one narrow family; 5 = many common tasks across domains.
  - distinctiveness: 1 = interchangeable with a look-alike; 5 = unique.
- **Mapping to the v1 prior:** `icon_usefulness` = f(scores), and `discard` = task_fit ≤ 1
  or recognizability ≤ 1. The mapping is fitted later on human labels.

**What the matcher uses:** readings, intents and example tasks become text views, as the
v1 fields do now. Depiction and modifiers are a separate "visual" view. Near-misses are
negatives for training and could become a penalty feature. `ambiguity` and the scores feed
the abstention gate: an ambiguous icon should need higher confidence to be shown.

## 4. Conditioning: light, public, contamination-free

**Style of example titles.** Each call gets about 8 *rotating* reference titles, so no
fixed list anchors the whole run. They come from a training-only pool:

- the MS-LaTTE titles not used by Public-short or Public-expanded (short, real phrasing);
- the fresh life-area synthetic pool planned in fine-tuning-plan §4 (long, author-like).

The prompt asks for a **length mix**: about a third 2–5 words (quick jots), about half
6–12 words (the author's typical style), and some 13–16 words.

**Topic coverage.** Each icon's examples must span at least 3 of its applicable domains.
The run-wide domain balance is monitored (§6). The author's interests (ML, programming,
career, photography, languages) are represented because they are ordinary domains of the
taxonomy, not because of private text. This is "light conditioning": the model is
*reminded* that these areas exist, never shown the author's tasks.

**Never used:** task bodies, private tasks, and any title from an eval dataset (Personal,
Realistic, Personal-synth, Public-short, Public-expanded).

**Contamination guard.** After generation, drop any example task whose normalized title
matches, or whose embedding is within cosine 0.95 of, an item in any eval dataset. Report
the drop rate. If eval tasks leaked into training positives, every later result would be
inflated.

**Multilingual.** Also generate 3 ES and 3 RU example tasks per icon, written natively
rather than translated. This serves lexical search and training in those languages, and it
is cheap.

## 5. Generation process

1. **Model pilot** (done 2026-11-06, `scripts/v2_pilot.py`, results in
   `results/v2_pilot/`): 50 glyphs, stratified by the manual review (16 useful, 28
   unsure, 6 reject, including 4 of the spot check's v1 misreads), × 5 configs with
   the same prompt, references and 256 px glyph. A fresh Claude Opus agent judged the
   readings blind (models hidden, slots shuffled per glyph); the author spot-checks.

   | Config | Wrong | Partly | Text realism / fit / near-miss (1–5) | Rejects discarded | Pass 1, batch |
   |---|---|---|---|---|---|
   | Gemini 4 Argon | 0 | 0 | 4.7 / 3.6 / 4.2 | 5 of 6 | ~$44 (promo until 30 Nov; then double) |
   | Gemini 3.1 Pro | 0 | 1 | 3.5 / 3.3 / 3.3 | 3 of 6 | ~$127 |
   | Gemini 3.8 Flash, low thinking | 1 | 2 | 4.1 / 3.6 / 3.9 | 3 of 6 | ~$14 |
   | Gemini 3.8 Flash | 1 | 3 | 4.1 / 3.2 / 3.7 | 3 of 6 | ~$24 |
   | Gemini 3 Flash | 2 | 4 | 3.5 / 2.5 / 2.6 | 1 of 6 | ~$24 |

   Misreadings concentrate on small corner badges, frames identified by one detail, and
   glyphs with two plausible readings. Argon leads on every measure, though with 50
   glyphs the error CIs overlap (0–7% for Argon, 0–10% for the 3.8 Flash variants).
   Argon's one defect, a runaway field listing hundreds of terms, is now a retried
   validation failure. **Decision: Argon for pass 1**, in batch mode (it hit rate limits
   at 10 parallel requests).
2. **Pass 1: describe.** One call per distinct glyph (3,486 kept + those discarded in v1, if
   re-judging `discard`), with the v2 prompt, the glyph at 256 px, rotating title
   references, and the domain list. Copy the result to aliases.
3. **Pass 2: contrast.** Group look-alikes by image-embedding similarity (e.g. SigLIP
   cosine; never by name). For each group of 3–8, send the glyphs together and ask what
   distinguishes each one, and which tasks suit one but not the others. This rewrites
   `readings`, `near_misses` and `distinctiveness` for about the 30% of icons in families.
4. **Pass 3: QA.** A stronger model re-checks depictions where the pass-1 model reported
   high ambiguity or low recognizability, or where pass-1 and v1 disagree most (embedding
   distance between the v1 and v2 readings).
5. **Guard and statistics** (§4, §6), then freeze v2 with the prompt version in `_meta`.

**What was run (2026-11-06).** All with Gemini 4 Argon.

- Pass 1: one batch job over 3,619 glyphs (the review's rejects skipped), $43.95.
  `scripts/describe_v2_pass1.py`.
- Pass 2: 350 look-alike groups of 2–8 (951 glyphs, 33%), from CLIP glyph embeddings at
  cosine ≥ 0.975, checked on contact sheets; one batch job, $4.15.
  `scripts/describe_v2_pass2.py`; the replaced pass-1 fields stay in `_meta.pass1`.
  Distinctiveness of those glyphs went from mostly 4 to an even spread (229 kept glyphs
  now score 1).
- Pass 3: the blind QA below; the 18 glyphs whose v2 reading was judged wrong or partly
  were re-described with the judge's fix as a reviewer note, and their groups
  contrasted again. `scripts/v2_qa.py`.
- Guard: 0.4% of the generated strings dropped (8 exact matches; the rest within e5
  cosine 0.95, mostly common chores and admin tasks). `scripts/v2_guard.py`.

The loader (`tcr.data.load_icons(version="v2")`, or `TCR_DESC=v2`) keeps 2,862 glyphs:
v2's own discards, except the 32 glyphs the review calls useful or somewhat.

**Cost:** per icon, about 2k input tokens and about 2.5k output (JSON plus thinking); ~4k
icons:

| Pass | 3 Flash | 3.5 Flash | 3.1 Pro |
|---|---|---|---|
| Pass 1 | ~$35 | ~$100 | ~$140 |
| Passes 2–3 | ~20–30% on top | | |
| **Batch mode** | halves everything | | |

## 6. Measuring whether v2 is better

**Intrinsic** (cheap, during generation):

- *Glyph-reading accuracy.* 200 random icons, v1 and v2 depictions/readings shown blind in
  random order next to the glyph. The author (or a strong judge validated on a subset)
  marks each as correct / partly / wrong. Report the error rate with a CI; v1 is about 20%
  wrong in the spot check.
- *Diversity and realism:*
  - example-length distribution vs. the target mix;
  - distinct-n and cross-icon reuse rate (v1: 6,438 reused strings);
  - share of negatives taken by the top 20 strings (v1: 35%);
  - domain entropy across all examples.
- *Discriminability:* the bootstrap eval (hold out one example per icon, retrieve among all
  icons) run with v1 and v2 descriptions. Only suggestive: both sides come from the same
  generator.

**Intrinsic results (2026-11-06).** Full tables: `results/v2_stats/report.md`,
`results/v2_qa/report.md`.

- *Glyph reading*, judged blind by Claude Opus subagents (v1 and v2 in shuffled
  slots, one common format). Random 200 glyphs kept by v1 or v2:

  | | wrong (95% CI) | wrong or partly (95% CI) |
  |---|---|---|
  | v1 | 2.5% (1.1–5.7%) | 27.0% (21.3–33.5%) |
  | v2 (after pass 2) | 0.0% (0–1.9%) | 0.5% (0.1–2.8%) |

  On 302 doubtful glyphs (low recognizability, review overrides, largest v1/v2
  disagreement) v2 was wrong or partly right on 5.6% and v1 on 65.9%. v1's partly
  verdicts are mostly a missed modifier (a slash read as a generic "off", a badge
  ignored). Caveat: the blinding is imperfect, since v1's keyword lists look different
  from v2's sentences, and v2's several hedged readings are easier to accept as fair.
- *Diversity* (kept glyphs): v2's example tasks have median 9 words (v1: 4) and a
  length mix of 0.37 / 0.46 / 0.17 against the 0.31 / 0.50 / 0.19 target. Strings reused
  across glyphs fell from 5,671 to 1,087, and the top-20 negatives' share from 36% to
  0.9%. The domain entropy is 0.92 of the maximum, though food, career and
  communication are thin.
- *Discriminability* (bootstrap eval, `python -m tcr.eval --desc v1|v2`, seed 0):
  M3 nDCG@10 0.106 → 0.261, M1 0.092 → 0.246, B1 0.139 → 0.201. This is only
  suggestive: the held-out queries come from the same generator, and v2 has fewer
  icons (2,862 vs 3,486).

**Extrinsic** (the decision metric, on human labels, **dev split only**):

- Same matchers (B1, M1, M2, M3), with v1 vs. v2 descriptions: nDCG@10, P@1, MRR, and
  precision at 50/70/90% coverage for the abstention gate. Per dataset, and for short vs.
  long tasks.
- **Pool fairness.** Labelling pools are built from matchers over *both* v1 and v2, plus
  random icons and search. Neither version's candidates are privileged, so its recall can be
  measured. Each label's provenance records which version proposed the chosen icon, which
  gives **recall-of-pool per version** directly.
- *Labelling efficiency* (free, from the tool's logs): share of tasks where the chosen icon
  came from search (a pool miss), median `ms_spent`, and the "no good icon" rate. Better
  descriptions → better pools → fewer searches and faster labels.

**Decision rule:** adopt v2 as the default if it improves dev nDCG@10 for the primary
matcher on Personal and does not hurt Public-short. Report the test split once, at the
end, for both versions: the v1 vs. v2 ablation is a result in itself.

## 7. Order relative to curation and labelling

**First, review the icon set by hand** (the labelling tool's `/curate` page; see
[labeller/README.md](../labeller/README.md)). Every distinct glyph, including the ones v1
discarded, is shown blind and given one of three levels: *reject*, *somewhat useful* or
*useful*. Rejects are, for example, property-only glyphs such as `1.5x` or a megapixel
count, which v1 kept with usefulness 8 and 7. Pass 1 then skips the rejected glyphs, and
the result (`data/labels/icon_curation.jsonl`) is the human reference for `discard` and
`task_fit` (§3): v1's `discard` and `icon_usefulness` are scored against it now, v2's
scores later. The three levels can also serve directly as the usefulness prior, which in
v1 has only three effective levels anyway (§1).

Do the pilot and pass 1 **before** heavy labelling, then rebuild the labelling pools over
v1 + v2 (§6, pool fairness). Pass 2 (contrast) can follow while labelling runs: the pools
already cover both versions, and the decision is made on dev labels.

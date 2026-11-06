# Icon descriptions v2: status and hand-off

Working notes for continuing [icon-descriptions-v2.md](icon-descriptions-v2.md) in a new
session. Update this file as steps finish; delete it once v2 is frozen.

## Done (2026-11-06)

- **Manual icon review** imported to `data/labels/icon_curation.jsonl`: 832 useful,
  280 reject, 13 somewhat, 2,776 seen but undecided ("unsure"). Rejects are skipped by
  pass 1; the three levels can serve as the usefulness prior. No more icon-only
  labelling planned: the middle is better judged through task labels.
- **Model pilot** (`scripts/v2_pilot.py`, `results/v2_pilot/`): 50 glyphs × 5 configs,
  judged blind by an Opus agent. **Gemini 4 Argon** won on every measure (0 misreadings,
  best example diversity, best near-misses, agreed most with the human rejects). The
  author agreed after reading the report: all models were decent, Argon more diverse.
  Results table in icon-descriptions-v2.md §5.
- **The v2 request** lives in `tcr/describe_v2.py` (prompt `v2.0`, schema, validation,
  rotating reference titles that exclude every eval title, crisp centred 256 px glyph).
  `tcr/llm.py` sends images, thinking level and media resolution, and returns usage.

## Pass 1 (describe): `scripts/describe_v2_pass1.py`

One Vertex batch job with every kept glyph: 3,619 requests (3,621 glyphs; two pairs
render identically once centred and are described once), plus 337 font aliases written
as copies. ~$44 at Argon's promotional batch price (until 2026-11-30; double after).

```bash
cd ~/projects/research/task-concept-retrieval
python scripts/describe_v2_pass1.py run 2>&1 | tee -a .cache/v2_pass1/pass1.log
```

- `run` = prepare → submit → wait (polls every 5 min) → collect → retry. The job runs
  on Google's side: once "submitted" is printed, the laptop may sleep or the command
  may be stopped (Ctrl-C). Nothing is lost.
- **Check on it:** `python scripts/describe_v2_pass1.py status`
- **Resume** (after stopping it, or the next morning): run the same `run` command
  again. It never resubmits a job that `state.json` already records; it waits, then
  collects and retries.
- **Collect by hand:** `collect`, then `retry` (synchronous Argon calls for whatever is
  missing or failed validation, 2 at a time).
- **State:** `.cache/v2_pass1/pass1/{input.jsonl, manifest.json, state.json, output/}`.
  Batch files sit in `gs://<project>-tcr-batch/v2_pass1/` (private bucket,
  us-central1); delete them when done.
- **Output:** `data/icon_descriptions_v2/<name>.json`: the v2 schema plus
  `schema_version: 2` and `_meta` (model, prompt version, glyph hash, usage, soft
  problems, `alias_of`). Committed, like v1.
- **Start over** (rarely needed): delete `.cache/v2_pass1/pass1/state.json`, which makes
  `run` submit a new job.

## Left to do

1. **Check pass 1**: count files against 3,619 + aliases. Look at the share with soft
   problems, the discard rate (vs the human levels), and the cost from the usage sums.
2. **Pass 2 (contrast)**: look-alike groups by image embedding (`tcr/glyphs.py`
   `glyph_embeddings` / `lookalike_order`; never names). Send 3–8 glyphs together and
   rewrite `readings`, `near_misses` and `distinctiveness` for the ~30% in families.
   The pilot's misreadings were exactly this kind of glyph (small badges, frames
   defined by one detail, two plausible readings).
3. **Pass 3 (QA)**: a second look at glyphs with low recognizability or high ambiguity,
   or where v1 and v2 readings disagree most. Judge: an Opus subagent, blind, as in the
   pilot (the author prefers Opus when the cost is a few % of a 5-hour window).
4. **Guard and statistics** (§4, §6): drop example tasks that equal an eval title or are
   within e5 cosine 0.95 of one (the pilot found 2–10 per 50 glyphs per model; the code
   is in `scripts/v2_pilot.py` `contamination()`). Then report length mix, cross-icon
   reuse, top-20 negative share and domain entropy, and freeze v2.
5. **Make v2 matchable**: `tcr/data.py` reads only the v1 fields. Add a v2 loader that
   maps readings, intents, examples (+ ES/RU) and depiction to the IconDoc field views,
   and a switch between v1 and v2. Then rebuild the labelling pools over v1 + v2
   (§6, pool fairness).

## Decisions

- One model for all glyphs (no split by usefulness or difficulty), for clean
  comparisons: Argon.
- Batch mode for pass 1: half price, no rate limits (Argon hit 429 at 10 parallel
  online requests).
- Default media resolution (1,120 image tokens); lower resolution would save ~$1–3.
- The schema carries no array or integer bounds (Vertex rejects it with them); counts
  are enforced by the prompt and checked by `soft_problems()`.

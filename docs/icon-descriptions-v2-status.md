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

## Done (2026-11-06, second session)

- **Pass 1 checked**: 3,619 glyphs + 339 alias copies, all valid ($43.95). 18% have soft
  problems (mostly 17–20 examples instead of 16, or the wrong parent count): harmless.
  v2 keeps 2,831 glyphs; it discards 27% of the review's "unsure", 20 of 832 "useful"
  and all 12 "somewhat". Report: `scripts/v2_report.py` → `results/v2_stats/report.md`.
- **Contamination guard** (`scripts/v2_guard.py`): dropped 483 generated strings within
  e5 cosine 0.95 of an eval title (8 exact), of 125,536. The files record only the
  counts (`_meta.guard`), never the text (Personal is private). Idempotent; rerun after
  every pass.
- **v2 is matchable**: `load_icons(version="v1"|"v2")`, default `TCR_DESC` (v1).
  v2 maps depiction → `visual_concepts`, readings → a new `readings` view,
  examples → EN ("Parent: task") + ES + RU, near-misses → `poor_matches`; usefulness =
  (recognizability + task_fit) / 10 until fitted on labels. The review overrides v2's
  discard for useful/somewhat glyphs (2,862 kept). `python -m tcr.eval --desc v2`.
  The embedding caches are now keyed on the encoded texts (before, v2 would have
  reused v1's vectors). Bootstrap eval after all passes (suggestive only): M3 nDCG@10
  0.106 (v1) → 0.261 (v2).
- **Pass 2 (contrast)** (`scripts/describe_v2_pass2.py`): look-alike groups from CLIP
  glyph embeddings at cosine ≥ 0.975: 350 groups of 2–8, 951 glyphs (33%), checked
  on contact sheets. All 350 groups merged (348 by batch, 1 retried and 2 test groups
  online; $4.15). Distinctiveness of the 951 glyphs moved from mostly 4 to an even
  spread (239 now score 1). 10 entries name a sibling by letter ("icon B"); harmless.
  The guard then dropped 5 new near-misses.
- **Pass 3 (QA)** (`scripts/v2_qa.py`): 485 glyphs judged blind, v1 vs v2, by four
  Opus subagents (12 packets each). Random 200: v1 wrong or partly 27%, v2 0.5%.
  The 18 v2 readings judged wrong or partly were re-described with the judge's fix
  and their groups contrasted again. Results in icon-descriptions-v2.md §6.
- **v2 frozen** (prompts `v2.0`, `v2.1-contrast`, `v2.0-qa` in `_meta`).
- **The labelling pools** (`scripts/build_label_bundle.py`) now draw from every matcher
  on both versions (provenance `M3.v2@1`), over the union of the kept icons. Not
  rebuilt yet.

## Left to do

1. Rebuild the labelling pools: `python scripts/build_label_bundle.py --refresh`.
   Without `--refresh`, existing grids are kept and get no v2 candidates. Check first
   that no task labels exist on the NAS; grids someone already labelled must not change.
2. Delete the batch files in `gs://<project>-tcr-batch/v2_pass1/` and `v2_pass2/`.
3. Then label, and compare v1 vs v2 on dev labels (icon-descriptions-v2.md §6,
   extrinsic). Delete this file once the pools are rebuilt.

## Decisions

- One model for all glyphs (no split by usefulness or difficulty), for clean
  comparisons: Argon.
- Look-alike threshold: CLIP cosine 0.975 (0.95 put 71% of glyphs in groups; CLIP
  sees all thin black glyphs as similar).
- The review's useful/somewhat level overrides v2's discard (32 glyphs).
- Batch mode for passes 1–2: half price, no rate limits (Argon hit 429 at 10 parallel
  online requests).
- Default media resolution (1,120 image tokens); lower resolution would save ~$1–3.
- The schema carries no array or integer bounds (Vertex rejects it with them); counts
  are enforced by the prompt and checked by `soft_problems()`.

# Experimentation Strategy

This document defines *how we experiment* on the task→icon matching problem: how a task is
represented, what data we use, how validation data is produced, and how methods are measured. The
catalogue of methods themselves lives in [matching-methods.md](matching-methods.md).

> **One-line goal.** Given an org-mode task, show a single Material Symbols icon that captures the
> task's idea at a glance — **or show nothing** if no icon is genuinely suitable. Matching must run
> in real time during agenda rendering.

---

## 1. Problem framing

This is **open-vocabulary retrieval + a decision to abstain**, not classification:

- **Corpus:** ~4,257 icons, each with a rich generated description (see schema below). Fixed,
  known in advance, embeddable offline.
- **Query:** an arbitrary task title (+ tags), unseen vocabulary, any language.
- **Output:** the single best icon **or `None`**. Because the agenda auto-shows the top-1, a wrong
  icon is actively harmful — worse than showing nothing. So the system must **retrieve then
  decide** (rank candidates, then gate on confidence). This abstention requirement is treated as a
  first-class concern throughout (§7, metrics in §8, and in every method).

We are **not** doing classification (the label space is the open set of tasks, the item set is the
~4k icons, and most icons are never the right answer for a given task). We frame it as
**information retrieval with an abstention gate**.

### Icon description schema (the only signal we use)

Each `data/icon_descriptions/<name>.json` (4,257 files) has:

```json
{
  "icon_usefulness": 8,                // 0–10, LLM-judged fitness for task labelling
  "discard": false,                    // true → unusable for our purpose
  "visual_concepts": ["bookmark", "ribbon", "tag", ...],   // 5–15 items
  "task_intents": ["saving for later", "prioritizing", ...],// 5–15 items
  "example_tasks": ["Save article for weekend reading", ...],// 10–20 realistic titles
  "poor_matches": ["Delete temporary files", ...],         // 5–10 tasks that should NOT use it
  "reasoning": "The bookmark icon is a universally recognized metaphor for ...",
  "_meta": {"model": "gemini-3.1-flash-lite"}
}
```

**Hard rule:** we **ignore the icon's official name and font metadata** (`name`, `tags`,
`categories` in `material_symbols_catalog.json`). Those were authored for UI/typography, not for our
purpose. We match on `visual_concepts` / `task_intents` / `example_tasks` / `reasoning` only, and we
respect `icon_usefulness` and `discard` for quality gating. (`name` is used only as a stable ID/key
and to render/identify the icon, never as a matching feature.)

---

## 2. Query representation: what we feed the matcher

**Decision: the normalized task title + its org tags.** The body/description is a *separate optional
experiment*, not the default (most agenda tasks have no body, and bodies add noise and latency).
Scheduling/timestamps are dropped — they carry no semantic signal for the icon.

We keep the representation **system-agnostic** (a unified plain-text query) so the approach
transfers to non-org task systems later. The normalization pipeline:

1. **Strip the TODO state and priority.** Remove leading keywords (`TODO`, `NEXT`, `INPR`, `WAIT`,
   `MAYB`, `DONE`, …) and priority cookies (`[#A]`, `[#B]`, `[#C]`).
2. **Split off tags.** Parse trailing `:tag1:tag2:` and treat tags as a separate field appended to
   the query text (they are short topic labels, e.g. `:career:planning:`).
3. **Drop scheduling lines** (`SCHEDULED:`, `DEADLINE:`, `CLOSED:`, `Started:`) and timestamps.
4. **Normalize links — protocol-agnostic, not Obsidian-specific.** Keep the human-readable text,
   drop the machinery:
   - org link with description `[[target][Desc]]` → `Desc`
   - bare org link `[[target]]` → readable text derived from `target` (last path/title segment,
     `_`/`-`/`/` → spaces); if it is opaque (e.g. a UUID), drop it
   - protocol/wiki links of the form `[[obsidian:Note Name]]`, `[[id:...]]`, `[[file:~/x.org]]`,
     `[[roam:Page]]`, `mailto:a@b.com`, `https://host/path` → strip the `protocol:` prefix and
     target, keep the readable label (or the last meaningful path segment for bare URLs)
   - implemented as a small **rule set keyed on `protocol:`**, so new schemes are easy to add
5. **Strip org markup:** `*bold*`, `/italic/`, `_underline_`, `=verbatim=`, `~code~`, `+strike+`.
6. **Collapse whitespace.**

**Optional experiments** (documented, off by default): URL-part expansion (break a domain/path into
words), query expansion with synonyms or an LLM ("what icon would represent this task?"), and
including the body. These are A/B'd, not baked in.

---

## 3. Icon corpus representation

For each non-discarded icon we build a **document** from the description fields. Several
representations are compared in [matching-methods.md](matching-methods.md); the defaults are:

- **Concatenated document** = `visual_concepts` + `task_intents` + `example_tasks` (+ optionally
  `reasoning`), used by lexical (BM25) and the single-vector bi-encoder.
- **Field-structured** = each field encoded separately for multi-vector / field-weighted methods.
- **Example-centric** = the `example_tasks` list kept as individual short texts, for
  nearest-example voting (these are *literally tasks*, so task↔task similarity is a strong signal).

**Quality gating** (applies everywhere): icons with `discard=true` are removed from the index;
`icon_usefulness` (0–10) is used as a **prior** that down-weights low-usefulness icons in scoring,
so a marginally-similar but "bad for labelling" icon loses to a slightly-less-similar but excellent
one.

`poor_matches` are retained as **hard negatives** — see §4.

---

## 4. Using `poor_matches` (hard negatives)

Each icon ships ~5–10 `poor_matches`: tasks that look superficially related but should **not** use
that icon. These are valuable supervised negatives. Concrete uses:

1. **Contrastive fine-tuning.** For icon *I* with poor-match task *t*, treat `(t, I)` as a **hard
   negative** in InfoNCE/triplet training: the encoder must push *t* away from *I* even though they
   share surface features. Combined with in-batch (random) negatives.
2. **Reranker supervision.** Negative labels for the cross-encoder / MLP / LambdaMART rerankers and
   the learned abstention gate.
3. **Abstention calibration.** `poor_matches` are exactly "high similarity but wrong" cases —
   ideal for setting/learning the show-vs-abstain threshold and training the OOD detector.
4. **Diagnostic metric — poor-match suppression rate.** For each method, the fraction of
   `poor_matches` correctly ranked low (and not shown). Reported alongside the main metrics; a
   method that scores well on positives but ranks poor-matches highly is dangerous given abstention.
5. **Hard-negative mining seed.** Starting point for iterative (ANCE-style) mining of additional
   near-miss icons beyond the provided ones.

---

## 5. Datasets

The icons' own `example_tasks` are **deliberately demoted** to weak *training-only* signal: they
are LLM-synthesized, narrow, and circular with the corpus (a method trained and tested on them
would look artificially good). They are **never** used as the held-out evaluation set.

### 5.1 Sources

| Source | What we extract | Role |
|---|---|---|
| **Vendored `samples/realistic`** | The existing ~22 org files / ~300 tasks, copied into `data/eval/realistic/` as a raw snapshot + parsed `tasks.jsonl` | In-domain held-out (small, trusted) |
| **Anonymized personal extraction** | De-identified titles+tags abstracted from the user's ~2,000 real tasks | **Primary in-domain corpus** (realistic distribution) |
| **wikiHow goal-step** (`tasksource/goal-step-wikihow`) | ~187k *goal* titles ("How to X" → imperative tasks) | Generic breadth, realistic phrasings |
| **Google Taskmaster-1/2/3** (`google-research-datasets/taskmaster{1,2,3}`) | User goals / action items across restaurants, food, movies, hotels, flights, music, sports | Generic, domain variety |
| **SNIPS / ATIS / CLINC150** | Short imperative intent/reminder utterances | Generic, reminder-style phrasings |
| **GitHub TODO/SATD corpora**: Tesoro (`NamCyan/tesoro`), SATDAUG, `yikun-li/satd-issue-tracker-data` | Real `TODO`/`FIXME` developer tasks | Technical-task domain |
| **Public org-mode / todo.txt repos** | Mined task headlines | Generic, native task formatting |
| **LLM synthesis** | Diverse tasks seeded by the anonymized real distribution | Coverage + targeted gaps; held out from training |
| **Multilingual** | LLM-translated slice of the above into **Spanish + Russian** | Multilingual eval |

> License note: each external dataset's license is checked before any redistribution; we cite all
> sources and, where redistribution is restricted, store only IDs/our-derived transforms or
> download scripts rather than the raw data.

### 5.2 Privacy-preserving personal extraction (primary realistic source)

The user has ~2,000 real personal tasks (in a separate directory, never copied here raw). We run an
LLM **anonymization / abstraction pass** that:

- strips PII — names, employers, places, account numbers, project codenames, anything identifying;
- rephrases each task into a **generic-but-faithful** title (+ generic tags), preserving the
  *concept* (what icon it implies) while discarding specifics;
- emits only de-identified output.

This is gated by a **manual review** before anything is committed. The raw personal tasks never
enter the repo. The result is the highest-value **in-domain** corpus because it matches the real
task distribution the system will face.

### 5.3 LLM choice per task (a decision rule, not one global model)

- **Bulk / cheap / repeatable** (synthesis, translation, silver judging): **Gemini** — free within
  500 requests/day when batched.
- **Sensitive / correctness-critical** (the PII anonymization pass): a **strong, instruction-
  faithful model — Claude Sonnet/Opus** — because PII leakage is a safety concern; plus the manual
  review gate. A one-off in-session pass can validate the pipeline on a small sample first.

### 5.4 Splits & anti-overfit

- **In-domain held-out** = vendored `realistic` snapshot **+** the anonymized personal extraction.
- **Generic** = wikiHow / Taskmaster / SNIPS / TODO corpora / synthesis.
- **Multilingual** = ES + RU.
- **Train/tune only on generic + synthesis.** Never tune on personal tasks.
- **Report metrics per split** so we can see in-domain quality *and* generality separately —
  guarding against overfitting to the user's personal style. Category-stratified cross-validation
  where applicable.

### 5.5 Artifact commit policy

- **Gitignore derived artifacts:** parsed `tasks.jsonl`, embeddings `.npz`, silver-label files,
  cached judge outputs. They are reproducible from source.
- **Commit raw inputs + generator scripts:** the vendored `.org` snapshot and all scripts, so the
  pipeline is reproducible.
- **Revisit later:** if regeneration becomes slow or eval needs a frozen split for comparability,
  commit a pinned snapshot under `data/eval/` at that point.

---

## 6. Validation data — two tracks + active learning

We build **both** an automated (silver) track and a hand-labelled (gold) track, and bridge them
with active learning. The whole catalogue (4,000+ icons) is far too large to put in an LLM prompt,
so both tracks operate on a **retrieved shortlist** (~20 candidates per task).

### 6.1 LLM-judge (silver labels)

For each task: embed-retrieve ~20 candidate icons, then ask the judge to score/rank them against
their descriptions. The rubric is **pointwise graded** (e.g. 0=poor … 3=excellent) and includes an
explicit **"no icon is appropriate"** verdict (needed for the abstention gate). Gemini is the
primary judge, **batched ~10–20 tasks per request** to live within 500 req/day (500 requests then
cover 5,000–10,000 tasks).

### 6.2 Calibration — two technical senses

**(a) Judge calibration.** The cheap Gemini judge is measured against the higher-quality Claude
judge on a shared subset:

- inter-rater agreement: **Cohen's / Krippendorff's κ** on the verdict labels, **Spearman ρ** on
  the graded scores;
- estimate Gemini's systematic **bias** (e.g. consistently over-rating) and **noise**;
- **bias-correct / reweight** Gemini's silver labels accordingly (or restrict silver labels to
  high-agreement regions).

This tells us how much to trust silver labels and where to spend the (limited) gold-labelling
budget.

**(b) Score calibration.** A method's raw similarity (e.g. cosine) is **not** a probability and is
not comparable across queries. We learn a mapping `score → P(acceptable match)` via **Platt scaling
(logistic regression), isotonic regression, or temperature scaling**, fit on labelled data. This
makes a single global **abstention threshold** meaningful and scores comparable across tasks. We
report **reliability diagrams + Expected Calibration Error (ECE)**.

### 6.3 Committee (query-by-committee)

A **committee of judges** — Gemini + Claude + Groq (a free Llama-70B-class model) — and/or a
committee of retrieval **methods**. For each (task, candidate):

- collect each member's vote/score;
- aggregate by **majority / Borda count / mean-score**;
- **consensus → confident silver labels** (cheap to scale);
- **disagreement → high-information items** routed to the gold (iPad) track for human resolution;
- track each judge's reliability (from §6.2) and **weight votes** accordingly.

The committee is itself a CV-worthy artifact (ensemble judging + disagreement-driven labelling).

### 6.4 iPad labelling tool (gold labels)

A **FastAPI server + lightweight browser UI** for relaxed, sofa-friendly labelling on an iPad over a
tunnel (e.g. tailscale/ngrok):

- shows a task and a **grid of candidate icons** (rendered PNGs from `data/icons/`);
- tap to rate each: **best / good / bad / no good icon** (the last is essential for abstention);
- stores to SQLite/JSONL.

Candidate generation = **union of top-k from several methods** (diverse), so the labels are not
biased toward one model's view.

**Search-augmented labelling (explicit user requirement).** When the auto-retrieved candidates are
poor, the user can pick up the keyboard and type a query into a **search box** in the UI; an
on-device search (lexical / ripgrep-style over descriptions, or any registered method via the POC's
`search` engine) pulls **additional** candidate icons into the grid to judge, on top of the
retrieved set. This both rescues bad shortlists and captures gold labels for icons the automatic
methods missed.

### 6.5 Active learning

Which tasks get the (expensive) human labelling is chosen by:

- **uncertainty** (low calibrated confidence / small top1–top2 margin),
- **committee disagreement** (§6.3),
- **diversity / core-set** (cover the query space, avoid redundant labels).

Then retrain/re-evaluate and repeat. Extensions noted for later: **BALD**, expected-model-change.

---

## 7. Abstention ("show nothing unless it's good")

Because the agenda auto-displays the top-1 icon, **a wrong icon is worse than no icon**. Every
matcher therefore returns `Optional[(icon, score, confidence)]` and a calibrated **decision gate**
decides whether to show it. Gate approaches we compare (details in
[matching-methods.md](matching-methods.md)):

- **Calibrated score threshold** (after §6.2(b) calibration);
- **margin / entropy** signals (top1−top2 gap; entropy over the candidate softmax);
- **OOD / density** detection in embedding space (Mahalanobis distance to the icon cloud; kNN
  density) — catches tasks with no good icon;
- **conformal prediction** — distribution-free guarantee on the "shown" set;
- a small **learned gate** (logistic regression / MLP) over features (top scores, margins, lexical
  overlap, `icon_usefulness`) predicting "acceptable@1".

Labels for the gate come from the "no good icon" verdict (§6.1, §6.4), deliberately-included
negative tasks (tasks that *should* get nothing), and `poor_matches` (§4).

---

## 8. Metrics & protocol

Because of abstention, the headline is **not** a single accuracy number but a **precision–coverage
(risk–coverage) curve**:

- **Precision@1 conditioned on showing** vs **coverage** (% of tasks given an icon). We report
  Precision@1 at fixed coverage operating points (e.g. coverage = 50%, 70%, 90%).
- **Gate AUROC / AUPRC** for the show/no-show decision; **abstention F1**.
- **Cost-weighted utility** — a single tunable number weighting *wrong-icon-shown* vs
  *good-icon-withheld*; the user sets the cost ratio to reflect their preference.
- Ranking quality (independent of the gate): **Recall@k, MRR, nDCG@k**.
- **Poor-match suppression rate** (§4).
- **Latency p50/p95** (the real-time constraint).

Protocol: fixed seeds and splits; cached embeddings; results written as JSON + a markdown
leaderboard. **Always report per split** (in-domain / generic / multilingual). Experiment tracking
is lightweight by default (JSON + leaderboard); W&B/MLflow are optional showcase add-ons.

---

## 9. Manual exploration (this round, in the POC)

A CLI/notebook entry point (`tcr.cli search "<query>"`) queries any method directly and returns the
top-k icons — so the user can hand-search the 4k catalogue when nothing auto-matches. This is the
**same search engine** the iPad tool's search box will call (§6.4), so building it now is reused
later.

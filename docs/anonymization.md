# Publishing a stand-in for a private task list

How the public **Personal-synth** dataset is derived from the author's private to-do list
(~2,950 tasks) without leaking it, and how the leakage is measured. The private list is the
primary evaluation set for the task→icon matcher. Personal-synth exists so that results on
the in-domain distribution can be shown and reproduced publicly.

Code: `scripts/datasets/synth_personal.py`, `tcr/privacy.py`, `scripts/datasets/make_style_exemplars.py`.

## 1. Threat model

- **Asset:** the private list. That covers individual tasks, and also what the list as a
  whole reveals about its owner: profession, employer, location, family, health, languages.
- **What gets published:** a set of synthetic tasks, and the icon labels for them.
- **Adversaries:**
  - (a) a reader looking for verbatim or near-verbatim private tasks;
  - (b) a reader who knows the author and looks for recognisable specifics (names,
    projects, places);
  - (c) an **LLM attribute-inference attacker** that reads the whole published set and profiles
    its author (Staab et al. 2024 show this works at scale on real text).
- **Not in scope:** the LLM providers used during generation. Private text is sent only to
  paid or subscription endpoints that do not train on it (Vertex, Claude Code), never to a
  free tier, and the `tcr/llm.py` wrapper enforces this with `private=True`. The owner judged
  the list low-sensitivity for this purpose: no credentials or account data.

## 2. Why not rewrite each private task?

The default industrial approach is de-identification: detect PII (names, emails, places) and
mask or replace it, as in Microsoft Presidio. Lison et al. (2021) argue this is **not
anonymization**. Text leaks through *quasi-identifiers*, combinations of individually
harmless facts (a city, an employer's product area, a rare hobby, a visa renewal), and NER
recall on those is poor. A personal to-do list is almost entirely quasi-identifiers. Even
"generic-but-faithful" per-item rewrites keep the set's joint profile intact, which is exactly
what adversary (c) exploits. Staab et al. (2025) show LLM-based adversarial rewriting beats
classic anonymizers, but it is still a one-to-one rewrite of the same records.

So we **generate new records from an abstracted description of the distribution**. The
private text is never rewritten; it is only summarized at the cluster level.

## 3. Protocol

1. **Canaries.** 15 fabricated tasks, each carrying two random pseudo-words (e.g. "Vozelibas"),
   are planted in a working copy of the private list. If any pseudo-word shows up in a
   skeleton or a generated task, the pipeline demonstrably copies rare private strings
   (after Meeus et al. 2025, "The Canary's Echo").
2. **Cluster.** Tasks are embedded locally (multilingual-e5-small) and grouped into 100
   clusters with agglomerative Ward clustering.
3. **Drop small clusters.** Clusters with fewer than *k* = 5 tasks are dropped: a topic the
   author touched fewer than five times is more likely to be identifying. This follows the
   spirit of k-anonymity, and MS-LaTTE's release rule, which kept only tasks created by at
   least 5 users.
4. **Abstract skeletons.** Claude sees at most 30 sampled members per cluster and writes a
   topic label, a description and 5–8 *generalized* task patterns. This is
   "self-disclosure abstraction" (Dou et al. 2024): names become roles, employers and
   projects become categories, places become kinds of places, and dates, amounts, health
   details and rare words are dropped. Widely known public topics (a library, a famous paper)
   may stay. The model marks clusters that cannot be generalized without losing their point
   (a specific employer's internal work, a named person); those are dropped.
5. **Noisy counts.** Each cluster's target count is proportional to its size, plus Laplace
   noise (scale 2). This is a cheap histogram-level differential-privacy step: it hides exact
   per-topic frequencies. It is not an end-to-end DP guarantee, because step 4 reads the text.
6. **Generate.** Claude writes new tasks from **the skeleton alone**, plus aggregate style
   statistics and a fixed set of 16 fictional **style exemplars** on unrelated topics
   (reviewed by the author and committed). Each task gets a *shape* and a *markup*
   instruction, sampled from the author's own distribution (body-length buckets, bold,
   verbatim, obsidian links, URLs, lists), so that style is matched in aggregate, not copied.
   The generator never sees a private task. About 30% extra tasks are generated to absorb
   rejections.
7. **Leakage gates.** A task is rejected if any of these fires:
   - its normalized title equals a private title;
   - it shares a word 5-gram with the private list (URLs compared whole);
   - it contains a private URL;
   - it contains a **rare private token**: a word used in at most 3 private tasks and absent
     from every public reference corpus (MS-LaTTE, MASSIVE, icon descriptions, the
     exemplars). This catches names and code names without having to recognise them;
   - its embedding's nearest private neighbour is closer than τ, the 95th percentile of the
     private-holdout → private-train nearest-neighbour similarity. In other words, it is
     closer to a real task than real tasks usually are to each other (a per-item
     distance-to-closest-record test);
   - Presidio finds a contact or account identifier (email, phone, IBAN, …). Invented person
     and place names are reported but allowed: fiction needs them, and private names are
     caught by the rare-token gate.
8. **Human review.** The author reviews every candidate while labelling it and flags
   anything recognisable. `publish` drops flagged items and is the only step that writes the
   public file.

## 4. Audit metrics

| Question | Metric |
|---|---|
| Does the pipeline copy rare strings? | Canary tokens echoed in skeletons and in generated tasks: must be 0 |
| Are published tasks near-copies? | Nearest-neighbour similarity quantiles: synth→private vs. holdout→private; gate firing rates |
| Can the author be profiled? | Attribute-inference attack (16 attributes: location, profession, employer, education, age, gender, family, languages, hobbies, health, income). Vertex Gemini Pro, a different model family from the generator, reads 500 random tasks, 3 runs per set, on the **private list (ceiling)**, **Personal-synth**, and **Public-expanded (floor: no private input)**. Scored by the author against a private ground-truth card. Privacy holds if Personal-synth sits near the floor |
| Is the stand-in still useful? | Topic histogram JS divergence vs. the private list (k-means over private embeddings), compared with a private holdout (ideal) and Public-expanded (baseline); style statistics (lengths, markup and link rates) |
| **Is it a valid eval proxy?** (follow-up) | **Eval-set fidelity:** Kendall τ between the ranking of matching methods on the private list and on Personal-synth, once both are labelled. This is the metric that justifies publishing results on the stand-in |

## 5. Results

Filled from `results/privacy_metrics.json` (public-safe aggregates) after the audit. The full
audit, including the attacker's guesses about the author, stays in
`data/private/privacy_audit.md`.

### First run (2026-09-27)

**Generation.**

- 100 clusters; 97 kept (k ≥ 5); 10 skeletons were judged identifying and dropped.
- 1,675 tasks generated in 3 rounds; 1,021 passed the gates and are candidates.
- Gate firings:
  - nearest-neighbour: 522 (τ = 0.908);
  - distinctive 5-gram: 74;
  - private URL: 9;
  - rare private token: 8;
  - hard PII: 0.

**Calibration lessons.**

- A naive rare-token gate, "rare in the private list and absent from a few public
  corpora", rejected 71% of tasks: it fired on ordinary English words.
- A general vocabulary (wordfreq, ~320k words) removed most of those false positives. An
  LLM vocabulary audit then cleared the remaining rare-but-public technical terms: 181 of
  186 were judged public (library, metric, textbook-author names); the 5 uncertain ones stay
  blocked.
- Generic shared 5-grams ("I should be able to") were likewise excluded by requiring
  ≥ 2 uncommon words and absence from public text.

**Canaries.** 0 of 30 canary tokens echoed, in either the skeletons or the generated tasks,
even though every canary was shown to the skeleton writer.

**Nearest-neighbour similarity to the private train half** (multilingual-e5):

| | median | q95 | q99 |
|---|---|---|---|
| private holdout (reference) | 0.883 | 0.908 | 0.930 |
| Personal-synth | 0.883 | 0.904 | 0.907 |
| Public-expanded (floor) | 0.853 | 0.873 | 0.881 |

Synthetic tasks are exactly as close to the private list as unseen real tasks are, and the
tail above τ is cut by the gate. The DCR share (closer to the train half than to the
holdout) is 0.524, close to the no-memorization value of 0.5.

**Topic fidelity** (JS divergence of k-means topic histograms vs. the private train half):

| holdout (ideal) | Personal-synth | Public-expanded (baseline) |
|---|---|---|
| 0.014 | 0.198 | 0.448 |

**Attribute inference** (16 attributes × 3 runs). Values stay in the private audit; only
the pattern is reported here.

- **The attacker gets from Personal-synth but not from the floor:** the profession and
  seniority, an age band, household composition, the income bracket, and the owner's
  *specific set of hobbies*. These travel through the *topic distribution*, which is exactly
  what the dataset is meant to preserve.
- **From the real list but not from Personal-synth:** city, country of origin, native
  language, employer history, and specific health conditions. On these, the synthetic set
  sits at the floor or is wrong.
- **From all three sets alike:** country of residence. The style exemplars carry British
  spelling, so the floor leaks it too.

### Second run: rebalancing toward everyday life (2026-09-27)

The proportional set is **82% ML study and side projects**: 67 of 87 usable clusters, as
classified by Claude from the abstracted skeletons. That mirrors the private list faithfully,
but as a benchmark of *personal* tasks it under-represents career and everyday life. The
owner asked for ~500 more everyday tasks. `synth_personal.py oversample` added **485**:

- **293 from the 19 non-ML skeletons** (career, life admin, general hobbies, health,
  travel): square-root allocation, at most 20 per cluster. The one *distinctive hobby*
  cluster is never oversampled.
- **192 from 16 generic life-area skeletons** written from a fixed list of life areas,
  with **no private input**: household admin, finances, appointments, family, workplace
  admin, and so on.

The new tasks pass the same gates, plus a near-duplicate filter (cosine > 0.95 to an
accepted synthetic task, which fired 4 times). Every record has `meta.origin`
(`skeleton` / `oversample` / `life_area`) and `meta.domain`, so the proportional subset
(`origin == "skeleton"`) stays available for eval-set-fidelity measurements.

Re-audit of the 1,506-task set:

- canaries 0 / 30;
- NN-similarity quantiles unchanged (q99 0.907 vs. holdout 0.930); DCR share 0.522;
- topic JS divergence 0.211, up from 0.198: the intended, deliberate shift.

**The generic life-area tasks act as chaff.** The attacker's guesses for relationship
status and health conditions moved *away* from the truth, to confident but wrong profiles
drawn from the invented everyday tasks. Its location guess also changed to a wrong region.
Profession and hobbies are still inferred, as before. Diluting a released set with
plausible, unrelated personal tasks is a cheap, measurable defence against
attribute inference. It deserves a proper ablation: attacker accuracy vs. chaff ratio.

**Open decision.** Topic-level attributes (profession, hobbies) are the utility–privacy
frontier: removing them means changing the topic mix. The candidate mitigation is *topic
substitution*: re-skin distinctive hobby clusters as other hobbies of the same broad kind,
keeping task shapes and the variety of icon concepts. It is left to the owner, who decides
which attributes are acceptable to reveal.

## 6. Related work and follow-ups

- **Lison, Pilán, Sánchez, Batet, Øvrelid (2021).** *Anonymisation Models for Text Data: State
  of the Art, Challenges and Future Directions.* ACL. Distinguishes de-identification from
  anonymization; explains quasi-identifiers and why NER masking is insufficient.
- **Pilán et al. (2022).** *The Text Anonymization Benchmark (TAB).* Computational Linguistics
  48(4). Its identifier taxonomy (direct vs. quasi) and recall-style metrics are a model for
  annotating a private audit subset.
- **Staab, Vero, Balunović, Vechev (2024).** *Beyond Memorization: Violating Privacy via
  Inference with Large Language Models.* ICLR. The attacker used in §4.
- **Staab et al. (2025).** *Large Language Models are Advanced Anonymizers.* ICLR.
  Adversarial rewrite loop; a stronger per-item baseline to compare against.
- **Dou et al. (2024).** *Reducing Privacy Risks in Online Self-Disclosures with Language
  Models.* ACL. Self-disclosure abstraction, used in skeletons.
- **Utpala, Hooker, Chen (2023).** *Locally Differentially Private Document Generation Using
  Zero Shot Prompting* (DP-Prompt). Findings of EMNLP. Per-item DP paraphrasing. For 5–15-word
  tasks the ε is too loose to mean much; better as stylometry scrubbing.
- **Yue et al. (2023).** *Synthetic Text Generation with Differential Privacy: A Simple and
  Practical Recipe.* ACL. DP fine-tuning of a generator; with ~3k records the noise would
  destroy utility.
- **Xie et al. (2024).** *Differentially Private Synthetic Data via Foundation Model APIs 2:
  Text* (Aug-PE). ICML. **Planned follow-up.** Private tasks only cast DP-noised
  nearest-neighbour votes over API-generated candidates, and the text never enters a prompt.
  This would turn step 4 into a formal (ε, δ) guarantee. Its behaviour on very short texts
  is under-studied.
- **Meeus et al. (2025).** *The Canary's Echo: Auditing Privacy Risks of LLM-Generated
  Synthetic Text.* ICML. Canary design for synthetic-data-only membership auditing.
- **Pillutla et al. (2021).** *MAUVE.* NeurIPS. A distribution-level utility metric; a
  possible addition to the topic JS divergence.

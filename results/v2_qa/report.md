# Blind QA of glyph readings: v1 vs v2

485 glyphs, 970 of 970 (glyph, version) judgments. Judge: a Claude Opus subagent, blind to the version (slots shuffled per glyph, one common format). v2 is judged after the contrast pass and before `redescribe`.

## Random sample (kept by v1 or v2): 200 glyphs

| Version | correct | partly | wrong | wrong, 95% CI | wrong or partly, 95% CI |
|---|---|---|---|---|---|
| v1 | 146 | 49 | 5 | 2.5% (1.1%–5.7%) | 27.0% (21.3%–33.5%) |
| v2 | 199 | 1 | 0 | 0.0% (0.0%–1.9%) | 0.5% (0.1%–2.8%) |

## Doubtful glyphs: 302 glyphs

| Version | correct | partly | wrong | wrong, 95% CI | wrong or partly, 95% CI |
|---|---|---|---|---|---|
| v1 | 103 | 124 | 75 | 24.8% (20.3%–30.0%) | 65.9% (60.4%–71.0%) |
| v2 | 285 | 13 | 4 | 1.3% (0.5%–3.4%) | 5.6% (3.5%–8.8%) |

## v2 verdicts by reason for sampling

| Reason | correct | partly | wrong |
|---|---|---|---|
| low_recognizability | 146 | 10 | 4 |
| random | 199 | 1 | 0 |
| review_override | 29 | 1 | 2 |
| v1_v2_disagree | 127 | 3 | 0 |

Re-described after QA: 18 glyphs (v2 judged wrong or partly).

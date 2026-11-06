# v2 pilot: 50 glyphs × 5 configs

Glyph reading (blind verdicts; 95% Wilson CI):

| config | model | judged | correct | partly | wrong | wrong CI | partly+wrong CI |
|---|---|---|---|---|---|---|---|
| 3-flash | gemini-3-flash-preview | 50 | 44 | 4 | 2 | 1%–13% | 6%–24% |
| 3.8-flash | gemini-3.8-flash | 50 | 46 | 3 | 1 | 0%–10% | 3%–19% |
| 3.8-flash-low | gemini-3.8-flash | 50 | 47 | 2 | 1 | 0%–10% | 2%–16% |
| 3.1-pro | gemini-3.1-pro-preview | 50 | 49 | 1 | 0 | 0%–7% | 0%–10% |
| 4-argon | gemini-4-argon | 50 | 50 | 0 | 0 | 0%–7% | 0%–7% |

Output quality:

| config | ok/failed | retries | soft problems/glyph | examples | 2-5 / 6-12 / 13-16 / other (target 31/50/19/0) | reused strings | domains (median) | traps/glyph | text realism / fit / near-miss |
|---|---|---|---|---|---|---|---|---|---|
| 3-flash | 50/0 | 0 | 0.3 | 16.0 | 45% / 40% / 11% / 4% | 6 | 6.5 | 3.7 | 3.5 / 2.5 / 2.6 |
| 3.8-flash | 50/0 | 0 | 0.6 | 16.4 | 44% / 50% / 6% / 0% | 0 | 5.0 | 4.0 | 4.1 / 3.2 / 3.7 |
| 3.8-flash-low | 50/0 | 0 | 1.2 | 16.1 | 46% / 54% / 0% / 0% | 1 | 6.0 | 3.6 | 4.1 / 3.6 / 3.9 |
| 3.1-pro | 50/0 | 0 | 0.2 | 16.0 | 34% / 47% / 19% / 1% | 1 | 6.0 | 3.9 | 3.5 / 3.3 / 3.3 |
| 4-argon | 50/0 | 0 | 0.2 | 16.4 | 38% / 44% / 17% / 1% | 2 | 6.5 | 4.4 | 4.7 / 3.6 / 4.2 |

Agreement with the manual icon review, and contamination:

| config | rejects discarded | useful kept | task_fit useful / unsure / reject | eval matches: exact / cos≥0.95 |
|---|---|---|---|---|
| 3-flash | 1/6 | 16/16 | 4 / 3.71 / 3.33 | 0 / 10 |
| 3.8-flash | 3/6 | 15/16 | 3.88 / 3.14 / 2.17 | 1 / 9 |
| 3.8-flash-low | 3/6 | 16/16 | 3.56 / 3.32 / 2.33 | 0 / 4 |
| 3.1-pro | 3/6 | 15/16 | 3.56 / 2.89 / 2.17 | 0 / 4 |
| 4-argon | 5/6 | 15/16 | 3.69 / 3.04 / 1.5 | 0 / 2 |

Cost (prices read 2026-11-05; pass 1 = 3621 glyphs):

| config | input tok | output tok | thinking tok | seconds | $/glyph | pass 1 $ | pass 1 batch $ |
|---|---|---|---|---|---|---|---|
| 3-flash | 3803 | 1186 | 2561 | 17 | 0.0131 | 48 | 24 |
| 3.8-flash | 3803 | 1397 | 1327 | 24 | 0.0131 | 47 | 24 |
| 3.8-flash-low | 3803 | 1347 | 0 | 12 | 0.0079 | 29 | 14 |
| 3.1-pro | 3803 | 1410 | 3797 | 50 | 0.0701 | 254 | 127 |
| 4-argon | 3923 | 1627 | 36 | 50 | 0.0245 | 89 | 44 |

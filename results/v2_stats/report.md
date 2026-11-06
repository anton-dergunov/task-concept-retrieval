# v2 icon descriptions: statistics vs v1

One description per distinct glyph. v1: 3901 glyphs; v2: 3619 glyphs (the manual review's rejects were not described). Passes: describe 2653, describe+contrast 948, describe+qa 15, describe+qa+contrast 3.

## Kept vs discarded, by manual review level

| Level | v1 kept | v1 discarded | v2 kept | v2 discarded |
|---|---|---|---|---|
| useful | 824 | 8 | 814 | 18 |
| somewhat | 1 | 12 | 0 | 12 |
| unsure | 2475 | 301 | 2018 | 757 |
| reject | 186 | 94 | 0 | 0 |

The loader keeps v2-discarded glyphs that the review calls useful or somewhat (the human label wins).

## Text

| Measure | v1 | v2 |
|---|---|---|
| kept glyphs | 3486 | 2832 |
| example tasks | 53178 | 46010 |
| examples per glyph (median) | 15.0 | 16.0 |
| words per example (median / p90) | (4.0, 6) | (9.0, 14) |
| length mix 2-5 / 6-12 / 13-16 / other | (0.853, 0.147, 0.0, 0.0) | (0.365, 0.461, 0.17, 0.004) |
| distinct examples | 40390 | 44363 |
| examples used by >1 glyph | 5671 | 1087 |
| most reused example (glyphs) | 70 | 14 |
| negatives | 24874 | 17425 |
| distinct negatives | 3231 | 16517 |
| top-20 negatives' share | 0.356 | 0.009 |

Target length mix: 5 / 8 / 3 of 16 = 0.31 / 0.50 / 0.19. v1 has no ES/RU examples; v2 adds 3 + 3 per glyph (not counted above).

## v2 only

- Example-task domains: entropy 3.68 bits, 0.92 of the maximum; home 7552, tech 5982, work 5751, social_family 3374, shopping 3086, admin 2956, travel 2865, health_fitness 2826, hobbies 2537, finance 2522, creative 1839, learning 1733, outdoors 1202, food 610, career 588, communication 587.
- Scores of kept glyphs (1–5): recognizability 1:0 2:135 3:506 4:1137 5:1054; task_fit 1:0 2:265 3:1252 4:1141 5:174; distinctiveness 1:229 2:326 3:654 4:1326 5:297.
- Soft problems: 655 of 3619 glyphs (18%): example_tasks 457, example_tasks with parent 151, near_misses use 2 traps (< 3) 39, domains 13, example_tasks_es 6, task_intents 3, near_misses use 1 traps (< 3) 2, example_tasks_es contains Cyrillic 1, example_tasks_ru 1.
- Contamination guard dropped 443 strings: near_misses 223, example_tasks 117, context_tasks 50, example_tasks_es 32, example_tasks_ru 21.
- Tokens: 14,201,725 in, 5,945,667 out; ≈ $43.93 at the batch price.

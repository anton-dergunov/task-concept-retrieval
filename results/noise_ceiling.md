# Judge-noise ceiling (argon-experiments.md #6)

Dev tasks of the Argon-judged eval sets, re-judged by the same judge (prompt j4, gemini-4-argon) in a fresh run with the candidates in another order. *Judge as a method*: the re-judge's ranking scored against the original gold; *model*: F1@all-large-all3k.v2+parent on the same tasks, plain gold.

| set | n | same top-1 | re-judge top-1 a hit | original top-1 a hit | no-good-icon agree | grade κ (quadratic) | judge as a method nDCG@5 / hit@1 | model nDCG@5 / hit@1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| personal | 150 | 0.74 | 0.95 | 0.94 | 1.00 | 0.33 | 0.870 / 0.95 | 0.491 / 0.61 |
| personal_synth | 75 | 0.72 | 0.97 | 0.97 | 1.00 | 0.36 | 0.903 / 0.97 | 0.490 / 0.63 |
| realistic | 75 | 0.73 | 0.97 | 0.97 | 0.99 | 0.39 | 0.907 / 0.97 | 0.534 / 0.69 |
| pooled | 300 | 0.73 | 0.96 | 0.96 | 1.00 | 0.35 | 0.888 / 0.96 | 0.502 / 0.63 |

## Does the leaderboard survive the re-judge?

Every stored method (75) on the 300 re-judged tasks, nDCG@5 against the original gold and against the re-judge's. Spearman ρ of the two method orders: **0.994**.

| method | nDCG@5 original | nDCG@5 re-judge | change |
|---|---:|---:|---:|
| F1@all-large-all3k.v2+parent | 0.502 | 0.492 | -0.010 |
| F1@all-large-e8.v2+parent | 0.493 | 0.486 | -0.007 |
| F1@all-large.v2+parent | 0.484 | 0.478 | -0.006 |
| F1@all-large-all3k.v2 | 0.470 | 0.456 | -0.014 |
| F1@all-lr-para-all3k.v2+parent | 0.463 | 0.455 | -0.008 |
| F1@all-large-e8.v2 | 0.463 | 0.459 | -0.004 |
| F1@all-kd-final.v2+parent | 0.462 | 0.463 | +0.001 |
| F1@all-kd-final.v2 | 0.452 | 0.450 | -0.002 |
| F1@all-lr-para.v2+parent | 0.449 | 0.438 | -0.011 |
| F1@all-large.v2 | 0.447 | 0.445 | -0.002 |
| F1@all-base-lr.v2+parent | 0.446 | 0.440 | -0.006 |
| F1@all-lr-para-all3k.v2 | 0.441 | 0.439 | -0.002 |
| M3.v2+parent | 0.300 | 0.297 | -0.003 |
| M3.v2 | 0.296 | 0.294 | -0.002 |
| M1.v2+parent | 0.248 | 0.246 | -0.002 |
| M1.v2 | 0.247 | 0.245 | -0.002 |
| B0.v2 | 0.236 | 0.236 | +0.000 |
| B1.v2 | 0.219 | 0.211 | -0.008 |
| M1.v1+parent | 0.172 | 0.172 | -0.000 |
| M3.v1 | 0.165 | 0.165 | +0.001 |
| M3.v1+parent | 0.159 | 0.159 | +0.000 |
| M1.v1 | 0.158 | 0.162 | +0.004 |
| B0.v1 | 0.157 | 0.160 | +0.003 |
| B1.v1 | 0.100 | 0.101 | +0.001 |
| M2.v1 | 0.087 | 0.089 | +0.001 |
| M2.v2 | 0.070 | 0.069 | -0.001 |

Over all 2775 pairs of methods: judge noise is a median **11%** of the per-task variance of a paired ΔnDCG@5 (the rest is which tasks happen to be in the set). A bootstrap over tasks resamples each task with its one label, so its interval already covers that noise. On the full pooled dev set (n = 3230), judge noise alone gives a paired Δ a standard deviation of a median **0.0015** (90% of pairs ≤ 0.0018).

Sign of the mean Δ on these tasks, original gold vs re-judge, by |Δ| under the original:

| \|Δ\| | pairs | same sign |
|---|---:|---:|
| 0.000–0.005 | 139 | 0.65 |
| 0.005–0.010 | 130 | 0.91 |
| 0.010–0.020 | 200 | 1.00 |
| 0.020–1.000 | 2306 | 1.00 |

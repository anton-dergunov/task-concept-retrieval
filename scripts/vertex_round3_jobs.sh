#!/usr/bin/env bash
# Round 3's Gemini jobs (docs/next-data-and-distillation.md; hand-off docs/vertex-round3-run.md).
#
#   v1  (any time; ~$45): the translated eval set (7 languages, eval only), rewrites of the
#       judged pool tasks, ~10k new Personal-style candidate tasks; Flash, Vertex batch
#   grade (after round 3 is scored; ~$10): Argon/Flash grades of the leaders' unjudged
#       top-1s on dev; set GRADE_METHODS to the method labels (silver table names)
#   v2  (after Kaggle round 4 and select_active.py; ~$65, before 2026-11-30): Argon judges
#       the selected tasks, then their ES/RU translations and parent headings
#
# Safe to rerun at any time: every step skips what is done, and a submitted batch job is
# never resubmitted. The budget cap is fixed on the first run (data/private/augment/
# cap-round3.json; delete it to set it again).
#
# Run (from the repo root, unattended, hours):
#   CREDIT_LEFT=<USD under Billing → Credits> scripts/vertex_round3_jobs.sh v1
#   CREDIT_LEFT=… GRADE_METHODS=F1@all-large.v2+parent,… scripts/vertex_round3_jobs.sh grade
#   CREDIT_LEFT=… scripts/vertex_round3_jobs.sh v2
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
A="$PY scripts/augment_pools.py"
B="$PY scripts/judge_batch.py"
G="$PY scripts/grade_unjudged.py"
POOL=data/private/train_pools/personal_synth_pool.jsonl
CAND=data/private/train_pools/personal_synth_cand.jsonl
SEL_SYNTH=data/private/train_pools/personal_synth_pool3.jsonl
SEL_MS=data/train_pools/mslatte2.jsonl
STAGE="${1:?usage: $0 v1|grade|v2}"
: "${CREDIT_LEFT:?set CREDIT_LEFT to the credit left in USD (Billing → Credits)}"
# Only the account and project pinned in .env (tcr/gcp.py); stop before any spend if not.
$PY scripts/gcp_check.py || { echo "GCP preflight failed: nothing submitted"; exit 1; }
BUDGET="${BUDGET:-100}"
V2_RESERVE="${V2_RESERVE:-75}"      # kept for v2 + grading while v1 sizes the candidates
N_CAND_MAX="${N_CAND_MAX:-10000}"
GEN_PER_TASK=0.0031                 # measured, augment_pools.py

CAP=$($A cap --credit-left "$CREDIT_LEFT" --reserve "${RESERVE:-5}" --budget "$BUDGET" --tag round3)
spent() { $PY -c 'from tcr import judge; print(round(judge.spent(), 2))'; }
echo "=== budget: stop when the spend log reaches \$$CAP (now \$$(spent))"

case "$STAGE" in
v1)
  echo "=== 1. the translated eval set (dev tasks of the public eval sets; eval only)"
  for RUN in public-short-j4 public-expanded-j4 realistic-j4; do
    $A translate --eval --judge-run "$RUN" --max-usd "$CAP"
  done
  $A xling-build

  echo "=== 2. rewrites of the judged training tasks"
  for RUN in pool-synth-j4 pool-synth2-j4 pool-mslatte-j4; do
    $A rewrite --judge-run "$RUN" --max-usd "$CAP"
  done

  echo "=== 3. new Personal-style candidate tasks (unlabelled; teacher scores + selection)"
  if [ ! -f "$CAND" ]; then
    LEFT=$($PY -c "print(max(0.0, $CAP - $(spent) - $V2_RESERVE))")
    K=$($PY -c "print(min($N_CAND_MAX, int($LEFT / $GEN_PER_TASK)))")
    if [ "$K" -lt 1000 ]; then
      echo "the budget buys only $K candidates; skipping the generation"
    else
      [ -f "${POOL%.jsonl}.before-cand.jsonl" ] || cp "$POOL" "${POOL%.jsonl}.before-cand.jsonl"
      N=$(( $(wc -l < "${POOL%.jsonl}.before-cand.jsonl") + K + K / 10 ))
      echo "generating ~$K new tasks (pool --n $N)"
      $PY scripts/datasets/synth_personal.py pool --n "$N" --rounds 4
      $A extra --out "$CAND"
    fi
  fi
  echo "v1 done. Next: f1_data.py build, upload, Kaggle round 4 (docs/f1-kaggle-run.md)."
  ;;
grade)
  : "${GRADE_METHODS:?set GRADE_METHODS to the silver method labels to grade}"
  NAME="${GRADE_NAME:-g1}"
  [ -f "data/private/judge/grade/$NAME/meta.json" ] || $G prepare --name "$NAME" --methods "$GRADE_METHODS"
  $G run --name "$NAME" --max-usd "$CAP"
  echo "grade done. Score with: silver_eval.py --methods … --extra-grades $NAME"
  ;;
v2)
  [ -f "$SEL_SYNTH" ] || [ -f "$SEL_MS" ] || { echo "run scripts/select_active.py first"; exit 1; }
  for spec in "pool-synth3-j4|$SEL_SYNTH|personal_synth_pool|--private" "pool-mslatte2-j4|$SEL_MS|train_mslatte|"; do
    IFS='|' read -r RUN FILE NAME PRIV <<< "$spec"
    [ -f "$FILE" ] || continue
    echo "=== judge $RUN ($(wc -l < "$FILE") tasks, Argon)"
    if [ ! -f "data/private/judge/batch/$RUN/meta.json" ]; then
      # shellcheck disable=SC2086
      $B prepare --run "$RUN" --pool "$FILE" --name "$NAME" $PRIV --model gemini-4-argon \
         --extra-candidates "${FILE%.jsonl}.extra.json"
    fi
    $B run --run "$RUN" --max-usd "$CAP"
    $B export --run "$RUN"
    echo "=== translations + parent headings of $RUN"
    if [ "$NAME" = personal_synth_pool ]; then
      $A run --judge-run "$RUN" --gen-parents --max-usd "$CAP"
    else
      $A run --judge-run "$RUN" --max-usd "$CAP"
    fi
  done
  echo "v2 done. Next: add the runs to tcr/silver.py POOL_RUNS, f1_data.py build, Kaggle round 5."
  ;;
*) echo "usage: $0 v1|grade|v2"; exit 1 ;;
esac
echo "spend log: \$$(spent) (cap \$$CAP)"

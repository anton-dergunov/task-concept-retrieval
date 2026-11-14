#!/usr/bin/env bash
# The last planned Gemini jobs (docs/next-label-free-work.md step 0), in order:
#   1. Spanish/Russian translations of the two judged pools, and parent headings for the
#      Personal-style pool (Gemini 3.8 Flash, Vertex batch; ~$5)
#   2. a larger Personal-style pool: generate more tasks (Flash, online), judge the new
#      ones (Argon, batch), as many as the credit left buys
#   3. translations and parent headings of the new tasks
#
# Safe to rerun at any time: every step skips what is done, and a submitted batch job is
# never resubmitted. The budget cap is fixed on the first run (data/private/augment/cap.json;
# delete that file to set it again).
#
# Run (from the repo root, ~hours, unattended):
#   CREDIT_LEFT=<USD under Billing → Credits> scripts/gemini_last_jobs.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
A="$PY scripts/augment_pools.py"
B="$PY scripts/judge_batch.py"
POOL=data/private/train_pools/personal_synth_pool.jsonl
EXTRA=data/private/train_pools/personal_synth_pool_extra.jsonl
RUN2=pool-synth2-j4
: "${CREDIT_LEFT:?set CREDIT_LEFT to the credit left in USD (Billing → Credits)}"

$A backfill
CAP=$($A cap --credit-left "$CREDIT_LEFT" --reserve "${RESERVE:-8}")
echo "=== budget: stop when the spend log reaches \$$CAP"

echo "=== 1. translations + parent headings of the judged pools"
$A run --judge-run pool-synth-j4 --gen-parents --max-usd "$CAP"
$A run --judge-run pool-mslatte-j4 --max-usd "$CAP"

echo "=== 2. a larger Personal-style pool"
if [ ! -f "data/private/judge/batch/$RUN2/meta.json" ]; then
  if [ ! -f "$EXTRA" ]; then
    K=$($A plan-extra --cap "$CAP")
    if [ "$K" -lt 200 ]; then
      echo "the budget buys only $K more tasks; skipping the pool extension"
      K=0
    else
      [ -f "${POOL%.jsonl}.before-extend.jsonl" ] || cp "$POOL" "${POOL%.jsonl}.before-extend.jsonl"
      N=$(( $(wc -l < "${POOL%.jsonl}.before-extend.jsonl") + K + K / 10 ))
      $PY scripts/datasets/synth_personal.py pool --n "$N" --rounds 4
      $A extra --out "$EXTRA"
    fi
  fi
  if [ -f "$EXTRA" ]; then
    K=$($A plan-extra --cap "$CAP" --generated)
    [ "$K" -ge 1 ] || { echo "no budget left to judge the new tasks"; exit 1; }
    $B prepare --run "$RUN2" --pool "$EXTRA" --name personal_synth_pool --private \
       --model gemini-4-argon --limit "$K"
  fi
fi
if [ -f "data/private/judge/batch/$RUN2/meta.json" ]; then
  $B run --run "$RUN2" --max-usd "$CAP"
  $B export --run "$RUN2"
  echo "=== 3. translations + parent headings of the new tasks"
  $A run --judge-run "$RUN2" --gen-parents --max-usd "$CAP"
fi
echo "all done; spend log: \$$($PY -c 'from tcr import judge; print(round(judge.spent(), 2))') (cap \$$CAP)"

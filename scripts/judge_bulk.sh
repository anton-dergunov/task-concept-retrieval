#!/usr/bin/env bash
# Run the LLM judge over every remaining dataset and pool, one after another, in priority
# order (docs/judge-bulk-run.md). Safe to rerun at any time: a prepared run is not prepared
# again, a submitted job is not resubmitted, a finished run is only exported again.
#
# The --max-usd caps are cumulative over all runs (the shared spend log), so the budget
# stops the least valuable runs at the end first.
#
# Run: scripts/judge_bulk.sh            (from the repo root; ~hours, unattended)
set -euo pipefail
cd "$(dirname "$0")/.."
B=".venv/bin/python scripts/judge_batch.py"

# run NAME CAP PREPARE-ARGS...
run() {
  local name=$1 cap=$2; shift 2
  echo "=== $name (cap \$$cap) ==="
  if [ ! -f "data/private/judge/batch/$name/meta.json" ]; then
    $B prepare --run "$name" "$@"
  fi
  $B run --run "$name" --max-usd "$cap"
  $B export --run "$name"
}

run personal-j4        90  --dataset personal --model gemini-4-argon
run pool-synth-j4      190 --pool data/private/train_pools/personal_synth_pool.jsonl \
                           --name personal_synth_pool --private --model gemini-4-argon
run synth-j4           220 --dataset personal_synth --model gemini-4-argon
run realistic-j4       235 --dataset realistic --model gemini-4-argon
run public-short-j4    245 --dataset public_short --model gemini-3.8-flash
run public-expanded-j4 250 --dataset public_expanded --model gemini-3.8-flash
run pool-mslatte-j4    280 --pool data/train_pools/mslatte.jsonl --name train_mslatte \
                           --model gemini-3.8-flash
echo "all runs finished; spend: $(.venv/bin/python -c 'from tcr import judge; print(round(judge.spent(), 2))') USD"

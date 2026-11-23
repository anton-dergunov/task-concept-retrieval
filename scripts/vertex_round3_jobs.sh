#!/usr/bin/env bash
# Round 3's Gemini jobs (docs/next-data-and-distillation.md; hand-off docs/vertex-round3-run.md).
#
#   v1  (any time; ~$45): the translated eval set (7 languages, eval only), rewrites of the
#       judged pool tasks, ~10k new Personal-style candidate tasks; Flash, Vertex batch
#   grade (after round 3 is scored; ~$10): Argon/Flash grades of the leaders' unjudged
#       top-1s on dev; set GRADE_METHODS to the method labels (silver table names)
#   v2  (after Kaggle round 4 and select_active.py; ~$65, before 2026-11-30): Argon judges
#       the selected tasks, then their ES/RU translations, parent headings and rewrites
#   v3a (round 6, batch 1; ~$30, before 2026-11-30; argon-experiments.md "Round 6's
#       choice"): #6 re-judge 300 eval dev tasks (the judge-noise ceiling) and #3 minimal
#       pairs (Flash edits ~1,000 judged tasks, Argon judges the edits). Own cap:
#       CAP_TAG=round6 BUDGET=100 (the author's approval of 2026-11-22)
#   v3b (round 6, batch 2; ~$85, before 2026-11-30; argon-experiments.md "Batch 2"): g5
#       (the leaders' unjudged top-10 icons on dev, graded), #5 VLM reranker (Argon ranks
#       the big model's top 20 as glyph images: dev, 1,000 judged training tasks, 3,000
#       unjudged candidates), #2 adversarial tasks (Flash writes 4 tasks per big-model
#       error on validation tasks, Argon judges them), then g6 (the VLM's unjudged top 3).
#       Own cap: CAP_TAG=round6b BUDGET=100 (the author's approval of 2026-11-23). Mac
#       steps run on one thread (TCR_F1_THREADS=1, nice)
#
# Safe to rerun at any time: every step skips what is done, and a submitted batch job is
# never resubmitted. The budget cap is fixed on the first run (data/private/augment/
# cap-round3.json; delete it to set it again).
#
# Run (from the repo root, unattended, hours):
#   CREDIT_LEFT=<USD under Billing → Credits> scripts/vertex_round3_jobs.sh v1
#   CREDIT_LEFT=… GRADE_METHODS=F1@all-large.v2+parent,… scripts/vertex_round3_jobs.sh grade
#   (later rounds: GRADE_NAME=g2 …; picks graded in earlier sets are skipped)
#   CREDIT_LEFT=… scripts/vertex_round3_jobs.sh v2
#   CREDIT_LEFT=155 CAP_TAG=round6 BUDGET=100 scripts/vertex_round3_jobs.sh v3a
#   CREDIT_LEFT=130 CAP_TAG=round6b BUDGET=100 scripts/vertex_round3_jobs.sh v3b
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
STAGE="${1:?usage: $0 v1|grade|v2|v3a|v3b}"
: "${CREDIT_LEFT:?set CREDIT_LEFT to the credit left in USD (Billing → Credits)}"
# Only the account and project pinned in .env (tcr/gcp.py); stop before any spend if not.
$PY scripts/gcp_check.py || { echo "GCP preflight failed: nothing submitted"; exit 1; }
BUDGET="${BUDGET:-100}"
V2_RESERVE="${V2_RESERVE:-75}"      # kept for v2 + grading while v1 sizes the candidates
N_CAND_MAX="${N_CAND_MAX:-10000}"
GEN_PER_TASK=0.0031                 # measured, augment_pools.py

CAP=$($A cap --credit-left "$CREDIT_LEFT" --reserve "${RESERVE:-5}" --budget "$BUDGET" --tag "${CAP_TAG:-round3}")
spent() { $PY -c 'from tcr import judge; print(round(judge.spent(), 2))'; }
echo "=== budget: stop when the spend log reaches \$$CAP (now \$$(spent))"

# Background jobs of a stage (v3a, v3b): one log each in $LOGS; waitall stops on a failure.
PIDS=()
bg() { local name=$1; shift; "$@" > "$LOGS/$name.log" 2>&1 & PIDS+=($!); echo "  started $name (log $LOGS/$name.log)"; }
waitall() { local bad=0; for p in ${PIDS[@]+"${PIDS[@]}"}; do wait "$p" || bad=1; done; PIDS=();
            [ "$bad" = 0 ] || { echo "a job failed: read $LOGS/*.log, then rerun (done steps are skipped)"; exit 1; }; }

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
  # Picks graded in earlier sets (extra_grades/*.jsonl) are not graded again.
  SKIP=$(ls data/private/judge/extra_grades/*.jsonl 2>/dev/null | xargs -n1 basename 2>/dev/null \
         | sed 's/\.jsonl$//' | grep -vx "$NAME" | paste -sd, - || true)
  [ -f "data/private/judge/grade/$NAME/meta.json" ] || \
    $G prepare --name "$NAME" --methods "$GRADE_METHODS" --skip "${GRADE_SKIP-$SKIP}"
  $G run --name "$NAME" --max-usd "$CAP"
  echo "grade done. Score with: silver_eval.py --methods … --extra-grades ${SKIP:+$SKIP,}$NAME"
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
    # Round 4: rewrites with the original's grades were the best cheap text (+0.019 for
    # e5-small), so the new tasks get them too (~$0.5; after the parents, which they use).
    echo "=== rewrites of $RUN"
    $A rewrite --judge-run "$RUN" --max-usd "$CAP"
  done
  echo "v2 done. The runs join the training pools by themselves (tcr/silver.py NEW_POOL_RUNS). Next: f1_data.py build, Kaggle round 5."
  ;;
v3a)
  [ "${CAP_TAG:-round3}" = round6 ] || { echo "v3a spends round 6's money: set CAP_TAG=round6 BUDGET=100"; exit 1; }
  LOGS=data/private/augment/v3a-logs
  mkdir -p "$LOGS"

  echo "=== #6 re-judge dev tasks of the Argon-judged eval sets (another candidate order)"
  [ -f data/private/judge/rejudge/personal.ids ] || $PY scripts/noise_ceiling.py select
  for D in personal personal_synth realistic; do
    RUN="rejudge-$D"
    [ -f "data/private/judge/batch/$RUN/meta.json" ] || \
      $B prepare --run "$RUN" --dataset "$D" --ids "data/private/judge/rejudge/$D.ids" \
         --shuffle-seed 1 --model gemini-4-argon
    bg "$RUN" sh -c "$B run --run $RUN --max-usd $CAP && $B export --run $RUN"
  done
  REJUDGE=(${PIDS[@]+"${PIDS[@]}"}); PIDS=()          # waited for at the end, not before the pairs

  echo "=== #3 minimal pairs: Flash edits judged tasks so the best icon should change"
  bg mp-gen-synth sh -c "$A minpair --judge-run pool-synth-j4 --n ${MP_SYNTH:-600} --max-usd $CAP && \
                         $A minpair --judge-run pool-synth2-j4 --n ${MP_SYNTH2:-150} --max-usd $CAP"
  bg mp-gen-mslatte $A minpair --judge-run pool-mslatte-j4 --n "${MP_MSLATTE:-250}" --max-usd "$CAP"
  waitall
  $A minpair-pool --judge-runs pool-synth-j4,pool-synth2-j4,pool-mslatte-j4

  echo "=== #3 Argon judges the edited tasks (the original's top icons among the candidates)"
  for spec in "mp-synth-j4|data/private/train_pools/personal_synth_pool.mp_pool.jsonl|personal_synth_pool|--private" \
              "mp-mslatte-j4|data/train_pools/train_mslatte.mp_pool.jsonl|train_mslatte|"; do
    IFS='|' read -r RUN FILE NAME PRIV <<< "$spec"
    [ -f "$FILE" ] || continue
    if [ ! -f "data/private/judge/batch/$RUN/meta.json" ]; then
      # shellcheck disable=SC2086
      $B prepare --run "$RUN" --pool "$FILE" --name "$NAME" $PRIV --model gemini-4-argon \
         --extra-candidates "${FILE%.jsonl}.extra.json"
    fi
    bg "$RUN" sh -c "$B run --run $RUN --max-usd $CAP && $B export --run $RUN"
  done
  PIDS+=(${REJUDGE[@]+"${REJUDGE[@]}"})
  waitall
  $PY scripts/noise_ceiling.py report
  echo "v3a done. Next: batch 2 (v3b: #5 VLM reranker, #2 adversarial tasks)."
  ;;
v3b)
  [ "${CAP_TAG:-round3}" = round6b ] || { echo "v3b spends batch 2's money: set CAP_TAG=round6b BUDGET=100"; exit 1; }
  LOGS=data/private/augment/v3b-logs
  mkdir -p "$LOGS"
  export TCR_F1_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
  V="nice -n 19 $PY scripts/vlm_rerank.py"
  ARM="${VLM_ARM:-img+desc}"
  if [ "$ARM" = img+desc ]; then VLM_LABEL="VLM@all-large-all3k.v2+parent"; else VLM_LABEL="VLM-img@all-large-all3k.v2+parent"; fi
  SEED_RUNS=(pool-synth-j4 pool-synth2-j4 pool-mslatte-j4)

  echo "=== the first stage's top 20 (Mac, one thread; skips what is scored)"
  $V topk --sets dev,val,train,cand
  echo "=== batch 2's estimate at measured rates (refuses above the cap)"
  $V budget --cap "$CAP"

  echo "=== g5: grade the leaders' unjudged top-10 icons on dev"
  [ -f data/private/judge/grade/g5/meta.json ] || nice -n 19 $G prepare --name g5 --depth 10 --skip g1,g2,g3,g4 \
      --methods F1@all-large-all3k.v2+parent,F1@all-kd-final.v2+parent
  # run re-logs the spend each time it collects, so a finished grade set is not run again
  [ -f data/private/judge/extra_grades/g5.jsonl ] || bg g5 $G run --name g5 --max-usd "$CAP"

  echo "=== #5 Argon reranks the top 20 (arm $ARM): dev, train, cand"
  for S in dev train cand; do
    [ -f "data/private/vlm/batch/$S/meta.json" ] || $V prepare --set "$S" --arm "$ARM"
    bg "vlm-$S" sh -c "$V run --set $S --arm $ARM --max-usd $CAP && $V export --set $S"
  done

  echo "=== #2 adversarial tasks: Flash writes them from the big model's errors, Argon judges them"
  adversarial() {
    for R in "${SEED_RUNS[@]}"; do $A adversarial --judge-run "$R" --max-usd "$CAP"; done
    nice -n 19 $A adversarial-pool --judge-runs "$(IFS=,; echo "${SEED_RUNS[*]}")"
    for spec in "adv-synth-j4|data/private/train_pools/personal_synth_pool.adv_pool.jsonl|personal_synth_pool|--private" \
                "adv-mslatte-j4|data/train_pools/train_mslatte.adv_pool.jsonl|train_mslatte|"; do
      IFS='|' read -r RUN FILE NAME PRIV <<< "$spec"
      [ -f "$FILE" ] || continue
      if [ ! -f "data/private/judge/batch/$RUN/meta.json" ]; then
        # shellcheck disable=SC2086
        nice -n 19 $B prepare --run "$RUN" --pool "$FILE" --name "$NAME" $PRIV --model gemini-4-argon \
           --extra-candidates "${FILE%.jsonl}.extra.json"
      fi
      $B run --run "$RUN" --max-usd "$CAP"
      $B export --run "$RUN"
      $A run --judge-run "$RUN" --max-usd "$CAP"          # ES/RU translations
      $A rewrite --judge-run "$RUN" --max-usd "$CAP"
    done
  }
  bg adversarial adversarial
  waitall

  echo "=== g6: grade the VLM's unjudged top 3 on dev"
  [ -f data/private/judge/grade/g6/meta.json ] || nice -n 19 $G prepare --name g6 --depth 3 \
      --skip g1,g2,g3,g4,g5 --methods "$VLM_LABEL"
  $G run --name g6 --max-usd "$CAP"

  echo "=== reports"
  $V report
  $A adversarial-report
  echo "v3b done. Score with: nice .venv/bin/python scripts/silver_eval.py --methods F1@all-large-all3k --versions v2 --extra-grades g1,g2,g3,g4,g5,g6"
  ;;
*) echo "usage: $0 v1|grade|v2|v3a|v3b"; exit 1 ;;
esac
echo "spend log: \$$(spent) (cap \$$CAP)"

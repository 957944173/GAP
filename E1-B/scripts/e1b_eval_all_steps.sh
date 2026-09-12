#!/usr/bin/env bash
# E1-B evaluation orchestrator (E1-B_task.md section 6).
#
# Evaluates the E1-B checkpoints with the ORIGINAL GAP evaluation protocol, one after
# another (the 4 GPUs cannot host two evaluations at once), never overwriting an
# existing evaluation directory:
#
#   bash E1-B/scripts/e1b_eval_all_steps.sh                  # step120 first, then 80..110
#   bash E1-B/scripts/e1b_eval_all_steps.sh 120              # only the mandatory step120
#   bash E1-B/scripts/e1b_eval_all_steps.sh 120 80 90 100 110
#
# A failure of one step is recorded and does NOT abort the remaining steps.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E1B_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG="$E1B_ROOT/logs/eval_all_steps.log"
mkdir -p "$E1B_ROOT/logs" "$E1B_ROOT/evaluations"

STEPS=("$@")
if [[ ${#STEPS[@]} -eq 0 ]]; then
    STEPS=(120 80 90 100 110)
fi

{
  echo "===== E1-B evaluation orchestrator $(date -Is) steps: ${STEPS[*]} ====="
  for s in "${STEPS[@]}"; do
      echo "--- evaluating step $s"
      if bash "$SCRIPT_DIR/run_e1b_eval_step.sh" "$s" >>"$E1B_ROOT/logs/eval_step${s}.log" 2>&1; then
          echo "OK   step $s"
      else
          echo "FAIL step $s (rc=$?) -- see logs/eval_step${s}.log"
      fi
  done
  echo "--- collecting artifacts"
  python3 "$SCRIPT_DIR/e1b_collect_evaluation_artifacts.py" \
      --eval-root "$E1B_ROOT/evaluations" --out "$E1B_ROOT/evaluations" || true
  echo "--- multi-way comparison"
  python3 "$SCRIPT_DIR/e1b_compare_all.py" \
      --eval-root "$E1B_ROOT/evaluations" --out "$E1B_ROOT/evaluations/comparison" || true
  echo "===== E1-B evaluation orchestrator DONE $(date -Is) ====="
} 2>&1 | tee -a "$LOG"

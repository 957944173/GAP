#!/usr/bin/env bash
# Wait for the E1-B training run to end, or for a suspected stall, then exit so the
# supervising agent gets a single notification instead of polling.
#
#   exits 0 -> TRAINING_FINISHED (no verl main_ppo process left)
#   exits 2 -> SUSPECTED_STALL  (stdout.log mtime older than STALL_MIN minutes while
#                                the training process is still alive)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E1B_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_DIR="$(cat "$E1B_ROOT/latest_run.txt")"
LOG="$RUN_DIR/stdout.log"
PROG="$E1B_ROOT/logs/wait_progress.log"
STALL_MIN="${STALL_MIN:-40}"
PROGRESS_MIN="${PROGRESS_MIN:-20}"
mkdir -p "$E1B_ROOT/logs"

say() { echo "$@" | tee -a "$PROG"; }

say "$(date -Is) waiter started run=$RUN_DIR stall_min=$STALL_MIN"
last_progress=0
while pgrep -f "verl.trainer.main_ppo" >/dev/null 2>&1; do
    now=$(date +%s)
    mtime=$(stat -c %Y "$LOG" 2>/dev/null || echo "$now")
    age=$(( now - mtime ))
    if (( age > STALL_MIN * 60 )); then
        say "$(date -Is) SUSPECTED_STALL: stdout.log untouched for ${age}s (> $((STALL_MIN*60))s)"
        tail -5 "$LOG" 2>/dev/null | cut -c1-200 | tee -a "$PROG"
        exit 2
    fi
    if (( now - last_progress >= PROGRESS_MIN * 60 )); then
        last_progress=$now
        step=$(grep -oE "step:[0-9]+ - " "$LOG" 2>/dev/null | tail -1 | tr -d ' -')
        ckpts=$(ls -d "$E1B_ROOT"/../experiments/DAPO-GAP3B-MHQA-Agent-E1B-step70to120-4gpu/global_step_* 2>/dev/null | wc -l)
        say "$(date -Is) progress: last=${step:-none} checkpoints=$ckpts log_age=${age}s"
    fi
    sleep 45
done
say "$(date -Is) TRAINING_FINISHED (no main_ppo process)"
exit 0

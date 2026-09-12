#!/usr/bin/env bash
# Wait for the E1-B post-training pipeline to finish (or for a stalled evaluation) and
# exit, so the supervising agent gets one notification instead of polling.
#   exits 0 -> PIPELINE_DONE / pipeline process gone
#   exits 2 -> SUSPECTED_EVAL_STALL (no eval log growth for STALL_MIN minutes while the
#              pipeline is still alive)
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E1B_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROG="$E1B_ROOT/logs/pipeline_wait.log"
STALL_MIN="${STALL_MIN:-45}"
PROGRESS_MIN="${PROGRESS_MIN:-20}"
say() { echo "$@" | tee -a "$PROG"; }
say "$(date -Is) pipeline waiter started"
last_progress=0
while pgrep -f "e1b_post_train_pipeline.sh" >/dev/null 2>&1; do
    now=$(date +%s)
    # newest eval log (the file that must keep growing during an evaluation)
    newest=$(ls -t "$E1B_ROOT"/logs/eval_step*.log 2>/dev/null | head -1)
    if [[ -n "$newest" ]]; then
        age=$(( now - $(stat -c %Y "$newest" 2>/dev/null || echo "$now") ))
        if (( age > STALL_MIN * 60 )); then
            say "$(date -Is) SUSPECTED_EVAL_STALL: $newest untouched for ${age}s"
            tail -5 "$newest" | cut -c1-200 | tee -a "$PROG"
            exit 2
        fi
    else
        age=0
    fi
    if (( now - last_progress >= PROGRESS_MIN * 60 )); then
        last_progress=$now
        done_steps=$(grep -c "^OK   step" "$E1B_ROOT/logs/eval_all_steps.log" 2>/dev/null || echo 0)
        say "$(date -Is) progress: eval_steps_done=$done_steps newest_log=$(basename "${newest:-none}") age=${age}s"
    fi
    sleep 60
done
say "$(date -Is) PIPELINE_FINISHED $(cat "$E1B_ROOT/logs/pipeline_status.txt" 2>/dev/null)"
exit 0

#!/usr/bin/env bash
# E1-E milestone watcher: block (as a background job) until a milestone path exists, then
# print a compact snapshot and exit.  Used to avoid busy-polling a long unattended run.
#   bash E1-E/scripts/e1e_wait_for.sh <path> [timeout_seconds]
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E1E_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TARGET="${1:?usage: e1e_wait_for.sh <path> [timeout_seconds]}"
TIMEOUT="${2:-86400}"
START=$(date +%s)
while true; do
    if [[ -e "$TARGET" ]]; then
        echo "[wait-for] ${TARGET} appeared after $(( $(date +%s) - START ))s at $(date -Is)"
        break
    fi
    if (( $(date +%s) - START > TIMEOUT )); then
        echo "[wait-for] TIMEOUT after ${TIMEOUT}s; ${TARGET} still absent at $(date -Is)"
        exit 2
    fi
    sleep 60
done
echo "--- pipeline status ---"
cat "$E1E_ROOT/logs/pipeline_status.txt" 2>/dev/null || echo "(no pipeline_status.txt)"
cat "$E1E_ROOT/logs/d1_gate_status.txt" 2>/dev/null || true
echo "--- run_all tail ---"
tail -n 5 "$E1E_ROOT/logs/run_all.log" 2>/dev/null | cut -c1-200
echo "--- D1/D2 steps seen ---"
for ph in D1 D2; do
    L=$(ls -1 "$E1E_ROOT"/logs/train_${ph}_*.log 2>/dev/null | tail -1)
    [[ -n "$L" ]] && echo "$ph: $(grep -oE 'step:[0-9]+' "$L" | sort -u | tr '\n' ' ')"
done
echo "--- diagnostics ---"
for ph in D1 D2; do
    R=$(cat "$E1E_ROOT/runs/latest_${ph}.txt" 2>/dev/null) || continue
    [[ -n "$R" ]] && echo "$ph: $(wc -l "$R"/diagnostics/*.jsonl 2>/dev/null | tail -1 | tr -s ' ')"
done
echo "--- checkpoints ---"
ls -d /data01/wyy/Graph-Agent-Planning/experiments/DAPO-GAP3B-MHQA-Agent-E1E-q2-step60to120-4gpu/global_step_* 2>/dev/null | tr '\n' ' '
echo
df -BG --output=avail /data01 | tail -1

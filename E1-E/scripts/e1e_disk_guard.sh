#!/usr/bin/env bash
# =====================================================================================
# E1-E disk recheck / safe cleanup before D2 (E1-E_task.md sections 10 and 25 item 13).
#
# E1-E needs ~216 GB for its own six checkpoints (70 + 80/90/100/110/120, ~36 GB each)
# plus the two FSDP->HF merges the evaluator creates.  This script:
#   1. reports free space;
#   2. if free space is already enough (>= NEED_GB) it does nothing (NO deletion is the
#      default: intermediates of a finished old run are only removed when actually needed);
#   3. otherwise it deletes ONLY the redundant intermediate checkpoints of the finished
#      E1-D run (global_step_{80,90,100,110}; E1-D's final step120 and every evaluation,
#      report, log, statistic and manifest are preserved), appending a full record to
#      E1-E/logs/disk_cleanup.log;
#   4. if space is still insufficient it writes STATUS=BLOCKED_DISK and exits non-zero --
#      it never touches a protected path.
#
# PROTECTED (never deleted here): original GAP step60/120/180/200, base/SFT model,
# E1-0/E1-A step70, E1-B step120, E1-D step70/step120, all code / parquet / eval results /
# reports / logs / manifests, and anything belonging to the running E1-E experiment.
# =====================================================================================
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E1E_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$E1E_ROOT/.." && pwd)"
NEED_GB="${E1E_NEED_GB:-250}"
LOG="$E1E_ROOT/logs/disk_cleanup.log"
STATUS="$E1E_ROOT/logs/disk_guard_status.txt"

free_gb() { df -BG --output=avail "$REPO_ROOT" | tail -1 | tr -dc '0-9'; }

FREE=$(free_gb)
echo "[disk-guard] free=${FREE} GB need=${NEED_GB} GB"
{
    echo "===== E1-E disk guard $(date -Is) ====="
    echo "free_before=${FREE} GB  need=${NEED_GB} GB"
} >> "$LOG"

if (( FREE >= NEED_GB )); then
    echo "[disk-guard] sufficient free space; NO deletion performed"
    echo "free_before=${FREE} GB need=${NEED_GB} GB" >> "$LOG"
    echo "STATUS=OK free_gb=$FREE deleted=none" > "$STATUS"
    exit 0
fi

E1D_EXP="$REPO_ROOT/experiments/DAPO-GAP3B-MHQA-Agent-E1D-step60to120-4gpu"
# guard: only ever delete when the finished run's final checkpoint is present
if [[ ! -d "$E1D_EXP/global_step_120" ]]; then
    echo "[disk-guard] BLOCKED: $E1D_EXP/global_step_120 missing; refusing to delete anything" >&2
    echo "STATUS=BLOCKED_DISK reason=no_final_checkpoint free_gb=$FREE" > "$STATUS"
    exit 3
fi
DELETED=0
for s in 80 90 100 110; do
    D="$E1D_EXP/global_step_${s}"
    [[ -d "$D" ]] || continue
    SZ=$(du -sb "$D" | cut -f1)
    MT=$(stat -c %y "$D")
    rm -rf "$D"
    DELETED=$((DELETED + SZ))
    printf 'deleted %s  bytes=%s  mtime=%s  experiment=E1-D(finished)  reason=disk guard before E1-E D2; final global_step_120, all evaluations, statistics, manifests and reports preserved\n' \
        "$D" "$SZ" "$MT" >> "$LOG"
    echo "[disk-guard] deleted $D ($SZ bytes)"
done
FREE2=$(free_gb)
{
    echo "freed_bytes=${DELETED}  free_after=${FREE2} GB"
    echo "protected paths untouched: GAP step60/120/180/200, base model, E1-0/E1-A step70, E1-B step120, E1-D step70/step120, E1-E run dir"
    echo "scientific variable changed: no"
} >> "$LOG"

if (( FREE2 < NEED_GB )); then
    echo "[disk-guard] BLOCKED: still only ${FREE2} GB free after safe cleanup (< ${NEED_GB} GB)" >&2
    echo "STATUS=BLOCKED_DISK free_gb=$FREE2 freed_bytes=$DELETED" > "$STATUS"
    exit 4
fi
echo "STATUS=OK free_gb=$FREE2 deleted_bytes=$DELETED" > "$STATUS"
echo "[disk-guard] free=${FREE2} GB after cleanup"

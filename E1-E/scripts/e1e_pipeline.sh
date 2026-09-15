#!/usr/bin/env bash
# E1-E unattended pipeline: waits for D1 to end, then runs
#   D1 integrity -> step70 eval -> D2 training -> D2 integrity -> step120 eval -> analysis
# Each stage gates the next; a failure writes STATUS=BLOCKED and stops the chain.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E1E_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$E1E_ROOT/.." && pwd)"
LOG="$E1E_ROOT/logs/pipeline.log"
STATUS="$E1E_ROOT/logs/pipeline_status.txt"
mkdir -p "$E1E_ROOT/logs"
say() { echo "$(date -Is) $*" | tee -a "$LOG"; }
say "pipeline started; waiting for D1 to finish"
while pgrep -f "verl.trainer.main_ppo" >/dev/null 2>&1; do sleep 60; done
say "D1 training process gone"

source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate parallel-agent
cd "$REPO_ROOT"

E1E_PHASE=D1 python3 "$SCRIPT_DIR/e1e_integrity.py" >>"$LOG" 2>&1 || { say "BLOCKED: D1 integrity failed"; echo "STATUS=BLOCKED_D1_INTEGRITY" > "$STATUS"; exit 3; }
say "D1 integrity PASS"

bash "$SCRIPT_DIR/run_e1e_eval_70.sh" >>"$LOG" 2>&1 || { say "BLOCKED: step70 eval failed"; echo "STATUS=BLOCKED_EVAL70" > "$STATUS"; exit 4; }
say "step70 eval done"

if ! python3 "$SCRIPT_DIR/e1e_d1_gate.py" >>"$LOG" 2>&1; then
    say "BLOCKED: D1 -> D2 gate FAILED (see E1-E/analysis/d1_gate.md)"
    echo "STATUS=BLOCKED_D1_GATE" > "$STATUS"; exit 8
fi
say "D1 gate PASS -> starting D2"

bash "$SCRIPT_DIR/e1e_disk_guard.sh" >>"$LOG" 2>&1 || { say "BLOCKED: disk guard failed"; echo "STATUS=BLOCKED_DISK" > "$STATUS"; exit 5; }
say "disk guard OK ($(cat "$E1E_ROOT/logs/disk_guard_status.txt"))"

source "$REPO_ROOT/environment.sh" >/dev/null
export WIKI_RAG_SERVER_URL="http://127.0.0.1:8008/retrieve"
bash "$SCRIPT_DIR/run_e1e_train_70_120.sh" >>"$LOG" 2>&1
RC=$?
if [[ $RC -ne 0 ]]; then say "D2 training exit=$RC"; fi

E1E_PHASE=D2 python3 "$SCRIPT_DIR/e1e_integrity.py" >>"$LOG" 2>&1 || { say "BLOCKED: D2 integrity failed (training rc=$RC)"; echo "STATUS=BLOCKED_D2_INTEGRITY train_rc=$RC" > "$STATUS"; exit 6; }
say "D2 integrity PASS"

bash "$SCRIPT_DIR/run_e1e_eval_120.sh" >>"$LOG" 2>&1 || { say "BLOCKED: step120 eval failed"; echo "STATUS=BLOCKED_EVAL120" > "$STATUS"; exit 7; }
say "step120 eval done"

bash "$SCRIPT_DIR/run_e1e_analysis.sh" >>"$LOG" 2>&1 || say "analysis reported warnings (continuing)"

echo "STATUS=DONE" > "$STATUS"
say "PIPELINE_DONE"
exit 0

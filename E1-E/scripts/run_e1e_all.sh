#!/usr/bin/env bash
# E1-E end-to-end driver (E1-E_task.md section 25): preflight -> selftest -> D1 train ->
# D1 integrity -> step70 eval -> automatic gate -> disk guard -> D2 train -> D2 integrity ->
# step120 eval -> analysis -> post-run provenance -> figures.
#
# Every stage is checked explicitly: the first failure stops the chain, writes
# logs/pipeline_status.txt = STATUS=BLOCKED_<stage> and exits non-zero.  (The earlier version
# relied on `set -e` plus `&&` chaining and a trailing echo, which swallowed a stage failure and
# reported rc=0 -- that is how the D1 integrity failure of 15:07 was missed.)
#
# Usage:
#   bash run_e1e_all.sh all                  # everything
#   bash run_e1e_all.sh allfrom eval70       # resume the chain at a given stage
#   bash run_e1e_all.sh <stage>              # one stage
# Stages in order: preflight d1 eval70 gate diskguard d2 eval120 analysis postmanifest figures
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E1E_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$E1E_ROOT/.." && pwd)"
LOG="$E1E_ROOT/logs/run_all.log"
STATUS="$E1E_ROOT/logs/pipeline_status.txt"
mkdir -p "$E1E_ROOT/logs"
exec > >(tee -a "$LOG") 2>&1

STAGES="preflight d1 eval70 gate diskguard d2 eval120 analysis postmanifest figures"
STAGE="${1:-all}"
START="${2:-}"

stage_preflight() {
    bash "$SCRIPT_DIR/e1e_preflight.sh" >/dev/null || return 1
    grep -q "STATUS=OK" "$E1E_ROOT/logs/preflight_status.txt" || return 1
    # E1-E_task.md section 11 gate: planner + runtime replay vs E1-E0 (q=0/q=2), the 12 mandated
    # invariants and the PART D empty-keep neutrality regression. Must pass before any GPU work.
    source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate parallel-agent
    (
        cd "$REPO_ROOT"
        export PYTHONPATH="$REPO_ROOT/verl:$E1E_ROOT/code/pythonpath_e1e${PYTHONPATH:+:$PYTHONPATH}"
        export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
        export E1E_LAMBDA="${E1E_LAMBDA:-0.05}"
        rm -rf "$E1E_ROOT/analysis/selftest_diag"
        python3 "$E1E_ROOT/code/e1e_runtime_selftest.py" --run-dir "$E1E_ROOT/analysis/selftest_diag"
    ) || return 1
    python3 - "$E1E_ROOT/analysis/preflight_selftest.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
ok = bool(r.get("all_pass")) and all(r["pass"].values())
print("preflight selftest all_pass:", ok, r.get("pass"))
sys.exit(0 if ok else 1)
PY
}
stage_d1() {
    source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate parallel-agent
    cd "$REPO_ROOT"; source environment.sh >/dev/null
    export WIKI_RAG_SERVER_URL="http://127.0.0.1:8008/retrieve"
    bash "$SCRIPT_DIR/run_e1e_train_60_70.sh" || return 1
    E1E_PHASE=D1 python3 "$SCRIPT_DIR/e1e_integrity.py" || return 1
}
stage_eval70() { bash "$SCRIPT_DIR/run_e1e_eval_70.sh" || return 1; }
stage_gate() {
    # E1-E_task.md section 15: automatic D1 -> D2 decision (no user prompt).
    python3 "$SCRIPT_DIR/e1e_d1_gate.py" || return 1
    grep -q "GATE=PASS" "$E1E_ROOT/logs/d1_gate_status.txt" || return 1
}
stage_diskguard() {
    # E1-E_task.md section 25 item 13: recheck disk and only clean up if actually needed.
    bash "$SCRIPT_DIR/e1e_disk_guard.sh" || return 1
    grep -q "STATUS=OK" "$E1E_ROOT/logs/disk_guard_status.txt" || return 1
}
stage_d2() {
    source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate parallel-agent
    cd "$REPO_ROOT"; source environment.sh >/dev/null
    export WIKI_RAG_SERVER_URL="http://127.0.0.1:8008/retrieve"
    bash "$SCRIPT_DIR/run_e1e_train_70_120.sh" || return 1
    E1E_PHASE=D2 python3 "$SCRIPT_DIR/e1e_integrity.py" || return 1
}
stage_eval120() { bash "$SCRIPT_DIR/run_e1e_eval_120.sh" || return 1; }
stage_analysis() { bash "$SCRIPT_DIR/run_e1e_analysis.sh" || return 1; }
stage_postmanifest() { python3 "$SCRIPT_DIR/e1e_verify_original_unchanged.py" || return 1; }
stage_figures() {
    [[ -f "$SCRIPT_DIR/e1e_make_figures.py" ]] || return 0
    source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate parallel-agent
    python3 "$SCRIPT_DIR/e1e_make_figures.py" || return 1
    python3 "$SCRIPT_DIR/e1e_prediction_vs_observation.py" --phase both || return 1
}

run_one() {
    local st="$1"
    echo ""
    echo "=== E1-E stage $st  $(date -Is) ==="
    "stage_$st"
    local rc=$?
    if (( rc != 0 )); then
        echo "=== E1-E stage $st FAILED (rc=$rc) $(date -Is) ==="
        echo "STATUS=BLOCKED_$st" > "$STATUS"
        return "$rc"
    fi
    echo "=== E1-E stage $st OK  $(date -Is) ==="
    return 0
}

rc=0
case "$STAGE" in
  all)
    for st in $STAGES; do
        run_one "$st" || { rc=$?; break; }
    done
    ;;
  allfrom)
    [[ -n "$START" ]] || { echo "usage: run_e1e_all.sh allfrom <stage>"; exit 2; }
    seen=0
    for st in $STAGES; do
        [[ "$st" == "$START" ]] && seen=1
        (( seen )) || continue
        run_one "$st" || { rc=$?; break; }
    done
    (( seen )) || { echo "unknown stage '$START'"; exit 2; }
    ;;
  preflight|d1|eval70|gate|diskguard|d2|eval120|analysis|postmanifest|figures)
    run_one "$STAGE" || rc=$?
    ;;
  *)
    echo "unknown stage '$STAGE' (stages: $STAGES)"; exit 2 ;;
esac

if (( rc == 0 )); then
    echo "STATUS=DONE" > "$STATUS"
    echo "=== E1-E ALL STAGES DONE $(date -Is) ==="
else
    echo "=== E1-E CHAIN STOPPED rc=$rc $(date -Is) (see $STATUS) ==="
fi
exit "$rc"

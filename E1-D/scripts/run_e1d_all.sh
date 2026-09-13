#!/usr/bin/env bash
# E1-D end-to-end driver: preflight -> selftest -> D1 train -> integrity -> eval70 -> D2
# train -> integrity -> eval120 -> analysis.  Every stage checks the previous return code;
# a failure stops the chain (no silent continuation).
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E1D_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$E1D_ROOT/.." && pwd)"
LOG="$E1D_ROOT/logs/run_all.log"
mkdir -p "$E1D_ROOT/logs"
exec > >(tee -a "$LOG") 2>&1
STAGE="${1:-all}"
echo "=== E1-D run_all stage=$STAGE $(date -Is) ==="

stage_preflight() {
    bash "$SCRIPT_DIR/e1d_preflight.sh" >/dev/null || return 1
    grep -q "STATUS=OK" "$E1D_ROOT/logs/preflight_status.txt" || return 1
    python3 "$E1D_ROOT/code/e1d_filter_selftest.py" "$SCRIPT_DIR/run_e1d_train.sh" || return 1
}
stage_d1() {
    source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate parallel-agent
    cd "$REPO_ROOT"; source environment.sh >/dev/null
    export WIKI_RAG_SERVER_URL="http://127.0.0.1:8008/retrieve"
    bash "$SCRIPT_DIR/run_e1d_train_60_70.sh" || return 1
    E1D_PHASE=D1 python3 "$SCRIPT_DIR/e1d_integrity.py" || return 1
}
stage_eval70() { bash "$SCRIPT_DIR/run_e1d_eval_70.sh" || return 1; }
stage_d2() {
    source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate parallel-agent
    cd "$REPO_ROOT"; source environment.sh >/dev/null
    export WIKI_RAG_SERVER_URL="http://127.0.0.1:8008/retrieve"
    bash "$SCRIPT_DIR/run_e1d_train_70_120.sh" || return 1
    E1D_PHASE=D2 python3 "$SCRIPT_DIR/e1d_integrity.py" || return 1
}
stage_eval120() { bash "$SCRIPT_DIR/run_e1d_eval_120.sh" || return 1; }
stage_analysis() { bash "$SCRIPT_DIR/run_e1d_analysis.sh" || return 1; }

case "$STAGE" in
  preflight) stage_preflight ;;
  d1)        stage_d1 ;;
  eval70)    stage_eval70 ;;
  d2)        stage_d2 ;;
  eval120)   stage_eval120 ;;
  analysis)  stage_analysis ;;
  all)
    stage_preflight && stage_d1 && stage_eval70 && stage_d2 && stage_eval120 && stage_analysis
    echo "E1-D ALL STAGES DONE $(date -Is)"
    ;;
  *) echo "unknown stage $STAGE"; exit 2 ;;
esac
rc=$?
echo "=== E1-D run_all stage=$STAGE finished rc=$rc $(date -Is) ==="
exit $rc

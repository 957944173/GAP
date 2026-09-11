#!/usr/bin/env bash
# E1-A evaluation entry: 7-benchmark evaluation of the E1-A step70 checkpoint.
#
# Generated from Agent/evaluation/mhqa_agent/eval_mhqa_agent_rl_step_70_4gpu.sh
# (same 7 benchmarks, same protocol); the model path is the E1-A checkpoint instead of
# the template's non-existent DAPO-GAP3B-MHQA-Agent-4gpu/global_step_70 (see issues.log I1).
#
# Outputs: E1-A/eval_results/<run_timestamp>/
#   accuracy.json benchmark_results.json trajectory_metrics.json
#   search_round_statistics.json parallel_statistics.json eval.log (+ eval_stderr.log)
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EVAL_TS="${E1A_EVAL_TS:-$(date +%Y%m%d_%H%M%S)}"
OUT="$SCRIPT_DIR/eval_results/${EVAL_TS}"
mkdir -p "$OUT"
echo "$OUT" > "$SCRIPT_DIR/eval_results/latest_eval.txt"
exec 1> >(tee -a "$OUT/eval.log") 2> >(tee -a "$OUT/eval_stderr.log" >&2)

# Required environment order (E1-A_task.md "Wiki环境要求"): wiki health was verified in
# E1-A/preflight/wiki_service_status.log (REUSED, no duplicate server started), then
# parallel-agent + environment.sh, then evaluation.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate parallel-agent
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source environment.sh >/dev/null
export WIKI_RAG_SERVER_URL="http://127.0.0.1:8008/retrieve"
if ! LD_LIBRARY_PATH="" curl --silent --show-error --fail --max-time 10 \
        -H 'Content-Type: application/json' -X POST "$WIKI_RAG_SERVER_URL" \
        -d '{"queries": []}' >/dev/null; then
    echo "ERROR: wiki service not reachable at $WIKI_RAG_SERVER_URL" >&2
    exit 1
fi

# shellcheck disable=SC1091
source "$SCRIPT_DIR/scripts/e1a_eval_common.sh"
e1a_run_eval \
    "experiments/DAPO-GAP3B-MHQA-Agent-E1A-step60to70-4gpu/global_step_70/actor" \
    "DAPO-GAP3B-MHQA-Agent-E1A-eval-step70-4gpu" \
    "$OUT" \
    "e1a_step70" \
    "$REPO_ROOT"

echo "[E1-A eval] results -> $OUT"

#!/usr/bin/env bash
# E1-A matched-baseline evaluation: "GAP step60 -> step70 with the ORIGINAL reward".
#
# That experiment is E1-0 (identical start checkpoint, identical 10 steps, identical
# hyper-parameters, original EM reward).  Evaluating its step70 checkpoint with the same
# 7-benchmark protocol gives the apples-to-apples baseline required by
# E1-A_task.md section 2 ("Baseline: GAP step60->70 original reward").
#
# Outputs: E1-A/eval_results/baseline_<run_timestamp>/
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EVAL_TS="${E1A_EVAL_TS:-$(date +%Y%m%d_%H%M%S)}"
OUT="$REPO_ROOT/E1-A/eval_results/baseline_${EVAL_TS}"
mkdir -p "$OUT"
echo "$OUT" > "$REPO_ROOT/E1-A/eval_results/latest_baseline_eval.txt"
exec 1> >(tee -a "$OUT/eval.log") 2> >(tee -a "$OUT/eval_stderr.log" >&2)

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
source "$REPO_ROOT/E1-A/scripts/e1a_eval_common.sh"
e1a_run_eval \
    "experiments/DAPO-GAP3B-MHQA-Agent-E10-step60to70-4gpu/global_step_70/actor" \
    "DAPO-GAP3B-MHQA-Agent-E10-eval-step70-4gpu" \
    "$OUT" \
    "baseline_e10_step70" \
    "$REPO_ROOT"

echo "[E1-A baseline eval] results -> $OUT"

#!/usr/bin/env bash
# E1-B evaluation entry: 7-benchmark evaluation of one E1-B checkpoint.
#
#   bash E1-B/scripts/run_e1b_eval_step.sh 120
#
# Outputs: E1-B/evaluations/step<STEP>_<timestamp>/
#   accuracy.json benchmark_results.json trajectory_metrics.json
#   search_round_statistics.json parallel_statistics.json eval_summary.json
#   eval.log eval_stderr.log
# plus the task-required flat files written by e1b_collect_evaluation_artifacts.py:
#   results.json benchmark_scores.csv behavior_metrics.csv
#
# Never overwrites an existing evaluation directory.
set -Eeuo pipefail

STEP="${1:?usage: run_e1b_eval_step.sh <step>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E1B_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$E1B_ROOT/.." && pwd)"
export REPO_ROOT

EXP_NAME="DAPO-GAP3B-MHQA-Agent-E1B-step70to120-4gpu"
CKPT_REL="experiments/${EXP_NAME}/global_step_${STEP}/actor"
if [[ ! -s "$REPO_ROOT/$CKPT_REL/model_world_size_4_rank_0.pt" ]]; then
    echo "ERROR: checkpoint actor not found: $REPO_ROOT/$CKPT_REL" >&2
    exit 3
fi

EVAL_TS="${E1B_EVAL_TS:-$(date +%Y%m%d_%H%M%S)}"
OUT="$E1B_ROOT/evaluations/step${STEP}_${EVAL_TS}"
if [[ -e "$OUT" ]]; then
    echo "ERROR: refusing to overwrite existing $OUT" >&2
    exit 9
fi
mkdir -p "$OUT"
echo "$OUT" > "$E1B_ROOT/evaluations/latest_step${STEP}_eval.txt"
exec 1> >(tee -a "$OUT/eval.log") 2> >(tee -a "$OUT/eval_stderr.log" >&2)

echo "[E1-B eval] step=$STEP out=$OUT $(date -Is)"

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
    exit 4
fi

# shellcheck disable=SC1091
source "$SCRIPT_DIR/e1b_eval_common.sh"
e1b_run_eval "$CKPT_REL" "DAPO-GAP3B-MHQA-Agent-E1B-eval-step${STEP}-4gpu" "$OUT" "e1b_step${STEP}" "$STEP"

# task-required flat artifacts (results.json / benchmark_scores.csv / behavior_metrics.csv)
python3 "$SCRIPT_DIR/e1b_collect_evaluation_artifacts.py" \
    --eval-root "$E1B_ROOT/evaluations" \
    --out "$E1B_ROOT/evaluations"

echo "[E1-B eval] step=$STEP results -> $OUT"

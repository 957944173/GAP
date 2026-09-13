#!/usr/bin/env bash
# E1-D evaluation entry: 7-benchmark evaluation of one E1-D checkpoint.
#   bash E1-D/scripts/run_e1d_eval_step.sh <step>
# Output: E1-D/eval_results/step<STEP>_<timestamp>/
set -Eeuo pipefail
STEP="${1:?usage: run_e1d_eval_step.sh <step>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E1D_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$E1D_ROOT/.." && pwd)"
export REPO_ROOT
EXP_NAME="DAPO-GAP3B-MHQA-Agent-E1D-step60to120-4gpu"
CKPT_REL="experiments/${EXP_NAME}/global_step_${STEP}/actor"
if [[ ! -s "$REPO_ROOT/$CKPT_REL/model_world_size_4_rank_0.pt" ]]; then
    echo "ERROR: checkpoint actor not found: $REPO_ROOT/$CKPT_REL" >&2; exit 3
fi
EVAL_TS="${E1D_EVAL_TS:-$(date +%Y%m%d_%H%M%S)}"
OUT="$E1D_ROOT/eval_results/step${STEP}_${EVAL_TS}"
if [[ -e "$OUT" ]]; then echo "ERROR: refusing to overwrite $OUT" >&2; exit 9; fi
mkdir -p "$OUT"
echo "$OUT" > "$E1D_ROOT/eval_results/latest_step${STEP}.txt"
echo "$OUT" > "$E1D_ROOT/eval_results/latest_step${STEP}_eval.txt"
exec 1> >(tee -a "$OUT/eval.log") 2> >(tee -a "$OUT/eval_stderr.log" >&2)
echo "[E1-D eval] step=$STEP out=$OUT $(date -Is)"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate parallel-agent
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source environment.sh >/dev/null
export WIKI_RAG_SERVER_URL="http://127.0.0.1:8008/retrieve"
if ! LD_LIBRARY_PATH="" curl --silent --show-error --fail --max-time 10 -H 'Content-Type: application/json' \
        -X POST "$WIKI_RAG_SERVER_URL" -d '{"queries": []}' >/dev/null; then
    echo "ERROR: wiki service not reachable" >&2; exit 4
fi
# shellcheck disable=SC1091
source "$SCRIPT_DIR/e1d_eval_common.sh"
e1d_run_eval "$CKPT_REL" "DAPO-GAP3B-MHQA-Agent-E1D-eval-step${STEP}-4gpu" "$OUT" "e1d_step${STEP}" "$STEP"
echo "[E1-D eval] step=$STEP results -> $OUT"

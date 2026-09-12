#!/usr/bin/env bash
# Re-aggregate the EXISTING reference evaluation dumps with the E1-B analyzer so that
# every comparison in E1-B/FINAL_REPORT.md is computed by one identical protocol.
#
# No model is re-run here: the dumps were produced by the original GAP evaluation
# scripts (and, for E1-A/E1-0, by the E1-A evaluation scripts, which reuse the original
# implementation).  This script only re-derives the metrics into E1-B/evaluations/.
#
# References:
#   gap_step60_reference   original GAP RL step60   (the E1-A/E1-B starting point)
#   gap_step120_reference  original GAP RL step120  (the same step count as E1-B end)
#   gap_step180_reference  original GAP RL step180  (long-run reference)
#   e1a_step70_reference   E1-A step70              (the E1-B starting checkpoint)
#   e10_step70_reference   E1-0 step70              (matched baseline: original reward,
#                                                    step60->70, same 10 steps as E1-A)
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E1B_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$E1B_ROOT/.." && pwd)"
OUT_ROOT="$E1B_ROOT/evaluations"
mkdir -p "$OUT_ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate parallel-agent

aggregate() {
    local label="$1" log="$2" dump="$3" out="$4"
    if [[ ! -s "$dump" ]]; then
        echo "SKIP $label (missing dump $dump)"
        return 0
    fi
    mkdir -p "$out"
    echo "=== $label -> $out"
    python3 "$SCRIPT_DIR/e1b_analyze_eval.py" \
        --label "$label" --log "$log" --dump "$dump" --out "$out" || echo "WARN: $label aggregation failed"
}

aggregate "gap_step60_reference" \
    "$REPO_ROOT/verl/logs/DAPO-GAP3B-MHQA-Agent-eval-rl-step-60-4gpu.log" \
    "$REPO_ROOT/experiments/DAPO-GAP3B-MHQA-Agent-eval-rl-step-60-4gpu/val_generations/0.jsonl" \
    "$OUT_ROOT/gap_step60_reference"

aggregate "gap_step120_reference" \
    "$REPO_ROOT/verl/logs/DAPO-GAP3B-MHQA-Agent-eval-rl-step-120-4gpu.log" \
    "$REPO_ROOT/experiments/DAPO-GAP3B-MHQA-Agent-eval-rl-step-120-4gpu/val_generations/0.jsonl" \
    "$OUT_ROOT/gap_step120_reference"

aggregate "gap_step180_reference" \
    "$REPO_ROOT/verl/logs/DAPO-GAP3B-MHQA-Agent-eval-rl-step-180-4gpu.log" \
    "$REPO_ROOT/experiments/DAPO-GAP3B-MHQA-Agent-eval-rl-step-180-4gpu/val_generations/0.jsonl" \
    "$OUT_ROOT/gap_step180_reference"

aggregate "e1a_step70_reference" \
    "$REPO_ROOT/verl/logs/DAPO-GAP3B-MHQA-Agent-E1A-eval-step70-4gpu.log" \
    "$REPO_ROOT/experiments/DAPO-GAP3B-MHQA-Agent-E1A-eval-step70-4gpu/val_generations/0.jsonl" \
    "$OUT_ROOT/e1a_step70_reference"

aggregate "e10_step70_reference" \
    "$REPO_ROOT/verl/logs/DAPO-GAP3B-MHQA-Agent-E10-eval-step70-4gpu.log" \
    "$REPO_ROOT/experiments/DAPO-GAP3B-MHQA-Agent-E10-eval-step70-4gpu/val_generations/0.jsonl" \
    "$OUT_ROOT/e10_step70_reference"

echo "[E1-B references] done -> $OUT_ROOT"

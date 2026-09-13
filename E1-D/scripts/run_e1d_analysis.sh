#!/usr/bin/env bash
# E1-D analysis entry: builds the causal matrix, paired per-prompt comparisons and the
# training-side mechanism statistics from artifacts that already exist.  Never overwrites:
# every output goes to a fresh E1-D/analysis/<name>_<timestamp>/ directory (or is merged
# into E1-D/analysis/ where the names are unique by construction).
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E1D_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$E1D_ROOT/.." && pwd)"
TS="${E1D_ANALYSIS_TS:-$(date +%Y%m%d_%H%M%S)}"
A="$E1D_ROOT/analysis"
LOGF="$E1D_ROOT/logs/analysis_${TS}.log"
mkdir -p "$A" "$A/comparison_${TS}" "$A/all_evals"
exec > >(tee -a "$LOGF") 2>&1
echo "=== E1-D analysis $TS ==="
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate parallel-agent
cd "$REPO_ROOT"

# ---------- 1. training-side diagnostics of both E1-D phases ----------
for ph in D1 D2; do
    [[ -f "$E1D_ROOT/runs/latest_${ph}.txt" ]] || { echo "no $ph run"; continue; }
    RUN=$(cat "$E1D_ROOT/runs/latest_${ph}.txt")
    echo "--- $ph run: $RUN"
    python3 "$SCRIPT_DIR/e1d_checkpoint_stats.py" --run-dir "$RUN" --out "$E1D_ROOT/diagnostics/checkpoint_stats_${ph}" || echo "WARN checkpoint_stats $ph"
    python3 "$SCRIPT_DIR/e1d_analyze_training.py" --run-dir "$RUN" --out "$E1D_ROOT/analysis/${ph}_vs_E1_0" || echo "WARN analyze_training $ph vs E1-0"
    python3 "$SCRIPT_DIR/e1d_analyze_training.py" --run-dir "$RUN" --out "$E1D_ROOT/analysis/${ph}_vs_E1_B" \
        --baseline-audit "$REPO_ROOT/E1-B/run_20260911_113432/diagnostics" || echo "WARN analyze_training $ph vs E1-B"
    E1D_PHASE=$ph E1D_RUN_DIR="$RUN" python3 "$SCRIPT_DIR/e1d_integrity.py" || echo "WARN integrity $ph"
done

# ---------- 2. merge every evaluation into one root (symlinks; originals untouched) ----
AE="$A/all_evals"
link() { # link <eval_dir>  (E1-D's own evals keep their step<N>_<ts> name; refs keep their dir name)
    local d="$1"; [[ -f "$d/eval_summary.json" ]] || return 0
    local b; b="$(basename "$d")"
    local name
    if [[ "$b" == step*_* ]]; then name="$b"; else name="${b}_$(basename "$(dirname "$d")")"; fi
    ln -sfn "$d" "$AE/$name"; echo "linked $name -> $d"
}
for d in "$E1D_ROOT"/eval_results/step*_*; do [[ -d "$d" ]] && link "$d"; done
for d in "$REPO_ROOT"/E1-B/evaluations/gap_step60_reference \
         "$REPO_ROOT"/E1-B/evaluations/gap_step120_reference \
         "$REPO_ROOT"/E1-B/evaluations/gap_step180_reference \
         "$REPO_ROOT"/E1-B/evaluations/e10_step70_reference \
         "$REPO_ROOT"/E1-B/evaluations/e1a_step70_reference \
         "$REPO_ROOT"/E1-B/evaluations/step120_20260911_235134; do
    [[ -d "$d" ]] && link "$d"
done
python3 "$SCRIPT_DIR/e1d_collect_evaluation_artifacts.py" --eval-root "$AE" --out "$A" || echo "WARN collector"
python3 "$SCRIPT_DIR/e1d_compare_all.py" --eval-root "$AE" --out "$A/comparison_${TS}" || echo "WARN compare_all"

# ---------- 3. paired per-prompt comparisons ----------
paired() { # paired <baseline_dump> <e1d_dump> <label>
    local b="$1" e="$2" l="$3"
    [[ -s "$b" && -s "$e" ]] || { echo "SKIP paired $l (missing dump)"; return 0; }
    python3 "$SCRIPT_DIR/e1d_paired_eval_compare.py" --baseline-dump "$b" --e1d-dump "$e" \
        --out "$A/comparison_${TS}" --label "$l" || echo "WARN paired $l"
}
GAP120_DUMP="$REPO_ROOT/experiments/DAPO-GAP3B-MHQA-Agent-eval-rl-step-120-4gpu/val_generations/0.jsonl"
GAP60_DUMP="$REPO_ROOT/experiments/DAPO-GAP3B-MHQA-Agent-eval-rl-step-60-4gpu/val_generations/0.jsonl"
E1A70_DUMP="$REPO_ROOT/experiments/DAPO-GAP3B-MHQA-Agent-E1A-eval-step70-4gpu/val_generations/0.jsonl"
E1B120_DUMP="$REPO_ROOT/experiments/DAPO-GAP3B-MHQA-Agent-E1B-eval-step120-4gpu/val_generations/0.jsonl"
E1D70_DUMP="$REPO_ROOT/experiments/DAPO-GAP3B-MHQA-Agent-E1D-eval-step70-4gpu/val_generations/0.jsonl"
E1D120_DUMP="$REPO_ROOT/experiments/DAPO-GAP3B-MHQA-Agent-E1D-eval-step120-4gpu/val_generations/0.jsonl"
paired "$GAP60_DUMP"  "$E1D70_DUMP"  "e1d70_vs_gap60"
paired "$E1A70_DUMP"  "$E1D70_DUMP"  "e1d70_vs_e1a70"
paired "$GAP120_DUMP" "$E1D120_DUMP" "e1d120_vs_gap120"
paired "$E1B120_DUMP" "$E1D120_DUMP" "e1d120_vs_e1b120"

# ---------- 4. summary.json ----------
python3 "$SCRIPT_DIR/e1d_build_summary.py" --analysis "$A" --comparison "$A/comparison_${TS}" \
    --out "$A/summary.json" || echo "WARN build_summary"
echo "=== E1-D analysis done -> $A ==="

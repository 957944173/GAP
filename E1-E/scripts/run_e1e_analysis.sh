#!/usr/bin/env bash
# E1-E analysis entry: builds the causal matrix, paired per-prompt comparisons and the
# training-side mechanism statistics from artifacts that already exist.  Never overwrites:
# every output goes to a fresh E1-E/analysis/<name>_<timestamp>/ directory (or is merged
# into E1-E/analysis/ where the names are unique by construction).
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E1E_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$E1E_ROOT/.." && pwd)"
TS="${E1E_ANALYSIS_TS:-$(date +%Y%m%d_%H%M%S)}"
A="$E1E_ROOT/analysis"
LOGF="$E1E_ROOT/logs/analysis_${TS}.log"
mkdir -p "$A" "$A/comparison_${TS}" "$A/all_evals"
exec > >(tee -a "$LOGF") 2>&1
echo "=== E1-E analysis $TS ==="
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate parallel-agent
cd "$REPO_ROOT"

# ---------- 0. resolved configs + checkpoint manifests (artifact completeness) ----------
mkdir -p "$E1E_ROOT/configs"
for ph in D1 D2; do
    [[ -f "$E1E_ROOT/runs/latest_${ph}.txt" ]] || continue
    LOGF=$(ls -1 "$E1E_ROOT"/logs/train_${ph}_*.log 2>/dev/null | tail -1)
    [[ -n "$LOGF" ]] && python3 "$SCRIPT_DIR/e1e_extract_config.py" --log "$LOGF" \
        --out "$E1E_ROOT/configs/resolved_${ph}.json" || echo "WARN extract_config $ph"
done
python3 "$SCRIPT_DIR/e1e_checkpoint_manifests.py" --out "$E1E_ROOT/manifests" || echo "WARN checkpoint_manifests"

# ---------- 1. training-side diagnostics of both E1-E phases ----------
for ph in D1 D2; do
    [[ -f "$E1E_ROOT/runs/latest_${ph}.txt" ]] || { echo "no $ph run"; continue; }
    RUN=$(cat "$E1E_ROOT/runs/latest_${ph}.txt")
    echo "--- $ph run: $RUN"
    python3 "$SCRIPT_DIR/e1e_checkpoint_stats.py" --run-dir "$RUN" --out "$E1E_ROOT/diagnostics/checkpoint_stats_${ph}" || echo "WARN checkpoint_stats $ph"
    python3 "$SCRIPT_DIR/e1e_analyze_training.py" --run-dir "$RUN" --out "$E1E_ROOT/analysis/${ph}_vs_E1_0" || echo "WARN analyze_training $ph vs E1-0"
    python3 "$SCRIPT_DIR/e1e_analyze_training.py" --run-dir "$RUN" --out "$E1E_ROOT/analysis/${ph}_vs_E1_B" \
        --baseline-audit "$REPO_ROOT/E1-B/run_20260911_113432/diagnostics" || echo "WARN analyze_training $ph vs E1-B"
    E1E_PHASE=$ph E1E_RUN_DIR="$RUN" python3 "$SCRIPT_DIR/e1e_integrity.py" || echo "WARN integrity $ph"
done

# ---------- 2. merge every evaluation into one root (symlinks; originals untouched) ----
AE="$A/all_evals"
link_as() { # link_as <name> <eval_dir>  (explicit name -> no ambiguity with E1-E's own steps)
    local name="$1" d="$2"
    [[ -n "$d" && -f "$d/eval_summary.json" ]] || return 0
    ln -sfn "$d" "$AE/$name"; echo "linked $name -> $d"
}
newest() { local h; h=$(ls -d $1 2>/dev/null | sort | tail -1); echo "$h"; }
# E1-E's own evaluations keep their step<N>_<timestamp> name
for d in "$E1E_ROOT"/eval_results/step*_*; do
    [[ -f "$d/eval_summary.json" ]] && link_as "$(basename "$d")" "$d"
done
# read-only reference evaluations (never written to).  Explicit names so that the
# E1-E branch's own `step<N>_<ts>` dirs can never be confused with a reference:
#   GAP60/120/180, E1-0 step70, E1-A step70 (from E1-B/evaluations)
#   E1-B step120, E1-D step70, E1-D step120 (the four-way comparison baselines)
link_as gap_step60_reference  "$REPO_ROOT/E1-B/evaluations/gap_step60_reference"
link_as gap_step120_reference "$REPO_ROOT/E1-B/evaluations/gap_step120_reference"
link_as gap_step180_reference "$REPO_ROOT/E1-B/evaluations/gap_step180_reference"
link_as e10_step70_reference  "$REPO_ROOT/E1-B/evaluations/e10_step70_reference"
link_as e1a_step70_reference  "$REPO_ROOT/E1-B/evaluations/e1a_step70_reference"
link_as e1b_step120_reference "$REPO_ROOT/E1-B/evaluations/step120_20260911_235134"
link_as e1d_step70_reference  "$(newest "$REPO_ROOT/E1-D/eval_results/step70_*")"
link_as e1d_step120_reference "$(newest "$REPO_ROOT/E1-D/eval_results/step120_*")"
python3 "$SCRIPT_DIR/e1e_collect_evaluation_artifacts.py" --eval-root "$AE" --out "$A" || echo "WARN collector"
python3 "$SCRIPT_DIR/e1e_compare_all.py" --eval-root "$AE" --out "$A/comparison_${TS}" || echo "WARN compare_all"

# ---------- 3. paired per-prompt comparisons ----------
paired() { # paired <baseline_dump> <e1e_dump> <label>
    local b="$1" e="$2" l="$3"
    [[ -s "$b" && -s "$e" ]] || { echo "SKIP paired $l (missing dump)"; return 0; }
    python3 "$SCRIPT_DIR/e1e_paired_eval_compare.py" --baseline-dump "$b" --e1e-dump "$e" \
        --out "$A/comparison_${TS}" --label "$l" || echo "WARN paired $l"
}
GAP120_DUMP="$REPO_ROOT/experiments/DAPO-GAP3B-MHQA-Agent-eval-rl-step-120-4gpu/val_generations/0.jsonl"
GAP60_DUMP="$REPO_ROOT/experiments/DAPO-GAP3B-MHQA-Agent-eval-rl-step-60-4gpu/val_generations/0.jsonl"
E1A70_DUMP="$REPO_ROOT/experiments/DAPO-GAP3B-MHQA-Agent-E1A-eval-step70-4gpu/val_generations/0.jsonl"
E1B120_DUMP="$REPO_ROOT/experiments/DAPO-GAP3B-MHQA-Agent-E1B-eval-step120-4gpu/val_generations/0.jsonl"
E1D120_DUMP="$REPO_ROOT/experiments/DAPO-GAP3B-MHQA-Agent-E1D-eval-step120-4gpu/val_generations/0.jsonl"
E1E70_DUMP="$REPO_ROOT/experiments/DAPO-GAP3B-MHQA-Agent-E1E-q2-eval-step70-4gpu/val_generations/0.jsonl"
E1E120_DUMP="$REPO_ROOT/experiments/DAPO-GAP3B-MHQA-Agent-E1E-q2-eval-step120-4gpu/val_generations/0.jsonl"
paired "$GAP60_DUMP"  "$E1E70_DUMP"  "e1e70_vs_gap60"
paired "$E1A70_DUMP"  "$E1E70_DUMP"  "e1e70_vs_e1a70"
paired "$GAP120_DUMP" "$E1E120_DUMP" "e1e120_vs_gap120"
paired "$E1B120_DUMP" "$E1E120_DUMP" "e1e120_vs_e1b120"
paired "$E1D120_DUMP" "$E1E120_DUMP" "e1e120_vs_e1d120"
paired "$E1E70_DUMP"  "$E1E120_DUMP" "e1e120_vs_e1e70"

# ---------- 3b. E1-E0 prediction vs online observation ----------
python3 "$SCRIPT_DIR/e1e_prediction_vs_observation.py" --phase both || echo "WARN prediction_vs_observation"

# ---------- 4. summary.json ----------
python3 "$SCRIPT_DIR/e1e_build_summary.py" --analysis "$A" --comparison "$A/comparison_${TS}" \
    --out "$A/summary.json" || echo "WARN build_summary"
echo "=== E1-E analysis done -> $A ==="

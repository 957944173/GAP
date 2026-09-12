#!/usr/bin/env bash
# E1-B post-training pipeline: runs every remaining step after the training job ends,
# unattended, and writes a single DONE/BLOCKED marker.
#
#   1. wait for the training process to disappear
#   2. verify checkpoints 80/90/100/110/120 (shards + data.pt + tracker)
#   3. extract the resolved config from the training stdout
#   4. per-checkpoint statistics        -> E1-B/checkpoints/
#   5. training-side analysis           -> <run>/analysis (vs E1-0 original-reward baseline)
#                                          <run>/analysis_vs_e1a (vs the E1-A branch)
#   6. post-training integrity check    -> <run>/analysis (steps 71..120, state resume)
#   7. 7-benchmark evaluation of step120 (then 80/90/100/110) with the original protocol
#   8. artifact collection + multi-way comparison + paired significance tests
#   9. original-file integrity re-verification
#
# Exit codes: 0 = everything done, 3 = training produced incomplete checkpoints
#             (evaluations are then skipped), 4 = no disk space for the merged models.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E1B_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$E1B_ROOT/.." && pwd)"
RUN_DIR="$(cat "$E1B_ROOT/latest_run.txt")"
EXP_NAME="DAPO-GAP3B-MHQA-Agent-E1B-step70to120-4gpu"
EXP_DIR="$REPO_ROOT/experiments/$EXP_NAME"
LOG="$E1B_ROOT/logs/post_train_pipeline.log"
STATUS="$E1B_ROOT/logs/pipeline_status.txt"
mkdir -p "$E1B_ROOT/logs" "$E1B_ROOT/checkpoints" "$E1B_ROOT/diagnostics"
: > "$LOG"

say() { echo "$(date -Is) $*" | tee -a "$LOG"; }

say "pipeline started (run dir $RUN_DIR)"

# ---------------------------------------------------------------- 1. wait
while pgrep -f "verl.trainer.main_ppo" >/dev/null 2>&1; do
    sleep 60
done
say "training process gone"

# ---------------------------------------------------------------- 2. checkpoints
MISSING=()
for s in 80 90 100 110 120; do
    d="$EXP_DIR/global_step_$s"
    ok=1
    for f in "$d/actor/model_world_size_4_rank_0.pt" "$d/actor/model_world_size_4_rank_1.pt" \
             "$d/actor/model_world_size_4_rank_2.pt" "$d/actor/model_world_size_4_rank_3.pt" \
             "$d/actor/optim_world_size_4_rank_0.pt" "$d/actor/extra_state_world_size_4_rank_0.pt" \
             "$d/data.pt"; do
        [[ -s "$f" ]] || ok=0
    done
    if [[ "$ok" == "1" ]]; then
        say "checkpoint $s OK ($(du -sh "$d" | cut -f1))"
    else
        say "checkpoint $s INCOMPLETE"
        MISSING+=("$s")
    fi
done
say "tracker=$(cat "$EXP_DIR/latest_checkpointed_iteration.txt" 2>/dev/null)"
say "steps seen: $(grep -oE 'step:[0-9]+ - ' "$RUN_DIR/stdout.log" | grep -oE '[0-9]+' | sort -n -u | tr '\n' ' ')"

# ---------------------------------------------------------------- 3. config
python3 "$SCRIPT_DIR/e1b_extract_config.py" --log "$RUN_DIR/stdout.log" \
    --out "$RUN_DIR/config/resolved_config.txt" --json "$RUN_DIR/config/resolved_config.json" >>"$LOG" 2>&1 || true

# ---------------------------------------------------------------- 4. checkpoint stats
python3 "$SCRIPT_DIR/e1b_checkpoint_stats.py" --run-dir "$RUN_DIR" --out "$E1B_ROOT/checkpoints" >>"$LOG" 2>&1 || say "checkpoint_stats FAILED"

# ---------------------------------------------------------------- 5. training analysis
python3 "$SCRIPT_DIR/e1b_analyze_training.py" --run-dir "$RUN_DIR" --out "$RUN_DIR/analysis" >>"$LOG" 2>&1 \
    || say "analyze_training (vs E1-0) FAILED"
E1A_DIAG="$REPO_ROOT/E1-A/run_20260911_024902/diagnostics"
if [[ -d "$E1A_DIAG" ]]; then
    python3 "$SCRIPT_DIR/e1b_analyze_training.py" --run-dir "$RUN_DIR" --out "$RUN_DIR/analysis_vs_e1a" \
        --baseline-audit "$E1A_DIAG" >>"$LOG" 2>&1 || say "analyze_training (vs E1-A) FAILED"
fi

# ---------------------------------------------------------------- 6. integrity
E1B_RUN_DIR="$RUN_DIR" E1B_EXPERIMENT_NAME="$EXP_NAME" LOG_FILE="$REPO_ROOT/verl/logs/$EXP_NAME.log" \
    python3 "$SCRIPT_DIR/e1b_check_training_integrity.py" >>"$LOG" 2>&1 || say "integrity check reported FAILED (see analysis)"

bash "$SCRIPT_DIR/e1b_verify_original_unchanged.sh" >>"$LOG" 2>&1 || say "original-file integrity FAILED"

if (( ${#MISSING[@]} > 0 )); then
    say "BLOCKED: incomplete checkpoints: ${MISSING[*]} -- skipping evaluations"
    echo "STATUS=BLOCKED_INCOMPLETE_CHECKPOINTS missing=${MISSING[*]}" > "$STATUS"
    exit 3
fi

FREE_GB=$(df -BG --output=avail /data01 | tail -1 | tr -dc '0-9')
say "disk free: ${FREE_GB} GB"
if (( FREE_GB < 90 )); then
    say "BLOCKED: only ${FREE_GB} GB free (< 90 GB needed for FSDP->HF merges)"
    echo "STATUS=BLOCKED_DISK free_gb=$FREE_GB" > "$STATUS"
    exit 4
fi

# ---------------------------------------------------------------- 7. evaluations
say "starting evaluations (120 first, then 80/90/100/110)"
bash "$SCRIPT_DIR/e1b_eval_all_steps.sh" 120 80 90 100 110 >>"$LOG" 2>&1 || say "eval orchestrator reported failures (continuing)"

# ---------------------------------------------------------------- 8. comparisons
python3 "$SCRIPT_DIR/e1b_collect_evaluation_artifacts.py" \
    --eval-root "$E1B_ROOT/evaluations" --out "$E1B_ROOT/evaluations" >>"$LOG" 2>&1 || say "collector FAILED"
python3 "$SCRIPT_DIR/e1b_compare_all.py" \
    --eval-root "$E1B_ROOT/evaluations" --out "$E1B_ROOT/evaluations/comparison" >>"$LOG" 2>&1 || say "compare_all FAILED"

GAP120_DUMP="$REPO_ROOT/experiments/DAPO-GAP3B-MHQA-Agent-eval-rl-step-120-4gpu/val_generations/0.jsonl"
E1A70_DUMP="$REPO_ROOT/experiments/DAPO-GAP3B-MHQA-Agent-E1A-eval-step70-4gpu/val_generations/0.jsonl"
E1B120_DUMP="$REPO_ROOT/experiments/DAPO-GAP3B-MHQA-Agent-E1B-eval-step120-4gpu/val_generations/0.jsonl"
if [[ -s "$E1B120_DUMP" ]]; then
    mkdir -p "$E1B_ROOT/evaluations/comparison"
    [[ -s "$GAP120_DUMP" ]] && python3 "$SCRIPT_DIR/e1b_paired_eval_compare.py" \
        --baseline-dump "$GAP120_DUMP" --e1b-dump "$E1B120_DUMP" \
        --out "$E1B_ROOT/evaluations/comparison" --label "e1b_step120_vs_gap_step120" >>"$LOG" 2>&1 \
        || say "paired vs GAP step120 FAILED"
    [[ -s "$E1A70_DUMP" ]] && python3 "$SCRIPT_DIR/e1b_paired_eval_compare.py" \
        --baseline-dump "$E1A70_DUMP" --e1b-dump "$E1B120_DUMP" \
        --out "$E1B_ROOT/evaluations/comparison" --label "e1b_step120_vs_e1a_step70" >>"$LOG" 2>&1 \
        || say "paired vs E1-A step70 FAILED"
fi

# ---------------------------------------------------------------- 9. final integrity
bash "$SCRIPT_DIR/e1b_verify_original_unchanged.sh" >>"$LOG" 2>&1 || say "final original-file integrity FAILED"

echo "STATUS=DONE run_dir=$RUN_DIR" > "$STATUS"
say "PIPELINE_DONE"
exit 0

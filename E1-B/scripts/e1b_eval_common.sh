#!/usr/bin/env bash
# =====================================================================================
# Shared E1-B evaluation helper.
#
# Reuses the ORIGINAL GAP evaluation implementation
# (Agent/evaluation/mhqa_agent/eval_mhqa_agent_4gpu_common.sh) **unmodified**, so the
# evaluation protocol (7 benchmarks, greedy, val_batch_size 512, max_prompt 4096,
# max_response 8192, max_model_len 12288, identical benchmark parquet files) is exactly
# the one used for the GAP baseline evaluations and for E1-A.
#
# The only things this helper changes are the checkpoint path, the experiment/output
# name and the label.  The shared script hardcodes
# `experiments/DAPO-GAP3B-MHQA-Agent-4gpu/global_step_${EVAL_RL_STEP}/actor` for
# EVAL_MODEL_KIND=rl, so a non-"rl" kind is used and the FSDP checkpoint is pre-merged
# here (same approach as E1-A).
#
# Usage (from an entrypoint):
#   e1b_run_eval <ckpt_actor_dir_rel_to_repo> <experiment_name> <out_dir> <label> <step>
# =====================================================================================
set -Eeuo pipefail

e1b_run_eval() {
    local ckpt_rel="$1" exp_name="$2" out_dir="$3" label="$4" step="$5"
    local repo_root="${REPO_ROOT:-/data01/wyy/Graph-Agent-Planning}"

    mkdir -p "$out_dir"
    local merged="$repo_root/$ckpt_rel/merged_hf"

    # ---- 1. merge FSDP shards -> HF (idempotent) ----
    if [[ ! -s "$merged/config.json" ]]; then
        echo "[E1-B eval:$label] merging FSDP shards -> $merged"
        ( cd "$repo_root/verl" && \
          PYTHONPATH="$repo_root/verl${PYTHONPATH:+:$PYTHONPATH}" \
          python3 "$repo_root/verl/scripts/model_merger.py" merge \
              --backend fsdp \
              --local_dir "$repo_root/$ckpt_rel" \
              --target_dir "$merged" ) 2>&1 | tail -20
    else
        echo "[E1-B eval:$label] merge already present: $merged"
    fi
    if [[ ! -s "$merged/config.json" ]]; then
        echo "[E1-B eval:$label] ERROR: merge failed ($merged/config.json missing)" >&2
        return 1
    fi

    # ---- 2. run the original evaluation ----
    # shellcheck disable=SC1090,SC1091
    source "$repo_root/Agent/evaluation/mhqa_agent/eval_mhqa_agent_4gpu_common.sh"
    export EVAL_MODEL_KIND="e1b"                 # non-"rl" => EVAL_BASE_MODEL_RELATIVE used verbatim
    export EVAL_BASE_MODEL_RELATIVE="$ckpt_rel/merged_hf"
    export EVAL_EXPERIMENT_NAME="$exp_name"
    export EVAL_RL_STEP="$step"
    export WIKI_RAG_SERVER_URL="http://127.0.0.1:8008/retrieve"
    run_mhqa_agent_4gpu_eval

    # ---- 3. aggregate the required artifacts ----
    local eval_log="$repo_root/verl/logs/${exp_name}.log"
    local dump="$repo_root/experiments/${exp_name}/val_generations/0.jsonl"
    python3 "$repo_root/E1-B/scripts/e1b_analyze_eval.py" \
        --label "$label" \
        --log "$eval_log" \
        --dump "$dump" \
        --out "$out_dir"
}

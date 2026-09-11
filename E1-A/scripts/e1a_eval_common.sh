#!/usr/bin/env bash
# Shared E1-A evaluation helper.
#
# Reuses the ORIGINAL GAP evaluation implementation
# (Agent/evaluation/mhqa_agent/eval_mhqa_agent_4gpu_common.sh) **unmodified**, so the
# evaluation protocol (7 benchmarks, n=8 ... val_kwargs echo, greedy, temperature 1.0,
# val_batch_size 512, max_model_len 12288) is identical to the GAP baseline evaluations.
#
# The only things this helper changes are the model path (as required: E1-A checkpoint)
# and the experiment/output name.  Because the shared script hardcodes
# `experiments/DAPO-GAP3B-MHQA-Agent-4gpu/global_step_${EVAL_RL_STEP}/actor` for
# EVAL_MODEL_KIND=rl, we use a non-"rl" kind and pre-merge the checkpoint ourselves.
#
# Usage (from an entrypoint):
#   e1a_run_eval <ckpt_actor_dir_rel_to_repo> <experiment_name> <out_dir> <label>
set -Eeuo pipefail

e1a_run_eval() {
    local ckpt_rel="$1" exp_name="$2" out_dir="$3" label="$4"
    local repo_root="$5"

    mkdir -p "$out_dir"
    local merged="$repo_root/$ckpt_rel/merged_hf"

    # ---- 1. merge FSDP shards -> HF (idempotent) ----
    if [[ ! -s "$merged/config.json" ]]; then
        echo "[E1-A eval:$label] merging FSDP shards -> $merged"
        ( cd "$repo_root/verl" && \
          PYTHONPATH="$repo_root/verl${PYTHONPATH:+:$PYTHONPATH}" \
          python3 "$repo_root/verl/scripts/model_merger.py" merge \
              --backend fsdp \
              --local_dir "$repo_root/$ckpt_rel" \
              --target_dir "$merged" ) 2>&1 | tail -20
    else
        echo "[E1-A eval:$label] merge already present: $merged"
    fi
    if [[ ! -s "$merged/config.json" ]]; then
        echo "[E1-A eval:$label] ERROR: merge failed ($merged/config.json missing)" >&2
        return 1
    fi

    # ---- 2. run the original evaluation ----
    # shellcheck disable=SC1090,SC1091
    source "$repo_root/Agent/evaluation/mhqa_agent/eval_mhqa_agent_4gpu_common.sh"
    export EVAL_MODEL_KIND="e1a"                 # non-"rl" => use EVAL_BASE_MODEL_RELATIVE verbatim
    export EVAL_BASE_MODEL_RELATIVE="$ckpt_rel/merged_hf"
    export EVAL_EXPERIMENT_NAME="$exp_name"
    export EVAL_RL_STEP=70
    export WIKI_RAG_SERVER_URL="http://127.0.0.1:8008/retrieve"
    run_mhqa_agent_4gpu_eval

    # ---- 3. aggregate the required artifacts ----
    local eval_log="$repo_root/verl/logs/${exp_name}.log"
    local dump="$repo_root/experiments/${exp_name}/val_generations/0.jsonl"
    python3 "$repo_root/E1-A/scripts/e1a_analyze_eval.py" \
        --label "$label" \
        --log "$eval_log" \
        --dump "$dump" \
        --out "$out_dir"
}

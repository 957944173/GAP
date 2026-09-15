#!/usr/bin/env bash
# Shared E1-E evaluation helper: reuses the ORIGINAL GAP evaluation implementation
# (Agent/evaluation/mhqa_agent/eval_mhqa_agent_4gpu_common.sh) unmodified; only the
# checkpoint path / experiment name / label change.
#   e1e_run_eval <ckpt_actor_rel_to_repo> <experiment_name> <out_dir> <label> <step>
set -Eeuo pipefail

e1e_run_eval() {
    local ckpt_rel="$1" exp_name="$2" out_dir="$3" label="$4" step="$5"
    local repo_root="${REPO_ROOT:-/data01/wyy/Graph-Agent-Planning}"
    local e1e_root="$repo_root/E1-E"
    mkdir -p "$out_dir"
    local merged="$repo_root/$ckpt_rel/merged_hf"
    if [[ ! -s "$merged/config.json" ]]; then
        echo "[E1-E eval:$label] merging FSDP shards -> $merged"
        ( cd "$repo_root/verl" && \
          PYTHONPATH="$repo_root/verl${PYTHONPATH:+:$PYTHONPATH}" \
          python3 "$repo_root/verl/scripts/model_merger.py" merge \
              --backend fsdp --local_dir "$repo_root/$ckpt_rel" --target_dir "$merged" ) 2>&1 | tail -20
    else
        echo "[E1-E eval:$label] merge already present: $merged"
    fi
    if [[ ! -s "$merged/config.json" ]]; then
        echo "[E1-E eval:$label] ERROR: merge failed" >&2; return 1
    fi
    # shellcheck disable=SC1090,SC1091
    source "$repo_root/Agent/evaluation/mhqa_agent/eval_mhqa_agent_4gpu_common.sh"
    export EVAL_MODEL_KIND="e1e"
    export EVAL_BASE_MODEL_RELATIVE="$ckpt_rel/merged_hf"
    export EVAL_EXPERIMENT_NAME="$exp_name"
    export EVAL_RL_STEP="$step"
    export WIKI_RAG_SERVER_URL="http://127.0.0.1:8008/retrieve"
    run_mhqa_agent_4gpu_eval
    local eval_log="$repo_root/verl/logs/${exp_name}.log"
    local dump="$repo_root/experiments/${exp_name}/val_generations/0.jsonl"
    python3 "$e1e_root/scripts/e1e_analyze_eval.py" --label "$label" --log "$eval_log" --dump "$dump" --out "$out_dir"
    python3 "$e1e_root/scripts/e1e_collect_evaluation_artifacts.py" --eval-root "$e1e_root/eval_results" --out "$e1e_root/eval_results" || true
}

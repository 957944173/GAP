#!/usr/bin/env bash
# =====================================================================================
# E1-E training launcher (both phases).
#
#   E1E_PHASE=D1   original GAP step60 -> step70   (total_training_steps=70)
#   E1E_PHASE=D2   E1-E  own step70  -> step120    (total_training_steps=120)
#
# E1-E reward semantics (E1-E_task.md section 1, frozen by E1-E0):
#   * mixed group (1 <= k <= 7)                 -> R_i = A_i        (no shaping)
#   * quota-selected all-correct efficiency grp -> R_i = 1 + 0.05*E_i, lambda = 0.05 fixed
#   * everything else                             -> dropped by the filter
# Group selection is pre-registered stream order: only mixed groups advance the stopping
# counter; k=8 efficiency groups are cached (q = 2) and injected at the stopping batch as
# mixed[:B-m] + cache[:m].  The trainer's `keep(group) iff std > 0` operates on the metric
# the manager publishes (`e1e_quota_metric`), while `data.batch['acc']` is restored to the
# ORIGINAL EM.  All numerics (lr, batch sizes, n=8, GRPO, DAPO clip, epochs, temperature,
# max turns, tool config, save/test freq, GPUs, scheduler, full-state resume) are unchanged
# from GAP/E1-B.
# =====================================================================================
set -Eeuo pipefail
set -x

ulimit -n 65535

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E1E_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$E1E_ROOT/.." && pwd)"
E1E_CODE="$E1E_ROOT/code"

PHASE="${E1E_PHASE:?E1E_PHASE must be D1 or D2}"

# ---------------------------------------------------------------- run directory
RUN_TS="${E1E_RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
export E1E_RUN_DIR="${E1E_RUN_DIR:-$E1E_ROOT/runs/${PHASE}_${RUN_TS}}"
if [[ -e "$E1E_RUN_DIR" ]]; then
    echo "[E1-E] refusing to overwrite existing run dir $E1E_RUN_DIR" >&2
    exit 9
fi
mkdir -p "$E1E_RUN_DIR"/{logs,diagnostics,config,preflight}
echo "$E1E_RUN_DIR" > "$E1E_ROOT/runs/latest_${PHASE}.txt"
TRAIN_LOG="$E1E_ROOT/logs/train_${PHASE}_${RUN_TS}.log"
exec 1> >(tee -a "$E1E_RUN_DIR/stdout.log" "$TRAIN_LOG") 2> >(tee -a "$E1E_RUN_DIR/stderr.log" >&2)
echo "[E1-E:$PHASE] run dir: $E1E_RUN_DIR"
echo "[E1-E:$PHASE] train log: $TRAIN_LOG"

# ---------------------------------------------------------------- environment
if [[ -z "${WIKI_RAG_SERVER_URL:-}" || -z "${GRM_API_KEY:-}" ]]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/environment.sh"
fi
export WIKI_RAG_SERVER_URL="http://127.0.0.1:8008/retrieve"
if [[ -n "${CONDA_PREFIX:-}" && ":${LD_LIBRARY_PATH:-}:" != *":$CONDA_PREFIX/lib:"* ]]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi
export CC=/usr/bin/gcc CXX=/usr/bin/g++ CUDAHOSTCXX=/usr/bin/g++
unset CFLAGS CXXFLAGS
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_CUMEM_ENABLE=0
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

# ---------------------------------------------------------------- params (identical to E1-B)
ACTOR_LR=1e-6
TRAIN_BS=32
PPO_MINI_BS=32
GEN_BS=160
EPOCHS=2
N=8
PPO_MICRO_BSZ_PER_GPU=2
LOG_PROB_MICRO_BSZ_PER_GPU=8
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 8))
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
N_GPUS_PER_NODE=4
GEN_TP=2
use_dynamic_bsz=True
offload=false

export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
CURRENT_DIR="$REPO_ROOT"
export NNODES=1
export PROJECT_NAME="Graph-Agent-Planning"
SAVE_MODEL_FOLDER="${CURRENT_DIR}/experiments"
export EXPERIMENT_NAME="${E1E_EXP_NAME:-DAPO-GAP3B-MHQA-Agent-E1E-q2-step60to120-4gpu}"
export BASE_MODEL="${CURRENT_DIR}/experiments/exp_1_lr1e-5_wr0.03_bs1_ga8_4gpu"
export VLLM_ATTENTION_BACKEND=XFORMERS
TRAIN_DATASETS="${CURRENT_DIR}/Agent/data/mhqa_agent/GAP-MHQA-RL-Dataset/GAP-RL-16w.cleaned.parquet"
VAL_NQ="${CURRENT_DIR}/Agent/data/mhqa_agent/test_benchmarks/nq.cleaned.parquet"
WIKI_SEARCH="${CURRENT_DIR}/verl/verl/tools/config/search_tool_config/wiki_rag_config.yaml"
LOG_DIR="${CURRENT_DIR}/verl/logs"
mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------- phase parameters
GAP_STEP60_CKPT="${SAVE_MODEL_FOLDER}/DAPO-GAP3B-MHQA-Agent-4gpu/global_step_60"
E1E_EXP_DIR="${SAVE_MODEL_FOLDER}/${EXPERIMENT_NAME}"
if [[ "$PHASE" == "D1" ]]; then
    STEPS=70
    RESUME_FROM="$GAP_STEP60_CKPT"
    EXPECTED_START_STEP=60
    ALLOW_EXISTING=0
else
    STEPS=120
    RESUME_FROM="${E1E_EXP_DIR}/global_step_70"
    EXPECTED_START_STEP=70
    ALLOW_EXISTING=1
fi
SAVE_FREQ=10
TEST_FREQ=10

# ---------------------------------------------------------------- preflight
E1E_SCORER="${E1E_CODE}/e1e_scorer.py"
PREFLIGHT_OK=1
{
  echo "E1-E:$PHASE preflight $(date -Is)"
  echo "repo_root=$REPO_ROOT"
  echo "run_dir=$E1E_RUN_DIR"
  echo "experiment_name=$EXPERIMENT_NAME"
  echo "default_local_dir=${E1E_EXP_DIR}"
  echo "resume_from_path=$RESUME_FROM"
  echo "expected_start_step=$EXPECTED_START_STEP"
  echo "filter_metric=e1e_quota_metric  (E1-E quota-gated filter metric, published by the scorer)"
  echo "quota_q=${E1E_QUOTA:-2} batch_target=${E1E_BATCH_TARGET:-32}"
  for p in "$RESUME_FROM" "$TRAIN_DATASETS" "$VAL_NQ" "$WIKI_SEARCH" "$BASE_MODEL" "$E1E_SCORER" \
           "${E1E_CODE}/e1e_reward_manager.py" "${E1E_CODE}/e1e_cost.py" "${E1E_CODE}/pythonpath_e1e/sitecustomize.py"; do
      if [[ -e "$p" ]]; then echo "OK      $p"; else echo "MISSING $p"; PREFLIGHT_OK=0; fi
  done
  for f in "$RESUME_FROM/actor/model_world_size_4_rank_0.pt" \
           "$RESUME_FROM/actor/model_world_size_4_rank_3.pt" \
           "$RESUME_FROM/actor/optim_world_size_4_rank_0.pt" \
           "$RESUME_FROM/actor/extra_state_world_size_4_rank_0.pt" \
           "$RESUME_FROM/data.pt"; do
      if [[ -s "$f" ]]; then echo "OK      $f"; else echo "MISSING $f"; PREFLIGHT_OK=0; fi
  done
  echo "wiki_url_effective=$WIKI_RAG_SERVER_URL"
} | tee "$E1E_RUN_DIR/preflight/preflight_train.txt"
if [[ "$PREFLIGHT_OK" != "1" ]]; then
    echo "[E1-E:$PHASE] PREFLIGHT FAILED -- see $E1E_RUN_DIR/preflight/preflight_train.txt" >&2
    echo "STATUS=BLOCKED_PREFLIGHT" > "$E1E_RUN_DIR/preflight/status.txt"
    exit 1
fi

# wiki reachability (reuse, never restart an already-running service)
if ! LD_LIBRARY_PATH="" curl --silent --show-error --fail --max-time 10 \
        -H 'Content-Type: application/json' -X POST "$WIKI_RAG_SERVER_URL" \
        -d '{"queries": []}' >/dev/null; then
    echo "[E1-E:$PHASE] wiki service unreachable at $WIKI_RAG_SERVER_URL" >&2
    echo "STATUS=BLOCKED_WIKI" > "$E1E_RUN_DIR/preflight/status.txt"
    exit 4
fi

# ---------------------------------------------------------------- no-overwrite guard
if [[ "$ALLOW_EXISTING" == "0" ]]; then
    if [[ -e "$E1E_EXP_DIR" ]]; then
        echo "[E1-E:D1] refusing to run: $E1E_EXP_DIR already exists" >&2
        exit 1
    fi
else
    TRACKER="$(cat "$E1E_EXP_DIR/latest_checkpointed_iteration.txt" 2>/dev/null || echo MISSING)"
    if [[ "$TRACKER" != "70" ]]; then
        echo "[E1-E:D2] refusing to run: tracker is '$TRACKER', expected exactly 70" >&2
        exit 1
    fi
    C70="$E1E_EXP_DIR/global_step_70"
    for f in "$C70/actor/model_world_size_4_rank_0.pt" "$C70/actor/optim_world_size_4_rank_0.pt" \
             "$C70/actor/extra_state_world_size_4_rank_0.pt" "$C70/data.pt"; do
        if [[ ! -s "$f" ]]; then echo "[E1-E:D2] incomplete step70: $f" >&2; exit 1; fi
    done
    echo "[E1-E:D2] continuing the existing E1-E experiment dir (tracker=70) - no other experiment is touched"
fi

# ---------------------------------------------------------------- provenance
{
  echo "## git status --short"
  git -C "$REPO_ROOT" status --short 2>&1 | head -40 || true
} > "$E1E_RUN_DIR/config/git_diff.txt" 2>&1 || true

cat > "$E1E_RUN_DIR/config/reward_config.json" <<JSON
{
  "experiment": "E1-E",
  "phase": "$PHASE",
  "reward_name": "solved-group gated/capped efficiency reward (E1-E: no mixed-group shaping)",
  "train_reward_manager": "e1e_quota_gated",
  "train_scorer": "compute_score_em_efficiency_batch",
  "train_scorer_path": "$E1E_SCORER",
  "formula": "mixed group (1<=k<=7): R_i = A_i; selected k=8 efficiency group: R_i = 1 + 0.05*E_i; wrong rollout -> 0",
  "lambda": 0.05,
  "cost": "logical_search_batches",
  "filter_metric": "e1e_quota_metric",
  "filter_metric_note": "algorithm.filter_groups.metric=e1e_quota_metric -> mixed groups publish EM, cached all-correct efficiency groups publish 1+0.05E; only groups materialised in the quota batch have std>0",
  "quota": "${E1E_QUOTA:-2}",
  "batch_target": "${E1E_BATCH_TARGET:-32}",
  "reward_mixed": "R_i = A_i (original EM, no shaping)",
  "reward_efficiency_group": "R_i = 1 + 0.05 * E_i",
  "token_level_rewards": "shaped reward (feeds GRPO advantage)",
  "acc_metric": "data.batch['acc'] restored to original EM",
  "val_scorer": "$CURRENT_DIR/verl/verl/utils/reward_score/mhqa_eval.py:compute_score_em_batch (UNCHANGED)"
}
JSON
cat > "$E1E_RUN_DIR/config/launch_command.txt" <<CMD
export E1E_PHASE=$PHASE
export E1E_RUN_DIR=$E1E_RUN_DIR
bash $SCRIPT_DIR/run_e1e_train.sh
CMD

python3 - "$E1E_CODE" "$E1E_RUN_DIR/preflight/code_manifest.txt" <<'PY'
import hashlib, os, sys
code_dir, out = sys.argv[1], sys.argv[2]
lines = ["# E1-E code manifest (sha256)", f"# code_dir={code_dir}"]
for dp, dn, fn in os.walk(code_dir):
    dn[:] = [d for d in dn if d != "__pycache__"]
    for f in sorted(fn):
        p = os.path.join(dp, f)
        lines.append(f"{hashlib.sha256(open(p,'rb').read()).hexdigest()}  {os.path.relpath(p, os.path.dirname(code_dir))}")
open(out, "w").write("\n".join(lines) + "\n")
PY

export E1E_START_GLOBAL_STEP="$EXPECTED_START_STEP"
export E1E_LAMBDA=0.05
export E1E_QUOTA="${E1E_QUOTA:-2}"
export E1E_BATCH_TARGET="${E1E_BATCH_TARGET:-$TRAIN_BS}"
export E1E_FILTER_METRIC="e1e_quota_metric"
export PYTHONPATH="${E1E_CODE}/pythonpath_e1e:${PYTHONPATH:-}"

# ---------------------------------------------------------------- passive monitors
python3 "${E1E_CODE}/e1e_stall_watchdog.py" --log "$TRAIN_LOG" --run-dir "$E1E_RUN_DIR" \
    >"$E1E_RUN_DIR/logs/watchdog_stdout.log" 2>&1 &
WATCHDOG_PID=$!
python3 "$E1E_ROOT/scripts/e1e_safety_monitor.py" --run-dir "$E1E_RUN_DIR" --log "$TRAIN_LOG" --interval 300 \
    >"$E1E_RUN_DIR/logs/safety_monitor_stdout.log" 2>&1 &
MONITOR_PID=$!
trap 'kill -TERM ${WATCHDOG_PID} ${MONITOR_PID} 2>/dev/null || true' EXIT
echo "[E1-E:$PHASE] watchdog=$WATCHDOG_PID monitor=$MONITOR_PID"

# ---------------------------------------------------------------- train
cd "$CURRENT_DIR/verl"
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    ray_init.num_cpus=16 \
    algorithm.adv_estimator=grpo \
    algorithm.filter_groups.enable=true \
    algorithm.filter_groups.metric=e1e_quota_metric \
    data.train_files=["${TRAIN_DATASETS}"] \
    data.val_files=["${VAL_NQ}"] \
    data.train_batch_size="${TRAIN_BS}" \
    data.gen_batch_size="${GEN_BS}" \
    data.val_batch_size=128 \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.shuffle=true \
    data.return_raw_chat=true \
    data.filter_overlong_prompts=False \
    actor_rollout_ref.model.path="$BASE_MODEL" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.hybrid_engine=true \
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
    actor_rollout_ref.actor.optim.lr_warmup_steps=3 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BS}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BSZ_PER_GPU \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.checkpoint.save_contents="['model', 'optimizer', 'extra']" \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio_low=$CLIP_RATIO_LOW \
    actor_rollout_ref.actor.clip_ratio_high=$CLIP_RATIO_HIGH \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.rollout.max_model_len=${actor_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BSZ_PER_GPU \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$GEN_TP \
    actor_rollout_ref.rollout.name=sglang_async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.n=$N \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BSZ_PER_GPU \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    trainer.logger=['console'] \
    trainer.val_only=false \
    trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=$NNODES \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs="${EPOCHS}" \
    trainer.total_training_steps="${STEPS}" \
    trainer.default_local_dir="${E1E_EXP_DIR}" \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path="${RESUME_FROM}" \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.max_turns=8 \
    +actor_rollout_ref.rollout.multi_turn.format=qwen \
    actor_rollout_ref.rollout.multi_turn.use_xml_tool_parser=true \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$WIKI_SEARCH" \
    reward_model.reward_manager="e1e_quota_gated" \
    custom_reward_function.train_path="${E1E_SCORER}" \
    custom_reward_function.train_name="compute_score_em_efficiency_batch" \
    custom_reward_function.val_path="${CURRENT_DIR}/verl/verl/utils/reward_score/mhqa_eval.py" \
    custom_reward_function.val_name="compute_score_em_batch" \
    2>&1 | tee -a "$TRAIN_LOG"

TRAIN_RC=${PIPESTATUS[0]}
echo "[E1-E:$PHASE] python main_ppo exit code: $TRAIN_RC"
kill -TERM ${WATCHDOG_PID} ${MONITOR_PID} 2>/dev/null || true
wait ${WATCHDOG_PID} ${MONITOR_PID} 2>/dev/null || true

# ---------------------------------------------------------------- post-checks
END_STEP=$STEPS
{
  echo "E1-E:$PHASE post-training check $(date -Is)"
  echo "tracker: $(cat "${E1E_EXP_DIR}/latest_checkpointed_iteration.txt" 2>/dev/null)"
  echo "resume line: $(grep -c "Setting global step to ${EXPECTED_START_STEP}" "$TRAIN_LOG" || true)"
  for s in 70 80 90 100 110 120; do
      d="${E1E_EXP_DIR}/global_step_${s}"
      if [[ -d "$d" ]]; then
          echo "OK   $d actor_bytes=$(du -sb "$d/actor" 2>/dev/null | cut -f1) data_pt=$([[ -s "$d/data.pt" ]] && echo yes || echo no)"
      else
          echo "absent $d"
      fi
  done
  echo "steps seen: $(grep -oE 'step:[0-9]+ -' "$TRAIN_LOG" | grep -oE '[0-9]+' | sort -n -u | tr '\n' ' ')"
  echo "errors (OOM/NCCL/Traceback): $(grep -cE 'OutOfMemoryError|NCCL error|Traceback \(most recent call last\)' "$TRAIN_LOG" || true)"
} | tee "$E1E_RUN_DIR/preflight/post_train_check.txt"

echo "[E1-E:$PHASE] done rc=$TRAIN_RC; run dir $E1E_RUN_DIR"
exit "$TRAIN_RC"

#!/usr/bin/env bash
# =====================================================================================
# E1-A training entry: step60 -> step70 with the
# "Success-conditioned group-relative efficiency reward" (lambda = 0.05).
#
# Derivation (E1-A_task.md "训练入口"): copied from the original GAP launcher chain
#   Agent/train/mhqa_agent/rl/run_gen_bs_supervisor.sh
#     -> Agent/train/mhqa_agent/rl/train_dapo_mhqa_agent_wiki.sh
# and then adapted, keeping EVERY training parameter byte-identical to the original
# except the reward function (this is the only intended change).
#
# Differences vs the original training script (complete list):
#   * resume_mode=resume_path / resume_from_path=<original step60 checkpoint>
#   * total_training_steps=70, save_freq=10, test_freq=10   (start 60 -> stop 70)
#   * EXPERIMENT_NAME / default_local_dir: unique E1-A identifier (never collides)
#   * GEN_BS fixed at 160 (the value the original supervisor had settled on at step20
#     and kept through step200, verified from supervisor.log; E1-A bypasses the
#     reactive gen_bs_supervisor.py because it only needs 10 fixed steps)
#   * reward_model.reward_manager=e1a_batch_shaped  +  custom_reward_function.train_path
#     -> E1-A shaped scorer; **the validation scorer is unchanged**
#     (mhqa_eval.py / compute_score_em_batch), so validation EM stays the original EM
#   * ray_init.num_cpus=16 (host-RAM mitigation proven necessary in E1-0)
#   * E1-A runtime fixes copied from E1-0 (SGLang GPU visibility under NOSET=1,
#     deferred CUDA-IPC reductions patch, checkpoint map_location="cpu") via PYTHONPATH
#   * HF-merge tail removed here (the merge is performed by run_e1a_eval_step70.sh,
#     which is where the merged model is actually consumed)
#
# Everything else (LR, batch sizes, clip ratios, context window, rollout/tool config,
# filter_groups, GRPO estimator, n=8, seed, max turns) is unchanged.
# =====================================================================================
set -Eeuo pipefail
set -x

ulimit -n 65535

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
E1A_CODE="$SCRIPT_DIR/code"

# ---------------------------------------------------------------- run directory
RUN_TS="${E1A_RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
export E1A_RUN_DIR="${E1A_RUN_DIR:-$SCRIPT_DIR/run_${RUN_TS}}"
mkdir -p "$E1A_RUN_DIR"/{logs,diagnostics,config,preflight}
echo "$E1A_RUN_DIR" > "$SCRIPT_DIR/latest_run.txt"
exec 1> >(tee -a "$E1A_RUN_DIR/stdout.log") 2> >(tee -a "$E1A_RUN_DIR/stderr.log" >&2)
echo "[E1-A] run dir: $E1A_RUN_DIR"

# ---------------------------------------------------------------- environment
if [[ -z "${WIKI_RAG_SERVER_URL:-}" || -z "${GRM_API_KEY:-}" ]]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/environment.sh"
fi
# environment.sh derives WIKI_RAG_SERVER_URL from `hostname -I` (LAN IP) while the RAG
# service is bound to 127.0.0.1 only; same override as gen_bs_supervisor.py:251.
export WIKI_RAG_SERVER_URL="http://127.0.0.1:8008/retrieve"

if [[ -n "${CONDA_PREFIX:-}" && ":${LD_LIBRARY_PATH:-}:" != *":$CONDA_PREFIX/lib:"* ]]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi
export CC=/usr/bin/gcc CXX=/usr/bin/g++ CUDAHOSTCXX=/usr/bin/g++
unset CFLAGS CXXFLAGS

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_CUMEM_ENABLE=0
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

# ---------------------------------------------------------------- params (identical to original)
ACTOR_LR=1e-6
TRAIN_BS=32
PPO_MINI_BS=32
GEN_BS=160
EPOCHS=2
STEPS=70
SAVE_FREQ=10
TEST_FREQ=10
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
export EXPERIMENT_NAME="DAPO-GAP3B-MHQA-Agent-E1A-step60to70-4gpu"
export BASE_MODEL="${CURRENT_DIR}/experiments/exp_1_lr1e-5_wr0.03_bs1_ga8_4gpu"
export VLLM_ATTENTION_BACKEND=XFORMERS
TRAIN_DATASETS="${CURRENT_DIR}/Agent/data/mhqa_agent/GAP-MHQA-RL-Dataset/GAP-RL-16w.cleaned.parquet"
VAL_NQ="${CURRENT_DIR}/Agent/data/mhqa_agent/test_benchmarks/nq.cleaned.parquet"
WIKI_SEARCH="${CURRENT_DIR}/verl/verl/tools/config/search_tool_config/wiki_rag_config.yaml"

LOG_DIR="${CURRENT_DIR}/verl/logs"
LOG_FILE="${LOG_DIR}/${EXPERIMENT_NAME}.log"
mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------- preflight
STEP60_CKPT="${SAVE_MODEL_FOLDER}/DAPO-GAP3B-MHQA-Agent-4gpu/global_step_60"
E1A_SCORER="${E1A_CODE}/e1a_scorer.py"
PREFLIGHT_OK=1
{
  echo "E1-A preflight $(date -Is)"
  echo "repo_root=$REPO_ROOT"
  echo "run_dir=$E1A_RUN_DIR"
  echo "experiment_name=$EXPERIMENT_NAME"
  echo "default_local_dir=${SAVE_MODEL_FOLDER}/${EXPERIMENT_NAME}"
  for p in "$STEP60_CKPT" "$TRAIN_DATASETS" "$VAL_NQ" "$WIKI_SEARCH" "$BASE_MODEL" "$E1A_SCORER" "${E1A_CODE}/e1a_reward_manager.py" "${E1A_CODE}/pythonpath_e1a/sitecustomize.py"; do
      if [[ -e "$p" ]]; then echo "OK      $p"; else echo "MISSING $p"; PREFLIGHT_OK=0; fi
  done
  for f in "$STEP60_CKPT/actor/model_world_size_4_rank_0.pt" "$STEP60_CKPT/actor/optim_world_size_4_rank_0.pt" "$STEP60_CKPT/data.pt"; do
      if [[ -s "$f" ]]; then echo "OK      $f"; else echo "MISSING $f"; PREFLIGHT_OK=0; fi
  done
  echo "wiki_url_effective=$WIKI_RAG_SERVER_URL"
} | tee "$E1A_RUN_DIR/preflight/preflight_train.txt"
if [[ "$PREFLIGHT_OK" != "1" ]]; then
    echo "[E1-A] PREFLIGHT FAILED -- see $E1A_RUN_DIR/preflight/preflight_train.txt" >&2
    exit 1
fi

# output-collision guard: never overwrite a previous E1-A result
if [[ -e "${SAVE_MODEL_FOLDER}/${EXPERIMENT_NAME}" ]]; then
    echo "[E1-A] refusing to run: ${SAVE_MODEL_FOLDER}/${EXPERIMENT_NAME} already exists" >&2
    echo "       (E1-A must not overwrite previous results; move/remove it explicitly)" >&2
    exit 1
fi

# provenance: git state + reward config + launch command
{
  echo "## git status --short (repo has no commits; see E1-A/preflight/original_files_manifest.txt)"
  git -C "$REPO_ROOT" status --short 2>&1 | head -40 || true
  echo
  echo "## git diff (empty: no original file is tracked/modified by E1-A)"
  git -C "$REPO_ROOT" diff 2>&1 | head -20 || true
} > "$E1A_RUN_DIR/config/git_diff.txt" 2>&1 || true

cat > "$E1A_RUN_DIR/config/reward_config.json" <<JSON
{
  "reward_name": "success-conditioned group-relative efficiency reward",
  "train_reward_manager": "e1a_batch_shaped",
  "train_scorer": "compute_score_em_efficiency_batch",
  "train_scorer_path": "$E1A_SCORER",
  "val_reward_manager": "e1a_batch_shaped (is_valid instance -> delegates to original verify)",
  "val_scorer": "$CURRENT_DIR/verl/verl/utils/reward_score/mhqa_eval.py:compute_score_em_batch (UNCHANGED)",
  "formula": "R_i = A_i * (1 + lambda * E_i)",
  "lambda": 0.05,
  "A_i": "EM correctness, computed by the ORIGINAL compute_score_em (mhqa_train.py, unmodified)",
  "cost": "logical_search_batches (assistant search rounds / logical <wiki_search> blocks)",
  "E_i": "(Cmax - C_i)/(Cmax - Cmin) for correct rollouts of a uid group with >=2 correct rollouts and correct-cost variation, else 0",
  "wrong_rollout_reward": 0.0,
  "acc_metric": "data.batch['acc'] is overwritten with the ORIGINAL EM after the base manager writes the shaped reward",
  "group_key": "uid (n=8 rollouts per group, computed before filter_groups slices the batch)"
}
JSON

cat > "$E1A_RUN_DIR/config/launch_command.txt" <<CMD
# E1-A launcher (exact checksums of the code used are in preflight/code_manifest.txt)
export E1A_RUN_DIR=$E1A_RUN_DIR
bash $SCRIPT_DIR/run_e1a_train.sh
CMD

# code manifest (the files that implement E1-A)
python3 - "$E1A_CODE" "$E1A_RUN_DIR/preflight/code_manifest.txt" <<'PY'
import hashlib, os, sys
code_dir, out = sys.argv[1], sys.argv[2]
lines = ["# E1-A code manifest (sha256)", f"# code_dir={code_dir}"]
for dp, dn, fn in os.walk(code_dir):
    dn[:] = [d for d in dn if d != "__pycache__"]
    for f in sorted(fn):
        p = os.path.join(dp, f)
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        lines.append(f"{h}  {os.path.relpath(p, os.path.dirname(code_dir))}")
open(out, "w").write("\n".join(lines) + "\n")
PY

# ---------------------------------------------------------------- E1-A audit registration
export E1A_START_GLOBAL_STEP=60
export E1A_LAMBDA=0.05
export PYTHONPATH="${E1A_CODE}/pythonpath_e1a:${PYTHONPATH:-}"

# ---------------------------------------------------------------- stall watchdog (passive)
python3 "${E1A_CODE}/e1a_stall_watchdog.py" \
    --log "$LOG_FILE" --run-dir "$E1A_RUN_DIR" \
    >"$E1A_RUN_DIR/logs/watchdog_stdout.log" 2>&1 &
WATCHDOG_PID=$!
trap 'kill -TERM ${WATCHDOG_PID} 2>/dev/null || true' EXIT
echo "[E1-A] stall watchdog pid=${WATCHDOG_PID}"

# ---------------------------------------------------------------- train
cd "$CURRENT_DIR/verl"
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    ray_init.num_cpus=16 \
    algorithm.adv_estimator=grpo \
    algorithm.filter_groups.enable=true \
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
    trainer.default_local_dir="${SAVE_MODEL_FOLDER}/${EXPERIMENT_NAME}" \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path="${STEP60_CKPT}" \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.max_turns=8 \
    +actor_rollout_ref.rollout.multi_turn.format=qwen \
    actor_rollout_ref.rollout.multi_turn.use_xml_tool_parser=true \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$WIKI_SEARCH" \
    reward_model.reward_manager="e1a_batch_shaped" \
    custom_reward_function.train_path="${E1A_SCORER}" \
    custom_reward_function.train_name="compute_score_em_efficiency_batch" \
    custom_reward_function.val_path="${CURRENT_DIR}/verl/verl/utils/reward_score/mhqa_eval.py" \
    custom_reward_function.val_name="compute_score_em_batch" \
    2>&1 | tee "$LOG_FILE"

TRAIN_RC=${PIPESTATUS[0]}
echo "[E1-A] python main_ppo exit code: $TRAIN_RC"

# ---------------------------------------------------------------- post-checks
CKPT70="${SAVE_MODEL_FOLDER}/${EXPERIMENT_NAME}/global_step_70"
{
  echo "E1-A post-training check $(date -Is)"
  if [[ -d "$CKPT70" ]]; then
      echo "OK   $CKPT70"
      ls -la "$CKPT70" "$CKPT70/actor" 2>&1 | head -30
      echo "tracker: $(cat "${SAVE_MODEL_FOLDER}/${EXPERIMENT_NAME}/latest_checkpointed_iteration.txt" 2>/dev/null)"
  else
      echo "MISSING $CKPT70"
  fi
  echo "steps seen in log:"
  grep -oE "step:[0-9]+ -" "$LOG_FILE" 2>/dev/null | sort -u -t: -k2 -n | tr '\n' ' '
  echo
} | tee "$E1A_RUN_DIR/preflight/post_train_check.txt"

exit "$TRAIN_RC"

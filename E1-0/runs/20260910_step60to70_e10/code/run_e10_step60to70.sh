#!/usr/bin/env bash
# E1-0 launch script: step60 -> step70, ORIGINAL GAP reward, passive audit only.
#
# Derived from Agent/train/mhqa_agent/rl/train_dapo_mhqa_agent_wiki.sh (copied,
# not modified in place). ALL original training parameters are kept identical;
# the only changes from the original are marked "E1-0 CHANGE" below:
#   - resume_mode=resume_path / resume_from_path=<step60 checkpoint>
#   - total_training_steps=70, save_freq=10, test_freq=10 (yields exactly one
#     save at step70, matching original SAVE_FREQ/TEST_FREQ cadence of 20 for
#     the *shape* of the schedule while landing exactly on step70)
#   - EXPERIMENT_NAME / default_local_dir: unique E1-0 identifier, never
#     colliding with the original experiment's output
#   - GEN_BS fixed at 160 (original run's steady-state value for the entire
#     step20-200 window per supervisor.log; this run bypasses the reactive
#     gen_bs_supervisor.py entirely since it only needs 10 fixed steps)
#   - reward_model.reward_manager=e10_batch_audit, PYTHONPATH prepended with
#     the E1-0 sitecustomize.py shim dir (registers the audit reward manager
#     via the existing @register() decorator -- zero modification to any
#     original verl file)
#   - custom_reward_function.train_path/train_name point at the E1-0 scorer,
#     which wraps (never reimplements) the original compute_score_em
#   - WIKI_RAG_SERVER_URL forced to 127.0.0.1 post-environment.sh (see
#     rationale below; same precedent as gen_bs_supervisor.py:251)
#   - trailing HF-merge block removed (E1-0 does not need an HF-merged model;
#     the FSDP checkpoint manifest is sufficient audit evidence)
#
# Everything else (ACTOR_LR, TRAIN_BS, PPO_MINI_BS, EPOCHS structure, N=8,
# clip ratios, context window, rollout config, tool config, filter_groups,
# GRPO estimator, val dataset, etc.) is copied byte-for-byte unchanged.
set -Eeuo pipefail
set -x

ulimit -n 65535
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# Auto-load environment.sh (WIKI_RAG_SERVER_URL / GRM_API_KEY placeholder / etc.)
# if the caller hasn't sourced it yet. Idempotent: re-sourcing is harmless.
if [[ -z "${WIKI_RAG_SERVER_URL:-}" || -z "${GRM_API_KEY:-}" ]]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/environment.sh"
fi

# E1-0 CHANGE: environment.sh derives WIKI_RAG_SERVER_URL from `hostname -I`
# (a LAN-facing IP), but the Wiki RAG server here is bound to 127.0.0.1 only
# (confirmed via `ss -tlnp` showing LISTEN on 127.0.0.1:8008 exclusively, and
# a functional POST to the hostname IP returning HTTP_CODE=000 / connection
# failure while the same POST to 127.0.0.1 succeeds). Overriding here matches
# the existing precedent in gen_bs_supervisor.py (line ~251, same override,
# same reasoning) rather than modifying the original environment.sh.
export WIKI_RAG_SERVER_URL="http://127.0.0.1:8008/retrieve"

# sglang spawns nvidia-smi / flashinfer-JIT subprocesses that need libcudart.so.12
# on the loader path. `conda activate` alone doesn't set this — export it here
# so `conda activate parallel-agent && bash <this>` is enough.
if [[ -n "${CONDA_PREFIX:-}" && ":${LD_LIBRARY_PATH:-}:" != *":$CONDA_PREFIX/lib:"* ]]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi

# conda's gcc_linux-64/CFLAGS/CXXFLAGS leak into flashinfer's JIT compile step and
# break it against this box's system GLIBC (2.31). Force the system toolchain.
export CC=/usr/bin/gcc CXX=/usr/bin/g++ CUDAHOSTCXX=/usr/bin/g++
unset CFLAGS CXXFLAGS

export CUDA_VISIBLE_DEVICES=0,1,2,3
# E1-0 SELF-FIX (attempts 3-8, full history in logs/nccl_duplicate_gpu_diagnosis.md,
# logs/checkpoint_load_oom_diagnosis.md and logs/sglang_gpu_collision_diagnosis.md):
# Every attempt run under Ray's *default* per-actor CUDA_VISIBLE_DEVICES remap
# (1, 2, 3, 6, and 7 -- 5 independent tries, including attempt 7 with
# CUDA_DEVICE_ORDER=PCI_BUS_ID added) crashed identically pre-training with
# `NCCL error ... Duplicate GPU detected: rank X and rank 0 both on CUDA
# device 34000` during FSDP's initial _sync_module_params_and_buffers -- a
# 5/5 reproduction rate under any non-NOSET config, ruling out both "rare
# nondeterministic race" and "inconsistent PCI enumeration order" as fixable
# without bypassing Ray's remap. The ONLY setting that has ever cleared this
# crash is RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1 (attempt 4), which
# disables Ray's own per-actor remap in favor of verl's own
# Worker._setup_env_cuda_visible_devices() calling
# torch.cuda.set_device(RAY_LOCAL_RANK) directly. That setting's side effect
# (breaking sglang_rollout.py's TP-group CUDA_VISIBLE_DEVICES union logic,
# see logs/checkpoint_load_oom_diagnosis.md and the "attempt 6" update in
# nccl_duplicate_gpu_diagnosis.md) is now fixed directly via a companion
# monkeypatch, code/e10_sglang_gpu_patch.py (see
# logs/sglang_gpu_collision_diagnosis.md), which replicates Ray's default-remap
# narrowing (CUDA_VISIBLE_DEVICES -> single physical GPU per rank,
# LOCAL_RANK="0") deterministically in our own code instead of relying on
# Ray's racy internal remap. Re-enabling NOSET=1 alongside that patch (attempt
# 8) is therefore the combined fix: it avoids Ray's FSDP/NCCL race AND
# preserves the per-rank single-GPU invariant that both FSDP/NCCL and SGLang
# depend on. NCCL_CUMEM_ENABLE=0 (harmless, vLLM-documented mitigation) is
# kept; CUDA_DEVICE_ORDER=PCI_BUS_ID (proven ineffective in attempt 7) is
# dropped.
export NCCL_CUMEM_ENABLE=0
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
# =====================================================================================================================
#                                      Param (all identical to original except where marked)
# =====================================================================================================================
ACTOR_LR=1e-6
TRAIN_BS=32
PPO_MINI_BS=32
GEN_BS=160                        # E1-0 CHANGE: fixed at original run's step20-200 steady-state value
EPOCHS=2
STEPS=70                          # E1-0 CHANGE: stop at step70 (was 200)
SAVE_FREQ=10                      # E1-0 CHANGE: land exactly on step70 (was 20)
TEST_FREQ=10                      # E1-0 CHANGE: matches SAVE_FREQ (was 20)
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
# =====================================================================================================================
#                                      Env
# =====================================================================================================================
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
CURRENT_DIR="$REPO_ROOT"
export NNODES=1
export PROJECT_NAME="Graph-Agent-Planning"
SAVE_MODEL_FOLDER="${CURRENT_DIR}/experiments"
export EXPERIMENT_NAME="DAPO-GAP3B-MHQA-Agent-E10-step60to70-4gpu"   # E1-0 CHANGE: unique identifier
export BASE_MODEL="${CURRENT_DIR}/experiments/exp_1_lr1e-5_wr0.03_bs1_ga8_4gpu"
export VLLM_ATTENTION_BACKEND=XFORMERS
TRAIN_DATASETS="${CURRENT_DIR}/Agent/data/mhqa_agent/GAP-MHQA-RL-Dataset/GAP-RL-16w.cleaned.parquet"
VAL_NQ="${CURRENT_DIR}/Agent/data/mhqa_agent/test_benchmarks/nq.cleaned.parquet"
# =====================================================================================================================
#                                      Tool
# =====================================================================================================================
WIKI_SEARCH="${CURRENT_DIR}/verl/verl/tools/config/search_tool_config/wiki_rag_config.yaml"

LOG_DIR="${CURRENT_DIR}/verl/logs"
LOG_FILE="${LOG_DIR}/${EXPERIMENT_NAME}.log"
mkdir -p "$LOG_DIR"

for required_path in "$BASE_MODEL" "$TRAIN_DATASETS" "$VAL_NQ" "$WIKI_SEARCH"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Required path does not exist: $required_path" >&2
        exit 1
    fi
done
TRAIN_FILES_OVERRIDE="data.train_files=[\"${TRAIN_DATASETS}\"]"
VAL_FILES_OVERRIDE="data.val_files=[\"${VAL_NQ}\"]"

# =====================================================================================================================
#                                      Wiki RAG server (reuse-only for E1-0)
# =====================================================================================================================
# E1-0 CHANGE: WIKI_SERVER_AUTOSTART=0 -- Wiki health was already verified
# functional (reused) during E1-0 preflight (section 5); this script must
# never start a duplicate server nor risk killing the user's existing one.
export WIKI_SERVER_AUTOSTART=0

# =====================================================================================================================
#                                      E1-0 audit code registration
# =====================================================================================================================
E10_CODE_DIR="${E10_RUN_DIR:?E10_RUN_DIR must be set}/code"
export E10_RUN_DIR
export E10_START_GLOBAL_STEP=60
export PYTHONPATH="${E10_CODE_DIR}/pythonpath_e10:${PYTHONPATH:-}"

STEP60_CKPT="${SAVE_MODEL_FOLDER}/DAPO-GAP3B-MHQA-Agent-4gpu/global_step_60"
if [[ ! -d "$STEP60_CKPT" ]]; then
    echo "step60 checkpoint not found: $STEP60_CKPT" >&2
    exit 1
fi

# =====================================================================================================================
#                                      Train
# =====================================================================================================================
cd "$CURRENT_DIR/verl"
# E1-0 SELF-FIX (attempts 10, 11, 12; see logs/host_ram_oom_diagnosis.md):
# ray_init.num_cpus defaults to null (os.cpu_count()=96 on this box), which
# makes Ray eagerly prestart ~96 idle worker processes at ray.init() time,
# each importing the full torch/sglang/verl dependency stack (~0.47GB RSS
# apiece, ~45GB total) before any actual training actor exists. Combined with
# the 4 real GPU worker processes' own (much larger) host-RAM footprint, this
# reproducibly pushed the node to ~121.3-121.9GB/125.53GB at TaskRunner init
# in both attempt 10 and attempt 12 -- a deterministic ceiling, not a
# transient fluke. This run only ever needs 4 GPU actors + a handful of
# CPU-only driver/reward tasks, so capping the idle worker pool at 16 is a
# pure Ray-infra resource-allocation change (not a rollout/reward/GRPO/
# sampling semantic) that eliminates the OOM without touching any
# section-7-audited config.
# ---------------------------------------------------------------------------------------------------------------------
# E1-0 passive stall watchdog (diagnostics only; added for attempt 19)
# ---------------------------------------------------------------------------------------------------------------------
# Attempt 18 stalled silently right after checkpoint-resume (log frozen, GPU 0%,
# no `test_gen_batch meta info` ever printed) and cost ~2h before it was noticed.
# ptrace/py-spy/gdb are unusable on this box, so the only way to obtain a stack
# is the SIGUSR1 faulthandler hook installed by code/e10_hang_diagnostics.py in
# every E1-0 process.  This watchdog watches the launcher log, and once a stall
# is confirmed (log static for pre-stall-seconds before the first validation
# batch, or post-stall-seconds afterwards) delivers SIGUSR1 to exactly the pids
# that provably loaded that hook, appending all-thread stacks to
# $E10_RUN_DIR/logs/stackdumps/.  It never touches training in any way.
mkdir -p "$E10_RUN_DIR/logs"
python3 "${E10_CODE_DIR}/e10_stall_watchdog.py" \
    --log "$LOG_FILE" \
    --run-dir "$E10_RUN_DIR" \
    >"$E10_RUN_DIR/logs/watchdog_stdout.log" 2>&1 &
WATCHDOG_PID=$!
trap 'kill -TERM ${WATCHDOG_PID} 2>/dev/null || true' EXIT
echo "[E1-0] stall watchdog started pid=${WATCHDOG_PID} (see $E10_RUN_DIR/logs/watchdog.log)"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    ray_init.num_cpus=16 \
    algorithm.adv_estimator=grpo \
    algorithm.filter_groups.enable=true \
    "$TRAIN_FILES_OVERRIDE" \
    "$VAL_FILES_OVERRIDE" \
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
    reward_model.reward_manager="e10_batch_audit" \
    custom_reward_function.train_path="${E10_CODE_DIR}/e10_scorer.py" \
    custom_reward_function.train_name="compute_score_em_batch_e10" \
    custom_reward_function.val_path="${CURRENT_DIR}/verl/verl/utils/reward_score/mhqa_eval.py" \
    custom_reward_function.val_name="compute_score_em_batch" \
    2>&1 | tee "$LOG_FILE"

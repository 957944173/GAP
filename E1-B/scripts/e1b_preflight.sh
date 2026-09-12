#!/usr/bin/env bash
# =====================================================================================
# E1-B preflight (E1-B_task.md section 3 "启动前检查").
#
#   1. E1-A step70 checkpoint must exist -> otherwise STOP AND REPORT (never retrain).
#   2. E1-B output must never be overwritten -> a fresh run_<ts> dir is always created.
#   3. retriever wiki service: reuse if healthy, otherwise start it, logging to
#      E1-B/logs/wiki_service.log (never a duplicate server if one is already up).
#   4. `source environment.sh` before entering the parallel-agent env.
#
# Everything is written under E1-B/ (run-scoped copy under E1-B/run_<ts>/preflight/).
# =====================================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E1B_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$E1B_ROOT/.." && pwd)"

WIKI_URL="http://127.0.0.1:8008/retrieve"
E1A_CKPT_REL="experiments/DAPO-GAP3B-MHQA-Agent-E1A-step60to70-4gpu/global_step_70"
E1A_CKPT="$REPO_ROOT/$E1A_CKPT_REL"

mkdir -p "$E1B_ROOT"/{code_changes,scripts,configs,checkpoints,logs,evaluations,diagnostics}
WIKI_LOG="$E1B_ROOT/logs/wiki_service.log"
mkdir -p "$(dirname "$WIKI_LOG")"

LOG="$E1B_ROOT/logs/preflight.log"
exec > >(tee -a "$LOG") 2>&1

echo "===== E1-B preflight $(date -Is) ====="

# ---------------------------------------------------------------- 1. start checkpoint
echo "--- [1/4] E1-A step70 checkpoint"
CKPT_OK=1
for f in "$E1A_CKPT/actor/model_world_size_4_rank_0.pt" \
         "$E1A_CKPT/actor/model_world_size_4_rank_1.pt" \
         "$E1A_CKPT/actor/model_world_size_4_rank_2.pt" \
         "$E1A_CKPT/actor/model_world_size_4_rank_3.pt" \
         "$E1A_CKPT/actor/optim_world_size_4_rank_0.pt" \
         "$E1A_CKPT/actor/extra_state_world_size_4_rank_0.pt" \
         "$E1A_CKPT/data.pt"; do
    if [[ -s "$f" ]]; then
        echo "OK      $f ($(du -h "$f" | cut -f1))"
    else
        echo "MISSING $f"
        CKPT_OK=0
    fi
done
echo "e1a_step70_tracker=$(cat "$REPO_ROOT/experiments/DAPO-GAP3B-MHQA-Agent-E1A-step60to70-4gpu/latest_checkpointed_iteration.txt" 2>/dev/null || echo '?')"
if [[ "$CKPT_OK" != "1" ]]; then
    echo "FATAL: E1-A step70 checkpoint is incomplete. E1-B_task.md forbids re-training it; STOPPING." >&2
    echo "STATUS=BLOCKED_MISSING_E1A_STEP70" > "$E1B_ROOT/logs/preflight_status.txt"
    exit 3
fi

# ---------------------------------------------------------------- 2. output collision
echo "--- [2/4] output-directory guards (never overwrite)"
E1B_EXP_NAME="${E1B_EXP_NAME:-DAPO-GAP3B-MHQA-Agent-E1B-step70to120-4gpu}"
CHECK_DIRS=(
    "$REPO_ROOT/experiments/$E1B_EXP_NAME"
    "$E1B_ROOT/logs/run"     # run_<ts> pattern
    "$E1B_ROOT/evaluations"
    "$E1B_ROOT/checkpoints"
)
for d in "${CHECK_DIRS[@]}"; do
    if [[ "$d" == *"/run" ]]; then
        echo "run dirs present: $(ls -d "$E1B_ROOT"/run_* 2>/dev/null | wc -l) (a NEW run_<ts> is always created)"
    elif [[ -e "$d" ]] && [[ -n "$(ls -A "$d" 2>/dev/null)" ]]; then
        echo "EXISTS_NONEMPTY $d (will NOT be overwritten; eval/checkpoint outputs get a new subdir)"
    else
        echo "OK_ABSENT_OR_EMPTY $d"
    fi
done

# ---------------------------------------------------------------- 3. wiki service
echo "--- [3/4] retriever wiki service"
probe() {
    LD_LIBRARY_PATH="" curl --silent --show-error --fail --max-time 5 \
        -H 'Content-Type: application/json' -X POST "$1" -d '{"queries": []}' >/dev/null 2>&1
}
real_probe() {
    LD_LIBRARY_PATH="" curl --silent --show-error --fail --max-time 20 \
        -H 'Content-Type: application/json' -X POST "$1" \
        -d '{"queries": ["what is the capital of France"]}'
}
ss -tlnp 2>/dev/null | grep -E ":8008" || echo "(no listener on 8008)"
pgrep -af "wiki_rag_server" || echo "(no wiki_rag_server process)"

WIKI_REUSED=false
WIKI_STARTED=false
if probe "$WIKI_URL"; then
    WIKI_REUSED=true
    echo "wiki service HEALTHY at $WIKI_URL -> REUSING (no duplicate server started, none stopped)"
    echo -n "real-query probe (truncated): "; real_probe "$WIKI_URL" | head -c 300; echo
else
    echo "wiki service NOT reachable -> starting it (log: $WIKI_LOG)"
    CUDA_VISIBLE_DEVICES="${WIKI_SERVER_GPUS:-0,1,2,3}" SERVER_HOST=127.0.0.1 \
        setsid bash "$REPO_ROOT/Agent/tool_servers/wiki_server/launch_rag_server.sh" \
        >"$WIKI_LOG" 2>&1 &
    deadline=$(( $(date +%s) + 900 ))
    while ! probe "$WIKI_URL"; do
        if (( $(date +%s) >= deadline )); then
            echo "FATAL: wiki service not ready in 900s; tail of $WIKI_LOG:" >&2
            tail -n 40 "$WIKI_LOG" >&2 || true
            echo "STATUS=BLOCKED_WIKI_SERVICE" > "$E1B_ROOT/logs/preflight_status.txt"
            exit 4
        fi
        sleep 5
    done
    WIKI_STARTED=true
    echo "wiki service started and healthy"
fi

# ---------------------------------------------------------------- 4. env
echo "--- [4/4] conda env + environment.sh"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source environment.sh >/dev/null
echo "WIKI_RAG_SERVER_URL from environment.sh: ${WIKI_RAG_SERVER_URL:-unset}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate parallel-agent
echo "conda env: ${CONDA_DEFAULT_ENV:-unset} ($CONDA_PREFIX)"
export WIKI_RAG_SERVER_URL="$WIKI_URL"
if real_probe "$WIKI_RAG_SERVER_URL" >/dev/null; then
    echo "functional probe OK: $WIKI_RAG_SERVER_URL"
else
    echo "FATAL: functional probe failed" >&2
    echo "STATUS=BLOCKED_WIKI_PROBE" > "$E1B_ROOT/logs/preflight_status.txt"
    exit 4
fi

# ---------------------------------------------------------------- inputs
echo "--- input / code checks"
for p in \
    "$REPO_ROOT/Agent/data/mhqa_agent/GAP-MHQA-RL-Dataset/GAP-RL-16w.cleaned.parquet" \
    "$REPO_ROOT/Agent/data/mhqa_agent/test_benchmarks/nq.cleaned.parquet" \
    "$E1B_ROOT/code_changes/e1b_scorer.py" \
    "$E1B_ROOT/code_changes/e1b_reward_manager.py" \
    "$E1B_ROOT/code_changes/pythonpath_e1b/sitecustomize.py" \
    "$E1B_ROOT/scripts/run_e1b_train.sh" ; do
    if [[ -s "$p" ]]; then echo "OK      $p"; else echo "MISSING $p"; fi
done

# ---------------------------------------------------------------- integrity manifest
echo "--- original GAP file integrity (sha256; must stay unchanged for the whole run)"
python3 - "$REPO_ROOT" "$E1B_ROOT/configs/original_files_manifest_pre.txt" <<'PY'
import hashlib, os, sys
repo, out = sys.argv[1], sys.argv[2]
files = [
    "Agent/train/mhqa_agent/rl/run_gen_bs_supervisor.sh",
    "Agent/train/mhqa_agent/rl/train_dapo_mhqa_agent_wiki.sh",
    "Agent/train/mhqa_agent/rl/gen_bs_supervisor.py",
    "Agent/evaluation/mhqa_agent/eval_mhqa_agent_rl_step_70_4gpu.sh",
    "Agent/evaluation/mhqa_agent/eval_mhqa_agent_4gpu_common.sh",
    "Agent/evaluation/mhqa_agent/summarize_eval.py",
    "verl/verl/workers/reward_manager/batch.py",
    "verl/verl/workers/reward_manager/__init__.py",
    "verl/verl/utils/reward_score/mhqa_train.py",
    "verl/verl/utils/reward_score/mhqa_eval.py",
    "verl/verl/workers/rollout/sglang_rollout/sglang_rollout.py",
    "verl/verl/workers/rollout/schemas.py",
    "verl/verl/trainer/ppo/ray_trainer.py",
    "verl/verl/trainer/ppo/reward.py",
    "verl/verl/trainer/ppo/core_algos.py",
    "verl/verl/utils/checkpoint/fsdp_checkpoint_manager.py",
    "environment.sh",
    "E1-0/runs/20260910_step60to70_e10/code/e10_reward_manager.py",
    "E1-0/runs/20260910_step60to70_e10/code/e10_sglang_gpu_patch.py",
    "E1-A/code/e1a_scorer.py",
    "E1-A/code/e1a_reward_manager.py",
    "E1-A/code/e1a_cost.py",
]
lines = ["# E1-B preflight original-file manifest (sha256)"]
for rel in files:
    p = os.path.join(repo, rel)
    if os.path.exists(p):
        lines.append(f"{hashlib.sha256(open(p,'rb').read()).hexdigest()}  {rel}")
    else:
        lines.append(f"MISSING  {rel}")
open(out, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
PY

# ---------------------------------------------------------------- host state
echo "--- host state"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv || true
df -h /data01 | tail -1
python3 -c "import torch" 2>/dev/null && echo "torch import OK (parallel-agent)" || echo "torch import FAILED"

{
  echo "STATUS=OK"
  echo "WIKI_REUSED=$WIKI_REUSED"
  echo "WIKI_STARTED=$WIKI_STARTED"
  echo "WIKI_URL=$WIKI_URL"
  echo "WIKI_PIDS=$(pgrep -d, -f wiki_rag_server || true)"
  echo "E1A_STEP70_CKPT=$E1A_CKPT"
  echo "E1A_STEP70_ACTOR_BYTES=$(du -sb "$E1A_CKPT/actor" 2>/dev/null | cut -f1)"
  echo "PARALLEL_AGENT_ENV=${CONDA_DEFAULT_ENV:-unset}"
  echo "TIMESTAMP=$(date -Is)"
} > "$E1B_ROOT/logs/preflight_status.txt"
cat "$E1B_ROOT/logs/preflight_status.txt"

# E1-B_task.md section 3.3: the wiki service state must be recorded in
# E1-B/logs/wiki_service.log (both when reused and when newly started).
{
  echo "===== E1-B preflight wiki record $(date -Is) ====="
  echo "WIKI_REUSED=$WIKI_REUSED WIKI_STARTED=$WIKI_STARTED URL=$WIKI_URL"
  echo "listener: $(ss -tlnp 2>/dev/null | grep -E ':8008' | tr -s ' ' || echo none)"
  echo "process:  $(pgrep -af wiki_rag_server | head -1 || echo none)"
  echo "real-query probe OK"
} >> "$WIKI_LOG"
echo "wiki record appended to $WIKI_LOG"
echo "===== E1-B preflight DONE (STATUS=OK) ====="

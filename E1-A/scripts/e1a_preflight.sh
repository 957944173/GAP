#!/usr/bin/env bash
# E1-A preflight: wiki environment (retriever) + parallel-agent env + input checks.
# Records everything under E1-A/preflight/. Never starts a duplicate wiki server.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E1A_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$E1A_ROOT/.." && pwd)"
OUT="$E1A_ROOT/preflight"
mkdir -p "$OUT"
STATUS="$OUT/wiki_service_status.log"

exec > >(tee -a "$STATUS") 2>&1

echo "===== E1-A preflight $(date -Is) ====="

# ---------------------------------------------------------------- phase 1: retriever env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate retriever
echo "conda env: ${CONDA_DEFAULT_ENV:-unset} ($CONDA_PREFIX)"

WIKI_URL="http://127.0.0.1:8008/retrieve"
probe() {
    LD_LIBRARY_PATH="" curl --silent --show-error --fail --max-time 5 \
        -H 'Content-Type: application/json' -X POST "$1" -d '{"queries": []}' >/dev/null 2>&1
}
real_probe() {
    LD_LIBRARY_PATH="" curl --silent --show-error --fail --max-time 20 \
        -H 'Content-Type: application/json' -X POST "$1" \
        -d '{"queries": ["what is the capital of France"]}'
}

echo "--- port/process check"
ss -tlnp 2>/dev/null | grep -E "8008" || echo "(no listener on 8008)"
pgrep -af "wiki_rag_server" || echo "(no wiki_rag_server process)"

WIKI_REUSED=false
WIKI_STARTED=false
if probe "$WIKI_URL"; then
    WIKI_REUSED=true
    echo "wiki service HEALTHY at $WIKI_URL -> REUSING (will not start or stop it)"
    echo "--- functional retrieval probes"
    echo -n "empty-queries probe response: "
    LD_LIBRARY_PATH="" curl --silent --max-time 10 -H 'Content-Type: application/json' \
        -X POST "$WIKI_URL" -d '{"queries": []}' | head -c 120
    echo
    echo -n "real-query probe response (truncated): "
    real_probe "$WIKI_URL" | head -c 400
    echo
else
    echo "wiki service NOT reachable at $WIKI_URL -> starting via the repo's own launcher"
    # Same mechanism the original supervisor uses (Agent/tool_servers/wiki_server/launch_rag_server.sh)
    WIKI_LOG="$REPO_ROOT/verl/logs/wiki_rag_server_e1a.log"
    CUDA_VISIBLE_DEVICES="${WIKI_SERVER_GPUS:-0,1,2,3}" SERVER_HOST=127.0.0.1 \
        setsid bash "$REPO_ROOT/Agent/tool_servers/wiki_server/launch_rag_server.sh" \
        >"$WIKI_LOG" 2>&1 &
    WIKI_PGID=$!
    echo "launched wiki server pgid=$WIKI_PGID log=$WIKI_LOG"
    deadline=$(( $(date +%s) + 900 ))
    while ! probe "$WIKI_URL"; do
        if (( $(date +%s) >= deadline )); then
            echo "ERROR: wiki service did not become ready in 900s; tail of $WIKI_LOG:" >&2
            tail -n 40 "$WIKI_LOG" >&2 || true
            exit 1
        fi
        sleep 5
    done
    WIKI_STARTED=true
    echo "wiki service started and healthy"
fi

# ---------------------------------------------------------------- phase 2: parallel-agent env
conda activate parallel-agent
echo "conda env: ${CONDA_DEFAULT_ENV:-unset} ($CONDA_PREFIX)"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source environment.sh >/dev/null
echo "WIKI_RAG_SERVER_URL from environment.sh: ${WIKI_RAG_SERVER_URL:-unset}"
if probe "$WIKI_RAG_SERVER_URL"; then
    echo "environment.sh URL is reachable"
else
    echo "environment.sh URL NOT reachable (LAN IP); the E1-A launcher overrides it to $WIKI_URL"
fi
export WIKI_RAG_SERVER_URL="$WIKI_URL"
if real_probe "$WIKI_RAG_SERVER_URL" >/dev/null; then
    echo "post-override functional probe OK: $WIKI_RAG_SERVER_URL"
else
    echo "ERROR: post-override functional probe FAILED" >&2
    exit 1
fi

# ---------------------------------------------------------------- phase 3: inputs
STEP60="$REPO_ROOT/experiments/DAPO-GAP3B-MHQA-Agent-4gpu/global_step_60"
echo "--- input checks"
for p in \
    "$STEP60/actor/model_world_size_4_rank_0.pt" \
    "$STEP60/actor/optim_world_size_4_rank_0.pt" \
    "$STEP60/actor/extra_state_world_size_4_rank_0.pt" \
    "$STEP60/data.pt" \
    "$REPO_ROOT/Agent/data/mhqa_agent/GAP-MHQA-RL-Dataset/GAP-RL-16w.cleaned.parquet" \
    "$REPO_ROOT/Agent/data/mhqa_agent/test_benchmarks/nq.cleaned.parquet" \
    "$E1A_ROOT/code/e1a_scorer.py" \
    "$E1A_ROOT/code/e1a_reward_manager.py" \
    "$E1A_ROOT/code/pythonpath_e1a/sitecustomize.py" \
    "$E1A_ROOT/run_e1a_train.sh" ; do
    if [[ -s "$p" ]]; then echo "OK      $p"; else echo "MISSING $p"; fi
done

echo "--- original file integrity (sha256, must be unmodified)"
python3 - "$REPO_ROOT" "$OUT/original_files_manifest.txt" <<'PY'
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
]
lines = ["# E1-A preflight original-file manifest (sha256)"]
for rel in files:
    p = os.path.join(repo, rel)
    if os.path.exists(p):
        lines.append(f"{hashlib.sha256(open(p,'rb').read()).hexdigest()}  {rel}")
    else:
        lines.append(f"MISSING  {rel}")
open(out, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
PY

{
  echo "WIKI_REUSED=$WIKI_REUSED"
  echo "WIKI_STARTED=$WIKI_STARTED"
  echo "WIKI_URL=$WIKI_URL"
  echo "WIKI_PIDS=$(pgrep -d, -f wiki_rag_server || true)"
  echo "RETRIEVER_PROBE_OK=true"
  echo "PARALLEL_AGENT_ENV=${CONDA_DEFAULT_ENV:-unset}"
  echo "TIMESTAMP=$(date -Is)"
} > "$OUT/wiki_service_status.json"

echo "===== E1-A preflight DONE ====="

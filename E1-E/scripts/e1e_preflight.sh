#!/usr/bin/env bash
# =====================================================================================
# E1-E preflight (E1-E_task.md §5, §6, §7).
#   1. Wiki retriever: probe the URL the repo tool config uses; REUSE if healthy,
#      otherwise start it the way the repo/E1-A/E1-B did.  Logged to E1-E/logs/.
#   2. conda activate parallel-agent + source environment.sh, record interpreter/versions.
#   3. GPU / process / Ray preflight (never kills an unknown process).
#   4. Input integrity: ORIGINAL GAP step60 (not E1-A step70 / E1-B step120) + dataset +
#      tool config; output-collision guards.
# =====================================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E1E_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$E1E_ROOT/.." && pwd)"

WIKI_URL="http://127.0.0.1:8008/retrieve"
GAP_STEP60="$REPO_ROOT/experiments/DAPO-GAP3B-MHQA-Agent-4gpu/global_step_60"
E1E_EXP_NAME="${E1E_EXP_NAME:-DAPO-GAP3B-MHQA-Agent-E1E-q2-step60to120-4gpu}"

mkdir -p "$E1E_ROOT"/{code,scripts,configs,logs,diagnostics,analysis,eval_results,manifests,runs,checkpoints}
WIKI_LOG="$E1E_ROOT/logs/wiki_service.log"
LOG="$E1E_ROOT/logs/preflight.log"
exec > >(tee -a "$LOG") 2>&1

echo "===== E1-E preflight $(date -Is) ====="

# ---------------------------------------------------------------- 1. wiki
echo "--- [1/4] wiki retriever service"
echo "tool config URL: $(grep -oE 'http://[^\"[:space:]]+' "$REPO_ROOT/verl/verl/tools/config/search_tool_config/wiki_rag_config.yaml" 2>/dev/null | head -1 || echo '(not found)')"
ss -tlnp 2>/dev/null | grep -E ':8008' || echo "(no listener on 8008)"
pgrep -af "wiki_rag_server" || echo "(no wiki_rag_server process)"
probe() { LD_LIBRARY_PATH="" curl --silent --show-error --fail --max-time 5 -H 'Content-Type: application/json' -X POST "$1" -d '{"queries": []}' >/dev/null 2>&1; }
real_probe() { LD_LIBRARY_PATH="" curl --silent --show-error --fail --max-time 20 -H 'Content-Type: application/json' -X POST "$1" -d '{"queries": ["what is the capital of France"]}'; }
WIKI_REUSED=false; WIKI_STARTED=false
if probe "$WIKI_URL"; then
    WIKI_REUSED=true
    echo "wiki HEALTHY at $WIKI_URL -> REUSING (no duplicate server; nothing stopped)"
    echo -n "functional retrieval probe (truncated): "; real_probe "$WIKI_URL" | head -c 300; echo
else
    echo "wiki NOT reachable -> starting with the repo's own launcher"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate retriever
    echo "retriever env: ${CONDA_DEFAULT_ENV:-unset}"
    CUDA_VISIBLE_DEVICES="${WIKI_SERVER_GPUS:-0,1,2,3}" SERVER_HOST=127.0.0.1 \
        setsid bash "$REPO_ROOT/Agent/tool_servers/wiki_server/launch_rag_server.sh" >"$WIKI_LOG" 2>&1 &
    deadline=$(( $(date +%s) + 900 ))
    while ! probe "$WIKI_URL"; do
        if (( $(date +%s) >= deadline )); then
            echo "FATAL: wiki not ready in 900s; tail of $WIKI_LOG:" >&2; tail -n 40 "$WIKI_LOG" >&2 || true; exit 4
        fi
        sleep 5
    done
    WIKI_STARTED=true
    echo "wiki started and healthy"
fi
{
  echo "===== E1-E wiki record $(date -Is) ====="
  echo "WIKI_REUSED=$WIKI_REUSED WIKI_STARTED=$WIKI_STARTED URL=$WIKI_URL"
  echo "listener: $(ss -tlnp 2>/dev/null | grep -E ':8008' | tr -s ' ' || echo none)"
  echo "process:  $(pgrep -af wiki_rag_server | head -1 || echo none)"
  echo "functional retrieval probe OK"
} >> "$WIKI_LOG"

# ---------------------------------------------------------------- 2. env
echo "--- [2/4] conda env + environment.sh"
cd "$REPO_ROOT"
[[ -f "$REPO_ROOT/environment.sh" ]] || { echo "FATAL: no $REPO_ROOT/environment.sh"; exit 5; }
# shellcheck disable=SC1091
source environment.sh >/dev/null
echo "sourced $REPO_ROOT/environment.sh"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate parallel-agent
echo "conda env: ${CONDA_DEFAULT_ENV:-unset} ($CONDA_PREFIX)"
export WIKI_RAG_SERVER_URL="$WIKI_URL"
echo "which python: $(which python)"
echo "python version: $(python --version 2>&1)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
real_probe "$WIKI_RAG_SERVER_URL" >/dev/null && echo "wiki functional probe OK" || { echo "FATAL: wiki probe failed"; exit 4; }
python3 - <<'PY'
import importlib, sys, torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
for m in ("ray", "sglang", "transformers", "pandas", "numpy"):
    try: print(m, getattr(importlib.import_module(m), "__version__", "?"))
    except Exception as e: print(m, "n/a", type(e).__name__)
print("python", sys.version.split()[0])
PY

# ---------------------------------------------------------------- 3. GPU / process
echo "--- [3/4] GPU / process / ray preflight"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
echo "GAP main_ppo processes: $(pgrep -af 'verl.trainer.main_ppo' | wc -l)"
pgrep -af "verl.trainer.main_ppo" || echo "(none)"
echo "ray processes: $(pgrep -c -f 'ray::' 2>/dev/null || echo 0)"
echo "sglang processes: $(pgrep -c -f 'sglang' 2>/dev/null || echo 0)"
echo "wiki processes: $(pgrep -c -f 'wiki_rag_server' 2>/dev/null || echo 0)"
echo "(E1-E kills no unknown process)"

# ---------------------------------------------------------------- 4. inputs + guards
echo "--- [4/4] input integrity + output guards"
CKPT_OK=1
for f in "$GAP_STEP60/actor/model_world_size_4_rank_0.pt" "$GAP_STEP60/actor/model_world_size_4_rank_1.pt" \
         "$GAP_STEP60/actor/model_world_size_4_rank_2.pt" "$GAP_STEP60/actor/model_world_size_4_rank_3.pt" \
         "$GAP_STEP60/actor/optim_world_size_4_rank_0.pt" "$GAP_STEP60/actor/optim_world_size_4_rank_3.pt" \
         "$GAP_STEP60/actor/extra_state_world_size_4_rank_0.pt" "$GAP_STEP60/actor/extra_state_world_size_4_rank_3.pt" \
         "$GAP_STEP60/data.pt"; do
    if [[ -s "$f" ]]; then echo "OK      $f ($(stat -c %s "$f") B)"; else echo "MISSING $f"; CKPT_OK=0; fi
done
for p in "$REPO_ROOT/Agent/data/mhqa_agent/GAP-MHQA-RL-Dataset/GAP-RL-16w.cleaned.parquet" \
         "$REPO_ROOT/Agent/data/mhqa_agent/test_benchmarks/nq.cleaned.parquet" \
         "$REPO_ROOT/verl/verl/tools/config/search_tool_config/wiki_rag_config.yaml" \
         "$REPO_ROOT/experiments/exp_1_lr1e-5_wr0.03_bs1_ga8_4gpu" \
         "$E1E_ROOT/code/e1e_scorer.py" "$E1E_ROOT/code/e1e_reward_manager.py" \
         "$E1E_ROOT/code/pythonpath_e1e/sitecustomize.py" "$E1E_ROOT/scripts/run_e1e_train.sh"; do
    if [[ -s "$p" ]]; then echo "OK      $p"; else echo "MISSING $p"; CKPT_OK=0; fi
done
echo "gap_step60 tracker=$(cat "$REPO_ROOT/experiments/DAPO-GAP3B-MHQA-Agent-4gpu/latest_checkpointed_iteration.txt" 2>/dev/null || echo '?')"
if [[ "$CKPT_OK" != "1" ]]; then
    echo "FATAL: missing inputs; E1-E_task.md §7 forbids substituting them. STOPPING." >&2
    echo "STATUS=BLOCKED_INPUTS" > "$E1E_ROOT/logs/preflight_status.txt"; exit 3
fi
for d in "$REPO_ROOT/experiments/$E1E_EXP_NAME" "$E1E_ROOT/runs/latest_D1.txt" "$E1E_ROOT/eval_results/latest_step70.txt"; do
    if [[ -e "$d" ]]; then echo "EXISTS $d (launchers never overwrite; D2 may continue its own dir)"; else echo "ABSENT $d"; fi
done

# ---------------------------------------------------------------- provenance pre
python3 - "$REPO_ROOT" "$E1E_ROOT/manifests/original_files_manifest_pre.json" <<'PY'
import hashlib, json, os, sys
repo, out = sys.argv[1], sys.argv[2]
files = [
    "Agent/train/mhqa_agent/rl/run_gen_bs_supervisor.sh",
    "Agent/train/mhqa_agent/rl/train_dapo_mhqa_agent_wiki.sh",
    "Agent/train/mhqa_agent/rl/gen_bs_supervisor.py",
    "Agent/evaluation/mhqa_agent/eval_mhqa_agent_4gpu_common.sh",
    "Agent/evaluation/mhqa_agent/summarize_eval.py",
    "verl/verl/workers/reward_manager/batch.py",
    "verl/verl/workers/reward_manager/__init__.py",
    "verl/verl/utils/reward_score/mhqa_train.py",
    "verl/verl/utils/reward_score/mhqa_eval.py",
    "verl/verl/workers/rollout/sglang_rollout.py",
    "verl/verl/workers/rollout/schemas.py",
    "verl/verl/trainer/ppo/ray_trainer.py",
    "verl/verl/trainer/ppo/reward.py",
    "verl/verl/trainer/ppo/core_algos.py",
    "verl/verl/utils/checkpoint/fsdp_checkpoint_manager.py",
    "environment.sh",
    "E1-0/runs/20260910_step60to70_e10/code/e10_reward_manager.py",
    "E1-A/code/e1a_scorer.py",
    "E1-A/code/e1a_reward_manager.py",
    "E1-B/code_changes/e1b_scorer.py",
    "E1-B/code_changes/e1b_reward_manager.py",
    "E1-C/FINAL_REPORT.md",
    "E1-C/code/e1c_common.py",
]
lines = []
for rel in files:
    p = os.path.join(repo, rel)
    if os.path.exists(p):
        lines.append({"path": rel, "bytes": os.path.getsize(p), "mtime": os.path.getmtime(p),
                      "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()})
    else:
        lines.append({"path": rel, "missing": True})
json.dump({"manifest": "pre", "files": lines}, open(out, "w"), indent=2)
print(f"wrote {out} ({len(lines)} files)")
PY

{
  echo "STATUS=OK"
  echo "WIKI_REUSED=$WIKI_REUSED"
  echo "WIKI_STARTED=$WIKI_STARTED"
  echo "WIKI_URL=$WIKI_URL"
  echo "WIKI_PIDS=$(pgrep -d, -f wiki_rag_server || true)"
  echo "GAP_STEP60_CKPT=$GAP_STEP60"
  echo "GAP_STEP60_ACTOR_BYTES=$(du -sb "$GAP_STEP60/actor" 2>/dev/null | cut -f1)"
  echo "PARALLEL_AGENT_ENV=${CONDA_DEFAULT_ENV:-unset}"
  echo "TIMESTAMP=$(date -Is)"
} > "$E1E_ROOT/logs/preflight_status.txt"
cat "$E1E_ROOT/logs/preflight_status.txt"
echo "===== E1-E preflight DONE (STATUS=OK) ====="

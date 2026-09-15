#!/usr/bin/env bash
# E1-E integrity re-verification: the original GAP files must be byte-identical before
# and after the whole E1-E experiment (E1-E_task.md "不要直接修改原 GAP 主代码").
#
# Recomputes the manifest written by scripts/e1e_preflight.sh and diffs it.
# Outputs: E1-E/configs/original_files_manifest_post.txt
#          E1-E/configs/original_files_diff.txt  (empty == unchanged)
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E1E_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$E1E_ROOT/.." && pwd)"
OUT="$E1E_ROOT/configs"
mkdir -p "$OUT"

python3 - "$REPO_ROOT" "$OUT/original_files_manifest_post.txt" <<'PY'
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
lines = ["# E1-E preflight original-file manifest (sha256)"]
for rel in files:
    p = os.path.join(repo, rel)
    if os.path.exists(p):
        lines.append(f"{hashlib.sha256(open(p,'rb').read()).hexdigest()}  {rel}")
    else:
        lines.append(f"MISSING  {rel}")
open(out, "w").write("\n".join(lines) + "\n")
PY

PRE="$OUT/original_files_manifest_pre.txt"
POST="$OUT/original_files_manifest_post.txt"
DIFF="$OUT/original_files_diff.txt"
if [[ -s "$PRE" ]]; then
    diff -u "$PRE" "$POST" > "$DIFF" || true
    if [[ -s "$DIFF" ]]; then
        echo "CHANGED ORIGINAL FILES DETECTED:"; cat "$DIFF"
        exit 1
    fi
    echo "OK: $(($(wc -l < "$POST") - 1)) original files unchanged (sha256 identical)"
else
    echo "no pre manifest at $PRE"
fi

# E1-0 / E1-A result directories must not have been touched either
echo "--- E1-0 / E1-A mtimes (must predate this experiment)"
stat -c '%y  %n' "$REPO_ROOT"/E1-0/runs/20260910_step60to70_e10/FINAL_REPORT.md 2>/dev/null || true
stat -c '%y  %n' "$REPO_ROOT"/E1-A/FINAL_REPORT.md 2>/dev/null || true

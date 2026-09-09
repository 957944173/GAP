#!/usr/bin/env python3
"""Run a tiny GRPO grouping audit against GAP's current implementation.

This script imports the repository's compute_grpo_outcome_advantage directly;
it intentionally does not start Ray, a model, a rollout worker, or training.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
VERL_ROOT = REPO_ROOT / "verl"
if str(VERL_ROOT) not in sys.path:
    sys.path.insert(0, str(VERL_ROOT))

from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage  # noqa: E402


CASES = {
    "A_[1,1,0x6]": [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "B_[1,0.8,0x6]": [1.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "C_[1,0.9,0.7,0x5]": [1.0, 0.9, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0],
}


def run_case(rewards: list[float]) -> dict:
    # Put each scalar at the final response token, matching BatchRewardManager.
    token_level_rewards = torch.zeros((len(rewards), 4), dtype=torch.float32)
    token_level_rewards[:, -1] = torch.tensor(rewards, dtype=torch.float32)
    response_mask = torch.ones_like(token_level_rewards)
    uid = np.array(["prompt-0"] * len(rewards), dtype=object)

    advantages, returns = compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=uid,
        norm_adv_by_std_in_grpo=True,
    )
    return {
        "input_outcome_rewards": rewards,
        "advantage_shape": list(advantages.shape),
        "advantage_per_trajectory": advantages[:, 0].tolist(),
        "advantage_tensor": advantages.tolist(),
        "returns_equal_advantages": bool(torch.equal(returns, advantages)),
    }


def main() -> None:
    output = {
        "implementation": "verl.trainer.ppo.core_algos.compute_grpo_outcome_advantage",
        "group_key": "uid = prompt-0 for all eight rows",
        "response_mask_shape": [8, 4],
        "cases": {name: run_case(values) for name, values in CASES.items()},
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""E1-A reward manager: shaped reward + accuracy separation + diagnostics.

Registered name: `e1a_batch_shaped` (subclass of the ORIGINAL
`verl/verl/workers/reward_manager/batch.py:BatchRewardManager`).

What it changes vs the original `batch` manager
-----------------------------------------------
1. For TRAINING it computes the shaped reward via
   `e1a_scorer.compute_score_em_efficiency_batch` (R = A*(1+0.05*E), cost =
   `logical_search_batches` from the structured rollout metadata, group key =
   uid, computed over the full unfiltered n=8 group the trainer hands to the
   reward function before `filter_groups` runs).
2. `data.batch["acc"]` is restored to the ORIGINAL EM after the base class
   wrote the shaped reward into it, so accuracy metrics never see the shaping
   (E1-A_task.md "batch.py 要求").  `token_level_scores`/`token_level_rewards`
   keep the shaped reward, which is what GRPO and the group filter consume.
3. Passive diagnostics are appended to `<E1A_RUN_DIR>/diagnostics/`.

For VALIDATION (`num_examine == 1`, i.e. the `val_reward_fn` instance created
by main_ppo.py:162) `verify()` delegates to the original implementation
verbatim, so validation behaviour/metrics are bit-identical to the baseline.

The original `batch.py` file is NOT modified: this is a runtime-registered
subclass, loaded through the PYTHONPATH sitecustomize shim (same mechanism the
original script uses to select `reward_model.reward_manager`).
"""
import json
import os
import threading
from datetime import datetime

import numpy as np
import torch

from verl.workers.reward_manager import register
from verl.workers.reward_manager.batch import BatchRewardManager

import e1a_cost
import e1a_state
import e1a_trainer_patch

E1A_RUN_DIR = os.environ.get("E1A_RUN_DIR")
E1A_START_GLOBAL_STEP = int(os.environ.get("E1A_START_GLOBAL_STEP", "60"))


@register("e1a_batch_shaped")
class E1ABatchRewardManager(BatchRewardManager):
    """Success-conditioned group-relative efficiency reward + passive audit."""

    _lock = threading.Lock()
    _call_index = {"train": 0}

    def __init__(self, tokenizer, num_examine, compute_score, reward_fn_key="data_source", **reward_kwargs):
        super().__init__(tokenizer, num_examine, compute_score, reward_fn_key=reward_fn_key, **reward_kwargs)
        self.is_train = num_examine == 0
        self.run_dir = E1A_RUN_DIR
        self.pid = os.getpid()
        self._last_step = None
        self._call_within_step = 0
        if self.is_train and self.run_dir:
            e1a_trainer_patch.ensure_driver_patch()
            diag = os.path.join(self.run_dir, "diagnostics")
            os.makedirs(diag, exist_ok=True)
            self.rollouts_path = os.path.join(diag, f"rollouts_pid{self.pid}.jsonl")
            self.groups_path = os.path.join(diag, f"groups_pid{self.pid}.jsonl")
            self.calls_path = os.path.join(diag, f"calls_pid{self.pid}.jsonl")
            self.errors_path = os.path.join(diag, f"diagnostics_errors_pid{self.pid}.log")

    # ------------------------------------------------------------------ #
    def _next_call_index(self):
        with E1ABatchRewardManager._lock:
            idx = E1ABatchRewardManager._call_index["train"]
            E1ABatchRewardManager._call_index["train"] += 1
            return idx

    # ------------------------------------------------------------------ #
    def verify(self, data):
        """Training path: identical string construction to the original
        `BatchRewardManager.verify`, plus uid / cost / query extraction so the
        scorer can compute the group-relative efficiency term."""
        if not self.is_train:
            # validation: delegate verbatim to the original implementation
            return super().verify(data)

        prompt_ids = data.batch["prompts"]
        response_ids = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]

        prompt_len = prompt_ids.shape[-1]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

        responses_str = []
        prompts_str = []
        questions = [info["question"] for info in data.non_tensor_batch["extra_info"]]
        for i in range(len(data)):
            valid_len = valid_response_lengths[i]
            valid_response_ids = response_ids[i][:valid_len]
            prompt_str = self.tokenizer.decode(prompt_ids[i], skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            prompts_str.append(prompt_str)
            responses_str.append(response_str)

        ground_truths = [item.non_tensor_batch["reward_model"].get("ground_truth", None) for item in data]
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        extras = data.non_tensor_batch.get("extra_info", [None] * len(data))

        n = len(data)
        uids = data.non_tensor_batch.get("uid", None)
        cost_info = e1a_cost.costs_from_batch(data.non_tensor_batch, n)

        scores = self.compute_score(
            questions=questions,
            ground_truths=ground_truths,
            responses=responses_str,
            data_sources=data_sources,
            prompts=prompts_str,
            extra_infos=extras,
            uids=uids,
            costs=cost_info["costs"],
            queries=cost_info["queries"],
            conversation_turns=cost_info["conversation_turns"],
            **self.reward_kwargs,
        )
        return scores

    # ------------------------------------------------------------------ #
    def __call__(self, data, return_dict=False):
        result = super().__call__(data, return_dict=True)

        if self.is_train:
            # ---- E1-A requirement: acc = original EM, reward = shaped ----
            try:
                info = result.get("reward_extra_info", {}) or {}
                em = info.get("em")
                if em is not None:
                    data.batch["acc"] = torch.tensor(
                        [float(x) for x in em], dtype=torch.float32,
                        device=data.batch["prompts"].device,
                    )
            except Exception:  # noqa: BLE001 -- must never affect training
                pass
            try:
                self._diagnostics(data, result)
            except Exception as e:  # noqa: BLE001 -- diagnostics must never affect training
                try:
                    if self.run_dir:
                        with open(self.errors_path, "a") as f:
                            f.write(f"DIAG ERROR (swallowed): {type(e).__name__}: {e}\n")
                except Exception:
                    pass

        if return_dict:
            return result
        return result["reward_tensor"]

    # ------------------------------------------------------------------ #
    def _diagnostics(self, data, result):
        """Passive diagnostics required by E1-A_task.md: reward / efficiency
        bonus distributions, logical_search_batches, search_queries,
        parallel_factor, group_correct_count and filter retention."""
        if not self.run_dir:
            return
        info = result.get("reward_extra_info", {}) or {}
        rt = result.get("reward_tensor")
        n = len(data)
        call_index = self._next_call_index()
        gstep = e1a_state.current_global_step()
        if gstep is not None and gstep != self._last_step:
            self._last_step = gstep
            self._call_within_step = 0
        else:
            self._call_within_step += 1
        ts = datetime.now().isoformat(timespec="seconds")

        uids = data.non_tensor_batch.get("uid", [None] * n)
        ds_list = data.non_tensor_batch.get(self.reward_fn_key, [None] * n)
        cr_list = data.non_tensor_batch.get("complete_reason", [None] * n)
        cost_info = e1a_cost.costs_from_batch(data.non_tensor_batch, n)
        prompt_ids = data.batch["prompts"]
        resp_ids = data.batch["responses"]
        prompt_len = prompt_ids.shape[-1]
        vlen = data.batch["attention_mask"][:, prompt_len:].sum(dim=-1)

        em_list = info.get("em", [None] * n)
        score_list = info.get("score", [None] * n)
        eff_list = info.get("efficiency", [None] * n)
        cost_list = info.get("cost", [None] * n)

        group_agg = {}
        rollout_records = []
        for i in range(n):
            uid = str(uids[i])
            reward_tensor_sum = None
            if rt is not None:
                try:
                    reward_tensor_sum = float(rt[i].sum().item())
                except Exception:  # noqa: BLE001
                    pass
            cost = cost_list[i] if i < len(cost_list) else cost_info["costs"][i]
            queries = cost_info["queries"][i]
            turns = cost_info["conversation_turns"][i]
            em = em_list[i] if i < len(em_list) else None
            eff = eff_list[i] if i < len(eff_list) else None
            score = score_list[i] if i < len(score_list) else None
            rec = {
                "run_id": os.path.basename(self.run_dir),
                "pid": self.pid,
                "call_index": call_index,
                "call_within_step": self._call_within_step,
                "global_step": gstep,
                "timestamp": ts,
                "uid": uid,
                "data_source": ds_list[i] if i < len(ds_list) else None,
                "em": em,
                "reward": score,
                "reward_tensor_sum": reward_tensor_sum,
                "efficiency": eff,
                "logical_search_batches": cost,
                "search_queries": queries,
                "parallel_factor": ((queries / cost) if (queries and cost) else 0.0),
                "conversation_turns": turns,
                "response_token_length": int(vlen[i].item()),
                "zero_search": bool(cost == 0),
                "metadata_valid": bool(cost_info["metadata_valid"][i]),
                "complete_reason": cr_list[i] if i < len(cr_list) else None,
            }
            rollout_records.append(rec)
            g = group_agg.setdefault(uid, {"em": [], "reward": [], "eff": [], "cost": [], "queries": []})
            g["em"].append(1.0 if (em == 1 or em == 1.0) else 0.0)
            g["reward"].append(float(score) if score is not None else 0.0)
            g["eff"].append(float(eff) if eff is not None else 0.0)
            g["cost"].append(cost)
            g["queries"].append(queries)

        with open(self.rollouts_path, "a") as f:
            for r in rollout_records:
                f.write(json.dumps(r, default=str) + "\n")

        group_records = []
        n_baseline_keep = n_shaped_keep = n_newly = 0
        for uid, g in group_agg.items():
            em = np.asarray(g["em"], dtype=float)
            rw = np.asarray(g["reward"], dtype=float)
            baseline_keep = bool(len(em) == 1 or np.std(em) > 0.0)
            shaped_keep = bool(len(rw) == 1 or np.std(rw) > 0.0)
            newly = bool(shaped_keep and not baseline_keep)
            n_baseline_keep += int(baseline_keep)
            n_shaped_keep += int(shaped_keep)
            n_newly += int(newly)
            cost_vals = [c for c in g["cost"] if c is not None]
            group_records.append({
                "pid": self.pid, "call_index": call_index, "global_step": gstep, "timestamp": ts,
                "group_id": uid, "group_size": len(em),
                "group_correct_count": int(em.sum()),
                "group_wrong_count": int(len(em) - em.sum()),
                "costs": cost_vals,
                "cost_min_correct": (min([c for c, a in zip(g["cost"], em) if a == 1.0 and c is not None])
                                     if any(a == 1.0 and c is not None for c, a in zip(g["cost"], em)) else None),
                "cost_max_correct": (max([c for c, a in zip(g["cost"], em) if a == 1.0 and c is not None])
                                     if any(a == 1.0 and c is not None for c, a in zip(g["cost"], em)) else None),
                "mean_efficiency": float(np.mean(g["eff"])),
                "max_efficiency": float(np.max(g["eff"])),
                "efficiency_active": bool(np.max(g["eff"]) > 0.0),
                "mean_reward": float(np.mean(rw)),
                "baseline_keep_em_std_gt_0": baseline_keep,
                "shaped_keep_reward_std_gt_0": shaped_keep,
                "newly_added_by_shaping": newly,
            })
        with open(self.groups_path, "a") as f:
            for gr in group_records:
                f.write(json.dumps(gr, default=str) + "\n")

        rw_all = [float(r["reward"]) for r in rollout_records if r["reward"] is not None]
        eff_all = [float(r["efficiency"]) for r in rollout_records if r["efficiency"] is not None]
        cost_all = [r["logical_search_batches"] for r in rollout_records if r["logical_search_batches"] is not None]
        q_all = [r["search_queries"] for r in rollout_records if r["search_queries"] is not None]
        call_rec = {
            "pid": self.pid, "call_index": call_index, "call_within_step": self._call_within_step,
            "global_step": gstep, "timestamp": ts, "n_rollouts": n, "n_groups": len(group_agg),
            "reward_mean": float(np.mean(rw_all)) if rw_all else None,
            "reward_nonzero_rate": float(np.mean([1.0 if x > 0 else 0.0 for x in rw_all])) if rw_all else None,
            "efficiency_mean": float(np.mean(eff_all)) if eff_all else None,
            "efficiency_nonzero_count": int(sum(1 for x in eff_all if x > 0)),
            "cost_mean": float(np.mean(cost_all)) if cost_all else None,
            "queries_mean": float(np.mean(q_all)) if q_all else None,
            "n_baseline_keep": n_baseline_keep,
            "n_shaped_keep": n_shaped_keep,
            "n_newly_added": n_newly,
            "n_efficiency_active_groups": int(sum(1 for gr in group_records if gr["efficiency_active"])),
            "metadata_invalid": int(sum(1 for r in rollout_records if not r["metadata_valid"])),
        }
        with open(self.calls_path, "a") as f:
            f.write(json.dumps(call_rec, default=str) + "\n")

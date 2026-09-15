"""E1-E reward manager: q=2 solved-group gated / capped efficiency training.

Registered name: ``e1e_quota_gated`` (subclass of the ORIGINAL
``verl/verl/workers/reward_manager/batch.py:BatchRewardManager``).

What happens for TRAINING (``num_examine == 0``)
------------------------------------------------
1. The full generation batch is classified per n=8 group (correctness `k`, search-cost
   variation) using the *same* scorer the trainer will use.
2. `e1e_quota_plan.plan_batch` decides, in stream order, which mixed groups (1<=k<=7) are
   kept and which cached all-correct efficiency groups (k=8 with cost variation) enter the
   optimizer batch, with quota q=2.  Only mixed groups count towards the generation
   stopping target `data.train_batch_size`; efficiency groups never advance it.
3. The batch is materialised in place as ``mixed[:B-m] + efficiency_cache[:m]``; efficiency
   groups seen in *earlier* generation batches of the same step are injected from a cache of
   stashed slices (the trainer drops them from their own batch, so they must be re-attached
   at the stopping batch).
4. The reward is group-type specific: mixed groups get the original EM (no shaping), the
   selected efficiency groups get ``1 + 0.05 * E``.  ``data.batch["acc"]`` is always restored
   to the TRUE EM.
5. The published filter metric ``e1e_quota_metric`` (see e1e_scorer) has std > 0 for exactly
   the groups in the materialised batch, so the trainer keeps all of them and stops after
   exactly the batch where the mixed count reaches B.

For VALIDATION (``num_examine == 1``) everything delegates to the original implementation.
"""
import json
import os
import threading
from collections import OrderedDict, defaultdict
from datetime import datetime

import numpy as np
import torch

from verl.protocol import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.batch import BatchRewardManager

import e1e_cost
import e1e_quota_plan
import e1e_state
import e1e_trainer_patch

E1E_RUN_DIR = os.environ.get("E1E_RUN_DIR")
E1E_QUOTA = int(os.environ.get("E1E_QUOTA", "2"))
E1E_BATCH_TARGET = int(os.environ.get("E1E_BATCH_TARGET", "0"))   # 0 -> read config
E1E_START_GLOBAL_STEP = int(os.environ.get("E1E_START_GLOBAL_STEP", "60"))


@register("e1e_quota_gated")
class E1EQuotaRewardManager(BatchRewardManager):
    _lock = threading.Lock()
    _call_index = {"train": 0}

    def __init__(self, tokenizer, num_examine, compute_score, reward_fn_key="data_source", **reward_kwargs):
        super().__init__(tokenizer, num_examine, compute_score, reward_fn_key=reward_fn_key, **reward_kwargs)
        self.is_train = num_examine == 0
        self.run_dir = E1E_RUN_DIR
        self.pid = os.getpid()
        self.quota = E1E_QUOTA
        self.batch_target = E1E_BATCH_TARGET or None
        self._q = e1e_quota_plan.new_state()
        self._cache_slices = OrderedDict()
        self._step_rows = []
        self._keep_nothing = False      # this call planned an empty keep set
        self._no_keep_calls = 0
        self.quota_errors = 0
        self.steps_seen = []
        if self.is_train and self.run_dir:
            e1e_trainer_patch.ensure_driver_patch()
            diag = os.path.join(self.run_dir, "diagnostics")
            os.makedirs(diag, exist_ok=True)
            self.calls_path = os.path.join(diag, f"quota_calls_pid{self.pid}.jsonl")
            self.steps_path = os.path.join(diag, f"quota_steps_pid{self.pid}.jsonl")
            self.groups_path = os.path.join(diag, f"quota_groups_pid{self.pid}.jsonl")
            self.rollouts_path = os.path.join(diag, f"rollouts_pid{self.pid}.jsonl")
            self.errors_path = os.path.join(diag, f"quota_errors_pid{self.pid}.log")

    # ------------------------------------------------------------------ helpers
    def _next_call_index(self):
        with E1EQuotaRewardManager._lock:
            idx = E1EQuotaRewardManager._call_index["train"]
            E1EQuotaRewardManager._call_index["train"] += 1
            return idx

    def _target(self):
        if self.batch_target:
            return self.batch_target
        try:
            cfg = self.config if hasattr(self, "config") else None
        except Exception:  # noqa: BLE001
            cfg = None
        return 32

    # ------------------------------------------------------------------ verify
    def verify(self, data):
        """Identical string construction to the original manager (and to E1-B/E1-D), plus the
        uid / cost / query extraction the E1-E scorer needs."""
        if not self.is_train:
            return super().verify(data)

        prompt_ids = data.batch["prompts"]
        response_ids = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]
        prompt_len = prompt_ids.shape[-1]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

        responses_str, prompts_str = [], []
        questions = [info["question"] for info in data.non_tensor_batch["extra_info"]]
        for i in range(len(data)):
            valid_len = valid_response_lengths[i]
            valid_response_ids = response_ids[i][:valid_len]
            prompts_str.append(self.tokenizer.decode(prompt_ids[i], skip_special_tokens=True))
            responses_str.append(self.tokenizer.decode(valid_response_ids, skip_special_tokens=True))

        ground_truths = [item.non_tensor_batch["reward_model"].get("ground_truth", None) for item in data]
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        extras = data.non_tensor_batch.get("extra_info", [None] * len(data))
        n = len(data)
        uids = data.non_tensor_batch.get("uid", None)
        cost_info = e1e_cost.costs_from_batch(data.non_tensor_batch, n)
        return self.compute_score(
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

    # ------------------------------------------------------------------ quota
    def _apply_quota_inplace(self, data):
        """Classify the generation batch, plan the quota batch and materialise it in place."""
        if not self.is_train or len(data) == 0:
            return
        gstep = e1e_state.current_global_step()
        if self._q["step"] != gstep:
            e1e_quota_plan.reset_state(self._q)
            self._q["step"] = gstep
            self._cache_slices = OrderedDict()
            self._step_rows = []

        target = self._target()
        scores_full = self.verify(data)

        uids = data.non_tensor_batch["uid"]
        order, idx = [], {}
        for i, u in enumerate(uids):
            u = str(u)
            if u not in idx:
                idx[u] = []
                order.append(u)
            idx[u].append(i)

        cur_groups, gid2group = [], {}
        for u in order:
            rows = idx[u]
            k = int(sum(1 for j in rows if scores_full[j]["em"] >= 0.5))
            gap = None
            if k == len(rows) and k >= 2:
                costs = [scores_full[j]["cost"] for j in rows if scores_full[j]["cost"] is not None]
                if costs:
                    gap = max(costs) - min(costs)
            gtype = e1e_quota_plan.classify_group(k, len(rows), gap)
            gid = f"{self._q['step']}:{u}"
            g = {"gid": gid, "uid": u, "idx": rows, "k": k, "cost_gap": gap, "type": gtype,
                 "source": str(data.non_tensor_batch["data_source"][rows[0]])
                 if "data_source" in data.non_tensor_batch else None}
            cur_groups.append(g)
            gid2group[gid] = g

        plan = e1e_quota_plan.plan_batch(cur_groups, self._q, target_batch=target, quota=self.quota)

        # ---- materialise: mixed kept (this batch) + efficiency kept (cache order) ----
        parts, kept_meta = [], []
        keep_set = set(plan["mixed_kept"])
        for g in cur_groups:
            if g["gid"] in keep_set:
                parts.append(data[g["idx"]])
                kept_meta.append((g, "mixed"))
        for gid in plan["efficiency_kept"]:
            if gid in gid2group:
                g = gid2group[gid]
                parts.append(data[g["idx"]])
            else:
                g = {"gid": gid, "uid": gid.split(":")[-1], "idx": [], "k": 8,
                     "cost_gap": None, "type": e1e_quota_plan.GROUP_EFFICIENCY, "source": None}
                parts.append(self._cache_slices[gid][0])
            kept_meta.append((g, "efficiency"))
        # stash this batch's efficiency groups for later injection (stream order, capped at q)
        # NOTE: the stash must happen even when `parts` is empty, otherwise a later injection
        # would find nothing to insert.
        if not plan["is_stopping"]:
            for g in cur_groups:
                if (g["type"] == e1e_quota_plan.GROUP_EFFICIENCY and g["gid"] in plan["cache_after"]
                        and g["gid"] not in self._cache_slices):
                    self._cache_slices[g["gid"]] = (data[g["idx"]],
                                                    [scores_full[j] for j in g["idx"]])

        if not parts:
            # The plan keeps nothing from this generation batch (all candidates are all-correct
            # without cost variation / all-wrong, or the frozen mixed cap is already reached).
            # The batch MUST NOT be handed to the trainer unchanged: the trainer would run its
            # own `std > 0` filter on the raw candidates and accumulate groups that the frozen
            # plan did not select -- in particular uncached efficiency groups would advance the
            # mixed-only stopping counter (observed live at D1 step 67 before this fix).
            # Instead, keep the planner state as advanced, leave the tensors untouched, and mark
            # the call so that `__call__` neutralises the published filter metric, which makes
            # the trainer drop every group of this batch (std == 0).
            self._keep_nothing = True
            self._no_keep_calls += 1
            self._record_no_keep(plan, cur_groups, scores_full, len(data), gstep, target)
            return

        new = parts[0] if len(parts) == 1 else DataProto.concat(parts)

        # ---- record + mutate ----
        n_before = len(data)
        data.batch = new.batch
        data.non_tensor_batch = new.non_tensor_batch

        self._record(plan, cur_groups, kept_meta, scores_full, n_before, len(data), gstep, target)

    def _record_no_keep(self, plan, cur_groups, scores_full, n_candidates, gstep, target):
        """Record a generation batch from which the frozen plan selected nothing."""
        call_index = self._next_call_index()
        rec = {
            "global_step": gstep, "call_index": call_index,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "quota_target": self.quota, "batch_target": target,
            "generation_batch_in_step": self._q["batches"],
            "candidate_groups_seen": len(cur_groups), "candidate_rollouts": n_candidates,
            "mixed_seen_cum": self._q["mixed_seen"], "mixed_kept_cum": self._q["mixed_kept"],
            "efficiency_seen_cum": self._q["eff_seen"],
            "efficiency_cache_size": len(self._q["cache"]),
            "is_stopping": bool(plan["is_stopping"]),
            "kept_nothing": True, "quota_actual": 0, "quota_filled": False,
            "eff_shortfall": int(plan.get("eff_shortfall", 0)),
            "mixed_dropped_this_batch": int(len(plan["dropped_mixed"])),
            "inserted_efficiency_group_ids": [], "displaced_mixed_group_ids": [],
            "mixed_selected_group_ids": [],
            "optimizer_batch_groups": 0, "needs_injection": False,
            "advantage_mean": None, "advantage_std": None, "advantage_min": None,
            "advantage_max": None, "advantage_nan": 0, "advantage_inf": 0,
            "reward_mean": None, "reward_min": None, "reward_max": None, "em_mean": None,
            "note": ("empty keep set: the published filter metric is neutralised so that the "
                     "trainer drops this batch entirely and the mixed-only stopping counter is "
                     "not advanced by uncached efficiency groups"),
        }
        try:
            with open(self.calls_path, "a") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
        except Exception:  # noqa: BLE001
            pass

    def _record(self, plan, cur_groups, kept_meta, scores_full, n_before, n_after, gstep, target):
        call_index = self._next_call_index()
        ts = datetime.now().isoformat(timespec="seconds")
        # per-rollout rewards of the materialised batch, for the advantage diagnostic
        by_gid = {}
        for g in cur_groups:
            gid = g["gid"]
            by_gid[gid] = [scores_full[j] for j in g["idx"]]
        adv_scores, adv_uids, rows = [], [], []
        _all_vals = []
        for g, role in kept_meta:
            gid = g["gid"]
            vals = by_gid.get(gid)
            if vals is None:                      # injected from an earlier generation batch
                cached = self._cache_slices.get(gid)
                vals = cached[1] if isinstance(cached, tuple) else None
            if vals is None:
                continue
            # `vals` is the per-rollout score list of this group (from the current batch, or
            # from the cache for an injected group whose `idx` is empty); iterate it directly
            # so injected groups also contribute to the advantage / reward diagnostics.
            for s in vals:
                adv_scores.append(float(s["score"]))
                adv_uids.append(gid)
            _all_vals.extend(vals)
            rows.append({
                "global_step": gstep, "call_index": call_index, "group_id": gid, "uid": g["uid"],
                "group_type": role, "k": g["k"], "cost_gap": g["cost_gap"], "source": g["source"],
                "reward_mean": float(np.mean([s["score"] for s in vals])),
                "reward_min": float(np.min([s["score"] for s in vals])),
                "reward_max": float(np.max([s["score"] for s in vals])),
                "em_mean": float(np.mean([s["em"] for s in vals])),
                "efficiency_mean": float(np.mean([s["efficiency"] for s in vals])),
                "cost_mean": float(np.mean([s["cost"] for s in vals if s["cost"] is not None]))
                if any(s["cost"] is not None for s in vals) else None,
                "queries_mean": float(np.mean([s["search_queries"] for s in vals
                                               if s["search_queries"] is not None]))
                if any(s["search_queries"] is not None for s in vals) else None,
                "turns_mean": float(np.mean([s["conversation_turns"] for s in vals
                                             if s["conversation_turns"] is not None]))
                if any(s["conversation_turns"] is not None for s in vals) else None,
            })
        with open(self.groups_path, "a") as fh:
            for r in rows:
                fh.write(json.dumps(r, default=str) + "\n")

        # advantage (exact GRPO replica on the materialised batch)
        adv = np.asarray([])
        if adv_scores:
            adv = _grpo_advantage(np.asarray(adv_scores, dtype=float), adv_uids)
        rec = {
            "global_step": gstep, "call_index": call_index, "timestamp": ts,
            "quota_target": self.quota, "batch_target": target,
            "generation_batch_in_step": self._q["batches"],
            "candidate_groups_seen": len(cur_groups),
            "candidate_rollouts": n_before,
            "mixed_seen_cum": self._q["mixed_seen"], "mixed_kept_cum": self._q["mixed_kept"],
            "efficiency_seen_cum": self._q["eff_seen"], "efficiency_cache_size": len(self._q["cache"]),
            "is_stopping": bool(plan["is_stopping"]),
            "quota_actual": int(plan["m"]), "quota_filled": bool(plan["m"] == self.quota),
            "eff_shortfall": int(plan.get("eff_shortfall", 0)),
            "mixed_dropped_this_batch": len(plan["dropped_mixed"]),
            "inserted_efficiency_group_ids": [g["gid"] for g, role in kept_meta
                                              if role == "efficiency"],
            "displaced_mixed_group_ids": list(plan["dropped_mixed"]),
            "mixed_selected_group_ids": [g["gid"] for g, role in kept_meta if role == "mixed"],
            "optimizer_batch_groups": n_after,
            "needs_injection": bool(plan["needs_injection"]),
            "advantage_mean": float(np.mean(adv)) if adv.size else None,
            "advantage_std": float(np.std(adv, ddof=1)) if adv.size > 1 else None,
            "advantage_min": float(np.min(adv)) if adv.size else None,
            "advantage_max": float(np.max(adv)) if adv.size else None,
            "advantage_nan": int(np.isnan(adv).sum()) if adv.size else 0,
            "advantage_inf": int(np.isinf(adv).sum()) if adv.size else 0,
            "reward_mean": float(np.mean(adv_scores)) if adv_scores else None,
            "reward_min": float(np.min(adv_scores)) if adv_scores else None,
            "reward_max": float(np.max(adv_scores)) if adv_scores else None,
            "em_mean": float(np.mean([s["em"] for s in _all_vals])) if _all_vals else None,
        }
        with open(self.calls_path, "a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
        self._step_rows.append(rec)

        if plan["is_stopping"]:
            step_rec = dict(rec)
            # `optimizer_batch_groups` (from rec) counts the rollouts materialised by THIS
            # generation batch; the OPTIMIZER batch is the accumulation over the step, i.e.
            # the mixed groups kept so far plus the injected efficiency groups (n = 8 each;
            # the trainer truncates to 32 groups = 256 rollouts).
            step_rec["optimizer_batch_groups_step"] = int(self._q["mixed_kept"]) + int(plan["m"])
            step_rec["stopping_call_rollouts"] = n_after
            step_rec["generation_batches_consumed"] = self._q["batches"]
            step_rec["candidates_seen_total"] = self._q["candidates_seen"]
            step_rec["eff_shortfall_total"] = self._q["eff_shortfall"]
            step_rec["mixed_dropped_total"] = self._q["mixed_dropped"]
            step_rec["inserted_total"] = sum(r["quota_actual"] for r in self._step_rows)
            with open(self.steps_path, "a") as fh:
                fh.write(json.dumps(step_rec, default=str) + "\n")
            e1e_quota_plan.reset_state(self._q)
            self._cache_slices = OrderedDict()
            self._step_rows = []

    # ------------------------------------------------------------------ call
    def __call__(self, data, return_dict=False):
        if self.is_train:
            self._keep_nothing = False
            try:
                self._apply_quota_inplace(data)
            except Exception as e:  # noqa: BLE001 -- never break training, but record loudly
                self.quota_errors += 1
                try:
                    with open(self.errors_path, "a") as fh:
                        fh.write(f"{datetime.now().isoformat()} QUOTA ERROR (data left unchanged): "
                                 f"{type(e).__name__}: {e}\n")
                except Exception:  # noqa: BLE001
                    pass

        result = super().__call__(data, return_dict=True)

        if self.is_train and self._keep_nothing:
            # The plan selected no group from this generation batch.  Replace the published
            # filter metric with a constant so that the trainer's `std > 0` filter drops every
            # group of this batch and the accumulation is left untouched (see the empty-keep
            # note in `_apply_quota_inplace`).  The reward tensor itself is discarded with the
            # batch, so the value written here does not matter.
            try:
                info = result.get("reward_extra_info")
                if info is not None and "e1e_quota_metric" in info:
                    info["e1e_quota_metric"] = [0.0] * len(data)
            except Exception as e:  # noqa: BLE001 -- never break training
                try:
                    with open(self.errors_path, "a") as fh:
                        fh.write(f"NO-KEEP METRIC NEUTRALISE FAILED: {type(e).__name__}: {e}\n")
                except Exception:  # noqa: BLE001
                    pass

        if self.is_train:
            try:
                info = result.get("reward_extra_info", {}) or {}
                em = info.get("em")
                if em is not None:
                    data.batch["acc"] = torch.tensor(
                        [float(x) for x in em], dtype=torch.float32,
                        device=data.batch["prompts"].device)
            except Exception:  # noqa: BLE001
                pass
            try:
                self._rollout_diagnostics(data, result)
            except Exception as e:  # noqa: BLE001
                try:
                    with open(self.errors_path, "a") as fh:
                        fh.write(f"DIAG ERROR (swallowed): {type(e).__name__}: {e}\n")
                except Exception:  # noqa: BLE001
                    pass

        if return_dict:
            return result
        return result["reward_tensor"]

    def _rollout_diagnostics(self, data, result):
        if not self.run_dir:
            return
        info = result.get("reward_extra_info", {}) or {}
        rt = result.get("reward_tensor")
        n = len(data)
        gstep = e1e_state.current_global_step()
        ts = datetime.now().isoformat(timespec="seconds")
        uids = data.non_tensor_batch.get("uid", [None] * n)
        cost_info = e1e_cost.costs_from_batch(data.non_tensor_batch, n)
        prompt_ids = data.batch["prompts"]
        prompt_len = prompt_ids.shape[-1]
        vlen = data.batch["attention_mask"][:, prompt_len:].sum(dim=-1)
        recs = []
        for i in range(n):
            rts = None
            if rt is not None:
                try:
                    rts = float(rt[i].sum().item())
                except Exception:  # noqa: BLE001
                    pass
            recs.append({
                "run_id": os.path.basename(self.run_dir), "pid": self.pid,
                "global_step": gstep, "timestamp": ts, "uid": str(uids[i]),
                "em": (info.get("em") or [None] * n)[i],
                "reward": (info.get("score") or [None] * n)[i],
                "reward_tensor_sum": rts,
                "efficiency": (info.get("efficiency") or [None] * n)[i],
                "group_type": ("efficiency" if (info.get("group_efficiency") or [False] * n)[i]
                               else ("mixed" if (info.get("group_mixed") or [False] * n)[i] else "excluded")),
                "group_k": (info.get("group_k") or [None] * n)[i],
                "quota_metric": (info.get("e1e_quota_metric") or [None] * n)[i],
                "logical_search_batches": cost_info["costs"][i],
                "search_queries": cost_info["queries"][i],
                "parallel_factor": ((cost_info["queries"][i] / cost_info["costs"][i])
                                    if (cost_info["costs"][i] and cost_info["queries"][i]) else 0.0),
                "conversation_turns": cost_info["conversation_turns"][i],
                "response_token_length": int(vlen[i].item()),
                "zero_search": bool(cost_info["costs"][i] == 0),
                "metadata_valid": bool(cost_info["metadata_valid"][i]),
            })
        with open(self.rollouts_path, "a") as fh:
            for r in recs:
                fh.write(json.dumps(r, default=str) + "\n")


def _grpo_advantage(scores, uids, epsilon=1e-6):
    """Exact replica of core_algos.compute_grpo_outcome_advantage (std with ddof=1)."""
    scores = np.asarray(scores, dtype=float)
    by = defaultdict(list)
    for s, u in zip(scores, uids):
        by[u].append(s)
    adv = np.zeros_like(scores)
    for u, vals in by.items():
        v = np.asarray(vals, dtype=float)
        idx = [i for i, uu in enumerate(uids) if uu == u]
        if len(v) == 1:
            mean, std = 0.0, 1.0
        else:
            mean, std = float(np.mean(v)), float(np.std(v, ddof=1))
        for i in idx:
            adv[i] = (scores[i] - mean) / (std + epsilon)
    return adv

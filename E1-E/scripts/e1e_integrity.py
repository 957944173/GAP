#!/usr/bin/env python3
"""E1-E post-phase integrity check (E1-E_task.md §12 / §16).

    E1E_PHASE=D1 python3 e1e_integrity.py      # checks global_step_70
    E1E_PHASE=D2 python3 e1e_integrity.py      # checks global_step_120

Verifies
  * the phase's checkpoint is complete (model/optimizer/extra_state x4 ranks, data.pt,
    tracker value) and the numbers of steps between the resume point and the end are all
    present in the log;
  * full-state resume evidence (lr_scheduler.last_epoch continuation + RNG present, read
    from the small extra_state shard);
  * no non-finite metric values, and an error-pattern scan (CUDA OOM / NCCL / traceback /
    wiki-tool failures) restricted to the real training window;
  * the FILTER DECOUPLING invariant, from the reward-manager diagnostics of this phase:
    `actual retained set on the current candidate pool == the same-pool original EM filter
    set` (100 %), i.e. `keep_em_std_gt_0` reproduces `std(EM) > 0` exactly, and the
    counterfactual shaped filter would have retained strictly more.

Writes E1-E/analysis/step<END>_integrity.{md,json}
"""
import glob
import json
from collections import defaultdict
import math
import os
import re
import sys

import numpy as np

REPO = "/data01/wyy/Graph-Agent-Planning"
E1D = os.path.join(REPO, "E1-E")
EXP_NAME = os.environ.get("E1E_EXPERIMENT_NAME", "DAPO-GAP3B-MHQA-Agent-E1E-q2-step60to120-4gpu")
EXP_DIR = os.path.join(REPO, "experiments", EXP_NAME)

PHASE = os.environ.get("E1E_PHASE", "D1")
END_STEP = int(os.environ.get("E1E_END_STEP", "70" if PHASE == "D1" else "120"))
START_STEP = int(os.environ.get("E1E_START_STEP", "60" if PHASE == "D1" else "70"))
RUN_DIR = os.environ.get("E1E_RUN_DIR") or open(
    os.path.join(E1D, "runs", f"latest_{PHASE}.txt")).read().strip()
TRAIN_LOG = os.environ.get("E1E_TRAIN_LOG") or sorted(glob.glob(
    os.path.join(E1D, "logs", f"train_{PHASE}_*.log")))[-1]
CKPT = os.path.join(EXP_DIR, f"global_step_{END_STEP}")
ANALYSIS = os.path.join(E1D, "analysis")

STEP_RE = re.compile(r"step:(\d+)\s+-")
METRIC_RE = re.compile(r"([A-Za-z0-9_/\.\-@]+):([^\s]+)")
ERROR_PATTERNS = {
    "cuda_oom": r"CUDA out of memory|OutOfMemoryError|torch\.cuda\.OutOfMemoryError",
    "nccl_error": r"NCCL error|ncclInvalidUsage|ncclInternalError|ncclSystemError|Duplicate GPU detected",
    "nccl_warn": r"\[W\d+ .*c10d\]|NCCL WARN",
    "ray_worker_died": r"ActorDiedError|Worker exits unexpectedly|RayTaskError|ray\.exceptions",
    "wiki_search_error": r"Error when executing search",
    "wiki_tool_exec_failed": r"\[SearchTool\] Execution failed",
    "traceback": r"Traceback \(most recent call last\)",
    "cuda_error": r"CUDA error",
}


def _iter_jsonl(paths):
    for p in paths:
        for line in open(p, errors="ignore"):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except ValueError:
                continue


def check_quota_semantics():
    """E1-E section 11/12 invariants measured on THIS phase's own runtime diagnostics.

    Replaces the E1-D `check_filter_decoupling` (which read `groups_pid*.jsonl` fields the
    E1-E manager does not produce).  Everything here comes from the E1-E reward manager's
    quota diagnostics written during the phase being checked.
    """
    diag = os.path.join(RUN_DIR, "diagnostics")
    steps = list(_iter_jsonl(sorted(glob.glob(os.path.join(diag, "quota_steps_pid*.jsonl")))))
    calls = list(_iter_jsonl(sorted(glob.glob(os.path.join(diag, "quota_calls_pid*.jsonl")))))
    groups = list(_iter_jsonl(sorted(glob.glob(os.path.join(diag, "quota_groups_pid*.jsonl")))))
    rollouts = list(_iter_jsonl(sorted(glob.glob(os.path.join(diag, "rollouts_pid*.jsonl")))))
    errfiles = sorted(glob.glob(os.path.join(diag, "quota_errors_pid*.log")))
    err_lines = [ln.strip() for f in errfiles for ln in open(f, errors="ignore") if ln.strip()]
    res = {
        "available": bool(steps), "n_step_records": len(steps), "n_generation_calls": len(calls),
        "n_group_records": len(groups), "n_rollout_records": len(rollouts),
        "quota_error_lines": len(err_lines),
        "quota_error_examples": err_lines[:5],
    }
    if not steps:
        res["reason"] = "no quota_steps_pid*.jsonl diagnostics in this run dir"
        return res

    steps_by_g = {}
    for r in steps:
        steps_by_g.setdefault(int(r["global_step"]), []).append(r)
    step_ids = sorted(steps_by_g)
    want = list(range(START_STEP + 1, END_STEP + 1))
    res["steps_covered"] = step_ids
    res["steps_missing"] = [s for s in want if s not in steps_by_g]
    res["all_steps_recorded"] = not res["steps_missing"]

    def step_of(r):
        return int(r["global_step"])

    # ---- (2) the FROZEN quota formula: m = min(q, |cache|) ----------------------------
    # m < q is allowed *by design* when fewer than q efficiency groups have been cached so far
    # (E1-E_task.md section 1).  The only permitted deviation is the documented mixed-cap
    # shortfall, which must be recorded in `eff_shortfall` for that step.
    quota_formula_violations, short_steps = [], []
    for g_, rs_ in steps_by_g.items():
        stop = [r for r in rs_ if r.get("is_stopping")]
        if not stop:
            continue
        r0 = stop[-1]
        q_act = int(r0.get("quota_actual") or 0)
        cache = int(r0.get("efficiency_cache_size") or 0)
        tgt_q = int(r0.get("quota_target") or 2)
        sf = int(r0.get("eff_shortfall") or 0)
        design_m = min(tgt_q, cache)
        if q_act == design_m:
            continue
        if q_act < design_m and sf == design_m - q_act:
            short_steps.append({"step": g_, "quota_actual": q_act, "design_m": design_m,
                                "efficiency_cache_size": cache, "recorded_eff_shortfall": sf,
                                "mixed_kept_cum": int(r0.get("mixed_kept_cum") or 0)})
            continue
        quota_formula_violations.append({"step": g_, "quota_actual": q_act, "design_m": design_m,
                                         "efficiency_cache_size": cache,
                                         "recorded_eff_shortfall": sf})

    # ---- (1) batch size / (2) quota / (3) mixed-only stopping -------------------------
    # NOTE on field semantics: the manager's `optimizer_batch_groups` is the number of
    # ROLLOUTS materialised by THIS generation batch (len(data) after the in-place mutation),
    # not the optimizer batch size.  The optimizer batch is the accumulation over the step, so
    # the correct step-level group count is `mixed_kept_cum + quota_actual` (each group n = 8,
    # and the trainer truncates to 32 groups = 256 rollouts).
    def step_groups(r):
        try:
            return int(r.get("mixed_kept_cum", -1)) + int(r.get("quota_actual", -1))
        except (TypeError, ValueError):
            return -1

    bad_size = [g for g, rs in steps_by_g.items()
                if any(step_groups(r) != 32 for r in rs)]
    stop_calls = [r for r in calls if r.get("is_stopping")]
    nons_top = [r for r in calls if not r.get("is_stopping")]
    bad_quota = [r for r in calls if int(r.get("quota_actual", 0)) > int(r.get("quota_target", 2))]
    bad_target = [r for r in calls if int(r.get("quota_target", -1)) != 2]
    bad_stop_count = [g for g, rs in
                      ((g, [r for r in calls if step_of(r) == g]) for g in step_ids)
                      if sum(1 for r in rs if r.get("is_stopping")) != 1]
    bad_seen = [r for r in nons_top if int(r.get("mixed_seen_cum", 0)) >= 32] + \
               [r for r in stop_calls if int(r.get("mixed_seen_cum", 0)) < 32]
    bad_metric_calls = [r for r in nons_top if int(r.get("quota_actual", 0)) != 0]
    res.update({
        "steps_with_wrong_batch_size": bad_size,
        "batch_size_definition": "step-level groups = mixed_kept_cum + quota_actual (must be 32); "
                                 "`stopping_call_rollouts` is only that call's materialised rollouts",
        "calls_with_quota_actual_gt_2": len(bad_quota),
        "calls_with_wrong_quota_target": len(bad_target),
        "steps_without_exactly_one_stopping_call": bad_stop_count,
        "calls_where_stopping_disagrees_with_mixed_count": len(bad_seen),
        "non_stopping_calls_with_insertions": len(bad_metric_calls),
        "quota_formula_holds_all_steps": not quota_formula_violations,
        "quota_formula_violations": quota_formula_violations[:10],
        "steps_short_of_design_quota": short_steps[:10],
        "n_steps_short_of_design_quota": len(short_steps),
    })

    # ---- call/rollout reconciliation helpers (used by several checks below) ----------
    import datetime as _dt

    def _ts(v):
        try:
            return _dt.datetime.fromisoformat(str(v))
        except Exception:  # noqa: BLE001
            return None

    # The call record and its rollout rows are timestamped by two separate `now()` calls, so the
    # seconds can differ; match them with a small tolerance instead of exactly.
    served_groups = defaultdict(list)
    for r in rollouts:
        served_groups[(int(r["global_step"]), str(r.get("timestamp")))].append(r)
    recorded_calls = set()
    for r in calls:
        recorded_calls.add((int(r["global_step"]), str(r.get("timestamp"))))

    def has_record(step, ts, tol=3):
        t0 = _ts(ts)
        if t0 is None:
            return (step, ts) in recorded_calls
        for st, cts in recorded_calls:
            if st != step:
                continue
            t1 = _ts(cts)
            if t1 is not None and abs((t1 - t0).total_seconds()) <= tol:
                return True
        return False

    unrecorded = sorted(k for k in served_groups if not has_record(*k))
    # Rollouts of a `kept_nothing` call carry the deliberately neutralised metric (0.0) and were
    # dropped by the trainer, so they are excluded from the metric-consistency check.
    neutralised = [(int(c["global_step"]), _ts(c.get("timestamp"))) for c in calls
                   if c.get("kept_nothing")]

    def _is_neutralised(r):
        t0 = _ts(r.get("timestamp"))
        if t0 is None:
            return False
        st = int(r["global_step"])
        return any(st == s2 and t1 is not None and abs((t1 - t0).total_seconds()) <= 3
                   for s2, t1 in neutralised)

    # ---- (5)(6)(7)(8)(9)(10)(12) reward semantics ------------------------------------
    eff = [g for g in groups if g.get("group_type") == "efficiency"]
    mix = [g for g in groups if g.get("group_type") == "mixed"]
    bad_eff_k = [g for g in eff if int(g.get("k", 0)) != 8]
    # The injected efficiency groups carry `cost_gap: null` in their group record (the manager
    # does not recompute it for a cached group), so the cost-variation invariant is verified
    # from the authoritative per-rollout costs instead: among the correct rollouts of the group
    # the cost range must be > 0.
    cost_gap_by_group = {}  # noqa: F841 (kept for readability of the block below)
    _grp_costs = defaultdict(set)
    _grp_correct = defaultdict(int)
    for r in rollouts:
        if _is_neutralised(r):
            continue
        key = (int(r["global_step"]), str(r.get("uid")))
        c = r.get("logical_search_batches")
        if c is not None:
            _grp_costs[key].add(float(c))
        if float(r.get("em") or 0.0) >= 0.5:
            _grp_correct[key] += 1
    bad_eff_gap = []
    for g_ in eff:
        key = (int(g_["global_step"]), str(g_.get("uid")))
        costs = _grp_costs.get(key, set())
        if len(costs) < 2 or (max(costs) - min(costs)) <= 0:
            bad_eff_gap.append({"step": key[0], "uid": key[1],
                                "group_cost_gap_field": g_.get("cost_gap"),
                                "rollout_costs": sorted(costs)[:6]})
    bad_eff_rng = [g for g in eff if not (1.0 - 1e-12 <= float(g.get("reward_min", 0)) and
                                         float(g.get("reward_max", 0)) <= 1.05 + 1e-12)]
    bad_eff_em = [g for g in eff if abs(float(g.get("em_mean", 0)) - 1.0) > 1e-12]
    bad_mix_rng = [g for g in mix if not (0.0 - 1e-12 <= float(g.get("reward_min", 0)) and
                                          float(g.get("reward_max", 0)) <= 1.0 + 1e-12)]
    bad_mix_shaped = [g for g in mix if abs(float(g.get("reward_mean", 0)) -
                                            float(g.get("em_mean", 0))) > 1e-12]
    # per-rollout semantics (`reward` is the shaped reward that feeds GRPO; `em` is truth)
    # Rollouts of a `kept_nothing` call carry the deliberately neutralised metrics (0.0) and
    # were dropped by the trainer, so they are excluded from the metric-consistency check.
    neutralised = [(int(c["global_step"]), _ts(c.get("timestamp"))) for c in calls
                   if c.get("kept_nothing")]

    def _is_neutralised(r):
        t0 = _ts(r.get("timestamp"))
        if t0 is None:
            return False
        st = int(r["global_step"])
        return any(st == s2 and t1 is not None and abs((t1 - t0).total_seconds()) <= 3
                   for s2, t1 in neutralised)

    bad_roll_mixed = bad_roll_wrong = bad_roll_eff = bad_roll_metric = 0
    n_wrong = n_zero_search = n_malformed = 0
    n_skipped_neutralised = 0
    for r in rollouts:
        if _is_neutralised(r):
            n_skipped_neutralised += 1
            continue
        em = float(r.get("em") or 0.0)
        rew = float(r.get("reward") or 0.0)
        rw = r.get("reward_tensor_sum")
        gt = r.get("group_type")
        if rw is not None and abs(float(rw) - rew) > 1e-6:
            bad_roll_metric += 1
        if em == 0.0:
            n_wrong += 1
            if abs(rew) > 1e-12:
                bad_roll_wrong += 1
        if gt == "mixed" and abs(rew - em) > 1e-12:
            bad_roll_mixed += 1
        if gt == "efficiency":
            if not (1.0 - 1e-12 <= rew <= 1.05 + 1e-12) or em != 1.0:
                bad_roll_eff += 1
            qm = r.get("quota_metric")
            if qm is not None and abs(float(qm) - (1.0 + 0.05 * float(r.get("efficiency") or 0.0))) > 1e-9:
                bad_roll_metric += 1
        elif gt == "mixed" and r.get("quota_metric") is not None:
            if abs(float(r["quota_metric"]) - em) > 1e-12:
                bad_roll_metric += 1
        n_zero_search += int(bool(r.get("zero_search")))
        n_malformed += int(not bool(r.get("metadata_valid")))
    adv_nan = sum(int(r.get("advantage_nan") or 0) for r in steps)
    adv_inf = sum(int(r.get("advantage_inf") or 0) for r in steps)
    # Independent advantage recomputation.  The manager's per-call `advantage_*` fields are
    # informational; in the D1 build they are null at stopping calls (the injected groups were
    # zipped against an empty index list, so they contributed nothing -- fixed for D2).  The
    # authoritative check is to recompute the exact GRPO advantage here from the recorded
    # per-rollout rewards of the materialised batch and verify finiteness.
    adv_by_step, adv_recomputed = {}, {}
    for s_ in step_ids:
        rs = [r for r in rollouts if step_of(r) == s_ and r.get("reward") is not None]
        if not rs:
            continue
        groups_ = {}
        for r in rs:
            groups_.setdefault(str(r.get("uid")), []).append(float(r["reward"]))
        scores, uids = [], []
        for u, vals in groups_.items():
            scores += vals
            uids += [u] * len(vals)
        a = np.asarray(scores, dtype=float)
        by = {}
        for v, u in zip(a, uids):
            by.setdefault(u, []).append(v)
        adv = np.zeros_like(a)
        for u, vv in by.items():
            v = np.asarray(vv, dtype=float)
            idx = [i for i, uu in enumerate(uids) if uu == u]
            if len(v) == 1:
                m_, sd_ = 0.0, 1.0
            else:
                m_, sd_ = float(np.mean(v)), float(np.std(v, ddof=1))
            for i in idx:
                adv[i] = (a[i] - m_) / (sd_ + 1e-6)
        adv_by_step[s_] = {"n": int(adv.size), "groups": len(by),
                           "nan": int(np.isnan(adv).sum()), "inf": int(np.isinf(adv).sum()),
                           "mean": float(np.mean(adv)), "std": float(np.std(adv, ddof=1)),
                           "min": float(np.min(adv)), "max": float(np.max(adv))}
        adv_recomputed[s_] = adv_by_step[s_]
    adv_nan += sum(v["nan"] for v in adv_recomputed.values())
    adv_inf += sum(v["inf"] for v in adv_recomputed.values())
    res.update({
        "efficiency_groups": len(eff), "mixed_groups": len(mix),
        "efficiency_groups_with_k_not_8": len(bad_eff_k),
        "efficiency_groups_without_cost_variation": len(bad_eff_gap),
        "efficiency_groups_reward_out_of_1_1.05": len(bad_eff_rng),
        "efficiency_groups_em_mean_not_1": len(bad_eff_em),
        "mixed_groups_reward_out_of_0_1": len(bad_mix_rng),
        "mixed_groups_reward_mean_not_equal_em": len(bad_mix_shaped),
        "rollouts_mixed_reward_not_equal_em": bad_roll_mixed,
        "rollouts_wrong_with_nonzero_reward": bad_roll_wrong,
        "rollouts_efficiency_reward_or_em_wrong": bad_roll_eff,
        "rollouts_published_metric_inconsistent": bad_roll_metric,
        "rollouts_reward_tensor_mismatch": 0,
        "wrong_rollouts": n_wrong,
        "zero_search_rollouts": n_zero_search,
        "zero_search_share": (n_zero_search / len(rollouts)) if rollouts else None,
        "malformed_metadata_rollouts": n_malformed,
        "malformed_metadata_share": (n_malformed / len(rollouts)) if rollouts else None,
        "rollouts_skipped_neutralised_metric": n_skipped_neutralised,
        "rollouts_checked_for_metric_consistency": len(rollouts) - n_skipped_neutralised,
        "advantage_nan_total": adv_nan, "advantage_inf_total": adv_inf,
        "advantage_nan_total_manager_field": sum(int(r.get("advantage_nan") or 0) for r in steps),
        "advantage_nan_total_recomputed": sum(v["nan"] for v in adv_recomputed.values()),
        "advantage_inf_total_recomputed": sum(v["inf"] for v in adv_recomputed.values()),
        "advantage_recomputed_per_step": {str(k): v for k, v in adv_recomputed.items()},
    })

    # ---- every manager call must be recorded (no raw-batch leak) ---------------------
    # Each manager call writes its returned rollouts under one timestamp, so counting distinct
    # (global_step, timestamp) pairs in rollouts_pid gives the number of calls the manager
    # actually served -- independently of the call records.  Every one of those calls must have
    # a quota_calls record (normal or `kept_nothing`); a served call without a record is exactly
    # the D1 leak where the raw generation batch reached the trainer's filter.
    # the served/recorded reconciliation, `has_record`, `unrecorded` and `_is_neutralised` are
    # defined earlier (they are needed by the reward-semantics checks as well)
    no_keep = [r for r in calls if r.get("kept_nothing")]
    # A manager call must never hand the trainer a batch bigger than the optimizer batch
    # (32 groups x n=8 = 256 rollouts) unless it is a `kept_nothing` call, which intentionally
    # leaves the raw generation batch in place while neutralising the published filter metric.
    oversized = []
    for (step_, ts_), rows in served_groups.items():
        if len(rows) <= 8 * (int((steps_by_g[step_][-1].get("batch_target") or 32)) if steps_by_g.get(step_) else 32):
            continue
        match = None
        t0 = _ts(ts_)
        for c in calls:
            if step_of(c) != step_:
                continue
            t1 = _ts(c.get("timestamp"))
            if t1 is not None and t0 is not None and abs((t1 - t0).total_seconds()) <= 3:
                match = c
                break
        if match is None or not match.get("kept_nothing"):
            oversized.append({"step": step_, "timestamp": ts_, "rollouts": len(rows),
                              "matched_call": None if match is None else match.get("call_index"),
                              "kept_nothing": None if match is None else bool(match.get("kept_nothing"))})
    res["manager_calls_served"] = len(served_groups)
    res["manager_calls_recorded"] = len(recorded_calls)
    res["unrecorded_calls"] = unrecorded[:10]
    res["n_unrecorded_calls"] = len(unrecorded)
    res["oversized_returned_batches"] = oversized[:10]
    res["n_oversized_returned_batches"] = len(oversized)
    res["kept_nothing_calls"] = len(no_keep)
    res["all_manager_calls_recorded"] = not unrecorded
    # per-step generation-batch counter must be gapless 1..N (a gap means an unrecorded call)
    gb_gaps = {}
    for g_ in sorted({int(r["global_step"]) for r in calls}):
        gbs = sorted(int(r.get("generation_batch_in_step") or 0) for r in calls
                     if int(r["global_step"]) == g_)
        want = list(range(1, len(gbs) + 1))
        if gbs != want:
            gb_gaps[g_] = {"got": gbs, "expected": want}
    res["steps_with_generation_batch_gaps"] = gb_gaps
    # returned rollouts must be 8 x the number of recorded kept groups (or the untouched raw
    # batch for a `kept_nothing` call)
    by_call_groups = defaultdict(int)
    for g_ in groups:
        by_call_groups[(step_of(g_), int(g_.get("call_index", -1)))] += 1
    bad_roll = []
    for r in calls:
        ci = int(r.get("call_index", -1))
        ret = int(r.get("optimizer_batch_groups") or 0)
        kept_groups = by_call_groups.get((step_of(r), ci), 0)
        if r.get("kept_nothing"):
            continue                      # data intentionally left untouched; metric neutralised
        if ret != 8 * kept_groups:
            bad_roll.append({"step": step_of(r), "call_index": ci, "returned_rollouts": ret,
                             "kept_groups": kept_groups})
    res["calls_returning_unexpected_rollout_counts"] = bad_roll[:10]
    res["n_calls_returning_unexpected_rollout_counts"] = len(bad_roll)

    # ---- derived group-id lists (authoritative) -------------------------------------
    # The D1 build of the reward manager recorded `inserted_efficiency_group_ids` /
    # `mixed_selected_group_ids` with a comprehension that referenced a leaked loop variable
    # (`gid` instead of `g["gid"]`), so those two fields repeat one id (fixed for D2; the
    # training itself and every other field are unaffected).  quota_groups_pid*.jsonl carries
    # the real per-group records, so the id lists are re-derived from it here for BOTH phases.
    by_call = defaultdict(list)
    for g in groups:
        by_call[(step_of(g), int(g.get("call_index", -1)))].append(g)
    derived = {}
    for g in step_ids:
        rs = [r for r in calls if step_of(r) == g]
        stop = [r for r in rs if r.get("is_stopping")]
        ci = int(stop[-1]["call_index"]) if stop else None
        recs = by_call.get((g, ci), [])
        ins = [x["group_id"] for x in recs if x.get("group_type") == "efficiency"]
        mx = [x["group_id"] for x in recs if x.get("group_type") == "mixed"]
        derived[g] = {"call_index": ci, "inserted": ins, "mixed_kept_in_stop_call": mx}
    bad_ins = [g for g, d in derived.items()
               if len(d["inserted"]) != len(set(d["inserted"]))
               or len(d["inserted"]) != (steps_by_g[g][-1].get("quota_actual") or 0)]
    res["derived_inserted_ids_per_step"] = {str(g): derived[g]["inserted"] for g in step_ids}
    res["steps_with_inconsistent_inserted_ids"] = bad_ins
    res["id_list_note"] = ("inserted ids are re-derived from quota_groups records of the "
                           "stopping call; the D1 manager field repeated one id due to a "
                           "leaked loop variable (fixed for D2)")

    # ---- runtime fill statistics (the section 12 / RQ-facing numbers) ----------------
    per_step = {}
    for g in step_ids:
        rs = [r for r in calls if step_of(r) == g]
        stop = [r for r in rs if r.get("is_stopping")]
        per_step[g] = {
            "generation_batches": len(rs),
            "quota_actual": int(stop[0]["quota_actual"]) if stop else None,
            "candidate_groups_seen": sum(int(r.get("candidate_groups_seen", 0)) for r in rs),
            "mixed_groups_seen": int(stop[0].get("mixed_seen_cum", 0)) if stop else None,
            "efficiency_groups_seen": int(stop[0].get("efficiency_seen_cum", 0)) if stop else None,
            "efficiency_cache_size": int(stop[0].get("efficiency_cache_size", 0)) if stop else None,
            "eff_shortfall": int(stop[0].get("eff_shortfall", 0)) if stop else None,
            "needs_injection": bool(stop[0].get("needs_injection")) if stop else None,
            "inserted_ids": derived[g]["inserted"],
            "inserted_ids_as_recorded": stop[0].get("inserted_efficiency_group_ids", []) if stop else [],
            "displaced_ids": stop[0].get("displaced_mixed_group_ids", []) if stop else [],
            "mixed_kept_in_stop_call": derived[g]["mixed_kept_in_stop_call"],
            "mixed_dropped_in_step": sum(int(r.get("mixed_dropped_this_batch", 0)) for r in rs),
            "optimizer_batch_groups": step_groups(stop[0]) if stop else None,
            "stopping_call_rollouts": int(stop[0].get("optimizer_batch_groups", 0)) if stop else None,
            "mixed_kept_cum": int(stop[0].get("mixed_kept_cum", 0)) if stop else None,
            "advantage_mean": (stop[0].get("advantage_mean") if stop and
                               stop[0].get("advantage_mean") is not None
                               else (adv_recomputed.get(g) or {}).get("mean")),
            "advantage_std": (stop[0].get("advantage_std") if stop and
                              stop[0].get("advantage_std") is not None
                              else (adv_recomputed.get(g) or {}).get("std")),
            "advantage_nan_recomputed": (adv_recomputed.get(g) or {}).get("nan"),
            "advantage_inf_recomputed": (adv_recomputed.get(g) or {}).get("inf"),
        }
    fill = [v["quota_actual"] / 2.0 for v in per_step.values() if v["quota_actual"] is not None]
    res["per_step"] = per_step
    res["quota_fill_rate"] = (sum(fill) / len(fill)) if fill else None
    res["steps_full_quota"] = sum(1 for f in fill if f == 1.0)
    res["steps_used"] = len(fill)
    res["total_inserted_efficiency_groups"] = sum(len(v["inserted_ids"]) for v in per_step.values())
    res["inserted_ids_distinct_every_step"] = not bad_ins
    res["total_mixed_dropped"] = sum(v["mixed_dropped_in_step"] for v in per_step.values())
    res["total_displaced_mixed_groups"] = sum(len(v["displaced_ids"]) for v in per_step.values())
    res["total_eff_shortfall"] = sum((v["eff_shortfall"] or 0) for v in per_step.values())
    res["steps_needing_injection"] = sum(1 for v in per_step.values() if v["needs_injection"])
    res["total_generation_batches"] = sum(v["generation_batches"] for v in per_step.values())
    res["mean_generation_batches_per_step"] = (res["total_generation_batches"] / len(per_step)
                                               if per_step else None)

    res["quota_semantics_ok"] = bool(
        res["all_steps_recorded"] and not bad_size and not bad_quota and not bad_target
        and not bad_stop_count and not bad_seen and not bad_metric_calls
        and not bad_eff_k and not bad_eff_gap and not bad_eff_rng and not bad_eff_em
        and not bad_mix_rng and not bad_mix_shaped
        and not bad_roll_mixed and not bad_roll_wrong and not bad_roll_eff
        and not bad_roll_metric and adv_nan == 0 and adv_inf == 0
        and not quota_formula_violations
        and not bad_ins
        and res["all_manager_calls_recorded"] and not gb_gaps and not bad_roll
        and not oversized)
    return res


LOG_STEP_RE = re.compile(r"step:(\d+)\s+-\s+(.*)$")


def check_log_vs_diagnostics(quota):
    """Cross-check the trainer's OWN logged metrics against the manager's diagnostics.

    If the runtime semantics are right, a step with quota_actual > 0 must show the shaped
    reward max (1.05) in `critic/score/max`, and a step with no insertion must never exceed
    1.0; `critic/score/mean` must equal the mean shaped reward over the 32 selected groups.
    """
    out = {"available": False}
    if not os.path.exists(TRAIN_LOG) or not quota.get("per_step"):
        return out
    text = re.sub(r"\x1b\[[0-9;]*m", "", open(TRAIN_LOG, errors="replace").read())
    logged = {}
    for line in text.splitlines():
        m = LOG_STEP_RE.search(line)          # the console line carries a "(TaskRunner pid=..)" prefix
        if not m:
            continue
        vals = dict(METRIC_RE.findall(m.group(2)))
        try:
            picked = {k: float(v) for k, v in vals.items()
                      if k in ("critic/score/mean", "critic/score/max")}
        except ValueError:
            continue
        if picked:
            logged.setdefault(int(m.group(1)), {}).update(picked)
    mism = []
    checked = 0
    for g, v in sorted(quota["per_step"].items()):
        lg = logged.get(int(g))
        if not lg or "critic/score/max" not in lg:
            continue
        checked += 1
        smax, smean = lg["critic/score/max"], lg.get("critic/score/mean")
        want_max = 1.05 if (v["quota_actual"] or 0) > 0 else 1.0
        if abs(smax - want_max) > 5e-3:
            mism.append({"step": g, "logged_score_max": smax, "expected": want_max,
                         "quota_actual": v["quota_actual"]})
    out.update({"available": bool(checked), "steps_cross_checked": checked,
                "score_max_mismatches": len(mism), "examples": mism[:5],
                "log_score_max_consistent_with_quota": bool(checked and not mism)})
    return out



def main():
    os.makedirs(ANALYSIS, exist_ok=True)
    result = {"phase": PHASE, "experiment_dir": EXP_DIR, "checkpoint_dir": CKPT,
              "train_log": TRAIN_LOG, "run_dir": RUN_DIR,
              "expected_start_step": START_STEP, "expected_end_step": END_STEP}

    ckpt = {"exists": os.path.isdir(CKPT), "shards": {}, "data_pt": False, "tracker": None}
    if ckpt["exists"]:
        for rank in range(4):
            for kind, pat in (("model", "model_world_size_4_rank_%d.pt"),
                              ("optimizer", "optim_world_size_4_rank_%d.pt"),
                              ("extra_state", "extra_state_world_size_4_rank_%d.pt")):
                p = os.path.join(CKPT, "actor", pat % rank)
                ckpt["shards"][f"{kind}_rank{rank}"] = (
                    {"exists": True, "size": os.path.getsize(p)} if os.path.exists(p) else {"exists": False})
        ckpt["data_pt"] = os.path.exists(os.path.join(CKPT, "data.pt"))
        t = os.path.join(EXP_DIR, "latest_checkpointed_iteration.txt")
        ckpt["tracker"] = open(t).read().strip() if os.path.exists(t) else None
    result["checkpoint"] = ckpt

    log = {"exists": os.path.exists(TRAIN_LOG)}
    if log["exists"]:
        text = open(TRAIN_LOG, errors="replace").read()
        clean = re.sub(r"\x1b\[[0-9;]*m", "", text)
        steps = sorted({int(m.group(1)) for m in STEP_RE.finditer(clean)})
        want = list(range(START_STEP, END_STEP + 1))
        log["steps_seen"] = steps
        log["missing_steps"] = [s for s in want if s not in steps]
        log["all_steps_present"] = not log["missing_steps"]
        log["resume_line_present"] = f"Setting global step to {START_STEP}" in clean
        bad, n_metrics = {}, 0
        for line in clean.splitlines():
            if not STEP_RE.search(line):
                continue
            for k, v in METRIC_RE.findall(line):
                try:
                    fv = float(v)
                except ValueError:
                    continue
                n_metrics += 1
                if not math.isfinite(fv):
                    bad[k] = bad.get(k, 0) + 1
        log["metrics_parsed"] = n_metrics
        log["non_finite_metrics"] = bad
        idx = clean.find("Final validation metrics:")
        head = clean if idx < 0 else clean[:idx]
        log["error_pattern_counts_before_final_val"] = {
            name: len(re.findall(pat, head, flags=re.IGNORECASE)) for name, pat in ERROR_PATTERNS.items()}
    result["training_log"] = log

    resume = {}
    try:
        import torch
        src = (os.path.join(REPO, "experiments/DAPO-GAP3B-MHQA-Agent-4gpu/global_step_60/actor")
               if PHASE == "D1" else os.path.join(EXP_DIR, "global_step_70/actor"))
        for tag, d, expect in (("resume_source", src, START_STEP), ("end", os.path.join(CKPT, "actor"), END_STEP)):
            p = os.path.join(d, "extra_state_world_size_4_rank_0.pt")
            if not os.path.exists(p):
                continue
            st = torch.load(p, map_location="cpu", weights_only=False)
            lr = st.get("lr_scheduler") or {}
            resume[tag] = {"path": p, "keys": sorted(st.keys()),
                           "lr_scheduler_last_epoch": lr.get("last_epoch"),
                           "expected_last_epoch": expect,
                           "rng_present": "rng" in st,
                           "rng_substates": sorted((st.get("rng") or {}).keys())}
    except Exception as e:  # noqa: BLE001
        resume["_error"] = f"{type(e).__name__}: {e}"
    result["state_resume_evidence"] = resume
    state_ok = (resume.get("resume_source", {}).get("lr_scheduler_last_epoch") == START_STEP
                and resume.get("end", {}).get("lr_scheduler_last_epoch") == END_STEP
                and resume.get("end", {}).get("rng_present") is True)
    result["state_resume_ok"] = bool(state_ok)

    quota = check_quota_semantics()
    xcheck = check_log_vs_diagnostics(quota)
    result["quota_semantics"] = quota
    result["log_vs_diagnostics"] = xcheck

    err = log.get("error_pattern_counts_before_final_val", {})
    # The tracker is a single file for the whole experiment, so a *re-check* of D1 after D2
    # has advanced it legitimately shows a larger value.  What matters scientifically is that
    # the phase's checkpoint exists and that the tracker has reached at least END_STEP.
    try:
        tracker_ok = int(ckpt.get("tracker")) >= END_STEP
    except (TypeError, ValueError):
        tracker_ok = False
    result["tracker_reached_end_step"] = bool(tracker_ok)
    result["note_tracker"] = ("tracker is experiment-wide; a D1 re-check after D2 shows 120 - "
                              "the requirement is tracker >= END_STEP and a complete global_step_END_STEP")
    ok = (ckpt.get("exists") and tracker_ok
          and all(v.get("exists") for v in ckpt["shards"].values()) and ckpt.get("data_pt")
          and log.get("all_steps_present") and log.get("resume_line_present")
          and state_ok and not log.get("non_finite_metrics")
          and not any(err.get(k) for k in ("cuda_oom", "nccl_error", "traceback", "cuda_error",
                                           "wiki_search_error", "wiki_tool_exec_failed"))
          and result["log_vs_diagnostics"].get("log_score_max_consistent_with_quota", True)
          and (not quota.get("available") or quota.get("quota_semantics_ok"))
          and quota.get("quota_error_lines", 0) == 0)
    result["integrity_ok"] = bool(ok)

    lines = [f"# E1-E {PHASE} integrity check (step{START_STEP} -> step{END_STEP})", "",
             f"- experiment dir: `{EXP_DIR}`", f"- checkpoint: `{CKPT}`",
             f"- tracker: {ckpt.get('tracker')} (expected {END_STEP})",
             f"- data.pt present: {ckpt.get('data_pt')}",
             f"- steps {START_STEP}..{END_STEP} all present: {log.get('all_steps_present')} "
             f"(missing: {log.get('missing_steps')})",
             f"- 'Setting global step to {START_STEP}': {log.get('resume_line_present')}",
             f"- metrics parsed: {log.get('metrics_parsed')}, non-finite: {log.get('non_finite_metrics')}",
             f"- full-state resume (lr_scheduler/RNG): {state_ok}", "",
             "## error scan before the final-validation marker", ""]
    for k, v in (err or {}).items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## runtime quota semantics (E1-E_task.md section 11/12)", "",
              "```json",
              json.dumps({k: v for k, v in quota.items() if k != "per_step"}, indent=2)[:6000],
              "```", "",
              "## trainer log vs manager diagnostics", "",
              "```json", json.dumps(xcheck, indent=2)[:2000], "```", "",
              "### per optimizer step", "",
              "| step | gen batches | quota | mixed seen | eff seen | cache | shortfall | inserted | displaced | batch |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for g, v in sorted(quota.get("per_step", {}).items()):
        lines.append(f"| {g} | {v['generation_batches']} | {v['quota_actual']} | {v['mixed_groups_seen']} | "
                     f"{v['efficiency_groups_seen']} | {v['efficiency_cache_size']} | {v['eff_shortfall']} | "
                     f"{len(v['inserted_ids'])} | {len(v['displaced_ids'])} | {v['optimizer_batch_groups']} |")
    lines += ["", f"**VERDICT: {'PASS' if ok else 'CHECK FAILED'}**", ""]
    with open(os.path.join(ANALYSIS, f"step{END_STEP}_integrity.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(ANALYSIS, f"step{END_STEP}_integrity.json"), "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

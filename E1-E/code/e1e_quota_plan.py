"""E1-E quota planner - the single pure implementation shared by the runtime reward
manager and the offline selftest.

Frozen semantics (E1-E0, q = 2, E1-E_task.md section 1):

  * a group is **mixed** iff 1 <= k <= 7 (k = correct rollouts out of n = 8); only mixed
    groups count towards the generation stopping target B = `data.train_batch_size`;
  * a group is an **efficiency group** iff k == 8 AND the correct rollouts differ in
    `logical_search_batches`;
  * while generating, efficiency groups are only *cached* (stream order) and never advance
    the stopping counter;
  * at the stopping generation batch:
        m = min(q, |cache|)
        optimizer batch = mixed[:B-m] + cache[:m]
    i.e. the last m of the B mixed groups the q = 0 baseline would have used are displaced
    by the m earliest cached efficiency groups;
  * selection is pure stream order - no difficulty / reward / source / length selection.

Implementation note (the only deviation mechanism)
--------------------------------------------------
The upstream trainer accumulates kept groups across generation batches and truncates to the
first B at the end, so a group already accumulated cannot be removed.  To guarantee that the
running count never reaches B before the mixed-only stopping batch, the planner caps the
number of *mixed* groups it keeps at ``B - min(q, |cache before this batch|)``.  Because the
cache is filled early in practice (E1-E0 measured min 3 per step), this cap is binding only
when more than B - q mixed groups arrive before the final batch, in which case the batch ends
up with slightly more mixed / fewer efficiency groups than the ideal.  Every step records
``eff_shortfall`` so the effect is measurable rather than silent.
"""
GROUP_MIXED = "mixed"
GROUP_EFFICIENCY = "efficiency"
GROUP_EXCLUDED = "excluded"


def classify_group(k, n_rollouts, cost_gap_correct):
    """E1-E group type for one n-rollout group."""
    if k == n_rollouts and cost_gap_correct is not None and cost_gap_correct > 0:
        return GROUP_EFFICIENCY
    if 1 <= k <= 7:
        return GROUP_MIXED
    return GROUP_EXCLUDED


def new_state():
    """Per-optimizer-step planner state."""
    return {
        "step": None,
        "mixed_seen": 0,       # every mixed group in the stopping prefix
        "mixed_kept": 0,       # mixed groups actually kept in the optimizer batch
        "cache": [],           # efficiency group ids in stream order, capped at q
        "batches": 0,
        "candidates_seen": 0,
        "eff_seen": 0,
        "eff_shortfall": 0,    # efficiency groups the ideal rule would have inserted but the
                               # accumulate-then-truncate design cannot place
        "mixed_dropped": 0,
        "stopping_done": False,
    }


def plan_batch(cur_groups, state, target_batch=32, quota=2):
    """Plan one generation batch.

    Args:
        cur_groups: dicts in stream order for this generation batch, with keys
            ``gid``, ``type`` (GROUP_*), ``k``, ``cost_gap``.
        state: dict from :func:`new_state` (mutated in place).
        target_batch: B.
        quota: q.

    Returns a plan dict (see the module docstring).  The returned ``mixed_kept`` ids are the
    mixed groups of THIS batch to keep; ``efficiency_kept`` are the cache-order efficiency
    group ids to place after them in the accumulated batch.
    """
    n_mixed_this = sum(1 for g in cur_groups if g["type"] == GROUP_MIXED)
    n_eff_this = sum(1 for g in cur_groups if g["type"] == GROUP_EFFICIENCY)

    state["batches"] += 1
    state["candidates_seen"] += len(cur_groups)
    state["eff_seen"] += n_eff_this

    cache_before = len(state["cache"])
    for g in cur_groups:                       # cache in stream order, capped at q
        if g["type"] == GROUP_EFFICIENCY and len(state["cache"]) < quota:
            state["cache"].append(g["gid"])

    m_prev = min(quota, cache_before)
    cap = target_batch - m_prev                # mixed kept may never exceed this

    mixed_this = [g["gid"] for g in cur_groups if g["type"] == GROUP_MIXED]
    a = state["mixed_seen"]
    new_mixed_seen = a + n_mixed_this
    is_stopping = new_mixed_seen >= target_batch
    state["mixed_seen"] = new_mixed_seen

    if not is_stopping:
        room = max(0, cap - state["mixed_kept"])
        keep = mixed_this[:room]
        dropped = mixed_this[room:]
        state["mixed_kept"] += len(keep)
        state["mixed_dropped"] += len(dropped)
        state["stopping_done"] = False
        return {
            "mixed_kept": keep, "efficiency_kept": [], "m": 0, "is_stopping": False,
            "needs_injection": False, "dropped_mixed": dropped,
            "cache_after": list(state["cache"]), "mixed_seen_after": new_mixed_seen,
            "eff_shortfall": 0,
        }

    # ---- stopping batch: mixed[:B-m] + cache[:m] ----
    m = min(quota, len(state["cache"]))
    need_mixed = max(0, target_batch - m - state["mixed_kept"])
    keep = mixed_this[:need_mixed]
    dropped = mixed_this[need_mixed:]
    state["mixed_kept"] += len(keep)
    state["mixed_dropped"] += len(dropped)

    # never let the accumulated batch exceed B: cap the efficiency slots to what is left
    room_eff = max(0, target_batch - state["mixed_kept"])
    eff_kept = list(state["cache"][:min(m, room_eff)])
    shortfall = m - len(eff_kept)
    state["eff_shortfall"] += shortfall
    state["stopping_done"] = True

    this_batch_gids = {g["gid"] for g in cur_groups}
    needs_injection = any(gid not in this_batch_gids for gid in eff_kept)
    return {
        "mixed_kept": keep, "efficiency_kept": eff_kept, "m": len(eff_kept),
        "is_stopping": True, "needs_injection": needs_injection, "dropped_mixed": dropped,
        "cache_after": list(state["cache"]), "mixed_seen_after": new_mixed_seen,
        "eff_shortfall": shortfall, "ideal_m": m,
    }


def reset_state(state):
    s = new_state()
    state.clear()
    state.update(s)
    return state

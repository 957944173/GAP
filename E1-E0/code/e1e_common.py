"""E1-E0 common utilities: E1-D D2 candidate stream, n=8 group reconstruction, and the
quota-constrained fixed-size optimizer-batch replay.

Reuse (E1-E0_task.md §0): this module is a copy-and-extend of
`E1-C/code/e1c_common.py` (itself validated in E1-C), which in turn reuses the group
reconstruction validated in E1-0.5.  What is kept verbatim:
    * group instance = `(call_index, uid)` (one n=8 group, exactly what the trainer's
      `filter_groups` operates on);
    * `logical_search_batches` as the search cost (never a new cost definition);
    * `kept iff np.std(reward) > 0 or |group| == 1`.
What E1-E0 adds: mixed / all-correct-efficiency classification, the mixed-only generation
stopping rule, the efficiency-group cache, and the quota batch construction
`mixed[:B-m] + efficiency_cache[:m]`.

READ-ONLY with respect to E1-0 / E1-0.5 / E1-A / E1-B / E1-C / E1-D: every artifact E1-E0
writes lands under `E1-E0/`.
"""
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np
import pandas as pd

REPO = "/data01/wyy/Graph-Agent-Planning"
E1E = os.path.join(REPO, "E1-E0")
RESULTS = os.path.join(E1E, "results")
LOGS = os.path.join(E1E, "logs")
FIGS = os.path.join(E1E, "figures")

# ---------------------------------------------------------------- candidate stream
D2_RUN = os.path.join(REPO, "E1-D/runs/D2_20260913_041846")
D2_ROLLOUTS = os.path.join(D2_RUN, "diagnostics/rollouts_pid1249393.jsonl")
D2_CALLS = os.path.join(D2_RUN, "diagnostics/calls_pid1249393.jsonl")
D2_CONFIG = os.path.join(REPO, "E1-D/configs/resolved_step70_to120.json")
E1_C_SUMMARY = os.path.join(REPO, "E1-C/results/summary.json")
E1_D_SUMMARY = os.path.join(REPO, "E1-D/analysis/summary.json")

ROLLOUT_COLS = ["call_index", "global_step", "uid", "data_source", "em",
                "logical_search_batches", "search_queries", "conversation_turns",
                "response_token_length", "metadata_valid"]

LAMBDA_MAIN = 0.05


def d2_files():
    files = sorted(glob.glob(os.path.join(D2_RUN, "diagnostics", "rollouts_pid*.jsonl")))
    if not files:
        raise FileNotFoundError(f"no E1-D D2 rollout diagnostics under {D2_RUN}/diagnostics")
    return files


def load_stream(verbose=True):
    """Load the E1-D D2 candidate stream in trainer order (file order == batch order)."""
    rows = []
    for path in d2_files():
        with open(path, errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                em = r.get("em")
                rows.append([
                    r.get("call_index"), r.get("global_step"), str(r.get("uid")),
                    r.get("data_source"),
                    (float(em) if em is not None else np.nan),
                    r.get("logical_search_batches"), r.get("search_queries"),
                    r.get("conversation_turns"), r.get("response_token_length"),
                    bool(r.get("metadata_valid", True)),
                ])
    df = pd.DataFrame(rows, columns=ROLLOUT_COLS)
    df["src_order"] = np.arange(len(df), dtype=np.int64)
    df["correct"] = (df["em"] >= 0.5).astype(float)
    if verbose:
        print(f"[e1e] loaded {len(df):,} rollouts from {len(d2_files())} file(s)")
    return df


def build_group_table(df):
    """Aggregate rollouts into n=8 group instances keyed by (call_index, uid).

    Classification (E1-E0_task.md §2.A/§2.B), computed from the raw rollouts:
        mixed-correctness group      : 1 <= k <= 7
        all-correct efficiency group : k == 8 AND max(cost) > min(cost) over correct rollouts
        all-wrong group              : k == 0      (never used by GAP training)
    """
    out = []
    for (call_index, uid), recs in df.groupby(["call_index", "uid"], sort=False):
        n = len(recs)
        correct = recs["correct"].to_numpy(dtype=float)
        cost = recs["logical_search_batches"].to_numpy(dtype=float)
        queries = recs["search_queries"].to_numpy(dtype=float)
        toks = recs["response_token_length"].to_numpy(dtype=float)
        turns = recs["conversation_turns"].to_numpy(dtype=float)
        cor = correct > 0
        k = int(cor.sum())
        c_cor = cost[cor]
        gap = float(c_cor.max() - c_cor.min()) if k >= 1 else 0.0
        out.append({
            "group_id": f"{int(call_index)}:{uid}",
            "call_index": int(call_index),
            "global_step": int(recs["global_step"].iloc[0]),
            "uid": str(uid),
            "data_source": recs["data_source"].iloc[0],
            "n_rollouts": n,
            "k": k,
            "group_success_rate": k / n if n else 0.0,
            "mixed": bool(1 <= k <= 7),
            "all_correct": bool(k == n),
            "all_wrong": bool(k == 0),
            "cost_gap_correct": gap,
            "efficiency_variable": bool(k == n and gap > 0.0),
            "efficiency_group": bool(k == n and gap > 0.0),   # alias used by the task wording
            "cost_all_mean": float(cost.mean()),
            "cost_correct_mean": float(c_cor.mean()) if k else None,
            "cost_correct_min": float(c_cor.min()) if k else None,
            "cost_correct_max": float(c_cor.max()) if k else None,
            "queries_all_mean": float(queries.mean()),
            "queries_correct_mean": float(queries[cor].mean()) if k else None,
            "tokens_all_mean": float(toks.mean()),
            "tokens_correct_mean": float(toks[cor].mean()) if k else None,
            "turns_all_mean": float(turns.mean()),
            "src_order": int(recs["src_order"].min()),
        })
    g = pd.DataFrame(out)
    # group-relative efficiency E for the correct rollouts of an efficiency group
    eff_mean = {}
    eff_max = {}
    for row in g.itertuples(index=False):
        key = row.group_id
        if not row.efficiency_variable:
            eff_mean[key] = 0.0
            eff_max[key] = 0.0
            continue
        recs = df[(df["call_index"] == row.call_index) & (df["uid"] == row.uid)]
        correct = recs["correct"].to_numpy(dtype=float)
        cost = recs["logical_search_batches"].to_numpy(dtype=float)
        E = efficiency_vector(correct, cost)
        eff_mean[key] = float(np.mean(E))
        eff_max[key] = float(np.max(E))
    g["efficiency_E_mean"] = g["group_id"].map(eff_mean)
    g["efficiency_E_max"] = g["group_id"].map(eff_max)
    return g


def group_arrays(df):
    """{(call_index, uid): (correct[], cost[], queries[], tokens[])}"""
    out = {}
    for (ci, uid), recs in df.groupby(["call_index", "uid"], sort=False):
        out[(int(ci), str(uid))] = (
            recs["correct"].to_numpy(dtype=float),
            recs["logical_search_batches"].to_numpy(dtype=float),
            recs["search_queries"].to_numpy(dtype=float),
            recs["response_token_length"].to_numpy(dtype=float),
        )
    return out


def efficiency_vector(correct, cost):
    """E_i = (Cmax - C_i)/(Cmax - Cmin) for the correct rollouts, else 0 (E1-A/E1-B definition)."""
    correct = np.asarray(correct, dtype=float)
    cost = np.asarray(cost, dtype=float)
    E = np.zeros_like(cost, dtype=float)
    cor = correct > 0
    if cor.sum() >= 2:
        c = cost[cor]
        cmin, cmax = c.min(), c.max()
        if cmax > cmin:
            E[cor] = (cmax - c) / (cmax - cmin)
    return E


def gap_filter_keep(scores):
    """The trainer's rule (ray_trainer.py:1780-1781): kept iff std > 0 or |group| == 1."""
    s = np.asarray(scores, dtype=float)
    return bool(len(s) == 1 or np.std(s) > 0.0)


# ---------------------------------------------------------------- quota replay
def replay_stream(g, target_batch, verbose=True):
    """Replay the generation loop and build the q=0 .. q=4 optimizer batches.

    The stream is consumed in trainer order.  Only **mixed-correctness** groups count
    towards the stopping target (E1-E0_task.md §2.C); all-correct efficiency groups are
    cached in the same pass, in stream order, and never influence when generation stops.

    Returns (per_step, per_quota_rows, batch_members) where
        per_step[step] = dict with the stopping point, mixed list, efficiency cache
        batch_members[(step, q)] = list of (group_id, role)
    """
    g = g.sort_values(["global_step", "src_order"], kind="stable")
    by_step = defaultdict(list)
    for row in g.itertuples(index=False):
        by_step[int(row.global_step)].append(row)

    per_step = {}
    for step in sorted(by_step):
        rows = sorted(by_step[step], key=lambda r: r.src_order)
        mixed, eff_cache = [], []
        calls_seen = set()
        mixed_all_in_step = eff_all_in_step = 0
        for r in rows:
            calls_seen.add(r.call_index)
            if r.mixed:
                mixed_all_in_step += 1
                if len(mixed) < target_batch:
                    mixed.append(r.group_id)
            elif r.efficiency_group:
                eff_all_in_step += 1
                if len(mixed) < target_batch:      # only groups inside the stopping prefix
                    eff_cache.append(r.group_id)
            if len(mixed) >= target_batch:
                break
        per_step[step] = {
            "global_step": step,
            "generation_batches_consumed": len(calls_seen),
            "candidate_groups_seen": len(rows),
            "candidate_groups_seen_prefix": sum(
                1 for r in rows if r.call_index in calls_seen),
            "mixed_groups_available": mixed_all_in_step,
            "efficiency_groups_available": eff_all_in_step,
            "mixed_prefix": mixed,
            "efficiency_cache": eff_cache,
            "mixed_prefix_count": len(mixed),
            "efficiency_cache_count": len(eff_cache),
            "stopping_reached": len(mixed) >= target_batch,
        }

    # sanity: every step's stopping point must be the same for every q by construction
    batch_members = {}
    for step, st in per_step.items():
        mixed = st["mixed_prefix"]
        eff = st["efficiency_cache"]
        for q in (0, 1, 2, 3, 4):
            m = min(q, len(eff))
            members = [(gid, "mixed") for gid in mixed[:target_batch - m]]
            members += [(gid, "efficiency") for gid in eff[:m]]
            batch_members[(step, q)] = members
    return per_step, batch_members


# ---------------------------------------------------------------- GRPO advantage
def grpo_advantage(scores, uids, epsilon=1e-6, norm_by_std=True):
    """Exact replica of verl `compute_grpo_outcome_advantage`
    (norm_adv_by_std_in_grpo=True, epsilon=1e-6, std with ddof=1)."""
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
            adv[i] = ((scores[i] - mean) / (std + epsilon)) if norm_by_std else (scores[i] - mean)
    return adv


# ---------------------------------------------------------------- stats helpers
def _rankdata(a):
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    sa = a[order]
    i = 0
    while i < len(sa):
        j = i
        while j + 1 < len(sa) and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def mannwhitney_u(a, b):
    a = np.asarray([x for x in a if x is not None and np.isfinite(x)], dtype=float)
    b = np.asarray([x for x in b if x is not None and np.isfinite(x)], dtype=float)
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return {"u": None, "z": None, "p": None, "n1": n1, "n2": n2,
                "mean_a": None, "mean_b": None}
    allv = np.concatenate([a, b])
    ranks = _rankdata(allv)
    u1 = ranks[:n1].sum() - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    _, counts = np.unique(allv, return_counts=True)
    tie = np.sum(counts ** 3 - counts)
    n = n1 + n2
    var = n1 * n2 / 12.0 * ((n + 1) - tie / (n * (n - 1))) if n > 1 else 0.0
    if var <= 0:
        return {"u": float(u1), "z": 0.0, "p": 1.0, "n1": n1, "n2": n2,
                "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b))}
    z = (u1 - mu) / math.sqrt(var)
    return {"u": float(u1), "z": float(z), "p": float(math.erfc(abs(z) / math.sqrt(2))),
            "n1": n1, "n2": n2, "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)),
            "median_a": float(np.median(a)), "median_b": float(np.median(b))}


def two_prop_z(k1, n1, k2, n2):
    if n1 == 0 or n2 == 0:
        return {"p": None, "z": None}
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return {"p": 1.0, "z": 0.0, "p1": p1, "p2": p2}
    z = (p1 - p2) / se
    return {"z": float(z), "p": float(math.erfc(abs(z) / math.sqrt(2))), "p1": float(p1), "p2": float(p2)}


def chi2_independence(table):
    """2-D contingency table independence test (no scipy): chi2 + p via erfc on the
    normal approximation of the chi-square with k-1 dof (Wilson-Hilferty)."""
    t = np.asarray(table, dtype=float)
    if t.ndim != 2 or t.sum() == 0 or (t.sum(axis=0) == 0).any() or (t.sum(axis=1) == 0).any():
        return {"chi2": None, "p": None, "dof": None}
    exp = np.outer(t.sum(axis=1), t.sum(axis=0)) / t.sum()
    chi2 = float(((t - exp) ** 2 / np.where(exp == 0, 1, exp)).sum())
    dof = (t.shape[0] - 1) * (t.shape[1] - 1)
    if dof <= 0:
        return {"chi2": chi2, "p": None, "dof": dof}
    z = ((chi2 / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / math.sqrt(2 / (9 * dof))
    p = math.erfc(abs(z) / math.sqrt(2))
    return {"chi2": chi2, "p": float(p), "dof": int(dof)}

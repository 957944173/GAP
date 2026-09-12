"""E1-C common utilities: multi-source rollout pools, group reconstruction, filter
replay, GRPO advantage, and small statistical tests.

Provenance / reuse (E1-C_task.md §0.3 and §2):
    This module is a copy-and-extend of `E1-0.5/scripts/e105_common.py`
    (which was itself validated in E1-0.5).  The group-reconstruction decision
    (`group = (call_index, uid)` == one n=8 group instance) and the filter
    semantics (`kept iff np.std(group_reward) > 0 or group size == 1`) are kept
    verbatim; what is added here is:
      * three candidate pools instead of one (E1-0 audit, E1-A diagnostics,
        E1-B diagnostics) behind one adapter interface;
      * λ-parameterised reward so the same pool can be replayed at
        λ ∈ {0, 0.01, 0.025, 0.05, 0.1, 0.2};
      * a sequential fixed-size batch-fill replay that returns the actual
        **selected group IDs** (not just counts), reproducing
        ray_trainer.py's `num_prompt_in_batch` loop + `batch[:traj_bsz]`
        truncation;
      * the exact GRPO outcome-advantage formula
        (`verl/trainer/ppo/core_algos.py::compute_grpo_outcome_advantage`,
        `norm_adv_by_std_in_grpo=True`, `epsilon=1e-6`, `std` with ddof=1);
      * dependency-free rank-sum / two-proportion tests (this environment has
        no scipy).

READ-ONLY with respect to E1-0 / E1-0.5 / E1-A / E1-B: every artifact E1-C
writes lands under `E1-C/`.
"""
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np
import pandas as pd

REPO = "/data01/wyy/Graph-Agent-Planning"
E1C = os.path.join(REPO, "E1-C")
RESULTS = os.path.join(E1C, "results")
LOGS = os.path.join(E1C, "logs")
FIGS = os.path.join(E1C, "figures")

TARGET_TRAIN_BATCH_SIZE_FALLBACK = 32

# --------------------------------------------------------------------------- #
# Pool definitions
# --------------------------------------------------------------------------- #
POOLS = {
    "E1_0": {
        "label": "E1-0 (steps 61-70, ORIGINAL correctness reward, actual filter = GAP)",
        "glob": os.path.join(REPO, "E1-0/runs/20260910_step60to70_e10/audit/rollouts_pid*.jsonl"),
        "files": [os.path.join(REPO, "E1-0/runs/20260910_step60to70_e10/audit/rollouts_pid2552655.jsonl")],
        "reward_field": "reward_original",
        "actual_filter": "GAP",
        "kind": "audit",
    },
    "E1_A": {
        "label": "E1-A (steps 61-70, E1 shaped reward λ=0.05, actual filter = E1)",
        "glob": os.path.join(REPO, "E1-A/run_20260911_024902/diagnostics/rollouts_pid*.jsonl"),
        "files": None,
        "reward_field": "reward",
        "actual_filter": "E1",
        "kind": "diag",
    },
    "E1_B": {
        "label": "E1-B (steps 71-120, E1 shaped reward λ=0.05, actual filter = E1)",
        "glob": os.path.join(REPO, "E1-B/run_20260911_113432/diagnostics/rollouts_pid*.jsonl"),
        "files": None,
        "reward_field": "reward",
        "actual_filter": "E1",
        "kind": "diag",
    },
}

CFG_E1B = os.path.join(REPO, "E1-B/run_20260911_113432/config/resolved_config.json")
CFG_E1A = os.path.join(REPO, "E1-A/run_20260911_024902/config/resolved_config.json")

ROLLOUT_COLS = [
    "call_index", "global_step", "call_within_step", "uid", "data_source", "em",
    "logical_search_batches", "search_queries", "conversation_turns",
    "response_token_length", "metadata_valid", "zero_search", "reward_raw",
]


def load_config(path):
    with open(path) as fh:
        return json.load(fh)


def train_batch_size():
    """Auto-read data.train_batch_size (E1-C_task.md §5: do not hardcode)."""
    for p in (CFG_E1B, CFG_E1A):
        if os.path.exists(p):
            cfg = load_config(p)
            return int(cfg["data"]["train_batch_size"]), p
    return TARGET_TRAIN_BATCH_SIZE_FALLBACK, "fallback-constant"


def rollout_files(pool):
    spec = POOLS[pool]
    if spec["files"]:
        return [p for p in spec["files"] if os.path.exists(p)]
    return sorted(glob.glob(spec["glob"]))


def load_pool(pool, verbose=True):
    """Load one candidate rollout pool into a DataFrame with a stable row order.

    The row order of the source files is the order in which the reward manager
    iterated the batch (`for i in range(n)` inside the manager), i.e. the batch
    order the trainer saw.  We keep it as `src_order` so the sequential
    batch-fill replay can reproduce the real ordering.
    """
    spec = POOLS[pool]
    files = rollout_files(pool)
    if not files:
        raise FileNotFoundError(f"no rollout files for pool {pool}: {spec['glob']}")
    rows = []
    for path in files:
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
                rw = r.get(spec["reward_field"])
                rows.append([
                    r.get("call_index"), r.get("global_step"), r.get("call_within_step"),
                    str(r.get("uid")), r.get("data_source"),
                    (float(em) if em is not None else np.nan),
                    r.get("logical_search_batches"), r.get("search_queries"),
                    r.get("conversation_turns"), r.get("response_token_length"),
                    bool(r.get("metadata_valid", True)), bool(r.get("zero_search", False)),
                    (float(rw) if rw is not None else np.nan),
                ])
    df = pd.DataFrame(rows, columns=ROLLOUT_COLS)
    df["src_order"] = np.arange(len(df), dtype=np.int64)
    df["correct"] = (df["em"] >= 0.5).astype(float)
    if verbose:
        print(f"[e1c] pool {pool}: {len(df):,} rollouts from {len(files)} file(s)")
    return df


# --------------------------------------------------------------------------- #
# Group table
# --------------------------------------------------------------------------- #
def build_group_table(df):
    """Aggregate rollouts into n=8 group instances keyed by (call_index, uid).

    Same grouping decision as E1-0.5 (validated there).  All derived quantities
    are RE-COMPUTED here from `correct` + `logical_search_batches` rather than
    trusted from the recorded diagnostics, so the E1-C numbers are independent
    of any field the earlier reward managers wrote.
    """
    out = []
    for (call_index, uid), recs in df.groupby(["call_index", "uid"], sort=False):
        n = len(recs)
        correct = recs["correct"].to_numpy(dtype=float)
        cost = recs["logical_search_batches"].to_numpy(dtype=float)
        queries = recs["search_queries"].to_numpy(dtype=float)
        toks = recs["response_token_length"].to_numpy(dtype=float)
        turns = recs["conversation_turns"].to_numpy(dtype=float)
        cor_mask = correct > 0
        cc = int(cor_mask.sum())
        c_cor = cost[cor_mask]
        gap = float(c_cor.max() - c_cor.min()) if cc >= 1 else 0.0
        active = bool(cc >= 2 and gap > 0)
        first_order = int(recs["src_order"].min())
        out.append({
            "group_id": f"{int(call_index)}:{uid}",
            "call_index": int(call_index),
            "global_step": int(recs["global_step"].iloc[0]),
            "call_within_step": (int(recs["call_within_step"].iloc[0])
                                 if recs["call_within_step"].notna().all() else -1),
            "uid": str(uid),
            "data_source": recs["data_source"].iloc[0],
            "n_rollouts": n,
            "correct_count": cc,
            "all_wrong": cc == 0,
            "all_correct": (cc == n),
            "mixed": 0 < cc < n,
            "group_success_rate": cc / n if n else 0.0,
            "cost_gap_correct": gap,
            "cost_correct_min": float(c_cor.min()) if cc else None,
            "cost_correct_max": float(c_cor.max()) if cc else None,
            "cost_all_mean": float(cost.mean()),
            "cost_correct_mean": float(c_cor.mean()) if cc else None,
            "queries_all_mean": float(queries.mean()),
            "queries_correct_mean": float(queries[cor_mask].mean()) if cc else None,
            "tokens_all_mean": float(toks.mean()),
            "turns_all_mean": float(turns.mean()),
            "is_active_efficiency": active,
            "src_order": first_order,
        })
    g = pd.DataFrame(out)
    g["all_correct_or_wrong"] = g["all_wrong"] | g["all_correct"]
    return g


def group_vectors(df, g):
    """Return {(call_index, uid): (correct[], cost[], E[])} for reward replay."""
    vec = {}
    for (ci, uid), recs in df.groupby(["call_index", "uid"], sort=False):
        correct = recs["correct"].to_numpy(dtype=float)
        cost = recs["logical_search_batches"].to_numpy(dtype=float)
        vec[(int(ci), str(uid))] = (correct, cost)
    return vec


def efficiency_vector(correct, cost):
    """E_i = (Cmax - C_i)/(Cmax - Cmin) for correct rollouts of an active group."""
    E = np.zeros_like(cost, dtype=float)
    cor = correct > 0
    if cor.sum() >= 2:
        c = cost[cor]
        cmin, cmax = c.min(), c.max()
        if cmax > cmin:
            E[cor] = (cmax - c) / (cmax - cmin)
    return E


def reward_vector(correct, cost, lam):
    """R_i = A_i * (1 + λ E_i); wrong rollout ⇒ exactly 0."""
    E = efficiency_vector(correct, cost)
    R = correct * (1.0 + lam * E)
    R[correct <= 0] = 0.0
    return R, E


def group_kept(scores):
    """GAP's filter (ray_trainer.py:1780-1781): kept iff std > 0 or |group| == 1."""
    s = np.asarray(scores, dtype=float)
    return bool(len(s) == 1 or np.std(s) > 0.0)


def classify_set_relation(n_gap, n_e1, inter):
    if n_gap == 0 and n_e1 == 0:
        return "BOTH_EMPTY"
    if n_gap == inter and n_e1 == inter:
        return "EQUAL"
    if n_gap == inter and n_e1 > inter:
        return "GAP_STRICT_SUBSET_OF_E1"
    if n_e1 == inter and n_gap > inter:
        return "E1_STRICT_SUBSET_OF_GAP"
    if inter == 0:
        return "DISJOINT"
    if inter / max(min(n_gap, n_e1), 1) < 0.10:
        return "NEAR_DISJOINT"
    return "PARTIAL_OVERLAP"


# --------------------------------------------------------------------------- #
# Sequential fixed-size batch-fill replay
# --------------------------------------------------------------------------- #
def replay_batchfill(g, vec, lam, target, mode=None, keep_col=None):
    """Reproduce the trainer's generation loop on an ORDERED candidate stream.

    ray_trainer.py:
        kept = [uid for uid, std in std_by_uid.items() if std > 0 or len == 1]
        num_prompt_in_batch += len(kept)          # per generation batch
        batch = concat([batch, new_batch])
        if num_prompt_in_batch < train_batch_size: continue generating
        else: batch = batch[: train_batch_size * n]     # truncation

    So the optimizer batch is the FIRST `target` kept groups in generation
    order.  Returns per-optimizer-step rows with the selected group IDs and how
    many generation batches were consumed.
    """
    g = g.sort_values(["global_step", "src_order"], kind="stable")
    per_call_keep = {}
    for row in g.itertuples(index=False):
        if keep_col is not None:
            kept = bool(getattr(row, keep_col))
        else:
            correct, cost = vec[(row.call_index, row.uid)]
            if mode == "GAP":
                kept = group_kept(correct)
            elif mode == "E1":
                R, _ = reward_vector(correct, cost, lam)
                kept = group_kept(R)
            else:
                raise ValueError("mode or keep_col required")
        per_call_keep.setdefault(int(row.global_step), []).append(
            (int(row.call_index), row.group_id, row.src_order, kept))

    steps = []
    for step in sorted(per_call_keep):
        calls = sorted(per_call_keep[step], key=lambda x: x[2])
        total_kept_if_all_calls = sum(1 for _ci, _gid, _so, kept in calls if kept)
        # reproduce: consume generation batches in order, accumulate kept groups,
        # stop as soon as the target count is reached, then take the first `target`
        acc, calls_needed, selected = 0, 0, []
        seen_calls = set()
        for call_index, gid, _so, kept in calls:
            if call_index not in seen_calls:
                seen_calls.add(call_index)
                calls_needed += 1
            if kept:
                acc += 1
                selected.append(gid)
            if acc >= target:
                break
        selected = selected[:target]
        steps.append({
            "global_step": step,
            "n_generation_batches_available": len(seen_calls),
            "calls_needed": calls_needed,
            "kept_in_prefix": acc,
            "selected": selected,
            "selected_count": len(selected),
            "dropped_surplus": max(0, acc - target),
            "total_kept_if_all_calls": total_kept_if_all_calls,
        })
    return steps


# --------------------------------------------------------------------------- #
# GRPO advantage (exact replica of core_algos.compute_grpo_outcome_advantage)
# --------------------------------------------------------------------------- #
def grpo_advantage(scores, uids, epsilon=1e-6, norm_by_std=True):
    """scores/uids: same-length arrays for the trajectories of one optimizer batch."""
    scores = np.asarray(scores, dtype=float)
    id2scores = defaultdict(list)
    for i, u in enumerate(uids):
        id2scores[u].append(scores[i])
    adv = np.zeros_like(scores)
    for u, vals in id2scores.items():
        v = np.asarray(vals, dtype=float)
        idx = [i for i, uu in enumerate(uids) if uu == u]
        if len(v) == 1:
            mean, std = 0.0, 1.0
        else:
            mean = float(np.mean(v))
            std = float(np.std(v, ddof=1))          # torch.std default: unbiased
        for i in idx:
            adv[i] = ((scores[i] - mean) / (std + epsilon)) if norm_by_std else (scores[i] - mean)
    return adv


# --------------------------------------------------------------------------- #
# dependency-free statistics
# --------------------------------------------------------------------------- #
def _rankdata(a):
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    # average ties
    sorted_a = a[order]
    i = 0
    while i < len(sorted_a):
        j = i
        while j + 1 < len(sorted_a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def mannwhitney_u(a, b):
    """Two-sided Mann-Whitney U via normal approximation with tie correction."""
    a = np.asarray([x for x in a if x is not None and np.isfinite(x)], dtype=float)
    b = np.asarray([x for x in b if x is not None and np.isfinite(x)], dtype=float)
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return {"u": None, "z": None, "p": None, "n1": n1, "n2": n2}
    allv = np.concatenate([a, b])
    ranks = _rankdata(allv)
    r1 = ranks[:n1].sum()
    u1 = r1 - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    _, counts = np.unique(allv, return_counts=True)
    tie = np.sum(counts ** 3 - counts)
    n = n1 + n2
    var = n1 * n2 / 12.0 * ((n + 1) - tie / (n * (n - 1))) if n > 1 else 0.0
    if var <= 0:
        return {"u": float(u1), "z": 0.0, "p": 1.0, "n1": n1, "n2": n2}
    z = (u1 - mu) / math.sqrt(var)
    p = math.erfc(abs(z) / math.sqrt(2))
    return {"u": float(u1), "z": float(z), "p": float(p), "n1": n1, "n2": n2,
            "median_a": float(np.median(a)), "median_b": float(np.median(b)),
            "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b))}


def two_prop_z(k1, n1, k2, n2):
    if n1 == 0 or n2 == 0:
        return {"p": None, "z": None}
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return {"p": 1.0, "z": 0.0, "p1": p1, "p2": p2}
    z = (p1 - p2) / se
    return {"z": float(z), "p": float(math.erfc(abs(z) / math.sqrt(2))),
            "p1": float(p1), "p2": float(p2)}


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)

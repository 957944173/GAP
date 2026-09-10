"""E1-0.5 common utilities: load the E1-0 rollout audit and build the n=8 group table.

READ-ONLY with respect to E1-0: this module never writes into `E1-0/`. All of its
own artifacts go to `E1-0.5/outputs/`.

Input (E1-0 audit, produced by the E1-0 passive-audit reward manager):
    E1-0/runs/20260910_step60to70_e10/audit/rollouts_pid2552655.jsonl
    147,200 records = 115 generation batches x 160 prompts x n=8 rollouts

Grouping decision (documented in README.md and the final report):
    The task says "group by uid; each group should contain 8 rollouts".  A `uid`
    is the dataset row index, and the SAME uid is re-sampled in later global
    steps / generation batches, so grouping by uid alone would merge up to 115
    different n=8 groups into one bucket of 8*115 rollouts.  The training-time
    grouping (and the grouping GAP's `filter_groups` actually operates on) is
    `(generation batch = call_index, uid)`, which yields exactly 8 rollouts per
    group -- 18,400 groups.  We therefore group by `(call_index, uid)` and also
    keep the uid-level multiplicity in the outputs.

Field semantics of the E1-0 audit (verified against the rollouts, see README):
    logical_search_batches   = number of original <wiki_search> blocks the
                               assistant emitted (== structured_search_rounds
                               in this run: the model never put two blocks in
                               one assistant turn)
    search_queries           = number of individual queries after the parser's
                               '|' split
    parallel_factor          = search_queries / logical_search_batches
    structured_conversation_turns = number of assistant messages in the trajectory
    response_token_length    = valid response tokens
    resp_xml_search_batches  = <wiki_search> blocks counted in the *re-rendered*
                               response text.  NOTE: SGLangRollout replaces the
                               tokenizer chat template (sglang_rollout.py:289)
                               and renders EACH parallel query as its own
                               <wiki_search> block, so this equals
                               `search_queries` (99.79% of records; the rest are
                               length-truncated), NOT logical_search_batches.
"""
import json
import os

import numpy as np
import pandas as pd

REPO = "/data01/wyy/Graph-Agent-Planning"
E10_AUDIT_CANDIDATES = [
    os.path.join(REPO, "E1-0/runs/20260910_step60to70_e10/audit/rollouts_pid2552655.jsonl"),
]
E105 = os.path.join(REPO, "E1-0.5")
OUTPUTS = os.path.join(E105, "outputs")
REPORTS = os.path.join(E105, "reports")
FIGURES = os.path.join(REPORTS, "figures")

ROLLOUT_FIELDS = [
    "call_index", "global_step", "uid", "extra_info_index", "data_source", "em",
    "logical_search_batches", "structured_search_rounds", "search_queries",
    "parallel_factor", "structured_conversation_turns", "response_token_length",
    "response_char_length", "queries_per_round", "rollout_pos_in_group",
    "zero_search", "metadata_valid", "resp_xml_search_batches",
]


def audit_path():
    for p in E10_AUDIT_CANDIDATES:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "E1-0 rollout audit not found. Looked in: %s" % E10_AUDIT_CANDIDATES
    )


def load_rollouts(verbose=True):
    """Stream the JSONL audit into a compact DataFrame with only needed fields."""
    path = audit_path()
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append([r.get(k) for k in ROLLOUT_FIELDS])
    df = pd.DataFrame(rows, columns=ROLLOUT_FIELDS)
    if verbose:
        print(f"[e105] loaded {len(df):,} rollouts from {path}")
    return df


def build_group_table(df):
    """Aggregate rollouts into n=8 group instances keyed by (call_index, uid)."""
    out = []
    for (call_index, uid), recs in df.groupby(["call_index", "uid"], sort=False):
        cost = recs["logical_search_batches"].to_numpy(dtype=float)
        queries = recs["search_queries"].to_numpy(dtype=float)
        pf = recs["parallel_factor"].to_numpy(dtype=float)
        toks = recs["response_token_length"].to_numpy(dtype=float)
        turns = recs["structured_conversation_turns"].to_numpy(dtype=float)
        em = recs["em"].to_numpy(dtype=float)
        correct = em >= 0.5
        n = len(recs)
        cc = int(correct.sum())

        c_correct = cost[correct]
        q_correct = queries[correct]
        pf_correct = pf[correct]
        tok_correct = toks[correct]
        turns_correct = turns[correct]

        # active efficiency group: >=2 correct rollouts AND correct-cost variation
        if cc >= 2:
            gap_correct = float(c_correct.max() - c_correct.min())
        else:
            gap_correct = 0.0
        is_active = bool(cc >= 2 and gap_correct > 0)

        # min/max cost among correct, and the trajectories at each end
        if cc >= 1:
            cmin, cmax = float(c_correct.min()), float(c_correct.max())
        else:
            cmin = cmax = None

        out.append({
            "group_id": f"{int(call_index)}:{uid}",
            "call_index": int(call_index),
            "global_step": int(recs["global_step"].iloc[0]),
            "uid": str(uid),
            "extra_info_index": int(recs["extra_info_index"].iloc[0]),
            "data_source": recs["data_source"].iloc[0],
            "n_rollouts": n,
            "correct_count": cc,
            "wrong_count": n - cc,
            "all_correct": cc == n,
            "all_wrong": cc == 0,
            "mixed": 0 < cc < n,
            # cost (logical_search_batches) over all / correct rollouts
            "cost_all_mean": float(cost.mean()),
            "cost_all_min": float(cost.min()),
            "cost_all_max": float(cost.max()),
            "cost_correct_mean": float(c_correct.mean()) if cc else None,
            "cost_correct_min": cmin,
            "cost_correct_max": cmax,
            "cost_gap_correct": gap_correct,
            "cost_unique_correct": sorted(set(int(x) for x in c_correct)),
            # queries
            "queries_all_mean": float(queries.mean()),
            "queries_correct_mean": float(q_correct.mean()) if cc else None,
            "queries_correct_max": float(q_correct.max()) if cc else None,
            # parallel factor
            "parallel_factor_all_mean": float(pf.mean()),
            "parallel_factor_correct_mean": float(pf_correct.mean()) if cc else None,
            "parallel_factor_max": float(pf.max()),
            # turns / tokens
            "turns_all_mean": float(turns.mean()),
            "turns_correct_mean": float(turns_correct.mean()) if cc else None,
            "response_tokens_all_mean": float(toks.mean()),
            "response_tokens_correct_mean": float(tok_correct.mean()) if cc else None,
            "is_active": is_active,
        })
    g = pd.DataFrame(out)
    return g


def efficiency_scores(costs, active):
    """Group-relative efficiency E in [0,1] for the corrected trajectories.

    E_i = (Cmax - C_i) / (Cmax - Cmin) for correct rollouts of an active group,
    else 0.  Matches the definition in E1-0_task.md section 14 / E1-0.5 Part 7.
    """
    costs = np.asarray(costs, dtype=float)
    if not active or len(costs) == 0:
        return np.zeros_like(costs)
    cmin, cmax = costs.min(), costs.max()
    if cmax <= cmin:
        return np.zeros_like(costs)
    return (cmax - costs) / (cmax - cmin)


def gap_filter_keep(rewards):
    """GAP's actual group filter, ray_trainer.py:1780-1781.

    kept iff np.std(group_rewards) > 0  (or the group has a single member)
    """
    r = np.asarray(rewards, dtype=float)
    return bool(len(r) == 1 or np.std(r) > 0.0)

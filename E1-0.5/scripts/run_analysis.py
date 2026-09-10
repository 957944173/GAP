#!/usr/bin/env python3
"""E1-0.5 offline diagnosis: Parts 2-7.

Reads the E1-0 rollout audit (read-only), builds the n=8 group table, and writes
every artifact under E1-0.5/.  Nothing here trains, samples, or modifies any GAP
code; it is a pure offline analysis of already-collected rollouts.

Usage:
    python3 E1-0.5/scripts/run_analysis.py
"""
import json
import os
import subprocess
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e105_common as C  # noqa: E402

LAMBDA = 0.05
TRAIN_BATCH_SIZE = 32   # data.train_batch_size in the E1-0 run


# --------------------------------------------------------------------------- #
# Part 2: active group analysis
# --------------------------------------------------------------------------- #
def part2(df, g):
    os.makedirs(C.OUTPUTS, exist_ok=True)

    # ---- per-rollout efficiency class, fully vectorised ----
    key = ["call_index", "uid"]
    gg = g[key + ["is_active", "cost_correct_min", "cost_correct_max"]]
    df = df.merge(gg, on=key, how="left", validate="many_to_one")

    is_correct = df["em"] >= 0.5
    cost = df["logical_search_batches"]
    act = df["is_active"].fillna(False)
    cls = pd.Series("inactive_group", index=df.index, dtype=object)
    cls[~is_correct] = "incorrect"
    cls[act & is_correct & (cost == df["cost_correct_min"])] = "efficient_correct"
    cls[act & is_correct & (cost == df["cost_correct_max"])] = "inefficient_correct"
    cls[act & is_correct & (cost > df["cost_correct_min"]) & (cost < df["cost_correct_max"])] = "middle_correct"
    df["efficiency_class"] = cls

    # ---- active group detail (self-contained for Parts 3/5/6) ----
    act_df = df[df["is_active"].fillna(False)]
    rows = []
    for (call_index, uid), recs in act_df.groupby(key, sort=True):
        cor = recs[recs["em"] >= 0.5]
        eff = cor[cor["efficiency_class"] == "efficient_correct"]
        ine = cor[cor["efficiency_class"] == "inefficient_correct"]
        rows.append({
            "group_id": f"{int(call_index)}:{uid}",
            "call_index": int(call_index),
            "global_step": int(recs["global_step"].iloc[0]),
            "uid": str(uid),
            "data_source": recs["data_source"].iloc[0],
            "correct_count": int((recs["em"] >= 0.5).sum()),
            "wrong_count": int((recs["em"] < 0.5).sum()),
            "logical_search_batches_correct": ";".join(str(int(x)) for x in sorted(cor["logical_search_batches"])),
            "logical_search_batches_all": ";".join(str(int(x)) for x in recs["logical_search_batches"]),
            "search_queries_correct": ";".join(str(int(x)) for x in sorted(cor["search_queries"])),
            "parallel_factor_correct": ";".join(f"{x:.3f}" for x in sorted(cor["parallel_factor"])),
            "response_tokens_correct": ";".join(str(int(x)) for x in sorted(cor["response_token_length"])),
            "turns_correct": ";".join(str(int(x)) for x in sorted(cor["structured_conversation_turns"])),
            "cost_min": int(cor["logical_search_batches"].min()),
            "cost_max": int(cor["logical_search_batches"].max()),
            "cost_gap": int(cor["logical_search_batches"].max() - cor["logical_search_batches"].min()),
            "n_efficient": int(len(eff)),
            "n_inefficient": int(len(ine)),
            "efficient_queries_mean": float(eff["search_queries"].mean()),
            "inefficient_queries_mean": float(ine["search_queries"].mean()),
            "efficient_parallel_factor_mean": float(eff["parallel_factor"].mean()),
            "inefficient_parallel_factor_mean": float(ine["parallel_factor"].mean()),
            "efficient_turns_mean": float(eff["structured_conversation_turns"].mean()),
            "inefficient_turns_mean": float(ine["structured_conversation_turns"].mean()),
            "efficient_response_tokens_mean": float(eff["response_token_length"].mean()),
            "inefficient_response_tokens_mean": float(ine["response_token_length"].mean()),
        })
    active = pd.DataFrame(rows).sort_values(key).reset_index(drop=True)

    g.sort_values(key).to_csv(os.path.join(C.OUTPUTS, "group_table.csv"), index=False)
    active.to_csv(os.path.join(C.OUTPUTS, "active_groups.csv"), index=False)

    uid_sizes = df.groupby("uid").size()
    summary = {
        "total_rollouts": int(len(df)),
        "total_groups": int(len(g)),
        "rollouts_per_group_min": int(g["n_rollouts"].min()),
        "rollouts_per_group_max": int(g["n_rollouts"].max()),
        "groups_with_8_rollouts": int((g["n_rollouts"] == 8).sum()),
        "active_groups": int(g["is_active"].sum()),
        "active_ratio_pct": round(100.0 * float(g["is_active"].mean()), 4),
        "unique_uids": int(df["uid"].nunique()),
        "uid_multiplicity_mean": round(float(uid_sizes.mean()), 3),
        "uid_max_multiplicity": int(uid_sizes.max()),
        "correct_rollouts": int((df["em"] >= 0.5).sum()),
        "correct_rate_pct": round(100.0 * float((df["em"] >= 0.5).mean()), 4),
        "efficient_correct_rollouts": int((df["efficiency_class"] == "efficient_correct").sum()),
        "inefficient_correct_rollouts": int((df["efficiency_class"] == "inefficient_correct").sum()),
        "middle_correct_rollouts": int((df["efficiency_class"] == "middle_correct").sum()),
    }
    return df, g, active, summary


# --------------------------------------------------------------------------- #
# Part 3: cost gap analysis
# --------------------------------------------------------------------------- #
def part3(active):
    os.makedirs(C.FIGURES, exist_ok=True)
    gap = active["cost_gap"].astype(int)
    n = len(active)
    gap_counts = Counter(gap)
    min_counts = Counter(active["cost_min"].astype(int))
    max_counts = Counter(active["cost_max"].astype(int))
    contain = Counter()
    for s in active["logical_search_batches_correct"]:
        for v in set(int(x) for x in s.split(";")):
            contain[v] += 1
    patterns = Counter()
    for s in active["logical_search_batches_correct"]:
        cnt = Counter(int(x) for x in s.split(";"))
        patterns["|".join(f"{k}x{cnt[k]}" for k in sorted(cnt))] += 1

    recs = []
    for v in sorted(gap_counts):
        recs.append(("cost_gap_distribution", str(v), gap_counts[v], 100.0 * gap_counts[v] / n))
    for v in sorted(min_counts):
        recs.append(("min_cost_distribution", str(v), min_counts[v], 100.0 * min_counts[v] / n))
    for v in sorted(max_counts):
        recs.append(("max_cost_distribution", str(v), max_counts[v], 100.0 * max_counts[v] / n))
    for v in sorted(contain):
        recs.append(("cost_contains_distribution", str(v), contain[v], 100.0 * contain[v] / n))
    for p, c in patterns.most_common():
        recs.append(("cost_pair_pattern", p, c, 100.0 * c / n))
    recs.append(("summary", "active_groups", n, 100.0))
    recs.append(("summary", "mean_cost_gap", round(float(gap.mean()), 4), None))
    recs.append(("summary", "median_cost_gap", float(gap.median()), None))
    recs.append(("summary", "max_cost_gap", int(gap.max()), None))
    recs.append(("summary", "distinct_cost_pair_patterns", len(patterns), None))
    long_df = pd.DataFrame(recs, columns=["record_type", "key", "count", "pct_of_active_groups"])
    long_df.to_csv(os.path.join(C.OUTPUTS, "cost_gap_distribution.csv"), index=False)

    wide = pd.DataFrame({
        "cost_gap": sorted(gap_counts),
        "n_groups": [gap_counts[v] for v in sorted(gap_counts)],
        "pct_groups": [round(100.0 * gap_counts[v] / n, 4) for v in sorted(gap_counts)],
    })
    wide.to_csv(os.path.join(C.OUTPUTS, "cost_gap_wide.csv"), index=False)
    pd.DataFrame([(p, c, round(100.0 * c / n, 4)) for p, c in patterns.most_common()],
                 columns=["cost_pattern", "n_groups", "pct_groups"]
                 ).to_csv(os.path.join(C.OUTPUTS, "cost_pair_patterns.csv"), index=False)
    active[["group_id", "uid", "data_source", "n_efficient", "n_inefficient", "cost_min",
            "cost_max", "cost_gap", "logical_search_batches_correct"]
           ].to_csv(os.path.join(C.OUTPUTS, "cost_gap_group_detail.csv"), index=False)
    return wide, long_df


# --------------------------------------------------------------------------- #
# Part 4: dataset / source breakdown
# --------------------------------------------------------------------------- #
def part4(df, g, active):
    act_keys = set(zip(active["call_index"].astype(int), active["uid"].astype(str)))
    rows = []
    for src, sub in g.groupby("data_source"):
        act = sum(1 for ci, u in zip(sub["call_index"], sub["uid"]) if (int(ci), str(u)) in act_keys)
        rsub = df[df["data_source"] == src]
        rows.append({
            "data_source": src,
            "total_groups": int(len(sub)),
            "active_groups": int(act),
            "active_ratio_pct": round(100.0 * act / len(sub), 4),
            "total_rollouts": int(len(rsub)),
            "correct_rollouts": int((rsub["em"] >= 0.5).sum()),
            "rollout_correct_rate_pct": round(100.0 * float((rsub["em"] >= 0.5).mean()), 4),
            "all_correct_groups": int(sub["all_correct"].sum()),
            "all_wrong_groups": int(sub["all_wrong"].sum()),
            "mixed_groups": int(sub["mixed"].sum()),
        })
    rows.append({
        "data_source": "TOTAL",
        "total_groups": int(len(g)),
        "active_groups": int(len(active)),
        "active_ratio_pct": round(100.0 * len(active) / len(g), 4),
        "total_rollouts": int(len(df)),
        "correct_rollouts": int((df["em"] >= 0.5).sum()),
        "rollout_correct_rate_pct": round(100.0 * float((df["em"] >= 0.5).mean()), 4),
        "all_correct_groups": int(g["all_correct"].sum()),
        "all_wrong_groups": int(g["all_wrong"].sum()),
        "mixed_groups": int(g["mixed"].sum()),
    })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(C.OUTPUTS, "source_breakdown.csv"), index=False)
    return out


# --------------------------------------------------------------------------- #
# Parts 5 & 6: reward hacking (query packing) + parallel behaviour
# --------------------------------------------------------------------------- #
METRICS = [
    ("logical_search_batches", "rounds / logical XML search blocks"),
    ("search_queries", "individual queries after the '|' split"),
    ("parallel_factor", "queries per block"),
    ("queries_per_round", "queries per round (alias of parallel_factor)"),
    ("structured_conversation_turns", "assistant turns"),
    ("response_token_length", "response tokens"),
]


def part5_6(df, active):
    cor = df[df["efficiency_class"].isin(["efficient_correct", "inefficient_correct"])]
    eff = cor[cor["efficiency_class"] == "efficient_correct"]
    ine = cor[cor["efficiency_class"] == "inefficient_correct"]

    rows = []
    for col, desc in METRICS:
        e, i = eff[col].astype(float), ine[col].astype(float)
        rows.append({
            "metric": col, "description": desc,
            "efficient_mean": round(float(e.mean()), 4),
            "inefficient_mean": round(float(i.mean()), 4),
            "delta_eff_minus_ineff": round(float(e.mean() - i.mean()), 4),
            "efficient_median": round(float(e.median()), 4),
            "inefficient_median": round(float(i.median()), 4),
            "efficient_p90": round(float(e.quantile(0.9)), 4),
            "inefficient_p90": round(float(i.quantile(0.9)), 4),
            "n_efficient": int(len(e)), "n_inefficient": int(len(i)),
        })
    qr = pd.DataFrame(rows)

    det = []
    packing_suspect = rounds_down = queries_up = 0
    for _, r in active.iterrows():
        er, ir = r["cost_min"], r["cost_max"]
        eq, iq = r["efficient_queries_mean"], r["inefficient_queries_mean"]
        epf, ipf = r["efficient_parallel_factor_mean"], r["inefficient_parallel_factor_mean"]
        rd = bool(er < ir)
        qu = bool(eq >= iq)
        qs = bool(rd and qu)
        packing_suspect += qs
        rounds_down += rd
        queries_up += bool(iq > eq)
        det.append({
            "group_id": r["group_id"], "uid": r["uid"], "data_source": r["data_source"],
            "cost_min": er, "cost_max": ir, "rounds_reduced": rd,
            "efficient_queries_mean": round(eq, 3), "inefficient_queries_mean": round(iq, 3),
            "queries_not_reduced": qu, "queries_increased_in_inefficient": bool(iq > eq),
            "efficient_parallel_factor_mean": round(epf, 3),
            "inefficient_parallel_factor_mean": round(ipf, 3),
            "parallel_packing_increased_in_efficient": bool(epf > ipf),
            "packing_suspect": qs,
        })
    det = pd.DataFrame(det)
    det.to_csv(os.path.join(C.OUTPUTS, "query_packing_group_detail.csv"), index=False)

    verdict = {
        "n_active_groups": int(len(active)),
        "groups_with_rounds_reduced": int(rounds_down),
        "groups_with_queries_increased_in_inefficient": int(queries_up),
        "groups_where_efficient_does_NOT_have_fewer_queries": int(len(active) - queries_up),
        "packing_suspect_groups": int(packing_suspect),
        "packing_suspect_ratio_pct": round(100.0 * packing_suspect / max(len(active), 1), 4),
        "groups_where_efficient_packs_more_queries_per_block": int(
            (det["parallel_packing_increased_in_efficient"]).sum()),
        "efficient_parallel_factor_mean": round(float(eff["parallel_factor"].mean()), 4),
        "inefficient_parallel_factor_mean": round(float(ine["parallel_factor"].mean()), 4),
        "efficient_queries_mean": round(float(eff["search_queries"].mean()), 4),
        "inefficient_queries_mean": round(float(ine["search_queries"].mean()), 4),
        "interpretation": (
            "packing_suspect = efficient correct rollouts use strictly fewer rounds while their "
            "TOTAL query count does not decrease (the round reduction would then be pure query "
            "packing). High value => the round-count reward can be gamed by packing."
        ),
    }
    with open(os.path.join(C.OUTPUTS, "query_packing_verdict.json"), "w") as f:
        json.dump(verdict, f, indent=2)
    qr.to_csv(os.path.join(C.OUTPUTS, "query_round_analysis.csv"), index=False)

    # ---- Part 6: parallel behaviour ----
    classes = {
        "efficient_correct": eff,
        "inefficient_correct": ine,
        "all_correct": df[df["em"] >= 0.5],
        "all_rollouts": df,
        "incorrect": df[df["em"] < 0.5],
    }
    prows, hist = [], {}
    for name, sub in classes.items():
        pf = sub["parallel_factor"].astype(float)
        prows.append({
            "class": name, "n_rollouts": int(len(sub)),
            "parallel_factor_mean": round(float(pf.mean()), 4),
            "parallel_factor_median": round(float(pf.median()), 4),
            "parallel_factor_p90": round(float(pf.quantile(0.9)), 4),
            "parallel_factor_max": round(float(pf.max()), 4),
            "parallel_sample_rate_pct": round(100.0 * float((pf > 1).mean()), 4),
            "multi_query_block_rate_pct": round(100.0 * float((pf >= 2).mean()), 4),
            "output_tokens_mean": round(float(sub["response_token_length"].mean()), 2),
            "turns_mean": round(float(sub["structured_conversation_turns"].mean()), 4),
            "queries_mean": round(float(sub["search_queries"].mean()), 4),
            "rounds_mean": round(float(sub["logical_search_batches"].mean()), 4),
        })
        hist[name] = {str(int(k)): int(v) for k, v in sorted(Counter(pf.astype(int)).items())}
    par = pd.DataFrame(prows)
    par.to_csv(os.path.join(C.OUTPUTS, "parallel_behavior.csv"), index=False)
    with open(os.path.join(C.OUTPUTS, "parallel_behavior_hist.json"), "w") as f:
        json.dump(hist, f, indent=2)
    return qr, par, verdict


# --------------------------------------------------------------------------- #
# Part 7: counterfactual filter replay
# --------------------------------------------------------------------------- #
def part7(df, g, active, lam=LAMBDA):
    grp = {k: v for k, v in df.groupby(["call_index", "uid"], sort=False)}
    rows = []
    for _, r in g.iterrows():
        recs = grp[(r["call_index"], r["uid"])]
        em01 = (recs["em"].to_numpy(dtype=float) >= 0.5).astype(float)
        cost = recs["logical_search_batches"].to_numpy(dtype=float)
        cor_mask = em01 > 0
        active_g = bool(r["is_active"])
        E = np.zeros_like(cost)
        if active_g and cor_mask.sum() >= 2:
            E[cor_mask] = C.efficiency_scores(cost[cor_mask], True)
        R = em01 * (1.0 + lam * E)
        rows.append({
            "group_id": r["group_id"], "call_index": int(r["call_index"]),
            "global_step": int(r["global_step"]), "uid": r["uid"],
            "data_source": r["data_source"], "is_active": active_g,
            "correct_count": int(r["correct_count"]), "all_correct": bool(r["all_correct"]),
            "baseline_keep": C.gap_filter_keep(em01),
            "new_keep": C.gap_filter_keep(R),
        })
    fr = pd.DataFrame(rows)
    fr["newly_added"] = fr["new_keep"] & ~fr["baseline_keep"]
    fr["efficiency_active"] = fr["is_active"] & fr["new_keep"]

    baseline_retained = int(fr["baseline_keep"].sum())
    new_retained = int(fr["new_keep"].sum())
    newly = fr[fr["newly_added"]]
    eff_active_retained = int(fr["efficiency_active"].sum())

    # ---- per-generation-batch accumulation replay ----
    # The trainer iterates generation batches; each batch contributes its kept
    # uids to `num_prompt_in_batch` and it keeps generating until that reaches
    # data.train_batch_size (32), then truncates.  A "call" in the audit IS one
    # generation batch (160 prompts), so this reproduces the real loop exactly.
    per_call = []
    for (step, call), sub in fr.groupby(["global_step", "call_index"]):
        per_call.append({
            "global_step": int(step), "call_index": int(call), "n_groups": int(len(sub)),
            "baseline_kept": int(sub["baseline_keep"].sum()),
            "new_kept": int(sub["new_keep"].sum()),
            "n_active": int(sub["is_active"].sum()),
        })
    pc = pd.DataFrame(per_call).sort_values(["global_step", "call_index"]).reset_index(drop=True)
    pc.to_csv(os.path.join(C.OUTPUTS, "filter_replay_per_call.csv"), index=False)

    sim = []
    for step, sub in pc.groupby("global_step"):
        observed = int(len(sub))          # == the run's train/num_gen_batches for that step
        row = {"global_step": int(step), "observed_calls_in_E1_0": observed}
        for mode, col in (("baseline", "baseline_kept"), ("new_reward", "new_kept")):
            acc, calls = 0, 0
            for v in sub[col]:
                calls += 1
                acc += int(v)
                if acc >= TRAIN_BATCH_SIZE:
                    break
            row[f"{mode}_calls_needed"] = calls
            row[f"{mode}_prompts_collected"] = acc
        row["baseline_reproduces_observed"] = row["baseline_calls_needed"] == observed
        sim.append(row)
    sim_df = pd.DataFrame(sim)
    sim_df.to_csv(os.path.join(C.OUTPUTS, "filter_replay_per_step.csv"), index=False)

    base_total = int(sim_df["baseline_calls_needed"].sum())
    new_total = int(sim_df["new_reward_calls_needed"].sum())
    obs_total = int(sim_df["observed_calls_in_E1_0"].sum())

    out = {
        "lambda": lam,
        "reward_formula": "R = A * (1 + lambda * E); E = (Cmax - C)/(Cmax - Cmin) for correct rollouts of an active group, else 0",
        "filter_simulated": "verl ray_trainer.py:1780-1781 -> kept iff np.std(group_rewards) > 0 or group size == 1",
        "total_groups": int(len(fr)),
        "active_groups": int(fr["is_active"].sum()),
        "baseline_retained": baseline_retained,
        "baseline_retained_ratio_pct": round(100.0 * baseline_retained / len(fr), 4),
        "new_retained": new_retained,
        "new_retained_ratio_pct": round(100.0 * new_retained / len(fr), 4),
        "newly_added_groups": int(len(newly)),
        "newly_added_all_correct_groups": int(newly["all_correct"].sum()),
        "newly_added_by_source": {k: int(v) for k, v in newly["data_source"].value_counts().items()},
        "newly_added_by_correct_count": {str(int(k)): int(v) for k, v in newly["correct_count"].value_counts().sort_index().items()},
        "efficiency_supervision_ratio": round(eff_active_retained / max(new_retained, 1), 6),
        "efficiency_supervision_ratio_pct": round(100.0 * eff_active_retained / max(new_retained, 1), 4),
        "efficiency_supervision_definition": (
            "share of groups retained under the new reward that actually carry a non-zero "
            "efficiency term (active groups); the rest supervise correctness only"
        ),
        "baseline_to_new_retained_multiple": round(new_retained / max(baseline_retained, 1), 4),
        "retained_groups_lost": int((fr["baseline_keep"] & ~fr["new_keep"]).sum()),
        "groups_retained_by_both": int((fr["baseline_keep"] & fr["new_keep"]).sum()),
        "efficiency_active_groups_retained": eff_active_retained,
        "generation_batch_replay": {
            "note": ("sequential accumulation of kept prompts until train_batch_size=32, per global "
                     "step, generation batches consumed in call_index order; one audit call == one "
                     "generation batch of 160 prompts"),
            "observed_calls_in_E1_0": obs_total,
            "baseline_calls_needed_total": base_total,
            "new_calls_needed_total": new_total,
            "baseline_reproduces_observed_all_steps": bool(sim_df["baseline_reproduces_observed"].all()),
            "baseline_per_step_match": {str(int(r.global_step)): bool(r.baseline_reproduces_observed)
                                        for r in sim_df.itertuples()},
            "call_reduction_pct": round(100.0 * (1 - new_total / max(base_total, 1)), 4),
            "per_step": sim,
        },
    }
    with open(os.path.join(C.OUTPUTS, "filter_replay_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    fr.to_csv(os.path.join(C.OUTPUTS, "filter_replay_group_level.csv"), index=False)

    # ---- lambda sensitivity (task asks whether lambda=0.05 is reasonable) ----
    # Since the filter keeps a group iff np.std(reward) > 0, any lambda > 0 that
    # makes an active group's reward vary gives the SAME retained set: lambda
    # changes the advantage magnitude, not the keep/drop decision.
    sens = []
    for lam2 in (0.0, 0.01, 0.05, 0.10, 0.20, 1.0):
        keep_new = 0
        for _, r in g.iterrows():
            recs = grp[(r["call_index"], r["uid"])]
            em01 = (recs["em"].to_numpy(dtype=float) >= 0.5).astype(float)
            cost = recs["logical_search_batches"].to_numpy(dtype=float)
            cor_mask = em01 > 0
            E = np.zeros_like(cost)
            if bool(r["is_active"]) and cor_mask.sum() >= 2:
                E[cor_mask] = C.efficiency_scores(cost[cor_mask], True)
            keep_new += int(C.gap_filter_keep(em01 * (1.0 + lam2 * E)))
        sens.append({
            "lambda": lam2, "baseline_retained": baseline_retained,
            "new_retained": keep_new, "newly_added_groups": keep_new - baseline_retained,
            "retained_set_identical_to_0.05": keep_new == new_retained,
        })
    pd.DataFrame(sens).to_csv(os.path.join(C.OUTPUTS, "filter_replay_lambda_sensitivity.csv"), index=False)
    out["lambda_sensitivity"] = sens
    return out, fr


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def render_figure(wide):
    """Bar chart of the cost-gap distribution. Uses matplotlib if importable in
    this interpreter, else shells out to a conda env that has it."""
    csv = os.path.join(C.OUTPUTS, "cost_gap_wide.csv")
    png = os.path.join(C.FIGURES, "cost_gap_distribution.png")
    code = (
        "import matplotlib; matplotlib.use('Agg');"
        "import matplotlib.pyplot as plt, pandas as pd;"
        f"d=pd.read_csv({csv!r});"
        "fig,ax=plt.subplots(figsize=(7,4.5),dpi=140);"
        "b=ax.bar(d['cost_gap'].astype(str),d['n_groups'],color='#3b6ea5');"
        "ax.bar_label(b,padding=2,fontsize=9);"
        "ax.set_xlabel('cost gap = max-min logical_search_batches over correct rollouts');"
        "ax.set_ylabel('number of active groups');"
        "ax.set_title('E1-0.5 active groups: correct-cost gap distribution (n=%d)'%int(d['n_groups'].sum()));"
        "ax.grid(axis='y',alpha=.3);"
        f"fig.tight_layout();fig.savefig({png!r});print('wrote',{png!r})"
    )
    if os.path.exists(png):
        os.remove(png)
    attempts = [[sys.executable, "-c", code]]
    for p in ("/home/nf5468m6/miniconda3/envs/huatuo/bin/python3",
              "/home/nf5468m6/miniconda3/envs/llama_factory/bin/python3"):
        if os.path.exists(p):
            attempts.append([p, "-c", code])
    last = ""
    for cmd in attempts:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            if r.returncode == 0 and os.path.exists(png):
                return {"figure": png, "ok": True, "renderer": cmd[0],
                        "attempted_interpreters": [a[0] for a in attempts]}
            last = f"{cmd[0]} rc={r.returncode} stderr={r.stderr.strip()[-300:]}"
        except Exception as e:  # noqa: BLE001
            last = f"{cmd[0]} raised {type(e).__name__}: {e}"
    return {"figure": png, "ok": False, "error": last,
            "attempted_interpreters": [a[0] for a in attempts]}


# --------------------------------------------------------------------------- #
def main():
    os.makedirs(C.OUTPUTS, exist_ok=True)
    os.makedirs(C.FIGURES, exist_ok=True)
    df = C.load_rollouts()
    g = C.build_group_table(df)
    df, g, active, s2 = part2(df, g)
    wide, _ = part3(active)
    s4 = part4(df, g, active)
    qr, par, packing = part5_6(df, active)
    s7, fr = part7(df, g, active, lam=LAMBDA)
    fig = render_figure(wide)

    summary = {
        "input_audit": C.audit_path(),
        "lambda": LAMBDA,
        "part2_active_groups": s2,
        "part3_cost_gap": {
            "distribution": dict(zip(wide["cost_gap"].astype(str), wide["n_groups"])),
            "mean_gap": round(float(active["cost_gap"].mean()), 4),
            "max_gap": int(active["cost_gap"].max()),
        },
        "part4_source_breakdown": s4.to_dict(orient="records"),
        "part5_6_packing_verdict": packing,
        "part7_filter_replay": s7,
        "figure": fig,
    }
    with open(os.path.join(C.OUTPUTS, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({k: summary[k] for k in
                      ("part2_active_groups", "part5_6_packing_verdict", "figure")},
                     indent=2, default=str))
    print("\npart7 headline:", json.dumps({k: v for k, v in s7.items()
                                           if not isinstance(v, (dict, list))}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

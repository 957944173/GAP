#!/usr/bin/env python3
"""E1-A training-side diagnostics + comparison against the E1-0 baseline run.

Reads the diagnostics written by the E1-A reward manager
(`<E1A_RUN_DIR>/diagnostics/{rollouts,groups,calls}_pid*.jsonl`) and, for the
matched baseline, the E1-0 audit
(`E1-0/runs/20260910_step60to70_e10/audit/rollouts_pid*.jsonl`).

Both runs are "step60 -> step70, 10 RL steps, identical hyper-parameters"; the only
difference is the reward (original EM vs success-conditioned group-relative efficiency),
so the two are directly comparable.

Outputs (into --out):
    training_summary.json               total/overall metrics for E1-A (+baseline)
    reward_distribution.json            shaped reward + EM distributions
    efficiency_bonus_distribution.json  E distribution / activation
    search_behavior.json                rounds / queries / parallel_factor / turns / tokens
    group_correct_count.json            group_correct_count histograms
    filter_retention.json               baseline vs shaped filter retention
    training_comparison.json/md         E1-A vs E1-0 per-step and overall
"""
import argparse
import glob
import json
import os
import statistics
from collections import Counter, defaultdict

E10_AUDIT = "/data01/wyy/Graph-Agent-Planning/E1-0/runs/20260910_step60to70_e10/audit"


def load_jsonl(pattern):
    rows = []
    for p in sorted(glob.glob(pattern)):
        with open(p, errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    return rows


def pctl(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    return s[min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))]


def hist(vals, buckets=None):
    c = Counter()
    for v in vals:
        if v is None:
            continue
        c[str(v)] += 1
    return {k: c[k] for k in sorted(c, key=lambda x: (float(x) if x.replace('.', '', 1).lstrip('-').isdigit() else 0))}


def summarise(rolls, groups=None, name=""):
    n = len(rolls)
    em = [float(r["em"]) if r.get("em") is not None else None for r in rolls]
    em = [x for x in em if x is not None]
    reward = [float(r["reward"]) if r.get("reward") is not None else
              (float(r["em"]) if r.get("em") is not None else None) for r in rolls]
    reward = [x for x in reward if x is not None]
    eff = [float(r["efficiency"]) if r.get("efficiency") is not None else None for r in rolls]
    eff = [x for x in eff if x is not None]
    rounds = [r["logical_search_batches"] for r in rolls if r.get("logical_search_batches") is not None]
    queries = [r["search_queries"] for r in rolls if r.get("search_queries") is not None]
    pf = [(q / c) if (q and c) else 0.0 for q, c in
          zip([r.get("search_queries") for r in rolls], [r.get("logical_search_batches") for r in rolls])
          if q is not None and c is not None]
    turns = [r.get("conversation_turns", r.get("structured_conversation_turns")) for r in rolls]
    turns = [x for x in turns if x is not None]
    toks = [r.get("response_token_length") for r in rolls if r.get("response_token_length") is not None]
    steps = sorted({r.get("global_step") for r in rolls if r.get("global_step") is not None})
    calls = sorted({(r.get("global_step"), r.get("call_index")) for r in rolls})

    out = {
        "name": name,
        "n_rollouts": n,
        "n_groups": len(set((r.get("global_step"), r.get("call_index"), r.get("uid")) for r in rolls)),
        "n_calls": len(calls),
        "steps": steps,
        "em_rate": (sum(em) / len(em)) if em else None,
        "reward_mean": (sum(reward) / len(reward)) if reward else None,
        "reward_nonzero_rate": (sum(1 for x in reward if x > 0) / len(reward)) if reward else None,
        "reward_hist": hist([round(x, 3) for x in reward]) if reward else {},
        "efficiency_mean": (sum(eff) / len(eff)) if eff else None,
        "efficiency_nonzero_rate": (sum(1 for x in eff if x > 0) / len(eff)) if eff else None,
        "efficiency_hist": hist([round(x, 2) for x in eff]) if eff else {},
        "rounds_mean": (sum(rounds) / len(rounds)) if rounds else None,
        "rounds_median": pctl(rounds, .5),
        "rounds_p90": pctl(rounds, .9),
        "rounds_max": max(rounds) if rounds else None,
        "rounds_hist": hist(rounds),
        "queries_mean": (sum(queries) / len(queries)) if queries else None,
        "queries_p90": pctl(queries, .9),
        "queries_max": max(queries) if queries else None,
        "queries_hist": hist(queries),
        "parallel_factor_mean": (sum(pf) / len(pf)) if pf else None,
        "parallel_factor_p90": pctl(pf, .9),
        "parallel_factor_max": max(pf) if pf else None,
        "parallel_sample_rate": (sum(1 for x in pf if x > 1) / len(pf)) if pf else None,
        "multi_query_round_rate": (sum(1 for q, c in zip(queries, rounds) if q > c) / len(rounds)) if rounds else None,
        "turns_mean": (sum(turns) / len(turns)) if turns else None,
        "tokens_mean": (sum(toks) / len(toks)) if toks else None,
        "zero_search_rate": (sum(1 for c in rounds if c == 0) / len(rounds)) if rounds else None,
    }
    if groups:
        cc = [g.get("group_correct_count") for g in groups if g.get("group_correct_count") is not None]
        out["group_correct_count_hist"] = hist(cc)
        out["group_correct_count_mean"] = (sum(cc) / len(cc)) if cc else None
        base = sum(1 for g in groups if g.get("baseline_keep_em_std_gt_0"))
        shaped = sum(1 for g in groups if g.get("shaped_keep_reward_std_gt_0"))
        newly = sum(1 for g in groups if g.get("newly_added_by_shaping"))
        active = sum(1 for g in groups if g.get("efficiency_active"))
        out["filter_retention_baseline"] = base
        out["filter_retention_shaped"] = shaped
        out["filter_newly_added"] = newly
        out["efficiency_active_groups"] = active
        out["efficiency_supervision_ratio"] = (active / shaped) if shaped else None
    return out


def per_step_table(rolls, groups, key="global_step"):
    gby = defaultdict(list)
    for g in groups:
        gby[g.get(key)].append(g)
    rby = defaultdict(list)
    for r in rolls:
        rby[r.get(key)].append(r)
    rows = []
    for step in sorted(x for x in rby if x is not None):
        rr, gg = rby[step], gby.get(step, [])
        em = [float(r["em"]) for r in rr if r.get("em") is not None]
        rw = [float(r["reward"]) for r in rr if r.get("reward") is not None]
        eff = [float(r["efficiency"]) for r in rr if r.get("efficiency") is not None]
        rounds = [r["logical_search_batches"] for r in rr if r.get("logical_search_batches") is not None]
        q = [r["search_queries"] for r in rr if r.get("search_queries") is not None]
        rows.append({
            "global_step": step,
            "n_rollouts": len(rr), "n_groups": len(gg),
            "em_rate": (sum(em) / len(em)) if em else None,
            "reward_mean": (sum(rw) / len(rw)) if rw else None,
            "efficiency_mean": (sum(eff) / len(eff)) if eff else None,
            "efficiency_nonzero_rate": (sum(1 for x in eff if x > 0) / len(eff)) if eff else None,
            "rounds_mean": (sum(rounds) / len(rounds)) if rounds else None,
            "queries_mean": (sum(q) / len(q)) if q else None,
            "baseline_kept": sum(1 for g in gg if g.get("baseline_keep_em_std_gt_0")),
            "shaped_kept": sum(1 for g in gg if g.get("shaped_keep_reward_std_gt_0")),
            "newly_added": sum(1 for g in gg if g.get("newly_added_by_shaping")),
            "efficiency_active": sum(1 for g in gg if g.get("efficiency_active")),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="E1-A run dir containing diagnostics/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline-audit", default=E10_AUDIT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rolls = load_jsonl(os.path.join(args.run_dir, "diagnostics", "rollouts_pid*.jsonl"))
    groups = load_jsonl(os.path.join(args.run_dir, "diagnostics", "groups_pid*.jsonl"))
    calls = load_jsonl(os.path.join(args.run_dir, "diagnostics", "calls_pid*.jsonl"))
    if not rolls:
        raise SystemExit(f"no E1-A diagnostics found under {args.run_dir}/diagnostics")

    e1a = summarise(rolls, groups, "E1-A shaped reward")

    base_rolls = load_jsonl(os.path.join(args.baseline_audit, "rollouts_pid*.jsonl"))
    baseline = summarise(base_rolls, None, "baseline E1-0 (original EM)") if base_rolls else None
    if base_rolls:
        gmap = defaultdict(list)
        for r in base_rolls:
            gmap[(r.get("global_step"), r.get("call_index"), r.get("uid"))].append(r)
        bkeep = 0
        active = 0
        cc_hist = Counter()
        for _k, recs in gmap.items():
            em = [1.0 if (x.get("em") == 1 or x.get("em") == 1.0) else 0.0 for x in recs]
            cost = [x.get("logical_search_batches") for x in recs]
            cc_hist[int(sum(em))] += 1
            if len(set(em)) > 1:
                bkeep += 1
            cc = [c for c, a in zip(cost, em) if a == 1.0 and c is not None]
            if len(cc) >= 2 and len(set(cc)) > 1:
                active += 1
        baseline["filter_retention_baseline"] = bkeep
        baseline["efficiency_active_groups"] = active
        baseline["baseline_kept_group_ratio"] = bkeep / max(len(gmap), 1)
        baseline["n_groups"] = len(gmap)
        baseline["group_correct_count_hist"] = {str(k): cc_hist[k] for k in sorted(cc_hist)}
        baseline["group_correct_count_mean"] = (
            sum(k * v for k, v in cc_hist.items()) / max(sum(cc_hist.values()), 1))

    # ---- required distributions ----
    reward_dist = {
        "e1a_reward_hist": e1a["reward_hist"],
        "e1a_em_rate": e1a["em_rate"],
        "e1a_reward_nonzero_rate": e1a["reward_nonzero_rate"],
        "note": "reward = A*(1+0.05*E); E=0 => reward == EM (0 or 1); active correct rollouts get 1.05/1.0333/... depending on cost",
        "baseline_em_rate": (baseline or {}).get("em_rate"),
    }
    eff_dist = {
        "e1a_efficiency_hist": e1a["efficiency_hist"],
        "e1a_efficiency_mean": e1a["efficiency_mean"],
        "e1a_efficiency_nonzero_rate": e1a["efficiency_nonzero_rate"],
        "e1a_efficiency_active_groups": e1a.get("efficiency_active_groups"),
    }
    search_behavior = {
        "e1a": {k: e1a[k] for k in ("rounds_mean", "rounds_median", "rounds_p90", "rounds_max",
                                     "rounds_hist", "queries_mean", "queries_p90", "queries_max",
                                     "queries_hist", "parallel_factor_mean", "parallel_factor_p90",
                                     "parallel_factor_max", "parallel_sample_rate",
                                     "multi_query_round_rate", "turns_mean", "tokens_mean",
                                     "zero_search_rate")},
        "baseline": ({k: baseline[k] for k in ("rounds_mean", "rounds_median", "rounds_p90", "rounds_max",
                                                "rounds_hist", "queries_mean", "queries_p90", "queries_max",
                                                "queries_hist", "parallel_factor_mean", "parallel_factor_p90",
                                                "parallel_factor_max", "parallel_sample_rate",
                                                "multi_query_round_rate", "turns_mean", "tokens_mean",
                                                "zero_search_rate")} if baseline else None),
        "delta_e1a_minus_baseline": ({k: (e1a[k] - baseline[k]) for k in
                                      ("rounds_mean", "queries_mean", "parallel_factor_mean", "turns_mean",
                                       "tokens_mean", "parallel_sample_rate", "zero_search_rate")
                                      if e1a.get(k) is not None and baseline.get(k) is not None}
                                     if baseline else None),
    }
    group_cc = {"e1a_group_correct_count_hist": e1a.get("group_correct_count_hist"),
                "e1a_group_correct_count_mean": e1a.get("group_correct_count_mean"),
                "baseline_group_correct_count_hist": (baseline or {}).get("group_correct_count_hist")}
    retention = {
        "e1a_baseline_kept_groups": e1a.get("filter_retention_baseline"),
        "e1a_shaped_kept_groups": e1a.get("filter_retention_shaped"),
        "e1a_newly_added_groups": e1a.get("filter_newly_added"),
        "e1a_efficiency_active_groups": e1a.get("efficiency_active_groups"),
        "e1a_efficiency_supervision_ratio": e1a.get("efficiency_supervision_ratio"),
        "baseline_kept_groups_recomputed": (baseline or {}).get("filter_retention_baseline"),
        "baseline_active_groups_recomputed": (baseline or {}).get("efficiency_active_groups"),
        "expected_newly_added_from_baseline": (baseline or {}).get("efficiency_active_groups"),
        "note": ("retention simulated exactly like ray_trainer.py:1780-1781 "
                 "(kept iff np.std(group reward) > 0 or |group| == 1) on the full pre-filter batch"),
    }

    # ---- safety analysis (query packing / zero-search / malformed / reward dist) ----
    def packing_stats(rolls_, label):
        gmap = defaultdict(list)
        for r in rolls_:
            gmap[(r.get("global_step"), r.get("call_index"), r.get("uid"))].append(r)
        eff, ineff = [], []
        suspects = 0
        active = 0
        for _k, recs in gmap.items():
            cor = [r for r in recs if (r.get("em") == 1 or r.get("em") == 1.0)]
            if len(cor) < 2:
                continue
            costs = [r.get("logical_search_batches") for r in cor]
            cmin, cmax = min(costs), max(costs)
            if cmax <= cmin:
                continue
            active += 1
            e = [r for r in cor if r.get("logical_search_batches") == cmin]
            i = [r for r in cor if r.get("logical_search_batches") == cmax]
            eff += e
            ineff += i
            eq = statistics.fmean([r.get("search_queries") or 0 for r in e]) if e else 0
            iq = statistics.fmean([r.get("search_queries") or 0 for r in i]) if i else 0
            if eq >= iq:
                suspects += 1

        def m(rows, key, default=None):
            vals = [r.get(key) for r in rows if r.get(key) is not None]
            return statistics.fmean(vals) if vals else default
        return {
            "label": label,
            "active_groups": active,
            "efficient_correct_rollouts": len(eff),
            "inefficient_correct_rollouts": len(ineff),
            "efficient_rounds_mean": m(eff, "logical_search_batches"),
            "inefficient_rounds_mean": m(ineff, "logical_search_batches"),
            "efficient_queries_mean": m(eff, "search_queries"),
            "inefficient_queries_mean": m(ineff, "search_queries"),
            "efficient_parallel_factor_mean": (m(eff, "search_queries") / m(eff, "logical_search_batches")
                                               if m(eff, "logical_search_batches") else None),
            "inefficient_parallel_factor_mean": (m(ineff, "search_queries") / m(ineff, "logical_search_batches")
                                                 if m(ineff, "logical_search_batches") else None),
            "efficient_tokens_mean": m(eff, "response_token_length"),
            "inefficient_tokens_mean": m(ineff, "response_token_length"),
            "efficient_turns_mean": m(eff, "conversation_turns", m(eff, "structured_conversation_turns")),
            "inefficient_turns_mean": m(ineff, "conversation_turns", m(ineff, "structured_conversation_turns")),
            "packing_suspect_groups": suspects,
            "packing_suspect_ratio": (suspects / active) if active else None,
        }

    safety = {
        "query_packing": {
            "e1a": packing_stats(rolls, "E1-A"),
            "baseline": packing_stats(base_rolls, "baseline E1-0") if base_rolls else None,
            "definition": ("packing_suspect = within an active (>=2 correct, cost-varying) group the "
                           "efficient correct rollouts use fewer rounds but NOT fewer total queries"),
        },
        "zero_search": {
            "e1a_zero_search_rate": e1a.get("zero_search_rate"),
            "baseline_zero_search_rate": (baseline or {}).get("zero_search_rate"),
            "e1a_zero_search_correct_rate": (
                sum(1 for r in rolls if (r.get("logical_search_batches") == 0
                                          and (r.get("em") == 1 or r.get("em") == 1.0)))
                / max(sum(1 for r in rolls if (r.get("em") == 1 or r.get("em") == 1.0)), 1)),
        },
        "malformed": {
            "e1a_metadata_invalid_rate": (
                sum(1 for r in rolls if not r.get("metadata_valid")) / max(len(rolls), 1)),
            "complete_reason_top": Counter(
                str(r.get("complete_reason"))[:80] for r in rolls).most_common(5),
        },
        "reward_distribution": reward_dist,
    }

    summary = {"e1a": e1a, "baseline": baseline,
               "e1a_per_step": per_step_table(rolls, groups),
               "e1a_calls": calls}
    for name, obj in (("training_summary.json", summary),
                      ("reward_distribution.json", reward_dist),
                      ("efficiency_bonus_distribution.json", eff_dist),
                      ("search_behavior.json", search_behavior),
                      ("group_correct_count.json", group_cc),
                      ("filter_retention.json", retention),
                      ("safety_analysis.json", safety),
                      ("training_comparison.json", {"e1a": e1a, "baseline": baseline,
                                                     "per_step_e1a": summary["e1a_per_step"]})):
        with open(os.path.join(args.out, name), "w") as f:
            json.dump(obj, f, indent=2, default=str)

    # human readable
    def fmt(v, n=4):
        return f"{v:.{n}f}" if isinstance(v, float) else str(v)
    lines = ["# E1-A training-side diagnostics (step 60 -> 70)", ""]
    lines.append("| metric | E1-A (shaped) | baseline E1-0 (original EM) | delta |")
    lines.append("|---|---|---|---|")
    keys = ["n_rollouts", "n_groups", "n_calls", "em_rate", "reward_mean", "efficiency_mean",
            "efficiency_nonzero_rate", "rounds_mean", "queries_mean", "parallel_factor_mean",
            "parallel_sample_rate", "multi_query_round_rate", "turns_mean", "tokens_mean",
            "zero_search_rate", "filter_retention_baseline", "filter_retention_shaped",
            "filter_newly_added", "efficiency_active_groups", "efficiency_supervision_ratio"]
    for k in keys:
        a = e1a.get(k)
        b = (baseline or {}).get(k)
        d = (a - b) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else ""
        lines.append(f"| {k} | {fmt(a)} | {fmt(b)} | {fmt(d)} |")
    lines.append("")
    lines.append("## per-step (E1-A)")
    lines.append("| step | rollouts | groups | EM | reward mean | E mean | E>0 rate | rounds | queries | base kept | shaped kept | newly added |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in summary["e1a_per_step"]:
        lines.append("| {global_step} | {n_rollouts} | {n_groups} | {em_rate} | {reward_mean} | "
                     "{efficiency_mean} | {efficiency_nonzero_rate} | {rounds_mean} | {queries_mean} | "
                     "{baseline_kept} | {shaped_kept} | {newly_added} |".format(
                         **{k: (fmt(v) if isinstance(v, float) else v) for k, v in r.items()}))
    with open(os.path.join(args.out, "training_diagnostics.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print(json.dumps({"e1a": {k: e1a.get(k) for k in
                              ("n_rollouts", "n_groups", "n_calls", "steps", "em_rate", "reward_mean",
                               "efficiency_mean", "efficiency_nonzero_rate", "rounds_mean", "queries_mean",
                               "parallel_factor_mean", "turns_mean", "tokens_mean", "zero_search_rate",
                               "filter_retention_baseline", "filter_retention_shaped", "filter_newly_added",
                               "efficiency_active_groups", "efficiency_supervision_ratio")},
                      "baseline": {k: (baseline or {}).get(k) for k in
                                   ("n_rollouts", "n_groups", "em_rate", "rounds_mean", "queries_mean",
                                    "parallel_factor_mean", "turns_mean", "tokens_mean", "zero_search_rate",
                                    "filter_retention_baseline", "efficiency_active_groups")}},
                     indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

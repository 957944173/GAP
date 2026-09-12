#!/usr/bin/env python3
"""E1-C main offline analysis driver (Analyses A-G + sanity checks).

Run:
    python3 E1-C/code/e1c_run_all.py

Everything written lands under E1-C/.  E1-0 / E1-0.5 / E1-A / E1-B are only read.
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e1c_common as C  # noqa: E402

PRIMARY_POOL = "E1_B"
LAMBDAS = [0.0, 0.01, 0.025, 0.05, 0.1, 0.2]
LAMBDA_MAIN = 0.05
METRIC_COLS = ["correct_count", "group_success_rate", "cost_all_mean", "cost_correct_mean",
               "cost_gap_correct", "queries_all_mean", "tokens_all_mean", "turns_all_mean"]


def log(msg, fh=None):
    line = f"[e1c] {msg}"
    print(line, flush=True)
    if fh:
        fh.write(line + "\n")
        fh.flush()


def group_vectors(df):
    vec = {}
    for (ci, uid), recs in df.groupby(["call_index", "uid"], sort=False):
        vec[(int(ci), str(uid))] = (recs["correct"].to_numpy(dtype=float),
                                    recs["logical_search_batches"].to_numpy(dtype=float))
    return vec


def keep_map_for_lambda(df, lam):
    """{group_id: bool} for the filter under reward R(λ). λ=0 is the GAP filter."""
    out = {}
    for (ci, uid), recs in df.groupby(["call_index", "uid"], sort=False):
        correct = recs["correct"].to_numpy(dtype=float)
        cost = recs["logical_search_batches"].to_numpy(dtype=float)
        if lam == 0.0:
            out[f"{int(ci)}:{uid}"] = C.group_kept(correct)
        else:
            R, _ = C.reward_vector(correct, cost, lam)
            out[f"{int(ci)}:{uid}"] = C.group_kept(R)
    return out


# --------------------------------------------------------------------------- #
# Analysis F helper: parse the trainer console for the real generation loop
# --------------------------------------------------------------------------- #
KEPT_RE = re.compile(r"len\(kept_prompt_uids\)\s*-\s*(\d+)")
STEP_RE = re.compile(r"step:(\d+)\s+-")


def parse_generation_loop(path):
    """{step: [kept counts per generation batch]} from a trainer console log.

    For every generation batch the trainer prints `len(kept_prompt_uids) - N`;
    the `step:N - ...critic/score/mean...` metric line closes the optimizer
    step, so all kept-prints since the previous metric line belong to it.
    """
    out = {}
    pending = []
    with open(path, errors="ignore") as fh:
        for line in fh:
            line = re.sub(r"\x1b\[[0-9;]*m", "", line)
            m = KEPT_RE.search(line)
            if m:
                pending.append(int(m.group(1)))
                continue
            s = STEP_RE.search(line)
            if s and "critic/score/mean" in line:
                step = int(s.group(1))
                if pending:
                    out.setdefault(step, []).extend(pending)
                    pending = []
    return out


def parse_calls_per_step(pattern):
    c = Counter()
    for p in sorted(glob.glob(pattern)):
        with open(p, errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("global_step") is not None:
                    c[int(r["global_step"])] += 1
    return dict(c)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=C.RESULTS)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(C.LOGS, exist_ok=True)
    runlog = open(os.path.join(C.LOGS, "run.log"), "a")
    log(f"=== E1-C run {pd.Timestamp.now().isoformat()} ===", runlog)

    target, target_src = C.train_batch_size()
    log(f"target optimizer batch size (data.train_batch_size) = {target} [source: {target_src}]", runlog)

    pools = {}
    for name in ("E1_0", "E1_A", "E1_B"):
        df = C.load_pool(name)
        g = C.build_group_table(df)
        p = {"df": df, "g": g, "vec": group_vectors(df)}
        p["gap_keep"] = keep_map_for_lambda(df, 0.0)
        p["e1_keep"] = keep_map_for_lambda(df, LAMBDA_MAIN)
        pools[name] = p
        log(f"pool {name}: {len(g):,} groups, size histogram={dict(Counter(g['n_rollouts']))}", runlog)

    results = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "target_train_batch_size": target, "target_source": target_src,
        "lambda_main": LAMBDA_MAIN, "lambdas": LAMBDAS,
    }

    # ======================= Analysis A =====================================
    log("--- Analysis A: same-pool retained-set relationship", runlog)
    pool_A = {}
    for name, p in pools.items():
        g = p["g"]
        ids = list(g["group_id"])
        S_gap = {i for i in ids if p["gap_keep"][i]}
        S_e1 = {i for i in ids if p["e1_keep"][i]}
        inter = S_gap & S_e1
        rel = C.classify_set_relation(len(S_gap), len(S_e1), len(inter))
        pool_A[name] = {
            "pool_label": C.POOLS[name]["label"],
            "total_candidate_groups": len(ids),
            "S_GAP": len(S_gap), "S_E1": len(S_e1), "intersection": len(inter),
            "GAP_only": len(S_gap - S_e1), "E1_only": len(S_e1 - S_gap),
            "union": len(S_gap | S_e1),
            "jaccard": C.jaccard(S_gap, S_e1),
            "containment_inter_over_GAP": len(inter) / max(len(S_gap), 1),
            "containment_inter_over_E1": len(inter) / max(len(S_e1), 1),
            "relation": rel,
        }
        log(f"  [{name}] |S_GAP|={len(S_gap)} |S_E1|={len(S_e1)} inter={len(inter)} "
            f"gap_only={len(S_gap - S_e1)} e1_only={len(S_e1 - S_gap)} "
            f"jaccard={C.jaccard(S_gap, S_e1):.4f} -> {rel}", runlog)
    results["same_pool_sets"] = pool_A

    gp, km = pools[PRIMARY_POOL]["g"], pools[PRIMARY_POOL]
    det = gp[["group_id", "call_index", "global_step", "uid", "data_source", "n_rollouts",
              "correct_count", "all_wrong", "all_correct", "mixed", "group_success_rate",
              "cost_all_mean", "cost_correct_mean", "cost_correct_min", "cost_correct_max",
              "cost_gap_correct", "queries_all_mean", "turns_all_mean",
              "tokens_all_mean", "is_active_efficiency"]].copy()
    det["keep_GAP"] = det["group_id"].map(km["gap_keep"])
    det["keep_E1_lambda0.05"] = det["group_id"].map(km["e1_keep"])
    det["is_E1_only"] = det["keep_E1_lambda0.05"] & ~det["keep_GAP"]
    det["is_GAP_only"] = det["keep_GAP"] & ~det["keep_E1_lambda0.05"]
    det["in_intersection"] = det["keep_GAP"] & det["keep_E1_lambda0.05"]
    det.to_csv(os.path.join(args.out, "same_pool_set_relationship.csv"), index=False)
    det[det["is_GAP_only"]].to_csv(os.path.join(args.out, "same_pool_gap_only.csv"), index=False)
    det[det["is_E1_only"]].to_csv(os.path.join(args.out, "same_pool_e1_only.csv"), index=False)
    det[det["in_intersection"]].to_csv(os.path.join(args.out, "same_pool_intersection.csv"), index=False)
    pd.DataFrame([{"pool": k, **v} for k, v in pool_A.items()]).to_csv(
        os.path.join(args.out, "same_pool_set_summary.csv"), index=False)

    # ======================= Analysis B =====================================
    log("--- Analysis B: group types of E1-only / GAP-only", runlog)
    rows = []
    for name, p in pools.items():
        sub = p["g"].copy()
        sub["keep_GAP"] = sub["group_id"].map(p["gap_keep"])
        sub["keep_E1"] = sub["group_id"].map(p["e1_keep"])
        for label, mask in (("candidates", np.ones(len(sub), dtype=bool)),
                            ("S_GAP", sub["keep_GAP"].to_numpy()),
                            ("S_E1", sub["keep_E1"].to_numpy()),
                            ("intersection", (sub["keep_GAP"] & sub["keep_E1"]).to_numpy()),
                            ("E1_only", (sub["keep_E1"] & ~sub["keep_GAP"]).to_numpy()),
                            ("GAP_only", (sub["keep_GAP"] & ~sub["keep_E1"]).to_numpy())):
            s = sub[mask]
            if len(s) == 0:
                rows.append({"pool": name, "set": label, "n_groups": 0})
                continue
            rows.append({
                "pool": name, "set": label, "n_groups": len(s),
                "all_wrong": int(s["all_wrong"].sum()), "mixed": int(s["mixed"].sum()),
                "all_correct": int(s["all_correct"].sum()),
                "all_correct_rate": float(s["all_correct"].mean()),
                "mixed_rate": float(s["mixed"].mean()),
                "mean_group_success_rate": float(s["group_success_rate"].mean()),
                "mean_correct_count": float(s["correct_count"].mean()),
                "mean_cost_all": float(s["cost_all_mean"].mean()),
                "mean_cost_gap_correct": float(s["cost_gap_correct"].mean()),
                "active_efficiency_groups": int(s["is_active_efficiency"].sum()),
                "mean_queries": float(s["queries_all_mean"].mean()),
                "mean_tokens": float(s["tokens_all_mean"].mean()),
                "mean_turns": float(s["turns_all_mean"].mean()),
                "top_sources": "; ".join(f"{k2}:{v2}" for k2, v2 in
                                         s["data_source"].value_counts().head(5).items()),
            })
    gt = pd.DataFrame(rows)
    gt.to_csv(os.path.join(args.out, "group_type_summary.csv"), index=False)
    det[det["is_E1_only"]].to_csv(os.path.join(args.out, "e1_only_details.csv"), index=False)
    det[det["is_GAP_only"]].to_csv(os.path.join(args.out, "gap_only_details.csv"), index=False)
    e1o = det[det["is_E1_only"]]
    if len(e1o):
        e1o.groupby("data_source").agg(
            n_groups=("group_id", "size"), all_correct=("all_correct", "sum"),
            mean_group_success_rate=("group_success_rate", "mean"),
            mean_cost_gap=("cost_gap_correct", "mean")).reset_index().to_csv(
            os.path.join(args.out, "e1_only_source_breakdown.csv"), index=False)

    # ======================= Analysis C =====================================
    log("--- Analysis C: sequential fixed-size batch-fill replay", runlog)
    batchfill = {}
    for name, p in pools.items():
        gap = C.replay_batchfill(p["g"], p["vec"], lam=0.0, target=target, mode="GAP")
        e1 = C.replay_batchfill(p["g"], p["vec"], lam=LAMBDA_MAIN, target=target, mode="E1")
        batchfill[name] = {"GAP": gap, "E1": e1}
        gs = {r["global_step"]: r for r in gap}
        es = {r["global_step"]: r for r in e1}
        recs = []
        for step in sorted(gs):
            G, E = gs[step], es[step]
            sg, se = set(G["selected"]), set(E["selected"])
            recs.append({
                "global_step": step,
                "calls_available": G["n_generation_batches_available"],
                "GAP_calls_needed": G["calls_needed"], "E1_calls_needed": E["calls_needed"],
                "calls_saved": G["calls_needed"] - E["calls_needed"],
                "GAP_selected": G["selected_count"], "E1_selected": E["selected_count"],
                "selected_intersection": len(sg & se),
                "GAP_selected_only": len(sg - se), "E1_selected_only": len(se - sg),
                "selected_jaccard": C.jaccard(sg, se),
                "GAP_kept_all_calls": G["total_kept_if_all_calls"],
                "E1_kept_all_calls": E["total_kept_if_all_calls"],
            })
        bf = pd.DataFrame(recs)
        if name == PRIMARY_POOL:
            bf.to_csv(os.path.join(args.out, "batchfill_overlap_by_step.csv"), index=False)
            disp, add = [], []
            for row in bf.itertuples(index=False):
                sg = set(gs[row.global_step]["selected"])
                se = set(es[row.global_step]["selected"])
                disp += [{"global_step": row.global_step, "group_id": gid} for gid in sorted(sg - se)]
                add += [{"global_step": row.global_step, "group_id": gid} for gid in sorted(se - sg)]
            pd.DataFrame(disp).to_csv(os.path.join(args.out, "batchfill_gap_only.csv"), index=False)
            pd.DataFrame(add).to_csv(os.path.join(args.out, "batchfill_e1_only.csv"), index=False)
        tot_gap = sum(r["GAP_selected"] for r in recs)
        tot_e1 = sum(r["E1_selected"] for r in recs)
        inter = sum(r["selected_intersection"] for r in recs)
        gonly = sum(r["GAP_selected_only"] for r in recs)
        eonly = sum(r["E1_selected_only"] for r in recs)
        agg = {
            "GAP_selected_total": tot_gap, "E1_selected_total": tot_e1, "intersection": inter,
            "GAP_selected_only": gonly, "E1_selected_only": eonly,
            "jaccard_overall": inter / max(tot_gap + eonly, 1),
            "GAP_calls_needed_total": sum(r["calls_needed"] for r in gap),
            "E1_calls_needed_total": sum(r["calls_needed"] for r in e1),
            "call_reduction_pct": 100.0 * (1 - sum(r["calls_needed"] for r in e1)
                                           / max(sum(r["calls_needed"] for r in gap), 1)),
            "steps": len(recs),
            "steps_with_identical_selection": int(sum(1 for r in recs if r["selected_jaccard"] == 1.0)),
        }
        agg["relation"] = C.classify_set_relation(tot_gap, tot_e1, inter)
        # ---- censoring diagnostics (critical on pools generated under the E1 filter):
        # the counterfactual GAP arm cannot reach the target within the batches the E1
        # run actually produced, so its "calls_needed" is right-censored.
        gap_calls_avail = sum(r["calls_available"] for r in recs)
        agg["GAP_censored_steps"] = int(sum(1 for r in recs if r["GAP_selected"] < target))
        agg["E1_censored_steps"] = int(sum(1 for r in recs if r["E1_selected"] < target))
        agg["target_slots"] = len(recs) * target
        agg["GAP_kept_per_batch"] = agg["GAP_selected_total"] / max(gap_calls_avail, 1)
        agg["E1_kept_per_batch"] = agg["E1_selected_total"] / max(gap_calls_avail, 1)
        agg["fill_rate_ratio_E1_over_GAP"] = (agg["E1_kept_per_batch"] / agg["GAP_kept_per_batch"]
                                              if agg["GAP_kept_per_batch"] else None)
        # censoring-free estimate of how many batches each filter would need, using the
        # per-step observed keep rate (INFERENCE, not an observation)
        est = 0.0
        for r in recs:
            rate = r["GAP_kept_all_calls"] / max(r["calls_available"], 1)
            est += target / rate if rate > 0 else float("inf")
        agg["GAP_calls_needed_projected_from_keeprate"] = est
        results[f"batchfill_{name}"] = agg
        log(f"  [{name}] GAP calls={agg['GAP_calls_needed_total']} E1 calls={agg['E1_calls_needed_total']} "
            f"({agg['call_reduction_pct']:.1f}% fewer) | selected GAP={tot_gap} E1={tot_e1} inter={inter} "
            f"gap_only={gonly} e1_only={eonly} -> {agg['relation']}", runlog)
    allbf = []
    for name in pools:
        for mode in ("GAP", "E1"):
            for r in batchfill[name][mode]:
                allbf.append({"pool": name, "mode": mode, "global_step": r["global_step"],
                              "calls_needed": r["calls_needed"],
                              "n_generation_batches_available": r["n_generation_batches_available"],
                              "kept_in_prefix": r["kept_in_prefix"], "selected_count": r["selected_count"],
                              "total_kept_if_all_calls": r["total_kept_if_all_calls"]})
    pd.DataFrame(allbf).to_csv(os.path.join(args.out, "batchfill_all_pools_by_step.csv"), index=False)

    # ======================= Analysis D =====================================
    log("--- Analysis D: difficulty of E1-added vs GAP-displaced groups", runlog)
    disp_ids = set(pd.read_csv(os.path.join(args.out, "batchfill_gap_only.csv"))["group_id"])
    add_ids = set(pd.read_csv(os.path.join(args.out, "batchfill_e1_only.csv"))["group_id"])
    D = det[det["group_id"].isin(disp_ids | add_ids)].copy()
    D["cohort"] = np.where(D["group_id"].isin(add_ids), "E1_added", "GAP_displaced")
    D.to_csv(os.path.join(args.out, "batchfill_cohort_members.csv"), index=False)

    # per-pool cohorts (the E1-0 pool is the uncensored counterfactual, the E1-B pool the
    # censored one) so the difficulty comparison is not tied to a single pool
    cohort_by_pool = {}
    for name, p in pools.items():
        gsel, esel = {}, {}
        gs = {r["global_step"]: r for r in batchfill[name]["GAP"]}
        es = {r["global_step"]: r for r in batchfill[name]["E1"]}
        add_ids_p, disp_ids_p = set(), set()
        for step in gs:
            a, b = set(gs[step]["selected"]), set(es[step]["selected"])
            disp_ids_p |= (a - b)
            add_ids_p |= (b - a)
        cohort_by_pool[name] = (disp_ids_p, add_ids_p)
        dd_p = p["g"][p["g"]["group_id"].isin(disp_ids_p | add_ids_p)].copy()
        dd_p["cohort"] = np.where(dd_p["group_id"].isin(add_ids_p), "E1_added", "GAP_displaced")
        dd_p.to_csv(os.path.join(args.out, f"batchfill_cohort_members_{name}.csv"), index=False)

    d_rows = []
    for metric in METRIC_COLS:
        a = D.loc[D.cohort == "E1_added", metric].to_numpy(dtype=float)
        b = D.loc[D.cohort == "GAP_displaced", metric].to_numpy(dtype=float)
        t = C.mannwhitney_u(a, b)
        d_rows.append({"metric": metric, "E1_added_mean": float(np.mean(a)) if len(a) else None,
                       "GAP_displaced_mean": float(np.mean(b)) if len(b) else None,
                       "E1_added_median": float(np.median(a)) if len(a) else None,
                       "GAP_displaced_median": float(np.median(b)) if len(b) else None,
                       "mannwhitney_p": t["p"], "n_E1_added": len(a), "n_GAP_displaced": len(b)})
    for rate, col in (("all_correct_rate", "all_correct"), ("mixed_rate", "mixed"),
                      ("all_wrong_rate", "all_wrong")):
        k1 = int(D.loc[D.cohort == "E1_added", col].sum()); n1 = int((D.cohort == "E1_added").sum())
        k2 = int(D.loc[D.cohort == "GAP_displaced", col].sum()); n2 = int((D.cohort == "GAP_displaced").sum())
        t = C.two_prop_z(k1, n1, k2, n2)
        d_rows.append({"metric": rate, "E1_added_mean": k1 / max(n1, 1), "GAP_displaced_mean": k2 / max(n2, 1),
                       "E1_added_median": None, "GAP_displaced_median": None,
                       "mannwhitney_p": t["p"], "n_E1_added": n1, "n_GAP_displaced": n2})
    for r in d_rows:
        r["pool"] = PRIMARY_POOL
    # per-pool summary rows (counts + success-rate + all-correct rate), pandas-readable
    pool_rows = []
    for name, (disp_ids_p, add_ids_p) in cohort_by_pool.items():
        gp2 = pools[name]["g"]
        A = gp2[gp2["group_id"].isin(add_ids_p)]
        B = gp2[gp2["group_id"].isin(disp_ids_p)]
        pool_rows.append({
            "pool": name, "n_E1_added": len(A), "n_GAP_displaced": len(B),
            "E1_added_correct_count_mean": float(A["correct_count"].mean()) if len(A) else None,
            "GAP_displaced_correct_count_mean": float(B["correct_count"].mean()) if len(B) else None,
            "E1_added_success_rate_mean": float(A["group_success_rate"].mean()) if len(A) else None,
            "GAP_displaced_success_rate_mean": float(B["group_success_rate"].mean()) if len(B) else None,
            "E1_added_all_correct_rate": float(A["all_correct"].mean()) if len(A) else None,
            "GAP_displaced_mixed_rate": float(B["mixed"].mean()) if len(B) else None,
            "E1_added_cost_mean": float(A["cost_all_mean"].mean()) if len(A) else None,
            "GAP_displaced_cost_mean": float(B["cost_all_mean"].mean()) if len(B) else None,
        })
    pd.DataFrame(pool_rows).to_csv(os.path.join(args.out, "difficulty_by_pool.csv"), index=False)
    dd = pd.DataFrame(d_rows)
    dd.to_csv(os.path.join(args.out, "difficulty_comparison.csv"), index=False)
    results["difficulty"] = {r["metric"]: r for r in d_rows}

    # ======================= Analysis E =====================================
    log("--- Analysis E: lambda sweep (membership / batch-fill / advantage)", runlog)
    keep_by_lam = {lam: keep_map_for_lambda(pools[PRIMARY_POOL]["df"], lam) for lam in LAMBDAS}
    S_by_lam = {lam: {i for i, v in km.items() if v} for lam, km in keep_by_lam.items()}
    gap_sel_steps = {r["global_step"]: r for r in batchfill[PRIMARY_POOL]["GAP"]}
    gap_selected = set()
    for r in batchfill[PRIMARY_POOL]["GAP"]:
        gap_selected |= set(r["selected"])

    sel05 = set()
    for r in C.replay_batchfill(pools[PRIMARY_POOL]["g"], pools[PRIMARY_POOL]["vec"],
                                lam=LAMBDA_MAIN, target=target, mode="E1"):
        sel05 |= set(r["selected"])

    lam_rows, lam_bf_rows, lam_adv_rows = [], [], []
    for lam in LAMBDAS:
        S = S_by_lam[lam]
        ref0, ref5 = S_by_lam[0.0], S_by_lam[LAMBDA_MAIN]
        st = C.replay_batchfill(pools[PRIMARY_POOL]["g"], pools[PRIMARY_POOL]["vec"],
                                lam=lam, target=target, mode="E1")
        sel = set()
        for r in st:
            sel |= set(r["selected"])
        lam_rows.append({
            "lambda": lam, "n_retained": len(S),
            "overlap_with_lambda0": len(S & ref0), "jaccard_with_lambda0": C.jaccard(S, ref0),
            "new_vs_lambda0": len(S - ref0), "removed_vs_lambda0": len(ref0 - S),
            "overlap_with_lambda0.05": len(S & ref5), "jaccard_with_lambda0.05": C.jaccard(S, ref5),
            "membership_identical_to_lambda0": S == ref0,
            "membership_identical_to_lambda0.05": S == ref5,
        })
        lam_bf_rows.append({
            "lambda": lam, "calls_needed_total": sum(r["calls_needed"] for r in st),
            "selected_total": len(sel),
            "overlap_with_lambda0_selected": len(sel & gap_selected),
            "jaccard_with_lambda0_selected": C.jaccard(sel, gap_selected),
            "selected_identical_to_lambda0": sel == gap_selected,
            "overlap_with_lambda0.05_selected": len(sel & sel05),
            "selected_identical_to_lambda0.05": sel == sel05,
        })
        # advantages on this λ's actual optimizer batch
        e1only_ids = S - ref0
        adv_all, adv_correct, adv_wrong, adv_eff, adv_ineff, adv_e1only = [], [], [], [], [], []
        nan_ct = inf_ct = 0
        for r in st:
            scores, uids, tags = [], [], []
            for gid in r["selected"]:
                ci, uid = gid.split(":")
                correct, cost = pools[PRIMARY_POOL]["vec"][(int(ci), uid)]
                R, E = C.reward_vector(correct, cost, lam)
                for i in range(len(R)):
                    scores.append(R[i]); uids.append(gid)
                    tags.append((correct[i] > 0, E[i], gid))
            adv = C.grpo_advantage(scores, uids)
            nan_ct += int(np.isnan(adv).sum()); inf_ct += int(np.isinf(adv).sum())
            for a, (ok, e, gid) in zip(adv, tags):
                adv_all.append(a)
                if ok:
                    adv_correct.append(a)
                    (adv_eff if e > 0 else adv_ineff).append(a)
                else:
                    adv_wrong.append(a)
                if gid in e1only_ids:
                    adv_e1only.append(a)
        lam_adv_rows.append({
            "lambda": lam, "n_trajectories": len(adv_all),
            "advantage_mean": float(np.mean(adv_all)) if adv_all else None,
            "advantage_std": float(np.std(adv_all, ddof=1)) if len(adv_all) > 1 else None,
            "advantage_mean_abs": float(np.mean(np.abs(adv_all))) if adv_all else None,
            "advantage_max_abs": float(np.max(np.abs(adv_all))) if adv_all else None,
            "correct_mean": float(np.mean(adv_correct)) if adv_correct else None,
            "wrong_mean": float(np.mean(adv_wrong)) if adv_wrong else None,
            "correct_minus_wrong_gap": (float(np.mean(adv_correct) - np.mean(adv_wrong))
                                        if adv_correct and adv_wrong else None),
            "efficient_correct_mean": float(np.mean(adv_eff)) if adv_eff else None,
            "inefficient_correct_mean": float(np.mean(adv_ineff)) if adv_ineff else None,
            "efficient_minus_inefficient_gap": (float(np.mean(adv_eff) - np.mean(adv_ineff))
                                                if adv_eff and adv_ineff else None),
            "e1_only_group_advantage_mean": float(np.mean(adv_e1only)) if adv_e1only else None,
            "n_e1_only_trajectories": len(adv_e1only),
            "nan_count": nan_ct, "inf_count": inf_ct,
        })
    # --- advantage sensitivity on a FIXED optimizer batch (λ=0.05 selection) -------
    # isolates "λ changes advantage magnitude" from "λ changes membership" (which is
    # nil for every positive λ, see lambda_membership.csv)
    fixed_rows = []
    sel05_steps = {r["global_step"]: r["selected"] for r in C.replay_batchfill(
        pools[PRIMARY_POOL]["g"], pools[PRIMARY_POOL]["vec"], lam=LAMBDA_MAIN, target=target, mode="E1")}
    for lam in LAMBDAS:
        all_, cor_, wr_, eff_, ineff_ = [], [], [], [], []
        for step, ids in sel05_steps.items():
            scores, uids, tags = [], [], []
            for gid in ids:
                ci, uid = gid.split(":")
                correct, cost = pools[PRIMARY_POOL]["vec"][(int(ci), uid)]
                R, E = C.reward_vector(correct, cost, lam)
                for i in range(len(R)):
                    scores.append(R[i]); uids.append(gid)
                    tags.append((correct[i] > 0, E[i]))
            adv = C.grpo_advantage(scores, uids)
            for a, (ok, e) in zip(adv, tags):
                all_.append(a)
                if ok:
                    cor_.append(a)
                    (eff_ if e > 0 else ineff_).append(a)
                else:
                    wr_.append(a)
        fixed_rows.append({
            "lambda": lam, "batch": "fixed_lambda0.05_selection", "n_trajectories": len(all_),
            "advantage_mean_abs": float(np.mean(np.abs(all_))) if all_ else None,
            "advantage_std": float(np.std(all_, ddof=1)) if len(all_) > 1 else None,
            "correct_mean": float(np.mean(cor_)) if cor_ else None,
            "wrong_mean": float(np.mean(wr_)) if wr_ else None,
            "correct_minus_wrong_gap": (float(np.mean(cor_) - np.mean(wr_)) if cor_ and wr_ else None),
            "efficient_correct_mean": float(np.mean(eff_)) if eff_ else None,
            "inefficient_correct_mean": float(np.mean(ineff_)) if ineff_ else None,
            "efficient_minus_inefficient_gap": (float(np.mean(eff_) - np.mean(ineff_))
                                                if eff_ and ineff_ else None),
        })
    pd.DataFrame(fixed_rows).to_csv(os.path.join(args.out, "lambda_advantage_fixed_batch.csv"), index=False)
    results["lambda_advantage_fixed_batch"] = fixed_rows

    pd.DataFrame(lam_rows).to_csv(os.path.join(args.out, "lambda_membership.csv"), index=False)
    pd.DataFrame(lam_bf_rows).to_csv(os.path.join(args.out, "lambda_batchfill.csv"), index=False)
    pd.DataFrame(lam_adv_rows).to_csv(os.path.join(args.out, "lambda_advantage.csv"), index=False)
    results["lambda_membership"] = lam_rows
    results["lambda_batchfill"] = lam_bf_rows
    results["lambda_advantage"] = lam_adv_rows

    # ======================= Analysis F =====================================
    log("--- Analysis F: actual GAP vs E1-B training-run composition", runlog)
    gap_files = sorted(glob.glob(os.path.join(C.REPO, "verl/logs/gen_bs_supervisor_run/train_console.run*.log")))
    gap_loop, gap_dup = {}, []
    for p in gap_files:
        for step, kept in parse_generation_loop(p).items():
            if step in gap_loop:
                gap_dup.append((os.path.basename(p), step, len(gap_loop[step]), len(kept)))
            gap_loop.setdefault(step, []).extend(kept)
    e1b_loop = parse_generation_loop(
        os.path.join(C.REPO, "verl/logs/DAPO-GAP3B-MHQA-Agent-E1B-step70to120-4gpu.log"))
    e1b_calls = parse_calls_per_step(os.path.join(C.REPO, "E1-B/run_20260911_113432/diagnostics/calls_pid*.jsonl"))
    e10_calls = parse_calls_per_step(os.path.join(C.REPO, "E1-0/runs/20260910_step60to70_e10/audit/calls_pid*.jsonl"))
    e1a_calls = parse_calls_per_step(os.path.join(C.REPO, "E1-A/run_20260911_024902/diagnostics/calls_pid*.jsonl"))
    act_rows = []
    for step in sorted(set(list(gap_loop) + list(e1b_loop))):
        if step < 70 or step > 120:
            continue
        gl, el = gap_loop.get(step, []), e1b_loop.get(step, [])
        act_rows.append({
            "global_step": step,
            "GAP_gen_batches": len(gl) if gl else None,
            "GAP_retained_per_batch_mean": float(np.mean(gl)) if gl else None,
            "E1B_gen_batches": len(el) if el else None,
            "E1B_retained_per_batch_mean": float(np.mean(el)) if el else None,
        })
    act = pd.DataFrame(act_rows)
    act.to_csv(os.path.join(args.out, "actual_run_comparison.csv"), index=False)
    both = act.dropna(subset=["GAP_gen_batches", "E1B_gen_batches"]) if len(act) else act
    results["actual_run"] = {
        "steps_compared": int(len(both)),
        "GAP_gen_batches_mean": float(both["GAP_gen_batches"].mean()) if len(both) else None,
        "E1B_gen_batches_mean": float(both["E1B_gen_batches"].mean()) if len(both) else None,
        "GAP_retained_per_batch_mean": float(both["GAP_retained_per_batch_mean"].mean()) if len(both) else None,
        "E1B_retained_per_batch_mean": float(both["E1B_retained_per_batch_mean"].mean()) if len(both) else None,
        "duplicate_steps_across_gap_logs": gap_dup,
        "note": ("OBSERVATIONAL / POLICY-CONFOUNDED: after the first policy update the two branches "
                 "generate different rollouts; the original GAP run keeps no per-group records (its "
                 "`std val` print is truncated by numpy), so only the generation-loop counts are "
                 "recoverable for it."),
    }

    e1b_obs = []
    for step, sub in gp.groupby("global_step"):
        d = pools[PRIMARY_POOL]["df"]
        d = d[d["global_step"] == step]
        e1b_obs.append({
            "global_step": int(step),
            "calls_observed": int(sub["call_index"].nunique()),
            "candidate_groups": int(len(sub)),
            "retained_observed_shaped": int(sub["group_id"].map(km["e1_keep"]).sum()),
            "retained_counterfactual_GAP": int(sub["group_id"].map(km["gap_keep"]).sum()),
            "candidate_em": float(d["correct"].mean()),
            "all_correct_rate": float(sub["all_correct"].mean()),
            "mixed_rate": float(sub["mixed"].mean()),
            "active_efficiency_groups": int(sub["is_active_efficiency"].sum()),
            "group_success_rate_mean": float(sub["group_success_rate"].mean()),
            "search_rounds_mean": float(sub["cost_all_mean"].mean()),
        })
    pd.DataFrame(e1b_obs).to_csv(os.path.join(args.out, "e1b_actual_composition_by_step.csv"), index=False)

    # ======================= Analysis G =====================================
    log("--- Analysis G: hypothesis verdicts", runlog)
    bf_agg = results[f"batchfill_{PRIMARY_POOL}"]
    pA = pool_A[PRIMARY_POOL]
    e1o_row = gt[(gt["pool"] == PRIMARY_POOL) & (gt["set"] == "E1_only")]
    e1o_all_correct = float(e1o_row.iloc[0]["all_correct_rate"]) if len(e1o_row) else None
    positive_lams = [l for l in LAMBDAS if l > 0]
    same_membership_all_positive = all(
        r["membership_identical_to_lambda0.05"] for r in lam_rows if r["lambda"] in positive_lams)
    dd_map = results["difficulty"]
    lam_mem = {r["lambda"]: r for r in lam_rows}
    adv_map = {r["lambda"]: r for r in lam_adv_rows}
    H = {
        "H1": {"statement": "the E1 reward adds new retained groups on the same candidate pool",
               "verdict": "SUPPORTED" if pA["E1_only"] > 0 else "NOT SUPPORTED",
               "evidence": f"|S_E1|-|S_GAP| = {pA['E1_only']} E1-only groups out of "
                           f"{pA['total_candidate_groups']} candidates ({pA['relation']})"},
        "H2": {"statement": "the new retained groups are mostly all-correct but efficiency-variable",
               "verdict": ("SUPPORTED" if (e1o_all_correct or 0) == 1.0 else
                           ("PARTIALLY SUPPORTED" if (e1o_all_correct or 0) > 0.5 else "NOT SUPPORTED")),
               "evidence": f"E1-only all-correct rate = {e1o_all_correct} "
                           f"(n={int(e1o_row.iloc[0]['n_groups']) if len(e1o_row) else 0}); "
                           f"active-efficiency groups among E1-only = "
                           f"{int(e1o_row.iloc[0]['active_efficiency_groups']) if len(e1o_row) else 0}"},
        "H3": {"statement": "fixed-size optimizer batch + early stop makes E1 displace mixed groups "
                            "GAP would have used, changing the actual training composition",
               "verdict": "SUPPORTED" if bf_agg["E1_selected_only"] > 0 else "NOT SUPPORTED",
               "evidence": (
                           f"on the E1-B pool the E1 filter fills all {bf_agg['target_slots']} optimizer slots "
                           f"while the counterfactual GAP filter is right-censored in "
                           f"{bf_agg['GAP_censored_steps']}/{bf_agg['steps']} steps (only "
                           f"{bf_agg['GAP_selected_total']} slots filled) - on the same stream E1 retains "
                           f"{bf_agg['fill_rate_ratio_E1_over_GAP']:.3f}x more groups per generation batch; "
                           f"on the uncensored E1-0 pool (generated under the GAP filter) E1 reaches 32 groups "
                           f"in {results['batchfill_E1_0']['E1_calls_needed_total']} generation batches vs GAP's "
                           f"{results['batchfill_E1_0']['GAP_calls_needed_total']} "
                           f"({results['batchfill_E1_0']['call_reduction_pct']:.1f}% fewer), replacing "
                           f"{results['batchfill_E1_0']['E1_selected_only']} of "
                           f"{results['batchfill_E1_0']['target_slots']} optimizer slots "
                           f"({100.0*results['batchfill_E1_0']['E1_selected_only']/results['batchfill_E1_0']['target_slots']:.1f}%) "
                           f"with E1-only groups")},
        "H4": {"statement": "the groups E1 adds are easier than the GAP groups they displace",
               "verdict": ("SUPPORTED" if (dd_map["group_success_rate"]["E1_added_mean"] or 0)
                           > (dd_map["group_success_rate"]["GAP_displaced_mean"] or 0) else "NOT SUPPORTED"),
               "evidence": f"mean group success rate E1-added "
                           f"{dd_map['group_success_rate']['E1_added_mean']:.4f} vs GAP-displaced "
                           f"{dd_map['group_success_rate']['GAP_displaced_mean']:.4f} "
                           f"(p={dd_map['group_success_rate']['mannwhitney_p']:.3g}); all-correct rate "
                           f"{dd_map['all_correct_rate']['E1_added_mean']:.4f} vs "
                           f"{dd_map['all_correct_rate']['GAP_displaced_mean']:.4f}"},
        "H5": {"statement": "different positive λ change advantage magnitude rather than retained membership",
               "verdict": "SUPPORTED" if same_membership_all_positive else "NOT SUPPORTED",
               "evidence": "retained membership identical across all positive λ = "
                           f"{same_membership_all_positive}; |S_λ| = "
                           + ", ".join(f"{l}:{lam_mem[l]['n_retained']}" for l in LAMBDAS)
                           + "; advantage magnitudes: "
                           + ", ".join(f"λ={l}: mean|adv|={adv_map[l]['advantage_mean_abs']:.4f}"
                                       for l in LAMBDAS)},
        "H6": {"statement": "the E1-B EM drop is consistent with the training-distribution-shift hypothesis",
               "verdict": "SUPPORTED BY ASSOCIATION",
               "evidence": f"{bf_agg['E1_selected_only']} of {bf_agg['E1_selected_total']} groups in the "
                           f"E1 optimizer batch are E1-only on the same pool (E1 retains "
                           f"{bf_agg['fill_rate_ratio_E1_over_GAP']:.3f}x more groups per generation batch); the E1-B "
                           f"step120 evaluation is micro EM -1.16 pp vs GAP step120 (paired p=2.1e-12). "
                           f"This is an association only: the policy also changes its own candidate "
                           f"distribution, and the EM cost is confounded with the efficiency objective."},
    }
    results["hypotheses"] = H
    for k in sorted(H):
        log(f"  {k}: {H[k]['verdict']} | {H[k]['evidence'][:200]}", runlog)

    # ======================= sanity checks ==================================
    sanity = {}
    e10g = pools["E1_0"]["g"]
    gap_retained_e10 = sum(1 for i in e10g["group_id"] if pools["E1_0"]["gap_keep"][i])
    sanity["baseline_retained_E1_0_pool"] = gap_retained_e10
    try:
        e105 = json.load(open(os.path.join(C.REPO, "E1-0.5/outputs/filter_replay_results.json")))
        sanity["E1_0.5_reported_baseline_retained"] = e105.get("baseline_retained")
        sanity["matches_E1_0.5"] = (e105.get("baseline_retained") == gap_retained_e10)
    except Exception as e:  # noqa: BLE001
        sanity["matches_E1_0.5"] = f"unavailable: {type(e).__name__}: {e}"
    bad0 = 0
    for gid, (correct, cost) in list(pools[PRIMARY_POOL]["vec"].items()):
        R0, _ = C.reward_vector(correct, cost, 0.0)
        if not np.allclose(R0, correct):
            bad0 += 1
    sanity["lambda0_equals_EM_all_groups"] = bool(bad0 == 0)
    sanity["lambda0_mismatch_groups"] = bad0
    sanity["group_size_histogram"] = {name: {int(k): int(v) for k, v in Counter(p["g"]["n_rollouts"]).items()}
                                      for name, p in pools.items()}
    sanity["all_groups_n8"] = all(set(Counter(p["g"]["n_rollouts"])) == {8} for p in pools.values())
    sanity["rollout_and_group_counts"] = {
        name: {"rollouts": int(len(p["df"])), "groups": int(len(p["g"])),
               "rollouts_in_groups": int(p["g"]["n_rollouts"].sum())} for name, p in pools.items()}
    rep = {r["global_step"]: r["calls_needed"] for r in batchfill["E1_B"]["E1"]}
    m = {s: rep.get(s) == len(e1b_loop.get(s, [])) for s in sorted(set(rep) & set(e1b_loop))}
    sanity["E1_replay_matches_observed_E1B_gen_batches"] = {
        "steps_compared": len(m), "n_mismatch": sum(1 for v in m.values() if not v),
        "all_match": all(m.values())}
    rep0 = {r["global_step"]: r["calls_needed"] for r in batchfill["E1_0"]["GAP"]}
    m0 = {s: rep0.get(s) == e10_calls.get(s) for s in sorted(set(rep0) & set(e10_calls))}
    sanity["GAP_replay_matches_observed_E1_0_gen_batches"] = {
        "steps_compared": len(m0), "n_mismatch": sum(1 for v in m0.values() if not v),
        "all_match": all(m0.values())}
    repA = {r["global_step"]: r["calls_needed"] for r in batchfill["E1_A"]["E1"]}
    mA = {s: repA.get(s) == e1a_calls.get(s) for s in sorted(set(repA) & set(e1a_calls))}
    sanity["E1_replay_matches_observed_E1A_gen_batches"] = {
        "steps_compared": len(mA), "n_mismatch": sum(1 for v in mA.values() if not v),
        "all_match": all(mA.values())}
    sanity["replay_calls_needed_by_pool"] = {
        "E1_0_GAP": sum(v for v in rep0.values()), "E1_0_observed": sum(e10_calls.values()),
        "E1_A_E1": sum(v for v in repA.values()), "E1_A_observed": sum(e1a_calls.values()),
        "E1_B_E1": sum(v for v in rep.values()), "E1_B_observed": sum(e1b_calls.values()),
    }
    # numerical consistency between summary.json and the emitted CSVs
    csv_sets = pd.read_csv(os.path.join(args.out, "same_pool_set_summary.csv"))
    row = csv_sets[csv_sets["pool"] == PRIMARY_POOL].iloc[0]
    sanity["summary_csv_consistent"] = {
        "S_GAP": int(row["S_GAP"]) == pA["S_GAP"], "S_E1": int(row["S_E1"]) == pA["S_E1"],
        "E1_only": int(row["E1_only"]) == pA["E1_only"],
        "difficulty_rows": int(len(dd)) == len(METRIC_COLS) + 3,
        "lambda_rows": int(len(pd.read_csv(os.path.join(args.out, "lambda_membership.csv")))) == len(LAMBDAS),
    }
    results["sanity"] = sanity
    for k, v in sanity.items():
        log(f"  sanity {k}: {v}", runlog)

    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    log(f"summary -> {os.path.join(args.out, 'summary.json')}", runlog)
    runlog.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

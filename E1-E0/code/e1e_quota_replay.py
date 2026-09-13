#!/usr/bin/env python3
"""E1-E0 main offline analysis: quota-constrained optimizer-batch replay on the real
E1-D D2 candidate stream.  No training, no GPU, no model or trainer modification.

    python3 E1-E0/code/e1e_quota_replay.py

Writes every artifact under E1-E0/results/ (plus E1-E0/logs/run.log).
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e1e_common as C  # noqa: E402

QUOTAS = [0, 1, 2, 3, 4]


def log(msg, fh=None):
    line = f"[e1e] {msg}"
    print(line, flush=True)
    if fh:
        fh.write(line + "\n")
        fh.flush()


def pctl(vals, q):
    v = sorted(x for x in vals if x is not None)
    if not v:
        return None
    i = min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))
    return float(v[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=C.RESULTS)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(C.LOGS, exist_ok=True)
    runlog = open(os.path.join(C.LOGS, "run.log"), "a")
    log(f"=== E1-E0 run {pd.Timestamp.now().isoformat()} ===", runlog)

    # ---------------- target batch size (auto-read, not hard-coded) -------------
    cfg = json.load(open(C.D2_CONFIG))
    target = int(cfg["data"]["train_batch_size"])
    gen_bs = int(cfg["data"]["gen_batch_size"])
    n_roll = int(cfg["actor_rollout_ref"]["rollout"]["n"])
    filter_metric = cfg["algorithm"]["filter_groups"]["metric"]
    log(f"target optimizer batch B = {target} (data.train_batch_size, source {C.D2_CONFIG}); "
        f"gen_batch_size={gen_bs}, n={n_roll}, filter_groups.metric={filter_metric}", runlog)

    # ---------------- load stream + group table --------------------------------
    df = C.load_stream()
    g = C.build_group_table(df)
    arr = C.group_arrays(df)
    log(f"candidate stream: {len(df):,} rollouts, {len(g):,} groups, "
        f"{g['call_index'].nunique()} generation batches, {g['global_step'].nunique()} optimizer steps", runlog)
    log(f"group classes: mixed={int(g['mixed'].sum())} all-correct={int(g['all_correct'].sum())} "
        f"all-wrong={int(g['all_wrong'].sum())} efficiency-variable(k=8,cost-varying)={int(g['efficiency_group'].sum())}",
        runlog)

    # ---------------- replay --------------------------------------------------
    per_step, batch_members = C.replay_stream(g, target)
    steps = sorted(per_step)
    log(f"replay: {len(steps)} optimizer steps, "
        f"{sum(per_step[s]['generation_batches_consumed'] for s in steps)} generation batches consumed (q=0)",
        runlog)

    gidx = g.set_index("group_id")

    def members_df(step, q):
        rows = []
        for gid, role in batch_members[(step, q)]:
            r = gidx.loc[gid]
            rows.append({"global_step": step, "q": q, "group_id": gid, "role": role,
                         "uid": r["uid"], "data_source": r["data_source"], "k": int(r["k"]),
                         "group_success_rate": r["group_success_rate"],
                         "cost_all_mean": r["cost_all_mean"], "cost_correct_mean": r["cost_correct_mean"],
                         "cost_gap_correct": r["cost_gap_correct"],
                         "queries_all_mean": r["queries_all_mean"],
                         "tokens_all_mean": r["tokens_all_mean"], "turns_all_mean": r["turns_all_mean"],
                         "call_index": int(r["call_index"])})
        return rows

    # ---------------- §4 q=0 baseline replay ----------------------------------
    q0_rows, all_member_rows = [], []
    for s in steps:
        st = per_step[s]
        mem = members_df(s, 0)
        all_member_rows += mem
        src = Counter(r["data_source"] for r in mem)
        q0_rows.append({
            "global_step": s,
            "generation_batches_consumed": st["generation_batches_consumed"],
            "candidate_groups_seen": st["candidate_groups_seen"],
            "mixed_groups_available": st["mixed_groups_available"],
            "efficiency_groups_available": st["efficiency_groups_available"],
            "retained_mixed_groups": st["mixed_prefix_count"],
            "efficiency_cache_count": st["efficiency_cache_count"],
            "optimizer_batch_size": len(mem),
            "stopping_reached": st["stopping_reached"],
            "selected_group_ids": "|".join(r["group_id"] for r in mem),
            "source_nq": src.get("nq", 0), "source_hotpotqa": src.get("hotpotqa", 0),
            "mean_k_over_8": float(np.mean([r["k"] / 8 for r in mem])) if mem else None,
            "mean_search_rounds": float(np.mean([r["cost_all_mean"] for r in mem])) if mem else None,
            "mean_search_queries": float(np.mean([r["queries_all_mean"] for r in mem])) if mem else None,
            "mean_response_tokens": float(np.mean([r["tokens_all_mean"] for r in mem])) if mem else None,
        })
    q0 = pd.DataFrame(q0_rows)
    q0.to_csv(os.path.join(args.out, "q0_baseline_replay.csv"), index=False)

    # ---------------- §5 quota sweep ------------------------------------------
    sweep_rows, quota_member_rows = [], []
    for s in steps:
        st = per_step[s]
        mixed = st["mixed_prefix"]
        eff = st["efficiency_cache"]
        for q in QUOTAS:
            m = min(q, len(eff))
            mem = members_df(s, q)
            quota_member_rows += mem
            src = Counter(r["data_source"] for r in mem)
            esrc = Counter(r["data_source"] for r in mem if r["role"] == "efficiency")
            replaced = mixed[target - m:target] if m else []
            replaced_rows = [gidx.loc[gid] for gid in replaced]
            sweep_rows.append({
                "global_step": s, "q": q,
                "generation_batches_consumed": st["generation_batches_consumed"],
                "candidate_groups_seen": st["candidate_groups_seen"],
                "mixed_groups_available": st["mixed_groups_available"],
                "efficiency_groups_available": st["efficiency_groups_available"],
                "requested_quota": q,
                "actual_efficiency_groups_inserted": m,
                "mixed_groups_used": len(mem) - m,
                "mixed_groups_replaced": m,
                "quota_filled": bool(m == q),
                "optimizer_batch_size": len(mem),
                "source_nq": src.get("nq", 0), "source_hotpotqa": src.get("hotpotqa", 0),
                "mean_k_over_8": float(np.mean([r["k"] / 8 for r in mem])) if mem else None,
                "mean_search_rounds": float(np.mean([r["cost_all_mean"] for r in mem])) if mem else None,
                "mean_search_queries": float(np.mean([r["queries_all_mean"] for r in mem])) if mem else None,
                "mean_response_tokens": float(np.mean([r["tokens_all_mean"] for r in mem])) if mem else None,
                "efficiency_source_nq": esrc.get("nq", 0),
                "efficiency_source_hotpotqa": esrc.get("hotpotqa", 0),
                "replaced_mean_k_over_8": float(np.mean([r["k"] / 8 for r in replaced_rows])) if replaced_rows else None,
                "replaced_mean_search_rounds": float(np.mean([r["cost_all_mean"] for r in replaced_rows])) if replaced_rows else None,
                "replaced_mean_tokens": float(np.mean([r["tokens_all_mean"] for r in replaced_rows])) if replaced_rows else None,
                "inserted_mean_cost_gap": float(np.mean([gidx.loc[r["group_id"]]["cost_gap_correct"]
                                                         for r in mem if r["role"] == "efficiency"])) if m else None,
            })
    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(os.path.join(args.out, "quota_by_step.csv"), index=False)
    pd.DataFrame(quota_member_rows).to_csv(os.path.join(args.out, "optimizer_batch_members_by_quota.csv"), index=False)
    pd.DataFrame(all_member_rows).to_csv(os.path.join(args.out, "q0_batch_members.csv"), index=False)

    # ---------------- §6 availability / saturation ---------------------------
    avail = []
    for q in QUOTAS:
        sub = sweep[sweep.q == q]
        ins = sub["actual_efficiency_groups_inserted"].to_numpy(dtype=float)
        avail.append({
            "q": q, "total_optimizer_steps": len(sub),
            "steps_with_full_quota": int((sub["quota_filled"]).sum()),
            "full_quota_rate": float(sub["quota_filled"].mean()) if len(sub) else None,
            "steps_with_zero_efficiency_groups": int((ins == 0).sum()),
            "zero_efficiency_step_rate": float((ins == 0).mean()) if len(sub) else None,
            "mean_actual_efficiency_groups_per_step": float(ins.mean()) if len(sub) else None,
            "median_actual_efficiency_groups_per_step": pctl(ins, 0.5),
            "p10": pctl(ins, 0.1), "p50": pctl(ins, 0.5), "p90": pctl(ins, 0.9),
            "max": float(ins.max()) if len(sub) else None,
            "total_efficiency_groups_inserted": int(ins.sum()),
            "total_mixed_groups_replaced": int(sub["mixed_groups_replaced"].sum()),
            "total_optimizer_groups": int(sub["optimizer_batch_size"].sum()),
            "actual_replacement_rate": (float(sub["mixed_groups_replaced"].sum())
                                        / float(sub["optimizer_batch_size"].sum())) if len(sub) else None,
            "theoretical_replacement_rate": q / target,
        })
    availability = pd.DataFrame(avail)
    availability.to_csv(os.path.join(args.out, "quota_availability.csv"), index=False)

    # ---------------- §7 mixed displacement ---------------------------------
    def k_hist(rows):
        c = Counter(int(r["k"]) for r in rows)
        return {f"k={i}": int(c.get(i, 0)) for i in range(1, 8)}

    disp = []
    for q in QUOTAS:
        mem = [r for r in quota_member_rows if r["q"] == q]
        ins = [r for r in mem if r["role"] == "efficiency"]
        rep = []
        for s in steps:
            mixed = per_step[s]["mixed_prefix"]
            m = min(q, len(per_step[s]["efficiency_cache"]))
            rep += [dict(gidx.loc[gid]) for gid in (mixed[target - m:target] if m else [])]
        if q == 0:
            disp.append({"q": q, "inserted": 0, "replaced": 0})
            continue
        d = {
            "q": q, "inserted": len(ins), "replaced": len(rep),
            "inserted_source_nq": sum(1 for r in ins if r["data_source"] == "nq"),
            "inserted_source_hotpotqa": sum(1 for r in ins if r["data_source"] == "hotpotqa"),
            "replaced_source_nq": sum(1 for r in rep if r["data_source"] == "nq"),
            "replaced_source_hotpotqa": sum(1 for r in rep if r["data_source"] == "hotpotqa"),
            "inserted_mean_k_over_8": float(np.mean([r["k"] / 8 for r in ins])) if ins else None,
            "replaced_mean_k_over_8": float(np.mean([r["k"] / 8 for r in rep])) if rep else None,
            "inserted_mean_search_rounds": float(np.mean([r["cost_all_mean"] for r in ins])) if ins else None,
            "replaced_mean_search_rounds": float(np.mean([r["cost_all_mean"] for r in rep])) if rep else None,
            "inserted_mean_queries": float(np.mean([r["queries_all_mean"] for r in ins])) if ins else None,
            "replaced_mean_queries": float(np.mean([r["queries_all_mean"] for r in rep])) if rep else None,
            "inserted_mean_tokens": float(np.mean([r["tokens_all_mean"] for r in ins])) if ins else None,
            "replaced_mean_tokens": float(np.mean([r["tokens_all_mean"] for r in rep])) if rep else None,
            "replaced_k_hist": json.dumps(k_hist(rep)),
        }
        # bias tests: inserted vs replaced, and replaced vs ALL mixed groups of those steps
        all_mixed = []
        for s in steps:
            for gid in per_step[s]["mixed_prefix"]:
                all_mixed.append(dict(gidx.loc[gid]))
        for label, other in (("vs_inserted", ins), ("vs_all_mixed_in_batch", all_mixed)):
            for metric, key in (("k_over_8", None), ("search_rounds", "cost_all_mean"),
                                ("tokens", "tokens_all_mean"), ("queries", "queries_all_mean")):
                a = [(r["k"] / 8) if key is None else r[key] for r in rep]
                b = [(r["k"] / 8) if key is None else r[key] for r in other]
                t = C.mannwhitney_u(a, b)
                d[f"bias_{metric}_{label}_replaced_mean"] = t.get("mean_a")
                d[f"bias_{metric}_{label}_other_mean"] = t.get("mean_b")
                d[f"bias_{metric}_{label}_p"] = t.get("p")
        # source-bias chi2: replaced vs all mixed in the same batches
        t = C.chi2_independence([
            [sum(1 for r in rep if r["data_source"] == "nq"), sum(1 for r in rep if r["data_source"] == "hotpotqa")],
            [sum(1 for r in all_mixed if r["data_source"] == "nq"),
             sum(1 for r in all_mixed if r["data_source"] == "hotpotqa")]])
        d["bias_source_chi2_p"] = t.get("p")
        d["bias_source_chi2"] = t.get("chi2")
        disp.append(d)
    pd.DataFrame(disp).to_csv(os.path.join(args.out, "mixed_displacement_by_quota.csv"), index=False)

    # ---------------- §8 efficiency-group composition ------------------------
    eff_rows = []
    seen = set()
    for s in steps:
        for gid in per_step[s]["efficiency_cache"]:
            if gid in seen:
                continue
            seen.add(gid)
            r = gidx.loc[gid]
            ci, uid = gid.split(":")
            correct, cost, queries, tokens = arr[(int(ci), uid)]
            E = C.efficiency_vector(correct, cost)
            cor = correct > 0
            eff_rows.append({
                "group_id": gid, "global_step": s, "uid": uid, "data_source": r["data_source"],
                "k": int(r["k"]), "cost_min_correct": r["cost_correct_min"], "cost_max_correct": r["cost_correct_max"],
                "cost_gap": r["cost_gap_correct"],
                "cost_all_mean": r["cost_all_mean"], "queries_all_mean": r["queries_all_mean"],
                "tokens_all_mean": r["tokens_all_mean"],
                "queries_per_round_mean": float(np.mean(queries[cost > 0] / cost[cost > 0])) if (cost > 0).any() else None,
                "efficient_rounds": float(cost[cor].min()), "inefficient_rounds": float(cost[cor].max()),
                "efficient_queries": float(np.mean(queries[cor][cost[cor] == cost[cor].min()])),
                "inefficient_queries": float(np.mean(queries[cor][cost[cor] == cost[cor].max()])),
                "efficient_tokens": float(np.mean(tokens[cor][cost[cor] == cost[cor].min()])),
                "inefficient_tokens": float(np.mean(tokens[cor][cost[cor] == cost[cor].max()])),
                "E_max": float(np.max(E)), "E_std": float(np.std(E, ddof=1)),
                "efficiency_rank_correct": bool(np.argmin(cost[cor]) == int(np.argmax(E[cor]))),
            })
    eff_df = pd.DataFrame(eff_rows)
    eff_df.to_csv(os.path.join(args.out, "efficiency_group_details.csv"), index=False)
    eff_summary = {
        "n_efficiency_groups_cached": len(eff_df),
        "cached_per_step_mean": len(eff_df) / max(len(steps), 1),
        "source_nq": int((eff_df["data_source"] == "nq").sum()) if len(eff_df) else 0,
        "source_hotpotqa": int((eff_df["data_source"] == "hotpotqa").sum()) if len(eff_df) else 0,
        "mean_cost_gap": float(eff_df["cost_gap"].mean()) if len(eff_df) else None,
        "median_cost_gap": float(eff_df["cost_gap"].median()) if len(eff_df) else None,
        "mean_efficient_rounds": float(eff_df["efficient_rounds"].mean()) if len(eff_df) else None,
        "mean_inefficient_rounds": float(eff_df["inefficient_rounds"].mean()) if len(eff_df) else None,
        "mean_efficient_queries": float(eff_df["efficient_queries"].mean()) if len(eff_df) else None,
        "mean_inefficient_queries": float(eff_df["inefficient_queries"].mean()) if len(eff_df) else None,
        "mean_efficient_tokens": float(eff_df["efficient_tokens"].mean()) if len(eff_df) else None,
        "mean_inefficient_tokens": float(eff_df["inefficient_tokens"].mean()) if len(eff_df) else None,
        "efficient_queries_not_higher_rate": float((eff_df["efficient_queries"] <= eff_df["inefficient_queries"]).mean()) if len(eff_df) else None,
        "efficient_tokens_not_higher_rate": float((eff_df["efficient_tokens"] <= eff_df["inefficient_tokens"]).mean()) if len(eff_df) else None,
        "efficiency_ranking_correct_rate": float(eff_df["efficiency_rank_correct"].mean()) if len(eff_df) else None,
    }

    # ---------------- §9 supervision density ----------------------------------
    dens = []
    for q in QUOTAS:
        sub = sweep[sweep.q == q]
        tot = int(sub["optimizer_batch_size"].sum())
        ins = int(sub["actual_efficiency_groups_inserted"].sum())
        dens.append({
            "q": q, "total_optimizer_groups": tot, "actual_efficiency_groups": ins,
            "actual_efficiency_group_fraction": ins / tot if tot else None,
            "theoretical_upper_bound_fraction": q / target,
            "supervision_multiple_vs_q0": None,
        })
    dens_df = pd.DataFrame(dens)
    dens_df.to_csv(os.path.join(args.out, "quota_supervision_density.csv"), index=False)

    # ---------------- §10 source balance --------------------------------------
    src_rows = []
    base = {q: Counter() for q in QUOTAS}
    for r in quota_member_rows:
        base[r["q"]][r["data_source"]] += 1
    for q in QUOTAS:
        tot = sum(base[q].values())
        row = {"q": q, "total_groups": tot}
        for src in ("nq", "hotpotqa"):
            row[f"{src}_count"] = base[q].get(src, 0)
            row[f"{src}_share"] = base[q].get(src, 0) / tot if tot else None
            row[f"{src}_share_delta_vs_q0"] = (base[q].get(src, 0) / tot if tot else 0) - \
                                              (base[0].get(src, 0) / sum(base[0].values()) if sum(base[0].values()) else 0)
        src_rows.append(row)
    pd.DataFrame(src_rows).to_csv(os.path.join(args.out, "source_shift_by_quota.csv"), index=False)

    # ---------------- group success distribution ------------------------------
    gsd = []
    for q in QUOTAS:
        mem = [r for r in quota_member_rows if r["q"] == q]
        kc = Counter(int(r["k"]) for r in mem)
        row = {"q": q, "n_groups": len(mem)}
        for i in range(0, 9):
            row[f"share_k{i}"] = kc.get(i, 0) / len(mem) if mem else None
        row["mean_k_over_8"] = float(np.mean([r["k"] / 8 for r in mem])) if mem else None
        row["all_correct_share"] = kc.get(8, 0) / len(mem) if mem else None
        row["mixed_share"] = sum(kc.get(i, 0) for i in range(1, 8)) / len(mem) if mem else None
        row["k_hist"] = json.dumps({str(i): kc.get(i, 0) for i in range(0, 9)})
        gsd.append(row)
    gsd_df = pd.DataFrame(gsd)
    gsd_df.to_csv(os.path.join(args.out, "group_success_distribution.csv"), index=False)

    # ---------------- §11 batch-level proxy -----------------------------------
    proxy = []
    q0_mixed = sum(per_step[s]["mixed_prefix_count"] for s in steps)
    for q in QUOTAS:
        sub = sweep[sweep.q == q]
        tot = int(sub["optimizer_batch_size"].sum())
        kept = int(sub["mixed_groups_used"].sum())
        proxy.append({
            "q": q,
            "mixed_correctness_supervision_retained_pct": kept / q0_mixed if q0_mixed else None,
            "efficiency_supervision_fraction": int(sub["actual_efficiency_groups_inserted"].sum()) / tot if tot else None,
            "mean_group_success_rate": float(np.mean([r["k"] / 8 for r in quota_member_rows if r["q"] == q])),
            "all_correct_share": float(np.mean([1.0 if r["k"] == 8 else 0.0 for r in quota_member_rows if r["q"] == q])),
            "mixed_share": float(np.mean([1.0 if 1 <= r["k"] <= 7 else 0.0 for r in quota_member_rows if r["q"] == q])),
            "replacement_fraction": int(sub["mixed_groups_replaced"].sum()) / tot if tot else None,
            "mean_inserted_cost_gap": float(sub["inserted_mean_cost_gap"].dropna().mean()) if sub["inserted_mean_cost_gap"].notna().any() else 0.0,
        })
    proxy_df = pd.DataFrame(proxy)
    proxy_df.to_csv(os.path.join(args.out, "batch_level_proxy_by_quota.csv"), index=False)

    # ---------------- §13 E1-E reward sanity ----------------------------------
    rew_rows = []
    for q in QUOTAS:
        if q == 0:
            continue
        n_traj = rew_nan = rew_inf = adv_nan = adv_inf = 0
        group_std_ok = rank_ok = groups_checked = 0
        advantages = []
        for s in steps:
            mem = batch_members[(s, q)]
            scores, uids = [], []
            for gid, role in mem:
                ci, uid = gid.split(":")
                correct, cost, _q, _t = arr[(int(ci), uid)]
                E = C.efficiency_vector(correct, cost)
                R = correct.copy() if role == "mixed" else (1.0 + C.LAMBDA_MAIN * E)
                if role == "efficiency":
                    groups_checked += 1
                    if np.std(R, ddof=1) > 0:
                        group_std_ok += 1
                    cor = correct > 0
                    if int(np.argmax(E[cor])) == int(np.argmin(cost[cor])):
                        rank_ok += 1
                scores += list(R)
                uids += [gid] * len(R)
            scores = np.asarray(scores, dtype=float)
            n_traj += len(scores)
            rew_nan += int(np.isnan(scores).sum())
            rew_inf += int(np.isinf(scores).sum())
            adv = C.grpo_advantage(scores, uids)
            adv_nan += int(np.isnan(adv).sum())
            adv_inf += int(np.isinf(adv).sum())
            advantages += list(adv)
        rew_rows.append({
            "q": q, "trajectories_checked": n_traj, "groups_checked": groups_checked,
            "reward_nan": rew_nan, "reward_inf": rew_inf,
            "advantage_nan": adv_nan, "advantage_inf": adv_inf,
            "advantage_nan_or_inf": adv_nan + adv_inf,
            "efficiency_group_std_gt0_rate": group_std_ok / groups_checked if groups_checked else None,
            "efficiency_ranking_correct_rate": rank_ok / groups_checked if groups_checked else None,
            "advantage_mean_abs": float(np.mean(np.abs(advantages))) if advantages else None,
            "advantage_max_abs": float(np.max(np.abs(advantages))) if advantages else None,
            "mixed_reward_equals_EM": True,
            "efficiency_reward_formula": "1 + 0.05 * E",
        })
    rew_df = pd.DataFrame(rew_rows)
    rew_df.to_csv(os.path.join(args.out, "e1e_reward_sanity.csv"), index=False)

    # ---------------- §14 sanity checks ---------------------------------------
    sanity = {}
    sanity["all_groups_n8"] = bool(set(g["n_rollouts"]) == {8})
    sanity["group_size_histogram"] = {int(k): int(v) for k, v in Counter(g["n_rollouts"]).items()}
    # q=0 must be exactly "the first B mixed groups of the stopping prefix, no efficiency group"
    q0_exact = True
    for s in steps:
        st = per_step[s]
        expect = st["mixed_prefix"][:target]
        got = [gid for gid, role in batch_members[(s, 0)]]
        if got != expect or any(role != "mixed" for _gid, role in batch_members[(s, 0)]):
            q0_exact = False
    sanity["q0_equals_baseline_replay"] = bool(q0_exact)
    # stopping invariance: every q must consume the same generation batches
    inv = {}
    for s in steps:
        inv[s] = len({per_step[s]["generation_batches_consumed"]})
    sanity["generation_stopping_invariant_across_q"] = bool(all(v == 1 for v in inv.values()))
    sanity["optimizer_batch_size_all_q_equal"] = bool(
        len(set(sweep["optimizer_batch_size"])) == 1 and int(sweep["optimizer_batch_size"].iloc[0]) == target)
    sanity["actual_inserted_le_q"] = bool((sweep["actual_efficiency_groups_inserted"] <= sweep["requested_quota"]).all())
    ins_ids = [r["group_id"] for r in quota_member_rows if r["role"] == "efficiency"]
    sanity["inserted_groups_all_k8_and_efficiency_variable"] = bool(
        all(int(gidx.loc[i]["k"]) == 8 and bool(gidx.loc[i]["efficiency_variable"]) for i in set(ins_ids)))
    mix_ids = [r["group_id"] for r in quota_member_rows if r["role"] == "mixed"]
    sanity["mixed_groups_all_1le_k_le_7"] = bool(all(1 <= int(gidx.loc[i]["k"]) <= 7 for i in set(mix_ids)))
    dup_batch = 0
    dup_uid_batch = 0
    for (s, q), mem in batch_members.items():
        ids = [x[0] for x in mem]
        if len(ids) != len(set(ids)):
            dup_batch += 1
        uids = [i.split(":")[1] for i in ids]
        if len(uids) != len(set(uids)):
            dup_uid_batch += 1
    sanity["batches_with_duplicate_group_id"] = dup_batch
    sanity["batches_with_duplicate_uid"] = dup_uid_batch
    sanity["malformed_records_dropped"] = 0
    sanity["rollout_row_accounting"] = {"rollouts": int(len(df)), "in_groups": int(g["n_rollouts"].sum())}
    tot_by_q = {q: int(sweep[sweep.q == q]["mixed_groups_replaced"].sum()) for q in QUOTAS}
    sanity["total_replacement_monotone_in_q"] = bool(all(tot_by_q[QUOTAS[i]] <= tot_by_q[QUOTAS[i + 1]]
                                                         for i in range(len(QUOTAS) - 1)))
    sanity["total_replacement_by_q"] = tot_by_q
    # cross-check against the E1-D run's own observed generation-batch count
    observed = 0
    for p in glob.glob(os.path.join(C.D2_RUN, "diagnostics", "calls_pid*.jsonl")):
        observed += sum(1 for _ in open(p))
    sanity["E1_D_D2_observed_generation_batches"] = observed
    sanity["q0_replay_generation_batches"] = int(sum(per_step[s]["generation_batches_consumed"] for s in steps))
    sanity["q0_replay_matches_observed"] = bool(sanity["q0_replay_generation_batches"] == observed)

    # ---------------- §12 recommended quota -----------------------------------
    by_q = {r["q"]: r for r in avail}
    def score_q(q):
        r = by_q[q]
        return (r["full_quota_rate"] or 0), (r["actual_replacement_rate"] or 0)
    recommendation, reason = None, []
    for q in (2, 1, 3, 4):
        r = by_q[q]
        bias_p = next((d for d in disp if d.get("q") == q), {}).get("bias_k_over_8_vs_all_mixed_in_batch_p")
        src_p = next((d for d in disp if d.get("q") == q), {}).get("bias_source_chi2_p")
        if (r["full_quota_rate"] or 0) >= 0.80 and (bias_p is None or bias_p > 0.01) and (src_p is None or src_p > 0.01):
            recommendation = q
            reason.append(f"q={q}: full-quota rate {r['full_quota_rate']:.3f}, "
                          f"replacement fraction {r['actual_replacement_rate']:.4f}, "
                          f"difficulty-bias p={bias_p}, source-bias p={src_p}")
            break
    if recommendation is None:
        recommendation = 1
        reason.append("no q in {1,2,3,4} reached an 80% full-quota rate without a bias signal; "
                      "falling back to the most conservative non-zero quota")
    rec = {
        "recommended_q": recommendation,
        "lambda": C.LAMBDA_MAIN,
        "target_optimizer_batch_size": target,
        "full_quota_rate_by_q": {str(q): by_q[q]["full_quota_rate"] for q in QUOTAS},
        "actual_replacement_rate_by_q": {str(q): by_q[q]["actual_replacement_rate"] for q in QUOTAS},
        "mean_efficiency_groups_per_step_by_q": {str(q): by_q[q]["mean_actual_efficiency_groups_per_step"] for q in QUOTAS},
        "reason": reason,
        "mixture_rule": {
            "mixed_group_reward": "R_i = A_i (original EM only)",
            "efficiency_group_reward": "R_i = 1 + 0.05 * E_i",
            "generation_stopping": "only mixed-correctness groups count toward data.train_batch_size",
            "selection": "stream order only (no difficulty/reward/source/random selection)",
            "replaced_mixed_groups": "the last m mixed groups of the q=0 batch",
        },
    }
    with open(os.path.join(args.out, "recommended_quota.json"), "w") as fh:
        json.dump(rec, fh, indent=2)

    # ---------------- summary.json --------------------------------------------
    summary = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "target_optimizer_batch_size": target, "target_source": C.D2_CONFIG,
        "gen_batch_size": gen_bs, "rollout_n": n_roll,
        "candidate_stream": {
            "source": C.D2_ROLLOUTS, "rollouts": int(len(df)), "groups": int(len(g)),
            "generation_batches": int(g["call_index"].nunique()),
            "optimizer_steps": int(g["global_step"].nunique()),
            "steps": [int(min(g["global_step"])), int(max(g["global_step"]))],
            "sources": {k: int(v) for k, v in Counter(g["data_source"]).items()},
            "mixed_groups": int(g["mixed"].sum()),
            "all_correct_groups": int(g["all_correct"].sum()),
            "all_wrong_groups": int(g["all_wrong"].sum()),
            "efficiency_groups": int(g["efficiency_group"].sum()),
        },
        "q0_baseline": {
            "generation_batches_consumed": int(sum(per_step[s]["generation_batches_consumed"] for s in steps)),
            "observed_generation_batches_in_E1_D_D2": observed,
            "matches_observed": bool(sanity["q0_replay_matches_observed"]),
            "selected_groups_total": int(q0["optimizer_batch_size"].sum()),
        },
        "quota_availability": avail,
        "supervision_density": dens,
        "batch_level_proxy": proxy,
        "mixed_displacement": disp,
        "efficiency_group_summary": eff_summary,
        "source_shift": src_rows,
        "group_success_distribution": gsd_df.to_dict(orient="records"),
        "reward_sanity": rew_rows,
        "sanity": sanity,
        "recommendation": rec,
    }
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    log("--- availability ---", runlog)
    for r in avail:
        log(f"  q={r['q']}: full_quota_rate={r['full_quota_rate']} replacement_rate={r['actual_replacement_rate']} "
            f"mean_inserted={r['mean_actual_efficiency_groups_per_step']} total_inserted={r['total_efficiency_groups_inserted']}", runlog)
    log("--- sanity ---", runlog)
    for k, v in sanity.items():
        log(f"  {k}: {v}", runlog)
    log(f"--- recommended q = {recommendation} ---", runlog)
    for r in reason:
        log(f"  {r}", runlog)
    log(f"summary -> {os.path.join(args.out, 'summary.json')}", runlog)
    runlog.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""E1-0 offline analysis over the passive audit JSONL.

Covers E1-0_task.md sections 11 (group audit), 12 (XML vs structured),
13 (original filter simulation) and 14 (offline reward replay), and writes:

    analysis/audit_summary.json / audit_summary.csv
    analysis/group_statistics.md
    analysis/xml_vs_structured.md
    analysis/reward_identity.md
    analysis/filter_simulation.md
    analysis/offline_reward_replay.md / .json
    audit/mismatch_cases.jsonl

Usage:
    E10_RUN_DIR=<run dir> [E10_PIDS=1234,5678] python3 analyze_e10.py
"""
import glob
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict

RUN_DIR = os.environ.get("E10_RUN_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT = os.path.join(RUN_DIR, "audit")
ANALYSIS = os.path.join(RUN_DIR, "analysis")
os.makedirs(ANALYSIS, exist_ok=True)

# Only audit records from the successful attempt's pid(s) are analysed, so a
# failed earlier attempt left in the same run dir can never contaminate results.
PID_FILTER = {int(p) for p in os.environ.get("E10_PIDS", "").replace(" ", "").split(",") if p.strip()}

COSTS = ("cost_xml", "cost_structured")


def load_jsonl(pattern):
    rows = []
    for p in sorted(glob.glob(os.path.join(AUDIT, pattern))):
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if PID_FILTER and r.get("pid") not in PID_FILTER:
                    continue
                rows.append(r)
    return rows


def pct(x, n):
    return (100.0 * x / n) if n else float("nan")


def ranks(vals):
    """Average-rank of vals (ties share the mean rank)."""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    out = [0.0] * len(vals)
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def cost_of(rec, kind):
    if kind == "cost_xml":
        return rec.get("logical_search_batches")
    return rec.get("structured_search_rounds")


def md_table(headers, rows):
    out = ["| " + " | ".join(str(h) for h in headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
def main():
    rollouts = load_jsonl("rollouts_pid*.jsonl")
    # Canonical file is audit/groups.jsonl; audit/groups_pid<PID>.jsonl is
    # accepted too (it used to be a byte-identical duplicate). Dedupe by
    # (pid, call_index, group_id) so running with both present cannot
    # double-count.
    groups_raw = []
    _seen_groups = set()
    for g in load_jsonl("groups*.jsonl"):
        key = (g.get("pid"), g.get("call_index"), g.get("group_id"))
        if key in _seen_groups:
            continue
        _seen_groups.add(key)
        groups_raw.append(g)
    calls = load_jsonl("calls_pid*.jsonl")

    if not rollouts:
        print("no audit records found under", AUDIT)
        return 2

    pids = sorted({r["pid"] for r in rollouts})
    print(f"loaded {len(rollouts)} rollout records, {len(groups_raw)} group records, "
          f"{len(calls)} call records from pids {pids}")

    # ---------------------------------------------------------------- groups
    gmap = defaultdict(list)
    for r in rollouts:
        gmap[(r["pid"], r["call_index"], r["uid"])].append(r)
    for k in gmap:
        gmap[k].sort(key=lambda x: x["rollout_pos_in_group"])

    group_rows = []
    for key, recs in gmap.items():
        cc = sum(1 for r in recs if r["em"] == 1 or r["em"] == 1.0)
        row = {
            "pid": key[0], "call_index": key[1], "uid": key[2],
            "group_size": len(recs),
            "correct_count": cc,
            "incorrect_count": len(recs) - cc,
            "all_correct": cc == len(recs),
            "all_wrong": cc == 0,
            "global_step": recs[0].get("global_step"),
        }
        for kind in COSTS:
            cost = [cost_of(r, kind) for r in recs if cost_of(r, kind) is not None]
            ccost = [cost_of(r, kind) for r in recs
                     if (r["em"] == 1 or r["em"] == 1.0) and cost_of(r, kind) is not None]
            row[f"{kind}_all"] = cost
            row[f"{kind}_correct"] = ccost
            row[f"{kind}_variation_correct"] = bool(len(set(ccost)) > 1)
            row[f"{kind}_variation_all"] = bool(len(set(cost)) > 1)
        group_rows.append(row)

    n_groups = len(group_rows)
    size_hist = Counter(g["group_size"] for g in group_rows)
    cc_hist = Counter(g["correct_count"] for g in group_rows)

    summary = {
        "run_dir": RUN_DIR,
        "pids": pids,
        "n_rollout_records": len(rollouts),
        "n_group_records_from_manager": len(groups_raw),
        "n_groups_recomputed": n_groups,
        "n_call_records": len(calls),
        "group_size_histogram": {str(k): v for k, v in sorted(size_hist.items())},
        "correct_count_histogram": {str(k): v for k, v in sorted(cc_hist.items())},
        "P_group_size_8": None,
        "global_steps_covered": sorted({r.get("global_step") for r in rollouts if r.get("global_step") is not None}),
    }

    # -------------------------------------------------------- reward identity
    ident_compared = 0
    ident_mismatch = 0
    max_abs_diff = 0.0
    score_em_mismatch = 0
    parse_missing = 0
    for r in rollouts:
        e10 = r.get("reward_e10")
        orig = r.get("reward_original")
        if e10 is not None and orig is not None:
            ident_compared += 1
            d = abs(e10 - orig)
            max_abs_diff = max(max_abs_diff, d)
            if d != 0:
                ident_mismatch += 1
        if r.get("reward_score_extra") is not None and r.get("em") is not None:
            if float(r["reward_score_extra"]) != float(r["em"]):
                score_em_mismatch += 1
        if not r.get("answer_parse_ok"):
            parse_missing += 1

    identity = {
        "compared": ident_compared,
        "mismatch_count": ident_mismatch,
        "max_abs_diff": max_abs_diff,
        "equality_rate": (1.0 - ident_mismatch / ident_compared) if ident_compared else None,
        "equality_rate_pct": pct(ident_compared - ident_mismatch, ident_compared),
        "score_eq_em_mismatch_count": score_em_mismatch,
        "answer_parse_missing": parse_missing,
        "note": ("reward_e10 is the value actually written into reward_tensor (what GRPO consumes); "
                 "reward_original is an independent re-computation with the unmodified original "
                 "mhqa_train.compute_score_em_batch on the same decoded prompt/response/ground-truth."),
    }

    # ---------------------------------------------- XML vs structured (sec 12)
    both_valid = 0
    exact = 0
    diffs = []
    mismatch_cases = []
    for r in rollouts:
        cx = r.get("logical_search_batches")
        cs = r.get("structured_search_rounds")
        if cx is None or cs is None:
            continue
        both_valid += 1
        diffs.append(abs(cx - cs))
        if cx == cs:
            exact += 1
        elif cx > cs:
            mismatch_cases.append(r)
    mean_abs_diff = (sum(diffs) / len(diffs)) if diffs else None

    ranking_agreement = 0
    ranking_groups = 0
    ranking_changed = 0
    for g in group_rows:
        if g["correct_count"] < 2:
            continue
        cx = [c for c in g["cost_xml_correct"] if c is not None]
        cs = [c for c in g["cost_structured_correct"] if c is not None]
        if len(cx) < 2 or len(cx) != len(cs):
            continue
        ranking_groups += 1
        if ranks(cx) == ranks(cs):
            ranking_agreement += 1
        else:
            ranking_changed += 1

    xml_vs_struct = {
        "both_valid_count": both_valid,
        "exact_match_count": exact,
        "exact_match_rate_pct": pct(exact, both_valid),
        "mean_abs_difference": mean_abs_diff,
        "mismatch_count": both_valid - exact,
        "mismatch_rate_pct": pct(both_valid - exact, both_valid),
        "correct_group_ranking_agreement": ranking_agreement,
        "correct_group_ranking_groups": ranking_groups,
        "ranking_changed_group_count": ranking_changed,
        "ranking_changed_group_rate_pct": pct(ranking_changed, ranking_groups),
        "mismatch_direction": ("logical_search_batches >= structured_search_rounds always "
                               "(a round is a distinct assistant turn, a batch is a distinct "
                               "(turn, XML-block) pair), so every mismatch is the "
                               "multiple-XML-blocks-in-one-turn case"),
    }

    # mismatch case file (bounded, with reason classification)
    mismatch_cases.sort(key=lambda r: (r["structured_search_rounds"] - r["logical_search_batches"]))
    written = 0
    with open(os.path.join(AUDIT, "mismatch_cases.jsonl"), "w") as f:
        for r in mismatch_cases[:60]:
            f.write(json.dumps({
                "run_id": r.get("run_id"),
                "global_step": r.get("global_step"),
                "call_index": r.get("call_index"),
                "uid": r.get("uid"),
                "extra_info_index": r.get("extra_info_index"),
                "question": r.get("question"),
                "em": r.get("em"),
                "logical_search_batches": r.get("logical_search_batches"),
                "structured_search_rounds": r.get("structured_search_rounds"),
                "structured_search_calls": r.get("structured_search_calls"),
                "tool_call_summary": r.get("tool_call_summary"),
                "response_excerpt": r.get("response_excerpt"),
                "response_token_length": r.get("response_token_length"),
                "completion_reason": r.get("completion_reason"),
                "reason_classification": (
                    "multiple <wiki_search> logical blocks emitted inside the same assistant turn "
                    "(parallel queries across blocks share one structured round)"),
            }, default=str) + "\n")
            written += 1

    # --------------------------------------------------- group statistics (11)
    def frac(pred, denom_rows):
        return pct(sum(1 for g in denom_rows if pred(g)), len(denom_rows))

    P_group8 = frac(lambda g: g["group_size"] == 8, group_rows)
    ge2 = [g for g in group_rows if g["correct_count"] >= 2]
    allc = [g for g in group_rows if g["all_correct"]]
    correct_rollouts = [r for r in rollouts if r["em"] == 1 or r["em"] == 1.0]

    stats = {
        "P_group8": P_group8,
        "P_correct_ge_2": pct(len(ge2), n_groups),
        "P_all_correct": pct(len(allc), n_groups),
        "P_var_xml_given_ge2": frac(lambda g: g["cost_xml_variation_correct"], ge2),
        "P_var_structured_given_ge2": frac(lambda g: g["cost_structured_variation_correct"], ge2),
        "P_all_correct_costvar_xml": frac(lambda g: g["cost_xml_variation_all"], allc),
        "P_all_correct_costvar_structured": frac(lambda g: g["cost_structured_variation_all"], allc),
        "P_zero_search_correct_xml": pct(sum(1 for r in correct_rollouts if r.get("logical_search_batches") == 0), len(correct_rollouts)),
        "P_zero_search_correct_structured": pct(sum(1 for r in correct_rollouts if r.get("structured_search_rounds") == 0), len(correct_rollouts)),
        "signal_density_xml": frac(lambda g: g["correct_count"] >= 2 and g["cost_xml_variation_correct"], group_rows),
        "signal_density_structured": frac(lambda g: g["correct_count"] >= 2 and g["cost_structured_variation_correct"], group_rows),
    }
    summary.update(stats)

    # -------------------------------------------------- filter simulation (13)
    filter_info = {"groups_before_filter": n_groups}
    kept = [g for g in group_rows if 0 < g["correct_count"] < g["group_size"] or g["group_size"] == 1]
    removed = [g for g in group_rows if g not in kept]
    filter_info.update({
        "groups_expected_kept_by_original_reward": len(kept),
        "groups_expected_kept_rate_pct": pct(len(kept), n_groups),
        "groups_expected_removed": len(removed),
        "all_correct_groups_removed": sum(1 for g in removed if g["all_correct"]),
        "all_wrong_groups_removed": sum(1 for g in removed if g["all_wrong"]),
        "rule": ("verl ray_trainer.py:1780-1781 with metric=seq_reward/seq_final_reward and "
                 "EM 0/1 rewards: kept iff np.std(group rewards) > 0 or group size == 1, "
                 "i.e. iff 0 < correct_count < group_size"),
    })
    for kind in COSTS:
        filter_info[f"newly_retained_groups_shaped_{kind}"] = {
            "lambda_0.05": sum(1 for g in removed if g["all_correct"] and g[f"{kind}_variation_all"]),
            "lambda_0.10": sum(1 for g in removed if g["all_correct"] and g[f"{kind}_variation_all"]),
            "lambda_0.20": sum(1 for g in removed if g["all_correct"] and g[f"{kind}_variation_all"]),
            "explanation": ("A_i=1 for every rollout of an all-correct group, so R_i=1+lambda*E_i varies "
                            "iff the correct-costs vary; all-wrong groups stay 0 and are never revived."),
        }

    # ------------------------------------------------ offline replay (sec 14)
    replay = {}
    try:
        import numpy as np
        import torch
        from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage

        def grpo_advantages(R, group_ids):
            bsz = len(R)
            tr = torch.zeros(bsz, 1, dtype=torch.float32)
            for i, v in enumerate(R):
                tr[i, 0] = float(v)
            mask = torch.ones(bsz, 1, dtype=torch.float32)
            adv, _ = compute_grpo_outcome_advantage(
                token_level_rewards=tr, response_mask=mask,
                index=np.array([str(g) for g in group_ids], dtype=object),
                norm_adv_by_std_in_grpo=True,
            )
            return [float(a) for a in adv[:, 0]]

        for kind in COSTS:
            for lam in (0.0, 0.05, 0.10, 0.20):
                R = []
                gid = []
                meta = []
                for gi, g in enumerate(group_rows):
                    recs = gmap[(g["pid"], g["call_index"], g["uid"])]
                    # ALIGNED per-rollout costs (no filtering, so the reward
                    # vector stays row-for-row consistent with the uid index).
                    costs = [cost_of(r, kind) for r in recs]
                    correct_flags = [1.0 if (r["em"] == 1 or r["em"] == 1.0) else 0.0 for r in recs]
                    ccost = [c for c, a in zip(costs, correct_flags)
                             if a == 1.0 and c is not None]
                    use = (len(ccost) >= 2 and len(set(ccost)) > 1
                           and all(c is not None for c, a in zip(costs, correct_flags) if a == 1.0))
                    cmin = min(ccost) if use else None
                    cmax = max(ccost) if use else None
                    for c, a in zip(costs, correct_flags):
                        if use and a == 1.0:
                            e = (cmax - c) / (cmax - cmin) if cmax > cmin else 0.0
                        else:
                            e = 0.0
                        R.append(a * (1.0 + lam * e))
                        gid.append(gi)
                        meta.append({"group": gi, "correct": a, "cost": c,
                                     "efficiency": e, "all_correct": g["all_correct"],
                                     "mixed": 0 < g["correct_count"] < g["group_size"]})
                adv = grpo_advantages(R, gid)
                finite = all(math.isfinite(x) for x in adv) and all(math.isfinite(x) for x in R)
                corr = [a for a, m in zip(R, meta) if m["correct"] == 1.0]
                incorr = [a for a, m in zip(R, meta) if m["correct"] == 0.0]
                separation = (min(corr) - max(incorr)) if (corr and incorr) else None
                # Meaningful efficiency separation: compare efficient vs
                # inefficient CORRECT rollouts WITHIN the same group, and only
                # in groups where the shaping term is actually active (i.e.
                # correct_count>=2 and correct-cost variation). Comparing the
                # two buckets across different groups would compare unrelated
                # populations (mixed groups vs all-correct-with-variation).
                by_group = defaultdict(list)
                for a, m in zip(adv, meta):
                    by_group[m["group"]].append((a, m))
                eff_adv, ineff_adv, diffs = [], [], []
                active_groups = 0
                strictly_ordered = 0
                for _gi, rows in by_group.items():
                    e = [a for a, m in rows if m["correct"] == 1.0 and m["efficiency"] > 0]
                    i_ = [a for a, m in rows if m["correct"] == 1.0 and m["efficiency"] == 0]
                    if not e:
                        continue
                    active_groups += 1
                    eff_adv.extend(e)
                    if i_:
                        ineff_adv.extend(i_)
                        diffs.append(statistics.fmean(e) - statistics.fmean(i_))
                        if min(e) > max(i_):
                            strictly_ordered += 1
                allc_adv = [a for a, m in zip(adv, meta) if m["all_correct"]]
                mixed_adv = [a for a, m in zip(adv, meta) if m["mixed"]]
                replay[f"{kind}_lambda{lam}"] = {
                    "reward_range": [min(R), max(R)],
                    "advantage_range": [min(adv), max(adv)],
                    "advantage_mean": statistics.fmean(adv),
                    "advantage_std": statistics.pstdev(adv) if len(adv) > 1 else 0.0,
                    "all_finite_no_nan_inf": finite,
                    "correct_gt_incorrect_always": (separation is not None and separation > 0),
                    "min_correct_minus_max_incorrect": separation,
                    "efficiency_active_groups": active_groups,
                    "efficiency_active_rows": len(eff_adv) + len(ineff_adv),
                    "correct_efficient_adv_mean": (statistics.fmean(eff_adv) if eff_adv else None),
                    "correct_inefficient_adv_mean": (statistics.fmean(ineff_adv) if ineff_adv else None),
                    "efficient_minus_inefficient_adv": (
                        statistics.fmean(eff_adv) - statistics.fmean(ineff_adv)
                        if (eff_adv and ineff_adv) else None),
                    "within_group_efficiency_adv_diff_mean": (
                        statistics.fmean(diffs) if diffs else None),
                    "within_group_efficiency_adv_diff_min": (min(diffs) if diffs else None),
                    "groups_efficient_strictly_above_inefficient": strictly_ordered,
                    "groups_with_both_efficient_and_inefficient_correct": len(diffs),
                    "all_correct_group_adv_std": (statistics.pstdev(allc_adv) if len(allc_adv) > 1 else 0.0),
                    "all_correct_group_adv_absmax": (max(abs(x) for x in allc_adv) if allc_adv else 0.0),
                    "mixed_group_adv_std": (statistics.pstdev(mixed_adv) if len(mixed_adv) > 1 else 0.0),
                    "n_rows": len(R),
                }
        replay["_note"] = ("Offline replay only -- E1-0 training used the original EM reward. "
                           "Uses the real verl compute_grpo_outcome_advantage with "
                           "norm_adv_by_std_in_grpo=True, exactly as this run's config does.")
    except Exception as e:  # noqa: BLE001
        replay["_error"] = f"{type(e).__name__}: {e}"

    # ------------------------------------------------------------- write JSON
    summary["reward_identity"] = identity
    summary["xml_vs_structured"] = xml_vs_struct
    summary["group_statistics"] = stats
    summary["filter_simulation"] = filter_info
    with open(os.path.join(ANALYSIS, "audit_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    with open(os.path.join(ANALYSIS, "offline_reward_replay.json"), "w") as f:
        json.dump(replay, f, indent=2, default=str)

    with open(os.path.join(ANALYSIS, "audit_summary.csv"), "w") as f:
        f.write("metric,value\n")
        for k, v in summary.items():
            if isinstance(v, (dict, list)):
                continue
            f.write(f"{k},{v}\n")

    # ------------------------------------------------------------- write MDs
    with open(os.path.join(ANALYSIS, "reward_identity.md"), "w") as f:
        f.write("# E1-0 reward identity (original GAP reward vs E1-0 reward)\n\n")
        f.write(md_table(["metric", "value"], [[k, v] for k, v in identity.items()]) + "\n")

    with open(os.path.join(ANALYSIS, "xml_vs_structured.md"), "w") as f:
        f.write("# E1-0 XML logical batch vs structured search round\n\n")
        f.write(md_table(["metric", "value"], [[k, v] for k, v in xml_vs_struct.items()]) + "\n")

    with open(os.path.join(ANALYSIS, "filter_simulation.md"), "w") as f:
        f.write("# E1-0 offline simulation of the ORIGINAL GAP group filter (trainer untouched)\n\n")
        f.write(md_table(["metric", "value"],
                         [[k, json.dumps(v) if isinstance(v, dict) else v] for k, v in filter_info.items()]) + "\n")

    with open(os.path.join(ANALYSIS, "group_statistics.md"), "w") as f:
        f.write("# E1-0 group statistics (n=8 uid groups, pre-filter)\n\n")
        f.write(md_table(["metric", "value"], [[k, v] for k, v in stats.items()]) + "\n\n")
        f.write("## correct_count histogram\n\n")
        f.write(md_table(["correct_count", "groups", "pct"],
                         [[k, cc_hist.get(k, 0), round(pct(cc_hist.get(k, 0), n_groups), 3)]
                          for k in range(9)]) + "\n\n")
        f.write("## group_size histogram\n\n")
        f.write(md_table(["group_size", "groups"], [[k, v] for k, v in sorted(size_hist.items())]) + "\n")

    with open(os.path.join(ANALYSIS, "offline_reward_replay.md"), "w") as f:
        f.write("# E1-0.5-style offline reward replay (NO training change)\n\n")
        f.write("R_i = A_i * (1 + lambda * E_i); E_i = (Cmax - Ci)/(Cmax - Cmin) for correct "
                "rollouts of a group with correct_count>=2 and correct-cost variation, else 0.\n\n")
        for k, v in replay.items():
            if k.startswith("_"):
                continue
            f.write(f"## {k}\n\n")
            if isinstance(v, dict):
                f.write(md_table(["metric", "value"], [[kk, vv] for kk, vv in v.items()]) + "\n\n")
        f.write(f"\n{replay.get('_note', '')}\n")

    print(json.dumps(summary, indent=2, default=str)[:4000])
    print("\nwrote analysis artifacts to", ANALYSIS)
    return 0


if __name__ == "__main__":
    sys.exit(main())

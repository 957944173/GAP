#!/usr/bin/env python3
"""Assemble E1-E/analysis/summary.json (E1-E_task.md section 26) from existing artifacts.

Reads (writes nothing outside E1-E/):
  * E1-E/eval_results/step{70,120}_*/eval_summary.json   (E1-E's own evaluations)
  * E1-E/analysis/step{70,120}_integrity.json            (runtime quota semantics + checkpoints)
  * E1-E/analysis/d1_gate.json                           (section 15 automatic decision)
  * reference evaluations: GAP60/120/180, E1-0 step70, E1-A step70 (E1-B/evaluations),
    E1-B step120 (E1-B/evaluations/step120_20260911_235134) and E1-D step70/120
    (E1-D/eval_results/step*)  -- the mandated four-way baselines
  * <comparison dir>/comparison.json + paired_comparison_*.json
"""
import argparse
import glob
import json
import os

REPO = "/data01/wyy/Graph-Agent-Planning"
E1E = os.path.join(REPO, "E1-E")

REFS = {
    "e1_0_step70": os.path.join(REPO, "E1-B/evaluations/e10_step70_reference/eval_summary.json"),
    "e1a_step70": os.path.join(REPO, "E1-B/evaluations/e1a_step70_reference/eval_summary.json"),
    "gap60": os.path.join(REPO, "E1-B/evaluations/gap_step60_reference/eval_summary.json"),
    "gap120": os.path.join(REPO, "E1-B/evaluations/gap_step120_reference/eval_summary.json"),
    "gap180": os.path.join(REPO, "E1-B/evaluations/gap_step180_reference/eval_summary.json"),
    "e1b120": os.path.join(REPO, "E1-B/evaluations/step120_20260911_235134/eval_summary.json"),
    "e1d70": "GLOB:E1-D/eval_results/step70_*/eval_summary.json",
    "e1d120": "GLOB:E1-D/eval_results/step120_*/eval_summary.json",
}

TOL_EM = 0.005          # 0.5 pp micro EM, the E1-E task's "no meaningful loss" band
ROUNDS_TARGET = -0.10   # -10% search rounds


def jload(p, default=None):
    try:
        with open(p) as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return default


def eval_block(path):
    s = jload(path)
    if not s:
        return None
    acc = s.get("accuracy", {})
    o = s.get("overall", {})
    return {
        "path": path,
        "label": s.get("label"),
        "macro_em": acc.get("macro_em_over_benchmarks"),
        "micro_em": acc.get("micro_em_over_samples"),
        "total_samples": acc.get("total_samples"),
        "per_benchmark_em": {k: v.get("em") for k, v in acc.get("per_benchmark", {}).items()},
        "search_rounds": o.get("retrieval_rounds_mean"),
        "queries": o.get("search_queries_mean"),
        "queries_per_round": (o.get("search_queries_mean") / o.get("retrieval_rounds_mean")
                              if o.get("retrieval_rounds_mean") else None),
        "parallel_factor": o.get("parallel_factor_mean"),
        "parallel_sample_rate": o.get("parallel_sample_rate"),
        "zero_search_rate": o.get("zero_search_rate"),
        "no_answer_tag_rate": o.get("no_answer_tag_rate"),
    }


def delta(a, b, key):
    if not a or not b or a.get(key) is None or b.get(key) is None:
        return None
    return b[key] - a[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default=os.path.join(E1E, "analysis"))
    ap.add_argument("--eval-root", default=os.path.join(E1E, "eval_results"))
    ap.add_argument("--comparison", default=None)
    ap.add_argument("--out", default=os.path.join(E1E, "analysis", "summary.json"))
    args = ap.parse_args()

    def newest(pattern):
        c = sorted(glob.glob(pattern))
        return c[-1] if c else None

    ev = {}
    e1e_path = {}
    for step in (70, 120):
        d = newest(os.path.join(args.eval_root, f"step{step}_*"))
        e1e_path[step] = os.path.join(d, "eval_summary.json") if d else None
        ev[f"e1e{step}"] = eval_block(e1e_path[step]) if e1e_path[step] else None
    for k, p in REFS.items():
        if p.startswith("GLOB:"):
            d = newest(os.path.join(REPO, p[5:]))
            ev[k] = eval_block(d) if d else None
        else:
            ev[k] = eval_block(p)

    integ = {70: jload(os.path.join(args.analysis, "step70_integrity.json")),
             120: jload(os.path.join(args.analysis, "step120_integrity.json"))}

    def mech(step):
        q = (integ.get(step) or {}).get("quota_semantics") or {}
        return {
            "available": q.get("available"),
            "steps_recorded": q.get("n_step_records"),
            "all_steps_recorded": q.get("all_steps_recorded"),
            "quota_fill_rate": q.get("quota_fill_rate"),
            "steps_full_quota": q.get("steps_full_quota"),
            "steps_used": q.get("steps_used"),
            "total_inserted_efficiency_groups": q.get("total_inserted_efficiency_groups"),
            "total_displaced_mixed_groups": q.get("total_displaced_mixed_groups"),
            "total_eff_shortfall": q.get("total_eff_shortfall"),
            "steps_needing_injection": q.get("steps_needing_injection"),
            "mixed_groups": q.get("mixed_groups"),
            "efficiency_groups": q.get("efficiency_groups"),
            "total_generation_batches": q.get("total_generation_batches"),
            "mean_generation_batches_per_step": q.get("mean_generation_batches_per_step"),
            "candidate_groups_seen": (q.get("n_group_records") or 0) + (q.get("n_rollout_records") or 0),
            "zero_search_share_train": q.get("zero_search_share"),
            "malformed_metadata_share": q.get("malformed_metadata_share"),
            "quota_semantics_ok": q.get("quota_semantics_ok"),
            "quota_error_lines": q.get("quota_error_lines"),
            "mixed_group_reward_not_equal_em": q.get("rollouts_mixed_reward_not_equal_em"),
            "efficiency_group_reward_or_em_wrong": q.get("rollouts_efficiency_reward_or_em_wrong"),
            "wrong_rollout_nonzero_reward": q.get("rollouts_wrong_with_nonzero_reward"),
            "advantage_nan_total": q.get("advantage_nan_total"),
            "advantage_inf_total": q.get("advantage_inf_total"),
            "per_step": q.get("per_step"),
        }

    comp = jload(os.path.join(args.comparison, "comparison.json")) if args.comparison else None
    paired = {}
    if args.comparison:
        for p in glob.glob(os.path.join(args.comparison, "paired_comparison_*.json")):
            pj = jload(p)
            if pj:
                paired[pj.get("label", os.path.basename(p))] = pj

    def pair_block(label):
        pj = paired.get(label)
        if not pj:
            return None
        o = pj["overall"]
        return {
            "paired_prompts": o["n_paired_prompts"],
            "rounds_baseline": o["rounds_baseline_mean"], "rounds_e1e": o["rounds_e1e_mean"],
            "rounds_delta": o["rounds_delta"],
            "rounds_delta_pct": (o["rounds_delta"] / o["rounds_baseline_mean"] * 100
                                 if o["rounds_baseline_mean"] else None),
            "rounds_decreased_prompts": o["rounds_decreased_prompts"],
            "rounds_increased_prompts": o["rounds_increased_prompts"],
            "rounds_unchanged_prompts": o.get("rounds_unchanged_prompts"),
            "sign_test_p_rounds": o["sign_test_p_rounds"],
            "queries_baseline": o["queries_baseline_mean"], "queries_e1e": o["queries_e1e_mean"],
            "queries_delta": o["queries_delta"],
            "em_baseline": o["em_baseline_mean"], "em_e1e": o["em_e1e_mean"],
            "em_delta": o["em_delta"],
            "em_up_prompts": o["em_up_prompts"], "em_down_prompts": o["em_down_prompts"],
            "sign_test_p_em": o["sign_test_p_em"],
        }

    def cmp_block(base_key, test_key, label):
        b, t = ev.get(base_key), ev.get(test_key)
        return {
            "baseline": base_key, "test": test_key,
            "delta_macro_em": delta(b, t, "macro_em"),
            "delta_micro_em": delta(b, t, "micro_em"),
            "delta_rounds": delta(b, t, "search_rounds"),
            "delta_rounds_pct": (delta(b, t, "search_rounds") / b["search_rounds"] * 100
                                 if b and t and b.get("search_rounds") else None),
            "delta_queries": delta(b, t, "queries"),
            "delta_queries_pct": (delta(b, t, "queries") / b["queries"] * 100
                                  if b and t and b.get("queries") else None),
            "delta_parallel_factor": delta(b, t, "parallel_factor"),
            "delta_zero_search": delta(b, t, "zero_search_rate"),
            "paired": pair_block(label),
        }

    comparisons = {
        "e1e120_vs_gap120": cmp_block("gap120", "e1e120", "e1e120_vs_gap120"),
        "e1e120_vs_e1b120": cmp_block("e1b120", "e1e120", "e1e120_vs_e1b120"),
        "e1e120_vs_e1d120": cmp_block("e1d120", "e1e120", "e1e120_vs_e1d120"),
        "e1e120_vs_e1e70": cmp_block("e1e70", "e1e120", "e1e120_vs_e1e70"),
        "e1e70_vs_e1_0_step70": cmp_block("e1_0_step70", "e1e70", "e1e70_vs_gap60"),
        "e1e70_vs_e1a70": cmp_block("e1a_step70", "e1e70", "e1e70_vs_e1a70"),
        "e1e70_vs_e1d70": cmp_block("e1d70", "e1e70", "e1e70_vs_e1d70"),
    }

    # ---- RQ-facing decision flags (mechanical; the report explains them) -------------
    c_gap = comparisons["e1e120_vs_gap120"]
    c_d = comparisons["e1e120_vs_e1d120"]
    c_b = comparisons["e1e120_vs_e1b120"]
    d_em_gap, d_r_gap = c_gap.get("delta_micro_em"), c_gap.get("delta_rounds_pct")
    d_em_d, d_r_d = c_d.get("delta_micro_em"), c_d.get("delta_rounds")
    flags = {
        "delta_EM_at_least_minus_0.5pp_vs_GAP120": (d_em_gap is not None and d_em_gap >= -TOL_EM),
        "delta_rounds_at_most_minus_10pct_vs_GAP120": (d_r_gap is not None and d_r_gap <= ROUNDS_TARGET * 100),
        "pareto_dominates_E1D120": (d_em_d is not None and d_r_d is not None
                                    and d_em_d >= -TOL_EM and d_r_d <= 0),
        "keeps_E1D_EM_and_beats_E1D_rounds": (d_em_d is not None and d_r_d is not None
                                              and d_em_d >= -TOL_EM and d_r_d < 0),
        "more_efficient_than_E1B120": (c_b.get("delta_rounds") is not None
                                       and c_b["delta_rounds"] < 0),
        "higher_EM_than_E1B120": (c_b.get("delta_micro_em") is not None
                                  and c_b["delta_micro_em"] > 0),
    }
    n_true = sum(1 for v in flags.values() if v)
    if flags["delta_EM_at_least_minus_0.5pp_vs_GAP120"] and \
            flags["delta_rounds_at_most_minus_10pct_vs_GAP120"] and \
            flags["pareto_dominates_E1D120"]:
        case = "STRONG_PASS"
    elif (flags["delta_rounds_at_most_minus_10pct_vs_GAP120"]
          and flags["delta_EM_at_least_minus_0.5pp_vs_GAP120"]):
        case = "PASS_PROMISING"
    elif ev["e1e120"] and n_true >= 2:
        case = "PASS_PROMISING"
    elif ev["e1e120"]:
        case = "NEGATIVE_BUT_INFORMATIVE"
    else:
        case = "INCOMPLETE"

    summary = {
        "experiment_status": "COMPLETE" if ev["e1e70"] and ev["e1e120"] else "PARTIAL",
        "design": {
            "quota_q": 2,
            "lambda": 0.05,
            "filter_metric": "e1e_quota_metric",
            "reward_mixed_group": "R_i = A_i (original EM, no shaping)",
            "reward_efficiency_group": "R_i = 1 + 0.05 * E_i (quota-selected all-correct k=8)",
            "reward_other_groups": "dropped by the filter (reward 0, never trained on)",
            "stopping_rule": "only mixed groups (1<=k<=7) advance the stopping counter; "
                             "efficiency groups are cached in stream order and injected at the "
                             "stopping batch as mixed[:32-m] + cache[:m], m = min(2, |cache|)",
            "cost": "logical_search_batches",
        },
        "checkpoints": {
            "start": os.path.join(REPO, "experiments/DAPO-GAP3B-MHQA-Agent-4gpu/global_step_60"),
            "step70": os.path.join(REPO, "experiments/DAPO-GAP3B-MHQA-Agent-E1E-q2-step60to120-4gpu/global_step_70"),
            "step120": os.path.join(REPO, "experiments/DAPO-GAP3B-MHQA-Agent-E1E-q2-step60to120-4gpu/global_step_120"),
        },
        "step70": dict(ev["e1e70"] or {}, **mech(70)),
        "step120": dict(ev["e1e120"] or {}, **mech(120)),
        "references": {k: v for k, v in ev.items() if k not in ("e1e70", "e1e120")},
        "comparisons": comparisons,
        "four_way_step120": (comp or {}).get("four_way_step120"),
        "d1_gate": (jload(os.path.join(args.analysis, "d1_gate.json")) or {}).get("gate"),
        "decision_flags": flags,
        "case": case,
        "causal_matrix": {
            "cell_GAP120": {"filter": "none (original GAP)", "efficiency_supervision": "none",
                            "reward": "EM", "run": "original GAP step120"},
            "cell_E1_B120": {"filter": "EM std > 0 (mixed only)", "efficiency_supervision": "none",
                             "reward": "EM", "run": "E1-B step120"},
            "cell_E1_D120": {"filter": "EM std > 0 (mixed only)",
                             "efficiency_supervision": "shaping inside mixed groups",
                             "reward": "R_i = A_i (1 + 0.05 E_i), all retained groups",
                             "run": "E1-D step120"},
            "cell_E1_E120": {"filter": "e1e_quota_metric (mixed EM + injected all-correct "
                                       "efficiency groups)",
                             "efficiency_supervision": "quota-gated (q = 2, 6.25% of groups)",
                             "reward": "mixed: A_i; gated efficiency group: 1 + 0.05 E_i",
                             "run": "E1-E step120 (this experiment)"},
        },
        "hypothesis_verdict": None,       # filled by e1e_report_numbers.py / FINAL_REPORT.md
        "recommended_next_experiment": None,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("references", "step70", "step120")},
                     indent=2, default=str)[:5000])
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

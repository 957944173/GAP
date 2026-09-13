#!/usr/bin/env python3
"""Assemble E1-D/analysis/summary.json (E1-D_task.md §26) from the artifacts that exist.

Reads (never writes outside E1-D/):
  * E1-D/eval_results/step{70,120}_*/eval_summary.json        (E1-D's own evaluations)
  * E1-D/analysis/step{70,120}_integrity.json                 (filter decoupling + checkpoints)
  * E1-B/evaluations/*/eval_summary.json                      (GAP/E1-A/E1-B references)
  * <comparison dir>/comparison.json, paired_comparison_*.json
"""
import argparse
import glob
import json
import os

REPO = "/data01/wyy/Graph-Agent-Planning"
E1D = os.path.join(REPO, "E1-D")

REFS = {
    "gap70_from_e1_0": os.path.join(REPO, "E1-B/evaluations/e10_step70_reference/eval_summary.json"),
    "e1a70": os.path.join(REPO, "E1-B/evaluations/e1a_step70_reference/eval_summary.json"),
    "gap120": os.path.join(REPO, "E1-B/evaluations/gap_step120_reference/eval_summary.json"),
    "gap60": os.path.join(REPO, "E1-B/evaluations/gap_step60_reference/eval_summary.json"),
    "gap180": os.path.join(REPO, "E1-B/evaluations/gap_step180_reference/eval_summary.json"),
    "e1b120": os.path.join(REPO, "E1-B/evaluations/step120_20260911_235134/eval_summary.json"),
}


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
    ap.add_argument("--analysis", default=os.path.join(E1D, "analysis"))
    ap.add_argument("--comparison", default=None)
    ap.add_argument("--out", default=os.path.join(E1D, "analysis", "summary.json"))
    args = ap.parse_args()

    def newest(pattern):
        c = sorted(glob.glob(pattern))
        return c[-1] if c else None

    e1d70_path = os.path.join(E1D, "eval_results",
                              "step70_" + os.path.basename(
                                  (newest(os.path.join(E1D, "eval_results", "step70_*")) or "step70_x")).split("step70_")[-1],
                              "eval_summary.json") if newest(os.path.join(E1D, "eval_results", "step70_*")) else None
    e1d120_path = os.path.join(E1D, "eval_results",
                               "step120_" + os.path.basename(
                                   (newest(os.path.join(E1D, "eval_results", "step120_*")) or "step120_x")).split("step120_")[-1],
                               "eval_summary.json") if newest(os.path.join(E1D, "eval_results", "step120_*")) else None

    ev = {"e1d70": eval_block(e1d70_path) if e1d70_path else None,
          "e1d120": eval_block(e1d120_path) if e1d120_path else None}
    for k, p in REFS.items():
        ev[k] = eval_block(p)

    integ = {70: jload(os.path.join(args.analysis, "step70_integrity.json")),
             120: jload(os.path.join(args.analysis, "step120_integrity.json"))}

    def mech(step):
        f = (integ.get(step) or {}).get("filter_decoupling") or {}
        return {
            "retained_groups": f.get("actual_retained"),
            "efficiency_active_retained": f.get("efficiency_active_retained_actual"),
            "all_correct_retained": f.get("retained_all_correct"),
            "mixed_retained": f.get("retained_mixed"),
            "generation_batches": f.get("generation_batches"),
            "efficiency_supervision_density": f.get("efficiency_supervision_density_actual"),
            "counterfactual_shaped_retained": f.get("counterfactual_shaped_retained"),
            "filter_decoupling_confirmed": f.get("filter_decoupling_confirmed"),
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
            "rounds_baseline": o["rounds_baseline_mean"], "rounds_e1d": o["rounds_e1d_mean"],
            "rounds_delta": o["rounds_delta"],
            "rounds_delta_pct": (o["rounds_delta"] / o["rounds_baseline_mean"] * 100
                                 if o["rounds_baseline_mean"] else None),
            "rounds_decreased_prompts": o["rounds_decreased_prompts"],
            "rounds_increased_prompts": o["rounds_increased_prompts"],
            "sign_test_p_rounds": o["sign_test_p_rounds"],
            "em_baseline": o["em_baseline_mean"], "em_e1d": o["em_e1d_mean"], "em_delta": o["em_delta"],
            "em_up_prompts": o["em_up_prompts"], "em_down_prompts": o["em_down_prompts"],
            "sign_test_p_em": o["sign_test_p_em"],
        }

    summary = {
        "experiment_status": "COMPLETE" if ev["e1d70"] and ev["e1d120"] else "PARTIAL",
        "filter_metric": "em",
        "reward_formula": "R_i = A_i * (1 + 0.05 * E_i); wrong rollout -> 0",
        "lambda": 0.05,
        "start_checkpoint": os.path.join(REPO, "experiments/DAPO-GAP3B-MHQA-Agent-4gpu/global_step_60"),
        "step70_checkpoint": os.path.join(REPO, "experiments/DAPO-GAP3B-MHQA-Agent-E1D-step60to120-4gpu/global_step_70"),
        "step120_checkpoint": os.path.join(REPO, "experiments/DAPO-GAP3B-MHQA-Agent-E1D-step60to120-4gpu/global_step_120"),
        "step70": dict(ev["e1d70"] or {}, **mech(70)),
        "step120": dict(ev["e1d120"] or {}, **mech(120)),
        "references": {k: v for k, v in ev.items() if k not in ("e1d70", "e1d120")},
        "comparisons": {
            "e1d70_vs_gap70": {"delta_macro_em": delta(ev["gap70_from_e1_0"], ev["e1d70"], "macro_em"),
                               "delta_micro_em": delta(ev["gap70_from_e1_0"], ev["e1d70"], "micro_em"),
                               "delta_rounds": delta(ev["gap70_from_e1_0"], ev["e1d70"], "search_rounds"),
                               "delta_queries": delta(ev["gap70_from_e1_0"], ev["e1d70"], "queries"),
                               "paired": pair_block("e1d70_vs_gap60")},
            "e1d70_vs_e1a70": {"delta_macro_em": delta(ev["e1a70"], ev["e1d70"], "macro_em"),
                               "delta_micro_em": delta(ev["e1a70"], ev["e1d70"], "micro_em"),
                               "delta_rounds": delta(ev["e1a70"], ev["e1d70"], "search_rounds"),
                               "delta_queries": delta(ev["e1a70"], ev["e1d70"], "queries"),
                               "paired": pair_block("e1d70_vs_e1a70")},
            "e1d120_vs_gap120": {"delta_macro_em": delta(ev["gap120"], ev["e1d120"], "macro_em"),
                                 "delta_micro_em": delta(ev["gap120"], ev["e1d120"], "micro_em"),
                                 "delta_rounds": delta(ev["gap120"], ev["e1d120"], "search_rounds"),
                                 "delta_queries": delta(ev["gap120"], ev["e1d120"], "queries"),
                                 "delta_parallel_factor": delta(ev["gap120"], ev["e1d120"], "parallel_factor"),
                                 "paired": pair_block("e1d120_vs_gap120")},
            "e1d120_vs_e1b120": {"delta_macro_em": delta(ev["e1b120"], ev["e1d120"], "macro_em"),
                                 "delta_micro_em": delta(ev["e1b120"], ev["e1d120"], "micro_em"),
                                 "delta_rounds": delta(ev["e1b120"], ev["e1d120"], "search_rounds"),
                                 "delta_queries": delta(ev["e1b120"], ev["e1d120"], "queries"),
                                 "delta_parallel_factor": delta(ev["e1b120"], ev["e1d120"], "parallel_factor"),
                                 "paired": pair_block("e1d120_vs_e1b120")},
        },
        "causal_matrix": {
            "cell_GAP": {"filter": "EM", "reward": "EM", "run": "original GAP step120"},
            "cell_E1_D": {"filter": "EM", "reward": "shaped", "run": "E1-D step120"},
            "cell_E1_B": {"filter": "shaped", "reward": "shaped", "run": "E1-B step120"},
        },
        "hypothesis_verdict": None,       # filled by the report / e1d_verdict.py
        "recommended_next_experiment": None,
    }
    # simple mechanical CASE classification (the report explains it)
    try:
        d_gap = summary["comparisons"]["e1d120_vs_gap120"]
        d_e1b = summary["comparisons"]["e1d120_vs_e1b120"]
        em_back = d_gap["delta_micro_em"] is not None and d_gap["delta_micro_em"] >= -0.005
        eff_kept = d_e1b["delta_rounds"] is not None and d_e1b["delta_rounds"] <= -0.05
        if em_back and eff_kept:
            case = "CASE_B"
        elif em_back and not eff_kept:
            case = "CASE_D"
        elif (not em_back) and eff_kept:
            case = "CASE_C"
        else:
            case = "CASE_A"
        summary["case"] = case
    except Exception:  # noqa: BLE001
        summary["case"] = None

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("references",)},
                     indent=2, default=str)[:4000])
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

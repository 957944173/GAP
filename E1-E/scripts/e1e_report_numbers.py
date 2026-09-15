#!/usr/bin/env python3
"""Print every number needed to write E1-E/FINAL_REPORT.md, from the artifacts on disk.

Usage: python3 E1-E/scripts/e1e_report_numbers.py [--comparison <dir>]
Read-only; writes nothing.
"""
import argparse
import glob
import json
import os

REPO = "/data01/wyy/Graph-Agent-Planning"
E1D = os.path.join(REPO, "E1-E")


def j(p, d=None):
    try:
        with open(p) as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return d


def newest(pat):
    c = sorted(glob.glob(pat))
    return c[-1] if c else None


def show(title, obj, keep=None):
    print(f"\n### {title}")
    if obj is None:
        print("  (missing)")
        return
    if keep:
        obj = {k: obj.get(k) for k in keep}
    print(json.dumps(obj, indent=2, default=str)[:3000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparison", default=None)
    a = ap.parse_args()
    comp_dir = a.comparison or newest(os.path.join(E1D, "analysis", "comparison_*"))

    print("=" * 90)
    print("E1-E report numbers")
    print("=" * 90)

    for step in (70, 120):
        integ = j(os.path.join(E1D, "analysis", f"step{step}_integrity.json"))
        if not integ:
            continue
        show(f"integrity step{step}", integ, ["phase", "integrity_ok", "state_resume_ok"])
        show(f"  checkpoint step{step}", integ.get("checkpoint"), ["exists", "data_pt", "tracker"])
        tl = integ.get("training_log") or {}
        show(f"  training_log step{step}", tl,
             ["steps_seen", "missing_steps", "all_steps_present", "resume_line_present",
              "metrics_parsed", "non_finite_metrics", "error_pattern_counts_before_final_val"])
        show(f"  state_resume step{step}", integ.get("state_resume_evidence"))
        show(f"  quota_semantics step{step}", integ.get("quota_semantics"))
        show(f"  log_vs_diagnostics step{step}", integ.get("log_vs_diagnostics"))

    for tag, pat in (("step70", os.path.join(E1D, "eval_results", "step70_*", "eval_summary.json")),
                     ("step120", os.path.join(E1D, "eval_results", "step120_*", "eval_summary.json"))):
        p = newest(pat)
        s = j(p)
        if not s:
            continue
        acc = s.get("accuracy", {})
        o = s.get("overall", {})
        print(f"\n### E1-E {tag} evaluation  ({p})")
        print(f"  macro_em={acc.get('macro_em_over_benchmarks')}  micro_em={acc.get('micro_em_over_samples')}  n={acc.get('total_samples')}")
        print("  per-benchmark EM: " + json.dumps({k: v.get("em") for k, v in acc.get("per_benchmark", {}).items()}))
        print("  overall: " + json.dumps({k: o.get(k) for k in
              ("retrieval_rounds_mean", "search_queries_mean", "parallel_factor_mean",
               "parallel_sample_rate", "zero_search_rate", "no_answer_tag_rate")}))

    refs = {
        "e1d70": newest(os.path.join(REPO, "E1-D/eval_results/step70_*", "eval_summary.json")),
        "e1d120": newest(os.path.join(REPO, "E1-D/eval_results/step120_*", "eval_summary.json")),
        "gap60": os.path.join(REPO, "E1-B/evaluations/gap_step60_reference/eval_summary.json"),
        "e10_70(GAP70)": os.path.join(REPO, "E1-B/evaluations/e10_step70_reference/eval_summary.json"),
        "e1a70": os.path.join(REPO, "E1-B/evaluations/e1a_step70_reference/eval_summary.json"),
        "gap120": os.path.join(REPO, "E1-B/evaluations/gap_step120_reference/eval_summary.json"),
        "gap180": os.path.join(REPO, "E1-B/evaluations/gap_step180_reference/eval_summary.json"),
        "e1b120": os.path.join(REPO, "E1-B/evaluations/step120_20260911_235134/eval_summary.json"),
    }
    print("\n### reference evaluations")
    for k, p in refs.items():
        s = j(p)
        if not s:
            print(f"  {k}: missing")
            continue
        acc, o = s.get("accuracy", {}), s.get("overall", {})
        print(f"  {k:14} macroEM={acc.get('macro_em_over_benchmarks'):.4f} microEM={acc.get('micro_em_over_samples'):.4f} "
              f"rounds={o.get('retrieval_rounds_mean'):.4f} queries={o.get('search_queries_mean'):.4f} "
              f"pf={o.get('parallel_factor_mean'):.4f} no-answer={o.get('no_answer_tag_rate')}")

    if comp_dir:
        c = j(os.path.join(comp_dir, "comparison.json"))
        if c:
            print(f"\n### multi-way comparison ({comp_dir})")
            for k, v in (c.get("runs") or {}).items():
                if not v:
                    continue
                o = v.get("overall") or {}
                print(f"  {k:34} macroEM={v.get('macro_em')} microEM={v.get('micro_em')} "
                      f"rounds={o.get('retrieval_rounds_mean')} queries={o.get('search_queries_mean')} "
                      f"pf={o.get('parallel_factor_mean')}")
            for key in ("vs_gap_step120", "vs_e1b_step120", "vs_e1d_step120",
                        "vs_e1a_step70", "vs_e10_step70", "vs_gap_step60", "e1e120_vs_e1e70"):
                b = c.get(key)
                if not b:
                    continue
                print(f"  [{key}] baseline={b['baseline_run']} test={b['test_run']} "
                      f"dmacroEM={b['macro_em']['delta']} dmicroEM={b['micro_em']['delta']} "
                      f"drounds={(b['behavior'].get('retrieval_rounds_mean') or {}).get('delta')} "
                      f"({(b['behavior'].get('retrieval_rounds_mean') or {}).get('pct')}%) "
                      f"dqueries={(b['behavior'].get('search_queries_mean') or {}).get('delta')}")
        fw = c.get("four_way_step120")
        if fw:
            print("\n### four-way step120")
            for row in fw:
                print(f"  {row['role']:14} microEM={row['micro_em']} macroEM={row['macro_em']} "
                      f"rounds={row['rounds']} queries={row['queries']}")
        for p in sorted(glob.glob(os.path.join(comp_dir, "paired_comparison_*.json"))):
            pj = j(p)
            if not pj:
                continue
            o = pj["overall"]
            print(f"\n### paired {pj.get('label')}  (n={o['n_paired_prompts']})")
            print(f"  rounds {o['rounds_baseline_mean']:.4f} -> {o['rounds_e1e_mean']:.4f} "
                  f"({o['rounds_delta']:+.4f}, {o['rounds_delta']/max(o['rounds_baseline_mean'],1e-9)*100:+.2f}%) "
                  f"dec={o['rounds_decreased_prompts']} inc={o['rounds_increased_prompts']} p={o['sign_test_p_rounds']:.3g}")
            print(f"  EM     {o['em_baseline_mean']:.4f} -> {o['em_e1e_mean']:.4f} ({o['em_delta']:+.4f}) "
                  f"up={o['em_up_prompts']} down={o['em_down_prompts']} p={o['sign_test_p_em']:.3g}")
            for ds, v in pj["per_benchmark"].items():
                print(f"    {ds:18} n={v['n_paired_prompts']:>6} drounds={v['rounds_delta']:+.3f} dEM={v['em_delta']:+.4f}")

    for tag in ("D1", "D2"):
        for base in ("vs_E1_0", "vs_E1_B"):
            p = os.path.join(E1D, "analysis", f"{tag}_{base}", "training_summary.json")
            t = j(p)
            if not t:
                continue
            e, b = t.get("e1d") or {}, t.get("baseline") or {}
            print(f"\n### training diagnostics {tag} {base}")
            for k in ("n_rollouts", "n_groups", "n_calls", "em_rate", "reward_mean", "efficiency_mean",
                      "efficiency_nonzero_rate", "rounds_mean", "queries_mean", "parallel_factor_mean",
                      "turns_mean", "tokens_mean", "zero_search_rate", "filter_retention_baseline",
                      "efficiency_active_groups", "efficiency_supervision_ratio",
                      "mixed_em_mean", "mixed_reward_mean", "efficiency_reward_mean",
                      "efficiency_cost_gap_mean", "efficiency_rounds_mean", "group_role_counts"):
                print(f"  {k:34} E1-E={e.get(k)}  baseline={b.get(k)}")
            q = t.get("quota")
            if q:
                print("  quota: " + json.dumps(q))
        sp = os.path.join(E1D, "analysis", f"{tag}_vs_E1_0", "safety_analysis.json")
        s = j(sp)
        if s:
            print(f"\n### safety {tag}")
            print("  query_packing: " + json.dumps({k: (v if not isinstance(v, dict) else {
                kk: v.get(kk) for kk in ("active_groups", "packing_suspect_groups", "packing_suspect_ratio",
                                         "efficient_rounds_mean", "inefficient_rounds_mean",
                                         "efficient_queries_mean", "inefficient_queries_mean")})
                for k, v in (s.get("query_packing") or {}).items() if k != "definition"}))
            print("  zero_search: " + json.dumps(s.get("zero_search")))
            print("  malformed: " + json.dumps({k: v for k, v in (s.get("malformed") or {}).items() if k != "complete_reason_top"}))

    s = j(os.path.join(E1D, "analysis", "summary.json"))
    if s:
        print("\n### summary.json")
        print(json.dumps({k: v for k, v in s.items() if k not in ("references", "comparisons")}, indent=2, default=str)[:2500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Collect E1-E evaluation artifacts (E1-E_task.md section 6).

Scans every evaluation directory under E1-E/evaluations/ that contains an
`eval_summary.json` (produced by e1e_analyze_eval.py) and writes:

    E1-E/evaluations/results.json           -- full nested results (per label: accuracy,
                                               per-benchmark behavior, overall metrics)
    E1-E/evaluations/benchmark_scores.csv   -- one row per (label, benchmark): EM / n
    E1-E/evaluations/behavior_metrics.csv   -- one row per (label, benchmark) with the
                                               search-behaviour metrics
    E1-E/evaluations/overall_metrics.csv    -- one row per label, macro/micro EM + behaviour

Usage: python3 e1e_collect_evaluation_artifacts.py [--eval-root DIR] [--out DIR]
"""
import argparse
import csv
import glob
import json
import os

BEHAVIOR_FIELDS = [
    "response_tokens_mean", "assistant_turns_mean_all_messages", "retrieval_rounds_mean",
    "search_queries_mean", "parallel_factor_mean", "parallel_sample_rate",
    "multi_query_round_rate", "zero_search_rate", "no_answer_tag_rate", "empty_output_rate",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    e1e_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = args.eval_root or os.path.join(e1e_root, "evaluations")
    out = args.out or root
    os.makedirs(out, exist_ok=True)

    entries = []
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(d):
            continue
        sp = os.path.join(d, "eval_summary.json")
        if not os.path.exists(sp):
            continue
        with open(sp) as fh:
            s = json.load(fh)
        entries.append({"dir": d, "name": os.path.basename(d), "summary": s})

    if not entries:
        print("no eval_summary.json found under", root)
        return 1

    results = {}
    for e in entries:
        s = e["summary"]
        results[e["name"]] = {
            "label": s.get("label", e["name"]),
            "dir": os.path.relpath(e["dir"], e1e_root),
            "accuracy": s.get("accuracy"),
            "benchmark_results": s.get("benchmark_results"),
            "overall": s.get("overall"),
        }
    with open(os.path.join(out, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2, default=str)

    with open(os.path.join(out, "benchmark_scores.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "label", "benchmark", "n_samples", "em", "em_from_log"])
        for e in entries:
            s = e["summary"]
            label = s.get("label", e["name"])
            acc = s.get("accuracy", {}).get("per_benchmark", {})
            for bench, v in acc.items():
                w.writerow([e["name"], label, bench, v.get("n"), v.get("em"), v.get("em_from_log")])

    with open(os.path.join(out, "behavior_metrics.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "label", "benchmark"] + BEHAVIOR_FIELDS)
        for e in entries:
            s = e["summary"]
            label = s.get("label", e["name"])
            for bench, v in (s.get("benchmark_results") or {}).items():
                w.writerow([e["name"], label, bench] + [v.get(f) for f in BEHAVIOR_FIELDS])
            o = s.get("overall") or {}
            w.writerow([e["name"], label, "OVERALL"] + [o.get(f) for f in BEHAVIOR_FIELDS])

    with open(os.path.join(out, "overall_metrics.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "label", "macro_em", "micro_em", "total_samples"] + BEHAVIOR_FIELDS)
        for e in entries:
            s = e["summary"]
            label = s.get("label", e["name"])
            acc = s.get("accuracy", {})
            o = s.get("overall") or {}
            w.writerow([e["name"], label,
                        acc.get("macro_em_over_benchmarks"), acc.get("micro_em_over_samples"),
                        acc.get("total_samples")] + [o.get(f) for f in BEHAVIOR_FIELDS])

    print(json.dumps({"out": out, "runs": [e["name"] for e in entries]}, indent=2))
    for e in entries:
        s = e["summary"]
        o = s.get("overall") or {}
        print(f"{e['name']:>28}  macroEM={s['accuracy'].get('macro_em_over_benchmarks'):.4f}  "
              f"rounds={o.get('retrieval_rounds_mean')}  queries={o.get('search_queries_mean')}  "
              f"pf={o.get('parallel_factor_mean')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

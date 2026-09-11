#!/usr/bin/env python3
"""E1-A vs baseline 7-benchmark comparison.

Usage:
    python3 e1a_compare_eval.py --baseline <dir with eval_summary.json> \
                                --e1a <dir with eval_summary.json> --out <dir>
Writes comparison.json and comparison.md.
"""
import argparse
import json
import os

ORDER = ["nq", "triviaqa", "popqa", "hotpotqa", "2wikimultihopqa", "musique", "bamboogle"]


def load(d):
    with open(os.path.join(d, "eval_summary.json")) as f:
        return json.load(f)


def fnum(v, n=4):
    return f"{v:.{n}f}" if isinstance(v, (int, float)) else "n/a"


def delta(a, b, n=4):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return f"{a - b:+.{n}f}"
    return "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--e1a", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    b = load(args.baseline)
    a = load(args.e1a)

    br, ar = b["benchmark_results"], a["benchmark_results"]
    ds_list = [d for d in ORDER if d in ar] + [d for d in ar if d not in ORDER]

    comparison = {"baseline_label": b["label"], "e1a_label": a["label"],
                  "per_benchmark": {}, "overall": {}}
    metrics = ["em", "response_tokens_mean", "assistant_turns_mean_all_messages",
               "retrieval_rounds_mean", "search_queries_mean", "parallel_factor_mean",
               "parallel_sample_rate", "multi_query_round_rate", "zero_search_rate",
               "no_answer_tag_rate"]
    for ds in ds_list:
        rec = {}
        for m in metrics:
            av = ar.get(ds, {}).get(m)
            bv = br.get(ds, {}).get(m)
            rec[m] = {"e1a": av, "baseline": bv,
                      "delta": (av - bv) if isinstance(av, (int, float)) and isinstance(bv, (int, float)) else None}
        comparison["per_benchmark"][ds] = rec

    ao, bo = a["overall"], b["overall"]
    for m in ["macro_em", "micro_em", "n_samples", "retrieval_rounds_mean", "search_queries_mean",
              "parallel_factor_mean", "parallel_sample_rate", "zero_search_rate", "no_answer_tag_rate"]:
        av, bv = ao.get(m), bo.get(m)
        comparison["overall"][m] = {"e1a": av, "baseline": bv,
                                    "delta": (av - bv) if isinstance(av, (int, float)) and isinstance(bv, (int, float)) else None}

    # how many of the 7 benchmarks improved / regressed on EM
    improved = [ds for ds in ds_list if (ar.get(ds, {}).get("em") is not None and br.get(ds, {}).get("em") is not None
                                         and ar[ds]["em"] > br[ds]["em"])]
    regressed = [ds for ds in ds_list if (ar.get(ds, {}).get("em") is not None and br.get(ds, {}).get("em") is not None
                                          and ar[ds]["em"] < br[ds]["em"])]
    comparison["em_improved_benchmarks"] = improved
    comparison["em_regressed_benchmarks"] = regressed
    em_b = [br[ds]["em"] for ds in ds_list if br.get(ds, {}).get("em") is not None]
    em_a = [ar[ds]["em"] for ds in ds_list if ar.get(ds, {}).get("em") is not None]
    if em_a and em_b:
        comparison["macro_em_e1a_recomputed"] = sum(em_a) / len(em_a)
        comparison["macro_em_baseline_recomputed"] = sum(em_b) / len(em_b)

    with open(os.path.join(args.out, "comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2, default=str)

    lines = ["# E1-A vs baseline (GAP step60->70 original reward): 7 benchmarks", "",
             f"- E1-A (shaped reward): `{a['label']}`",
             f"- baseline (original EM): `{b['label']}`", "",
             "## EM", "", "| benchmark | E1-A EM | baseline EM | delta |", "|---|---|---|---|"]
    for ds in ds_list:
        e = ar.get(ds, {}).get("em"); bb = br.get(ds, {}).get("em")
        lines.append(f"| {ds} | {fnum(e)} | {fnum(bb)} | {delta(e, bb)} |")
    lines.append(f"| **macro (mean of benchmarks)** | {fnum(ao.get('macro_em'))} | {fnum(bo.get('macro_em'))} | {delta(ao.get('macro_em'), bo.get('macro_em'))} |")
    lines.append(f"| **micro (mean of samples)** | {fnum(ao.get('micro_em'))} | {fnum(bo.get('micro_em'))} | {delta(ao.get('micro_em'), bo.get('micro_em'))} |")
    lines += ["", "## Trajectory / search metrics (macro over benchmarks)", "",
              "| metric | E1-A | baseline | delta |", "|---|---|---|---|"]
    for m, label in [("retrieval_rounds_mean", "search rounds"),
                     ("search_queries_mean", "search queries"),
                     ("assistant_turns_mean_all_messages", "turns (all assistant msgs)"),
                     ("response_tokens_mean", "response tokens"),
                     ("parallel_factor_mean", "parallel factor"),
                     ("parallel_sample_rate", "parallel sample rate"),
                     ("multi_query_round_rate", "multi-query-round rate"),
                     ("zero_search_rate", "zero-search rate"),
                     ("no_answer_tag_rate", "no <answer> rate")]:
        av = [ar[ds][m] for ds in ds_list if ar.get(ds, {}).get(m) is not None]
        bv = [br[ds][m] for ds in ds_list if br.get(ds, {}).get(m) is not None]
        am = sum(av) / len(av) if av else None
        bm = sum(bv) / len(bv) if bv else None
        lines.append(f"| {label} | {fnum(am)} | {fnum(bm)} | {delta(am, bm)} |")
    lines += ["", f"- EM improved on: {', '.join(improved) if improved else 'none'}",
              f"- EM regressed on: {', '.join(regressed) if regressed else 'none'}"]
    with open(os.path.join(args.out, "comparison.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

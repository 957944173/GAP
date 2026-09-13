#!/usr/bin/env python3
"""Multi-way comparison of all E1-D / reference evaluations (E1-D_task.md 8/9/10).

Reads every `eval_summary.json` under E1-D/evaluations/ (E1-D step80..120 plus the
re-aggregated references gap_step60, gap_step120, gap_step180, e1a_step70, e10_step70)
and emits:

    E1-D/evaluations/comparison/comparison.json
    E1-D/evaluations/comparison/comparison.md            (full table, all runs)
    E1-D/evaluations/comparison/vs_gap_step120.md        (E1-D step120 vs GAP baseline)
    E1-D/evaluations/comparison/vs_e1a_step70.md         (E1-D step120 vs E1-A start)
    E1-D/evaluations/comparison/per_benchmark_deltas.md  (per-benchmark behaviour deltas)

Comparison logic: every run uses the identical protocol (7 benchmarks, greedy, 51,201
samples), so macro-EM and the behaviour means are directly comparable.  Deltas are
reported in absolute terms and as percentages.
"""
import argparse
import glob
import json
import os

BEHAV = ["retrieval_rounds_mean", "search_queries_mean", "parallel_factor_mean",
         "parallel_sample_rate", "assistant_turns_mean_all_messages", "response_tokens_mean",
         "zero_search_rate", "no_answer_tag_rate", "empty_output_rate"]

BENCH_ORDER = ["nq", "triviaqa", "popqa", "hotpotqa", "2wikimultihopqa", "musique", "bamboogle"]


def fnum(v, n=4):
    return "-" if v is None else f"{v:.{n}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    e1d_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = args.eval_root or os.path.join(e1d_root, "evaluations")
    out = args.out or os.path.join(root, "comparison")
    os.makedirs(out, exist_ok=True)

    runs = {}
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        sp = os.path.join(d, "eval_summary.json")
        if not os.path.isdir(d) or not os.path.exists(sp):
            continue
        with open(sp) as fh:
            s = json.load(fh)
        overall = dict(s.get("overall", {}))
        # the evaluator's `overall` block omits turns/tokens; derive them as
        # sample-weighted means over the per-benchmark blocks so the report tables
        # are complete and consistent
        br = s.get("benchmark_results", {}) or {}
        n_tot = sum((v.get("n_samples") or 0) for v in br.values())
        if n_tot:
            for key in ("assistant_turns_mean_all_messages", "response_tokens_mean"):
                overall[key] = sum((v.get(key) or 0) * (v.get("n_samples") or 0)
                                   for v in br.values()) / n_tot
        runs[os.path.basename(d)] = {
            "label": s.get("label"),
            "macro_em": s["accuracy"].get("macro_em_over_benchmarks"),
            "micro_em": s["accuracy"].get("micro_em_over_samples"),
            "total_samples": s["accuracy"].get("total_samples"),
            "per_benchmark": {b: v.get("em") for b, v in s["accuracy"].get("per_benchmark", {}).items()},
            "behavior": s.get("benchmark_results", {}),
            "overall": overall,
        }

    def pick(prefix):
        for k in sorted(runs):
            if k.startswith(prefix):
                return k
        return None

    gap120 = pick("gap_step120_reference")
    gap60 = pick("gap_step60_reference")
    gap180 = pick("gap_step180_reference")
    e1a70 = pick("e1a_step70_reference")
    e10_70 = pick("e10_step70_reference")
    e1d_steps = {int(k.split("_")[0][4:]): k for k in runs if k.startswith("step")}

    result = {"runs": runs, "roles": {"gap_step120": gap120, "gap_step60": gap60,
                                      "gap_step180": gap180, "e1a_step70": e1a70,
                                      "e10_step70": e10_70,
                                      "e1d_steps": {str(k): v for k, v in sorted(e1d_steps.items())}}}

    def delta_block(base_key, test_key):
        if not base_key or not test_key or base_key not in runs or test_key not in runs:
            return None
        b, t = runs[base_key], runs[test_key]
        d = {"baseline_run": base_key, "test_run": test_key,
             "macro_em": {"baseline": b["macro_em"], "test": t["macro_em"],
                          "delta": (t["macro_em"] - b["macro_em"]) if (b["macro_em"] is not None and t["macro_em"] is not None) else None},
             "micro_em": {"baseline": b["micro_em"], "test": t["micro_em"],
                          "delta": (t["micro_em"] - b["micro_em"]) if (b["micro_em"] is not None and t["micro_em"] is not None) else None},
             "per_benchmark": {}, "behavior": {}}
        for bench in BENCH_ORDER:
            bv = b["per_benchmark"].get(bench)
            tv = t["per_benchmark"].get(bench)
            d["per_benchmark"][bench] = {"baseline": bv, "test": tv,
                                         "delta": (tv - bv) if (bv is not None and tv is not None) else None}
        for m in BEHAV:
            bv = b["overall"].get(m)
            tv = t["overall"].get(m)
            d["behavior"][m] = {"baseline": bv, "test": tv,
                                "delta": (tv - bv) if (bv is not None and tv is not None) else None,
                                "pct": ((tv - bv) / bv * 100.0) if (bv not in (None, 0) and tv is not None) else None}
        return d

    result["vs_gap_step120"] = delta_block(gap120, e1d_steps.get(120))
    result["vs_e1a_step70"] = delta_block(e1a70, e1d_steps.get(120))
    result["vs_gap_step60"] = delta_block(gap60, e1d_steps.get(120))
    result["vs_e10_step70"] = delta_block(e10_70, e1d_steps.get(120))
    # trajectory of the E1-D branch, anchored at its own start (E1-A step70)
    traj = {}
    for s in sorted(e1d_steps):
        traj[s] = delta_block(e1a70, e1d_steps[s])
    result["trajectory_from_e1a_step70"] = traj

    with open(os.path.join(out, "comparison.json"), "w") as fh:
        json.dump(result, fh, indent=2, default=str)

    def table(keys, title, fh):
        fh.write(f"## {title}\n\n")
        cols = ["run", "macro_EM", "micro_EM", "rounds", "queries", "parallel factor",
                "turns", "tokens", "zero-search", "no-answer"]
        fh.write("| " + " | ".join(cols) + " |\n")
        fh.write("|" + "---|" * len(cols) + "\n")
        for k in keys:
            if not k or k not in runs:
                continue
            r = runs[k]
            o = r["overall"]
            fh.write("| `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n".format(
                k, fnum(r["macro_em"]), fnum(r["micro_em"]),
                fnum(o.get("retrieval_rounds_mean"), 3), fnum(o.get("search_queries_mean"), 3),
                fnum(o.get("parallel_factor_mean"), 3),
                fnum(o.get("assistant_turns_mean_all_messages"), 3),
                fnum(o.get("response_tokens_mean"), 2),
                fnum(o.get("zero_search_rate"), 5), fnum(o.get("no_answer_tag_rate"), 5)))
        fh.write("\n")

    ordered_refs = [gap60, gap120, gap180, e10_70, e1a70]
    ordered_e1b = [e1d_steps[s] for s in sorted(e1d_steps)]
    with open(os.path.join(out, "comparison.md"), "w") as fh:
        fh.write("# E1-D evaluation comparison (identical protocol: 7 benchmarks, greedy, 51,201 samples)\n\n")
        table(ordered_refs + ordered_e1b, "All runs", fh)

    def delta_md(block, title, path):
        with open(path, "w") as fh:
            fh.write(f"# {title}\n\n")
            if not block:
                fh.write("(missing run)\n")
                return
            fh.write(f"baseline `{block['baseline_run']}` -> test `{block['test_run']}`\n\n")
            fh.write(f"- macro EM: {fnum(block['macro_em']['baseline'])} -> {fnum(block['macro_em']['test'])} "
                     f"(delta {fnum(block['macro_em']['delta'])}, "
                     f"{fnum((block['macro_em']['delta'] or 0)*100, 2)} pp)\n")
            fh.write(f"- micro EM: {fnum(block['micro_em']['baseline'])} -> {fnum(block['micro_em']['test'])} "
                     f"(delta {fnum(block['micro_em']['delta'])})\n\n")
            fh.write("| benchmark | baseline EM | test EM | delta (pp) |\n|---|---|---|---|\n")
            for bench in BENCH_ORDER:
                d = block["per_benchmark"].get(bench, {})
                if d.get("delta") is None:
                    continue
                fh.write(f"| {bench} | {fnum(d['baseline'])} | {fnum(d['test'])} | "
                         f"{d['delta']*100:+.2f} |\n")
            fh.write("\n| behavior metric | baseline | test | delta | delta % |\n|---|---|---|---|---|\n")
            for m in BEHAV:
                d = block["behavior"].get(m, {})
                if d.get("baseline") is None or d.get("test") is None:
                    continue
                pct = "-" if d.get("pct") is None else f"{d['pct']:+.2f}%"
                fh.write(f"| {m} | {fnum(d['baseline'], 5)} | {fnum(d['test'], 5)} | "
                         f"{fnum(d['delta'], 5)} | {pct} |\n")
            fh.write("\n")

    delta_md(result.get("vs_gap_step120"), "E1-D step120 vs GAP baseline (step120)",
             os.path.join(out, "vs_gap_step120.md"))
    delta_md(result.get("vs_e1a_step70"), "E1-D step120 vs E1-A step70 (E1-D start)",
             os.path.join(out, "vs_e1a_step70.md"))
    delta_md(result.get("vs_e10_step70"), "E1-D step120 vs E1-0 step70 (matched original-reward baseline)",
             os.path.join(out, "vs_e10_step70.md"))

    with open(os.path.join(out, "per_benchmark_deltas.md"), "w") as fh:
        fh.write("# E1-D step120 per-benchmark behavior, vs GAP step120 and E1-A step70\n\n")
        for bench in BENCH_ORDER:
            fh.write(f"## {bench}\n\n")
            fh.write("| metric | GAP step120 | E1-A step70 | E1-D step120 | d(E1B-GAP120) | d(E1B-E1A70) |\n")
            fh.write("|---|---|---|---|---|---|\n")
            row = {}
            for key, ref in (("gap", gap120), ("e1a", e1a70), ("e1b", e1d_steps.get(120))):
                row[key] = (runs.get(ref, {}).get("behavior", {}) or {}).get(bench, {}) if ref else {}
            for m in ["em"] + BEHAV:
                if m == "em":
                    vals = {k: (runs.get(v, {}).get("per_benchmark", {}) or {}).get(bench) if v else None
                            for k, v in (("gap", gap120), ("e1a", e1a70), ("e1b", e1d_steps.get(120)))}
                else:
                    vals = {k: row[k].get(m) for k in ("gap", "e1a", "e1b")}
                def dd(a, b):
                    return "-" if (a is None or b is None) else f"{b - a:+.4f}"
                fh.write(f"| {m} | {fnum(vals['gap'], 4)} | {fnum(vals['e1a'], 4)} | {fnum(vals['e1b'], 4)} | "
                         f"{dd(vals['gap'], vals['e1b'])} | {dd(vals['e1a'], vals['e1b'])} |\n")
            fh.write("\n")

    print(json.dumps({"out": out, "runs": sorted(runs), "roles": result["roles"]}, indent=2, default=str))
    if result.get("vs_gap_step120"):
        b = result["vs_gap_step120"]
        print("E1-D step120 vs GAP step120: macroEM delta",
              fnum(b["macro_em"]["delta"]), "rounds delta",
              fnum(b["behavior"]["retrieval_rounds_mean"]["delta"], 4),
              f"({fnum(b['behavior']['retrieval_rounds_mean']['pct'], 2)}%)")
    if result.get("vs_e1a_step70"):
        b = result["vs_e1a_step70"]
        print("E1-D step120 vs E1-A step70: macroEM delta", fnum(b["macro_em"]["delta"]),
              "rounds delta", fnum(b["behavior"]["retrieval_rounds_mean"]["delta"], 4),
              f"({fnum(b['behavior']['retrieval_rounds_mean']['pct'], 2)}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

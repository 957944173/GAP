#!/usr/bin/env python3
"""Paired per-prompt comparison of two eval dumps (E1-E vs matched baseline).

Both evaluations run the SAME 7 benchmarks with greedy decoding (val_kwargs
do_sample=False), so samples can be paired by (data_source, input prompt).  Pairing
removes the between-prompt variance that dominates the unpaired means and turns
"rounds went down 1.8%" into "on X% of prompts the model now uses fewer rounds,
on Y% more".

Outputs paired_comparison.json / paired_comparison.md in --out.
"""
import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict

OBS_SPLIT = re.compile(r"<observation>.*?</observation>", re.S)
TOOL_TAGS = ("<wiki_search>", "<web_search>", "<crawl_page>", "<search>", "<tool>",
             "<python>", "<calculator>")
WIKI_BLOCK = re.compile(r"<wiki_search>(.*?)</wiki_search>", re.S)
ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.S)


def load(path):
    per_key = defaultdict(list)
    with open(path, errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            ds = r.get("data_source", "?")
            inp = (r.get("input") or "")[:4000]
            out = r.get("output") or ""
            rounds = sum(1 for s in OBS_SPLIT.split(out) if any(t in s for t in TOOL_TAGS))
            queries = sum(1 for m in WIKI_BLOCK.finditer(out) if m.group(1).strip())
            per_key[(ds, inp)].append({
                "rounds": rounds, "queries": queries, "chars": len(out),
                "score": float(r.get("score", 0.0) or 0.0),
                "has_answer": bool(ANSWER_TAG.search(out)),
            })
    return per_key


def sign_test_p(k, n):
    """two-sided exact binomial p-value with p=0.5 (normal approx for large n)"""
    if n == 0:
        return None
    if n < 1000:
        from math import comb
        p = sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / (2 ** n) * 2
        return min(1.0, p)
    z = abs(k - n / 2) / math.sqrt(n / 4)
    return math.erfc(z / math.sqrt(2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-dump", required=True)
    ap.add_argument("--e1e-dump", "--e1d-dump", dest="e1e_dump", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="e1e_vs_baseline")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    b = load(args.baseline_dump)
    a = load(args.e1e_dump)

    per_ds = defaultdict(lambda: {"n": 0, "rounds_b": 0, "rounds_a": 0, "q_b": 0, "q_a": 0,
                                  "less": 0, "more": 0, "same": 0,
                                  "em_b": 0.0, "em_a": 0.0, "em_up": 0, "em_down": 0,
                                  "chars_b": 0, "chars_a": 0})
    tot = defaultdict(int)
    for key in set(b) & set(a):
        lb, la = b[key], a[key]
        if len(lb) != 1 or len(la) != 1:      # skip popqa duplicate prompts
            continue
        ds = key[0]
        rb, ra = lb[0], la[0]
        d = per_ds[ds]
        d["n"] += 1
        d["rounds_b"] += rb["rounds"]; d["rounds_a"] += ra["rounds"]
        d["q_b"] += rb["queries"]; d["q_a"] += ra["queries"]
        d["chars_b"] += rb["chars"]; d["chars_a"] += ra["chars"]
        d["em_b"] += rb["score"]; d["em_a"] += ra["score"]
        if ra["rounds"] < rb["rounds"]:
            d["less"] += 1
        elif ra["rounds"] > rb["rounds"]:
            d["more"] += 1
        else:
            d["same"] += 1
        if ra["score"] > rb["score"]:
            d["em_up"] += 1
        elif ra["score"] < rb["score"]:
            d["em_down"] += 1
        # accumulate the overall totals from the single-sample values (NOT from
        # the running per-benchmark aggregates, which would double count)
        tot["n"] += 1
        tot["rounds_b"] += rb["rounds"]; tot["rounds_a"] += ra["rounds"]
        tot["q_b"] += rb["queries"]; tot["q_a"] += ra["queries"]
        tot["chars_b"] += rb["chars"]; tot["chars_a"] += ra["chars"]
        if ra["rounds"] < rb["rounds"]:
            tot["less"] += 1
        elif ra["rounds"] > rb["rounds"]:
            tot["more"] += 1
        else:
            tot["same"] += 1
        if ra["score"] > rb["score"]:
            tot["em_up"] += 1
        elif ra["score"] < rb["score"]:
            tot["em_down"] += 1
        tot["em_b"] += rb["score"]; tot["em_a"] += ra["score"]

    out = {"label": args.label, "paired_prompts": tot["n"], "per_benchmark": {}, "overall": {}}

    def summarize(d):
        n = max(d["n"], 1)
        changed = d["less"] + d["more"]
        return {
            "n_paired_prompts": d["n"],
            "rounds_baseline_mean": d["rounds_b"] / n,
            "rounds_e1e_mean": d["rounds_a"] / n,
            "rounds_delta": (d["rounds_a"] - d["rounds_b"]) / n,
            "queries_baseline_mean": d["q_b"] / n,
            "queries_e1e_mean": d["q_a"] / n,
            "queries_delta": (d["q_a"] - d["q_b"]) / n,
            "chars_baseline_mean": d["chars_b"] / n,
            "chars_e1e_mean": d["chars_a"] / n,
            "em_baseline_mean": d["em_b"] / n,
            "em_e1e_mean": d["em_a"] / n,
            "em_delta": (d["em_a"] - d["em_b"]) / n,
            "rounds_decreased_prompts": d["less"],
            "rounds_increased_prompts": d["more"],
            "rounds_unchanged_prompts": d["same"],
            "rounds_decreased_rate": d["less"] / n,
            "rounds_increased_rate": d["more"] / n,
            "sign_test_p_rounds": sign_test_p(d["less"], max(changed, 1)),
            "em_up_prompts": d["em_up"],
            "em_down_prompts": d["em_down"],
            "sign_test_p_em": sign_test_p(d["em_up"], max(d["em_up"] + d["em_down"], 1)),
        }

    for ds, d in sorted(per_ds.items()):
        out["per_benchmark"][ds] = summarize(d)
    out["overall"] = summarize(tot)
    out["note"] = ("paired by (data_source, prompt); popqa entries with duplicate prompts are "
                   "skipped; both evals use greedy decoding so a pair is the same question "
                   "answered by the two checkpoints")

    # label-suffixed so several paired comparisons can share one output directory
    # (e.g. E1-E step120 vs GAP step120 AND vs E1-A step70) without overwriting
    for name in (f"paired_comparison_{args.label}.json", "paired_comparison.json"):
        with open(os.path.join(args.out, name), "w") as f:
            json.dump(out, f, indent=2, default=str)

    o = out["overall"]
    lines = ["# Paired per-prompt comparison (E1-E vs matched baseline)", "",
             f"- paired prompts: **{o['n_paired_prompts']:,}**", "",
             "| metric | E1-E | baseline | delta |", "|---|---|---|---|",
             f"| search rounds | {o['rounds_e1e_mean']:.4f} | {o['rounds_baseline_mean']:.4f} | "
             f"{o['rounds_delta']:+.4f} ({(o['rounds_delta']/max(o['rounds_baseline_mean'],1e-9))*100:+.2f}%) |",
             f"| search queries | {o['queries_e1e_mean']:.4f} | {o['queries_baseline_mean']:.4f} | "
             f"{o['queries_delta']:+.4f} |",
             f"| response chars | {o['chars_e1e_mean']:.1f} | {o['chars_baseline_mean']:.1f} | "
             f"{o['chars_e1e_mean']-o['chars_baseline_mean']:+.1f} |",
             f"| EM | {o['em_e1e_mean']:.4f} | {o['em_baseline_mean']:.4f} | {o['em_delta']:+.4f} |",
             "",
             f"- rounds **decreased** on {o['rounds_decreased_prompts']:,} prompts "
             f"({o['rounds_decreased_rate']*100:.2f}%), **increased** on "
             f"{o['rounds_increased_prompts']:,} ({o['rounds_increased_rate']*100:.2f}%), "
             f"unchanged on {o['rounds_unchanged_prompts']:,}",
             f"- sign test on round changes: p = {o['sign_test_p_rounds']:.3g}",
             f"- EM up on {o['em_up_prompts']:,} / down on {o['em_down_prompts']:,} prompts "
             f"(sign test p = {o['sign_test_p_em']:.3g})", "",
             "## per benchmark", "",
             "| benchmark | n | rounds E1-E | rounds base | Δrounds | decreased | increased | ΔEM |",
             "|---|---|---|---|---|---|---|---|"]
    for ds, v in out["per_benchmark"].items():
        lines.append(f"| {ds} | {v['n_paired_prompts']:,} | {v['rounds_e1e_mean']:.3f} | "
                     f"{v['rounds_baseline_mean']:.3f} | {v['rounds_delta']:+.3f} | "
                     f"{v['rounds_decreased_rate']*100:.1f}% | {v['rounds_increased_rate']*100:.1f}% | "
                     f"{v['em_delta']:+.4f} |")
    for name in (f"paired_comparison_{args.label}.md", "paired_comparison.md"):
        with open(os.path.join(args.out, name), "w") as f:
            f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

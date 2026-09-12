#!/usr/bin/env python3
"""E1-B evaluation aggregation: 7-benchmark metrics + trajectory/search diagnostics.

Reuses the repo's own conventions:
  * EM / response_length / num_turns come from the eval log's `val-core`/`val-aux`
    metrics (parsed exactly like Agent/evaluation/mhqa_agent/summarize_eval.py).
  * retrieval rounds ("search rounds") come from the val_generations dump by counting
    assistant spans between `<observation>` blocks that issue >=1 search tool call --
    the same convention summarize_eval.py uses for the paper's #Turns.
  * search queries = non-empty `<wiki_search>` blocks in the decoded response
    (= one per executed query, because SGLangRollout's custom chat template renders
    each parsed tool call as its own block).

Outputs (in --out):
    accuracy.json
    benchmark_results.json
    trajectory_metrics.json
    search_round_statistics.json
    parallel_statistics.json
    eval_summary.json
"""
import argparse
import glob
import json
import os
import re
from collections import Counter, defaultdict

ANSI = re.compile(r"\x1b\[[0-9;]*m")
OBS_SPLIT = re.compile(r"<observation>.*?</observation>", re.S)
TOOL_TAGS = ("<wiki_search>", "<web_search>", "<crawl_page>", "<search>", "<tool>",
             "<python>", "<calculator>")
WIKI_BLOCK = re.compile(r"<wiki_search>(.*?)</wiki_search>", re.S)
ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.S)

BENCH_ORDER = ["nq", "triviaqa", "popqa", "hotpotqa", "2wikimultihopqa", "musique", "bamboogle"]


def parse_metrics_line(log_path):
    if not os.path.isfile(log_path):
        return {"core": {}, "aux": {}}
    best = None
    with open(log_path, errors="ignore") as f:
        for raw in f:
            line = ANSI.sub("", raw).rstrip("\n")
            idx = line.find("step:")
            if idx != -1 and "val-core/" in line:
                best = line[idx:]
    if best is None:
        return {"core": {}, "aux": {}}
    out = {"core": defaultdict(dict), "aux": defaultdict(dict)}
    for token in best.split(" - "):
        key, sep, valstr = token.partition(":")
        if not sep:
            continue
        if key.startswith("val-core/"):
            section, rest = "core", key[len("val-core/"):]
        elif key.startswith("val-aux/"):
            section, rest = "aux", key[len("val-aux/"):]
        else:
            continue
        ds, _, sub = rest.partition("/")
        try:
            out[section][ds][sub] = float(valstr)
        except ValueError:
            continue
    return out


def get_em(core, aux, ds):
    """mean@1 reward from whichever section carries it (see summarize_eval.py)."""
    v = core.get(ds, {}).get("reward/mean@1")
    if v is None:
        v = aux.get(ds, {}).get("reward/mean@1")
    return v


def pct(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


def hist(vals, max_bucket=10):
    c = Counter()
    for v in vals:
        c[str(v) if v <= max_bucket else f">{max_bucket}"] += 1
    return dict(sorted(c.items(), key=lambda kv: (kv[0].startswith(">"), int(kv[0].lstrip(">")) if kv[0].lstrip(">").isdigit() else 0)))


def analyze_dump(dump_path):
    """Per-data_source trajectory statistics from the val generations dump."""
    agg = defaultdict(lambda: {
        "n": 0, "score_sum": 0.0, "rounds": [], "queries": [], "tokens_chars": [],
        "zero_search": 0, "no_answer": 0, "empty_output": 0,
        "rounds_hist": Counter(), "queries_hist": Counter(), "pf_hist": Counter(),
        "parallel_samples": 0, "multi_query_rounds": 0,
    })
    with open(dump_path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            ds = r.get("data_source", "unknown")
            out = r.get("output") or ""
            a = agg[ds]
            a["n"] += 1
            try:
                a["score_sum"] += float(r.get("score", 0.0) or 0.0)
            except (TypeError, ValueError):
                pass
            spans = OBS_SPLIT.split(out)
            rounds = sum(1 for s in spans if any(t in s for t in TOOL_TAGS))
            queries = 0
            for m in WIKI_BLOCK.finditer(out):
                body = m.group(1).strip()
                if body:
                    queries += 1
            a["rounds"].append(rounds)
            a["queries"].append(queries)
            a["tokens_chars"].append(len(out))
            a["rounds_hist"][rounds] += 1
            a["queries_hist"][queries] += 1
            pf = (queries / rounds) if rounds else 0.0
            a["pf_hist"][round(int(pf)) if pf >= 0 else 0] += 1
            if rounds == 0:
                a["zero_search"] += 1
            if pf > 1:
                a["parallel_samples"] += 1
            if queries > rounds:
                a["multi_query_rounds"] += 1
            if not out.strip():
                a["empty_output"] += 1
            if not ANSWER_TAG.search(out):
                a["no_answer"] += 1
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--benchmarks-dir", default="/data01/wyy/Graph-Agent-Planning/Agent/data/mhqa_agent/test_benchmarks")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    metrics = parse_metrics_line(args.log)
    core, aux = metrics["core"], metrics["aux"]
    dump = analyze_dump(args.dump) if os.path.isfile(args.dump) else {}

    counts = {}
    try:
        import pandas as pd
        for f in glob.glob(os.path.join(args.benchmarks_dir, "*.cleaned.parquet")):
            df = pd.read_parquet(f, columns=["data_source"])
            for ds, n in df["data_source"].value_counts().items():
                counts[ds] = counts.get(ds, 0) + int(n)
    except Exception:
        pass

    datasets = [ds for ds in BENCH_ORDER if ds in dump or ds in core or ds in aux]
    datasets += [ds for ds in (set(dump) | set(core) | set(aux)) if ds not in datasets]

    accuracy = {"label": args.label, "per_benchmark": {}, "n_by_benchmark": dict(counts)}
    benchmark_results, traj, rounds_stats, parallel_stats = {}, {}, {}, {}
    ems, weights = [], []

    for ds in datasets:
        d = dump.get(ds, {})
        n = d.get("n", 0)
        rounds = d.get("rounds", [])
        queries = d.get("queries", [])
        pf = [(q / r) if r else 0.0 for q, r in zip(queries, rounds)]
        em_log = get_em(core, aux, ds)
        em_dump = (d.get("score_sum", 0.0) / n) if n else None
        em = em_log if em_log is not None else em_dump
        resp_len = aux.get(ds, {}).get("response_length/mean@1")
        num_turns = aux.get(ds, {}).get("num_turns/mean@1")
        accuracy["per_benchmark"][ds] = {
            "em": em, "em_from_log": em_log, "em_from_dump": em_dump,
            "n": n, "n_parquet": counts.get(ds),
        }
        if em is not None and n:
            ems.append(em)
            weights.append(n)
        benchmark_results[ds] = {
            "n_samples": n,
            "em": em,
            "response_tokens_mean": resp_len,
            "assistant_turns_mean_all_messages": num_turns,
            "retrieval_rounds_mean": (sum(rounds) / len(rounds)) if rounds else None,
            "search_queries_mean": (sum(queries) / len(queries)) if queries else None,
            "parallel_factor_mean": (sum(pf) / len(pf)) if pf else None,
            "parallel_sample_rate": (d.get("parallel_samples", 0) / n) if n else None,
            "multi_query_round_rate": (d.get("multi_query_rounds", 0) / n) if n else None,
            "zero_search_rate": (d.get("zero_search", 0) / n) if n else None,
            "no_answer_tag_rate": (d.get("no_answer", 0) / n) if n else None,
            "empty_output_rate": (d.get("empty_output", 0) / n) if n else None,
        }
        traj[ds] = {
            "n": n,
            "retrieval_rounds": {"mean": (sum(rounds) / len(rounds)) if rounds else None,
                                  "p50": pct(rounds, .5), "p90": pct(rounds, .9),
                                  "p99": pct(rounds, .99), "max": max(rounds) if rounds else None,
                                  "hist": {str(k): v for k, v in sorted(d.get("rounds_hist", {}).items())}},
            "search_queries": {"mean": (sum(queries) / len(queries)) if queries else None,
                                "p50": pct(queries, .5), "p90": pct(queries, .9),
                                "p99": pct(queries, .99), "max": max(queries) if queries else None,
                                "hist": {str(k): v for k, v in sorted(d.get("queries_hist", {}).items())}},
            "parallel_factor": {"mean": (sum(pf) / len(pf)) if pf else None,
                                 "p50": pct(pf, .5), "p90": pct(pf, .9), "p99": pct(pf, .99),
                                 "max": max(pf) if pf else None},
            "response_tokens_mean_from_log": resp_len,
            "assistant_turns_mean_from_log": num_turns,
        }
        rounds_stats[ds] = {
            "retrieval_rounds_mean": (sum(rounds) / len(rounds)) if rounds else None,
            "retrieval_rounds_hist": {str(k): v for k, v in sorted(d.get("rounds_hist", {}).items())},
            "zero_search_rate": (d.get("zero_search", 0) / n) if n else None,
            "n": n,
        }
        parallel_stats[ds] = {
            "parallel_factor_mean": (sum(pf) / len(pf)) if pf else None,
            "parallel_factor_p90": pct(pf, .9),
            "parallel_factor_max": max(pf) if pf else None,
            "parallel_sample_rate": (d.get("parallel_samples", 0) / n) if n else None,
            "multi_query_round_rate": (d.get("multi_query_rounds", 0) / n) if n else None,
            "parallel_factor_hist": {str(k): v for k, v in sorted(d.get("pf_hist", {}).items())},
            "n": n,
        }

    accuracy["macro_em_over_benchmarks"] = (sum(ems) / len(ems)) if ems else None
    accuracy["micro_em_over_samples"] = (
        sum(accuracy["per_benchmark"][ds]["em"] * accuracy["per_benchmark"][ds]["n"]
            for ds in datasets if accuracy["per_benchmark"][ds]["em"] is not None)
        / max(sum(accuracy["per_benchmark"][ds]["n"] for ds in datasets
                  if accuracy["per_benchmark"][ds]["em"] is not None), 1)
    )
    accuracy["total_samples"] = sum(accuracy["per_benchmark"][ds]["n"] for ds in datasets)

    overall = {
        "n_samples": accuracy["total_samples"],
        "macro_em": accuracy["macro_em_over_benchmarks"],
        "micro_em": accuracy["micro_em_over_samples"],
        "retrieval_rounds_mean": None,
        "search_queries_mean": None,
        "parallel_factor_mean": None,
        "parallel_sample_rate": None,
        "zero_search_rate": None,
        "no_answer_tag_rate": None,
    }
    tot = sum(dump.get(ds, {}).get("n", 0) for ds in datasets)
    if tot:
        overall["retrieval_rounds_mean"] = sum(sum(dump[ds]["rounds"]) for ds in datasets if ds in dump) / tot
        overall["search_queries_mean"] = sum(sum(dump[ds]["queries"]) for ds in datasets if ds in dump) / tot
        overall["parallel_sample_rate"] = sum(dump[ds].get("parallel_samples", 0) for ds in datasets if ds in dump) / tot
        overall["zero_search_rate"] = sum(dump[ds].get("zero_search", 0) for ds in datasets if ds in dump) / tot
        overall["no_answer_tag_rate"] = sum(dump[ds].get("no_answer", 0) for ds in datasets if ds in dump) / tot
        q = sum(sum(dump[ds]["queries"]) for ds in datasets if ds in dump)
        r = sum(sum(dump[ds]["rounds"]) for ds in datasets if ds in dump)
        overall["parallel_factor_mean"] = q / r if r else None

    results = {
        "label": args.label,
        "eval_log": args.log,
        "dump": args.dump,
        "benchmarks": {ds: {"em": benchmark_results[ds]["em"],
                            "n": benchmark_results[ds]["n_samples"]} for ds in datasets},
    }

    def dump_json(name, obj):
        with open(os.path.join(args.out, name), "w") as f:
            json.dump(obj, f, indent=2, default=str)

    dump_json("accuracy.json", accuracy)
    dump_json("benchmark_results.json", benchmark_results)
    dump_json("trajectory_metrics.json", {"per_benchmark": traj, "overall": overall})
    dump_json("search_round_statistics.json", {"per_benchmark": rounds_stats, "overall": overall})
    dump_json("parallel_statistics.json", {"per_benchmark": parallel_stats, "overall": overall})
    dump_json("eval_summary.json", {"label": args.label, "accuracy": accuracy,
                                    "benchmark_results": benchmark_results, "overall": overall})
    print(json.dumps({"label": args.label, "macro_em": accuracy["macro_em_over_benchmarks"],
                      "micro_em": accuracy["micro_em_over_samples"],
                      "total_samples": accuracy["total_samples"], "overall": overall},
                     indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""E1-E continuous safety monitor (E1-E_task.md section 7).

Watches, for the whole step70 -> step120 run:

  1. accuracy明显下降        -- training-side EM (all rollouts) and the periodic
                               `val-core/nq/reward/mean@1` from the training log.
  2. search rounds 是否下降   -- mean `logical_search_batches`.
  3. queries 是否异常增加     -- mean `search_queries`.
  4. zero-search correct      -- rollouts with cost == 0 AND em == 1.
  5. query packing 异常增加   -- active (>=2 correct, cost-varying) groups in which the
                               efficient correct rollouts do NOT use fewer queries.

RECORD-ONLY: it never kills, throttles or patches the training process (the task asks
for "记录并尝试最小修复", i.e. detect + report so a human/agent can apply a *minimal*
fix; silently mutating a 50-step run would be a larger variable).  Alerts are appended
to <run_dir>/logs/safety_alerts.log and E1-E/logs/safety_alerts.log.

Incremental: reads only the bytes appended to the diagnostics jsonl since the last
cycle (state in <run_dir>/logs/safety_monitor_state.json), so it stays cheap over 50
steps / ~700k rollouts.
"""
import argparse
import glob
import json
import os
import re
import statistics
import time
from collections import defaultdict

STEP_RE = re.compile(r"step:(\d+) - ")
KV_RE = re.compile(r"([A-Za-z0-9_\-/\.@]+):(-?[0-9\.eE\+]+)")

METRIC_KEYS = [
    "critic/score/mean",
    "critic/score/max",
    "critic/rewards/mean",
    "actor/entropy",
    "actor/pg_loss",
    "actor/pg_clipfrac",
    "actor/grad_norm",
    "response_length/mean",
    "tools/avg_calls_per_traj",
    "tools/avg_turns_per_traj",
]


def iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


class Agg:
    """Per-global-step incremental aggregates."""

    def __init__(self):
        self.steps = {}

    def step(self, s, gstep):
        if gstep is None:
            gstep = -1
        d = self.steps.setdefault(str(gstep), {
            "global_step": gstep,
            "n_rollouts": 0, "n_groups": 0,
            "sum_em": 0.0, "sum_reward": 0.0, "sum_eff": 0.0,
            "n_eff_nonzero": 0,
            "cost_sum": 0.0, "cost_n": 0,
            "query_sum": 0.0, "query_n": 0,
            "turns_sum": 0.0, "turns_n": 0,
            "tokens_sum": 0.0, "tokens_n": 0,
            "zero_search": 0, "zero_search_correct": 0,
            "metadata_invalid": 0,
            "parallel_sum": 0.0, "parallel_n": 0,
            "uids": {},
        })
        return d

    def add_rollout(self, s, r):
        d = self.step(s, r.get("global_step"))
        d["n_rollouts"] += 1
        em = r.get("em")
        em = 1.0 if (em == 1 or em == 1.0) else 0.0
        d["sum_em"] += em
        rw = r.get("reward")
        if rw is not None:
            d["sum_reward"] += float(rw)
        eff = r.get("efficiency")
        if eff is not None:
            d["sum_eff"] += float(eff)
            if float(eff) > 0:
                d["n_eff_nonzero"] += 1
        cost = r.get("logical_search_batches")
        if cost is not None:
            d["cost_sum"] += float(cost)
            d["cost_n"] += 1
        q = r.get("search_queries")
        if q is not None:
            d["query_sum"] += float(q)
            d["query_n"] += 1
        if cost and q:
            d["parallel_sum"] += float(q) / float(cost)
            d["parallel_n"] += 1
        t = r.get("conversation_turns")
        if t is not None:
            d["turns_sum"] += float(t)
            d["turns_n"] += 1
        tk = r.get("response_token_length")
        if tk is not None:
            d["tokens_sum"] += float(tk)
            d["tokens_n"] += 1
        if r.get("zero_search"):
            d["zero_search"] += 1
            if em == 1.0:
                d["zero_search_correct"] += 1
        if not r.get("metadata_valid", True):
            d["metadata_invalid"] += 1
        uid = str(r.get("uid"))
        g = d["uids"].setdefault(uid, {"n": 0, "correct": 0, "costs": [], "queries": []})
        g["n"] += 1
        if em == 1.0:
            g["correct"] += 1
            if cost is not None:
                g["costs"].append(cost)
                g["queries"].append(q if q is not None else 0.0)

    def add_call(self, s, c):
        d = self.step(s, c.get("global_step"))
        d["n_groups"] += int(c.get("n_groups") or 0)
        d["n_baseline_keep"] = d.get("n_baseline_keep", 0) + int(c.get("n_baseline_keep") or 0)
        d["n_shaped_keep"] = d.get("n_shaped_keep", 0) + int(c.get("n_shaped_keep") or 0)
        d["n_newly_added"] = d.get("n_newly_added", 0) + int(c.get("n_newly_added") or 0)
        d["n_efficiency_active_groups"] = d.get("n_efficiency_active_groups", 0) + int(
            c.get("n_efficiency_active_groups") or 0)

    def summary(self):
        out = []
        for k in sorted(self.steps, key=lambda x: int(x)):
            d = self.steps[k]
            n = d["n_rollouts"]
            active = 0
            suspects = 0
            for uid, g in d["uids"].items():
                if g["correct"] < 2 or len(g["costs"]) < 2:
                    continue
                cmin, cmax = min(g["costs"]), max(g["costs"])
                if cmax <= cmin:
                    continue
                active += 1
                eq = statistics.fmean([q for c, q in zip(g["costs"], g["queries"]) if c == cmin]) if any(
                    c == cmin for c in g["costs"]) else 0.0
                iq = statistics.fmean([q for c, q in zip(g["costs"], g["queries"]) if c == cmax]) if any(
                    c == cmax for c in g["costs"]) else 0.0
                if eq >= iq:
                    suspects += 1
            row = {
                "global_step": d["global_step"],
                "n_rollouts": n,
                "n_groups": d["n_groups"],
                "em_mean": (d["sum_em"] / n) if n else None,
                "reward_mean": (d["sum_reward"] / n) if n else None,
                "efficiency_mean": (d["sum_eff"] / n) if n else None,
                "efficiency_nonzero_rate": (d["n_eff_nonzero"] / n) if n else None,
                "efficiency_active_groups": d.get("n_efficiency_active_groups", 0),
                "efficiency_active_groups_recomputed": active,
                "retained_groups_shaped": d.get("n_shaped_keep", 0),
                "retained_groups_baseline_em": d.get("n_baseline_keep", 0),
                "newly_added_by_shaping": d.get("n_newly_added", 0),
                "search_batches_mean": (d["cost_sum"] / d["cost_n"]) if d["cost_n"] else None,
                "queries_mean": (d["query_sum"] / d["query_n"]) if d["query_n"] else None,
                "parallel_factor_mean": (d["parallel_sum"] / d["parallel_n"]) if d["parallel_n"] else None,
                "conversation_turns_mean": (d["turns_sum"] / d["turns_n"]) if d["turns_n"] else None,
                "response_tokens_mean": (d["tokens_sum"] / d["tokens_n"]) if d["tokens_n"] else None,
                "zero_search_rate": (d["zero_search"] / n) if n else None,
                "zero_search_correct_count": d["zero_search_correct"],
                "metadata_invalid": d["metadata_invalid"],
                "packing_active_groups": active,
                "packing_suspect_groups": suspects,
                "packing_suspect_ratio": (suspects / active) if active else None,
                "uids": d["uids"],
            }
            out.append(row)
        return out


def load_offsets(path):
    if os.path.exists(path):
        try:
            with open(path) as fh:
                return json.load(fh)
        except ValueError:
            return {}
    return {}


def read_new(path, offsets):
    off = offsets.get(path, 0)
    rows = []
    if not os.path.exists(path):
        return rows, off
    size = os.path.getsize(path)
    if off > size:  # file rotated/truncated
        off = 0
    with open(path, errors="ignore") as fh:
        fh.seek(off)
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
        offsets[path] = fh.tell()
    return rows, offsets[path]


def parse_log(log_path):
    """Return (latest_step, latest metrics dict, {step: val_nq_em})."""
    latest_step = None
    latest = {}
    vals = {}
    if not os.path.exists(log_path):
        return latest_step, latest, vals
    with open(log_path, errors="ignore") as fh:
        for line in fh:
            m = STEP_RE.search(line)
            if not m:
                continue
            step = int(m.group(1))
            if "val-core/nq/reward/mean@1" in line:
                for k, v in KV_RE.findall(line):
                    if k == "val-core/nq/reward/mean@1":
                        try:
                            vals[step] = float(v)
                        except ValueError:
                            pass
            if "critic/score/mean" not in line:
                continue
            kv = dict(KV_RE.findall(line))
            cur = {"step": step}
            for key in METRIC_KEYS:
                if key in kv:
                    try:
                        cur[key] = float(kv[key])
                    except ValueError:
                        pass
            latest_step = step
            latest = cur
    return latest_step, latest, vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--out", default=None,
                    help="extra aggregation output (default E1-E/diagnostics/safety_monitor.jsonl)")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    run_dir = args.run_dir
    out_path = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "diagnostics", "safety_monitor.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    snap_path = os.path.join(run_dir, "logs", "safety_monitor.jsonl")
    alert_run = os.path.join(run_dir, "logs", "safety_alerts.log")
    alert_glob = os.path.join(os.path.dirname(out_path), "safety_alerts.log")
    state_path = os.path.join(run_dir, "logs", "safety_monitor_state.json")
    offsets = load_offsets(state_path)
    agg = Agg()
    # restore previously aggregated state (so the snapshot stays complete)
    state_agg_path = os.path.join(run_dir, "logs", "safety_monitor_agg.json")
    if os.path.exists(state_agg_path):
        try:
            with open(state_agg_path) as fh:
                agg.steps = json.load(fh)
        except ValueError:
            pass

    baseline = {"em_mean": None, "search_batches_mean": None, "queries_mean": None,
                "parallel_factor_mean": None, "val_nq_em": None}

    cycle = 0
    while True:
        cycle += 1
        diag = os.path.join(run_dir, "diagnostics")
        rollouts = sorted(glob.glob(os.path.join(diag, "rollouts_pid*.jsonl")))
        calls = sorted(glob.glob(os.path.join(diag, "calls_pid*.jsonl")))
        for p in rollouts:
            rows, _ = read_new(p, offsets)
            for r in rows:
                agg.add_rollout(p, r)
        for p in calls:
            rows, _ = read_new(p, offsets)
            for c in rows:
                agg.add_call(p, c)

        latest_step, latest_metrics, val_nq = parse_log(args.log)
        rows = agg.summary()

        if baseline["em_mean"] is None and rows:
            b = rows[0]
            baseline.update({
                "em_mean": b["em_mean"],
                "search_batches_mean": b["search_batches_mean"],
                "queries_mean": b["queries_mean"],
                "parallel_factor_mean": b["parallel_factor_mean"],
            })
        if baseline["val_nq_em"] is None and val_nq:
            first_step = min(val_nq)
            baseline["val_nq_em"] = val_nq[first_step]
        if val_nq:
            baseline["val_nq_em_baseline_step"] = min(val_nq)

        alerts = []
        if rows:
            last = rows[-1]
            if last["zero_search_correct_count"] and last["zero_search_correct_count"] > 0:
                alerts.append(f"ZERO_SEARCH_CORRECT={last['zero_search_correct_count']} at step "
                              f"{last['global_step']} (unexpected: correct answer with no retrieval)")
            if last["packing_suspect_ratio"] is not None and last["packing_suspect_ratio"] > 0.10:
                alerts.append(f"PACKING_SUSPECT_RATIO={last['packing_suspect_ratio']:.3f} > 0.10 at step "
                              f"{last['global_step']}")
            if (baseline["em_mean"] is not None and last["em_mean"] is not None
                    and last["em_mean"] < baseline["em_mean"] - 0.01):
                alerts.append(f"TRAIN_EM_DROP step {last['global_step']}: {last['em_mean']:.4f} vs "
                              f"{baseline['em_mean']:.4f} (< -1pp)")
            if (baseline["queries_mean"] is not None and last["queries_mean"] is not None
                    and last["queries_mean"] > baseline["queries_mean"] * 1.10):
                alerts.append(f"QUERIES_INCREASE step {last['global_step']}: {last['queries_mean']:.4f} vs "
                              f"{baseline['queries_mean']:.4f} (> +10%)")
            if (baseline["parallel_factor_mean"] is not None and last["parallel_factor_mean"] is not None
                    and last["parallel_factor_mean"] > baseline["parallel_factor_mean"] * 1.05):
                alerts.append(f"PARALLEL_FACTOR_INCREASE step {last['global_step']}: "
                              f"{last['parallel_factor_mean']:.4f} vs {baseline['parallel_factor_mean']:.4f} (> +5%)")
            if last["metadata_invalid"]:
                alerts.append(f"METADATA_INVALID={last['metadata_invalid']} at step {last['global_step']}")

        snap = {
            "timestamp": iso(),
            "cycle": cycle,
            "latest_train_step": latest_step,
            "latest_train_metrics": latest_metrics,
            "val_nq_em_by_step": val_nq,
            "baseline": baseline,
            "per_step": [{k: v for k, v in r.items() if k != "uids"} for r in rows],
            "alerts": alerts,
        }
        with open(snap_path, "a") as fh:
            fh.write(json.dumps(snap, default=str) + "\n")
        with open(out_path, "a") as fh:
            fh.write(json.dumps(snap, default=str) + "\n")
        # persist incremental aggregation state (uid maps included: needed later to
        # re-derive packing statistics without re-reading the whole rollout file)
        with open(state_agg_path, "w") as fh:
            json.dump(agg.steps, fh)
        with open(state_path, "w") as fh:
            json.dump(offsets, fh)
        if alerts:
            with open(alert_run, "a") as fh:
                for a in alerts:
                    fh.write(f"{iso()} {a}\n")
            with open(alert_glob, "a") as fh:
                for a in alerts:
                    fh.write(f"{iso()} {a}\n")

        print(f"[monitor] cycle={cycle} step={latest_step} steps_agg={len(rows)} alerts={len(alerts)}", flush=True)
        if args.once:
            print(json.dumps(snap, indent=2, default=str)[:4000])
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""E1-E report figures (matplotlib, Agg backend, no seaborn).

Reads the artifacts produced by the run and writes PNGs into `E1-E/figures/`:
  fig1_four_way_pareto.png       micro EM vs mean search rounds: GAP120 / E1-B120 / E1-D120 /
                                 E1-E120 (+ step70 points) with the pre-registered target box
  fig2_quota_runtime.png         per optimizer step: quota filled, generation batches,
                                 efficiency groups seen, eff_shortfall
  fig3_reward_separation.png     reward distribution by group role (mixed vs efficiency)
  fig4_efficiency_ranking.png    within efficiency groups: cost vs assigned efficiency/reward
  fig5_search_behavior.png       per-step mean rounds / queries / zero-search rate
  fig6_prediction_vs_observation.png  E1-E0 frozen prediction vs E1-E online observation

Every panel is skipped with a printed notice when its inputs are missing; the script never
fails the pipeline because a figure cannot be drawn.
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = "/data01/wyy/Graph-Agent-Planning"
E1E = os.path.join(REPO, "E1-E")
FIG = os.path.join(E1E, "figures")


def jload(p):
    try:
        return json.load(open(p))
    except Exception:  # noqa: BLE001
        return None


def newest(pat):
    c = sorted(glob.glob(pat))
    return c[-1] if c else None


def iter_jsonl(paths):
    for p in paths:
        for line in open(p, errors="ignore"):
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except ValueError:
                    continue


def load_runtime():
    steps, groups, rolls = [], [], []
    for ph in ("D1", "D2"):
        f = os.path.join(E1E, "runs", f"latest_{ph}.txt")
        if not os.path.exists(f):
            continue
        diag = os.path.join(open(f).read().strip(), "diagnostics")
        steps += list(iter_jsonl(sorted(glob.glob(os.path.join(diag, "quota_steps_pid*.jsonl")))))
        groups += list(iter_jsonl(sorted(glob.glob(os.path.join(diag, "quota_groups_pid*.jsonl")))))
        rolls += list(iter_jsonl(sorted(glob.glob(os.path.join(diag, "rollouts_pid*.jsonl")))))
    return steps, groups, rolls


def eval_overall(tag):
    if tag == "e1e70":
        p = newest(os.path.join(E1E, "eval_results", "step70_*", "eval_summary.json"))
    elif tag == "e1e120":
        p = newest(os.path.join(E1E, "eval_results", "step120_*", "eval_summary.json"))
    elif tag == "e1d70":
        p = newest(os.path.join(REPO, "E1-D/eval_results/step70_*", "eval_summary.json"))
    elif tag == "e1d120":
        p = newest(os.path.join(REPO, "E1-D/eval_results/step120_*", "eval_summary.json"))
    else:
        p = os.path.join(REPO, f"E1-B/evaluations/{tag}/eval_summary.json")
    d = jload(p) if p else None
    if not d:
        return None
    return {"micro_em": d["accuracy"].get("micro_em_over_samples"),
            "rounds": d.get("overall", {}).get("retrieval_rounds_mean"),
            "queries": d.get("overall", {}).get("search_queries_mean")}


def fig1():
    pts = {"GAP120": eval_overall("gap_step120_reference"),
           "E1-B120": eval_overall("e1b120") or eval_overall("step120_20260911_235134"),
           "E1-D120": eval_overall("e1d120"),
           "E1-E120": eval_overall("e1e120"),
           "E1-D70": eval_overall("e1d70"),
           "E1-E70": eval_overall("e1e70")}
    pts = {k: v for k, v in pts.items() if v}
    if len(pts) < 2:
        print("[fig1] skipped: not enough evaluations")
        return
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    colors = {"GAP120": "#444444", "E1-B120": "#1f77b4", "E1-D120": "#ff7f0e", "E1-E120": "#d62728"}
    for k, v in pts.items():
        c = colors.get(k, "#999999")
        ax.scatter(v["rounds"], v["micro_em"], s=110 if k.startswith("E1-E120") else 70,
                   color=c, marker="o" if "120" in k else "s", zorder=3)
        ax.annotate(k, (v["rounds"], v["micro_em"]), textcoords="offset points",
                    xytext=(6, 5), fontsize=9, color=c)
    g = pts.get("GAP120")
    if g:
        # pre-registered target: dEM >= -0.5 pp and dRounds <= -10% vs GAP120
        em_line = g["micro_em"] - 0.005
        r_line = g["rounds"] * 0.90
        ax.axhline(em_line, color="#2ca02c", ls="--", lw=1.2,
                   label=f"micro EM >= {em_line:.4f} (GAP120 - 0.5 pp)")
        ax.axvline(r_line, color="#2ca02c", ls=":", lw=1.2,
                   label=f"rounds <= {r_line:.3f} (GAP120 - 10%)")
        ax.axhspan(em_line, ax.get_ylim()[1], xmin=0, xmax=1, color="#2ca02c", alpha=0.05)
    ax.set_xlabel("mean search rounds (7 benchmarks, greedy)")
    ax.set_ylabel("micro EM")
    ax.set_title("E1-E vs baselines: correctness / search-cost frontier")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_four_way_pareto.png"), dpi=150)
    plt.close(fig)
    print("[fig1] wrote fig1_four_way_pareto.png")


def fig2(steps):
    if not steps:
        print("[fig2] skipped: no runtime step records")
        return
    steps = sorted(steps, key=lambda r: int(r["global_step"]))
    x = [int(r["global_step"]) for r in steps]
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True)
    axes[0].step(x, [r.get("quota_actual") or 0 for r in steps], where="mid", color="#d62728")
    axes[0].axhline(2, color="k", ls=":", lw=1)
    axes[0].set_ylabel("inserted efficiency groups\n(quota actual)")
    axes[0].set_ylim(-0.2, 2.6)
    axes[1].step(x, [r.get("generation_batches_consumed") or 0 for r in steps], where="mid",
                 color="#1f77b4", label="generation batches consumed")
    axes[1].step(x, [r.get("efficiency_seen_cum") or 0 for r in steps], where="mid",
                 color="#2ca02c", label="efficiency groups seen (cum.)")
    axes[1].step(x, [r.get("eff_shortfall_total") or 0 for r in steps], where="mid",
                 color="#ff7f0e", label="eff_shortfall (cum.)")
    axes[1].set_xlabel("optimizer step")
    axes[1].set_ylabel("count")
    axes[1].legend(fontsize=8)
    for a in axes:
        a.grid(alpha=0.3)
        if x[0] <= 70 <= x[-1]:
            a.axvline(70, color="#888888", ls="--", lw=1)
    axes[0].set_title("E1-E runtime quota behaviour (D1 steps 61-70, D2 steps 71-120)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_quota_runtime.png"), dpi=150)
    plt.close(fig)
    print("[fig2] wrote fig2_quota_runtime.png")


def fig3(rolls):
    if not rolls:
        print("[fig3] skipped: no rollout records")
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for ax, role, title in ((axes[0], "mixed", "mixed groups (1<=k<=7): R_i = EM"),
                            (axes[1], "efficiency", "quota-selected k=8 groups: R_i = 1 + 0.05 E_i")):
        vals = [float(r["reward"]) for r in rolls
                if r.get("group_type") == role and r.get("reward") is not None]
        if not vals:
            ax.set_visible(False)
            continue
        ax.hist(vals, bins=40 if role == "efficiency" else 3, color="#1f77b4" if role == "mixed"
                else "#d62728", alpha=0.85)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("per-rollout reward")
        ax.set_ylabel("rollouts")
        ax.grid(alpha=0.3)
    fig.suptitle("Reward separation by group role (E1-E runtime diagnostics)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_reward_separation.png"), dpi=150)
    plt.close(fig)
    print("[fig3] wrote fig3_reward_separation.png")


def fig4(rolls):
    eff = [r for r in rolls if r.get("group_type") == "efficiency"]
    if not eff:
        print("[fig4] skipped: no efficiency rollouts")
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    cost = [r.get("logical_search_batches") for r in eff]
    rew = [float(r["reward"]) for r in eff]
    axes[0].scatter([c if c is not None else -0.1 for c in cost], rew, s=4, alpha=0.25,
                    color="#d62728")
    axes[0].set_xlabel("logical_search_batches (correct rollouts of efficiency groups)")
    axes[0].set_ylabel("reward  =  1 + 0.05 E_i")
    axes[0].set_title("cost -> reward ranking", fontsize=9)
    effi = [r.get("efficiency") for r in eff]
    axes[1].hist([float(e) for e in effi if e is not None], bins=40, color="#2ca02c", alpha=0.85)
    axes[1].set_xlabel("E_i = (Cmax - C_i)/(Cmax - Cmin)")
    axes[1].set_ylabel("rollouts")
    axes[1].set_title("efficiency distribution inside gated groups", fontsize=9)
    for a in axes:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_efficiency_ranking.png"), dpi=150)
    plt.close(fig)
    print("[fig4] wrote fig4_efficiency_ranking.png")


def fig5(rolls):
    if not rolls:
        print("[fig5] skipped: no rollouts")
        return
    by = {}
    for r in rolls:
        s = r.get("global_step")
        if s is None:
            continue
        d = by.setdefault(int(s), {"rounds": [], "queries": [], "zero": 0, "n": 0, "tok": []})
        d["n"] += 1
        if r.get("logical_search_batches") is not None:
            d["rounds"].append(float(r["logical_search_batches"]))
        if r.get("search_queries") is not None:
            d["queries"].append(float(r["search_queries"]))
        if r.get("response_token_length") is not None:
            d["tok"].append(float(r["response_token_length"]))
        d["zero"] += int(bool(r.get("zero_search")))
    if not by:
        print("[fig5] skipped: no per-step rollouts")
        return
    x = sorted(by)
    fig, axes = plt.subplots(3, 1, figsize=(9, 7.2), sharex=True)
    axes[0].plot(x, [sum(by[s]["rounds"]) / max(len(by[s]["rounds"]), 1) for s in x], color="#1f77b4")
    axes[0].set_ylabel("mean rounds")
    axes[1].plot(x, [sum(by[s]["queries"]) / max(len(by[s]["queries"]), 1) for s in x], color="#2ca02c")
    axes[1].set_ylabel("mean queries")
    axes[2].plot(x, [by[s]["zero"] / max(by[s]["n"], 1) for s in x], color="#d62728")
    axes[2].set_ylabel("zero-search rate")
    axes[2].set_xlabel("optimizer step")
    for a in axes:
        a.grid(alpha=0.3)
        if x[0] <= 70 <= x[-1]:
            a.axvline(70, color="#888888", ls="--", lw=1)
    axes[0].set_title("E1-E training search behaviour (materialised optimizer batches)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_search_behavior.png"), dpi=150)
    plt.close(fig)
    print("[fig5] wrote fig5_search_behavior.png")


def fig6():
    p = os.path.join(E1E, "analysis", "prediction_vs_observation.json")
    d = jload(p)
    if not d or not d.get("available"):
        print("[fig6] skipped: prediction_vs_observation.json not available")
        return
    obs = d["comparison"]["observed"]
    pred = d["comparison"]["predicted_E1E0"]
    keys = [k for k in ("full_quota_rate", "actual_replacement_rate",
                        "mean_k_over_8_of_optimizer_batch", "mean_search_rounds",
                        "mean_search_queries", "mean_response_tokens")
            if obs.get(k) is not None and pred.get(k) is not None]
    if not keys:
        print("[fig6] skipped: no comparable quantities")
        return
    fig, ax = plt.subplots(figsize=(8, 4.6))
    idx = range(len(keys))
    ax.bar([i - 0.2 for i in idx], [pred[k] for k in keys], width=0.4, label="E1-E0 prediction (q=2)",
           color="#1f77b4")
    ax.bar([i + 0.2 for i in idx], [obs[k] for k in keys], width=0.4, label="E1-E online observation",
           color="#d62728")
    ax.set_xticks(list(idx))
    ax.set_xticklabels([k.replace("mean_", "").replace("_of_optimizer_batch", "") for k in keys],
                       rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("value")
    ax.set_title("E1-E0 frozen prediction vs E1-E online observation")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig6_prediction_vs_observation.png"), dpi=150)
    plt.close(fig)
    print("[fig6] wrote fig6_prediction_vs_observation.png")


def main():
    os.makedirs(FIG, exist_ok=True)
    steps, groups, rolls = load_runtime()
    print(f"[figures] runtime records: steps={len(steps)} groups={len(groups)} rollouts={len(rolls)}")
    fig1()
    fig2(steps)
    fig3(rolls)
    fig4(rolls)
    fig5(rolls)
    fig6()
    print("[figures] done ->", FIG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

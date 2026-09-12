#!/usr/bin/env python3
"""E1-C figures (optional but requested by E1-C_task.md §11). Reads only the CSVs in
E1-C/results/ and writes PNGs into E1-C/figures/.

Uses csv+numpy+matplotlib only (the `parallel-agent` env has no matplotlib; run with e.g.
/home/nf5468m6/miniconda3/envs/huatuo/bin/python3).
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

E1C = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(E1C, "results")
FIG = os.path.join(E1C, "figures")
os.makedirs(FIG, exist_ok=True)


def read_csv(name):
    with open(os.path.join(RES, name)) as fh:
        return list(csv.DictReader(fh))


def fnum(rows, col, cast=float):
    return [cast(float(r[col])) if r[col] not in ("", "None") else None for r in rows]


def fig_composition():
    rows = read_csv("group_type_summary.csv")
    pools = ["E1_0", "E1_A", "E1_B"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, p in zip(axes, pools):
        r = [x for x in rows if x["pool"] == p and x["set"] == "candidates"][0]
        vals = [int(float(r["all_wrong"])), int(float(r["mixed"])), int(float(r["all_correct"]))]
        ax.bar(["all-wrong\nk=0", "mixed\n1<=k<=7", "all-correct\nk=8"], vals,
               color=["#888", "#2b7bba", "#d95f02"])
        ax.set_title(f"{p} candidate pool\n(n={int(float(r['n_groups'])):,} groups)")
        ax.set_ylabel("groups")
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Candidate group composition — only the mixed band passes the GAP filter")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "group_composition.png"), dpi=130)
    plt.close(fig)


def fig_sets():
    rows = read_csv("same_pool_set_summary.csv")
    pools = [r["pool"] for r in rows]
    gap = fnum(rows, "S_GAP", int)
    e1o = fnum(rows, "E1_only", int)
    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(pools))
    ax.barh(y, gap, color="#2b7bba", label="S_GAP (kept by correctness-only reward)")
    ax.barh(y, e1o, left=gap, color="#d95f02", label="E1-only (added by shaped reward)")
    for i, (g, e) in enumerate(zip(gap, e1o)):
        ax.text(g + e, i, f"  {g:,} + {e:,} = {g+e:,}", va="center", fontsize=9)
    ax.set_yticks(y, pools)
    ax.set_xlabel("retained groups on the same candidate pool")
    ax.set_title("Same-pool retained sets: S_GAP is a strict subset of S_E1 (GAP-only = 0)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "set_relationship.png"), dpi=130)
    plt.close(fig)


def fig_difficulty():
    rows = read_csv("difficulty_by_pool.csv")
    r = [x for x in rows if x["pool"] == "E1_B"][0]
    d = read_csv("difficulty_comparison.csv")
    p_success = [x for x in d if x["metric"] == "group_success_rate"][0]["mannwhitney_p"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    labels = ["group success rate\n(k/8)", "all-correct rate", "mixed rate"]
    a = [float(r["E1_added_success_rate_mean"]), float(r["E1_added_all_correct_rate"]), 0.0]
    b = [float(r["GAP_displaced_success_rate_mean"]), 0.0, float(r["GAP_displaced_mixed_rate"])]
    x = np.arange(3)
    ax.bar(x - 0.2, a, 0.4, label="E1-added (enters E1 batch)", color="#d95f02")
    ax.bar(x + 0.2, b, 0.4, label="GAP-displaced (pushed out)", color="#2b7bba")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Optimizer-batch cohorts (E1-B pool)\nsuccess-rate difference p = {float(p_success):.2e}")
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.bar(["E1-added", "GAP-displaced"],
           [float(r["E1_added_cost_mean"]), float(r["GAP_displaced_cost_mean"])],
           color=["#d95f02", "#2b7bba"])
    ax.set_ylabel("mean logical_search_batches")
    ax.set_title("Search depth of the two cohorts")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "difficulty_cohorts.png"), dpi=130)
    plt.close(fig)


def fig_lambda():
    mem = read_csv("lambda_membership.csv")
    adv = read_csv("lambda_advantage_fixed_batch.csv")
    lam = [float(r["lambda"]) for r in mem]
    n = [int(float(r["n_retained"])) for r in mem]
    gap = []
    for r in adv:
        v = r["efficient_minus_inefficient_gap"]
        gap.append(float(v) if v not in ("", "None") else None)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.step(lam, n, where="post", marker="o", color="#2b7bba")
    ax.set_xlabel("λ")
    ax.set_ylabel("|S_λ| (retained groups)")
    ax.set_title("Membership is a step function of λ:\nλ=0 → 1412, every λ>0 → 1685 (identical set)")
    ax = axes[1]
    ax.plot(lam, gap, marker="o", color="#d95f02")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("λ")
    ax.set_ylabel("advantage(efficient correct) − advantage(inefficient correct)")
    ax.set_title("Efficiency gradient on a FIXED optimizer batch:\nnegative at λ=0, positive and ≈λ-invariant for λ>0")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "lambda_effect.png"), dpi=130)
    plt.close(fig)


def fig_actual():
    rows = read_csv("actual_run_comparison.csv")
    rows = [r for r in rows if r["GAP_gen_batches"] not in ("", "None") and r["E1B_gen_batches"] not in ("", "None")]
    steps = [int(r["global_step"]) for r in rows]
    g = [float(r["GAP_gen_batches"]) for r in rows]
    e = [float(r["E1B_gen_batches"]) for r in rows]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(steps, g, marker=".", label="original GAP run (observed)", color="#2b7bba")
    ax.plot(steps, e, marker=".", label="E1-B run (observed)", color="#d95f02")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("generation batches used")
    ax.set_title("OBSERVATIONAL (policy-confounded): actual generation batches per optimizer step, steps 70-120")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "actual_gen_batches.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    fig_composition()
    fig_sets()
    fig_difficulty()
    fig_lambda()
    fig_actual()
    for f in sorted(os.listdir(FIG)):
        print("wrote", os.path.join(FIG, f))

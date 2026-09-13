#!/usr/bin/env python3
"""E1-E0 figures (+ the unrestricted-admission reference used in FINAL_REPORT §16).

Reads only the CSVs in E1-E0/results/ and writes PNGs into E1-E0/figures/ plus
`results/unrestricted_reference.csv`.

matplotlib is absent from the parallel-agent env; run with
    /home/nf5468m6/miniconda3/envs/huatuo/bin/python3 E1-E0/code/e1e_figures.py
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

E1E = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(E1E, "results")
FIG = os.path.join(E1E, "figures")
os.makedirs(FIG, exist_ok=True)


def rd(name):
    with open(os.path.join(RES, name)) as fh:
        return list(csv.DictReader(fh))


def f(v, d=None):
    if v in (None, "", "None"):
        return d
    return float(v)


def main():
    avail = rd("quota_availability.csv")
    dens = rd("quota_supervision_density.csv")
    proxy = rd("batch_level_proxy_by_quota.csv")
    src = rd("source_shift_by_quota.csv")
    disp = rd("mixed_displacement_by_quota.csv")
    q0 = rd("q0_baseline_replay.csv")

    q = [int(r["q"]) for r in avail]
    full = [f(r["full_quota_rate"]) for r in avail]
    repl = [f(r["actual_replacement_rate"]) for r in avail]
    frac = [f(r["actual_efficiency_group_fraction"]) for r in dens]
    theo = [f(r["theoretical_upper_bound_fraction"]) for r in dens]

    # ---------- unrestricted reference (E1-E0_task.md §16) ---------------------
    # "unrestricted" = admit every cached efficiency group into the same batch
    # (E1-B-style admission), still bounded by the 32-group batch size.
    tot_cache = sum(int(r["efficiency_cache_count"]) for r in q0)
    tot_mixed = sum(int(r["retained_mixed_groups"]) for r in q0)
    unbounded_m = sum(min(32, int(r["efficiency_cache_count"])) for r in q0)
    unrestricted = {
        "definition": "m = min(32, cached efficiency groups of that step) - i.e. admit every "
                      "cached efficiency group, the E1-B-style unrestricted admission",
        "efficiency_groups_cached_total": tot_cache,
        "mixed_groups_available_total": tot_mixed,
        "efficiency_groups_inserted_total": unbounded_m,
        "optimizer_groups_total": 32 * len(q0),
        "efficiency_group_fraction": unbounded_m / (32 * len(q0)),
        "replacement_fraction": unbounded_m / (32 * len(q0)),
        "mean_per_step": unbounded_m / len(q0),
        "cache_mean_per_step": tot_cache / len(q0),
        "note": "E1-B's own run (different policy) had 291 efficiency-active groups retained over "
                "50 steps = 5.82/step = 18.2 % of a 32-group batch (E1-C on the E1-B pool). "
                "The two pools are different policies, so the comparison is indicative only.",
    }
    with open(os.path.join(RES, "unrestricted_reference.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["quantity", "value"])
        for k, v in unrestricted.items():
            w.writerow([k, v])
    print(json.dumps(unrestricted, indent=1))

    # ---------- fig 1: efficiency-group fraction vs quota ----------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(q, [x * 100 for x in theo], "o--", color="#888", label="theoretical upper bound q/B")
    ax.plot(q, [x * 100 for x in frac], "o-", color="#d95f02", label="actual (replay)")
    ax.axhline(unrestricted["efficiency_group_fraction"] * 100, color="#2b7bba", ls=":",
               label=f"unrestricted admission ({unrestricted['efficiency_group_fraction']*100:.1f} %)")
    for x, y in zip(q, frac):
        ax.annotate(f"{y*100:.2f}%", (x, y * 100), textcoords="offset points", xytext=(0, 6), fontsize=8)
    ax.set_xlabel("quota q (max efficiency groups per 32-group batch)")
    ax.set_ylabel("efficiency-group fraction of optimizer groups (%)")
    ax.set_title("E1-E0: actual efficiency supervision vs quota")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "q_vs_efficiency_fraction.png"), dpi=130)
    plt.close(fig)

    # ---------- fig 2: full-quota rate and replacement fraction ---------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar([str(x) for x in q], [x * 100 for x in full], color="#2b7bba")
    axes[0].set_ylim(0, 105)
    axes[0].set_xlabel("q"); axes[0].set_ylabel("full-quota rate (%)")
    axes[0].set_title("Steps where the quota is actually filled")
    for i, v in enumerate(full):
        axes[0].text(i, v * 100, f"{v*100:.0f}%", ha="center", va="bottom", fontsize=9)
    axes[1].bar([str(x) for x in q], [x * 100 for x in repl], color="#d95f02")
    axes[1].set_xlabel("q"); axes[1].set_ylabel("replacement fraction (%)")
    axes[1].set_title("Mixed-correctness groups replaced")
    for i, v in enumerate(repl):
        axes[1].text(i, v * 100, f"{v*100:.2f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "q_vs_fullquota_and_replacement.png"), dpi=130)
    plt.close(fig)

    # ---------- fig 3: source distribution ------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    nq = [f(r["nq_share"]) * 100 for r in src]
    hp = [f(r["hotpotqa_share"]) * 100 for r in src]
    x = np.arange(len(q))
    ax.bar(x - 0.2, nq, 0.4, label="nq", color="#2b7bba")
    ax.bar(x + 0.2, hp, 0.4, label="hotpotqa", color="#d95f02")
    ax.set_xticks(x, [str(v) for v in q])
    ax.set_xlabel("q"); ax.set_ylabel("share of optimizer groups (%)")
    ax.set_title("Source balance by quota (max shift: %.2f pp)" % max(abs(a - b) for a, b in zip(nq, hp)))
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "q_vs_source_distribution.png"), dpi=130)
    plt.close(fig)

    # ---------- fig 4: displaced mixed k/8 distribution -----------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    ks = list(range(1, 8))
    width = 0.8 / max(1, len([d for d in disp if d.get("replaced")]))
    idx = 0
    for d in disp:
        if not d.get("replaced"):
            continue
        hist = json.loads(d["replaced_k_hist"]) if d.get("replaced_k_hist", "").startswith("{") else {}
        vals = [hist.get(f"k={i}", 0) for i in ks]
        tot = max(sum(vals), 1)
        ax.bar([k + idx * width for k in ks], [v / tot * 100 for v in vals], width,
               label=f"q={d['q']} (n={int(float(d['replaced']))})")
        idx += 1
    ax.set_xticks([k + 0.4 - width for k in ks], [f"{k}/8" for k in ks])
    ax.set_xlabel("correctness count k of the displaced mixed group")
    ax.set_ylabel("share of displaced groups (%)")
    ax.set_title("Which mixed groups get displaced? (no significant bias vs all mixed groups)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "displaced_mixed_k_distribution.png"), dpi=130)
    plt.close(fig)

    # ---------- fig 5: mixed supervision retained vs efficiency gained --------
    fig, ax = plt.subplots(figsize=(7, 4))
    kept = [f(r["mixed_correctness_supervision_retained_pct"]) * 100 for r in proxy]
    eff = [f(r["efficiency_supervision_fraction"]) * 100 for r in proxy]
    ax.plot(kept, eff, "o-", color="#2b7bba")
    for x, y, qq in zip(kept, eff, q):
        ax.annotate(f"q={qq}", (x, y), textcoords="offset points", xytext=(6, -3), fontsize=9)
    ax.set_xlabel("mixed-correctness supervision retained (%)")
    ax.set_ylabel("efficiency supervision fraction (%)")
    ax.set_title("E1-E0 trade-off curve: correctness supervision vs efficiency signal")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "tradeoff_curve.png"), dpi=130)
    plt.close(fig)

    for n in sorted(os.listdir(FIG)):
        print("wrote", os.path.join(FIG, n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

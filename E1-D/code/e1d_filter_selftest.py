#!/usr/bin/env python3
"""E1-D preflight offline selftest (E1-D_task.md §9) — CPU only, no GPU, no server.

Runs the seven mandatory tests on REAL saved rollouts and writes
`E1-D/analysis/preflight_selftest.{md,json}`.

  TEST 1  reward equivalence: E1-D scorer == E1-B scorer, rollout by rollout, on the real
          E1-0 audit (147,200 rollouts) and the real E1-B run (641,280 rollouts)
  TEST 2  lambda is exactly 0.05
  TEST 3  filter baseline equivalence: the `em`-metric filter retains exactly the same
          groups as the original GAP correctness filter on both real pools, and reproduces
          the S_GAP counts E1-C reported (340 for E1-0, 1,412 for E1-B)
  TEST 4  no all-correct revival: all-correct efficiency-variable groups are NOT retained
          by the em filter (while the shaped filter would retain them)
  TEST 5  mixed-group shaping preserved: inside a baseline-retained mixed group with cost
          variation, the efficient and inefficient correct rollouts get different shaped
          rewards and different GRPO advantages
  TEST 6  validation unchanged: the validation scorer/manager path is the original GAP one
  TEST 7  no NaN/Inf in rewards or advantages
"""
import hashlib
import importlib.util
import json
import os
import sys
from collections import defaultdict

import numpy as np

REPO = "/data01/wyy/Graph-Agent-Planning"
E1D = os.path.join(REPO, "E1-D")
OUT_MD = os.path.join(E1D, "analysis", "preflight_selftest.md")
OUT_JSON = os.path.join(E1D, "analysis", "preflight_selftest.json")

E1B_SCORER = os.path.join(REPO, "E1-B/code_changes/e1b_scorer.py")
E1D_SCORER = os.path.join(E1D, "code/e1d_scorer.py")
E1D_MANAGER = os.path.join(E1D, "code/e1d_reward_manager.py")
E1D_TRAIN_LAUNCHER = None  # set from argv if given

POOLS = {
    "E1_0": os.path.join(REPO, "E1-0/runs/20260910_step60to70_e10/audit/rollouts_pid2552655.jsonl"),
    "E1_B": os.path.join(REPO, "E1-B/run_20260911_113432/diagnostics/rollouts_pid3388475.jsonl"),
}
EI_C_S_GAP = {"E1_0": 340, "E1_B": 1412}          # E1-C reference values (sanity only)
EI_C_S_E1 = {"E1_0": 424, "E1_B": 1685}
EI_C_E1_ONLY = {"E1_0": 84, "E1_B": 273}

LAMBDA_EXPECTED = 0.05
TARGET_TRAIN_BATCH_SIZE = 32


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_rollouts(path, limit_groups=None):
    """[(call_index, uid, em, cost, queries, tokens)] with file order preserved."""
    rows = []
    with open(path, errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            em = r.get("em")
            rows.append((int(r.get("call_index")), str(r.get("uid")),
                         0.0 if em is None else float(1.0 if em in (1, 1.0) else 0.0),
                         r.get("logical_search_batches"),
                         r.get("search_queries"), r.get("response_token_length")))
    return rows


def group_rows(rows):
    g = defaultdict(list)
    for i, (ci, uid, em, cost, q, tok) in enumerate(rows):
        g[(ci, uid)].append((em, cost, q, tok))
    return g


def gap_filter_keep(vals):
    v = np.asarray(vals, dtype=float)
    return bool(len(v) == 1 or np.std(v) > 0.0)


def efficiency(correct, cost):
    E = np.zeros_like(np.asarray(cost, dtype=float))
    cor = np.asarray(correct, dtype=float) > 0
    if cor.sum() >= 2:
        c = np.asarray(cost, dtype=float)[cor]
        cmin, cmax = c.min(), c.max()
        if cmax > cmin:
            E[cor] = (cmax - c) / (cmax - cmin)
    return E


def grpo_advantage(scores, uids, eps=1e-6):
    scores = np.asarray(scores, dtype=float)
    by = defaultdict(list)
    for s, u in zip(scores, uids):
        by[u].append(s)
    out = np.zeros_like(scores)
    for u, vals in by.items():
        v = np.asarray(vals, dtype=float)
        idx = [i for i, uu in enumerate(uids) if uu == u]
        if len(v) == 1:
            m, sd = 0.0, 1.0
        else:
            m, sd = float(np.mean(v)), float(np.std(v, ddof=1))
        for i in idx:
            out[i] = (scores[i] - m) / (sd + eps)
    return out


def main():
    launcher = sys.argv[1] if len(sys.argv) > 1 else None
    tests = {}
    mod_b = load_module(E1B_SCORER, "e1b_scorer_selftest")
    mod_d = load_module(E1D_SCORER, "e1d_scorer_selftest")

    # ---------------- TEST 2: lambda -------------------------------------
    lam_b = float(mod_b.DEFAULT_LAMBDA)
    lam_d = float(mod_d.DEFAULT_LAMBDA)
    lam_src = float(os.environ.get("E1D_LAMBDA", "0.05"))
    tests["TEST2_lambda"] = {
        "pass": abs(lam_b - LAMBDA_EXPECTED) < 1e-12 and abs(lam_d - LAMBDA_EXPECTED) < 1e-12
                and lam_src == LAMBDA_EXPECTED,
        "e1b_default_lambda": lam_b, "e1d_default_lambda": lam_d,
        "e1d_env_lambda_default": lam_src, "expected": LAMBDA_EXPECTED,
    }

    # ---------------- TEST 1: reward equivalence on real rollouts --------
    equiv = {}
    total_checked = 0
    total_mismatch = 0
    for pool, path in POOLS.items():
        rows = load_rollouts(path)
        g = group_rows(rows)
        mism = 0
        checked = 0
        nan_ct = 0
        for (ci, uid), recs in g.items():
            em = np.array([r[0] for r in recs], dtype=float)
            cost = np.array([r[1] if r[1] is not None else 0.0 for r in recs], dtype=float)
            box_d = {"i": 0}
            box_b = {"i": 0}

            def stub(box):
                def f(ds, p, s, gt):
                    v = float(em[box["i"]]); box["i"] += 1
                    return {"score": v, "em": v}
                return f
            mod_d.compute_score_em = stub(box_d)
            mod_b.compute_score_em = stub(box_b)
            args = ([None] * len(em),) * 5
            rd = mod_d.compute_score_em_efficiency_batch(*args, uids=[uid] * len(em), costs=cost.tolist())
            rb = mod_b.compute_score_em_efficiency_batch(*args, uids=[uid] * len(em), costs=cost.tolist())
            for a, b in zip(rd, rb):
                checked += 1
                nan_ct += int(not np.isfinite(a["score"]) or not np.isfinite(a["efficiency"]))
                if abs(a["score"] - b["score"]) > 1e-12 or abs(a["efficiency"] - b["efficiency"]) > 1e-12 \
                        or a["em"] != b["em"]:
                    mism += 1
        equiv[pool] = {"rollouts_checked": checked, "mismatches": mism,
                       "non_finite": nan_ct, "groups": len(g)}
        total_checked += checked
        total_mismatch += mism
    tests["TEST1_reward_equivalence_with_E1B"] = {
        "pass": total_mismatch == 0, "total_rollouts_checked": total_checked,
        "total_mismatches": total_mismatch, "per_pool": equiv,
        "method": ("both scorers called on the same real rollouts with compute_score_em stubbed by the "
                   "recorded EM, so only the group/efficiency math is compared"),
    }
    tests["TEST7_no_nan_inf"] = {
        "pass": all(v["non_finite"] == 0 for v in equiv.values()),
        "non_finite_per_pool": {k: v["non_finite"] for k, v in equiv.items()},
    }

    # ---------------- TEST 3/4: filter equivalence + no all-correct revival
    filt = {}
    for pool, path in POOLS.items():
        rows = load_rollouts(path)
        g = group_rows(rows)
        s_em, s_shaped, all_correct_eff = set(), set(), set()
        n_all_correct = n_mixed = n_all_wrong = 0
        for (ci, uid), recs in g.items():
            em = np.array([r[0] for r in recs], dtype=float)
            cost = np.array([r[1] if r[1] is not None else 0.0 for r in recs], dtype=float)
            E = efficiency(em, cost)
            R = em * (1.0 + LAMBDA_EXPECTED * E)
            key = f"{ci}:{uid}"
            keep_em_metric = gap_filter_keep(em)          # filter metric = "em"
            keep_shaped = gap_filter_keep(R)
            cc = int((em > 0).sum())
            n_all_correct += int(cc == len(em)); n_mixed += int(0 < cc < len(em)); n_all_wrong += int(cc == 0)
            if keep_em_metric:
                s_em.add(key)
            if keep_shaped:
                s_shaped.add(key)
            if cc == len(em) and float(E.max()) > 0.0:
                all_correct_eff.add(key)
        revived = all_correct_eff & s_shaped
        revived_by_em = all_correct_eff & s_em
        filt[pool] = {
            "candidate_groups": len(g),
            "all_correct": n_all_correct, "mixed": n_mixed, "all_wrong": n_all_wrong,
            "S_em_metric": len(s_em), "S_shaped_metric": len(s_shaped),
            "e1_only_shaped": len(s_shaped - s_em),
            "gap_only": len(s_em - s_shaped),
            "all_correct_efficiency_variable": len(all_correct_eff),
            "of_those_revived_by_shaped_filter": len(revived),
            "of_those_revived_by_em_filter": len(revived_by_em),
            "reference_E1C_S_GAP": EI_C_S_GAP[pool],
            "reference_E1C_S_E1": EI_C_S_E1[pool],
            "reference_E1C_E1_only": EI_C_E1_ONLY[pool],
            "matches_E1C_S_GAP": len(s_em) == EI_C_S_GAP[pool],
            "matches_E1C_S_E1": len(s_shaped) == EI_C_S_E1[pool],
        }
    tests["TEST3_filter_baseline_equivalence"] = {
        "pass": all(v["matches_E1C_S_GAP"] and v["matches_E1C_S_E1"] and v["gap_only"] == 0
                    for v in filt.values()),
        "per_pool": filt,
        "note": ("filter metric `em` keeps iff std(EM)>0; the GAP baseline reward IS EM, so the "
                 "retained set must be identical - it reproduces E1-C's S_GAP exactly on both pools"),
    }
    tests["TEST4_no_all_correct_revival"] = {
        "pass": all(v["of_those_revived_by_em_filter"] == 0
                    and v["of_those_revived_by_shaped_filter"] == v["all_correct_efficiency_variable"]
                    for v in filt.values()),
        "per_pool": {k: {"all_correct_efficiency_variable": v["all_correct_efficiency_variable"],
                         "revived_by_em_filter": v["of_those_revived_by_em_filter"],
                         "revived_by_shaped_filter": v["of_those_revived_by_shaped_filter"]}
                     for k, v in filt.items()},
    }

    # ---------------- TEST 5: mixed-group shaping preserved --------------
    rng = np.random.default_rng(0)
    checked = 0
    reward_splits = 0
    adv_splits = 0
    example = None
    for pool, path in POOLS.items():
        rows = load_rollouts(path)
        g = group_rows(rows)
        keys = [k for k in g if 0 < int(sum(r[0] for r in g[k])) < len(g[k])]
        for k in keys:
            recs = g[k]
            em = np.array([r[0] for r in recs], dtype=float)
            cost = np.array([r[1] if r[1] is not None else 0.0 for r in recs], dtype=float)
            if not gap_filter_keep(em):
                continue
            E = efficiency(em, cost)
            cor = em > 0
            if E[cor].max() <= 0:
                continue
            checked += 1
            R = em * (1.0 + LAMBDA_EXPECTED * E)
            if len(set(np.round(R[cor], 12))) > 1:
                reward_splits += 1
            adv = grpo_advantage(R, [f"{k}"] * len(R))
            eff_adv = adv[cor & (E > 0)]
            ineff_adv = adv[cor & (E == 0)]
            if len(eff_adv) and len(ineff_adv) and abs(float(np.mean(eff_adv)) - float(np.mean(ineff_adv))) > 1e-9:
                adv_splits += 1
                if example is None:
                    example = {"pool": pool, "group": f"{k[0]}:{k[1]}",
                               "efficient_correct_advantage_mean": float(np.mean(eff_adv)),
                               "inefficient_correct_advantage_mean": float(np.mean(ineff_adv)),
                               "shaped_rewards": [float(x) for x in R]}
    tests["TEST5_mixed_group_shaping_preserved"] = {
        "pass": checked > 0 and reward_splits == checked and adv_splits > 0,
        "mixed_groups_checked": checked,
        "with_distinct_shaped_reward_among_correct": reward_splits,
        "with_distinct_grpo_advantage_between_efficient_and_inefficient": adv_splits,
        "example": example,
        "note": ("filter channel closed (EM std==0 all-correct groups are dropped) while the "
                 "objective channel stays open inside retained mixed groups"),
    }

    # ---------------- TEST 6: validation unchanged -----------------------
    val_ok = True
    details = {}
    try:
        mgr_src = open(E1D_MANAGER).read()
        details["manager_val_delegates_to_super"] = "return super().verify(data)" in mgr_src
        details["manager_has_is_train_flag"] = "self.is_train = num_examine == 0" in mgr_src
        val_ok &= details["manager_val_delegates_to_super"] and details["manager_has_is_train_flag"]
    except OSError as e:
        details["manager_read_error"] = str(e)
        val_ok = False
    if launcher and os.path.exists(launcher):
        src = open(launcher).read()
        details["val_scorer_is_original_mhqa_eval"] = "mhqa_eval.py" in src and "compute_score_em_batch" in src
        details["filter_metric_is_em"] = "algorithm.filter_groups.metric=em" in src
        val_ok &= details["val_scorer_is_original_mhqa_eval"] and details["filter_metric_is_em"]
    else:
        details["launcher_checked"] = None
    tests["TEST6_validation_unchanged"] = {"pass": val_ok, "details": details}

    # ---------------- TEST 7: advantage finiteness on a real batch -------
    adv_batch = None
    try:
        rows = load_rollouts(POOLS["E1_B"])
        g = group_rows(rows)
        # use the ACTUAL optimizer batch: the first 32 groups kept by the EM filter in
        # generation order for step 71 (ray_trainer: accumulate kept, then batch[:32*8])
        ordered = sorted(g.keys(), key=lambda k: (k[0],))
        sel = []
        for k in ordered:
            recs = g[k]
            em = np.array([r[0] for r in recs], dtype=float)
            if gap_filter_keep(em):
                sel.append(k)
            if len(sel) >= 32:
                break
        scores, uids = [], []
        for k in sel:
            recs = g[k]
            em = np.array([r[0] for r in recs], dtype=float)
            cost = np.array([r[1] if r[1] is not None else 0.0 for r in recs], dtype=float)
            E = efficiency(em, cost)
            R = em * (1.0 + LAMBDA_EXPECTED * E)
            scores += list(R)
            uids += [f"{k[0]}:{k[1]}"] * len(R)
        adv = grpo_advantage(scores, uids)
        adv_batch = {"n": len(adv), "n_groups": len(sel), "nan": int(np.isnan(adv).sum()),
                     "inf": int(np.isinf(adv).sum()), "max_abs": float(np.max(np.abs(adv))),
                     "mean_abs": float(np.mean(np.abs(adv))),
                     "note": "advantage on the EM-filter-selected optimizer batch of step 71 (shaped reward)"}
    except Exception as e:  # noqa: BLE001
        adv_batch = {"error": f"{type(e).__name__}: {e}"}
    tests["TEST7_advantage_finite"] = {
        "pass": bool(adv_batch and adv_batch.get("nan") == 0 and adv_batch.get("inf") == 0),
        "detail": adv_batch,
    }

    all_pass = all(t["pass"] for t in tests.values())
    result = {"all_pass": all_pass, "tests": tests,
              "scorer_sha256": {
                  "e1b": hashlib.sha256(open(E1B_SCORER, "rb").read()).hexdigest(),
                  "e1d": hashlib.sha256(open(E1D_SCORER, "rb").read()).hexdigest()},
              "target_train_batch_size": TARGET_TRAIN_BATCH_SIZE}
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        json.dump(result, fh, indent=2, default=str)

    md = ["# E1-D preflight offline selftest (E1-D_task.md §9)", "",
          f"**ALL TESTS: {'PASS' if all_pass else 'FAIL'}**", ""]
    for name in sorted(tests):
        t = tests[name]
        md.append(f"## {name}: {'PASS' if t['pass'] else 'FAIL'}")
        md.append("")
        md.append("```json")
        md.append(json.dumps({k: v for k, v in t.items() if k != "pass"}, indent=2, default=str)[:4000])
        md.append("```")
        md.append("")
    with open(OUT_MD, "w") as fh:
        fh.write("\n".join(md))
    print(json.dumps({"all_pass": all_pass,
                      "tests": {k: v["pass"] for k, v in tests.items()}}, indent=2))
    print("wrote", OUT_MD, "and", OUT_JSON)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

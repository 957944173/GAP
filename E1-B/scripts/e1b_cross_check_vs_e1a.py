#!/usr/bin/env python3
"""E1-B reward-equivalence proof vs the E1-A implementation.

E1-B_task.md s4: "优先复用 E1-A reward 实现。不要重新设计 reward。仅确认
compute_score_em_efficiency_batch 正常工作。"

E1-B therefore ships a **verbatim copy** of E1-A's implementation
(`E1-A/code/*.py` -> `E1-B/code_changes/e1b_*.py`, identifier-only rename
e1a_->e1b_, E1A_->E1B_); no reward logic was re-designed.  This script proves it
two ways:

  1. SOURCE IDENTITY -- re-apply the same mechanical rename to every E1-A source
     file and byte-compare with the E1-B copy.
  2. NUMERIC IDENTITY ON REAL DATA -- replay both scorers' group-relative math on
     the 147,200 real rollouts of the E1-0 step60->70 audit (real uid groups, real
     `logical_search_batches`).  `compute_score_em` is stubbed with the EM the
     original reward produced for that rollout, so the comparison isolates exactly
     the part E1-B could have broken (the efficiency/group math).

Output: E1-B/diagnostics/reward_equivalence_check.json
"""
import hashlib
import importlib.util
import json
import os
import re
import sys

REPO = "/data01/wyy/Graph-Agent-Planning"
E1A_CODE = os.path.join(REPO, "E1-A", "code")
E1B_CODE = os.path.join(REPO, "E1-B", "code_changes")
AUDIT = os.path.join(REPO, "E1-0", "runs", "20260910_step60to70_e10", "audit", "rollouts_pid2552655.jsonl")
OUT = os.path.join(REPO, "E1-B", "diagnostics", "reward_equivalence_check.json")

RENAMES = [
    (re.compile(r"E1AB"), "E1BB"),
    (re.compile(r"E1A_"), "E1B_"),
    (re.compile(r"e1a_"), "e1b_"),
    (re.compile(r"\[E1-A"), "[E1-B"),
    (re.compile(r"E1-A"), "E1-B"),
    (re.compile(r"code/pythonpath_e1a"), "code_changes/pythonpath_e1b"),
]


def rename(text):
    for pat, rep in RENAMES:
        text = pat.sub(rep, text)
    return text


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def source_identity():
    files = sorted(f for f in os.listdir(E1A_CODE) if f.endswith(".py"))
    pairs = []
    ok = True
    for f in files:
        nb = f.replace("e1a_", "e1b_")
        with open(os.path.join(E1A_CODE, f)) as fh:
            a = fh.read()
        with open(os.path.join(E1B_CODE, nb)) as fh:
            b = fh.read()
        same = rename(a) == b
        ok = ok and same
        pairs.append({
            "e1a": f"E1-A/code/{f}",
            "e1b": f"E1-B/code_changes/{nb}",
            "sha256_e1a": hashlib.sha256(a.encode()).hexdigest(),
            "sha256_e1b": hashlib.sha256(b.encode()).hexdigest(),
            "identical_after_rename": same,
        })
    shim_a = os.path.join(E1A_CODE, "pythonpath_e1a", "sitecustomize.py")
    shim_b = os.path.join(E1B_CODE, "pythonpath_e1b", "sitecustomize.py")
    with open(shim_a) as fh:
        a = fh.read()
    with open(shim_b) as fh:
        b = fh.read()
    same = rename(a) == b
    ok = ok and same
    pairs.append({
        "e1a": "E1-A/code/pythonpath_e1a/sitecustomize.py",
        "e1b": "E1-B/code_changes/pythonpath_e1b/sitecustomize.py",
        "sha256_e1a": hashlib.sha256(a.encode()).hexdigest(),
        "sha256_e1b": hashlib.sha256(b.encode()).hexdigest(),
        "identical_after_rename": same,
    })
    return ok, pairs


def numeric_identity():
    mod_a = load_module(os.path.join(E1A_CODE, "e1a_scorer.py"), "e1a_scorer_ck")
    mod_b = load_module(os.path.join(E1B_CODE, "e1b_scorer.py"), "e1b_scorer_ck")
    if float(mod_a.DEFAULT_LAMBDA) != float(mod_b.DEFAULT_LAMBDA):
        return False, {"error": "DEFAULT_LAMBDA differs"}
    lam = float(mod_a.DEFAULT_LAMBDA)

    groups = {}
    with open(AUDIT) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (r.get("call_index"), r.get("uid"))
            groups.setdefault(key, []).append(r)

    checked = 0
    mismatches = []
    active_groups = 0
    shaped_rows = 0
    for (call_index, uid), rows in groups.items():
        ems = [float(r.get("reward_original") or 0.0) for r in rows]
        costs = [r.get("logical_search_batches") for r in rows]
        uids = [str(uid)] * len(rows)
        n = len(rows)

        # `compute_score_em` is looked up in module globals on every call, so
        # stubbing it with the EM recorded by the original reward isolates the
        # group/efficiency math that E1-B could have broken.
        def make_stub(vals):
            box = {"i": 0}

            def stub(ds, p, s, gt):
                v = vals[box["i"]]
                box["i"] += 1
                return {"score": v, "em": v}

            return stub

        mod_a.compute_score_em = make_stub(ems)
        mod_b.compute_score_em = make_stub(ems)
        ra = mod_a.compute_score_em_efficiency_batch(
            [None] * n, [None] * n, [None] * n, [None] * n, [None] * n,
            uids=uids, costs=costs)
        rb = mod_b.compute_score_em_efficiency_batch(
            [None] * n, [None] * n, [None] * n, [None] * n, [None] * n,
            uids=uids, costs=costs)
        for i in range(n):
            checked += 1
            for k in ("score", "em", "efficiency", "cost", "group_size",
                      "group_correct_count", "group_cost_min", "group_cost_max",
                      "efficiency_active", "lambda"):
                va, vb = ra[i][k], rb[i][k]
                if isinstance(va, float) or isinstance(vb, float):
                    if va is None or vb is None:
                        same = va == vb
                    else:
                        same = abs(float(va) - float(vb)) <= 1e-12
                else:
                    same = va == vb
                if not same:
                    mismatches.append({"call_index": call_index, "uid": uid, "row": i,
                                       "field": k, "e1a": va, "e1b": vb})
            if ra[i]["efficiency_active"]:
                shaped_rows += 1
        if any(r["efficiency_active"] for r in ra):
            active_groups += 1

    return (not mismatches), {
        "lambda": lam,
        "groups_replayed": len(groups),
        "rollouts_replayed": checked,
        "efficiency_active_groups": active_groups,
        "rollouts_with_nonzero_efficiency": shaped_rows,
        "mismatch_count": len(mismatches),
        "mismatches_head": mismatches[:5],
    }


def main():
    src_ok, pairs = source_identity()
    num_ok, num_info = numeric_identity()
    report = {
        "purpose": "prove the E1-B reward is a verbatim reuse of the E1-A reward (no redesign)",
        "source_identity_ok": src_ok,
        "numeric_identity_ok": num_ok,
        "source_pairs": pairs,
        "numeric": num_info,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps({"source_identity_ok": src_ok, "numeric_identity_ok": num_ok,
                      **num_info}, indent=2))
    print(f"report -> {OUT}")
    return 0 if (src_ok and num_ok) else 1


if __name__ == "__main__":
    sys.exit(main())

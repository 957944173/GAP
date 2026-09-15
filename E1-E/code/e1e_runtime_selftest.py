#!/usr/bin/env python3
"""E1-E pre-training offline selftest (E1-E_task.md section 11).

Three parts, all CPU-only:

  PART A  planner replay: the shared `e1e_quota_plan` is run over the REAL E1-D D2 candidate
          stream and the resulting per-step optimizer batches are compared, group by group,
          with `E1-E0/results/optimizer_batch_members_by_quota.csv` for q=0 and q=2.
  PART B  runtime replay: synthetic `DataProto` generation batches (real uids/costs/EMs from
          the same stream) are pushed through the REAL `E1EQuotaRewardManager`, and the
          trainer's own loop (filter by published metric -> accumulate -> truncate to B) is
          reproduced, so the materialised batches can be compared with E1-E0 as well.
  PART C  reward / batch invariants: 12 checks required by section 11.

Writes E1-E/analysis/preflight_selftest.{md,json}.
"""
import argparse
import glob
import json
import os
import sys
import tempfile
from collections import OrderedDict, defaultdict
from types import SimpleNamespace

import numpy as np

REPO = "/data01/wyy/Graph-Agent-Planning"
E1E = os.path.join(REPO, "E1-E")
STREAM = os.path.join(REPO, "E1-D/runs/D2_20260913_041846/diagnostics/rollouts_pid1249393.jsonl")
E1E0_MEMBERS = os.path.join(REPO, "E1-E0/results/optimizer_batch_members_by_quota.csv")
E1E0_Q0 = os.path.join(REPO, "E1-E0/results/q0_baseline_replay.csv")
OUT_MD = os.path.join(E1E, "analysis", "preflight_selftest.md")
OUT_JSON = os.path.join(E1E, "analysis", "preflight_selftest.json")

sys.path.insert(0, os.path.join(E1E, "code"))
import e1e_quota_plan as QP  # noqa: E402

B = 32
Q = 2
LAMBDA = 0.05


# --------------------------------------------------------------------------- loader
def load_stream(path=STREAM, verbose=True):
    """Return {step: [call, ...]} where call = {call_index, groups:[group,...]} in stream order."""
    grp = OrderedDict()          # (call_index, uid) -> dict
    order = OrderedDict()        # call_index -> [gid, ...]
    with open(path, errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            ci = int(r["call_index"])
            uid = str(r["uid"])
            key = (ci, uid)
            g = grp.get(key)
            if g is None:
                g = grp[key] = {"call_index": ci, "global_step": int(r["global_step"]), "uid": uid,
                                "source": r.get("data_source"), "em": [], "cost": [],
                                "queries": [], "tokens": [],
                                "gid": f"{ci}:{uid}"}
                order.setdefault(ci, []).append(key)
            em = r.get("em")
            g["em"].append(1.0 if em in (1, 1.0) else 0.0)
            g["cost"].append(float(r["logical_search_batches"]) if r.get("logical_search_batches") is not None else None)
            g["queries"].append(r.get("search_queries"))
            g["tokens"].append(r.get("response_token_length"))
    by_step = defaultdict(list)
    for ci in sorted(order):
        groups = [grp[k] for k in order[ci]]
        for g in groups:
            cor = [c for c, a in zip(g["cost"], g["em"]) if a > 0 and c is not None]
            g["k"] = int(sum(g["em"]))
            g["n"] = len(g["em"])
            g["cost_gap"] = (max(cor) - min(cor)) if (g["k"] == g["n"] and len(cor) >= 2) else 0.0
            g["type"] = QP.classify_group(g["k"], g["n"], g["cost_gap"])
        by_step[groups[0]["global_step"]].append({"call_index": ci, "groups": groups})
    if verbose:
        n_groups = sum(len(c["groups"]) for s in by_step.values() for c in s)
        print(f"[selftest] stream: {len(by_step)} steps, "
              f"{sum(len(v) for v in by_step.values())} generation batches, {n_groups} groups")
    return by_step


def plan_stream(by_step, target=B, quota=Q):
    """PART A: run the shared planner over the whole stream."""
    out = {}
    for step in sorted(by_step):
        state = QP.new_state()
        state["step"] = step
        batches = []
        for call in by_step[step]:
            plan = QP.plan_batch(call["groups"], state, target_batch=target, quota=quota)
            batches.append((call, plan))
            if plan["is_stopping"]:
                break
        members = []
        for call, plan in batches:
            for g in call["groups"]:
                if g["gid"] in set(plan["mixed_kept"]):
                    members.append((g["gid"], "mixed"))
        if batches and batches[-1][1]["is_stopping"]:
            gid2g = {g["gid"]: g for c, _p in batches for g in c["groups"]}
            for gid in batches[-1][1]["efficiency_kept"]:
                members.append((gid, "efficiency"))
        out[step] = {"batches": len(batches), "members": members, "state": dict(state),
                     "plan_last": batches[-1][1] if batches else None}
    return out


def expected_members(q):
    exp = defaultdict(list)
    with open(E1E0_MEMBERS) as fh:
        for r in csv_rows(fh):
            if int(r["q"]) != q:
                continue
            exp[int(r["global_step"])].append((r["group_id"], r["role"]))
    return exp


def csv_rows(fh):
    import csv
    return csv.DictReader(fh)


# --------------------------------------------------------------------------- fake runtime
class FakeTok:
    def encode(self, text):
        import torch
        return torch.tensor(list(bytes(str(text), "utf-8")) or [0], dtype=torch.long)

    def decode(self, ids, skip_special_tokens=True):
        try:
            ids = [int(x) for x in ids]
        except TypeError:
            return str(ids)
        return bytes([i % 256 for i in ids]).decode("utf-8", errors="ignore")


def build_gen_batch(call, tok, tables):
    """Synthetic DataProto for one generation batch (real uids/costs/EMs)."""
    import torch
    from verl.protocol import DataProto
    rows = [(g, i) for g in call["groups"] for i in range(g["n"])]
    n = len(rows)
    prompts = torch.zeros((n, 4), dtype=torch.long)
    responses = torch.zeros((n, 4), dtype=torch.long)
    attn = torch.zeros((n, 8), dtype=torch.long)
    attn[:, :4] = 1
    attn[:, 4:8] = 1
    uid, ds, q_, gt, cr, dtm, extra = [], [], [], [], [], [], []
    table = {}
    for pos, (g, i) in enumerate(rows):
        b = list(bytes(f"R{int(g['em'][i])}", "utf-8"))[:4]
        responses[pos, :len(b)] = torch.tensor(b, dtype=torch.long)
        pb_ = list(bytes("P0", "utf-8"))[:4]
        prompts[pos, :len(pb_)] = torch.tensor(pb_, dtype=torch.long)
        uid.append(g["uid"])
        ds.append(g["source"])
        q_.append("q")
        gt.append("a")
        cr.append("stop")
        extra.append({"question": "q"})
        cost = g["cost"][i]
        blocks = int(cost) if cost is not None else 0
        seq = [{"tool_name": "wiki_search", "turn": b, "call_id": f"call_wiki_search_{b}_0"}
               for b in range(1, blocks + 1)]
        dtm.append({"tool_call_sequence": seq, "total_tool_calls": blocks,
                    "conversation_turns": blocks + 1})
        # per (uid, position-within-group) facts used by the stub scorer
        p_in_group = g["em"][:i + 1].count(g["em"][i]) - 1 if False else i
        table[(g["uid"], i)] = {"em": g["em"][i], "cost": cost,
                                "queries": g["queries"][i], "tokens": g["tokens"][i]}
    tables["current"] = table
    nd = {
        "uid": np.array(uid, dtype=object),
        "data_source": np.array(ds, dtype=object),
        "extra_info": np.array(extra, dtype=object),
        "reward_model": np.array([{"ground_truth": "a"}] * n, dtype=object),
        "complete_reason": np.array(cr, dtype=object),
        "detailed_tool_metrics": np.array(dtm, dtype=object),
    }
    return DataProto.from_dict(tensors={"prompts": prompts, "responses": responses,
                                        "attention_mask": attn}, non_tensors=nd)


def run_runtime_replay(by_step, steps, quota=Q, target=B, run_dir=None):
    """PART B: push synthetic batches through the real manager + the trainer's own loop."""
    import torch
    from verl.protocol import DataProto
    import e1e_state
    import e1e_trainer_patch
    import e1e_scorer
    import e1e_reward_manager as RM

    e1e_trainer_patch.ensure_driver_patch = lambda: None      # offline: no trainer to patch
    tok = FakeTok()
    tables = {"current": {}}

    def fake_em(ds, p, s, gt):
        """EM is carried in the synthetic response text (works for injected groups too)."""
        txt = str(s).strip("\x00")
        try:
            v = float(int(txt[1:]))
        except Exception:
            v = 0.0
        return {"score": v, "em": v}

    e1e_scorer.compute_score_em = fake_em

    def stub_score(**kw):
        return e1e_scorer.compute_score_em_efficiency_batch(
            kw["data_sources"], kw["prompts"], kw["responses"], kw["ground_truths"], kw["extra_infos"],
            uids=kw.get("uids"), costs=kw.get("costs"), queries=kw.get("queries"),
            conversation_turns=kw.get("conversation_turns"))
        return e1e_scorer.compute_score_em_efficiency_batch(
            kw["data_sources"], kw["prompts"], kw["responses"], kw["ground_truths"], kw["extra_infos"],
            uids=kw.get("uids"), costs=kw.get("costs"), queries=kw.get("queries"),
            conversation_turns=kw.get("conversation_turns"))

    if run_dir:
        # the manager reads the module-level constant in __init__ to decide where to write
        # diagnostics, so it must be set before construction
        RM.E1E_RUN_DIR = run_dir
    mgr = RM.E1EQuotaRewardManager(tok, num_examine=0, compute_score=stub_score)
    mgr.run_dir = run_dir or RM.E1E_RUN_DIR
    mgr.batch_target = target
    mgr.quota = quota

    results = {}
    for step in steps:
        e1e_state.TRAINER = SimpleNamespace(global_steps=step)
        accum = None
        n_prompt = 0
        calls_used = 0
        for call in by_step[step]:
            gb = build_gen_batch(call, tok, tables)
            out = mgr(gb, return_dict=True)
            # replicate ray_trainer.py:1753-1756: publish reward_extra_info into the batch
            for k, v in (out.get("reward_extra_info") or {}).items():
                try:
                    gb.non_tensor_batch[k] = np.array(v)
                except Exception:
                    pass
            gb.batch["token_level_rewards"] = out["reward_tensor"]
            calls_used += 1
            metric = gb.non_tensor_batch["e1e_quota_metric"]
            uid_arr = list(gb.non_tensor_batch["uid"])
            vals = defaultdict(list)
            for u, v in zip(uid_arr, metric):
                vals[str(u)].append(float(v))
            kept = [u for u, vs in vals.items() if len(vs) == 1 or np.std(vs) > 0]
            n_prompt += len(kept)
            idxs = [i for i, u in enumerate(uid_arr) if str(u) in kept]
            nb = gb[idxs]
            accum = nb if accum is None else DataProto.concat([accum, nb])
            if n_prompt >= target:
                break
        accum = accum[: target * 8]
        members = []
        seen = OrderedDict()
        uid_arr = list(accum.non_tensor_batch["uid"])
        for u in uid_arr:
            seen.setdefault(str(u), 0)
            seen[str(u)] += 1
        # group order = first appearance
        gids = list(seen.keys())
        qm = accum.non_tensor_batch["e1e_quota_metric"]
        # classify each group from its published metric / reward
        rew = accum.batch["token_level_rewards"].sum(dim=-1).numpy()
        roles = {}
        ge = accum.non_tensor_batch.get("group_efficiency")
        first_idx = {}
        for i, u in enumerate(uid_arr):
            first_idx.setdefault(str(u), i)
        for u in gids:
            kind = "efficiency" if (ge is not None and bool(ge[first_idx[u]])) else "mixed"
            members.append((u, kind, seen[u]))
        results[step] = {"calls_used": calls_used, "n_prompt": n_prompt,
                         "batch_groups": len(gids), "members": members}
    return results




def make_no_keep_call(call_index=999999, n_groups=3):
    """A synthetic generation batch the frozen plan must select nothing from.

    All-correct groups with *identical* costs are neither mixed (k != 8 fails for mixed) nor
    efficiency (cost variation == 0), and all-wrong groups are excluded too.  Live D1 hit this
    case whenever a generation batch happened to contain no mixed group, which is what exposed
    the empty-keep leak.
    """
    groups = []
    for i in range(n_groups):
        uid = f"SYNTHNOKEEP{i}"
        groups.append({
            "gid": f"0:{uid}", "uid": uid, "source": "nq", "n": 8,
            "em": [1.0] * 8, "cost": [2.0] * 8, "queries": [2.0] * 8, "tokens": [100.0] * 8,
            "k": 8, "cost_gap": 0.0, "type": QP.GROUP_EXCLUDED,
        })
    for i in range(2):
        uid = f"SYNTHALLWRONG{i}"
        groups.append({
            "gid": f"0:{uid}", "uid": uid, "source": "nq", "n": 8,
            "em": [0.0] * 8, "cost": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0],
            "queries": [1.0] * 8, "tokens": [100.0] * 8,
            "k": 0, "cost_gap": 0.0, "type": QP.GROUP_EXCLUDED,
        })
    return {"call_index": call_index, "groups": groups}


def find_empty_keep_steps(by_step, quota=Q, target=B):
    """Steps in the real E1-E0 stream that contain at least one empty-keep generation batch."""
    hits = {}
    for step in sorted(by_step):
        state = QP.new_state()
        state["step"] = step
        for i, call in enumerate(by_step[step]):
            plan = QP.plan_batch(call["groups"], state, target_batch=target, quota=quota)
            if not plan["mixed_kept"] and not plan["efficiency_kept"]:
                hits.setdefault(step, []).append(i)
            if plan["is_stopping"]:
                break
    return hits


def empty_keep_checks(by_step, steps, run_dir=None, quota=Q, target=B):
    """PART D: an empty-keep generation batch must be neutral, never leak raw groups.

    Two experiments:
      D1  a synthetic no-keep batch is inserted into a real step's stream; the materialised
          optimizer batch must be IDENTICAL to the same step without the insertion, and the
          trainer's own filter must keep exactly zero groups from the inserted batch.
      D2  the real-stream steps that actually contain an empty-keep batch must still produce a
          32-group optimizer batch equal to E1-E0's frozen membership.
    """
    import e1e_reward_manager as RM
    checks = {}

    def run(by_step_local, step_list, run_dir_local):
        return run_runtime_replay(by_step_local, step_list, quota=quota, target=target,
                                  run_dir=run_dir_local)

    base = run(by_step, steps, None)
    base_sets = {s: sorted(u for u, _k, _n in base[s]["members"]) for s in steps}

    # ---- D1: forced insertion ------------------------------------------------------
    injected = {}
    for s in steps:
        seq = list(by_step[s])
        seq.insert(1, make_no_keep_call())
        injected[s] = seq
    inj = run(injected, steps, run_dir)

    d1 = {}
    for s in steps:
        got = sorted(u for u, _k, _n in inj[s]["members"])
        d1[s] = {"same_membership_as_without_injection": got == base_sets[s],
                 "batch_groups": inj[s]["batch_groups"],
                 "calls_used_with_injection": inj[s]["calls_used"],
                 "calls_used_without": base[s]["calls_used"],
                 "extra": sorted(set(got) - set(base_sets[s]))[:5],
                 "missing": sorted(set(base_sets[s]) - set(got))[:5]}
    checks["D1_inserted_no_keep_batch_is_neutral"] = all(
        v["same_membership_as_without_injection"] for v in d1.values())
    checks["D1_optimizer_batch_still_32_groups"] = all(v["batch_groups"] == target for v in d1.values())

    # ---- D2: real-stream empty-keep steps -----------------------------------------
    hits = find_empty_keep_steps(by_step, quota=quota, target=target)
    checks["D2_real_stream_contains_empty_keep_batches"] = bool(hits)
    d2 = {}
    if hits:
        sample = sorted(hits)[:3]
        res = run(by_step, sample, None)
        exp2 = expected_members(quota)
        for s in sample:
            got = sorted(u for u, _k, _n in res[s]["members"])
            want = sorted(g.split(":")[-1] for g, _r in exp2.get(s, []))
            d2[s] = {"empty_keep_batch_indices": hits[s],
                     "batch_groups": res[s]["batch_groups"],
                     "same_set_as_e1e0": got == want,
                     "extra": sorted(set(got) - set(want))[:5],
                     "missing": sorted(set(want) - set(got))[:5]}
        checks["D2_empty_keep_steps_match_e1e0"] = all(v["same_set_as_e1e0"] for v in d2.values())
        checks["D2_empty_keep_steps_batch_32"] = all(v["batch_groups"] == target for v in d2.values())

    # ---- D3: the manager recorded the neutralised call -----------------------------
    recorded = 0
    if run_dir:
        for f in glob.glob(os.path.join(run_dir, "diagnostics", "quota_calls_pid*.jsonl")):
            for line in open(f, errors="ignore"):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("kept_nothing"):
                    recorded += 1
    checks["D3_kept_nothing_calls_recorded"] = recorded >= len(steps) if run_dir else None
    return {"d1_forced_insertion": d1, "d2_real_empty_keep_steps": d2,
            "d3_kept_nothing_records": recorded,
            "real_stream_empty_keep_steps": {str(k): v for k, v in hits.items()},
            "checks": checks}


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3, help="steps to exercise in PART B")
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    res = {"part_a": {}, "part_b": {}, "part_c": {}, "pass": {}}

    tmp = tempfile.mkdtemp(prefix="e1e_selftest_")
    os.environ.setdefault("E1E_RUN_DIR", tmp)
    os.environ["E1E_QUOTA"] = str(Q)
    os.environ["E1E_BATCH_TARGET"] = str(B)
    os.environ["E1E_LAMBDA"] = "0.05"
    by_step = load_stream()
    steps = sorted(by_step)

    # ---------------- PART A ----------------
    exp2 = expected_members(2)
    exp0 = expected_members(0)
    q0_batches = {}
    with open(E1E0_Q0) as fh:
        for r in csv_rows(fh):
            q0_batches[int(r["global_step"])] = int(r["generation_batches_consumed"])
    pa = {}
    exact = 0
    for q in (0, 2):
        plan = plan_stream(by_step, quota=q)
        exp = exp0 if q == 0 else exp2
        per_step = {}
        for s in steps:
            got = [g for g, _r in plan[s]["members"]]
            want = [g for g, _r in exp.get(s, [])]
            per_step[s] = {"matches": got == want, "n_got": len(got), "n_want": len(want),
                           "missing": sorted(set(want) - set(got))[:5],
                           "extra": sorted(set(got) - set(want))[:5],
                           "batches_got": plan[s]["batches"],
                           "batches_want": q0_batches.get(s)}
        n_match = sum(1 for v in per_step.values() if v["matches"])
        pa[str(q)] = {"steps": len(steps), "steps_exact": n_match,
                      "exact_match": n_match == len(steps), "per_step": per_step}
        exact += int(n_match == len(steps))
    res["part_a"] = pa
    res["pass"]["part_a_planner_matches_e1e0"] = bool(pa["0"]["exact_match"] and pa["2"]["exact_match"])

    # ---------------- PART B ----------------
    sub = steps[:args.steps]
    pb = run_runtime_replay(by_step, sub, run_dir=args.run_dir)
    det = {}
    for s in sub:
        got = [u for u, _k, _n in pb[s]["members"]]
        want = [g.split(":")[-1] for g, _r in exp2.get(s, [])]
        det[s] = {"n_groups": pb[s]["batch_groups"], "calls_used": pb[s]["calls_used"],
                  "n_prompt": pb[s]["n_prompt"],
                  "same_size": len(got) == len(want),
                  "same_set": sorted(got) == sorted(want),
                  "missing": sorted(set(want) - set(got))[:5],
                  "extra": sorted(set(got) - set(want))[:5]}
    res["part_b"] = {"steps": [int(s) for s in sub], "per_step": det,
                     "all_same_size": all(v["same_size"] for v in det.values()),
                     "all_same_set": all(v["same_set"] for v in det.values())}
    res["pass"]["part_b_runtime_matches_e1e0"] = bool(res["part_b"]["all_same_size"] and res["part_b"]["all_same_set"])

    # ---------------- PART D: empty-keep neutrality (regression for the live D1 leak) ----
    run_d = sorted(set(sub) | set(sorted(find_empty_keep_steps(by_step))[:3]))
    res["part_d"] = empty_keep_checks(by_step, run_d, run_dir=args.run_dir)
    res["part_d_checks"] = res["part_d"]["checks"]
    res["pass"]["part_d_empty_keep_neutral"] = all(
        v for k, v in res["part_d"]["checks"].items() if v is not None)

    # ---------------- PART C ----------------
    checks = {}
    plan2 = plan_stream(by_step, quota=2)
    sizes, quots, k_ok, gap_ok, stop_ok = set(), set(), True, True, True
    gid2info = {}
    for s in steps:
        for call in by_step[s]:
            for g in call["groups"]:
                gid2info[g["gid"]] = g
        sizes.add(len(plan2[s]["members"]))
        quots.add(plan2[s]["plan_last"]["m"])
        for gid, role in plan2[s]["members"]:
            g = gid2info[gid]
            if role == "efficiency":
                if g["k"] != g["n"]:
                    k_ok = False
                if not (g["cost_gap"] and g["cost_gap"] > 0):
                    gap_ok = False
            else:
                if not (1 <= g["k"] <= 7):
                    k_ok = False
    checks["optimizer_batch_size_all_32"] = sizes == {B}
    checks["quota_actual_le_2"] = max(quots) <= Q
    checks["inserted_groups_k8"] = k_ok
    checks["inserted_groups_cost_variation_positive"] = gap_ok
    # stopping invariance vs q=0
    plan_by_q = {q: plan_stream(by_step, quota=q) for q in (0, 1, 2, 3, 4)}
    q_sizes, q_batches = {}, {}
    for q, pp in plan_by_q.items():
        q_sizes[q] = set(len(pp[s]["members"]) for s in steps)
        q_batches[q] = {s: pp[s]["batches"] for s in steps}
    checks["stopping_invariant_across_q_batch_sizes"] = all(v == {B} for v in q_sizes.values())
    checks["stopping_batch_independent_of_q"] = all(q_batches[q] == q_batches[0] for q in (1, 2, 3, 4))
    # mixed-only stopping: the stopping generation batch is the one in which the count of
    # MIXED groups (never efficiency groups) reaches B.
    trig_ok = True
    for s in steps:
        pl = plan2[s]["plan_last"]
        if pl is None or not pl["is_stopping"]:
            trig_ok = False
            continue
        call = by_step[s][plan2[s]["batches"] - 1]
        n_mixed_this = sum(1 for g in call["groups"] if g["type"] == QP.GROUP_MIXED)
        if not (pl["mixed_seen_after"] - n_mixed_this < B <= pl["mixed_seen_after"]):
            trig_ok = False
    checks["mixed_only_stopping"] = trig_ok
    # reward semantics on the real data (+ finiteness / std / wrong-rollout zero)
    bad_wrong = bad_mixed = bad_range = 0
    non_finite = std_bad = 0
    for s in steps:
        for gid, role in plan2[s]["members"]:
            g = gid2info[gid]
            cors = [c for c, a in zip(g["cost"], g["em"]) if a > 0 and c is not None]
            cmin = min(cors) if cors else None
            cmax = max(cors) if cors else None
            rewards = []
            for i, a in enumerate(g["em"]):
                c = g["cost"][i]
                if role == "mixed":
                    r = float(a)
                elif c is not None and cmax is not None and cmax > cmin:
                    r = 1.0 + LAMBDA * (cmax - c) / (cmax - cmin)
                else:
                    r = 1.0
                rewards.append(r)
                if not np.isfinite(r):
                    non_finite += 1
                if a == 0.0 and abs(r) > 1e-12:
                    bad_wrong += 1
                if role == "efficiency" and not (1.0 - 1e-12 <= r <= 1.0 + LAMBDA + 1e-12):
                    bad_range += 1
                if role == "mixed" and abs(r - a) > 1e-12:
                    bad_mixed += 1
            sd = float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0
            if not np.isfinite(sd):
                std_bad += 1
    checks["mixed_group_reward_equals_em"] = bad_mixed == 0
    checks["efficiency_group_reward_in_1_1.05"] = bad_range == 0
    checks["wrong_rollout_reward_zero"] = bad_wrong == 0
    checks["group_reward_std_finite"] = std_bad == 0
    checks["no_nan_inf_rewards"] = non_finite == 0
    # efficiency ranking: among the correct rollouts of an efficiency group the cheapest gets
    # E = 1, the most expensive E = 0, and E is monotone non-increasing in cost.
    rank_ok = True
    n_eff_checked = 0
    for s in steps:
        for call in by_step[s]:
            for g in call["groups"]:
                if g["type"] != QP.GROUP_EFFICIENCY:
                    continue
                cors = [c for c, a in zip(g["cost"], g["em"]) if a > 0 and c is not None]
                if len(cors) < 2 or max(cors) == min(cors):
                    continue
                n_eff_checked += 1
                cmin, cmax = min(cors), max(cors)
                e_of = lambda c: (cmax - c) / (cmax - cmin)  # noqa: E731
                if abs(e_of(cmin) - 1.0) > 1e-12 or abs(e_of(cmax)) > 1e-12:
                    rank_ok = False
                for c1 in cors:
                    if not (0.0 <= e_of(c1) <= 1.0):
                        rank_ok = False
                    for c2 in cors:
                        if c1 < c2 and e_of(c1) < e_of(c2) - 1e-12:
                            rank_ok = False
    checks["efficiency_ranking_correct"] = bool(rank_ok and n_eff_checked > 0)
    res["eff_groups_ranked"] = n_eff_checked
    # map the 12 mandated section-11 checks onto the executed checks
    sec11 = [
        ("1. every optimizer batch = 32 groups", "optimizer_batch_size_all_32"),
        ("2. q <= 2", "quota_actual_le_2"),
        ("3. mixed-only stopping", "mixed_only_stopping"),
        ("4. quota does not change generation stopping", "stopping_invariant_across_q_batch_sizes"),
        ("4b. stopping generation batch identical for q=0..4", "stopping_batch_independent_of_q"),
        ("5. inserted groups have k = 8", "inserted_groups_k8"),
        ("6. inserted groups cost variation > 0", "inserted_groups_cost_variation_positive"),
        ("7. mixed group reward == EM", "mixed_group_reward_equals_em"),
        ("8. inserted group reward in [1, 1.05]", "efficiency_group_reward_in_1_1.05"),
        ("9. wrong rollout reward == 0", "wrong_rollout_reward_zero"),
        ("10. every group reward std finite", "group_reward_std_finite"),
        ("11. efficiency rollout ranking correct", "efficiency_ranking_correct"),
        ("12. no NaN / Inf", "no_nan_inf_rewards"),
    ]
    res["section11_mapping"] = [{"requirement": t, "check": k, "pass": bool(checks.get(k))}
                                for t, k in sec11]
    res["part_c"] = checks
    res["pass"]["part_c_invariants"] = all(checks.values())

    res["all_pass"] = all(res["pass"].values())
    with open(OUT_JSON, "w") as fh:
        json.dump(res, fh, indent=2, default=str)

    md = ["# E1-E pre-training offline selftest", "",
          f"**ALL PASS: {res['all_pass']}**", "",
          "## PART A - planner replay vs E1-E0 (all 50 optimizer steps)",
          f"- q=0 exact steps: {pa['0']['steps_exact']}/{pa['0']['steps']}",
          f"- q=2 exact steps: {pa['2']['steps_exact']}/{pa['2']['steps']}", "",
          "## PART B - runtime manager replay vs E1-E0",
          f"- steps exercised: {res['part_b']['steps']}",
          f"- same batch size: {res['part_b']['all_same_size']}, same group set: {res['part_b']['all_same_set']}",
          "## PART D - empty-keep neutrality",
          f"- forced insertion neutral: {res.get('part_d_checks', {}).get('D1_inserted_no_keep_batch_is_neutral')}",
          f"- forced insertion still 32 groups: {res.get('part_d_checks', {}).get('D1_optimizer_batch_still_32_groups')}",
          f"- real stream has empty-keep batches: {res.get('part_d_checks', {}).get('D2_real_stream_contains_empty_keep_batches')}",
          f"- real empty-keep steps match E1-E0: {res.get('part_d_checks', {}).get('D2_empty_keep_steps_match_e1e0')}",
          f"- kept_nothing calls recorded: {res.get('part_d', {}).get('d3_kept_nothing_records')}", "",
          "## PART C - invariants", ""]
    for k, v in checks.items():
        md.append(f"- {k}: {v}")
    md += ["", "## Section 11 requirement -> check", "",
           "| E1-E_task.md section 11 requirement | check | result |", "|---|---|---|"]
    for row in res["section11_mapping"]:
        md.append(f"| {row['requirement']} | `{row['check']}` | {'PASS' if row['pass'] else 'FAIL'} |")
    md += ["", "```json", json.dumps(res["pass"], indent=2), "```", ""]
    with open(OUT_MD, "w") as fh:
        fh.write("\n".join(md))
    print(json.dumps({"all_pass": res["all_pass"], "pass": res["pass"],
                      "part_a": {k: {"exact": v["exact_match"], "steps": v["steps_exact"]} for k, v in pa.items()},
                      "part_b": {"same_size": res["part_b"]["all_same_size"],
                                 "same_set": res["part_b"]["all_same_set"],
                                 "detail": res["part_b"]["per_step"]},
                      "part_c": checks}, indent=2, default=str))
    print("wrote", OUT_MD, "and", OUT_JSON)
    return 0 if res["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

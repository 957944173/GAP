#!/usr/bin/env python3
"""E1-A CPU-only self-test for the shaped reward + accuracy separation.

Verifies (no GPU, no training):
  1. R_i = A_i * (1 + 0.05 * E_i) exactly, with
       E_i = (Cmax - C_i)/(Cmax - Cmin) among the group's correct rollouts when
       the group has >= 2 correct rollouts and their costs differ, else 0.
  2. wrong rollouts get reward exactly 0.
  3. groups with < 2 correct rollouts, or with no correct-cost variation, get
     E = 0 for every member (reward == EM).
  4. `data.batch["acc"]` is the ORIGINAL EM, while `token_level_scores`
     (the reward tensor) is the SHAPED reward.
  5. the validation manager (num_examine=1) behaves exactly like the original
     `BatchRewardManager` (returns bare EM, no diagnostics written).
  6. `compute_score_em_batch` (original) still exists and is unchanged.
  7. diagnostics JSONL (rollouts/groups/calls) is written with the required
     fields and the filter-retention counters match an independent recount.

Usage:  E1A_RUN_DIR=/tmp/e1a_selftest python3 e1a_selftest.py
"""
import json
import os
import shutil
import sys
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "verl"))

RUN_DIR = os.environ.get("E1A_RUN_DIR", "/tmp/e1a_selftest")


class FakeTokenizer:
    def __init__(self):
        self._map = {}

    def add(self, text):
        i = len(self._map) + 1
        self._map[i] = text
        return i

    def decode(self, ids, skip_special_tokens=True):
        out = []
        for t in ids:
            t = int(t)
            if t == 0 and skip_special_tokens:
                continue
            out.append(self._map.get(t, ""))
        return "".join(out)


PROMPT_LEN, RESP_LEN = 6, 16

# (uid, correct?, cost, queries)  -- one uid = one n=8 group (like the real run)
SPEC = [
    # uid "0": 8 correct, costs 1,1,2,2,2,2,2,2 -> active, E in {0,1}
    ("0", True, 1, 1), ("0", True, 1, 1), ("0", True, 2, 2), ("0", True, 2, 2),
    ("0", True, 2, 2), ("0", True, 2, 2), ("0", True, 2, 2), ("0", True, 2, 2),
    # uid "1": 8 correct, all cost 2 -> >=2 correct but NO variation -> E = 0
    ("1", True, 2, 2), ("1", True, 2, 2), ("1", True, 2, 2), ("1", True, 2, 2),
    ("1", True, 2, 2), ("1", True, 2, 2), ("1", True, 2, 2), ("1", True, 2, 2),
    # uid "2": 1 correct (cost 1) + 7 wrong (cost 2/3) -> <2 correct -> E = 0
    ("2", True, 1, 1), ("2", False, 2, 2), ("2", False, 2, 2), ("2", False, 3, 2),
    ("2", False, 3, 2), ("2", False, 2, 1), ("2", False, 2, 1), ("2", False, 3, 3),
    # uid "3": 8 wrong -> reward all 0
    ("3", False, 1, 1), ("3", False, 2, 2), ("3", False, 2, 2), ("3", False, 2, 2),
    ("3", False, 1, 1), ("3", False, 2, 2), ("3", False, 2, 2), ("3", False, 2, 2),
]
GT = {"0": ["paris"], "1": ["berlin"], "2": ["rome"], "3": ["oslo"]}


def build_data():
    from verl import DataProto

    tok = FakeTokenizer()
    prompts, responses, masks = [], [], []
    uids, extras, dss, creasons, dtms, rms = [], [], [], [], [], []
    for uid, correct, cost, queries in SPEC:
        ans = GT[uid][0] if correct else "wrong"
        text = "<answer>%s</answer>" % ans
        prompts.append([tok.add("PROMPT")] + [0] * (PROMPT_LEN - 1))
        responses.append([tok.add(text)] + [0] * (RESP_LEN - 1))
        masks.append([1] * PROMPT_LEN + [1] * RESP_LEN)
        uids.append(uid)
        extras.append({"index": int(uid), "question": "q-%s" % uid})
        dss.append("nq")
        creasons.append("stop")
        # tool_call_sequence: `queries` calls spread over `cost` blocks/turns
        seq = []
        for b in range(cost):
            j = 0
            # distribute queries over blocks: block0 gets the remainder
            seq.append({"tool_name": "wiki_search", "call_id": f"call_wiki_search_{b}_{j}",
                        "turn": 2 + b, "arguments": json.dumps({"query": f"q{b}"})})
        for extra_q in range(queries - cost):
            seq.append({"tool_name": "wiki_search", "call_id": f"call_wiki_search_0_{extra_q + 1}",
                        "turn": 2, "arguments": json.dumps({"query": f"q{extra_q}"})})
        dtms.append({
            "total_tool_calls": len(seq),
            "tool_calls_by_type": {"wiki_search": len(seq)},
            "conversation_turns": cost + 1,
            "tool_call_sequence": seq,
            "unique_tools_used": ["wiki_search"],
            "unique_tools_count": 1,
        })
        rms.append({"ground_truth": {"target": np.array(GT[uid], dtype=object)}})

    tensors = {
        "prompts": torch.tensor(prompts, dtype=torch.long),
        "responses": torch.tensor(responses, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
    }
    non_tensors = {
        "uid": np.array(uids, dtype=object),
        "extra_info": np.array(extras, dtype=object),
        "data_source": np.array(dss, dtype=object),
        "complete_reason": np.array(creasons, dtype=object),
        "detailed_tool_metrics": np.array(dtms, dtype=object),
        "reward_model": np.array(rms, dtype=object),
    }
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors), tok


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    if os.path.isdir(RUN_DIR):
        shutil.rmtree(RUN_DIR)
    os.makedirs(RUN_DIR, exist_ok=True)
    os.environ["E1A_RUN_DIR"] = RUN_DIR

    import e1a_cost
    import e1a_scorer
    import e1a_state
    from e1a_reward_manager import E1ABatchRewardManager

    failures = []
    data, tok = build_data()
    n = len(SPEC)

    # ---- 6. original batch scorer still available ----
    if not callable(getattr(e1a_scorer, "compute_score_em_batch", None)):
        failures.append("compute_score_em_batch missing from e1a_scorer")

    # ---- 1/2/3. scorer-level math on the raw group spec ----
    cost_info = e1a_cost.costs_from_batch(data.non_tensor_batch, n)
    ds = list(data.non_tensor_batch["data_source"])
    extras = list(data.non_tensor_batch["extra_info"])
    prompts_txt = [tok.decode(data.batch["prompts"][i], True) for i in range(n)]
    resp_txt = [tok.decode(data.batch["responses"][i], True) for i in range(n)]
    gts = [data[i].non_tensor_batch["reward_model"].get("ground_truth") for i in range(n)]
    uids = list(data.non_tensor_batch["uid"])
    scored = e1a_scorer.compute_score_em_efficiency_batch(
        ds, prompts_txt, resp_txt, gts, extras,
        uids=uids, costs=cost_info["costs"], queries=cost_info["queries"],
        conversation_turns=cost_info["conversation_turns"],
    )
    for i, (uid, correct, cost, _q) in enumerate(SPEC):
        r = scored[i]
        a = 1.0 if correct else 0.0
        if r["em"] != a:
            failures.append(f"row {i}: em {r['em']} != {a}")
        if cost_info["costs"][i] != cost:
            failures.append(f"row {i}: extracted cost {cost_info['costs'][i]} != {cost}")
        want_e = 0.0
        if uid == "0" and correct:
            want_e = 1.0 if cost == 1 else 0.0
        if abs(r["efficiency"] - want_e) > 1e-12:
            failures.append(f"row {i} (uid {uid}): E {r['efficiency']} != {want_e}")
        want_r = a * (1.0 + 0.05 * want_e)
        if a == 0.0:
            want_r = 0.0
        if abs(r["score"] - want_r) > 1e-12:
            failures.append(f"row {i}: R {r['score']} != {want_r}")
        if a == 0.0 and r["score"] != 0.0:
            failures.append(f"row {i}: wrong rollout reward != 0")

    # ---- 4. manager: shaped reward tensor, original EM in acc ----
    tm = E1ABatchRewardManager(tok, num_examine=0, compute_score=e1a_scorer.compute_score_em_efficiency_batch)
    e1a_state.TRAINER = types.SimpleNamespace(global_steps=61)
    out = tm(data, return_dict=True)
    rt, em_extra = out["reward_tensor"], out["reward_extra_info"]["em"]
    acc = data.batch["acc"]
    for i in range(n):
        a = 1.0 if SPEC[i][1] else 0.0
        want_r = 0.0 if a == 0 else (1.05 if (SPEC[i][0] == "0" and SPEC[i][2] == 1) else 1.0)
        if abs(float(rt[i].sum()) - want_r) > 1e-6:
            failures.append(f"manager row {i}: reward_tensor {float(rt[i].sum())} != {want_r}")
        if abs(float(acc[i]) - a) > 1e-6:
            failures.append(f"manager row {i}: acc {float(acc[i])} != original EM {a}")
        if abs(float(em_extra[i]) - a) > 1e-6:
            failures.append(f"manager row {i}: reward_extra_info em {em_extra[i]} != {a}")

    # ---- 7. diagnostics ----
    pid = os.getpid()
    diag = os.path.join(RUN_DIR, "diagnostics")
    roll = read_jsonl(os.path.join(diag, f"rollouts_pid{pid}.jsonl"))
    grp = read_jsonl(os.path.join(diag, f"groups_pid{pid}.jsonl"))
    cal = read_jsonl(os.path.join(diag, f"calls_pid{pid}.jsonl"))
    if len(roll) != n:
        failures.append(f"diagnostics rollouts {len(roll)} != {n}")
    if len(grp) != 4:
        failures.append(f"diagnostics groups {len(grp)} != 4")
    if len(cal) != 1:
        failures.append(f"diagnostics calls {len(cal)} != 1")
    g_by_id = {g["group_id"]: g for g in grp}
    # uid 0: baseline keep (all correct -> std(EM)=0 -> drop), shaped keep (variance -> keep)
    if not (g_by_id["0"]["baseline_keep_em_std_gt_0"] is False
            and g_by_id["0"]["shaped_keep_reward_std_gt_0"] is True
            and g_by_id["0"]["newly_added_by_shaping"] is True):
        failures.append("uid 0 retention should be baseline=False -> shaped=True (newly added)")
    # uid 1: all correct, no cost variation -> no efficiency -> still dropped
    if g_by_id["1"]["shaped_keep_reward_std_gt_0"] is not False:
        failures.append("uid 1 (no variation) should remain dropped")
    # uid 2: 1 correct + wrongs -> baseline keep True, shaped keep True, not newly added
    if not (g_by_id["2"]["baseline_keep_em_std_gt_0"] is True
            and g_by_id["2"]["newly_added_by_shaping"] is False):
        failures.append("uid 2 should be kept by both")
    # uid 3: all wrong -> dropped by both
    if g_by_id["3"]["shaped_keep_reward_std_gt_0"] is not False:
        failures.append("uid 3 (all wrong) should be dropped")
    if cal[0]["n_newly_added"] != 1 or cal[0]["n_baseline_keep"] != 1 or cal[0]["n_shaped_keep"] != 2:
        failures.append(f"call retention counters wrong: {cal[0]}")

    # ---- 5. validation manager = original behaviour, no diagnostics ----
    before = len(read_jsonl(os.path.join(diag, f"rollouts_pid{pid}.jsonl")))
    data2, tok2 = build_data()
    vm = E1ABatchRewardManager(tok2, num_examine=1, compute_score=e1a_scorer.compute_score_em_batch)
    vout = vm(data2, return_dict=True)
    vrt = vout["reward_tensor"]
    for i in range(n):
        a = 1.0 if SPEC[i][1] else 0.0
        if abs(float(vrt[i].sum()) - a) > 1e-6:
            failures.append(f"val row {i}: EM reward {float(vrt[i].sum())} != {a}")
    after = len(read_jsonl(os.path.join(diag, f"rollouts_pid{pid}.jsonl")))
    if after != before:
        failures.append("validation manager wrote diagnostics (should not)")

    print(json.dumps({
        "scored_sample": scored[0], "call_record": cal[0],
        "group_uids_0_2": [g_by_id["0"], g_by_id["2"]],
        "reward_tensor": [round(float(rt[i].sum()), 6) for i in range(n)],
        "acc": [round(float(acc[i]), 6) for i in range(n)],
    }, indent=2, default=str))

    if failures:
        print("\nSELFTEST FAILURES:")
        for f in failures:
            print("  -", f)
        return 1
    print("\nE1-A SELFTEST PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

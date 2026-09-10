"""E1-0 audit self-test (CPU-only, no GPU, no training).

Exercises the rewritten E10BatchAuditRewardManager on a synthetic DataProto
that mimics the real runtime schema, and asserts the properties E1-0 depends on:

  * reward_tensor == original EM (score == em, no shaped reward)
  * `acc` == em
  * TRAIN manager writes audit records; VAL manager (num_examine=1) writes none
  * global_step comes from the live trainer, not a call counter
  * logical_search_batches is recovered from the parser block index in
    tool_call_sequence[].call_id, NOT from a string that does not exist
  * the response-text cross-check counts the rendered <tool_call> blocks
  * per-rollout reward identity vs the ORIGINAL compute_score_em_batch is exact

Run:
    conda activate parallel-agent
    cd <repo>
    E10_RUN_DIR=/tmp/e10_selftest \
    PYTHONPATH=<repo>/verl:<code_dir> python3 e10_selftest.py
"""
import json
import os
import shutil
import sys
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "verl"))

RUN_DIR = os.environ.get("E10_RUN_DIR", "/tmp/e10_selftest")


class FakeTokenizer:
    """Maps single token ids to text blocks; decode(ids) concatenates them."""

    def __init__(self):
        self._map = {}

    def add(self, text):
        idx = len(self._map) + 1
        self._map[idx] = text
        return idx

    def decode(self, ids, skip_special_tokens=True):
        out = []
        for t in ids:
            t = int(t)
            if t == 0 and skip_special_tokens:
                continue
            out.append(self._map.get(t, ""))
        return "".join(out)


def tool_call_block(query):
    payload = json.dumps({"query": query})
    return "<tool_call>\n" + json.dumps({"name": "wiki_search", "arguments": payload}) + "\n</tool_call>\n"


PROMPT_LEN, RESP_LEN = 8, 32

# (uid, correct, [(turn, block_i, query_j), ...])
SPEC = [
    ("0", True, [(2, 0, 0)]),
    ("0", True, [(2, 0, 0), (4, 0, 0), (4, 0, 1)]),   # 2 blocks / 3 queries
    ("0", False, [(2, 0, 0)]),
    ("0", False, []),
    ("1", False, [(2, 0, 0), (2, 1, 0)]),             # 2 blocks in one turn
    ("1", False, [(2, 0, 0)]),
    ("1", False, []),
    ("1", False, [(2, 0, 0)]),
]
GT = {"0": ["paris"], "1": ["tokyo"]}


def build_data():
    from verl import DataProto

    tok = FakeTokenizer()
    prompts, responses, masks = [], [], []
    uids, extras, dss, creasons, freasons, msgs, dtms, rms = [], [], [], [], [], [], [], []

    for uid, correct, calls in SPEC:
        text = ""
        if calls:
            text += "let me search\n"
            for (_t, i, j) in calls:
                text += tool_call_block(f"q{i}_{j}")
        text += f"<answer>{GT[uid][0] if correct else 'wrong'}</answer>"
        prompts.append([tok.add("PROMPT")] + [0] * (PROMPT_LEN - 1))
        responses.append([tok.add(text)] + [0] * (RESP_LEN - 1))
        masks.append([1] * PROMPT_LEN + [1] * RESP_LEN)
        uids.append(uid)
        extras.append({"index": int(uid), "question": f"question-{uid}"})
        dss.append("nq")
        creasons.append("stop")
        freasons.append({"type": "stop", "matched": None})
        msgs.append({"messages": [{"role": "assistant", "content": "let me search"}]})
        seq = [
            {
                "tool_name": "wiki_search",
                "call_id": f"call_wiki_search_{i}_{j}",
                "turn": t,
                "arguments": json.dumps({"query": f"q{i}_{j}"}),
            }
            for (t, i, j) in calls
        ]
        dtms.append(
            {
                "total_tool_calls": len(seq),
                "tool_calls_by_type": {"wiki_search": len(seq)} if seq else {},
                "conversation_turns": len({t for (t, _, _) in calls}),
                "tool_call_sequence": seq,
                "unique_tools_used": ["wiki_search"] if seq else [],
                "unique_tools_count": 1 if seq else 0,
            }
        )
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
        "finish_reason": np.array(freasons, dtype=object),
        "messages": np.array(msgs, dtype=object),
        "detailed_tool_metrics": np.array(dtms, dtype=object),
        "reward_model": np.array(rms, dtype=object),
    }
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors), tok


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    if os.path.isdir(RUN_DIR):
        shutil.rmtree(RUN_DIR)
    os.makedirs(RUN_DIR, exist_ok=True)
    os.environ["E10_RUN_DIR"] = RUN_DIR

    import e10_state
    from e10_reward_manager import E10BatchAuditRewardManager
    from e10_scorer import compute_score_em_batch_e10

    data, tok = build_data()
    n = len(SPEC)
    failures = []

    # ---------------- training manager (audits) ----------------
    tm = E10BatchAuditRewardManager(tok, num_examine=0, compute_score=compute_score_em_batch_e10)
    e10_state.TRAINER = types.SimpleNamespace(global_steps=61)  # fake live trainer
    out = tm(data, return_dict=True)
    rt = out["reward_tensor"]
    em = out["reward_extra_info"]["em"]
    score = out["reward_extra_info"]["score"]
    acc = data.batch["acc"]

    for i, (uid, correct, _calls) in enumerate(SPEC):
        want = 1.0 if correct else 0.0
        got = float(rt[i].sum())
        if got != want:
            failures.append(f"row {i}: reward_tensor {got} != {want}")
        if float(score[i]) != want or float(em[i]) != want:
            failures.append(f"row {i}: score/em {float(score[i])}/{float(em[i])} != {want}")
        if float(acc[i]) != want:
            failures.append(f"row {i}: acc {float(acc[i])} != {want}")

    pid = os.getpid()
    rollouts = read_jsonl(os.path.join(RUN_DIR, "audit", f"rollouts_pid{pid}.jsonl"))
    groups = read_jsonl(os.path.join(RUN_DIR, "audit", f"groups_pid{pid}.jsonl"))
    calls = read_jsonl(os.path.join(RUN_DIR, "audit", f"calls_pid{pid}.jsonl"))

    if len(rollouts) != n:
        failures.append(f"expected {n} rollout records, got {len(rollouts)}")
    if len(groups) != 2:
        failures.append(f"expected 2 group records, got {len(groups)}")
    if len(calls) != 1:
        failures.append(f"expected 1 call record, got {len(calls)}")

    for k, r in enumerate(rollouts):
        exp_logical = len({(t, i) for (t, i, _j) in SPEC[k][2]})
        exp_calls = len(SPEC[k][2])
        if r["global_step"] != 61 or r["global_step_source"] != "trainer.global_steps":
            failures.append(f"row {k}: bad step attribution {r['global_step']}/{r['global_step_source']}")
        if r["logical_search_batches"] != exp_logical:
            failures.append(f"row {k}: logical_search_batches {r['logical_search_batches']} != {exp_logical}")
        if r["structured_search_calls"] != exp_calls:
            failures.append(f"row {k}: structured_search_calls {r['structured_search_calls']} != {exp_calls}")
        if r["resp_wiki_tool_call_blocks"] != exp_calls:
            failures.append(f"row {k}: resp_wiki_tool_call_blocks {r['resp_wiki_tool_call_blocks']} != {exp_calls}")
        if r["reward_identity_abs_diff"] != 0:
            failures.append(f"row {k}: reward_identity_abs_diff {r['reward_identity_abs_diff']} != 0")
        if r["reward_original"] != r["reward_e10"]:
            failures.append(f"row {k}: reward_original {r['reward_original']} != reward_e10 {r['reward_e10']}")
        if not r["answer_parse_ok"]:
            failures.append(f"row {k}: answer_parse_ok False")
        if not r.get("response_excerpt"):
            failures.append(f"row {k}: response_excerpt empty")
        if not isinstance(r.get("tool_call_summary"), list):
            failures.append(f"row {k}: tool_call_summary not a list")
        if r["rollout_pos_in_group"] != (k % 4):
            failures.append(f"row {k}: rollout_pos_in_group {r['rollout_pos_in_group']} != {k % 4}")

    if calls and calls[0]["identity_mismatch"] != 0:
        failures.append(f"call identity_mismatch {calls[0]['identity_mismatch']} != 0")
    if calls and calls[0]["global_step"] != 61:
        failures.append(f"call global_step {calls[0]['global_step']} != 61")

    # ---------------- validation manager must NOT audit ----------------
    before = len(rollouts)
    vm = E10BatchAuditRewardManager(tok, num_examine=1, compute_score=compute_score_em_batch_e10)
    if vm.is_train:
        failures.append("val manager marked is_train=True")
    vm(data, return_dict=True)
    after = len(read_jsonl(os.path.join(RUN_DIR, "audit", f"rollouts_pid{pid}.jsonl")))
    if after != before:
        failures.append(f"val manager wrote audit records ({before} -> {after})")

    # ---------------- group content sanity ----------------
    g_by_id = {g["group_id"]: g for g in groups}
    if g_by_id.get("0", {}).get("correct_count") != 2:
        failures.append("uid 0 correct_count != 2")
    if g_by_id.get("1", {}).get("correct_count") != 0:
        failures.append("uid 1 correct_count != 0")
    if not g_by_id.get("0", {}).get("xml_variation_correct"):
        failures.append("uid 0 correct rollouts should show logical-batch variation (1 vs 2)")
    if g_by_id.get("1", {}).get("all_wrong") is not True:
        failures.append("uid 1 should be all_wrong")

    print("reward_tensor:", [float(rt[i].sum()) for i in range(n)])
    print("em           :", [float(e) for e in em])
    print("calls record :", json.dumps(calls, default=str))
    print("\nsample rollout:", json.dumps(rollouts[1], indent=2, default=str))
    print("\ngroups:", json.dumps(groups, indent=2, default=str))

    if failures:
        print("\nSELFTEST FAILURES:")
        for f in failures:
            print("  -", f)
        return 1
    print("\nSELFTEST PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

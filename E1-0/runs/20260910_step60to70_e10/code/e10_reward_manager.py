"""E1-0 passive-audit reward manager (rewritten 2026-09-10 after code review).

Subclasses the ORIGINAL BatchRewardManager
(verl/verl/workers/reward_manager/batch.py) and reuses essentially all of its
logic verbatim (verify()/__call__ structure, reward_tensor construction, acc
tensor).  The only addition is a passive audit hook: after computing scores
exactly as the base class would, this class also writes per-rollout, per-group
and per-call audit JSONL records to RUN_DIR/audit/, using metadata that is
ALREADY present on the DataProto at this point in the pipeline (uid,
detailed_tool_metrics, tool_call_sequence, messages, tool_call_info,
complete_reason, finish_reason, extra_info).  ray_trainer.py:1749 calls
compute_reward strictly BEFORE the filter_groups slice at line 1832, so the
hook observes the complete, unfiltered n=8-per-uid batch.

Does NOT alter: reward_tensor values, data.batch["acc"], control flow, or any
training-visible state.  If audit logging throws, it is caught and swallowed
(logged to stderr) so it can never affect reward_tensor/acc, matching task
section 16 ("audit logger error must never alter training math") -- this is
belt-and-suspenders on top of reward.py's own compute_reward() try/except.

=====================================================================
CODE-REVIEW CORRECTIONS (2026-09-10) vs the first E1-0 draft
=====================================================================
(1) VALIDATION CONTAMINATION.  `load_reward_manager(config, ..., is_valid=True)`
    instantiates the SAME class named by `reward_model.reward_manager`, so
    `val_reward_fn` is also an E10BatchAuditRewardManager.  `_validate()` calls
    it once per val batch (29 batches here), which (a) wrote NQ validation
    rollouts into the training audit JSONL and (b) advanced the shared
    call_index, corrupting every subsequent training record's step estimate.
    Fix: the audit hook is enabled ONLY for the training instance.  main_ppo.py
    constructs the train manager with num_examine=0 and the val manager with
    num_examine=1 (main_ppo.py:161-162); that is the discriminator used here.
    Val calls still return the exact base-class result -- only auditing is
    skipped.

(2) WRONG global_step.  filter_groups is enabled, so one global step consumes
    ~8-14 generation batches (verified: original run's supervisor.log line for
    step 60 reads `num_gen_batches=11`).  A `start + 1 + call_index` formula is
    wrong by roughly an order of magnitude.  Fix: read the live
    `RayPPOTrainer.global_steps` via e10_trainer_patch/e10_state (driver-only
    read-only reference), and record `global_step_source`.

(3) XML BATCH COST WAS ALWAYS ZERO.  The task asks for the number of non-empty
    `<wiki_search>...</wiki_search>` blocks in the response.  In THIS repo that
    string does not exist in the response at all: `add_assistant_message`
    (schemas.py:171-181) stores the tool call in the structured `tool_calls`
    field and re-renders the whole assistant turn through the Qwen chat
    template, which emits `<tool_call>{"name": "wiki_search", ...}</tool_call>`.
    The first draft counted the regex over assistant `messages[].content`,
    which is `normed_content` -- the text with the tool call already stripped.
    -> logical_search_batches was 0 for every rollout, making the whole
    XML-vs-structured analysis vacuous.  Fix:
      * `logical_search_batches` is reconstructed faithfully from the parser
        block index carried in each structured call id
        (`XMLToolParser.parse_non_stream` -> `call_<tool>_<block_i>_<query_j>`
        -> OpenAIFunctionToolCall.id -> detailed_tool_metrics.tool_call_sequence
        [].call_id), i.e. one logical block per distinct (assistant turn, i).
        This is exactly the `<wiki_search>` block count the task wants, just
        recovered from the structured record instead of a string that the repo
        never writes.
      * an INDEPENDENT response-text count is recorded alongside it
        (`resp_wiki_tool_call_blocks` from the rendered `<tool_call>` blocks,
        `resp_wiki_search_tag_count` for the literal XML tag) so the mapping
        text<->structure is empirically verified in the report instead of
        assumed.

(4) REWARD IDENTITY was vacuous.  Task section 0 requires per-rollout
    verification that the E1-0 reward equals the ORIGINAL GAP reward.  The
    first draft recorded the E1-0 scorer's own `score` and `em` -- trivially
    equal by construction.  Fix: additionally re-run the ORIGINAL
    `compute_score_em_batch` (imported unmodified from
    verl/verl/utils/reward_score/mhqa_train.py) on the same decoded
    prompt/response/ground-truth, and compare it against the value that
    actually reached GRPO (`reward_tensor[i].sum()`), recording
    `reward_original`, `reward_e10`, `abs_diff`, `reward_identity_ok`.
"""
import importlib.util
import json
import os
import re
import sys
import threading
from datetime import datetime

from verl.workers.reward_manager.batch import BatchRewardManager  # noqa: E402
from verl.workers.reward_manager import register  # noqa: E402

import e10_state  # noqa: E402
import e10_trainer_patch  # noqa: E402

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "..", ".."))


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Original GAP scorer, imported unmodified: used ONLY for the independent
# reward-identity cross-check (never for the reward that trains).
_ORIGINAL_MHQA_TRAIN = _load_module(
    "e10_original_mhqa_train_for_identity",
    os.path.join(_REPO_ROOT, "verl", "verl", "utils", "reward_score", "mhqa_train.py"),
)
compute_score_em_batch_original = _ORIGINAL_MHQA_TRAIN.compute_score_em_batch
extract_solution_original = _ORIGINAL_MHQA_TRAIN.extract_solution

# Real XML tool parser (for the response-side cross-check of the '|'
# parallel-query split rule and the rendered <tool_call> blocks).
_XML_PARSER_MOD = _load_module(
    "e10_xml_tool_parser",
    os.path.join(_REPO_ROOT, "verl", "verl", "tools", "xml_tool_parser.py"),
)
_XML_PARSER = _XML_PARSER_MOD.XMLToolParser(tools=[{"function": {"name": "wiki_search"}}])
_WIKI_SEARCH_REGEX = _XML_PARSER.tool_regexes["wiki_search"]
# Qwen chat template rendering of an assistant tool call, e.g.
#   <tool_call>
#   {"name": "wiki_search", "arguments": "{\"query\": \"...\"}"}
#   </tool_call>
_TOOL_CALL_RENDER_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
# call_<tool_name>_<block_index>_<query_index> (search) or call_<tool_name>_<i>.
# The tool name group MUST be non-greedy: a greedy `(.+)` swallows the block
# index for two-digit-style ids such as call_wiki_search_0_1 (it would parse
# block=1,query=None instead of block=0,query=1) -- caught by e10_selftest.
_CALL_ID_RE = re.compile(r"^call_(.+?)_(\d+)(?:_(\d+))?$")

E10_RUN_DIR = os.environ.get("E10_RUN_DIR")
E10_START_GLOBAL_STEP = int(os.environ.get("E10_START_GLOBAL_STEP", "60"))


def _count_logical_search_batches_and_queries_from_text(text):
    """Independent, response-text based count of non-empty <wiki_search> blocks
    and total queries, using the REAL XMLToolParser regex and its
    `_parse_search_queries()` ('|'-split rule).  Kept for cross-checking (see
    correction (3)); in this repo the rendered response normally contains
    `<tool_call>` rather than `<wiki_search>`, so this count is expected to be
    0 and `resp_wiki_tool_call_blocks` carries the text-side signal."""
    batches = 0
    total_queries = 0
    for m in _WIKI_SEARCH_REGEX.finditer(text or ""):
        content = m.group(1).strip()
        if not content:
            continue
        queries = _XML_PARSER._parse_search_queries(content)
        if not queries:
            continue
        batches += 1
        total_queries += len(queries)
    return batches, total_queries


def _count_rendered_wiki_tool_calls(text):
    """Count rendered <tool_call> blocks whose function name is wiki_search.

    Each ParsedToolCall (one per parallel query) is rendered as its own
    <tool_call> block, so this equals the number of executed wiki_search
    queries as seen in the response text."""
    total = 0
    wiki = 0
    for m in _TOOL_CALL_RENDER_RE.finditer(text or ""):
        total += 1
        try:
            obj = json.loads(m.group(1))
        except Exception:  # noqa: BLE001
            continue
        name = obj.get("name") if isinstance(obj, dict) else None
        if name == "wiki_search":
            wiki += 1
    return total, wiki


def _response_excerpt(text, limit=400):
    """Bounded, newline-flattened excerpt kept in the audit record so task
    section 12's "typical mismatch cases with a response excerpt" can be
    produced WITHOUT storing full 8k-token trajectories (~180k of them).
    Prefers the window around the first rendered tool call / answer tag."""
    if not text:
        return ""
    flat = text.replace("\n", "\\n")
    if len(flat) <= limit:
        return flat
    idx = -1
    for tag in ("<tool_call>", "<answer>"):
        j = flat.find(tag)
        if j != -1 and (idx == -1 or j < idx):
            idx = j
    if idx == -1:
        return flat[:limit]
    start = max(0, idx - 120)
    return flat[start:start + limit]


def _tool_call_summary(seq, max_items=16):
    """Compact 'tool@turn#block.query' summary of the structured call sequence."""
    out = []
    for c in seq[:max_items]:
        if not isinstance(c, dict):
            continue
        m = _CALL_ID_RE.match(str(c.get("call_id", "")))
        block = m.group(2) if m else "?"
        query = m.group(3) if (m and m.group(3) is not None) else "-"
        out.append(f"{c.get('tool_name')}@turn{c.get('turn')}#b{block}q{query}")
    if len(seq) > max_items:
        out.append(f"...(+{len(seq) - max_items})")
    return out


def _stats(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0, "min": None, "max": None, "mean": None, "var": None}
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    return {"n": n, "min": min(vals), "max": max(vals), "mean": mean, "var": var}


@register("e10_batch_audit")
class E10BatchAuditRewardManager(BatchRewardManager):
    """Passive-audit variant of BatchRewardManager. reward_manager name: 'e10_batch_audit'."""

    _lock = threading.Lock()
    _call_index = {"train": 0}

    def __init__(self, tokenizer, num_examine, compute_score, reward_fn_key="data_source", **reward_kwargs):
        super().__init__(tokenizer, num_examine, compute_score, reward_fn_key=reward_fn_key, **reward_kwargs)
        # main_ppo.py constructs the TRAIN manager with num_examine=0 and the
        # VAL manager with num_examine=1 -> only the former audits (correction 1).
        self.is_train = num_examine == 0
        self.run_dir = E10_RUN_DIR
        self.pid = os.getpid()
        self._last_step = None
        self._call_within_step = 0
        if self.is_train and self.run_dir:
            e10_trainer_patch.ensure_driver_patch()
            audit_dir = os.path.join(self.run_dir, "audit")
            os.makedirs(audit_dir, exist_ok=True)
            self.rollouts_path = os.path.join(audit_dir, f"rollouts_pid{self.pid}.jsonl")
            self.groups_path = os.path.join(audit_dir, f"groups_pid{self.pid}.jsonl")
            self.calls_path = os.path.join(audit_dir, f"calls_pid{self.pid}.jsonl")
            self.errors_path = os.path.join(audit_dir, f"audit_errors_pid{self.pid}.log")

    def _next_call_index(self):
        with E10BatchAuditRewardManager._lock:
            idx = E10BatchAuditRewardManager._call_index["train"]
            E10BatchAuditRewardManager._call_index["train"] += 1
            return idx

    def __call__(self, data, return_dict=False):
        result = super().__call__(data, return_dict=True)

        if self.is_train:
            try:
                self._audit(data, result)
            except Exception as e:  # noqa: BLE001 -- audit must never affect training
                try:
                    if self.run_dir:
                        with open(self.errors_path, "a") as f:
                            f.write(f"AUDIT ERROR (swallowed, training unaffected): {type(e).__name__}: {e}\n")
                except Exception:
                    pass

        if return_dict:
            return result
        else:
            return result["reward_tensor"]

    # ------------------------------------------------------------------ #
    # passive metadata helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _unwrap_messages(raw_messages_entry):
        """sglang_rollout.py appends {"messages": req.messages} (one extra
        nesting level) into non_tensor_batch['messages'] -- unwrap that here."""
        if raw_messages_entry is None:
            return None, False
        if isinstance(raw_messages_entry, dict) and "messages" in raw_messages_entry:
            return raw_messages_entry["messages"], True
        if isinstance(raw_messages_entry, list):
            return raw_messages_entry, True
        return None, False

    def _structured_metrics(self, dtm):
        """dtm = one entry of non_tensor_batch['detailed_tool_metrics'].

        Real schema (schemas.py:201-247 + sglang_rollout.py:1189):
            {total_tool_calls, tool_calls_by_type, conversation_turns,
             tool_call_sequence: [{tool_name, call_id, turn, arguments}, ...],
             unique_tools_used, unique_tools_count}
        `turn` is len(req.messages) at tracking time, i.e. one distinct value
        per assistant turn that issued tool calls.  `call_id` is the parser's
        tool_index, `call_<tool>_<block_i>_<query_j>`, so distinct (turn, i)
        pairs reconstruct the exact number of <wiki_search> blocks.
        """
        if not isinstance(dtm, dict) or not dtm:
            return {
                "structured_search_rounds": None,
                "structured_search_calls": None,
                "structured_logical_batches": None,
                "structured_total_tool_calls": None,
                "structured_conversation_turns": None,
                "structured_nonwiki_tool_calls": None,
                "metadata_valid": False,
                "fallback_degraded": True,
                "block_index_available": False,
                "structured_sequence_summary": [],
            }

        seq = dtm.get("tool_call_sequence", [])
        if not isinstance(seq, list):
            seq = []
        wiki_calls = [c for c in seq if isinstance(c, dict) and c.get("tool_name") == "wiki_search"]
        nonwiki_calls = [c for c in seq if isinstance(c, dict) and c.get("tool_name") != "wiki_search"]

        turns = set()
        blocks = set()
        block_index_available = False
        for c in wiki_calls:
            turn = c.get("turn")
            if turn is not None:
                turns.add(turn)
            m = _CALL_ID_RE.match(str(c.get("call_id", "")))
            if m:
                block_index_available = True
                blocks.add((turn, int(m.group(2))))

        seq_empty_but_calls = (len(seq) == 0) and bool(dtm.get("total_tool_calls"))
        return {
            "structured_search_rounds": len(turns),
            "structured_search_calls": len(wiki_calls),
            "structured_logical_batches": len(blocks) if blocks else (len(turns) if not block_index_available else 0),
            "structured_total_tool_calls": dtm.get("total_tool_calls"),
            "structured_conversation_turns": dtm.get("conversation_turns"),
            "structured_nonwiki_tool_calls": len(nonwiki_calls),
            "metadata_valid": True,
            # "fallback" = the degraded dict sglang_rollout emits when
            # req.detailed_tool_metrics was falsy.  It always carries an empty
            # tool_call_sequence, so it is only detectable as
            # "sequence empty although counts claim calls happened".
            "fallback_degraded": seq_empty_but_calls,
            "block_index_available": block_index_available,
            "structured_sequence_summary": _tool_call_summary(seq),
        }

    # ------------------------------------------------------------------ #
    # audit
    # ------------------------------------------------------------------ #
    def _audit(self, data, result):
        if not self.run_dir:
            return

        reward_extra_info = result.get("reward_extra_info", {}) or {}
        reward_tensor = result.get("reward_tensor")
        n = len(data)

        call_index = self._next_call_index()
        global_step = e10_state.current_global_step()
        global_step_source = "trainer.global_steps" if global_step is not None else "unavailable"
        if global_step is not None and global_step != self._last_step:
            self._last_step = global_step
            self._call_within_step = 0
        else:
            self._call_within_step += 1
        call_within_step = self._call_within_step
        timestamp = datetime.now().isoformat(timespec="seconds")

        uids = data.non_tensor_batch.get("uid", [None] * n)
        extra_infos = data.non_tensor_batch.get("extra_info", [None] * n)
        data_sources = data.non_tensor_batch.get(self.reward_fn_key, [None] * n)
        complete_reasons = data.non_tensor_batch.get("complete_reason", [None] * n)
        finish_reasons = data.non_tensor_batch.get("finish_reason", [None] * n)
        messages_raw = data.non_tensor_batch.get("messages", [None] * n)
        detailed_tool_metrics = data.non_tensor_batch.get("detailed_tool_metrics", [None] * n)

        response_ids = data.batch["responses"]
        prompt_ids = data.batch["prompts"]
        prompt_len = prompt_ids.shape[-1]
        attention_mask = data.batch["attention_mask"]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)
        prompt_lengths = attention_mask[:, :prompt_len].sum(dim=-1)

        em_list = reward_extra_info.get("em", [None] * n)
        score_list = reward_extra_info.get("score", [None] * n)

        # Ground truths exactly the way BatchRewardManager.verify() obtains them.
        ground_truths = [
            item.non_tensor_batch["reward_model"].get("ground_truth", None) for item in data
        ]

        group_agg = {}
        rollout_records = []
        seen_uid_pos = {}
        identity_mismatch = 0
        identity_max_abs_diff = 0.0
        identity_compared = 0

        for i in range(n):
            uid = uids[i]
            uid_str = str(uid)
            extra_info = extra_infos[i] if extra_infos[i] is not None else {}
            extra_index = extra_info.get("index") if isinstance(extra_info, dict) else None
            question = extra_info.get("question") if isinstance(extra_info, dict) else None
            rollout_pos = seen_uid_pos.get(uid_str, 0)
            seen_uid_pos[uid_str] = rollout_pos + 1

            valid_len = int(valid_response_lengths[i].item())
            response_str = self.tokenizer.decode(response_ids[i][:valid_len], skip_special_tokens=True)
            prompt_str = self.tokenizer.decode(prompt_ids[i], skip_special_tokens=True)
            response_char_len = len(response_str)

            messages, messages_ok = self._unwrap_messages(messages_raw[i] if i < len(messages_raw) else None)

            dtm = detailed_tool_metrics[i] if i < len(detailed_tool_metrics) else None
            structured = self._structured_metrics(dtm)

            # ---- response-text side (independent cross-check) ----
            xml_batches, xml_queries = _count_logical_search_batches_and_queries_from_text(response_str)
            resp_tool_call_blocks, resp_wiki_blocks = _count_rendered_wiki_tool_calls(response_str)
            resp_wiki_tag_count = len(_WIKI_SEARCH_REGEX.findall(response_str))

            em = em_list[i] if i < len(em_list) else None
            score = score_list[i] if i < len(score_list) else None
            is_correct = bool(em == 1 or em == 1.0)

            # ---- answer parse status, via the ORIGINAL extractor ----
            try:
                extracted = extract_solution_original(response_str)
            except Exception:  # noqa: BLE001
                extracted = None
            answer_parse_ok = extracted is not None

            # ---- independent reward identity check (task section 0) ----
            reward_e10 = None
            if reward_tensor is not None:
                try:
                    reward_e10 = float(reward_tensor[i].sum().item())
                except Exception:  # noqa: BLE001
                    reward_e10 = None
            reward_original = None
            try:
                gt = ground_truths[i] if i < len(ground_truths) else None
                reward_original = float(
                    compute_score_em_batch_original(
                        [data_sources[i] if i < len(data_sources) else None],
                        [prompt_str],
                        [response_str],
                        [gt],
                        [extra_info],
                    )[0]
                )
            except Exception:  # noqa: BLE001
                reward_original = None

            if reward_e10 is not None and reward_original is not None and score is not None:
                identity_compared += 1
                d = abs(reward_e10 - reward_original)
                identity_max_abs_diff = max(identity_max_abs_diff, d)
                if d != 0 or float(score) != reward_original or float(em) != reward_original:
                    identity_mismatch += 1

            # ---- E1-0 cost candidates ----
            logical_batches = structured["structured_logical_batches"]
            rounds = structured["structured_search_rounds"]
            queries = structured["structured_search_calls"]
            zero_search = bool((queries or 0) == 0 and (logical_batches or 0) == 0)
            queries_per_round = (queries / rounds) if (queries and rounds) else 0.0
            parallel_factor = (queries / logical_batches) if (queries and logical_batches) else 0.0

            record = {
                "run_id": os.path.basename(self.run_dir),
                "rank": 0,
                "pid": self.pid,
                "process_role": "driver_taskrunner",
                "call_index": call_index,
                "call_within_step": call_within_step,
                "global_step": global_step,
                "global_step_source": global_step_source,
                "timestamp": timestamp,
                # identity
                "uid": uid_str,
                "rollout_pos_in_group": rollout_pos,
                "extra_info_index": extra_index,
                "question": question,
                "data_source": data_sources[i] if i < len(data_sources) else None,
                # correctness
                "em": em,
                "reward_score_extra": score,
                "reward_e10": reward_e10,
                "reward_original": reward_original,
                "reward_identity_abs_diff": (
                    abs(reward_e10 - reward_original)
                    if (reward_e10 is not None and reward_original is not None)
                    else None
                ),
                "answer_parse_ok": answer_parse_ok,
                "answer_extracted": extracted if isinstance(extracted, list) else None,
                "completion_reason": complete_reasons[i] if i < len(complete_reasons) else None,
                "finish_reason": finish_reasons[i] if i < len(finish_reasons) else None,
                "response_excerpt": _response_excerpt(response_str),
                "tool_call_summary": structured["structured_sequence_summary"],
                # length
                "prompt_token_length": int(prompt_lengths[i].item()),
                "response_token_length": valid_len,
                "response_char_length": response_char_len,
                # search behaviour (structured = primary, text = cross-check)
                "logical_search_batches": logical_batches,
                "structured_logical_batches": logical_batches,
                "structured_search_rounds": rounds,
                "structured_search_calls": queries,
                "structured_total_tool_calls": structured["structured_total_tool_calls"],
                "structured_conversation_turns": structured["structured_conversation_turns"],
                "structured_nonwiki_tool_calls": structured["structured_nonwiki_tool_calls"],
                "search_queries": queries,
                "queries_per_round": queries_per_round,
                "parallel_factor": parallel_factor,
                "zero_search": zero_search,
                "metadata_valid": bool(structured["metadata_valid"] and messages_ok),
                "messages_nesting_ok": messages_ok,
                "fallback_degraded": structured["fallback_degraded"],
                "block_index_available": structured["block_index_available"],
                # response-text side cross-check
                "resp_xml_search_batches": xml_batches,
                "resp_xml_search_queries": xml_queries,
                "resp_tool_call_blocks": resp_tool_call_blocks,
                "resp_wiki_tool_call_blocks": resp_wiki_blocks,
                "resp_wiki_search_tag_count": resp_wiki_tag_count,
            }
            rollout_records.append(record)

            g = group_agg.setdefault(
                uid_str,
                {
                    "indices": [],
                    "correct": 0,
                    "incorrect": 0,
                    "logical_all": [],
                    "logical_correct": [],
                    "rounds_all": [],
                    "rounds_correct": [],
                    "queries_all": [],
                    "queries_correct": [],
                },
            )
            g["indices"].append(extra_index)
            g["logical_all"].append(logical_batches)
            g["rounds_all"].append(rounds)
            g["queries_all"].append(queries)
            if is_correct:
                g["correct"] += 1
                g["logical_correct"].append(logical_batches)
                g["rounds_correct"].append(rounds)
                g["queries_correct"].append(queries)
            else:
                g["incorrect"] += 1

        with open(self.rollouts_path, "a") as f:
            for r in rollout_records:
                f.write(json.dumps(r, default=str) + "\n")

        group_records = []
        for uid_str, g in group_agg.items():
            group_size = len(g["indices"])
            logical_stats = _stats(g["logical_all"])
            rounds_stats = _stats(g["rounds_all"])
            logical_correct_stats = _stats(g["logical_correct"])
            rounds_correct_stats = _stats(g["rounds_correct"])
            queries_stats = _stats(g["queries_all"])
            queries_correct_stats = _stats(g["queries_correct"])
            group_records.append(
                {
                    "run_id": os.path.basename(self.run_dir),
                    "pid": self.pid,
                    "call_index": call_index,
                    "call_within_step": call_within_step,
                    "global_step": global_step,
                    "timestamp": timestamp,
                    "group_id": uid_str,
                    "group_size": group_size,
                    "extra_info_indices": g["indices"],
                    "correct_count": g["correct"],
                    "incorrect_count": g["incorrect"],
                    "all_correct": g["correct"] == group_size,
                    "all_wrong": g["correct"] == 0,
                    "logical_search_batches_all": g["logical_all"],
                    "logical_search_batches_correct": g["logical_correct"],
                    "structured_search_rounds_all": g["rounds_all"],
                    "structured_search_rounds_correct": g["rounds_correct"],
                    "search_queries_all": g["queries_all"],
                    "search_queries_correct": g["queries_correct"],
                    "cost_xml_all_stats": logical_stats,
                    "cost_xml_correct_stats": logical_correct_stats,
                    "cost_structured_all_stats": rounds_stats,
                    "cost_structured_correct_stats": rounds_correct_stats,
                    "cost_queries_all_stats": queries_stats,
                    "cost_queries_correct_stats": queries_correct_stats,
                    "xml_variation_correct": bool(
                        logical_correct_stats["n"] >= 2 and logical_correct_stats["max"] > logical_correct_stats["min"]
                    ),
                    "structured_variation_correct": bool(
                        rounds_correct_stats["n"] >= 2 and rounds_correct_stats["max"] > rounds_correct_stats["min"]
                    ),
                }
            )
        with open(self.groups_path, "a") as f:
            for gr in group_records:
                f.write(json.dumps(gr, default=str) + "\n")

        call_record = {
            "run_id": os.path.basename(self.run_dir),
            "pid": self.pid,
            "call_index": call_index,
            "call_within_step": call_within_step,
            "global_step": global_step,
            "global_step_source": global_step_source,
            "timestamp": timestamp,
            "n_rollouts": n,
            "n_groups": len(group_agg),
            "correct_count_total": sum(g["correct"] for g in group_agg.values()),
            "identity_compared": identity_compared,
            "identity_mismatch": identity_mismatch,
            "identity_max_abs_diff": identity_max_abs_diff,
            "metadata_valid_false": sum(1 for r in rollout_records if not r["metadata_valid"]),
        }
        with open(self.calls_path, "a") as f:
            f.write(json.dumps(call_record, default=str) + "\n")

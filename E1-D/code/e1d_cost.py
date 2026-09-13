"""E1-D cost extraction: `logical_search_batches` for a rollout.

E1-D's efficiency term is group-relative in the *cost* `logical_search_batches`,
defined (E1-D_task.md) as the number of logical `<wiki_search>` blocks the
assistant emitted, i.e. the number of sequential search rounds.

Why this cannot simply be read off the response text
----------------------------------------------------
`SGLangRollout.__init__` replaces the tokenizer chat template
(`verl/verl/workers/rollout/sglang_rollout/sglang_rollout.py:289`) with one that
renders every tool call as `<name>arguments</name>`.  `XMLToolParser
.parse_non_stream` splits one `<wiki_search>a|b|c</wiki_search>` block into
**three** tool calls, so the re-rendered response contains one `<wiki_search>`
block per *query*, not per *logical block*.  Measured on the 147,200-rollout
E1-0 audit: `response <wiki_search> count == search_queries` for 99.79% of
rollouts, while `logical blocks == search rounds` for 100%.

`XMLToolParser.parse_non_stream` assigns each call the id
`call_<tool>_<block_index>_<query_index>`, and `schemas.py::track_tool_call`
stores that id together with the assistant `turn` in
`detailed_tool_metrics.tool_call_sequence`.  Counting distinct `(turn, block)`
pairs therefore reconstructs the logical block count **exactly**.  That logic is
the same one validated in E1-0 (and re-derived here so E1-D is self-contained).

Fallbacks (never fatal):
  * no metadata at all           -> cost None, metadata_valid False
  * calls but unparseable ids    -> cost = number of distinct assistant turns
"""
import os
import re

_CALL_ID_RE = re.compile(r"^call_(.+?)_(\d+)(?:_(\d+))?$")
# response-text fallback: one rendered block per query (see module docstring)
_RESP_BLOCK_RE = re.compile(r"<wiki_search>(.*?)</wiki_search>", re.DOTALL)
_OBS_RE = re.compile(r"<observation>.*?</observation>", re.DOTALL)


def structured_cost(dtm):
    """Return per-rollout cost info from one `detailed_tool_metrics` entry.

    Returns a dict with:
        logical_search_batches : int | None   (None => metadata unusable)
        search_queries         : int | None
        search_turns           : list[int]    (assistant turns that issued search)
        total_tool_calls       : int | None
        conversation_turns     : int | None
        metadata_valid         : bool
        block_index_available  : bool
    """
    if not isinstance(dtm, dict) or not dtm:
        return {
            "logical_search_batches": None, "search_queries": None, "search_turns": [],
            "total_tool_calls": None, "conversation_turns": None,
            "metadata_valid": False, "block_index_available": False,
        }
    seq = dtm.get("tool_call_sequence", [])
    if not isinstance(seq, list):
        seq = []
    wiki = [c for c in seq if isinstance(c, dict) and c.get("tool_name") == "wiki_search"]
    turns, blocks, block_index_available = set(), set(), False
    for c in wiki:
        turn = c.get("turn")
        if turn is not None:
            turns.add(turn)
        m = _CALL_ID_RE.match(str(c.get("call_id", "")))
        if m:
            block_index_available = True
            blocks.add((turn, int(m.group(2))))
    if blocks:
        logical = len(blocks)
    elif turns:
        # calls present but ids unparseable: distinct search turns is the best
        # available upper bound (and equals the block count whenever the model
        # emits at most one block per turn, as it does throughout E1-0).
        logical = len(turns)
    else:
        logical = 0
    return {
        "logical_search_batches": int(logical),
        "search_queries": int(len(wiki)),
        "search_turns": sorted(int(t) for t in turns if t is not None),
        "total_tool_calls": dtm.get("total_tool_calls"),
        "conversation_turns": dtm.get("conversation_turns"),
        "metadata_valid": True,
        "block_index_available": block_index_available,
    }


def text_cost(response_str):
    """Response-text fallback: number of rendered search blocks (= queries) and
    number of assistant spans that issue >=1 search (== retrieval rounds, the
    repo's own `summarize_eval.py` convention)."""
    if not response_str:
        return {"resp_search_blocks": 0, "resp_search_rounds": 0}
    blocks = 0
    for m in _RESP_BLOCK_RE.finditer(response_str):
        if m.group(1).strip():
            blocks += 1
    spans = _OBS_RE.split(response_str)
    rounds = sum(1 for s in spans if "<wiki_search>" in s)
    return {"resp_search_blocks": blocks, "resp_search_rounds": rounds}


def costs_from_batch(non_tensor_batch, n):
    """Extract cost arrays for a whole training batch (length n)."""
    dtms = non_tensor_batch.get("detailed_tool_metrics", [None] * n)
    costs, queries, turns, valid = [], [], [], []
    for i in range(n):
        info = structured_cost(dtms[i] if i < len(dtms) else None)
        costs.append(info["logical_search_batches"])
        queries.append(info["search_queries"])
        turns.append(info["conversation_turns"])
        valid.append(bool(info["metadata_valid"]))
    return {
        "costs": costs,
        "queries": queries,
        "conversation_turns": turns,
        "metadata_valid": valid,
    }


def run_dir():
    return os.environ.get("E1D_RUN_DIR")

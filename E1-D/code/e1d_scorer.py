"""E1-D reward scorer: success-conditioned group-relative efficiency reward.

    A_i = EM correctness (original GAP rule, unchanged)
    C_i = logical_search_batches of rollout i
    E_i = (Cmax - C_i) / (Cmax - Cmin)   if the uid group has >= 2 correct
                                          rollouts AND their costs differ
        = 0                              otherwise
    R_i = A_i * (1 + lambda * E_i)        lambda = 0.05, wrong rollout -> 0

Design constraints (E1-D_task.md):
  * `compute_score_em_batch()` is NOT deleted -- the original
    `verl/verl/utils/reward_score/mhqa_train.py` is untouched and its function
    is re-exported here for reference/back-compat.
  * EM is computed by calling the ORIGINAL `compute_score_em()` (never
    re-implemented), so A_i is byte-identical to the baseline reward.
  * `R_i` (the shaped reward) and `A_i` (accuracy) are returned in separate
    dict fields; the reward manager writes `score` into the reward tensor and
    `em` into `data.batch["acc"]`, so the shaped reward never contaminates the
    accuracy metric.
"""
import importlib.util
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_ORIGINAL_MHQA_TRAIN = os.path.join(_REPO_ROOT, "verl", "verl", "utils", "reward_score", "mhqa_train.py")


def _load_original_scorer():
    spec = importlib.util.spec_from_file_location("e1d_original_mhqa_train", _ORIGINAL_MHQA_TRAIN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ORIG = _load_original_scorer()

# The original single-item and batch scorers, re-exported untouched.
compute_score_em = _ORIG.compute_score_em
compute_score_em_batch = _ORIG.compute_score_em_batch

DEFAULT_LAMBDA = 0.05


def get_lambda():
    try:
        return float(os.environ.get("E1D_LAMBDA", DEFAULT_LAMBDA))
    except (TypeError, ValueError):
        return DEFAULT_LAMBDA


def compute_score_em_efficiency_batch(
    data_sources,
    prompts,
    responses,
    ground_truths,
    extra_infos,
    uids=None,
    costs=None,
    queries=None,
    conversation_turns=None,
    lambda_=None,
    **kwargs,
):
    """Group-relative efficiency-shaped batch reward.

    Args:
        data_sources/prompts/responses/ground_truths/extra_infos: as in the
            original batch scorer.
        uids: per-rollout uid (the n=8 group key). Required for any non-zero E.
        costs: per-rollout `logical_search_batches`. Required for E.
        queries, conversation_turns: optional, forwarded for diagnostics only.
        lambda_: override for the 0.05 default.

    Returns:
        list[dict] with, per rollout:
            score       -> R_i (shaped reward; this is what enters GRPO)
            em          -> A_i (original EM; this is the accuracy metric)
            efficiency  -> E_i
            cost, search_queries, group_size, group_correct_count,
            group_cost_min, group_cost_max, efficiency_active
    """
    lam = DEFAULT_LAMBDA if lambda_ is None else float(lambda_)
    n = len(responses)

    # ---- A_i: original EM, unchanged code path ----
    em = [0.0] * n
    for i in range(n):
        ds = data_sources[i]
        p = prompts[i]
        s = responses[i]
        gt = ground_truths[i]
        original = compute_score_em(ds, p, s, gt)
        em[i] = float(original["em"])
        # E1-D invariant: the original scorer's score is the EM (no shaping there)
        assert float(original["score"]) == em[i], "original scorer score != em"

    # ---- group bookkeeping on uid ----
    groups = {}
    if uids is not None:
        for i in range(n):
            groups.setdefault(str(uids[i]), []).append(i)

    out = []
    for i in range(n):
        uid = str(uids[i]) if uids is not None else None
        members = groups.get(uid, []) if uid is not None else []
        correct_members = []
        if costs is not None:
            correct_members = [j for j in members if em[j] >= 0.5 and costs[j] is not None]
        group_size = len(members) if members else None
        group_correct = len([j for j in members if em[j] >= 0.5]) if members else None
        cmin = cmax = None
        active = False
        if len(correct_members) >= 2:
            vals = [float(costs[j]) for j in correct_members]
            cmin, cmax = min(vals), max(vals)
            active = cmax > cmin
        e = 0.0
        if active and em[i] >= 0.5 and costs is not None and costs[i] is not None:
            e = (cmax - float(costs[i])) / (cmax - cmin)
        a = em[i]
        r = a * (1.0 + lam * e)
        if a == 0.0:
            r = 0.0  # wrong rollout: reward exactly 0 (explicit, not just implied)
        out.append({
            "score": float(r),
            "em": float(a),
            "efficiency": float(e),
            "cost": (float(costs[i]) if (costs is not None and costs[i] is not None) else None),
            "search_queries": (float(queries[i]) if (queries is not None and queries[i] is not None) else None),
            "conversation_turns": (float(conversation_turns[i]) if (conversation_turns is not None and conversation_turns[i] is not None) else None),
            "group_size": group_size,
            "group_correct_count": group_correct,
            "group_cost_min": cmin,
            "group_cost_max": cmax,
            "efficiency_active": bool(active),
            "lambda": lam,
        })
    return out

"""E1-E reward scorer - group-type-specific reward for q=2 solved-group gated training.

Frozen semantics (E1-E_task.md section 1/5, pre-registered by E1-E0):

    mixed-correctness group (1 <= k <= 7):
        R_i = A_i                     (original GAP reward, NO shaping)
    all-correct efficiency-variable group (k == 8 and max(C) > min(C) over correct rollouts):
        E_i = (Cmax - C_i) / (Cmax - Cmin)
        R_i = 1 + 0.05 * E_i
    everything else (all-wrong, all-correct without cost variation):
        R_i = A_i                     (never enter the E1-E optimizer batch)

`em` is always the TRUE original EM computed by the untouched
`mhqa_train.py::compute_score_em`, so `data.batch["acc"]` can never be contaminated.

`e1e_quota_metric` is the value published to the trainer's group filter
(`algorithm.filter_groups.metric=e1e_quota_metric`); it is built so that exactly the groups
E1-E keeps have std > 0:

    mixed group       -> A_i                (0/1 -> std > 0)
    efficiency group  -> 1 + 0.05 * E_i     (varies -> std > 0)
    anything else     -> constant 0.0       (std == 0)
"""
import importlib.util
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_ORIGINAL_MHQA_TRAIN = os.path.join(_REPO_ROOT, "verl", "verl", "utils", "reward_score", "mhqa_train.py")


def _load_original_scorer():
    spec = importlib.util.spec_from_file_location("e1e_original_mhqa_train", _ORIGINAL_MHQA_TRAIN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ORIG = _load_original_scorer()

# original single-item / batch scorers, re-exported untouched
compute_score_em = _ORIG.compute_score_em
compute_score_em_batch = _ORIG.compute_score_em_batch

DEFAULT_LAMBDA = 0.05


def get_lambda():
    try:
        return float(os.environ.get("E1E_LAMBDA", DEFAULT_LAMBDA))
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
    """Group-type-specific E1-E batch reward.  See the module docstring."""
    lam = DEFAULT_LAMBDA if lambda_ is None else float(lambda_)
    n = len(responses)

    # ---- A_i: the original EM, unchanged code path ----
    em = [0.0] * n
    for i in range(n):
        original = compute_score_em(data_sources[i], prompts[i], responses[i], ground_truths[i])
        em[i] = float(original["em"])
        assert float(original["score"]) == em[i], "original scorer score != em"

    groups = {}
    if uids is not None:
        for i in range(n):
            groups.setdefault(str(uids[i]), []).append(i)

    out = []
    for i in range(n):
        uid = str(uids[i]) if uids is not None else None
        members = groups.get(uid, []) if uid is not None else []
        size = len(members) if members else None
        k = int(sum(1 for j in members if em[j] >= 0.5)) if members else None

        cmin = cmax = None
        gap = 0.0
        if k is not None and costs is not None and k >= 2:
            vals = [float(costs[j]) for j in members if em[j] >= 0.5 and costs[j] is not None]
            if vals:
                cmin, cmax = min(vals), max(vals)
                gap = cmax - cmin
        group_efficiency = bool(k is not None and size is not None and k == size and gap > 0.0)
        group_mixed = bool(k is not None and 1 <= k <= 7)

        e = 0.0
        if (k is not None and k >= 2 and gap > 0.0 and em[i] >= 0.5
                and costs is not None and costs[i] is not None):
            e = (cmax - float(costs[i])) / gap

        a = em[i]
        r = (1.0 + lam * e) if group_efficiency else a

        if group_mixed:
            quota_metric = a
        elif group_efficiency:
            quota_metric = 1.0 + lam * e
        else:
            quota_metric = 0.0

        out.append({
            "score": float(r),
            "em": float(a),
            "efficiency": float(e if group_efficiency else 0.0),
            "cost": (float(costs[i]) if (costs is not None and costs[i] is not None) else None),
            "search_queries": (float(queries[i]) if (queries is not None and queries[i] is not None) else None),
            "conversation_turns": (float(conversation_turns[i]) if (conversation_turns is not None and conversation_turns[i] is not None) else None),
            "group_size": size,
            "group_k": k,
            "group_correct_count": k,
            "group_cost_min": cmin,
            "group_cost_max": cmax,
            "group_cost_gap": float(gap),
            "group_mixed": group_mixed,
            "group_efficiency": group_efficiency,
            "group_active": group_efficiency,
            "e1e_quota_metric": float(quota_metric),
            "efficiency_active": group_efficiency,
            "lambda": lam,
        })
    return out


def reward_matrix_check(per_rollout):
    """Verify the E1-E reward invariants; returns a list of violations."""
    bad = []
    for i, r in enumerate(per_rollout):
        if r["group_mixed"]:
            if abs(r["score"] - r["em"]) > 1e-12:
                bad.append((i, "mixed reward != EM"))
        elif r["group_efficiency"]:
            if not (1.0 - 1e-12 <= r["score"] <= 1.0 + r["lambda"] + 1e-12):
                bad.append((i, "efficiency reward out of [1, 1+lambda]"))
            if r["em"] != 1.0:
                bad.append((i, "efficiency group contains a wrong rollout"))
            if r["efficiency"] <= 0.0:
                bad.append((i, "efficiency group with E == 0"))
        else:
            if abs(r["score"] - r["em"]) > 1e-12:
                bad.append((i, "excluded group reward != EM"))
        if r["em"] == 0.0 and abs(r["score"]) > 1e-12:
            bad.append((i, "wrong rollout reward != 0"))
    return bad

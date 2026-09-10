"""E1-0 batch scorer.

Passive-audit wrapper around the ORIGINAL GAP scorer
(verl/verl/utils/reward_score/mhqa_train.py:compute_score_em). Does not
reimplement normalize/extract/EM rules; imports and calls the original
function unmodified, then repackages its result. Guarantees
`score == em` for every item (no shaped reward, no bonus/penalty).

Reward-identity contract required by E1-0_task.md section 0:
    E1-0 reward (score) == original GAP reward for the identical input.
Since this file calls compute_score_em() directly (not a reimplementation),
this holds by construction; the training-time audit additionally verifies
it empirically per rollout (see e10_reward_manager.py).
"""
import os
import sys
import importlib.util

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "..", ".."))
_ORIGINAL_MHQA_TRAIN_PATH = os.path.join(
    _REPO_ROOT, "verl", "verl", "utils", "reward_score", "mhqa_train.py"
)

_spec = importlib.util.spec_from_file_location("e10_original_mhqa_train", _ORIGINAL_MHQA_TRAIN_PATH)
_original_mhqa_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_original_mhqa_train)

compute_score_em = _original_mhqa_train.compute_score_em


def compute_score_em_batch_e10(
    data_sources,
    prompts,
    responses,
    ground_truths,
    extra_infos,
    questions=None,
    **kwargs,
):
    """E1-0 train-time batch scorer.

    Returns a list of dicts (not bare floats) so that
    BatchRewardManager.__call__'s `isinstance(score, dict)` branch fires and
    every key here is forwarded into `non_tensor_batch` via
    `reward_extra_info` -- this is how `em` (audit-only accuracy field,
    numerically identical to `score` for E1-0) reaches the trainer without
    touching batch.py or ray_trainer.py.
    """
    out = []
    n = len(responses)
    for i in range(n):
        ds = data_sources[i]
        p = prompts[i]
        s = responses[i]
        gt = ground_truths[i]
        original = compute_score_em(ds, p, s, gt)
        score = original["score"]
        em = original["em"]
        assert score == em, f"E1-0 invariant violated: score({score}) != em({em}) at index {i}"
        out.append({
            "score": score,
            "em": em,
        })
    return out

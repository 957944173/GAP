"""E1-0 shared passive state (audit + diagnostics).

Holds a reference to the live RayPPOTrainer inside the driver process so the
E1-0 audit reward manager can record the *real* `trainer.global_steps` for each
reward call, instead of guessing it from a call counter.

Why this is needed (code-review finding, 2026-09-10):
    filter_groups is enabled and `num_gen_batches` is ~8-14 per global step for
    this run (verified from the original run's supervisor.log at step 60:
    `observed step=60 num_gen_batches=11`).  That means the reward manager is
    invoked ~8-14 times PER global step, not once.  A naive
    `global_step = start + 1 + call_index` attribution is therefore wrong by
    roughly an order of magnitude.  Reading the trainer's live counter is exact
    even when several generation batches are consumed by one step.

Nothing here changes any training-visible value: it is a read-only reference
plus a `print()` helper.
"""
import time

# Set by e10_trainer_patch.ensure_driver_patch() when RayPPOTrainer is
# constructed (driver process only).
TRAINER = None


def diag(msg):
    """Rank/process-tagged, unbuffered passive diagnostic print."""
    print(f"[E1-0 diag][driver] {msg} (t={time.time():.1f})", flush=True)


def current_global_step():
    """Return the live trainer.global_steps, or None if unavailable."""
    t = TRAINER
    if t is None:
        return None
    try:
        return int(t.global_steps)
    except Exception:  # noqa: BLE001 -- must never break training
        return None

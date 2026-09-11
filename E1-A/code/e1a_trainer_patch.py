"""E1-0 driver-side passive instrumentation + exact global_step capture.

Loaded lazily by e1a_reward_manager.E10BatchAuditRewardManager.__init__ (which
only ever runs in the driver process, inside main_ppo.TaskRunner.run, *after*
`verl.trainer.ppo.ray_trainer` has already been imported by main_ppo).  That
lazy trigger is deliberate: importing ray_trainer from sitecustomize at
interpreter start-up would pull an extra dependency stack into EVERY process on
PYTHONPATH (including spawned SGLang scheduler subprocesses), which is not
needed and would be a gratuitous deviation from the original runtime.

What is installed (all pure observation, no control-flow/reward/sampling change):
  1. RayPPOTrainer.__init__ wrapper -> registers the live trainer instance in
     e1a_state so the audit can read the true `global_steps`.
  2. RayPPOTrainer._load_checkpoint wrapper -> prints entry/return.  This is
     what distinguishes "driver still blocked inside the load_checkpoint RPC"
     from "RPC returned, driver stuck later" for the attempt-18 post-load hang
     (attempt 18's launcher log froze right after the last
     `Resuming from ...` line with no `test_gen_batch meta info` ever printed).
  3. RayPPOTrainer._validate wrapper -> prints entry/exit per call, so a hang
     inside the pre-training validation is immediately localizable.
"""
import functools

import e1a_state

_PATCHED = False


def ensure_driver_patch():
    """Idempotently install the driver-side wrappers. Safe to call repeatedly."""
    global _PATCHED
    if _PATCHED:
        return

    import verl.trainer.ppo.ray_trainer as rt

    trainer_cls = rt.RayPPOTrainer

    orig_init = trainer_cls.__init__

    @functools.wraps(orig_init)
    def _e1a_init(self, *args, **kwargs):
        result = orig_init(self, *args, **kwargs)
        e1a_state.TRAINER = self
        try:
            e1a_state.diag(
                "RayPPOTrainer registered "
                f"(default_local_dir={self.config.trainer.default_local_dir}, "
                f"save_freq={self.config.trainer.save_freq}, "
                f"test_freq={self.config.trainer.test_freq}, "
                f"total_training_steps={self.total_training_steps})"
            )
        except Exception:  # noqa: BLE001
            pass
        return result

    trainer_cls.__init__ = _e1a_init

    orig_load = trainer_cls._load_checkpoint

    @functools.wraps(orig_load)
    def _e1a_load_checkpoint(self, *args, **kwargs):
        e1a_state.diag("_load_checkpoint: entered (driver)")
        result = orig_load(self, *args, **kwargs)
        e1a_state.diag(
            f"_load_checkpoint: returned to driver (global_steps={getattr(self, 'global_steps', None)})"
        )
        return result

    trainer_cls._load_checkpoint = _e1a_load_checkpoint

    orig_validate = trainer_cls._validate

    @functools.wraps(orig_validate)
    def _e1a_validate(self, *args, **kwargs):
        e1a_state.diag(f"_validate: entered (global_steps={getattr(self, 'global_steps', None)})")
        result = orig_validate(self, *args, **kwargs)
        e1a_state.diag(f"_validate: returned (global_steps={getattr(self, 'global_steps', None)})")
        return result

    trainer_cls._validate = _e1a_validate

    _PATCHED = True
    e1a_state.diag("driver-side passive instrumentation installed")

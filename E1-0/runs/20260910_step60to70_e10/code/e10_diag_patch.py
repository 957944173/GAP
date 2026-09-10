"""E1-0 worker-side passive diagnostics (attempt 19+).

Purpose: pin down the attempt-18 stall, whose only *positive* evidence was
negative (all 4 ranks returned from `FSDPCheckpointManager.load_checkpoint`
at t=...629.6, then the launcher log stayed frozen for ~2 hours and the
driver never reached `_validate()`'s first `test_gen_batch meta info` print).

These wrappers add entry/return `print()`s at the exact boundaries that were
previously invisible, so a repeat stall is localizable to one line instead of
being inferred from absence of output:

  * ActorRolloutRefWorker.load_checkpoint : worker side of the resume RPC.
      - if this never prints "entered", the driver never dispatched the RPC;
      - if it prints "entered" but not "returned", the stall is inside the
        worker (already instrumented further down by e10_checkpoint_patch);
      - if all 4 print "returned" and the driver still does not print
        "_load_checkpoint: returned to driver", the stall is driver-side
        *after* the RPC (data.pt / StatefulDataLoader.load_state_dict) or in
        Ray's result-return path.
  * FSDPSGLangShardingManager.__enter__/__exit__ : the FSDP -> SGLang weight
      resharding / CUDA-IPC handoff that must run before the first rollout.
      This is the handoff identified as the leading suspect in attempt 18's
      diagnosis and is the only place where a silent cross-process deadlock
      (sender waiting on a collective, engine waiting on tensors) would show
      up as 0% GPU util with no GPU error.

CRITICAL: `ActorRolloutRefWorker.load_checkpoint` / `generate_sequences` are
@register()-decorated. verl's `_bind_workers_method_to_parent` copies the
bound function objects into the dynamically-created WorkerDict class at
`init_workers()` time, i.e. AFTER sitecustomize ran, so wrapping the class
attributes here is picked up by the dispatch machinery -- but only if the
MAGIC dispatch attribute is preserved on the wrapper. functools.wraps copies
__dict__ (which holds it) and we re-set it explicitly anyway.

Pure logging: no control flow, reward, sampling, GRPO, optimizer, checkpoint,
rollout or tool-execution semantics are touched.
"""
import functools

from verl.single_controller.base.decorator import MAGIC_ATTR

import e10_state


def _wrap_method(cls, name):
    orig = getattr(cls, name, None)
    if orig is None or getattr(orig, "_e10_wrapped", False):
        return False
    label = f"{cls.__name__}.{name}"

    @functools.wraps(orig)
    def wrapper(self, *args, **kwargs):
        print(f"[E1-0 diag][worker] {label} entered", flush=True)
        result = orig(self, *args, **kwargs)
        print(f"[E1-0 diag][worker] {label} returned", flush=True)
        return result

    wrapper._e10_wrapped = True
    if hasattr(orig, MAGIC_ATTR):
        setattr(wrapper, MAGIC_ATTR, getattr(orig, MAGIC_ATTR))
    wrapper.__wrapped__ = orig
    setattr(cls, name, wrapper)
    return True


def _install():
    installed = []
    try:
        import verl.workers.fsdp_workers as fw

        for cls_name in ("ActorRolloutRefWorker", "AsyncActorRolloutRefWorker"):
            cls = getattr(fw, cls_name, None)
            if cls is None:
                continue
            for meth in ("load_checkpoint", "generate_sequences"):
                if _wrap_method(cls, meth):
                    installed.append(f"{cls_name}.{meth}")
    except Exception as e:  # noqa: BLE001 -- diagnostics must never break training
        print(f"[E1-0 diag][worker] fsdp_workers instrumentation failed (non-fatal): {e}", flush=True)

    try:
        from verl.workers.sharding_manager.fsdp_sglang import FSDPSGLangShardingManager

        for meth in ("__enter__", "__exit__"):
            if _wrap_method(FSDPSGLangShardingManager, meth):
                installed.append(f"FSDPSGLangShardingManager.{meth}")
    except Exception as e:  # noqa: BLE001
        print(f"[E1-0 diag][worker] sharding-manager instrumentation failed (non-fatal): {e}", flush=True)

    return installed


INSTALLED = _install()

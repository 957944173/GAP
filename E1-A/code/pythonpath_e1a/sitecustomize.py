"""E1-A registration + runtime-fix shim (PYTHONPATH-injected).

Auto-imported by the Python interpreter at start-up in every process that
inherits `PYTHONPATH=<E1-A>/code/pythonpath_e1a`.  It does two things:

1. Registers the E1-A reward manager (`e1a_reward_manager` -> its
   `@register("e1a_batch_shaped")` decorator) so the otherwise-unmodified
   `verl` registry can resolve `reward_model.reward_manager=e1a_batch_shaped`.
   No original verl file is edited.

2. Loads the runtime fixes that were root-caused and validated in E1-0 and are
   required for training to run at all on this host (copies live in E1-A/code,
   so E1-0's files are untouched):
     * e1a_hang_diagnostics  - on-demand SIGUSR1 faulthandler stacks
     * e1a_checkpoint_patch  - FSDP `load_checkpoint(..., map_location="cpu")`
     * e1a_sglang_gpu_patch  - per-rank SGLang GPU visibility under
       RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1; the CUDA-IPC reductions
       patch is applied lazily inside the WorkerDict actor (NOT at import), which
       is the fix for the fork/DataLoader deadlock found in E1-0
     * e1a_diag_patch        - passive worker boundary prints
"""
import os
import sys

_E1A_SHIM_DIR = os.path.dirname(os.path.abspath(__file__))
_E1A_CODE_DIR = os.path.abspath(os.path.join(_E1A_SHIM_DIR, ".."))
if _E1A_CODE_DIR not in sys.path:
    sys.path.insert(0, _E1A_CODE_DIR)

try:
    import e1a_hang_diagnostics  # noqa: F401
except Exception as e:  # pragma: no cover
    sys.stderr.write(f"E1-A sitecustomize: e1a_hang_diagnostics failed (non-fatal): {e}\n")

try:
    import e1a_reward_manager  # noqa: F401  -- runs @register("e1a_batch_shaped")
except Exception as e:  # pragma: no cover
    sys.stderr.write(f"E1-A sitecustomize: failed to import e1a_reward_manager: {e}\n")
    raise

try:
    import e1a_checkpoint_patch  # noqa: F401
except Exception as e:  # pragma: no cover
    sys.stderr.write(f"E1-A sitecustomize: failed to import e1a_checkpoint_patch: {e}\n")
    raise

try:
    import e1a_sglang_gpu_patch  # noqa: F401
except Exception as e:  # pragma: no cover
    sys.stderr.write(f"E1-A sitecustomize: failed to import e1a_sglang_gpu_patch: {e}\n")
    raise

try:
    import e1a_diag_patch  # noqa: F401
except Exception as e:  # pragma: no cover
    sys.stderr.write(f"E1-A sitecustomize: e1a_diag_patch failed (non-fatal): {e}\n")

# NCCL "Duplicate GPU detected" crash: diagnosis and self-fix (attempts 1-2 -> attempt 3)

## Symptom (identical on attempt 1 and attempt 2)

```
torch.distributed.DistBackendError: NCCL error in: /pytorch/torch/csrc/distributed/c10d/NCCLUtils.hpp:268, invalid usage (run with NCCL_DEBUG=WARN for details), NCCL version 2.21.5
ncclInvalidUsage: This usually reflects invalid usage of NCCL library.
Last error:
Duplicate GPU detected : rank X and rank 0 both on CUDA device 34000
```

Occurs during `fsdp_workers.py:_build_model_optimizer` -> `FSDP.__init__` -> `_recursive_wrap`
-> `_init_param_handle_from_module` -> `_sync_module_params_and_buffers` ->
`dist._broadcast_coalesced`, i.e. during the *first* FSDP wrap of the freshly
HF-loaded model. Both crashes occurred strictly **before** checkpoint/optimizer
resume and before any training step (confirmed: no `experiments/DAPO-GAP3B-MHQA-Agent-E10-step60to70-4gpu`
dir was ever created; no optimizer update occurred; all training processes exited
cleanly afterward; the Wiki server, pid 1016459, was unaffected both times).

CUDA device `34000` is GPU 0's PCI bus id suffix (`00000000:34:00.0`, confirmed via
`nvidia-smi --query-gpu=index,pci.bus_id`). Different rank pairs were reported as
duplicating rank 0 across the two attempts (rank2-vs-0, rank1-vs-0, rank3-vs-0),
consistent with a nondeterministic race rather than a fixed misconfiguration.

## Ruled out

- **E1-0-introduced regression in `CUDA_VISIBLE_DEVICES`**: ruled out.
  `export CUDA_VISIBLE_DEVICES=0,1,2,3` in `code/run_e10_step60to70.sh` is
  byte-identical to `Agent/train/mhqa_agent/rl/train_dapo_mhqa_agent_wiki.sh:30`.
- **Stale Ray session state from the two failed attempts**: ruled out. Between
  attempts 1 and 2, `pgrep -af "ray::"` / `ps aux | grep raylet` showed zero
  leftover Ray processes; `nvidia-smi --query-compute-apps` showed only the
  Wiki server (pid 1016459) holding GPU memory. Each attempt starts a fresh Ray
  cluster (single-node `ray.init()` in `verl/verl/trainer/main_ppo.py`), not a
  pre-existing one (`ray status` reports "Could not find any running Ray instance"
  when checked between attempts).
- **`RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES`-type env leakage**: ruled out.
  `env | grep -i RAY_` in the launch shell is empty; `verl/verl/utils/ray_utils.py`'s
  `ray_noset_visible_devices()` reads exactly this var and it is unset, so Ray's
  normal per-actor `CUDA_VISIBLE_DEVICES` remapping (one physical GPU per
  `WorkerDict` actor, via placement-group bundling in
  `verl/verl/single_controller/ray/base.py:_init_with_resource_pool`,
  `max_colocate_count=1`) is active as designed -- this is the same code path the
  original (untouched) `train_dapo_mhqa_agent_wiki.sh` run also uses, and that run
  completed step 0 -> step 200 successfully.
- **Multi-node NCCL network plugin gaps** (a cause seen in some upstream reports,
  e.g. lasgroup/SDPO#39): not applicable -- this is a single-node, 4-GPU,
  PIX-interconnected (`nvidia-smi topo -m`) box.

## Root cause (per upstream reports)

This is a known, previously-reported Ray + FSDP + vLLM/SGLang NCCL race:
- verl upstream issue `verl-project/verl#1096` -- identical
  "Duplicate GPU detected: rank 9 and rank 1 both on CUDA device 34000" during
  the same `fsdp_workers.py` `_build_model_optimizer` step, on an unrelated
  verl DAPO recipe.
- Ray upstream issue `ray-project/ray#48012` -- same NCCL error traced to a
  mismatch between how Ray assigns/remaps `CUDA_VISIBLE_DEVICES` per actor and
  how the NCCL communicator (using the cuMem/VMM allocator path) performs its
  bootstrap-time duplicate-GPU check via PCI bus id.
- vLLM's own troubleshooting docs describe forcing `NCCL_CUMEM_ENABLE=0` for
  exactly this class of "Duplicate GPU detected" failure when an
  SGLang/vLLM-style inference engine and a separate FSDP/training NCCL
  communicator coexist in the same process group lifecycle.

This is environmental/infra nondeterminism (a race in NCCL's device-dedup check
under the cuMem allocator when combined with SGLang's own NCCL usage in the same
Ray worker process), not something caused by any E1-0 code, config, or GPU-visibility
change -- E1-0 did not alter GPU count, placement strategy, `max_colocate_count`,
or any Ray/NCCL-related setting from the original script.

## Fix applied (attempt 3)

Added one line to `code/run_e10_step60to70.sh`, immediately after
`export CUDA_VISIBLE_DEVICES=0,1,2,3`:

```
export NCCL_CUMEM_ENABLE=0
```

This disables NCCL's cuMem/VMM allocator path (a pure memory-allocation-strategy
toggle with no effect on rollout sampling, reward computation, GRPO math, or any
audited-by-task-section-7 semantics), which is the standard workaround reported
for this exact error signature. Per task section 16, this qualifies as an
"ordinary error" (environment/runtime-stability) self-fix: no optimizer update had
occurred in either failed attempt, so attempt 3 reruns within the same `RUN_DIR`
(no new attempt-subdirectory needed) with logs from attempts 1-2 preserved as
`launcher_attempt1_nccl_crash.log` / `launcher_attempt2_nccl_crash.log`.

If attempt 3 still fails with the same error, the next diagnostic step would be
running with `NCCL_DEBUG=WARN` (already default-enabled via
`verl/verl/trainer/main_ppo.py`'s `ray.init(runtime_env={"env_vars": {"NCCL_DEBUG": "WARN", ...}})`)
for a fuller trace, and if that still does not resolve it, this would become a
BLOCKER (environmental issue outside E1-0's control, potentially affecting any
launch on this box regardless of experiment).

## UPDATE: attempt 3 also failed -- second fix required (attempt 4)

Attempt 3 (`NCCL_CUMEM_ENABLE=0` alone) crashed with the **identical**
"Duplicate GPU detected" error at the identical point
(`logs/launcher_attempt3_nccl_crash.log`). This proves `NCCL_CUMEM_ENABLE=0`
alone was insufficient for this box/version combination.

Root cause refined: `verl/verl/single_controller/base/worker.py`'s
`_setup_env_cuda_visible_devices()` calls `ray_noset_visible_devices()`
(`verl/verl/utils/ray_utils.py`) to check whether Ray's automatic
per-actor `CUDA_VISIBLE_DEVICES` remapping should be bypassed in favor of
verl self-managing `torch.cuda.set_device(RAY_LOCAL_RANK)` directly. This is
verl's own documented escape hatch for exactly the Ray-remap x NCCL
duplicate-GPU-detection race described in `verl-project/verl#1096` /
`ray-project/ray#48012`. Added, alongside the existing `NCCL_CUMEM_ENABLE=0`:

```
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
```

**Result: attempt 4 succeeded in clearing the NCCL crash entirely.** The run
progressed cleanly through FSDP model construction, dataset loading,
worker/tool registration, CUDA-graph capture, and into genuine checkpoint-resume
logic (`Setting global step to 60`, `Resuming from .../global_step_60`) -- the
first attempt across all 4 to get past this blocker. See
`logs/checkpoint_load_oom_diagnosis.md` for the new (different, later-stage)
CUDA OOM error attempt 4 then hit during checkpoint loading, and its fix.

Both `NCCL_CUMEM_ENABLE` and `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES`
remain pure Ray/NCCL device-plumbing toggles with no effect on rollout
sampling, reward computation, GRPO math, or any section-7-audited semantic --
both qualify as section-16 "ordinary error" self-fixes.

## UPDATE: RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES reverted (attempt 6)

Attempt 4's NCCL fix, while clearing the FSDP/NCCL crash, was found to break a
DIFFERENT layer of GPU-visibility logic: `sglang_rollout.py`'s
`_init_distributed_env()` (~line 354) does
`torch.distributed.all_gather_object(visible_devices, os.environ["CUDA_VISIBLE_DEVICES"], tp_group)`
to compute each DP-group's SGLang engine GPU assignment, implicitly assuming
Ray's *default* per-actor remap (each actor sees exactly one distinct
physical GPU as its own "cuda:0" before this call). With
`RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`, every actor instead sees the
full unremapped `CUDA_VISIBLE_DEVICES=0,1,2,3`, so this all_gather union
collapses to the same GPU set for every DP group, and both groups' SGLang
engines requested the same physical GPU pair (`base_gpu_id=0`) -- causing an
immediate GPU-memory collision and crash:
`RuntimeError: Not enough memory. Please try to increase --mem-fraction-static`
(SGLang scheduler, both TP ranks), followed by `ray.exceptions.ActorDiedError`.
See `logs/launcher_attempt5_sglang_gpu_collision_crash.log` and
`logs/checkpoint_load_oom_diagnosis.md` (which also documents the intermediate,
now-superseded checkpoint-load OOM diagnosis and its `map_location="cpu"` fix,
kept as a harmless defensive improvement regardless).

Checked upstream: `verl-project/verl#1096`'s own comment thread shows the
**actual community-verified fix** is downgrading `ray` to `2.40.0` (3
independent +1 reactions), not `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES`.
Downgrading the shared `parallel-agent` conda environment's `ray` version was
judged OUT OF SCOPE for an E1-0 "ordinary error" self-fix: it is a
hard-to-reverse, shared-infrastructure dependency change (not scoped to this
experiment; would affect any other process using this environment), unlike
the launch-script-local env var toggles used so far.

**Decision (attempt 6): reverted `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES`**
(no longer set in `code/run_e10_step60to70.sh`), restoring the exact
`CUDA_VISIBLE_DEVICES` handling of the original, byte-identical,
200-step-successful `train_dapo_mhqa_agent_wiki.sh`. `NCCL_CUMEM_ENABLE=0` is
kept (harmless, vLLM-documented mitigation; attempt 3 showed no adverse
side effect from it, only that it alone was insufficient). Rationale for
retrying under the reverted config rather than treating this as a BLOCKER:
the NCCL "Duplicate GPU detected" crash is nondeterministic (different rank
pairs in attempts 1-3), and the original run already proves this exact
GPU/TP geometry (`GEN_TP=2`, `hybrid_engine=true`, 4 GPUs) can complete a full
200-step run without ever hitting it -- so a clean retry has a genuine chance
of succeeding, and each attempt costs only a few minutes of wall-clock before
any optimizer update occurs. If attempt 6 (and further retries) continue to
hit the NCCL race deterministically, this becomes a genuine BLOCKER requiring
either a `ray` downgrade (a decision beyond E1-0's self-fix scope) or explicit
user sign-off, and will be reported as such in FINAL_REPORT.md.

## UPDATE: attempts 6 and 7 both also failed identically -- combined fix (attempt 8)

Attempt 6 (NOSET reverted, `NCCL_CUMEM_ENABLE=0` only) crashed with the
identical NCCL "Duplicate GPU detected" error
(`logs/launcher_attempt6_nccl_crash.log`). Attempt 7 (same, plus
`CUDA_DEVICE_ORDER=PCI_BUS_ID` to test the alternative "inconsistent PCI
enumeration order" hypothesis) also crashed identically
(`logs/launcher_attempt7_nccl_crash.log`), disproving that hypothesis as a
standalone fix. This brings the total to 5/5 reproductions (attempts 1, 2, 3,
6, 7) under every configuration that does NOT set
`RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` -- confirming that setting is
not merely one workaround among several but the only mechanism, of everything
tried, that avoids Ray's internal per-actor remap race entirely (rather than
hoping the race doesn't trigger).

Per the threshold stated above, this is the point at which continuing to
retry the *same* reverted config would no longer be a good-faith self-fix
attempt. However, a full combined fix -- re-enabling NOSET=1 together with a
direct fix for the SGLang-side GPU-collision issue it exposes (rather than
either accepting the collision or reverting NOSET again) -- had not yet been
tried. That combined fix is implemented in attempt 8; see
`logs/sglang_gpu_collision_diagnosis.md` for the SGLang-side root cause and
the `code/e10_sglang_gpu_patch.py` monkeypatch. `run_e10_step60to70.sh` now
sets `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` again (dropping the
ineffective `CUDA_DEVICE_ORDER=PCI_BUS_ID`). If attempt 8 still fails, the
next diagnosis update will assess whether this constitutes a genuine BLOCKER
per task section 20 (since the only other upstream-verified fix, a `ray`
downgrade, remains judged out of scope for autonomous self-fixing).

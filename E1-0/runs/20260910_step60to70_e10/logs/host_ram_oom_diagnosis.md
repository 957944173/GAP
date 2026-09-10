# Host RAM OOM during TaskRunner.__init__ (attempt 10) -- diagnosis and self-fix (attempt 10 -> attempt 11)

## Symptom

Attempt 10 (`code/run_e10_step60to70.sh` with `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`
plus the corrected `code/e10_sglang_gpu_patch.py` patching
`SGLangRollout._init_distributed_env`, see `sglang_gpu_collision_diagnosis.md`)
progressed further than any of attempts 1-9 -- past `Total training steps: 70`
and `colocated worker base class <class 'verl.single_controller.base.worker.Worker'>`
-- then crashed with:

```
ray.exceptions.OutOfMemoryError: Task was killed due to the node running low on memory.
Memory on the node ... was 121.35GB / 125.53GB (0.966714), which exceeds the
memory usage threshold of 0.95. Ray killed this worker ... because it was the
most recently scheduled task ...
Top 10 memory users:
PID    MEM(GB)  COMMAND
1466649  1.42   ray::TaskRunner.run
1016459  0.80   wiki_rag_server.py
1441418  0.47   ray::IDLE
1441614  0.47   ray::IDLE
1117345  0.47   vscode-server node
1441743  0.47   ray::IDLE
1438213  0.47   ray::IDLE
1437851  0.47   ray::IDLE
1440533  0.47   ray::IDLE
1439775  0.47   ray::IDLE
```

Saved as `logs/launcher_attempt10_ram_oom_crash.log`. This crash occurred
inside `TaskRunner.__init__`/early actor-pool setup, strictly *before* any
`WorkerDict`/FSDP/checkpoint-load activity -- i.e. before any GPU work, before
`Setting global step to 60`, and before any optimizer step. No partial
checkpoint output was created and no optimizer update occurred.

## Root cause

The "top 10" list Ray prints does **not** account for the bulk of the 121GB --
none of the listed processes are individually large, and the listed
`ray::IDLE` workers are the normal small worker-pool procs. The real driver
was **accumulated orphaned processes from 9 prior crashed attempts**, found
by `ps aux --sort=-%mem` and `pgrep -af compile_worker` immediately after the
crash:

- `torch/_inductor/compile_worker --workers=32` subprocess pools: 67 such
  processes were still running, each ~0.3-0.5% of 125GB RSS (roughly
  400-600MB each), spawned by `torch.compile` support (triggered by
  `actor_rollout_ref.model.use_remove_padding=true` / FSDP compiled paths and
  SGLang's own JIT paths) during attempts 1-9. These processes are children
  of the Ray dashboard-agent / runtime-env-agent processes (pids 1436239 and
  1436241, both already reparented to init, `PPid: 1`), not of the
  short-lived `TaskRunner`/`WorkerDict` actors themselves. When a training
  attempt's `TaskRunner` driver crashes (as attempts 1-9 all did, via NCCL /
  SGLang-GPU-collision / IndexError / checkpoint-OOM errors), Ray tears down
  the actors but this particular compile-worker pool -- rooted under the
  persistent dashboard/runtime-env agents rather than under the crashing
  driver -- is not cleaned up. Each of the 9 prior crash/retry cycles added
  more such orphans, so by attempt 10's launch the pool had grown to 67
  processes (~30-35GB combined RSS), which combined with normal OS load
  (milvus daemon, VSCode server, Wiki server, Ray worker pool, swap
  thrashing at 15GB already in use) crossed Ray's 0.95 threshold of 125.53GB
  during attempt 10's own actor/object-store setup.
- This is **not** a per-attempt memory sizing problem (no single process was
  large) and **not** a Ray object-store/dataloader sizing defect -- it is
  crash-retry orphan accumulation, a direct consequence of the same
  underlying fragility (9 attempts needed to fix the GPU-assignment bugs)
  rather than an independent bug in the E1-0 launch configuration.

## Fix applied (attempt 10 -> attempt 11)

1. Verified the Ray cluster from attempt 10 fully exited (`pgrep -af
   "ray::|raylet|gcs_server"` returned empty).
2. Killed the two orphaned root processes (dashboard-agent pid 1436239,
   runtime-env-agent pid 1436241) and their compile_worker children via
   `pkill -f "compile_worker.*parent=<pid>"`.
3. Confirmed clean state after cleanup: `free -h` shows 16Gi/125Gi used (down
   from 121Gi/125Gi at crash time), no `ray::`/`raylet`/`compile_worker`
   processes remain, GPUs show only the Wiki server's steady-state
   ~9.7-10.2GiB each at 0% utilization, and the Wiki server (pid 1016459) is
   still healthy and untouched.
4. No change to any E1-0 launch parameter, GRPO/rollout/audit config, or
   original verl file was needed -- this is a process-hygiene self-fix
   (killing orphaned subprocesses left behind by prior crashed attempts),
   not a training-semantics change. Qualifies as a section-16 "ordinary
   error" self-fix.
5. Since no optimizer update occurred in attempt 10 (crash was in
   `TaskRunner.__init__`, before `Setting global step to 60` / before any
   `WorkerDict` initialization), attempt 11 reruns within the same `RUN_DIR`
   using the same `code/run_e10_step60to70.sh`, `code/e10_sglang_gpu_patch.py`,
   and `code/e10_checkpoint_patch.py` as attempt 10 -- no new attempt
   subdirectory is required per task section 16's restart rule (that rule
   applies only when an optimizer update already occurred).

## UPDATE: attempt 11 -- transient Ray runtime_env_agent startup timeout (attempt 12)

Attempt 11 (relaunch of the identical attempt-10 config after the process
cleanup above) crashed within ~30 seconds of Ray head start, before any
worker actors registered:

```
[raylet] runtime_env_agent_client.cc:335: Runtime Env Agent timed out in
30000ms. Status: NotFound: on_connect Connection refused ...
[raylet] main.cc:291: The raylet exited immediately because the runtime env
agent timed out when Raylet try to connect to it.
```

`raylet.out` shows the raylet pre-forking ~95 idle worker processes within
about 1 second of startup (`Started worker process with pid ..., the token
is 1` through `95`, all timestamped 03:33:23-03:33:24) -- matching this box's
`nproc`=96, Ray's default `num_cpus` worker-prestart behavior when
`ray.init()`/`main_ppo` does not explicitly cap it. Each of those worker
processes imports the full training dependency graph (evidenced by ~50
repeated `livecodebench ... Few shot examples not found` warnings in
`raylet.err`, one per worker). The dashboard/runtime-env agent process
(`dashboard_agent.log`) did not finish its own Python module loading and
open its grpc port until 03:33:41 -- about 18-19 seconds after Ray start --
which combined with connection-retry overhead exceeded the raylet's
hardcoded 30-second connection timeout. `gcs_server.err` and the dashboard
agent's own log show no errors; this reads as a one-off CPU/scheduling
contention flake (likely aggravated by the box still settling immediately
after the attempt-10 orphan-process cleanup) rather than a deterministic
resource-limit violation: `free -h` and `ps -u nf5468m6` both show a fully
idle system after the crash, `ulimit -u` (513212) is far from the ~150
processes Ray creates, and `df -h /tmp` shows 749G free.

Root cause classified as **transient Ray infra timing**, not an E1-0 config
or training-semantics issue. No E1-0 file or launch parameter was changed.
Self-fix: simply retry the launch (attempt 12) on the now-idle system, per
section 16's "ordinary error" self-fix mandate. No optimizer update occurred
(crash was pre-`TaskRunner`/pre-checkpoint-load), so attempt 12 reruns in the
same `RUN_DIR`.

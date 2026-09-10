# Attempt 18 diagnosis: checkpoint-load hang RESOLVED, but new stall found immediately downstream

## Summary

Attempt 18 (driver pid 2244126, launched ~15:45, killed ~17:42 after ~2h with
zero forward progress past checkpoint-load) is the first attempt where the
`e10_checkpoint_patch.py` diagnostic instrumentation (added after attempts
14/16/17) actually fired and produced positive evidence. It shows that the
long-standing checkpoint-load hang (attempts 14, 16, 17 -- silence after
"Resuming from .../global_step_60") **did not recur**: all 4 ranks completed
the full `load_checkpoint` sequence, including `torch.load(model)` ->
`model.load_state_dict()` -> `torch.load(optim)` -> `optimizer.load_state_dict()`
-> exiting `get_fsdp_state_ctx` -> the final `torch.distributed.barrier()`,
with rank 0's `load_checkpoint` fully returning at t=1789026629.6 (~76s after
entry). This is the best evidence yet that the previously-hypothesized
FSDP-collective hang inside `load_state_dict()` either does not deterministically
reproduce, or was tied to a since-resolved environmental condition. No code
fix was applied to `e10_checkpoint_patch.py` between attempt 17 and 18 (only
the passive diagnostic prints were added, prior to attempt 18's launch) --
so this is evidence the hang is non-deterministic, not evidence a fix was made.

However, attempt 18 then stalled at the **very next step**: `_validate()`,
architecturally the immediate next call in `fit()` after `_load_checkpoint()`
returns (`ray_trainer.py` ~line 1557-1585, guarded by `val_before_train=true`
which is set for this run). No forward progress was observed for ~2 hours
after checkpoint-load completed, at which point the attempt was terminated.

## Evidence for the new stall (post-checkpoint-load, pre/in-validation)

1. **Launcher log frozen at 591 lines**, last write at 15:50:29 (t=1789026629.6,
   matching rank 0's `load_checkpoint: fully returned` line exactly). No new
   lines appeared over the following ~2 hours of wall-clock time, confirmed via
   repeated `wc -l` / `stat` checks showing a monotonically growing
   "seconds since last write" gap (320s, 781s, 1159s, ..., up to ~6900s at
   termination).

2. **`_validate()`'s unconditional driver-side print never appeared.**
   `ray_trainer.py`'s `_validate()` (lines ~794-910) does
   `print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")`
   synchronously, on the driver, before dispatching each of 29 val-batch
   `generate_sequences` calls (val set: 3610 examples / batch_size 128).
   `PYTHONUNBUFFERED=1` is set (`run_e10_step60to70.sh` line 190), so this
   print should appear in the launcher log within seconds of `_validate()`
   starting. `grep -c "test_gen_batch meta info"` on the full log: **0**.
   This means there is no positive evidence `_validate()` ever began
   processing its first batch.

3. **All 4 WorkerDict `.out`/`.err` files froze at checkpoint-load completion.**
   Direct per-pid files under `/tmp/ray/session_latest/logs/`
   (`worker-*-2247667.{out,err}`, `-2249281.*`, `-2249283.*`, `-2249285.*`)
   all stopped growing at 15:48-15:50 and never grew again (confirmed via
   `ls -la` mtime checks at 17:42, ~1h50m later). File sizes (1686-9807 bytes)
   are consistent with only pre-checkpoint-load startup output (CUDA graph
   capture, FSDP deprecation warnings, tool-schema dumps, NCCL version print)
   plus the checkpoint-load diagnostic lines -- i.e. zero output attributable
   to `generate_sequences` or any post-checkpoint-load rollout activity.

4. **Ray's `debug_state.txt` showed 0 executing / 0 waiting tasks** on the
   node at the time checked, and the TaskRunner driver's own core-worker log
   (`python-core-worker-...2244126.log`), while still growing in size/mtime,
   contained (after filtering boilerplate) only repeating periodic Ray
   telemetry ("Task execution event stats", "IO Service Stats"); its
   "total number of task attempts sent" counter (39) was static across a
   30s re-check -- no new Ray task was being dispatched from the driver.

5. **GPU utilization read a flat, unbroken 0% on all 4 GPUs** for the entire
   ~2-hour stall window (checked via `nvidia-smi` repeatedly), with memory
   pinned at the post-load footprint. This is inconsistent with genuine
   prefill/decode activity, which produces visible utilization spikes even
   under light load.

6. **Sustained SGLang-scheduler CPU usage (~95-115% of one core per rank,
   dozens of checks over 90+ minutes) was initially misread as evidence of
   ongoing generation work.** Reading SGLang's own source
   (`sglang/srt/managers/scheduler.py`, `recv_requests()` ~lines 802-871)
   shows the scheduler's main loop polls its inbound ZMQ queue
   non-blockingly (`recv_pyobj(zmq.NOBLOCK)` in a `while True` /
   `except zmq.ZMQError: break`) with no idle backoff sleep in this code
   path. This means a scheduler that is completely idle -- waiting for a
   request that will never arrive -- can sustain ~100% single-core CPU
   indefinitely purely from busy-polling an empty queue. The observed CPU
   pattern is therefore **not diagnostic**: it looks identical whether the
   scheduler is genuinely processing requests or sitting idle. Combined with
   items 1-5 above (zero GPU util, zero validate print, frozen per-rank
   output, no new Ray tasks dispatched), the busy-poll-while-idle
   interpretation is far better supported by the totality of evidence than
   the "genuinely slow validation" interpretation this investigation
   initially favored.

## Interpretation (INFERRED, not proven)

The stall most likely sits between `load_checkpoint: fully returned`
(confirmed complete on the driver, rank 0) and the first dispatched
`generate_sequences` RPC inside `_validate()`. Candidate locations not yet
distinguished from each other:
- The FSDP -> SGLang weight-sync / resharding handoff
  (`rollout_sharding_manager` context entry in `fsdp_workers.py`'s
  `generate_sequences`, ~line 644/656) that must occur between a freshly
  loaded FSDP state and the SGLang engine being ready to serve requests.
- Driver-side setup inside `_validate()` before the first
  `test_gen_batch.meta_info` print (unlikely, since that print is the very
  first statement using `test_gen_batch`, but the surrounding
  `val_dataloader` iteration/`DataProto` construction has not been fully
  ruled out).
- The RPC dispatch path itself (driver never sends the request, vs. worker
  receives it and stalls before touching SGLang).

No stack trace could be obtained (`ptrace_scope=1`, no passwordless sudo --
same confirmed environment limitation as prior attempts). This is a NEW,
DISTINCT problem from the checkpoint-load hang, which attempt 18 shows is
very likely no longer the blocking issue.

## Action taken

No optimizer update occurred (checkpoint-load only; never reached the
training loop, let alone an optimizer step). Per task section 16, this is
an ordinary pre-training-loop failure with no mixed-state risk -- cleanly
terminated in the same run directory, no new attempt subdirectory needed:

    kill -TERM 2238979   # original launcher bash (already exited on its own)
    kill -TERM 2244126 2238985   # TaskRunner driver + main_ppo entrypoint
    # ray stop --force attempted but `ray` was not on PATH in that shell;
    # process list confirmed independently that the kill cascade already
    # terminated every Ray/SGLang/WorkerDict process

Verified after termination:
- `ps aux | grep -E "main_ppo|sglang|ray::|WorkerDict"` returns empty.
- GPU memory returned to Wiki-server-only baseline (~9.7-10.2GB/GPU on all 4).
- Wiki server confirmed still healthy (`POST /retrieve` -> HTTP 200).
- No partial checkpoint exists under
  `experiments/DAPO-GAP3B-MHQA-Agent-E10-step60to70-4gpu/`.

## Status update to checkpoint_load_hang_investigation.md

The checkpoint-load hang (attempts 14/16/17's shared signature) should now be
considered **likely resolved / non-deterministic and not currently
reproducing** -- attempt 18's diagnostics show a full, clean pass through
`load_checkpoint` on all 4 ranks. The active, unresolved problem going
forward is the new post-checkpoint-load / pre-or-in-`_validate()` stall
documented here, which needs its own diagnostic instrumentation (e.g. a
print immediately before/after the `rollout_sharding_manager` context enter/
exit in `fsdp_workers.py`'s `generate_sequences`, and immediately before the
`val_dataloader` loop starts in `_validate()`) before the next attempt, to
pinpoint the exact stall location rather than inferring it indirectly again.

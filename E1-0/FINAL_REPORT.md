# E1-0 FINAL REPORT — summary

**E1-0 STATUS: PASS_WITH_CAVEATS** · 2026-09-10

| | |
|---|---|
| RUN_DIR | `E1-0/runs/20260910_step60to70_e10` |
| start / end step | **60 → 70** (exactly 10 global steps, steps 61..70 in the log) |
| step70 checkpoint | `experiments/DAPO-GAP3B-MHQA-Agent-E10-step60to70-4gpu/global_step_70/` (tracker = 70) |
| Wiki | **reused** existing healthy server (`127.0.0.1:8008`), never started/stopped by E1-0 |
| reward identity | **mismatch_count = 0, max_abs_diff = 0.0, equality_rate = 100%** over 147,200 rollouts |
| group verification | 18,400 groups, **P(group_size == 8) = 100%**, group key = `uid` (= dataset index) |
| signal density | **0.52%** of groups (95/18,400): ≥2 correct rollouts *and* correct-cost variation |
| XML vs structured | **100.00% exact match** (0 mismatches), correct-rollout ranking changed in **0/8,456** groups |
| offline λ replay | λ = 0.05 / 0.1 / 0.2: no NaN/Inf, correct > incorrect always, 95/95 active groups strictly ordered |
| recommended λ | **0.05** |
| original repo files | 24/24 sha256 unchanged (pre- and post-run) |
| E1-A | may proceed (with the powering caveat below) |

**Full report (all 25 required questions, control-variable table, evidence tags):
`E1-0/runs/20260910_step60to70_e10/FINAL_REPORT.md`.**

## What E1-0 established

1. The step60 → step70 continuation under the ORIGINAL GAP reward is valid: the
   reward is bit-for-bit the original EM reward on all 147,200 audited rollouts,
   `score == em`, no shaping, and GRPO/filter/optimizer/LR/sampling all match
   the original run's settings (only `ray_init.num_cpus=16` and the intended
   audit/output/stop-step changes differ — all documented).
2. Full-state resume is proven, not assumed: the LR scheduler's `last_epoch`
   continues 60 → 70 and the RNG states are restored.
3. The n=8 group structure is confirmed at runtime, and the group key is the
   dataset row index (`uid`).
4. The original group filter keeps only **1.85%** of groups (the 340 mixed
   ones), because the checkpoint policy is nearly deterministic per prompt
   (53.7% all-wrong, 44.5% all-correct). This is why each global step needs
   8–14 generation batches (audit call counts per step exactly match the log's
   `train/num_gen_batches`).
5. The efficiency signal is real but sparse: 84 all-correct groups (1.03% of
   all-correct, 0.46% of all groups) have cost variation, so a shaped reward
   could newly retain **84 groups ≈ +0.73 prompts per 160-prompt generation
   batch** on top of the ≈2.96 mixed groups.
6. `logical_search_batches` (XML/logical block count) and
   `structured_search_rounds` are **identical on every rollout** here, and the
   correct-rollout ranking is unchanged in every group ⇒ the simpler
   **logical-XML-batch** cost is recommended for E1-A.
7. Reward-hacking audit: zero-search is not exploited (all 24 zero-search
   rollouts were wrong); a parallel-query explosion exists (max 22 queries in
   one block, 45 rollouts >8) but a *block-count* cost does not reward it —
   a query-count cost would; no malformed-search or metadata mismatches.
8. Root cause of the long-standing silent hangs (attempts 14–18 and this
   session's 19a) was found and fixed: `e10_sglang_gpu_patch.py` called
   `monkey_patch_torch_reductions()` at import time in **every** process,
   including the driver, which deadlocks fork-based `StatefulDataLoader`
   workers (CUDA work inside a forked child). The call is now deferred to the
   WorkerDict actor that actually needs it. Details:
   `analysis/root_cause_dataloader_fork.md`.

## Recommended E1-A design

```
cost        = logical_search_batches  (XML logical block count == structured round count here)
lambda      = 0.05
reward      = R_i = A_i * (1 + 0.05 * E_i)
              E_i = (Cmax - C_i)/(Cmax - Cmin) for correct rollouts of a group with
                    correct_count >= 2 and correct-cost variation, else 0
signal      ~ 0.52% of pre-filter groups (84 new groups; ~+0.73 retained prompts / 160-prompt batch)
risks       sparse signal (power E1-A long enough); shortening pressure on multi-round
            questions; use block-count not query-count (parallel-query explosion exists);
            all-wrong groups (53.7%) carry no efficiency signal at all
```

**E1-0 ends here — E1-A was not started.**

## Caveats

* Signal density is low (0.52%); the shaped reward is clean but sparse.
* This session's first valid attempt is 19b; 19a hung on the fork/CUDA deadlock
  above (now fixed), and the earlier attempts share that same root cause.

## Key artifacts

* `E1-0/runs/20260910_step60to70_e10/FINAL_REPORT.md` — full report
* `E1-0/runs/20260910_step60to70_e10/analysis/` — group statistics, XML-vs-structured,
  reward identity, filter simulation, offline replay, post-train integrity, root cause
* `E1-0/runs/20260910_step60to70_e10/audit/` — 147,200 rollout records, 18,400 group
  records, 115 call records, summaries
* `E1-0/runs/20260910_step60to70_e10/logs/` — training/launcher logs, every attempt log,
  the stall watchdog log and the SIGUSR1 stack dumps
* `E1-0/Current_Progress.md` — session history and handoff notes

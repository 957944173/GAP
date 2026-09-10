# E1-0 FINAL REPORT (run `20260910_step60to70_e10`)

**Experiment**: E1-0 — resume the original GAP RL run from `global_step_60` and
train **exactly 10 further global steps to step70** under the **ORIGINAL GAP
reward** (pure EM), adding **passive audit only**.

**RUN_DIR**: `/data01/wyy/Graph-Agent-Planning/E1-0/runs/20260910_step60to70_e10`
**Date**: 2026-09-10 (run window ~20:14 – 23:14 CST, ≈3 h 0 m wall clock)
**Status**: **E1-0 STATUS: PASS_WITH_CAVEATS** (see §1 for the caveat list)

Evidence tags used throughout: **VERIFIED** (measured/derived from artifacts in
this RUN_DIR), **OBSERVED** (directly seen in logs/stacks), **INFERRED**
(reasoned, not directly proven), **BLOCKER** (none remain).

---

## 0. Executive summary

| Item | Result |
|---|---|
| start / end global step | **60 → 70** (exactly 10 steps, steps 61..70 all present in the log) — VERIFIED |
| step70 checkpoint | `experiments/DAPO-GAP3B-MHQA-Agent-E10-step60to70-4gpu/global_step_70/` (model+optimizer+extra_state ×4 ranks + `data.pt`, 36 GB, tracker = `70`) — VERIFIED |
| optimizer/scheduler resume | full-state, not weights-only: `lr_scheduler.last_epoch` continues **60 → 70**, RNG state present — VERIFIED |
| reward identity (task §0) | over **147,200** audited rollouts: `mismatch_count = 0`, `max_abs_diff = 0.0`, `equality_rate = 100%` — VERIFIED |
| n=8 groups at runtime | **P(group_size == 8) = 100.00%** of 18,400 groups — VERIFIED |
| group key | `uid` = the original dataset row index (`extra_info["index"]`), as set at `ray_trainer.py:1734`; verified by `extra_info_index == uid` for every group — VERIFIED |
| logical XML batch vs structured search round | **exact match 100.00%** (0 mismatches in 147,200), correct-rollout group ranking changed in **0 / 8,456** groups — VERIFIED |
| offline λ replay | λ = 0.05 / 0.10 / 0.20 all NaN/Inf-free, `correct > incorrect` always, 95/95 active groups strictly ordered; **λ = 0.05 recommended** — VERIFIED |
| Wiki server | **reused** (never started or stopped by E1-0); 0 tool failures in the whole run — VERIFIED |
| original repo files | 24/24 sha256 unchanged before **and** after the run — VERIFIED |
| E1-0 / E1-A gate | E1-0 **PASS_WITH_CAVEATS**; E1-A may proceed, see §21 |

The single caveat that prevents a clean `PASS` is **signal density**: only
**0.52%** of pre-filter groups (95 / 18,400) have ≥2 correct rollouts *and*
correct-cost variation, so the efficiency term can only revive **84**
all-correct groups (≈0.73 extra retained prompts per 160-prompt generation
batch). Everything else is a clean, control-variable-matched continuation.
A second, process-level caveat is documented in §17: the first 19 attempts
failed because of an E1-0-introduced fork/CUDA deadlock, which is now fixed and
root-caused, but it means this run is the first valid attempt rather than a
continuation of earlier ones.

---

## 1. Caveats / deviations (all explained)

1. **Signal density is low** (0.52% of groups). Not a defect of this run; it is
   the measurement E1-0 was designed to make. It is the main input to the E1-A
   design decision.
2. **`ray_init.num_cpus=16`** (original: unset ⇒ 96). Pure Ray idle-worker-pool
   cap, added after attempts 10/12 reproducibly hit ~121.9/125.5 GB host RAM.
   Not a training-semantics parameter; see §20.
3. **`data.gen_batch_size` is fixed at 160** instead of driven by
   `gen_bs_supervisor.py`. 160 is the value the original supervisor had settled
   on at step 20 and never changed again through step 200
   (`supervisor.log`: `step=20 128 -> 160`, no later adjustment), so step 60→70
   is matched. E1-0 needs only 10 steps and deliberately bypasses the reactive
   supervisor.
4. **HF-merge tail removed** from the E1-0 launcher (the original script merges
   the final FSDP checkpoint to HF at the end). E1-0 does not need an HF model;
   the FSDP checkpoint + manifest is the audit evidence. This changes nothing
   about training.
5. **`trainer.save_freq = test_freq = 10`** matches what the original supervisor
   actually passed (`--save-freq` default 10; `env["TEST_FREQ"] = save_freq`),
   not the base script's textual default of 20.
6. **Reward-manager/scorer selection differs by design** (`e10_batch_audit` /
   `compute_score_em_batch_e10`), and is numerically identical to the original
   (see §5).

---

## 2. Answers to the 25 required questions

**1. Did E1-0 start from the correct step60? — VERIFIED YES.**
`Setting global step to 60` in the log; `resume_mode=resume_path`,
`resume_from_path=…/DAPO-GAP3B-MHQA-Agent-4gpu/global_step_60`; the driver
wrapper recorded `_load_checkpoint: returned to driver (global_steps=60)`.
There is exactly one `global_step_60` directory in the repo, and it is the one
the original supervisor's own `latest_checkpointed_iteration.txt` state pointed
at. See `checkpoints_manifest/step60_source.md`.

**2. Exactly 10 steps to step70? — VERIFIED YES.**
`logs/training.log` contains `step:60 … step:70` (all of 61..70 present),
`Total training steps: 70`, `is_last_step` fired at 70, final validation ran,
and no `step:71` exists. The tracker file reads `70`.

**3. Real optimizer/scheduler resume? — VERIFIED YES (not weights-only).**
`extra_state_world_size_4_rank_0.pt` contains `{lr_scheduler, rng}`; its
`lr_scheduler.last_epoch` is **60 at step60 and 70 at step70**, i.e. the
scheduler continued for 10 more steps instead of restarting at 0. RNG substates
(`cpu/numpy/random/cuda`) are present in both. The run also used
`actor.checkpoint.save_contents=['model','optimizer','extra']` and
`FSDPCheckpointManager.load_checkpoint` executed
`torch.load(optim) → optimizer.load_state_dict()` (checkpoint diagnostics in
the log). Model shard bytes differ between step60 and step70 ⇒ real updates.

**4. Per-rollout reward identity? — VERIFIED IDENTICAL.**
`reward_e10` (the value actually written to `reward_tensor`, i.e. what GRPO
consumes) vs `reward_original` (independent re-run of the unmodified
`mhqa_train.compute_score_em_batch` on the same decoded prompt/response/GT):

| metric | value |
|---|---|
| compared | 147,200 |
| mismatch_count | **0** |
| max_abs_diff | **0.0** |
| equality_rate | **100.000%** |
| `score == em` mismatches | 0 |
| rollouts with no parseable `<answer>` | 360 (0.24%), all scored 0 by the original rule |

**5. n=8 validated at runtime? — VERIFIED YES.** 18,400 audited groups, every
one of size 8 (`P(group_size==8) = 100.00%`), and the manager's own group
records agree (`groups.jsonl`).

**6. Group key? — VERIFIED `uid`** (= `extra_info["index"]`, the dataset row
index). For every group, all 8 members share `uid` and `extra_info_index`, and
`rollout_pos_in_group` spans 0..7.

**7. P(group size = 8)? — 100.00%** (18,400 / 18,400).

**8. correct_count distribution?** (18,400 pre-filter groups)

| correct_count | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| groups | 9,876 | 68 | 45 | 40 | 40 | 45 | 33 | 69 | 8,184 |
| share | 53.67% | 0.37% | 0.24% | 0.22% | 0.22% | 0.24% | 0.18% | 0.38% | 44.48% |

The policy is close to deterministic per prompt: **53.7% all-wrong, 44.5%
all-correct, only 1.85% mixed**.

**9. P(correct_count ≥ 2)? — 45.96%** (8,456 / 18,400) — but this is dominated
by all-correct groups; see Q10.

**10. Fraction of ≥2-correct groups with efficiency variation? — 1.12%**
(95 / 8,456). Broken down: of the 8,184 all-correct groups, 84 (1.03%) have
correct-cost variation; of the 340 mixed groups, 11 have variation.

**11. XML logical batch vs structured round agreement? — 100.00%.**
`both_valid = 147,200`, `exact_match = 147,200`, `mean_abs_difference = 0.0`,
`mismatch_count = 0`. `mismatch_cases.jsonl` is therefore empty (0 bytes).
Note on definitions (see §6): in this repo the assistant's `<wiki_search>`
blocks are re-rendered into `<tool_call>` blocks, so the XML/logical batch count
is reconstructed from the parser's block index in
`tool_call_sequence[].call_id`; an independent response-text count
(`resp_wiki_tool_call_blocks`) matched it exactly on every rollout.

**12. Do the two costs change correct-rollout ranking? — NO.**
For all 8,456 groups with ≥2 correct rollouts, average-rank vectors by
`logical_search_batches` and by `structured_search_rounds` are identical:
`ranking_changed_group_count = 0`, rate 0.00%.

**13. Recommended cost for E1-A? — logical search batch (XML/logical block
count)**, i.e. "number of assistant search rounds = number of distinct
`(assistant turn, <wiki_search> block)` pairs". Evidence: exact alignment is
100% (≫ the 98% threshold), correct-group ranking is unchanged in every group,
and it needs no extra structured metadata. See §7.

**14. How many all-correct groups? — 8,184 (44.48%).**

**15. All-correct with efficiency variation? — 84 groups (1.03% of all-correct,
0.46% of all groups).**

**16. How many of those would the ORIGINAL filter delete? — all 8,184
all-correct groups are deleted** (plus all 9,876 all-wrong ones). The original
rule `kept iff std(group reward) > 0 or size == 1` keeps only the 340 mixed
groups = **1.85%** of groups. This is why the run needs 8–14 generation batches
per global step to accumulate 32 kept prompts (audit call counts per step
61..70: 13,14,13,9,13,13,12,8,10,10 — exactly the log's
`train/num_gen_batches` values).

**17. How many groups could shaped reward re-activate? — 84** (the all-correct
groups with cost variation), identical for λ = 0.05/0.10/0.20 because the
*existence* of variance, not its magnitude, drives the filter. All-wrong groups
stay at reward 0 and are never revived. In throughput terms: ≈0.73 extra
retained prompts per 160-prompt generation batch, on top of the ≈2.96
mixed-group prompts per batch (+25%).

**18. λ = 0.05 / 0.10 / 0.20 offline replay?** (real
`compute_grpo_outcome_advantage`, `norm_adv_by_std_in_grpo=True`, CPU only, no
GPU run — task §14)

| λ | reward range | all finite | correct > incorrect always | active groups | within-group efficient−inefficient advantage | groups efficient strictly > inefficient | all-correct group max abs advantage |
|---|---|---|---|---|---|---|---|
| 0.00 (baseline EM) | [0, 1] | yes | yes | 95 | 0.000 | 0 / 95 | 0.0000 |
| 0.05 | [0, 1.05] | yes | yes | 95 | **+2.077** | **95 / 95** | 2.4747 |
| 0.10 | [0, 1.10] | yes | yes | 95 | +2.089 | 95 / 95 | 2.4748 |
| 0.20 | [0, 1.20] | yes | yes | 95 | +2.111 | 95 / 95 | 2.4748 |

No NaN/Inf anywhere. Mixed-correctness group advantage std is unchanged
(0.9354) because shaping only touches correct rollouts of already-variable
groups; the visible effect is that previously **zero-variance all-correct
groups become trainable** (max |advantage| 0 → 2.47).

**19. Recommended λ for E1-A? — λ = 0.05.** It is the smallest tested value
that fully activates every active group; larger λ adds no newly-retained groups
and only widens the reward range (more distribution shift / off-policy risk).
Because GRPO divides by the group std, the effective gradient scale is largely
λ-insensitive, so taking the smallest λ is the conservative choice.

**20. Reward-hacking risks found?**

| risk | finding |
|---|---|
| zero-search | **not exploited**: 24/147,200 rollouts (0.02%) used no search, and **all 24 were wrong**; `P(cost==0 | correct) = 0.00%` |
| parallel-query explosion | **present but not rewarded by a block-count cost**: max parallel factor = 22 queries in a single block; 45 rollouts (0.03%) > 8, 280 (0.19%) > 4. A *query-count* cost would reward this; a *logical-block* cost does not — an argument for the recommended cost |
| malformed search | none material: 99.7% of trajectories ended on the normal `</answer>` stop; 419 hit max_turns(8), 24 hit response length, 16 stopped on an over-long `</wiki_search>` context |
| metadata mismatch | none: `metadata_valid=false` count is 0 over all 147,200 rollouts; every structured `tool_call_sequence` was present and produced a block index |
| perverse cost/correctness link | more search correlates with **lower** correctness (P(correct): 49.5% at cost 1, 42.8% at 2, 14.7% at ≥3; by query count 50.7%/48.3%/29.0%/20.5%/11.4% for 1/2/3/4/5+). So an efficiency bonus is aligned with correctness on this checkpoint — but it also means aggressive shortening pressure could hurt the multi-round questions |

**21. Wiki service reused or started? — REUSED.**
`wiki/wiki_status.json`: `reused=true`, `started_this_run=false`, pid 1016459
(initially), `http://127.0.0.1:8008/retrieve`, functional probe with a real
query returned genuine Wikipedia passages (HTTP 200). E1-0 set
`WIKI_SERVER_AUTOSTART=0` and never started, stopped or killed a server. At the
end of the run the server is still up (`POST /retrieve` → 200) and 0 wiki tool
failures appear in the training log.

**22. Any file overwritten? — NO.**
New identifiers everywhere: experiment `DAPO-GAP3B-MHQA-Agent-E10-step60to70-4gpu`,
log `verl/logs/<that name>.log`, RUN_DIR `E1-0/runs/20260910_step60to70_e10`.
`trainer.logger=['console']`, W&B disabled — no experiment tracker run was
created. The original experiment dir (step60/120/180/200) was only read.
Rejected/failed attempts were preserved under distinct names
(`launcher_attempt19a_hung_dataloader_fork.log`, etc.), never overwritten.

**23. Original code intact? — VERIFIED YES.**
`environment/original_files_reverify_after_run.txt`: 24/24 sha256 match the
pre-run manifest. Zero lines changed in any original file; all E1-0 logic is in
new files under the RUN_DIR.

**24. Which E1-0 files were changed/created?** See §18. Summary: new
`e10_scorer.py`, `e10_reward_manager.py`, `e10_checkpoint_patch.py`,
`e10_sglang_gpu_patch.py`, `e10_hang_diagnostics.py`, `e10_diag_patch.py`,
`e10_state.py`, `e10_trainer_patch.py`, `e10_stall_watchdog.py`,
`e10_selftest.py`, `pythonpath_e10/sitecustomize.py`,
`run_e10_step60to70.sh`, plus analysis scripts and this report. No original
file was edited.

**25. Is E1-0 ready for E1-A? — YES, with the caveat that E1-A must be powered
for a low base rate.** E1-0 `PASS_WITH_CAVEATS`. The reward-identity control is
airtight, n=8 groups are confirmed, the cost definition is unambiguous
(100% XML ≡ structured), and the offline replay shows the shaped reward
separates efficient from inefficient correct rollouts in 95/95 active groups
with no NaNs. See §21.

---

## 3. Run configuration (resolved)

`configs/resolved_training_config.yaml`, `configs/launch_command.txt`.
Launcher: `code/run_e10_step60to70.sh` (a new file; the original
`Agent/train/mhqa_agent/rl/train_dapo_mhqa_agent_wiki.sh` is untouched).

Mechanically-diffed against the original supervisor's Hyperparameter set: the
only CLI differences are `ray_init.num_cpus=16`,
`trainer.resume_mode=resume_path`, `trainer.resume_from_path=<step60>`,
`reward_model.reward_manager=e10_batch_audit`, and the E1-0 scorer path/name.
`STEPS=70`, `SAVE_FREQ=TEST_FREQ=10`, `GEN_BS=160`; everything else is a
byte-for-byte copy.

---

## 4. Control-variable check (task §21)

| | Original GAP | E1-0 | same? |
|---|---|---|---|
| Starting checkpoint | resume from a `global_step_N` in `experiments/DAPO-GAP3B-MHQA-Agent-4gpu` | `…/global_step_60` (same dir) | ✔ (matched — E1-0 fixes it at 60) |
| Dataset | `GAP-RL-16w.cleaned.parquet` (169,595 rows) | same file, sha256 identical | ✔ |
| Prompt | dataset `prompt` column, unchanged | same | ✔ |
| Rollout n | 8 | 8 | ✔ |
| Seed | `data.seed` default 1 + restored dataloader RNG state | same (state restored from step60 `data.pt`) | ✔ |
| Sampling params | temperature 1.0 + ppo_trainer.yaml defaults | same CLI, same defaults | ✔ |
| Tool config | `wiki_rag_config.yaml` (wiki_search) | same file, sha256 identical | ✔ |
| Wiki server | reused healthy `127.0.0.1:8008` | reused, never restarted | ✔ |
| Reward correctness | EM = `compute_score_em`, `score == em` | identical (independently verified per rollout) | ✔ |
| GRPO estimator | `algorithm.adv_estimator=grpo` | same | ✔ |
| Group filtering | `filter_groups.enable=true`, std>0 rule | trainer untouched; offline simulation only | ✔ |
| LR | 1e-6, warmup 3 | same | ✔ |
| Optimizer | default AdamW, `weight_decay=0.1` | same; optimizer state resumed | ✔ |
| Scheduler | warmup+constant, `last_epoch` continued | `last_epoch` 60→70 | ✔ |
| Batch sizes | train 32 / mini 32 / gen 160 (steady state) / micro 2 / logprob 8 | identical | ✔ |
| Max turns | 8 | 8 | ✔ |
| Max response length | 8192 (prompt 2048) | identical | ✔ |
| **Reward manager** | `batch` | `e10_batch_audit` (subclass of `batch`, audit-only) | intended difference |
| **Output path / experiment name** | `DAPO-GAP3B-MHQA-Agent-4gpu` | `DAPO-GAP3B-MHQA-Agent-E10-step60to70-4gpu` | intended difference |
| **Stop step** | 200 | 70 | intended difference |
| **ray_init.num_cpus** | unset (96) | 16 | documented deviation (RAM) |
| **HF merge tail** | yes | no | documented deviation (not needed) |

---

## 5. Reward identity (task §0) — detail

Definitions: `reward_e10` = `reward_tensor[i].sum()` as written by the E1-0
manager (what GRPO consumes); `reward_original` = independent call of the
**unmodified** `verl/verl/utils/reward_score/mhqa_train.py:compute_score_em_batch`
on the same decoded prompt/response/ground-truth; `reward_score_extra` /
`em` = the E1-0 scorer's dict fields forwarded through `reward_extra_info`.

```
mismatch_count = 0        max_abs_diff = 0.0        equality_rate = 100.000%   (n = 147,200)
score == em mismatches = 0
```

The E1-0 scorer asserts `score == em` per item and adds no shaped term; the
manager additionally re-derives the original score independently at audit time.
All three agree bit-for-bit. Full table: `analysis/reward_identity.md`; raw
evidence: `audit/rollouts_pid2552655.jsonl`, `audit/calls_pid2552655.jsonl`
(per-call `identity_mismatch=0`, `identity_max_abs_diff=0.0`).

---

## 6. Important schema finding: what "XML batch" means in this repo

The task defines `logical_search_batches` as the number of non-empty
`<wiki_search>…</wiki_search>` blocks in the response. **In this repository that
string never appears in the response or in `messages`:**

* `add_assistant_message` (`verl/verl/workers/rollout/schemas.py:171-181`) stores
  the parser's tool calls in the structured `tool_calls` field and stores
  `content = normed_content`, i.e. the assistant text *with the tag stripped*;
* the Qwen chat template then re-renders the turn as
  `<tool_call>{"name": "wiki_search", "arguments": …}</tool_call>` and that is
  what lands in `response_ids` (confirmed by rendering the real tokenizer, and
  empirically: `resp_wiki_search_tag_count = 0` on all 147,200 rollouts while
  `resp_wiki_tool_call_blocks` matched `structured_search_calls` exactly).

Counting the literal tag (as a first E1-0 draft did, over assistant
`messages[].content`) yields **0 for every rollout**, which would have made the
whole §12 analysis vacuous. E1-0 therefore defines, and cross-checks, two
faithful measures:

* `logical_search_batches` = # distinct `(assistant turn, block index)` pairs,
  the block index being recovered from the XML parser's own
  `call_<tool>_<block_i>_<query_j>` id carried in
  `detailed_tool_metrics.tool_call_sequence[].call_id` — i.e. exactly the
  number of `<wiki_search>` blocks the model emitted;
* `structured_search_rounds` = # distinct assistant turns that issued ≥1
  wiki_search call (`turn` field);
* independent text-side controls `resp_wiki_tool_call_blocks`,
  `resp_xml_search_batches`, `resp_wiki_search_tag_count`.

The block index is therefore the only quantity that distinguishes "one turn with
two blocks" from "two turns", which is precisely what §12 asks about.

---

## 7. XML logical batch vs structured search round (task §12)

| metric | value |
|---|---|
| both_valid_count | 147,200 |
| exact_match_count | 147,200 |
| exact_match_rate | **100.00%** |
| mean_abs_difference | 0.0 |
| mismatch_count | **0** |
| correct_group_ranking_agreement | 8,456 / 8,456 = **100.00%** |
| ranking_changed_group_count / rate | 0 / 0.00% |

`logical_search_batches ≥ structured_search_rounds` holds by construction (a
round is a distinct assistant turn; a batch is a distinct (turn, block) pair),
so the only possible mismatch is "several XML blocks inside one turn". **That
case did not occur once in this run**: the model emits exactly one
`<wiki_search>` block per search turn.

**Recommendation (task §12 judgement call)**: use the **logical XML batch**
count. Alignment is 100% (threshold was ≥98%), ranking effects are nil, and it
is the cheaper to implement (no dependence on structured call-id internals);
the structured round count remains the cross-check.

---

## 8. Group statistics (task §11) — pre-filter, steps 61..70

| metric | value |
|---|---|
| groups | 18,400 |
| P(group_size == 8) | **100.00%** |
| P(correct_count ≥ 2) | 45.96% |
| P(correct_count == 8) | 44.48% |
| P(xml cost variation \| cc ≥ 2) | **1.12%** |
| P(structured cost variation \| cc ≥ 2) | 1.12% |
| P(all-correct group has cost variation) | 1.03% |
| P(cost == 0 \| correct) | **0.00%** |
| **signal_density** = P(cc ≥ 2 ∧ correct-cost variation) | **0.52%** (95 groups) |

Cost distributions over all 147,200 rollouts (`logical_search_batches` ≡
`structured_search_rounds` everywhere):

| | mean | median | p90 | max |
|---|---|---|---|---|
| logical batches / rounds | 1.441 | 1 | 2 | 7 |
| search queries | 1.990 | 2 | 3 | 23 |
| assistant turns / trajectory | 2.441 | — | — | — |

Correct rollouts are *more* efficient than average: mean cost 1.317 vs 1.441.
Details: `analysis/group_statistics.md`.

---

## 9. Original filter, simulated offline (task §13)

Rule as implemented in this repo (`ray_trainer.py:1780-1781`,
`metric = seq_reward`, EM∈{0,1}): keep a uid iff `std(group rewards) > 0`, i.e.
iff `0 < correct_count < 8`.

| metric | value |
|---|---|
| groups_before_filter | 18,400 |
| expected kept by original reward | 340 (1.85%) |
| expected removed | 18,060 |
| all-correct removed | 8,184 |
| all-wrong removed | 9,876 |
| newly retained under shaped reward (λ any) | **84** |
| per-generation-batch effect | ≈ +0.73 retained prompts per 160-prompt batch (+25% over the ≈2.96 mixed) |

E1-0 did **not** modify the filter; this is an offline simulation only.
Details: `analysis/filter_simulation.md`.

---

## 10. Offline reward replay (task §14) — no GPU run

`R_i = A_i · (1 + λ·E_i)`, `A_i` = EM, `E_i = (Cmax − C_i)/(Cmax − Cmin)` for
correct rollouts of a group with `correct_count ≥ 2` and correct-cost variation,
else 0; advantages computed with the real
`verl.trainer.ppo.core_algos.compute_grpo_outcome_advantage`
(`norm_adv_by_std_in_grpo=True`, matching this run's config) on the 147,200
real audited rollouts. Full results: `analysis/offline_reward_replay.md|json`.

Headline (per λ): all finite; `min(correct reward) − max(incorrect reward) = 1.0`
always; in every one of the 95 active groups the efficient correct rollouts get
strictly larger advantages than the inefficient correct ones; the all-correct
groups go from zero-variance (max |advantage| 0.0000) to trainable
(max |advantage| 2.4747 at λ=0.05). Recommendation: **λ = 0.05**.

---

## 11. Training integrity (task §18) — `analysis/post_train_integrity.md`

VERDICT: **PASS**.

* checkpoint: `global_step_70` exists, dir-name step 70, tracker `70`,
  `model/optim_world_size_4_rank_{0..3}.pt` + `extra_state_…` for all ranks +
  `data.pt`; 36 GB total.
* log: `step:60..step:70` all present, `Final validation metrics:` present,
  621 parsed metric values, **0 non-finite**.
* error scan **before** the final-validation marker: cuda_oom 0, nccl_error 0,
  nccl_warn 6 (harmless `c10d` "hostname of the client socket cannot be
  retrieved"), traceback 0, cuda_error 0, wiki_connection 0, wiki_http_error 0,
  `Error when executing search` 0, `[SearchTool] Execution failed` 0,
  tool_error 0. After the marker: only the expected SIGTERM/actor-teardown
  messages.
* **Wiki tool failure rate = 0 failures** (denominator: 292,900 individual
  search queries = Σ`search_queries` over 147,200 rollouts; the log's
  `tools/wiki_search_total_calls` is per-last-gen-batch only, so it is not the
  right denominator).

---

## 12. Root cause of the earlier silent hangs (attempts 14–19a) — FIXED

Full write-up: `analysis/root_cause_dataloader_fork.md`. Summary:

* **Cause (VERIFIED)**: `e10_sglang_gpu_patch.py` executed
  `monkey_patch_torch_reductions()` at *import* time, and sitecustomize puts
  that module on PYTHONPATH in **every** process — including the Ray
  `TaskRunner` driver. With the patch active, fork-based `StatefulDataLoader`
  workers can never deliver a batch (the patched `reduce_tensor` maps a
  serialized tensor's device through `torch.cuda.get_device_properties()`,
  i.e. CUDA work inside a forked child → deadlock). Driver parked in
  `stateful_dataloader._try_get_data`, workers parked in `index_queue.get()`,
  GPU 0%, log frozen — exactly the observed signature in every hung attempt.
* **Evidence**: SIGUSR1 `faulthandler` stacks captured live by the new
  `code/e10_stall_watchdog.py`
  (`logs/attempt19a_driver_and_dataloader_worker_SIGUSR1_stacks.txt`), plus a
  deterministic CPU-only bisection: importing the reduction patch ⇒ real
  8-worker `StatefulDataLoader` over the nq parquet hangs; without it ⇒ first
  batch in 0.5 s. Importing sglang itself (and the torch-inductor
  `_read_thread`) is harmless.
* **Fix**: the call is deferred to `_ensure_monkey_patched()` invoked from the
  patched `SGLangRollout._init_distributed_env`, i.e. inside the WorkerDict
  actor that actually serializes FSDP weights over CUDA IPC (the SGLang
  scheduler side already calls the same function itself in
  `model_runner.py:1295`). No training semantics touched.
* Why the original 200-step run never hit it: its driver never had sglang
  imported, so the reduction patch was never applied to its DataLoader forks.

---

## 13. Wiki service

`wiki/wiki_status.json`, `wiki/wiki_health_check.txt`: **reused** an existing
healthy server (bound `127.0.0.1:8008`), functional probe with a real query
returned genuine passages (HTTP 200); `WIKI_SERVER_AUTOSTART=0` so E1-0 could
not start a duplicate or stop the user's server; `environment.sh`'s LAN-IP
`WIKI_RAG_SERVER_URL` was overridden to `127.0.0.1` inside the E1-0 launcher
(same precedent as `gen_bs_supervisor.py:251`), with `environment.sh` itself
untouched. Server still healthy after the run; no `wiki/wiki_server.log` is
present because E1-0 never started a server.

---

## 14. Output inventory (task §19)

| path | status |
|---|---|
| `E1-0/latest_run.txt` | ✔ points at this RUN_DIR |
| `preflight/preflight_report.md`, `preflight/preflight.json` | ✔ |
| `environment/environment.txt`, `gpu_before.txt`, `gpu_after.txt`, `original_files_reverify_after_run.txt` | ✔ |
| `code/original_files_manifest.txt`, `code/e10_files_manifest.txt` | ✔ |
| `patches/e10.patch` | ✔ (explains the zero-original-modification approach + the attempt-19 fork fix) |
| `configs/resolved_training_config.yaml`, `configs/launch_command.txt` | ✔ |
| `wiki/wiki_status.json`, `wiki/wiki_health_check.txt` | ✔ (`wiki_server.log` correctly absent: reused) |
| `logs/training.log`, `logs/launcher.log` | ✔ (plus every attempt log and the watchdog log) |
| `audit/rollouts_pid2552655.jsonl` (147,200 records, 240 MB) | ✔ |
| `audit/groups.jsonl` (18,400 groups; canonical copy — the byte-identical `groups_pid2552655.jsonl` duplicate was removed in the 2026-09-11 cleanup, see `logs/cleanup_record.md`) | ✔ |
| `audit/calls_pid2552655.jsonl` (115 calls) | ✔ |
| `audit/audit_summary.json`, `audit/audit_summary.csv` | ✔ |
| `audit/mismatch_cases.jsonl` | ✔ (empty — 0 mismatches) |
| `analysis/group_statistics.md`, `xml_vs_structured.md`, `reward_identity.md`, `filter_simulation.md`, `offline_reward_replay.md`, `offline_reward_replay.json` | ✔ |
| `analysis/post_train_integrity.md/.json`, `root_cause_dataloader_fork.md`, `analyze_e10.py`, `check_training_integrity.py` | ✔ |
| `checkpoints_manifest/step60_source.md`, `checkpoints_manifest/step70.md` | ✔ |
| `FINAL_REPORT.md` (this file) and `E1-0/FINAL_REPORT.md` | ✔ |
| model checkpoints inside E1-0 | correctly absent — real checkpoint lives at `experiments/DAPO-GAP3B-MHQA-Agent-E10-step60to70-4gpu/global_step_70/`, only manifests kept |

---

## 15. Audit reliability (task §17)

The reward manager runs **only in the driver process**
(`reward_model.launch_reward_fn_async=False` in this version's default config),
which is why a single `rollouts_pid2552655.jsonl` is correct. Every record
carries `run_id`, `pid`, `rank` (0), `global_step`, `global_step_source`,
`call_index`, `call_within_step`, `uid`, `timestamp`. Cross-check: the per-step
call counts derived from the audit equal the log's `train/num_gen_batches`
exactly for all ten steps (13,14,13,9,13,13,12,8,10,10). An `audit_errors_*.log`
was created by the manager but is empty (0 audit exceptions).

---

## 16. Code-review corrections made to the pre-existing E1-0 code

All four were real defects in the first E1-0 draft, found by re-reading the
runtime code before this run; each is verified by `code/e10_selftest.py`
(CPU-only, PASS) and by the run's own audit records.

1. **Validation contamination.** `load_reward_manager(..., is_valid=True)`
   instantiates the *same* manager class, so `val_reward_fn` was also auditing —
   it would have written NQ validation rollouts into the training audit file and
   advanced the shared call counter. Fixed by enabling the audit hook only for
   the training instance (`num_examine == 0`, `main_ppo.py:161-162`).
2. **Wrong `global_step`.** A `start + 1 + call_index` estimate is wrong by
   ~13× because one global step consumes 8–14 generation batches. Fixed by
   reading the live `RayPPOTrainer.global_steps` (`e10_trainer_patch.py`); the
   run confirms `global_step` 61..70 with the right call counts.
3. **`logical_search_batches` always 0** (see §6). Fixed by reconstructing the
   block count from the parser block index, with an independent response-text
   cross-check; the run shows 100% agreement with the structured round count.
4. **Vacuous reward identity.** Recording the E1-0 scorer's own `score`/`em`
   proves nothing. Fixed by re-running the **original** `compute_score_em_batch`
   per rollout and comparing to the reward actually fed to GRPO.

Additionally, `patches/e10.patch` and the module docstrings now record the
attempt-19 fork/CUDA root cause and fix.

---

## 17. Attempt history for this session

| attempt | result | cause |
|---|---|---|
| 19a (2 runs) | hung silently after checkpoint load, `_validate: entered` but no first `test_gen_batch meta info` | `monkey_patch_torch_reductions()` in the driver → DataLoader fork deadlock (root-caused this session) |
| 19b | **completed 60 → 70, tracker 70, exit 0** | deferred reductions patch; all fixes above |

The 18 earlier attempts (pre-session) are documented in
`Current_Progress.md` and `logs/*_diagnosis.md`; their symptom is now explained
by the same root cause.

---

## 18. Summary of E1-0 code (all under `RUN_DIR/code/`)

| file | role |
|---|---|
| `e10_scorer.py` | wraps (never reimplements) the original `compute_score_em`; returns `{score, em}` with `score == em` |
| `e10_reward_manager.py` | `e10_batch_audit`, a subclass of the original `BatchRewardManager`; passive audit of 147,200 rollouts + 18,400 groups + 115 calls; independent original-reward recomputation |
| `pythonpath_e10/sitecustomize.py` | PYTHONPATH shim that registers the manager and loads the patches; no original registry file edited |
| `e10_checkpoint_patch.py` | `map_location="cpu"` in `FSDPCheckpointManager.load_checkpoint` (NOSET=1 CUDA-OOM fix) + phase diagnostics |
| `e10_sglang_gpu_patch.py` | per-rank SGLang GPU-visibility fix under `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`; **deferred** CUDA-IPC reductions patch (this session's fork fix) |
| `e10_hang_diagnostics.py` | on-demand SIGUSR1 `faulthandler` stacks (periodic watchdog permanently disabled) |
| `e10_diag_patch.py` | passive worker-side `load_checkpoint`/`generate_sequences`/sharding-manager boundary prints |
| `e10_state.py`, `e10_trainer_patch.py` | live trainer reference ⇒ exact `global_step`; passive driver-phase prints |
| `e10_stall_watchdog.py` | automates the SIGUSR1 stack capture when the log stalls |
| `e10_selftest.py` | CPU-only regression test of the whole audit path (PASS) |
| `run_e10_step60to70.sh` | E1-0 launcher (original parameters; unique E1-0 identity) |
| `analysis/analyze_e10.py`, `analysis/check_training_integrity.py` | the analysis in §8–§11 |

---

## 19. Nothing forbidden was done

No original file edited; no baseline checkpoint touched; no existing log or
results overwritten; no `rm -rf`; no healthy wiki server stopped; no dataset
modified or cleaned; no reward shaping enabled in training; no change to GRPO,
group filtering, rollout concurrency, tool execution, backbone, n, sampling
params, batch composition/order, optimizer, LR or scheduler. The passive audit
writes only to `E1-0/runs/20260910_step60to70_e10/audit/`.

---

## 20. Recommended E1-A design

```
cost                      = logical_search_batches
                            (= number of assistant search rounds, one per
                               non-empty <wiki_search> block; identical to the
                               structured round count on this checkpoint)
lambda                    = 0.05
reward formula            = R_i = A_i * (1 + 0.05 * E_i)
                            A_i = EM correctness (0/1, original GAP rule)
                            E_i = (Cmax - C_i)/(Cmax - Cmin) for correct
                                  rollouts of a group with correct_count >= 2
                                  and correct-cost variation, else 0
window                    = same n=8 uid groups as E1-0; measure with the same
                            audit manager, keep the trainer's filter untouched
expected signal density   = ~0.52% of pre-filter groups (84 / 18,400 newly
                            retained all-correct groups; ~+0.73 retained
                            prompts per 160-prompt generation batch, +25% over
                            the 2.96 mixed groups/call)
important risks           = (1) low base rate -> a 10-step run may see only tens
                            of newly-retained groups; run E1-A long enough (or
                            measure over a window) for the effect to be
                            detectable;
                            (2) shortening pressure vs multi-round questions
                            (P(correct) falls from ~49% at cost 1 to ~15% at
                            cost >=3, so the model does not currently profit
                            from extra searches -- but a too-large lambda could
                            teach it to stop early);
                            (3) use the block-count cost, not a query count:
                            45 rollouts already emit >8 parallel queries in one
                            block (max 22), which a query-count cost would
                            actively reward;
                            (4) all-wrong groups (53.7%) carry no efficiency
                            signal at all -- the dominant untapped signal is
                            correctness on hard prompts, not efficiency.
```

**Do not start E1-A from this report** (task §20). E1-0 ends here.

---

## 21. Final status

```
E1-0 STATUS: PASS_WITH_CAVEATS

  PASS  : correct step60 start; exactly 10 steps to step70; full-state resume
          (scheduler last_epoch 60->70, RNG restored); reward identity 0/147,200
          mismatches, max_abs_diff 0.0, equality 100%; n=8 groups 100%;
          XML vs structured 100% exact and 0/8,456 ranking changes; no NaN/Inf/
          OOM/NCCL/Wiki failures; no original file modified; no overwrite.

  CAVEAT: signal density 0.52% of groups (84 newly-retainable all-correct
          groups) -- the efficiency signal exists and is clean, but it is
          sparse; E1-A must be powered accordingly.

  CAVEAT: this is the first valid attempt in this session (19a hung on an
          E1-0-introduced fork/CUDA deadlock, now root-caused and fixed); the
          pre-session attempts 14-18 share that same cause.

  NO BLOCKER REMAINS.
```

**Primary artifacts**

* this report: `E1-0/runs/20260910_step60to70_e10/FINAL_REPORT.md`
* summary: `E1-0/FINAL_REPORT.md`
* step70 checkpoint: `experiments/DAPO-GAP3B-MHQA-Agent-E10-step60to70-4gpu/global_step_70/`
* audit: `E1-0/runs/20260910_step60to70_e10/audit/`
* analysis: `E1-0/runs/20260910_step60to70_e10/analysis/`
* root cause of the hangs: `E1-0/runs/20260910_step60to70_e10/analysis/root_cause_dataloader_fork.md`

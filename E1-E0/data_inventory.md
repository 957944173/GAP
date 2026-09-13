# E1-E0 data inventory / provenance

Generated from `E1-E0/results/summary.json`; every file below was opened **read-only**.
E1-E0 writes nothing outside `E1-E0/`.

## 1. Candidate stream (the E1-D D2 run's own generation stream)

| item | value |
|---|---|
| source file | `E1-D/runs/D2_20260913_041846/diagnostics/rollouts_pid1249393.jsonl` |
| rollouts | 669,440 |
| n=8 groups | 83,680 |
| generation batches (`call_index`) | 523 |
| optimizer steps (`global_step`) | 50 (71–120) |
| data sources | {'nq': 39127, 'hotpotqa': 44553} |
| mixed-correctness groups (1 ≤ k ≤ 7) | 1,679 |
| all-correct groups (k = 8) | 37,926 |
| all-wrong groups (k = 0) | 44,075 |
| **all-correct + efficiency-variable groups** | **434** |

Target optimizer batch size **B = 32**, read automatically from
`/data01/wyy/Graph-Agent-Planning/E1-D/configs/resolved_step70_to120.json` (`data.train_batch_size`), together with `gen_batch_size =
160`, `rollout.n = 8` and `filter_groups.metric = em`.

## 2. Source files and hashes

| role | path | bytes | sha256 (16) |
|---|---|---|---|
| E1-D D2 candidate stream (rollouts) | `E1-D/runs/D2_20260913_041846/diagnostics/rollouts_pid1249393.jsonl` | 361966409 | `90528cb98af497aa` |
| E1-D D2 generation-batch records | `E1-D/runs/D2_20260913_041846/diagnostics/calls_pid1249393.jsonl` | 423611 | `fd7ac0f88b86896f` |
| E1-D D2 group records (cross-check) | `E1-D/runs/D2_20260913_041846/diagnostics/groups_pid1249393.jsonl` | 57869462 | `86f534a67493bcda` |
| E1-D D2 resolved config (target B) | `E1-D/configs/resolved_step70_to120.json` | 11012 | `730cb32c1dd01947` |
| E1-C summary (unrestricted comparison) | `E1-C/results/summary.json` | 24468 | `a6d266db530a4541` |
| E1-D summary (baseline cross-check) | `E1-D/analysis/summary.json` | 11417 | `6fa4fcff036b0587` |
| E1-C common code copied | `E1-C/code/e1c_common.py` | 16356 | `0764c9e1d96d15d9` |
| E1-C replay code copied | `E1-C/code/e1c_run_all.py` | 40612 | `5d938aa967c2a328` |
| trainer filter source | `verl/verl/trainer/ppo/ray_trainer.py` | 108763 | `29aecd3dcabfaa8a` |
| GRPO advantage source | `verl/verl/trainer/ppo/core_algos.py` | 28396 | `127ef53e92e89093` |

## 3. Available / missing fields

**Available per rollout** (from the E1-D reward-manager diagnostics):
`call_index`, `global_step`, `uid`, `data_source`, `em`, `logical_search_batches`,
`search_queries`, `conversation_turns`, `response_token_length`, `metadata_valid`,
`zero_search`, `complete_reason`, plus the recorded `reward` (= shaped) and `efficiency`.
Per group (recomputed here from the raw rollouts, never trusted from derived fields):
n=8 membership, `k`, mixed/all-correct/all-wrong flags, correction-cost variation,
mean/min/max search cost, queries, tokens, turns.

**Missing / not used**: raw response text (only excerpts exist in older audits, and none in
E1-D's diagnostics); per-mini-batch assignment inside an optimizer step; the actual GRPO
mini-batch shuffle.  None of these is needed by the E1-E0 quota replay.

## 4. Strictly replayable vs proxy

* **Strict**: the generation-stream order (`src_order` = file order, which is the batch order
  the trainer saw, with `call_index` monotone), the n=8 grouping `(call_index, uid)`, the
  mixed/efficiency classification, the mixed-only stopping rule, the quota batch
  construction, the group-level composition, and the GRPO outcome-advantage formula.
* **Proxy**: everything about *training effects*.  E1-E0 never runs the model, so no EM or
  search-round change is claimed; §11 of the report only reports transparent batch-level
  quantities (supervision retained %, efficiency fraction, mean k/8, source shift).

## 5. Cross-checks performed

* q=0 replay consumes **523** generation batches
  = exactly the 523 batches the E1-D D2 run
  actually used (`q0_replay_matches_observed = True`).
* q=0 selects **1600** groups = exactly the
  1679 mixed groups that E1-D's EM filter retained (E1-D report §24).
* the efficiency-group count (434) equals E1-D's reported
  `would_be_revived_by_shaped_filter = 434`.
* all 83,680 reconstructed groups have n = 8 and all
  669,440 rollouts are accounted for.

## 6. Limitations

* The stream comes from **one run (E1-D D2)**, i.e. one policy trajectory; quota availability
  is a property of *that* policy's candidate distribution.
* `uid` is the dataset row index, so the same prompt can be re-sampled in later generation
  batches — the group identity is `(call_index, uid)`, matching the trainer's own filter unit.
* No measurement of what a quota would do to learning is possible offline (by construction).

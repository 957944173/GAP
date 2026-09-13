# E1-D runtime proof that the group filter uses the ORIGINAL EM (E1-D_task.md §11)

Generated 2026-09-13T00:09:53+08:00 from the live D1 training log `/data01/wyy/Graph-Agent-Planning/E1-D/runs/D1_20260912_235330/stdout.log`.

## 1. The trainer's own filter-metric print

`ray_trainer.py:1766` prints `std val <new_batch.non_tensor_batch[metric_name]>` for every
generation batch, where `metric_name = algorithm.filter_groups.metric`.  With
`algorithm.filter_groups.metric=em` the printed array must contain only the original EM
values 0.0/1.0 — a shaped value (1.05) would prove the filter was still reading the shaped
reward.

```
(TaskRunner pid=1022500) std val  [0. 0. 0. ... 1. 1. 1.]
(TaskRunner pid=1022500) std val  [1. 1. 1. ... 0. 0. 0.]
```

Observed values are only 0 and 1 -> the filter sees **EM**, not the shaped reward.
The shaped reward is still produced (it is what `token_level_rewards` carries into GRPO);
the E1-D diagnostics record both channels separately.

## 2. Concrete config passed to the trainer

```
algorithm.filter_groups.metric=em
algorithm.filter_groups.enable=true
```

## 3. Data path (static, from the pinned sources)

```
verl/workers/reward_manager/batch.py  : reward_extra_info[key].append(value) for every key of the scorer dict
                                        -> includes 'em' (E1-D scorer returns em per rollout)
verl/trainer/ppo/reward.py::compute_reward : reward_result['reward_extra_info'] -> reward_extra_infos_dict
verl/trainer/ppo/ray_trainer.py:1755  : new_batch.non_tensor_batch.update({k: np.array(v) ...})
verl/trainer/ppo/ray_trainer.py:1762  : metric_name = config.algorithm.filter_groups.metric  (== 'em')
verl/trainer/ppo/ray_trainer.py:1772  : zip(non_tensor_batch['uid'], non_tensor_batch['em']) -> std per uid
verl/trainer/ppo/ray_trainer.py:1781  : kept iff std > 0 or len(group) == 1
verl/trainer/ppo/ray_trainer.py:1860  : accumulate until num_prompt_in_batch >= train_batch_size(32), batch[:32*8]
```

## 4. Retained counts observed while training (EM filter is sparse by design)

```
(TaskRunner pid=1022500) len(kept_prompt_uids) -  4
(TaskRunner pid=1022500) len(kept_prompt_uids) -  5
```

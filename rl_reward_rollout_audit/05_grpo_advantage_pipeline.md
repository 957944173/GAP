# 05. GRPO advantage pipeline

## 结论

当前训练实际使用 GRPO：

```text
algorithm.adv_estimator=grpo
algorithm.norm_adv_by_std_in_grpo=True（默认配置，训练脚本未覆盖）
```

GRPO 直接读取 `batch.batch["token_level_rewards"]`，先对每条 trajectory 的 response token reward 求和，再按 `data.non_tensor_batch["uid"]` 做组内 mean/std normalization。新 reward 只要在 `BatchRewardManager` 写入 token-level reward 之前产出每 trajectory 的 scalar，就不需要修改 advantage estimator。

## 1. 从 reward 到 GRPO 的实际链路

```text
BatchRewardManager.verify()
  -> scores: list[scalar]
BatchRewardManager.__call__()
  -> reward_tensor[i, valid_response_length - 1] = score_i
RayPPOTrainer.fit()
  -> token_level_scores
  -> token_level_rewards
filter_groups（当前开启）
  -> seq_final_reward = token_level_rewards.sum(-1)
compute_advantage(..., adv_estimator="grpo")
  -> compute_grpo_outcome_advantage()
  -> batch.batch["advantages"], batch.batch["returns"]
ActorRolloutRefWorker.update_actor()
  -> compute_policy_loss(..., advantages=advantages)
```

对应位置：

- reward 写入：`verl/verl/workers/reward_manager/batch.py:80-128`；
- trainer 接收 reward：`verl/verl/trainer/ppo/ray_trainer.py:1740-1756`；
- filter：`ray_trainer.py:1758-1832`；
- estimator dispatch：`ray_trainer.py:216-274`；
- trainer 调用：`ray_trainer.py:1938-1954`；
- GRPO 实现：`verl/verl/trainer/ppo/core_algos.py:172-227`。

## 2. 当前 GRPO 实现的关键细节

`compute_grpo_outcome_advantage()`（`core_algos.py:174-227`）执行：

1. `scores = token_level_rewards.sum(dim=-1)`，每条 trajectory 变成一个 outcome scalar；
2. 用传入的 `index` 建立 `id2score`；
3. 每个 group 计算 mean 和 `torch.std`；
4. `norm_adv_by_std_in_grpo=True` 时：

   ```text
   normalized_i = (score_i - group_mean) / (group_std + 1e-6)
   ```

5. 把 normalized scalar `unsqueeze(-1)` 后乘 `response_mask`，广播到 response tokens；
6. 当前实现直接返回 `(scores, scores)`，所以 `returns` 与 `advantages` 相同。

单样本 group 的特殊行为是 mean=0、std=1，不会做组内中心化；当前 n=8 训练通常不属于该情形，但它解释了为什么必须使用稳定且唯一的 group key。

GRPO 的 group key 不是 batch position，而是 `compute_advantage()` 传入的：

```python
index=data.non_tensor_batch["uid"]
```

trainer 在 reward 前将 `uid` 设为原始 `index`，然后通过 `DataProto.repeat(..., interleave=True)` 对齐 8 条 rollout（`ray_trainer.py:1734-1738`）。

## 3. 当前没有额外的 reward normalization

在 GRPO 之前，活动路径没有发现独立的 reward clip、scale 或 dense token shaping：

| 项目 | 当前状态 | 证据 |
|---|---|---|
| reward model | 关闭 | `ppo_trainer.yaml:687-695`；训练脚本未打开 |
| in-reward KL | 关闭 | `train_dapo_mhqa_agent_wiki.sh:143-145`；默认 `ppo_trainer.yaml:828-829` |
| rule reward clipping/scaling | 未发现 | `BatchRewardManager` 直接写 scorer 返回值 |
| group mean/std normalization | 开启 | `core_algos.py:201-225` |
| policy ratio clipping | 存在 | actor policy loss；它不是 reward clipping |
| group filtering | 开启 | `train_dapo_mhqa_agent_wiki.sh:117-118` |

所以最终进入 GRPO 的字段就是：

```text
batch.batch["token_level_rewards"]
```

活动路径中它与 `token_level_scores` 相同；每 trajectory scalar 位于最后一个有效 response token，其余 response token 为 0。

## 4. toy 运行结果

`scripts/inspect_grouping.py` 直接 import 当前仓库的 `compute_grpo_outcome_advantage()`，没有重新实现公式，也没有启动 Ray/模型/训练。每个 case 都使用同一个 `uid` 的 8 条 trajectory，response mask 为 `[8,4]`。下面给出每条 trajectory 的 advantage（由于 mask 全 1，同一行的每个 response token 相同）：

| case | 输入 outcome reward | 实际每 trajectory advantage |
|---|---|---|
| A | `[1, 1, 0, 0, 0, 0, 0, 0]` | `[1.62018, 1.62018, -0.54006, -0.54006, -0.54006, -0.54006, -0.54006, -0.54006]` |
| B | `[1.0, 0.8, 0, 0, 0, 0, 0, 0]` | `[1.84508, 1.36893, -0.53567, -0.53567, -0.53567, -0.53567, -0.53567, -0.53567]` |
| C | `[1.0, 0.9, 0.7, 0, 0, 0, 0, 0]` | `[1.48054, 1.26120, 0.82252, -0.71285, -0.71285, -0.71285, -0.71285, -0.71285]` |

因此，正确 rollout 之间只要 reward 不同，GRPO 就会产生不同的相对训练信号。目标 reward 中“正确且更省 search batch”的 trajectory 可以获得更高 outcome reward，不需要另写 advantage estimator。

## 5. `filter_groups` 对新 reward 的影响

当前 `algorithm.filter_groups.enable=true`。在 advantage 之前，trainer：

1. 对每条 trajectory 求 `seq_final_reward`；
2. 按 `uid` 收集 reward；
3. 计算每个 uid 的 std；
4. 当 group 有多条样本且 std=0 时丢弃整个 group；否则保留该 uid 的全部 trajectory。

这一步发生在 `ray_trainer.py:1758-1832`，并且使用已经被新 reward 改写的 scalar。因此：

- `#correct=0` 且所有 reward 都为 0 的 group 会被过滤；
- `#correct=1` 通常已有 0/1 方差，会保留；
- `#correct>=2` 且正确 rollout 的 efficiency reward 不同，会产生非零 std，通常会保留；
- 如果新 reward 对一个 group 的所有成员仍给相同值，则 group 仍可能因 std=0 被丢弃。

这不是 advantage estimator 的问题，而是训练样本选择策略的一部分。实验报告必须同时记录“过滤前 group 统计”和“过滤后实际进入 PPO 的 group 统计”。

## 6. 对目标 reward 的判断

若在 reward 阶段计算：

```text
A_i = EM correctness
B_i = logical search-batch cost
B_best = min(B_i for A_i == 1)
R_i = A_i - lambda * A_i * (B_i - B_best)
```

则推荐直接返回每条 trajectory 的 `R_i`，由现有 manager 写入最后一个 token，再由现有 GRPO 做组内标准化。无需修改：

- `compute_grpo_outcome_advantage()`；
- `advantages` / `returns` schema；
- actor policy loss。

需要在 runtime logging 中确认的，是 `B_i` 的定义和数据来源，而不是 GRPO 数学路径。


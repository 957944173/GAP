# 01. Reward pipeline

## 结论

当前默认 supervisor 路径为：

```text
run_gen_bs_supervisor.sh
  -> gen_bs_supervisor.py
  -> train_dapo_mhqa_agent_wiki.sh
  -> python -m verl.trainer.main_ppo
  -> TaskRunner.run()
  -> RayPPOTrainer.fit()
  -> compute_reward(new_batch, self.reward_fn)
  -> BatchRewardManager.__call__()
  -> BatchRewardManager.verify()
  -> mhqa_train.compute_score_em_batch()
  -> mhqa_train.compute_score_em()
  -> token_level_scores / token_level_rewards
  -> compute_advantage(..., adv_estimator=GRPO)
  -> batch.batch["advantages"]
  -> ActorRolloutRefWorker.update_actor()
  -> DataParallelPPOActor.update_policy()
  -> compute_policy_loss(..., advantages=...)
```

活动 reward 是规则式的 batch EM reward，不是模型 reward model。当前配置将 `reward_model.enable` 保持为默认 `False`，并使用 `reward_model.reward_manager=batch`。

## 入口、注册和加载

| 阶段 | 位置 | 证据 |
|---|---|---|
| supervisor 入口 | `Agent/train/mhqa_agent/rl/run_gen_bs_supervisor.sh:1-38` | 启动 `gen_bs_supervisor.py` |
| 实际训练命令 | `Agent/train/mhqa_agent/rl/train_dapo_mhqa_agent_wiki.sh:115-185` | 调用 `python3 -m verl.trainer.main_ppo` |
| Hydra 主入口 | `verl/verl/trainer/main_ppo.py:30-55` | `main()` -> `run_ppo()` |
| reward manager 加载 | `verl/verl/trainer/main_ppo.py:160-181` | 分别加载 train/val reward function 并传给 trainer |
| 自定义函数动态导入 | `verl/verl/trainer/ppo/reward.py:22-70` | 按 `train_path/train_name` 或 `val_path/val_name` 导入函数 |
| manager 注册 | `verl/verl/workers/reward_manager/batch.py:25-32`; `registry.py:20-49` | `@register("batch")` |
| manager 实例化 | `verl/verl/trainer/ppo/reward.py:72-85` | `get_reward_manager_cls("batch")` + custom scorer |

这里有两个不同的“注册”概念：`BatchRewardManager` 通过 registry 注册；`compute_score_em_batch` 不进入 reward registry，而是由 `get_custom_reward_fn()` 动态导入后作为 callable 注入 manager。

## 训练 reward 的具体实现

当前训练脚本明确指定：

```text
custom_reward_function.train_path = verl/verl/utils/reward_score/mhqa_train.py
custom_reward_function.train_name = compute_score_em_batch
reward_model.reward_manager = batch
```

位置：`Agent/train/mhqa_agent/rl/train_dapo_mhqa_agent_wiki.sh:180-184`。

`mhqa_train.py` 的核心行为：

1. `extract_solution()`（`mhqa_train.py:88-110`）寻找最后一个 `<answer>...</answer>`，取其内容并按 `|` 分割。
2. `normalize_answer()`（`38-52`）执行小写化、去标点、去冠词和空白规整。
3. `em_check()`（`55-72`）要求提取出的每个答案片段都在 golden answers 中；训练 scorer 的 `score=1.0`，格式不正确或 EM 失败为 `0.0`。
4. `compute_score_em()`（`113-142`）返回 `{"score": rw, "em": rw}`。
5. `compute_score_em_batch()`（`144-158`）对整批 `(data_source, prompt, response, ground_truth)` 逐条调用 `compute_score_em()`，当前不使用 `extra_infos`，也不做 group-wise 计算。

验证 scorer 位于 `verl/verl/utils/reward_score/mhqa_eval.py:40-158`。它也使用最后一个 `<answer>`，但 `mhqa_eval.em_check()` 在 `mhqa_eval.py:57-72` 只保留第一个预测片段；这与训练 scorer 的 all-pieces 检查存在实现差异。该差异不改变本次训练 reward 的入口判断，但实现新 reward 时应明确只修改 train scorer 还是 train/val 同步修改。

## manager 的输入

`BatchRewardManager.verify()` 位于 `verl/verl/workers/reward_manager/batch.py:34-70`。在 reward 调用时，`data` 已经是 rollout 展开后的 `DataProto`，它读取：

```text
data.batch["prompts"]       -> prompt token ids
data.batch["responses"]     -> response token ids
data.batch["attention_mask"]
data.non_tensor_batch["extra_info"]
data.non_tensor_batch["reward_model"]
data.non_tensor_batch["data_source"]
```

随后：

- `valid_response_lengths = attention_mask[:, prompt_len:].sum(-1)`（`35-40`）；
- 每条 response 按有效长度解码成完整 `response_str`（`46-54`）；
- `questions` 来自 `extra_info[i]["question"]`（`45`）；
- `ground_truths` 来自 `reward_model[i]["ground_truth"]`（`56`）；
- `extras` 是逐样本 `extra_info`（`58`）；
- 一次调用 `self.compute_score(...)`，传入完整列表（`60-68`）。

因此，当前 scorer 调用不是逐 trajectory RPC；它一次拿到当前 `DataProto` 中的整批列表。更重要的是，`extra_infos` 已经包含原始样本的 `index`，所以新 scorer 无须新增显式参数，也可以用 `extra_infos[i]["index"]` 做 group key；该值与 trainer 后续设置的 `uid` 语义相同。`mhqa_train.compute_score_em_batch()` 当前只是没有使用它。

## reward 输出 shape 和落点

`BatchRewardManager.__call__()` 位于 `verl/verl/workers/reward_manager/batch.py:72-128`：

```text
responses shape                  = (N, L_response)
reward_tensor shape              = (N, L_response)
score list                       = N 个 Python scalar 或 score dict
data.batch["acc"]               = (N,)
```

具体写入行为是 `batch.py:80-89` 创建全零 tensor、`89` 调用 `verify()`，并在 `92-107` 对每条样本执行：

```python
reward_tensor[i, length - 1] = reward
```

所以当前 reward 的语义是“每 trajectory 一个 outcome scalar”，只是被编码到最后一个有效 response token；不是 dense token reward。若 scorer 返回 dict，`batch.py:94-105` 取其 `score` 作为 reward，并把其他键累计进 `reward_extra_info`。

trainer 在 `verl/verl/trainer/ppo/ray_trainer.py:1740-1756` 接收结果：

```python
new_batch.batch["token_level_scores"] = reward_tensor
new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]
```

当前 scorer 返回纯 float list，因此 `reward_extra_infos_dict` 通常为空；`data.batch["acc"]` 也不是 GRPO estimator 的输入字段。

## 从 reward 到 policy loss

1. `RayPPOTrainer.fit()` 在 `ray_trainer.py:1740-1756` 完成 reward 写入。
2. 若 `n > 1`，`ray_trainer.py:1758-1832` 先用 reward 求 `seq_final_reward` 并按 `uid` 过滤整组 trajectory。
3. `response_mask` 在 `ray_trainer.py:1874-1875` 添加。
4. `compute_advantage()` 在 `ray_trainer.py:1944-1954` 被调用；GRPO 分支位于 `ray_trainer.py:239-256`。
5. GRPO 调用 `core_algos.compute_grpo_outcome_advantage()`，输入是 `token_level_rewards`、`response_mask` 和 `data.non_tensor_batch["uid"]`。
6. actor 更新入口是 `ray_trainer.py:1963-1969` 的 `update_actor(batch)`；FSDP worker 在 `verl/verl/workers/fsdp_workers.py:598-640` 转到 actor optimizer；实际 policy loss 使用 `verl/verl/workers/actor/dp_actor.py:336-451` 的 `compute_policy_loss(..., advantages=advantages)`。

## normalization、clipping、scaling、filtering

当前活动配置和代码证据：

| 项目 | 当前状态 | 位置 |
|---|---|---|
| reward model | 关闭 | `ppo_trainer.yaml:688-695`；训练脚本未打开 |
| in-reward KL | 关闭 | 训练脚本未覆盖；默认 `ppo_trainer.yaml:828-829` 为 `False` |
| reward clipping/scaling | 未发现活动逻辑 | `BatchRewardManager` 只写 scalar；无 reward clip/scale |
| group filtering | 开启 | 训练脚本 `117-118`；trainer `1758-1832` |
| filter metric | `seq_final_reward` | `ppo_trainer.yaml:811-814` |
| GRPO mean/std | 开启 | `norm_adv_by_std_in_grpo=True`，`ray_trainer.py:1944` |
| PPO policy clip | 存在，但作用于 policy ratio，不是 reward | 训练脚本 `146-148`；`dp_actor.py:389-412` |

当前训练的最终 GRPO reward field 是 `batch.batch["token_level_rewards"]`。在没有 KL penalty 的活动配置下，它与 `token_level_scores` 相同；GRPO 再将其按 response token 求和成每 trajectory scalar。

## 审计判断

新 reward 如果在 `token_level_scores` 写入之前把每条 trajectory 的最终 scalar 计算好，那么现有 filtering、GRPO normalization 和 policy loss 都可以直接复用。需要注意：新 reward 的绝对值仍会被 GRPO group mean/std 归一化；`filter_groups` 也会看到修改后的 reward，因此它的保留/丢弃行为会随 efficiency reward 改变。


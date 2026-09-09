# 02. n=8 rollout grouping

## 结论

**DIRECTLY_POSSIBLE（架构层）**：当前 reward manager 一次接收完整 generation batch，generation batch 中包含同一 prompt 的全部 8 条 rollout；同组身份可由 `uid` 或 `extra_info[i]["index"]` 恢复。因此可以在现有 batch reward 位置实现 group-relative reward。

**NEED_RUNTIME_VERIFICATION（数据层）**：当前没有落盘的训练 `DataProto`/training rollout artifact，尚未运行一次审计 logging 来验证 live batch 中每个 `uid` 恰好出现 8 次、结构化 tool metadata 的序列化形态和实际排序。静态代码与数据文件已经支持上述结论，但经验统计仍不能从现有文件给出。

## 配置和 batch 层级

训练脚本 `Agent/train/mhqa_agent/rl/train_dapo_mhqa_agent_wiki.sh` 的关键值：

| 配置 | 位置 | 值/含义 |
|---|---|---|
| `data.train_batch_size` | `121` | 32，最终 PPO prompt 数 |
| `data.gen_batch_size` | `122` | supervisor 动态注入；已有日志为 96、128、160 |
| `actor_rollout_ref.rollout.n` | `157` | 8 |
| `algorithm.filter_groups.enable` | `118` | true |
| `actor_rollout_ref.rollout.multi_turn.enable` | `175` | true |

`RayPPOTrainer._create_dataloader()`（`verl/verl/trainer/ppo/ray_trainer.py:477-505`）使用的是：

```python
batch_size = config.data.get("gen_batch_size", config.data.train_batch_size)
```

因此当前生成阶段不是每次只取 32 个 prompt。已有 `verl/logs/gen_bs_supervisor_run/train_console.run1.log` 记录实际命令为 `data.gen_batch_size=96`；supervisor 后续把它调整到 128、160。一个 generation batch 先产生 `gen_batch_size * 8` 条 trajectory，过滤后 trainer 才积累/截取 `32 * 8 = 256` 条进入 PPO update。

## 一个 prompt 如何展开成 8 条

活动路径是异步 multi-turn SGLang：

1. `SGLangRollout.generate_sequences()`（`verl/verl/workers/rollout/sglang_rollout/sglang_rollout.py:572-577`）检测 `multi_turn.enable`，转到 `_req_level_generate_sequences()`。
2. `ray_trainer.py:1634-1648` 将原始 `new_batch` 中的输入字段取出，调用 async rollout manager。
3. `_req_level_generate_sequences()`（`sglang_rollout.py:1119-1130`）在训练时传入 `n=self.config.n`；验证时强制 `n=1`（`1125`）。
4. `_preprocess_prompt_to_async_rollout_requests()`（`sglang_rollout.py:1273-1325`）对每个 `data_idx` 执行：

   ```python
   for rollout_offset in range(n):
   ```

   并创建不同 `request_id` 的 `AsyncRolloutRequest`。

5. 请求完成后按 `(batch_data_id, rollout_offset)` 排序（`1129-1130`）。所以同一原始 prompt 的 8 条结果在 rollout output 内连续，顺序可恢复。

6. trainer 在 reward 前设置 group key，然后调用 `DataProto.repeat(..., interleave=True)`（`ray_trainer.py:1734-1738`）：

   ```python
   new_batch.non_tensor_batch["uid"] = new_batch.non_tensor_batch.get(
       "index", uuid_array
   )
   new_batch = new_batch.repeat(repeat_times=8, interleave=True)
   new_batch = new_batch.union(gen_batch_output)
   ```

`DataProto.repeat()` 的实现（`verl/verl/protocol.py:721-758`）对 tensor 使用 `repeat_interleave`，对 non-tensor 使用 `np.repeat`。`union()`（`protocol.py:575-592`）和 `concat()`（`691-711`）保持 dim-0 行对齐。

## group identity

当前 identity 层次如下：

| 字段 | 是否存在 | 生命周期/可靠性 |
|---|---|---|
| `uid` | 是 | trainer 在 reward 前生成，值取原始 `index`；GRPO 直接使用 |
| `index` | 是 | `RLHFDataset.__getitem__()` 在 `rl_dataset.py:309-316` 写入，之后被 repeat；与 `uid` 相同语义 |
| `extra_info.index` | 是 | manager 已经通过 `extra_infos` 传给当前 custom scorer；可作为不改 manager 的 group key |
| `batch_data_id` | 是但仅内部 | `AsyncRolloutRequest` 字段 `schemas.py:69-74`；最终 rollout `DataProto` 未写出 |
| `rollout_offset` | 是但仅内部 | `schemas.py:72-74`；最终 rollout `DataProto` 未写出 |
| `request_id` | 是但未导出 | 用于单条请求和 tool instance，不是 GRPO group key |
| `data_source` | 是 | 任务来源，不足以区分同 source 的不同 prompt |
| batch position | 不是稳定身份 | 当前 reward 前连续，但后续可以被过滤/concat/mini-batch 改变 |

当前 RL parquet 的静态数据检查：`GAP-RL-16w.cleaned.parquet` 共 169,595 行，`extra_info.index` 无缺失、无重复，首尾为 `0..169614`（实际行数与最大 index 的差异来自中间过滤/编号历史，唯一性是关键）。

数据风险：`RLHFDataset` 对缺失 `extra_info.index` 使用默认值 `0`（`rl_dataset.py:309-315`），而 trainer 看到的 `index` key 仍然存在，所以可能导致所有样本共享 `uid=0`；trainer 的 UUID fallback 只有在整个 `index` key 不存在时才会触发。未来运行前应加唯一性断言。

## reward 阶段是否看到完整 group

是。当前调用顺序为：

```text
raw prompt batch (gen_batch_size prompts)
  -> n=8 rollout output (gen_batch_size * 8 rows)
  -> repeat original metadata with interleave=True
  -> union output and metadata
  -> BatchRewardManager.__call__(whole DataProto)
```

`compute_reward()`（`verl/verl/trainer/ppo/reward.py:87-139`）没有对 batch 做逐条拆分；活动配置 `launch_reward_fn_async=False`，所以实际是同步调用 `self.reward_fn(data, return_dict=True)`。`BatchRewardManager.verify()` 一次构造完整的 questions/ground_truths/responses 列表（`batch.py:42-68`）。

当前 custom scorer 还没有使用 group metadata，但它已经收到 `extra_infos`；因此可按 `extra_infos[i]["index"]` 直接 group。若只按 list position，则不应依赖，因为 position 不是长期稳定身份。

## reward 前后是否 shuffle

- `data.shuffle=true` 由 `RandomSampler` 影响原始 prompt 的采样顺序（`main_ppo.py:246-258`），不是 n 条 rollout 的组内 shuffle。
- async rollout 的 request list 和 output 都按 `(batch_data_id, rollout_offset)` 排序；同 prompt 的 8 条结果连续。
- group filter 使用 `new_batch[kept_traj_idxs]`（`ray_trainer.py:1827-1832`），保留原行顺序；`DataProto.concat()` 也按 generation batch 顺序追加。
- `_balance_batch()` 的调用在当前训练循环被注释掉（`ray_trainer.py:1875-1882`）。即使启用，`DataProto.reorder()` 会同步重排 tensor 与所有 non-tensor 字段（`protocol.py:713-719`），`uid` 仍可用于 grouping。
- actor 的后续 mini-batch 处理才可能发生 shuffle；advantage 已在此前由 trainer 按 `uid` 计算。

## filter 过程对 group 的影响

`ray_trainer.py:1758-1832`：

1. 每条 trajectory 的 `token_level_rewards` 求和成 `seq_final_reward`。
2. 用 `uid` 聚合成 `prompt_uid2metric_vals`。
3. 对每个 uid 求 `np.std`。
4. `filter_groups.enable=true` 时保留 `std > 0` 的 group；含多条 trajectory 的常数 reward group 被丢弃。
5. 保留的是整个 group 的所有成员，不是单条 trajectory。

随后 trainer 累积 group，达到 `train_batch_size=32` 后截取 `traj_bsz=32*8`（`1860-1872`）。因此 reward scorer 可以在一个 generation batch 内比较多个 prompt group，但不会跨独立 generation batch 共享 group 统计；同一 prompt 的 8 条不会被拆到不同 generation batch。

## 最终判断

对目标逻辑：

```text
for each prompt group keyed by extra_infos[i]["index"] / uid:
    find correct rollouts
    compute search cost for each rollout
    compare only correct rollouts
    assign scalar reward before token_level_scores is written
```

当前架构给出 **DIRECTLY_POSSIBLE**。最小实现甚至可以只修改 `mhqa_train.compute_score_em_batch()`，因为 `extra_infos` 已经随调用传入；若要使用结构化 `detailed_tool_metrics` 而不是解析 response 文本，则需要给 scorer 增加少量 metadata plumbing，见 `06_candidate_patch_points.md`。


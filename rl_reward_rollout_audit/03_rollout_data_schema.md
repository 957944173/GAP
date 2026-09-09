# 03. Rollout / DataProto schema

## 结论

reward 计算阶段可以拿到完整 response text：`BatchRewardManager.verify()` 根据 `responses` token 和 `attention_mask` 解码有效 response（`verl/verl/workers/reward_manager/batch.py:34-54`）。因此可以从 response 中解析 `<wiki_search>...</wiki_search>`，但当前已有的结构化 tool metadata 更适合做严格统计；它存在于 live `DataProto`，却没有传给 custom scorer。

## 数据阶段

`RLHFDataset.__getitem__()`（`verl/verl/utils/dataset/rl_dataset.py:212-317`）从当前训练 parquet 产生 tensor/non-tensor 混合 sample；`collate_fn()`（`39-67`）将 tensor stack 成 batch，将其他值转成 `np.ndarray(dtype=object)`。

当前训练 parquet 首行静态样本来自：

```text
Agent/data/mhqa_agent/GAP-MHQA-RL-Dataset/GAP-RL-16w.cleaned.parquet
```

实际 parquet 列：`question`, `answer`, `data_source`, `prompt`, `reward_model`, `extra_info`。

| field | type / shape / example | source | reward stage 可用？ |
|---|---|---|---|
| `question` | source string；实际 RL parquet 存在 | parquet -> RLHFDataset row | 是，通常也位于 `extra_info.question` |
| `answer` | list[str]；如 `['2,718']` | parquet | 是，作为 non-tensor metadata；当前 scorer主要用 nested ground truth |
| `data_source` | string，如 `nq` / `hotpotqa` | parquet，保留到 `non_tensor_batch` | 是 |
| `prompt` | list[dict]，`[{role, content}]` | parquet；`return_raw_chat=true` 时也生成 `raw_prompt` | 是，原始 `prompt` 保留；`raw_prompt` 在生成前被 pop |
| `reward_model` | dict，训练样本中 `ground_truth -> target -> list[str]` | parquet | 是，`BatchRewardManager.verify():56` 读取 |
| `extra_info` | dict；含 `question`, `answer`, `index`, `need_tools_kwargs`, `tools_kwargs` | parquet + dataset | 是；当前 scorer 已收到完整 `extra_infos` |
| `extra_info.index` | 当前 RL parquet 为唯一 int；全表静态检查无重复 | parquet / `rl_dataset.py:309-315` | 是；可直接作为 group key |
| `tools_kwargs` | dict；如 `wiki_search.create_kwargs` | `rl_dataset.py:311-316` | 生成前有；`ray_trainer.py:1627-1637` 在生成前 pop，reward stage通常不可用 |
| `input_ids` | torch tensor `[B, 2048]`（当前配置） | dataset tokenization `rl_dataset.py:251-285` | 在生成前有；之后作为完整 sequence 字段保留 |
| `attention_mask` | torch tensor `[B, 10240]` after rollout padding | dataset + rollout postprocess | 是 |
| `position_ids` | torch tensor，和 sequence 对齐 | dataset / rollout | 是 |
| `prompts` | rollout output tensor `[N, prompt_len]` | `sglang_rollout.py:1243-1253` | 是 |
| `responses` | rollout output tensor `[N, response_len]` | `sglang_rollout.py:1243-1253` | 是 |
| `loss_mask` | rollout output tensor `[N, prompt+response]` | `sglang_rollout.py:1243-1253` | 是；multi-turn GRPO 使用其 response 部分 |
| `messages` | `np.ndarray(dtype=object)`；每条为 `{"messages": req.messages}` | `sglang_rollout.py:1182`, `1261-1270` | 是，但当前 custom scorer未接收 |
| `reward_scores` | 每条 request 的 tool reward dict | `sglang_rollout.py:1183`, `1265` | 是，但不是最终 RL reward；当前 scorer未接收 |
| `complete_reason` | string | `sglang_rollout.py:1184`, `1266` | 是，但当前 scorer未接收 |
| `finish_reason` | dict | `sglang_rollout.py:1185`, `1267` | 是，但当前 scorer未接收 |
| `tool_call_info` | dict[str,int]，如 `{'wiki_search': 2}` | `sglang_rollout.py:1186`, `1268` | 是，但只计 tool call，不计逻辑 block；当前 scorer未接收 |
| `detailed_tool_metrics` | dict；含 total、按工具计数、turns、`tool_call_sequence` | request schema `schemas.py:201-247`; output `sglang_rollout.py:1188-1200`, `1269` | 是，live DataProto 可见；当前 scorer未接收 |
| `uid` | `np.ndarray`，每组 8 行相同 | trainer `ray_trainer.py:1734-1738` | 是；GRPO直接使用，当前 scorer通过 `extra_infos.index`间接可得 |
| `token_level_scores` | float tensor `[N, response_len]` | trainer `ray_trainer.py:1751` | reward manager返回后写入；在 scorer调用时尚未存在 |
| `token_level_rewards` | float tensor `[N, response_len]` | trainer `ray_trainer.py:1754-1756` | reward manager返回后存在；GRPO直接使用 |
| `acc` | float tensor `[N]` | `BatchRewardManager.__call__:123` | scorer调用时由 manager 写入；主要是辅助 metric，不是 GRPO输入 |
| `response_mask` | bool/int tensor `[N, response_len]` | `ray_trainer.py:1874-1875` | reward 后、advantage 前 |
| `advantages` / `returns` | float tensor `[N, response_len]` | `ray_trainer.py:1946-1954` | reward stage不可用；GRPO后产生 |

`DataProto` 对 batch 和 non-tensor batch 的一致性要求见 `verl/verl/protocol.py:310-328`；dim-0 长度必须一致。`repeat()`、`slice`、`concat` 和 `reorder` 都同步处理 non-tensor arrays，因此 `uid`/`extra_info` 不会因这些操作脱离 trajectory。

## reward stage 实际可见的字段集合

在 `ray_trainer.py:1734-1739` union 完成后，活动路径至少应包含：

```text
batch:
  prompts, responses, input_ids, attention_mask, position_ids, loss_mask

non_tensor_batch:
  data_source, prompt, reward_model, extra_info, answer, index, uid,
  messages, reward_scores, complete_reason, finish_reason,
  tool_call_info, detailed_tool_metrics
```

这里“至少”指代码构造的字段并集；不同数据、worker backend 或多模态配置可能增加字段。`BatchRewardManager.verify()` 当前主动取出的只是 prompt/response/mask、question、ground truth、data source、extra_info；它没有把其他 non-tensor 字段转发给 `compute_score_em_batch()`。

## response text 是否完整

是。`verify()` 用：

```python
valid_response_ids = response_ids[i][:valid_len]
response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
```

这里的 `response_ids` 是 multi-turn rollout 从 `req.input_ids[len(req.prompt_ids):]` 得到的完整 response sequence，包含后续 assistant/tool turns；`AsyncRolloutRequest.finalize()` 位于 `schemas.py:249-294`，rollout output 再在 `sglang_rollout.py:1169-1181` 组装。

已有 validation generation 文件也实际显示完整 `<wiki_search>...</wiki_search>`、`<observation>...</observation>` 和最终 `<answer>...</answer>` 文本。这证明 response text 解析路径可工作；它不是训练 group 数据的替代品，因为 validation 强制 `n=1` 且 dump 中没有 uid。

## 当前保存 artifact 的限制

训练脚本没有设置 `trainer.rollout_data_dir`。已有训练日志中的 resolved config 明确记录：

```text
rollout_data_dir: None
validation_data_dir: None
```

代码虽然支持 `ray_trainer.py:1972-1986` 的训练 rollout dump，但 `_dump_generations()`（`543-589`）只会保存 `input`, `output`, `score`, `step`，训练调用点也没有传 `data_sources`、`ground_truths` 或 uid。因此即便仅把 `rollout_data_dir` 打开，现有 dump 也不足以做可靠 group 统计，除非同时把 group/aux fields 作为 `reward_extra_infos_dict` 传入或增加 audit-only logging。

## 对 response 解析 search cost 的判断

可以从完整 response 文本计算逻辑 search 指标，但要遵守 `04_search_batch_tracking.md` 中的 tag、空块、重复调用和 observation 伪标签约束。更稳妥的实现顺序是：

1. 首选 live `detailed_tool_metrics.tool_call_sequence`，因为它来自实际 parser/tool call；
2. 若 scorer 不接收该字段，则解析 `responses`，并将结果标记为 logical XML-block counts；
3. 需要“实际成功 HTTP 请求数”或错误率时，当前输出 schema 不够，必须把 `req.metrics`/status 一并导出。


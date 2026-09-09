# 06. Candidate patch points

本节只定位最小修改路径，不实施任何 patch。目标 reward 假设为 success-conditioned group-relative search-batch reward：先算每条 trajectory 的 EM correctness，再只在同一 prompt 的正确 rollout 中比较 search cost。

## 推荐顺序

正式修改前先按第 07 节做短 runtime logging。若只需要逻辑 search batch，优先方案 A；若要使用真实 rollout metadata，优先方案 B；方案 C 是 reward API 无法承载 group-wise 输入时的兜底路径。

## 方案 A：只改 train scorer（最小侵入）

| 项目 | 结论 |
|---|---|
| 文件/函数 | `verl/verl/utils/reward_score/mhqa_train.py:144-158`，`compute_score_em_batch()` |
| 新增输入 | 不必新增 API 输入；已有 `responses`、`ground_truths`、`extra_infos`，其中 `extra_infos[i]["index"]` 可作 group key |
| 需要改 DataProto schema | 否 |
| 需要改 rollout worker | 否 |
| 需要改 reward manager | 否 |
| 需要改 GRPO estimator | 否 |
| 风险 | 代码侵入 LOW；指标语义 MEDIUM |

实现逻辑应是：

1. 逐条调用现有 `compute_score_em()` 得到 `A_i`；
2. 对每条 response 计算 `logical_num_search_batches`；
3. 按 `extra_infos[i]["index"]` 分组；
4. 只在 `A_i=1` 的成员中求 `B_best`；
5. 输出每条 trajectory 的 scalar reward；
6. 如果某组没有正确 rollout，保持全组 reward=0；如果只有一个正确 rollout，可把其 cost penalty 定义为 0，避免单样本“相对效率”任意化。

`compute_score_em_batch()` 目前已经接收 `extra_infos`，只是 153-158 完全没有使用。因此这是最短的架构路径。也可以返回带 `score` 的 dict，同时携带 `logical_num_search_batches`、`correctness` 等诊断字段；`BatchRewardManager` 会在 94-105 把 `score` 取出，并把其他键收集到 `reward_extra_info`。

主要风险是：response text 解析只能近似恢复 parser 语义；它不能可靠区分 duplicate query、HTTP retry、tool error 和真正的网络请求数。方案 A 适合作为 prototype，不应在未经 runtime 对齐时把该数称为实际 API cost。

## 方案 B：BatchRewardManager 转发结构化 rollout metadata（推荐的规范路径）

| 项目 | 结论 |
|---|---|
| 文件/函数 | `verl/verl/workers/reward_manager/batch.py:34-70`，`BatchRewardManager.verify()`；同时扩展 `mhqa_train.compute_score_em_batch()` |
| 新增输入 | `uid`、`detailed_tool_metrics`、可选 `tool_call_info` / `messages`，从 `data.non_tensor_batch` 转发给 scorer |
| 需要改 DataProto schema | 当前字段已存在，不需要新增 schema；若要准确 HTTP cost，需要 rollout worker 增加更细字段 |
| 需要改 rollout worker | 仅当需要 block-level/HTTP-level status、retry、duplicate 字段时需要 |
| 需要改 reward manager | 是，增加少量 kwargs plumbing |
| 需要改 GRPO estimator | 否 |
| 风险 | MEDIUM |

具体可以在 `verify()` 已有调用（60-68）中加入例如：

```text
uids=data.non_tensor_batch.get("uid")
detailed_tool_metrics=data.non_tensor_batch.get("detailed_tool_metrics")
tool_call_info=data.non_tensor_batch.get("tool_call_info")
```

然后在 scorer 中优先使用结构化 metrics，response text 作为 fallback。需要保持所有数组与 `responses` 的 dim-0 顺序一致，并对缺失 metrics 使用显式 sentinel，而不是把缺失误当成零 search。

注意：当前 `detailed_tool_metrics.total_tool_calls` 更接近 parsed query/tool-call 数；它没有直接存 `num_search_batches`。若只利用现有 `tool_call_sequence.call_id`，可以在 scorer 侧按 `call_wiki_search_<block>_<query>` 解析，但这依赖 parser 的命名约定。更稳妥的版本是在 rollout request 的 `detailed_tool_metrics` 中新增：

```text
search_batch_count
search_query_count
parallel_batch_count
executed_http_count
duplicate_count
error_count
```

## 方案 C：自定义 group-aware reward manager / trainer fallback

### C1. 自定义 reward manager

| 项目 | 结论 |
|---|---|
| 文件/函数 | 新增 `verl/verl/workers/reward_manager/<custom_group_manager>.py`；参考 `batch.py:25-128`；通过 `registry.py:20-49` 注册 |
| 新增输入 | 直接使用完整 `DataProto`，读取 `uid`、`extra_info`、response token 和 rollout metadata |
| 需要改 DataProto schema | 否，除非要求新增 worker metadata |
| 需要改 rollout worker | 可选；只为真实 HTTP cost |
| 需要改 reward manager | 是，新 manager 取代 batch manager |
| 需要改 GRPO estimator | 否 |
| 风险 | MEDIUM-HIGH |

这个方案把“先得到 correctness、再 group-wise 改写 scalar、最后落到 reward tensor”的逻辑封装在 manager 内，避免把大量业务逻辑塞进通用 scorer API。需要新增 registry import/config，并为 train/val 明确区分行为。

### C2. trainer 侧 fallback

如果 custom reward API 或 manager 不能改，可在 `verl/verl/trainer/ppo/ray_trainer.py:1740-1756` 的 `compute_reward()` 返回之后，使用当前 `new_batch` 的 `uid`、response 和 rollout fields 改写 `reward_tensor`，再写回 `token_level_scores/token_level_rewards`。这属于更大侵入的 fallback：它将 reward 业务放入 trainer，且容易与 `filter_groups` 的边界、异步 reward 和 validation 逻辑发生耦合。建议只作为临时诊断方案。

## 方案对比

| 方案 | 结构化 metadata | 原始源码改动面 | DataProto 变化 | worker 变化 | GRPO 变化 | 适用场景 |
|---|---|---|---|---|---|---|
| A | 否，解析 response | 最小 | 否 | 否 | 否 | 快速 prototype / 逻辑成本 |
| B | 是，manager 转发 | 小 | 通常否 | 可选 | 否 | 正式实验的推荐路径 |
| C | 是，manager/trainer 直接控制 | 较大 | 通常否 | 可选 | 否 | scorer API 受限或需要强控制 |

## 最终建议

先做 audit-only runtime logging，验证 group size、uid 对齐以及正确 rollout 的 search-cost 差异。验证通过后：

1. 若只研究“模型输出的逻辑 search batch”，采用方案 A；
2. 若要依赖 `detailed_tool_metrics` 或区分 duplicate/error，采用方案 B，并补充 worker-level metadata；
3. 不要改 GRPO estimator；修改点应停在 outcome scalar 进入 `token_level_scores` 之前。


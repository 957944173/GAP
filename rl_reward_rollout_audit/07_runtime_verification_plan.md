# 07. Runtime verification plan

## 当前状态

本仓库目前没有可直接用于本专项统计的训练 rollout artifact：

- resolved training config 记录 `trainer.rollout_data_dir: None`、`trainer.validation_data_dir: None`（`verl/logs/gen_bs_supervisor_run/train_console.run1.log:393-400`）；
- trainer 的 `_dump_generations()`（`ray_trainer.py:543-590`）默认只保存 input/output/score/step，以及显式传入的 reward extra info；训练调用点 `ray_trainer.py:1972-1986` 没有传 uid、group identity 或详细 rollout metadata；
- 已有 `experiments/*/val_generations/*.jsonl` 是 validation 结果，字段包括 input/output/score/reward/response_length/num_turns 等，但没有 `uid`；validation rollout 在 `sglang_rollout.py:1125` 强制 `n=1`。它们可用于验证 response 中是否出现工具轨迹，不能用于 n=8 training group 统计。

因此以下指标当前都标记为：

```text
NEED_RUNTIME_VERIFICATION
```

没有启动训练，也没有把 validation dump 当作训练样本估计 group-relative reward 的收益。

## 1. 建议的 audit-only logging 位置

在不改变训练逻辑的临时诊断分支中，建议在：

```text
RayPPOTrainer.fit()
  reward_tensor 已生成之后（ray_trainer.py:1749-1756）
  filter_groups 之前（ray_trainer.py:1758）
```

采集原始 generation batch。必须在 filter 前记录，因为 filter 会丢弃 std=0 的 group，事后无法恢复完整分布。该 logging 不应修改 reward 值、DataProto 顺序或 checkpoint。

每条 trajectory 建议记录一行 JSONL：

```text
step
generation_batch_id
row_position
uid
extra_info.index
data_source
em_correct
raw_score
response_length
logical_num_search_batches
logical_num_queries
parallel_batch_count
parallel_sample
parallel_factor
tool_call_info
detailed_tool_metrics
parse_status
```

如果要审计实际 retrieval 成本，还需要记录：

```text
search_batch_id, query, call_id, duplicate, status, error, retry_attempts,
executed_http_count
```

这些字段当前不是完整的现成 schema，应在 audit-only 版本中明确标注 `unknown`，不能用 0 代替未知。

## 2. 只需运行 1–5 个 training steps 的采集方案

### Step 0：启动前静态断言

对每个将进入 generation 的原始 batch 检查：

```text
index/extra_info.index 存在
index 唯一（至少在原始 prompt batch 内）
每行 uid == extra_info.index
config rollout.n == 8
```

缺失 `extra_info.index` 时，当前 dataset 代码可能使用默认值 0（`rl_dataset.py:309-315`），会把多个不同 prompt 错误地聚成同一 uid；这一步必须 fail fast。

### Step 1–5：每个 generation batch 记录

每个 prompt 预期产生 8 条 rollout。当前 `gen_batch_size` 由 supervisor 动态调整：脚本默认 64，supervisor 默认初始 128、范围 96–160；已有 run log 曾实际使用 96、128、160。每个 batch 的预期 trajectory 数应是：

```text
observed_prompt_count * 8
```

对每个 `uid` 聚合并记录：

```text
group_size
correct_count
search_batch_values
query_values
response_length_values
```

同时保留 batch-level 原始 rows，避免只保存聚合统计导致排错困难。

### 结束后汇总

对 1–5 steps 的全部过滤前 group 计算：

1. group size 分布，检查是否全部为 8；
2. `#correct` 的 0 / 1 / >=2 分布；
3. `P(#correct >= 2)`；
4. 在 `#correct >= 2` 的 group 中，正确 rollout 的 `num_search_batches` 是否有不同值；
5. `max(B_correct)-min(B_correct)` 的分布；
6. 同组正确 rollout 的 query-count range；
7. 同组正确 rollout 的 response-length range；
8. `uid == extra_info.index` 的通过率；
9. response parser 与 `detailed_tool_metrics` 的一致率；
10. logical query count、parsed tool-call count、实际 HTTP count（若可得）的差异。

## 3. 一致性检查

建议把以下检查作为 runtime audit 的硬性输出：

| 检查 | 通过条件 | 失败时含义 |
|---|---|---|
| group size | 每个 uid 恰好 8 行 | rollout 丢失、padding、batch 边界或 identity 处理异常 |
| identity | 所有 row 的 `uid == extra_info.index` | group key 错位或 dataset index 缺失 |
| order-independent grouping | 打乱 rows 后聚合结果不变 | 实现错误地依赖位置 |
| parser agreement | response logical count 与 structured sequence 可解释地对齐 | response 序列化、解析或 metadata 丢失 |
| correct-only cost | `B_best` 只在 `em_correct=1` 中求 | 错误 rollout 被误用为效率基准 |
| actual-vs-logical cost | duplicate/retry/error 单独统计 | 把 tool call 数错误当成 HTTP cost |

现有训练日志只能提供 aggregate tool metrics，例如 step 1 有 `tools/total_calls`、`tools/avg_calls_per_traj` 和 `tools/avg_turns_per_traj`；它没有 uid 维度，所以不能替代上述检查。

## 4. 推荐输出格式

建议新增（未来正式运行时，仍放在本审计目录或单独 audit output 目录）：

```text
rl_reward_rollout_audit/runtime/
  step_001_rows.jsonl
  step_001_groups.jsonl
  step_002_rows.jsonl
  ...
  summary.json
```

`summary.json` 至少保存：

```json
{
  "rollout_n_config": 8,
  "observed_group_size_histogram": {},
  "correct_count_histogram": {},
  "p_correct_at_least_2": null,
  "p_correct_cost_diff_given_at_least_2": null,
  "correct_search_batch_range": {},
  "correct_query_range": {},
  "correct_response_length_range": {},
  "uid_index_match_rate": null,
  "status": "NEED_RUNTIME_VERIFICATION"
}
```

所有尚未采集的数值保持 `null`，不要填入从 validation 或 aggregate log 推断出的伪统计。

## 5. 进入 Experiment 1 的判定

在 runtime logging 前，不建议直接进入 Experiment 1，状态为 `NEED_RUNTIME_VERIFICATION`。建议满足以下证据后再进入：

- n=8 group identity 在 live generation batch 中稳定且 group size 正确；
- `uid` 与 `extra_info.index` 全程对齐；
- search logical metrics 与结构化 tool sequence 的差异有明确解释；
- `#correct >= 2` 的 group 非零且数量足以观察组内相对效率；
- 这些 group 中确实存在正确 rollout 间的 search-batch 差异；
- 过滤前后样本量和 reward 分布都已记录。

如果正确 rollout 几乎没有 cost variation，先做诊断/可视化，不应假设 group-relative reward 会提供额外学习信号。


# GAP RL reward pipeline + rollout grouping audit

## 先给结论

1. 当前 reward 计算入口：`RayPPOTrainer.fit()` 中的 `compute_reward(new_batch, self.reward_fn)`，进入 `BatchRewardManager.__call__()` / `verify()`。
2. 当前 EM/correctness scorer：`verl/verl/utils/reward_score/mhqa_train.py:compute_score_em_batch`，逐条调用 `compute_score_em`。
3. rollout n：`8`。
4. 同 prompt rollout 的 grouping key：`uid`；trainer 将它设为原始 `index`，其源头是 `extra_info["index"]`。
5. reward 阶段是否能看到完整 group：架构上 `YES`；live batch 的 group size 和排序仍需 `NEED_RUNTIME_VERIFICATION`。
6. reward 阶段是否能看到完整 response：`YES`，`BatchRewardManager.verify()` 解码完整有效 response token 序列。
7. 是否能可靠统计 search batches：逻辑 XML batch 可从 response/parser 恢复，但实际 HTTP cost 受 duplicate/retry/error 影响；最终状态 `NEED_RUNTIME_VERIFICATION`。
8. 实现 group-relative reward 是否需要修改 advantage estimator：`NO`，在现有 outcome scalar 位置写入新 reward 即可。
9. 推荐的最小 patch 点：先审计后修改 `mhqa_train.compute_score_em_batch()`；若需结构化 tool metadata，再给 `BatchRewardManager.verify()` 增加少量转发。
10. 是否建议进入 Experiment 1：`NEED_RUNTIME_VERIFICATION`，先运行 1–5 steps 的 audit-only logging。

专项审计已完成。本次只新增 `rl_reward_rollout_audit/` 下的诊断文件和脚本，没有修改训练源码、数据或 checkpoint，也没有启动正式训练。


### 补充审计结论

- Reward：`compute_score_em_batch()` 对每条 trajectory 输出一个 scalar，写入该 trajectory 最后一个有效 response token，最终使用 `token_level_rewards`。
- Search：实际标签是 `<wiki_search>...</wiki_search>`；`|` 会拆成并发 query，但当前执行的是多个单 query HTTP 请求。逻辑 batch 可以解析，实际 HTTP cost 仍需 `NEED_RUNTIME_VERIFICATION`。
- GRPO：按 `uid` 做组内 mean/std normalization。toy 实验已证明 `[1,1,0,...]` 与 `[1,0.8,0,...]` 会产生不同 advantage，因此无需修改 GRPO estimator。
- 训练 artifact：当前没有训练 rollout artifact；已有 validation dump 为 n=1 且无 `uid`。Experiment 1 状态为 `NEED_RUNTIME_VERIFICATION`，建议先运行 1–5 steps audit logging。

## 最终决策表

| Question | Answer | Evidence |
|---|---|---|
| Can same-prompt rollouts be grouped? | YES（架构）；runtime group size NEED_RUNTIME_VERIFICATION | `ray_trainer.py:1734-1738`; `compute_advantage` 使用 uid；`02_rollout_grouping.md` |
| Can search batches be computed? | 逻辑 batch YES；实际 HTTP batch NEED_RUNTIME_VERIFICATION | `xml_tool_parser.py:63-135`; `search_tool.py:201-235`; `04_search_batch_tracking.md` |
| Can correct rollouts be identified? | YES | `mhqa_train.py:113-158` 的 EM scorer 可先得到 `A_i` |
| Can reward distinguish correct-efficient vs correct-inefficient? | YES（只要 cost 指标完成并在同组内计算） | manager 收到完整 generation batch；`05_grpo_advantage_pipeline.md` toy 结果 |
| Need to modify GRPO estimator? | NO | `core_algos.py:172-227` 接受任意 outcome scalar |
| Need DataProto changes? | NO for response-text prototype；structured/exact HTTP metrics may need fields | `03_rollout_data_schema.md`; `06_candidate_patch_points.md` |
| Need rollout-worker changes? | NO for text prototype；YES only for exact block/HTTP status accounting | `04_search_batch_tracking.md` |
| Estimated implementation complexity | A: LOW–MEDIUM；B: MEDIUM；C: MEDIUM–HIGH | `06_candidate_patch_points.md` |

## 核心调用链

```text
run_gen_bs_supervisor.sh
  -> gen_bs_supervisor.py
  -> train_dapo_mhqa_agent_wiki.sh
  -> verl.trainer.main_ppo / TaskRunner
  -> RayPPOTrainer.fit
  -> rollout.n=8 (sglang_async)
  -> uid/index repeat + DataProto.union
  -> compute_reward
  -> BatchRewardManager.verify
  -> mhqa_train.compute_score_em_batch
  -> token_level_scores / token_level_rewards
  -> filter_groups by uid
  -> compute_grpo_outcome_advantage
  -> advantages / returns
  -> actor policy loss
```

## 关键判断

当前架构最重要的事实是：reward manager 在一次调用中收到整个展开后的 generation `DataProto`，不是每条 trajectory 一个独立 RPC；同组 8 条 rollout 共享 `uid`，且 `extra_infos` 已经传入 custom scorer。因此 group-relative reward 在 reward 阶段是 `DIRECTLY_POSSIBLE`。

当前最大的不确定性不是 GRPO，而是 search cost 的语义：活动 parser 把 `|` 拆为多个并发的单 query tool call，`detailed_tool_metrics` 记录 parsed call，而 duplicate、HTTP retry 和 error 可能改变实际网络成本。正式 reward 前应先完成 runtime 对齐。

## 文件导航

报告入口：`rl_reward_rollout_audit/README.md`

主要附录：

- [04_search_batch_tracking.md](04_search_batch_tracking.md)：XML search、parallel query、HTTP 语义和 edge cases。
- [05_grpo_advantage_pipeline.md](05_grpo_advantage_pipeline.md)：当前 GRPO 代码与 toy 结果。
- [06_candidate_patch_points.md](06_candidate_patch_points.md)：三个候选最小修改点。
- [07_runtime_verification_plan.md](07_runtime_verification_plan.md)：1–5 steps 的 audit-only logging 方案。

配置索引：

- [key_locations.csv](key_locations.csv)：路径、函数、行号和用途索引。
- [config_summary.json](config_summary.json)：当前实际相关配置摘要。

其他审计材料：

- [01_reward_pipeline.md](01_reward_pipeline.md)：reward 注册、输入输出、聚合和 policy loss 链路。
- [02_rollout_grouping.md](02_rollout_grouping.md)：n=8 展开、uid/index、过滤和 group 可见性。
- [03_rollout_data_schema.md](03_rollout_data_schema.md)：parquet、DataProto 和 rollout metadata 字段。

## 已运行的轻量检查

使用训练环境 `/home/nf5468m6/miniconda3/envs/parallel-agent/bin/python`：

- 两个 audit script 通过 `py_compile`；
- `inspect_grouping.py` 直接运行仓库的 `compute_grpo_outcome_advantage`，得到 A/B/C 三组不同 advantage；
- `inspect_rollout_schema.py --rows 1` 读取 RL parquet 的一行和 schema；
- 同一脚本读取已有 validation JSONL，确认 dump 没有 uid，不能替代 training group artifact。

仓库根目录不是 Git worktree（`git status` 返回 `fatal: not a git repository`），因此本次变更范围以新增目录文件和源码/日志只读检查为准。


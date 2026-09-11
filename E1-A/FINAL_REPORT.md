# E1-A FINAL REPORT

**Experiment**: E1-A — *Success-conditioned group-relative efficiency reward*
**Start → end**: original GAP `step60` → `step70` (10 RL steps)
**Date**: 2026-09-11 (training 02:49–05:42, evaluations 05:45–09:10 CST)
**Training entry**: `E1-A/run_e1a_train.sh`  ·  **Eval entry**: `E1-A/run_e1a_eval_step70.sh`
**Training run dir**: `E1-A/run_20260911_024902/`
**Checkpoint**: `experiments/DAPO-GAP3B-MHQA-Agent-E1A-step60to70-4gpu/global_step_70/`
**Verdict**: **E1-A 目标达成，建议 GO 进入 E1-B（step70 → step120）** — 相对"同样 10 步、
原始 reward"的匹配 baseline，search rounds 在 **7/7 benchmark 上下降**
（macro −1.80%，逐 prompt 配对 −2.04%，符号检验 p≈3×10⁻²⁵），
**EM 统计上不变**（macro −0.22 pp，配对 −0.08 pp，p=0.56），且无任何安全回退。详见 §5、§7。

---

## 1. 实验目标 (Goal)

在最小代码修改原则下验证 **Success-conditioned group-relative efficiency reward** 能否在
**保持 accuracy 的同时降低 sequential search depth**：

```
R_i = A_i * (1 + 0.05 * E_i)
A_i = EM correctness（原始 GAP 判定，未改动）
C_i = logical_search_batches（assistant 的逻辑搜索轮数 / 原始 <wiki_search> block 数）
E_i = (Cmax - C_i)/(Cmax - Cmin)   若同一 uid group 内 correct rollout >= 2 且 cost 有差异
    = 0                            否则
wrong rollout -> reward = 0
```

对照变量：只改 reward；model / optimizer / scheduler / RL dataset / GRPO-DAPO trainer /
rollout 配置 / n=8 / wiki 环境 / 评估协议全部保持不变。

---

## 2. Baseline: GAP step60 → step70 with the ORIGINAL reward

两个 baseline 参考点：

| baseline | 说明 | 数据来源 |
|---|---|---|
| **B0 = GAP step60** | 起始 checkpoint（任何训练之前） | 仓库已有的 `DAPO-GAP3B-MHQA-Agent-eval-rl-step-60-4gpu` 评估，用同一套脚本重新聚合（`E1-A/eval_results/step60_reference/`） |
| **B1 = GAP step60 → step70，原始 EM reward** | **匹配 baseline**：同起点、同 10 步、同超参，唯一差别是 reward | E1-0 实验（`experiments/DAPO-GAP3B-MHQA-Agent-E10-step60to70-4gpu/global_step_70`），训练侧用其 rollout audit，评估侧用同一 7-benchmark 协议（`E1-A/eval_results/baseline_*/`） |

B0 与 B1 都在 7 benchmark（nq, triviaqa, popqa, hotpotqa, 2wikimultihopqa, musique,
bamboogle）上以完全相同的协议评估：n=8 的评估 rollout 配置、greedy、temperature 1.0、
`val_batch_size=512`、`max_prompt_length=4096`、`max_response_length=8192`、
`max_model_len=12288`，原始 `mhqa_eval.compute_score_em_batch` 作为 EM 判定。

---

## 3. 方法 (Method)

### 3.1 Reward 实现

| 组件 | 文件 | 说明 |
|---|---|---|
| scorer | `E1-A/code/e1a_scorer.py::compute_score_em_efficiency_batch` | 逐条调用**原始** `compute_score_em` 得到 `A_i`；用传入的 `uid`/`costs` 计算组内 `E_i`；返回 `{score: R, em: A, efficiency: E, cost, ...}` |
| cost 提取 | `E1-A/code/e1a_cost.py` | `logical_search_batches` = `detailed_tool_metrics.tool_call_sequence` 中 distinct `(assistant turn, block index)`；block index 取自 XML parser 的 `call_wiki_search_<block>_<query>` id |
| reward manager | `E1-A/code/e1a_reward_manager.py`（`@register("e1a_batch_shaped")`） | 训练路径：`verify()` 复用原始 `BatchRewardManager.verify` 的字符串构造并额外传 `uid`/`costs`；**验证路径直接 `super().verify()`（与原实现逐位一致）**；`data.batch["acc"]` 在基类写入 shaped reward 后被**还原为原始 EM** |
| 注册 | `E1-A/code/pythonpath_e1a/sitecustomize.py` | PYTHONPATH 注入，零修改原始 registry |
| 自测 | `E1-A/code/e1a_selftest.py` | CPU-only，PASS（见 §3.3） |

**acc / reward 分离**（E1-A_task.md「batch.py 要求」）：原始 `BatchRewardManager.__call__`
会把 scorer 的 `score` 写进 `data.batch["acc"]`。E1-A 的 manager 在基类调用之后把
`acc` 覆盖回 `reward_extra_info["em"]`（原始 EM），因此 **accuracy metric 永远不会被
shaped reward 污染**，而 `token_level_scores`/`token_level_rewards`（→ GRPO advantage、
→ group filter）使用 shaped reward。`batch.py` 本身未被修改。

### 3.2 关于 cost 的一个关键事实（影响可观测性）

`SGLangRollout.__init__` 会替换 tokenizer 的 chat template
(`verl/verl/workers/rollout/sglang_rollout/sglang_rollout.py:289`)，把每个 tool call
渲染成 `<name>arguments</name>`；而 `XMLToolParser.parse_non_stream` 会把
`<wiki_search>a|b|c</wiki_search>` 拆成 3 个 tool call，于是**解码后的 response 里
每个 query 一个 `<wiki_search>` block**（在 E1-0 的 147,200 条 rollout 上实测 block 数
== query 数占 99.79%，logical block 数 == search round 数占 100%）。
所以 E1-A 的 cost **不能**从 response 文本数 block 得到，必须用结构化 metadata 中的
parser block index 重建；评估侧的 rounds 则采用仓库自带的
`Agent/evaluation/mhqa_agent/summarize_eval.py` 约定（`<observation>` 之间的
assistant span 中含 search 即为一轮）。

### 3.3 自测（训练前）

`E1-A/code/e1a_selftest.py`（CPU-only）验证：`R=A(1+0.05E)` 逐条成立；wrong rollout
reward 恰为 0；`<2 correct` 或 `cost 无差异` 时 `E=0`；`acc` 为原始 EM 而 reward tensor
为 shaped；验证实例（`num_examine=1`）行为与原始实现一致且不写 diagnostics；
`compute_score_em_batch` 未被删除（原始文件原样保留）。结果：**PASS**。

---

## 4. 训练过程 (Training)

### 4.1 配置与恢复

* `resume_mode=resume_path`，`resume_from_path=experiments/DAPO-GAP3B-MHQA-Agent-4gpu/global_step_60` —
  日志中 `Setting global step to 60` / `Resuming from …/global_step_60`。
* `total_training_steps=70`，`save_freq=10`，`test_freq=10`；日志含 `step:60 … step:70`，
  `trainer.global_steps` 从 60 递增到 70，**没有 step 71**。
* 解析出的 resolved config：`config/resolved_training_config.json`
  （`reward_manager=e1a_batch_shaped`，`train_name=compute_score_em_efficiency_batch`，
  **`val_name=compute_score_em_batch`（未改动）**，`rollout.n=8`）。
* 训练用 4 GPU，`RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` + E1-0 验证过的运行时修复
  （SGLang GPU 可见性、**延迟** CUDA-IPC reductions patch、checkpoint `map_location="cpu"`）。

### 4.2 完成度与完整性

| 检查 | 结果 |
|---|---|
| python main_ppo 退出码 | **0** |
| checkpoint | `global_step_70/`（model+optimizer+extra_state ×4 ranks + `data.pt`，36 GB），tracker = `70` |
| step 序列 | 60,61,…,70 全部出现 |
| full-state resume | `lr_scheduler.last_epoch` 60→70，RNG 状态存在 |
| 训练窗口错误扫描 | CUDA OOM 0 / NCCL error 0 / traceback 0 / wiki 工具失败 0（仅 6 条无害 `c10d` 警告） |
| 非有限 metric | 0（解析 621 个 metric 值） |
| VERDICT | **PASS**（`run_*/analysis/post_train_integrity.md`） |

训练墙钟：02:49:02 → 05:42:31（≈2 h 53 m，含启动与两次验证；tqdm 计 10 个训练 step
用时 2:38:28）。

### 4.3 Reward / efficiency 诊断（必需项）

聚合自 `run_*/diagnostics/*.jsonl`（109 个 generation batch，139,520 条 rollout，
17,440 个 n=8 group），并与 B1（E1-0 审计，115 batch / 147,200 rollout / 18,400 group）对比：

| 指标 | E1-A（shaped） | B1（原始 EM） | Δ |
|---|---|---|---|
| generation batches | **109** | 115 | **−6 (−5.2%)** |
| rollouts / groups | 139,520 / 17,440 | 147,200 / 18,400 | — |
| EM（batch 内均值） | **0.45277** | 0.45391 | **−0.00114 (−0.11 pp)** |
| shaped reward 均值 | 0.45286 | — | — |
| efficiency 均值 | 0.001742 | — | — |
| E>0 rollout 数 / 比例 | **244 / 0.175%** | — | — |
| rounds 均值 | 1.43288 | 1.44073 | −0.00785 (−0.54%) |
| queries 均值 | 1.98378 | 1.98976 | −0.00598 (−0.30%) |
| parallel factor 均值 | 1.51280 | 1.50911 | +0.00369 |
| turns 均值（assistant 消息数） | 2.43278 | 2.44059 | −0.00782 |
| response tokens 均值 | 1275.68 | 1277.76 | −2.08 |
| zero-search 比例 | 0.0172% | 0.0163% | ≈ |
| 训练 batch 保留 group（baseline 规则） | 294 | 340 | （rollout 总量少 5%） |
| 训练 batch 保留 group（shaped 规则） | **349** | — | — |
| 其中由 shaping 新增保留 | **55** | — | — |
| efficiency-active group | 58 | 95 | — |
| efficiency supervision ratio | **16.6%** | — | — |

`reward distribution` / `efficiency bonus distribution` / `logical_search_batches` /
`search_queries` / `parallel_factor` / `group_correct_count` / `filter retention`
的完整分布见 `run_*/analysis/{reward_distribution,efficiency_bonus_distribution,
search_behavior,group_correct_count,filter_retention}.json` 与 `training_diagnostics.md`。

**逐 step（E1-A）**

| step | rollouts | groups | EM | reward | E 均值 | E>0 率 | rounds | queries | base kept | shaped kept | newly added |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 61 | 15,360 | 1,920 | 0.4550 | 0.4551 | 0.00241 | 0.25% | 1.4436 | 1.9844 | 28 | 37 | 9 |
| 62 | 14,080 | 1,760 | 0.4509 | 0.4509 | 0.00000 | 0.00% | 1.4396 | 2.0000 | 34 | 34 | 0 |
| 63 | 19,200 | 2,400 | 0.4523 | 0.4524 | 0.00167 | 0.17% | 1.4121 | 1.9608 | 31 | 37 | 6 |
| 64 | 12,800 | 1,600 | 0.4587 | 0.4588 | 0.00266 | 0.27% | 1.4571 | 2.0036 | 31 | 36 | 5 |
| 65 | 11,520 | 1,440 | 0.4650 | 0.4651 | 0.00191 | 0.19% | 1.4431 | 1.9867 | 28 | 34 | 6 |
| 66 | 12,800 | 1,600 | 0.4675 | 0.4676 | 0.00141 | 0.14% | 1.4349 | 1.9895 | 30 | 35 | 5 |
| 67 | 12,800 | 1,600 | 0.4413 | 0.4414 | 0.00219 | 0.22% | 1.4464 | 2.0004 | 26 | 32 | 6 |
| 68 | 15,360 | 1,920 | 0.4421 | 0.4422 | 0.00189 | 0.19% | 1.4350 | 1.9876 | 28 | 36 | 8 |
| 69 | 12,800 | 1,600 | 0.4489 | 0.4490 | 0.00109 | 0.11% | 1.3862 | 1.9638 | 30 | 34 | 4 |
| 70 | 12,800 | 1,600 | 0.4493 | 0.4494 | 0.00227 | 0.23% | 1.4390 | 1.9702 | 28 | 34 | 6 |

**shaped reward 确实进入了优化目标**：训练日志的 `critic/score/max = 1.050`
（10 步中 8 步；step62 因该步无 active group 而为 1.000），即
`A(1+0.05E)` 上限 1.05 出现在实际训练 batch 中；`critic/advantages/max = 2.475`。
`actor/entropy` 0.906–0.979，`actor/grad_norm` 0.777（step70），无 NaN/OOM。

---

## 5. 完整 7 benchmark 结果 (Evaluation)

评估协议与原 GAP 评估完全一致（复用未修改的
`Agent/evaluation/mhqa_agent/eval_mhqa_agent_4gpu_common.sh`）。
E1-A step70 的结果：`E1-A/eval_results/20260911_054537/`
（`accuracy.json` / `benchmark_results.json` / `trajectory_metrics.json` /
`search_round_statistics.json` / `parallel_statistics.json` / `eval.log`）。

### 5.1 E1-A step70 vs GAP step60（起点，B0）

| benchmark | E1-A EM | step60 EM | Δ EM | E1-A rounds | step60 rounds | Δ rounds | E1-A queries | Δ queries |
|---|---|---|---|---|---|---|---|---|
| nq | 0.3780 | 0.3740 | +0.0040 | 1.264 | 1.263 | +0.001 | 1.923 | −0.004 |
| triviaqa | 0.5630 | 0.5630 | 0.0000 | 1.284 | 1.281 | +0.003 | 1.862 | +0.001 |
| popqa | 0.3970 | 0.4010 | −0.0040 | 1.501 | 1.512 | −0.011 | 1.891 | −0.017 |
| hotpotqa | 0.3970 | 0.3970 | 0.0000 | 1.615 | 1.604 | +0.011 | 2.121 | +0.004 |
| 2wikimultihopqa | 0.4380 | 0.4470 | −0.0090 | 1.951 | 1.937 | +0.014 | 2.505 | +0.010 |
| musique | 0.1680 | 0.1750 | −0.0070 | 2.212 | 2.198 | +0.014 | 2.576 | +0.014 |
| bamboogle | 0.4480 | 0.4480 | 0.0000 | 1.856 | 1.808 | +0.048 | 2.040 | +0.072 |
| **macro** | **0.3984** | **0.4007** | **−0.0023** | **1.6691** | **1.6546** | **+0.0145** | **2.1312** | **+0.0193** |
| micro | 0.4317 | 0.4351 | −0.0033 | — | — | — | — | — |

### 5.2 E1-A step70 vs **matched baseline B1**（GAP step60→70，原始 reward）

`E1-A/eval_results/baseline_20260911_072826/`（同一 7-benchmark 协议、同一 seedless greedy
解码、同一 51,201 个样本）。

| benchmark | E1-A EM | B1 EM | Δ EM | E1-A rounds | B1 rounds | **Δ rounds** | rounds ↓ 比例 | rounds ↑ 比例 |
|---|---|---|---|---|---|---|---|---|
| nq | 0.3780 | 0.3730 | **+0.0050** | 1.264 | 1.299 | **−0.035** | 9.4% | 7.8% |
| triviaqa | 0.5630 | 0.5640 | −0.0010 | 1.284 | 1.302 | **−0.018** | 8.9% | 7.5% |
| popqa | 0.3970 | 0.3980 | −0.0010 | 1.501 | 1.524 | **−0.023** | 12.8% | 11.2% |
| hotpotqa | 0.3970 | 0.3940 | **+0.0030** | 1.615 | 1.645 | **−0.030** | 10.8% | 8.5% |
| 2wikimultihopqa | 0.4380 | 0.4420 | −0.0040 | 1.951 | 2.004 | **−0.053** | 13.9% | 10.8% |
| musique | 0.1680 | 0.1690 | −0.0010 | 2.212 | 2.252 | **−0.040** | 18.4% | 14.5% |
| bamboogle | 0.4480 | 0.4640 | −0.0160 | 1.856 | 1.872 | **−0.016** | 8.8% | 6.4% |
| **macro** | **0.3984** | **0.4006** | **−0.0021** | **1.6691** | **1.6997** | **−0.0306 (−1.80%)** | — | — |
| micro | 0.4317 | 0.4325 | −0.0008 | 1.5977* | 1.6296* | **−0.0318 (−1.95%)** | — | — |

\* sample-weighted over all 51,201 evaluation samples.

**逐 prompt 配对检验**（`E1-A/eval_results/comparison_vs_baseline/paired_comparison.md`；
两次评估都是 greedy 解码，因此同一个 prompt 的两次回答可直接配对；popqa 的重复 prompt
被排除，剩 49,947 对）：

| 指标 | E1-A | B1 | Δ | 显著性 |
|---|---|---|---|---|
| search rounds | 1.5999 | 1.6332 | **−0.0334 (−2.04%)** | rounds 减少 5,947 条 vs 增加 4,868 条，**符号检验 p = 3.2×10⁻²⁵** |
| search queries | 2.1085 | 2.1346 | −0.0260 | — |
| response chars | 5,526.5 | 5,573.8 | −47.4 | — |
| EM | 0.43280 | 0.43362 | −0.0008 | EM 上升 2,480 条 vs 下降 2,521 条，**p = 0.56（不显著）** |

**结论（§1 目标的答案）**：相对"同样 10 步、原始 reward"的匹配 baseline，E1-A 在
**全部 7 个 benchmark 上降低了 sequential search depth**（macro −1.8%，配对 −2.0%，
p≈10⁻²⁵），而 **accuracy 统计上不变**（macro −0.22 pp，配对 −0.08 pp，p=0.56）。
搜索轮数的下降伴随着 `parallel factor` 的小幅上升（1.3884→1.4032，+1.1%），即模型把
"更多轮串行"换成了"更少轮、每轮略多并行"——这正是该 reward 的设计意图。

### 5.3 汇总对比（宏观平均 + 必备指标）

| 指标 | E1-A step70 | B0 (step60) | B1 (step60→70, 原始 reward) |
|---|---|---|---|
| EM (macro) | 0.3984 | 0.4007 | 0.4006 |
| EM (micro) | 0.4317 | 0.4351 | 0.4325 |
| search rounds (macro) | **1.6691** | 1.6546 | 1.6997 |
| search queries (macro) | 2.1312 | 2.1120 | 2.1459 |
| turns（全部 assistant 消息，macro） | 2.6631 | 2.6480 | 2.6917 |
| response tokens（macro） | 322.18 | 318.29 | 322.70 |
| parallel factor（macro） | 1.4032 | 1.4001 | 1.3884 |
| parallel sample rate | 40.63% | 40.17% | 39.44% |
| multi-query-round rate | 40.63% | 40.17% | 39.44% |
| zero-search rate | 0.0293% | 0.0273% | 0.0312% |
| no `<answer>` rate | 0.568% | 0.580% | 0.744% |
| filter retention（训练侧） | 349 shaped-kept / 294 baseline-kept，+55 | — | 340 baseline-kept / 18,400 groups |

读法：B0 是起点，B1 是"同 10 步但用原始 reward"。E1-A 相对 **B1** 才是控制变量比较
（rounds −1.8%，EM −0.22 pp）；相对 B0 则叠加了"10 步训练本身"的影响（B1 相对 B0 的
rounds 是 **+2.7%**，即原始 reward 训练 10 步反而略微变长，而 E1-A 把它压回去了）。

注：EM 表使用评估日志的 `val-core/<ds>/reward/mean@1`（与仓库自带
`summarize_eval.py` 一致）；`popqa` 存在重复 prompt，其 log 值与 dump 逐样本均值相差
~0.024（verl 按 prompt 分组产生 mean@1/mean@N），`accuracy.json` 同时保留两者。

---

## 6. Safety analysis

### 6.1 Query packing（本轮重点）

在 training-side 的 active group（≥2 correct 且 correct cost 有差异）内比较
efficient（cost=cmin）与 inefficient（cost=cmax）correct rollout：

| | E1-A | B1 (E1-0) |
|---|---|---|
| active groups | 58 | 95 |
| efficient rounds / inefficient rounds | **1.554 / 2.406** | 1.488 / 2.492 |
| efficient queries / inefficient queries | **2.074 / 2.866** | 2.040 / 2.932 |
| efficient tokens / inefficient tokens | 1,205 / 1,830 | 1,343 / 1,829 |
| **packing-suspect groups**（rounds 减少但 query 未减少） | **2 / 58 = 3.45%** | 4 / 95 = 4.21% |

结论：**E1-A 训练后 packing 并没有增加**（3.45% < 4.21%）；efficient rollout 同时
减少 rounds(−35%)、queries(−28%)、tokens(−34%)，说明"省 round"在本 checkpoint 上
仍然是真实的效率提升，而不是把 query 塞进更少的 block。
评估侧同样没有恶化：`parallel factor` 1.4032 vs step60 1.4001，
`multi-query-round rate` 40.63% vs 40.17%，与 B1 的差异在噪声量级。

**残余风险**：reward 只按 block 计费，理论上可通过"一个 block 塞很多 query"套利。
E1-A 评估中仍存在极端个案（nq 有 1 条 24 queries/round、2wiki 有 2 条 pf=6），
但数量与 baseline 同量级且集中在错误轨迹上（`A=0 ⇒ reward=0`）。E1-B 必须继续监控
`parallel_factor` 与 `search_queries/round`。

### 6.2 Zero-search

| | E1-A | B1 | B0 |
|---|---|---|---|
| 训练侧 zero-search 比例 | 0.0172% | 0.0163% | — |
| 训练侧 zero-search 且 correct | **0** | 0 | — |
| 评估侧 zero-search 比例 | 0.0293% | 0.0312% | 0.0273% |

没有出现"用不搜索换取效率奖励"的迹象（正确的 zero-search 为 0；shaped reward 对
`A=0` 的轨迹恒为 0，zero-search 本身不带来任何收益）。

### 6.3 Malformed output

| | E1-A | B1 | B0 |
|---|---|---|---|
| 训练侧 `metadata_valid=false` | 0（139,520 条） | 0（147,200 条） | — |
| 训练侧正常 `</answer>` 结束 | 99.7%（139,158/139,520）；max_turns 305；overlong 31 | 99.7% | — |
| 评估侧无 `<answer>` 比例 | **0.568%** | 0.744% | 0.580% |

### 6.4 Reward distribution

* 训练 batch 内 shaped reward：`critic/score/mean` 0.526–0.666（保留 batch 本身偏向
  高正确率 group），`max=1.050`，`min=0.000`；
* 全部 139,520 条 rollout：reward 均值 0.45286、非零比例 45.29%（= EM 比例），
  efficiency>0 的 rollout 仅 0.175%（244 条），efficiency 分布集中在 0.1–1.0 的
  离散档位（`cost gap` 多为 1，因此 `E∈{0,1}` 占主导）；
* `critic/advantages` ∈ [−2.475, 2.475]（GRPO 组内标准化），无 NaN/Inf。

---

## 7. 最终判断：是否建议进入 E1-B step120

### 结论：**建议进入 E1-B（step70 → step120）**

E1-A 的核心假设已经得到支持：相对"同样 10 步、原始 reward"的匹配 baseline，
**7/7 benchmark 的 sequential search depth 都下降**（macro −1.8%、逐 prompt 配对
−2.0%、符号检验 p≈3×10⁻²⁵），而 **accuracy 在统计上不变**（macro −0.22 pp；
逐 prompt 配对 −0.08 pp，p=0.56），且没有引入任何安全回退。

**支撑 GO 的证据**

| # | 证据 | 数值 |
|---|---|---|
| 1 | 搜索轮数下降（**核心目标**） | macro 1.6691 vs 1.6997（−1.80%）；sample-weighted 1.5977 vs 1.6296（−1.95%）；逐 prompt 配对 −0.0334（−2.04%），p=3.2×10⁻²⁵；**7/7 benchmark 全部下降** |
| 2 | accuracy 保持 | EM macro −0.22 pp（0.3984 vs 0.4006）、micro −0.08 pp；2 个 benchmark 上升（nq +0.5 pp、hotpotqa +0.3 pp）、5 个微降（最大 −1.6 pp 在 n=125 的 bamboogle） |
| 3 | 搜索 query 也下降 | macro 2.1312 vs 2.1459（−0.69%）；配对 −0.0260 |
| 4 | 机制确实接通 | `critic/score/max=1.050`（8/10 步）、244 条 rollout 带非零 E、**55 个 all-correct group 由"被过滤丢弃"变为参与训练**（349 vs 294 retained，+18.7%），0 个 group 被误删 |
| 5 | 无 reward hacking | packing-suspect 3.45%（baseline 4.21%）；zero-search 且 correct = **0**；`metadata_valid=false` = 0；无 `<answer>` 比例 0.568%（baseline 0.744%，E1-A 更好） |
| 6 | 计算成本略降 | 凑满 32-prompt batch 用 109 个 generation batch（baseline 115，−5.2%） |
| 7 | 原始代码零改动、baseline 仍可运行 | 19/19 原始文件 sha256 未变 |

**为什么仍然要带判据进入 E1-B（残余不确定性）**

* 效应量小但方向一致：轮数只降了 ~2%，且监督覆盖仍然稀疏（efficiency-active group
  占 0.33%，带非零 E 的 rollout 占 0.175%）。50 步是否会把 −2% 扩大，还是会饱和，
  目前无法从 10 步外推。
* 下降的一部分来自 `parallel factor` 上升（1.3884 → 1.4032，+1.1%），即
  "更少轮 + 每轮略多并行"。这符合设计意图，但需要继续监控它是否演变为
  query packing（评估里仍有 24 queries/round 的极端个案）。
* EM 的 macro 差 −0.22 pp 虽然不显著，但方向为负；需要确认延长训练后不会累积成
  真实退化。

**E1-B 建议设计（step70 → step120）**

```
起点      = E1-A step70 checkpoint（reward/配置保持不变，只延长 horizon 到 step120）
reward    = R_i = A_i*(1 + 0.05*E_i)，cost = logical_search_batches（不变）
对照      = 继续与 B1 口径一致（同起点同协议；若 E1-B 需要更长的原始 reward 对照，
            可用 GAP 自身 step120 checkpoint 的既有评估结果做趋势参考）
监控      = 每 step：EM、rounds/queries/parallel_factor 分布、zero-search、
            packing-suspect 比例、efficiency supervision ratio（E1-A 的 diagnostics 已就绪）
成功判据  = macro EM 下降不超过 0.5 pp，且 (a) macro retrieval rounds 相对 E1-A
            step70 再降 >= 1.5%，或 (b) 多跳 benchmark（hotpotqa/2wiki/musique/
            bamboogle）中 1-round 解比例再升 >= 3 pp
中止判据  = EM 相对 E1-A step70 累计下降 > 1 pp；或 packing-suspect 比例 > 10%；
            或 parallel_factor 均值上升 > 5%
若仍无进一步效应 = 说明 λ=0.05 的稀疏监督在 10 步内已基本释放完毕，此时应提高
            信号密度（把 shaping 限定到 multi-round 才可能正确的 group，或把 λ 提到
            0.10–0.20）而不是继续延长 horizon
```

理由：E1-A 已经证明该 reward **安全且有效（小幅但显著）**，因此下一步的自然问题是
"效果会不会随训练继续累积"，这需要 step120 这一次实验来回答；同时给出明确的中止/成功
判据，避免为一个小效应量无限投入算力。

---

## 附录 A. 产物清单

| 路径 | 内容 |
|---|---|
| `E1-A/run_e1a_train.sh` | 训练入口（step60→70，shaped reward，含 preflight / run dir / watchdog / 后检查） |
| `E1-A/run_e1a_eval_step70.sh` | 评估入口（7 benchmark，E1-A step70） |
| `E1-A/scripts/e1a_eval_baseline_step70.sh` | 匹配 baseline（E1-0 step70）评估入口 |
| `E1-A/run_20260911_024902/stdout.log`, `stderr.log` | 训练 stdout/stderr |
| `…/config/` | `resolved_training_config.{txt,json}`、`reward_config.json`、`launch_command.txt`、`git_diff.txt`、`code_manifest.txt` |
| `…/preflight/` | `preflight_train.txt`、`post_train_check.txt`、`code_manifest.txt`、`original_files_manifest.txt`（+ after-run 复核） |
| `…/diagnostics/` | `rollouts_pid*.jsonl`（139,520）、`groups_pid*.jsonl`（17,440）、`calls_pid*.jsonl`（109） |
| `…/analysis/` | `training_summary.json`、`reward_distribution.json`、`efficiency_bonus_distribution.json`、`search_behavior.json`、`group_correct_count.json`、`filter_retention.json`、`safety_analysis.json`、`training_comparison.json`、`training_diagnostics.md`、`post_train_integrity.{md,json}` |
| `…/checkpoints_manifest/step70.md` | step70 checkpoint manifest（真实 checkpoint 不复制进 E1-A） |
| `E1-A/eval_results/20260911_054537/` | E1-A step70 评估（5 个必需 JSON + eval.log） |
| `E1-A/eval_results/baseline_*/` | 匹配 baseline 评估（同一套 5 个 JSON + eval.log） |
| `E1-A/eval_results/step60_reference/` | GAP step60 起点评估（同一套 JSON） |
| `E1-A/eval_results/comparison_vs_*/` | `comparison.{json,md}` 对比表 |
| `E1-A/preflight/` | `wiki_service_status.{log,json}`、`original_files_manifest.txt`、`original_files_reverify_after_e1a.txt` |
| `E1-A/issues.log` | 执行中发现的问题与修复记录 |
| `E1-A/README.md` | 目录说明与复现步骤 |

## 附录 B. 原始代码完整性

`E1-A/preflight/original_files_reverify_after_e1a.txt`：19/19 原始文件
（GAP 训练/评估脚本、`batch.py`、`mhqa_train.py`、`mhqa_eval.py`、`sglang_rollout.py`、
`schemas.py`、`ray_trainer.py`、`reward.py`、`core_algos.py`、`fsdp_checkpoint_manager.py`、
`environment.sh`，以及 E1-0 的代码）
sha256 **全部未变**；所有新增代码/数据/报告都在 `E1-A/` 下。

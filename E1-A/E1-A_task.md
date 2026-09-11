你现在负责完成 GAP 项目的 E1-A 实验。

目标：
基于已经完成的 E1-0 和 E1-0.5 结果，在最小代码修改原则下验证：

Success-conditioned group-relative efficiency reward

是否能够在保持 accuracy 的同时降低 sequential search depth。


========================
实验约束（必须遵守）
========================

1. 不修改原始 GAP 主代码。
2. 采用复制方式创建 E1-A 实验分支。
3. 所有 E1-A 文件必须保存到：

E1-A/

目录。

4. 原始 baseline reward、训练脚本、评估脚本必须保持可运行。


========================
实验配置
========================

起始 checkpoint：

step60


训练：

step60 -> step70

10 RL steps。


保持不变：

- model
- optimizer
- scheduler
- RL dataset
- GRPO/DAPO trainer
- rollout configuration
- n=8
- wiki environment
- evaluation protocol


唯一修改：

reward function。


========================
Reward 修改
========================

新增：

compute_score_em_efficiency_batch()


不要删除：

compute_score_em_batch()


Reward:

R_i = A_i * (1 + 0.05 * E_i)


其中：

A_i:

EM correctness。


cost:

logical_search_batches


对于同一个 uid group:

如果：

correct rollout 数 < 2

或者：

correct rollout cost 无差异：

E_i = 0


否则：

E_i =
(Cmax - Ci)/(Cmax-Cmin)


wrong rollout:

reward=0


========================
batch.py要求
========================

修改 reward pipeline：

必须区分：

acc

和

reward


acc:

保持原始 EM。


reward:

使用 shaped reward。


不要让 shaped reward 写入 accuracy metric。


========================
训练入口
========================

从：

Agent/train/mhqa_agent/rl/run_gen_bs_supervisor.sh


复制生成：

E1-A/run_e1a_train.sh


启动前检查：

1.
step60 checkpoint 是否存在。

2.
dataset 是否存在。

3.
reward scorer 是否存在。


输出禁止覆盖。

自动生成：

E1-A/run_timestamp/


保存：

- stdout.log
- stderr.log
- config
- git diff
- reward config
- checkpoint step70
- training metrics


========================
Wiki环境要求
========================

启动训练前：

1.
检查 retriever 环境 wiki service。


如果已经运行：

直接复用。


如果没有：

启动。


保存：

wiki_service_status.log


然后：

进入 parallel-agent 环境：

source environment.sh


之后才能启动 RL。


========================
评估
========================

复制：

Agent/evaluation/mhqa_agent/eval_mhqa_agent_rl_step_70_4gpu.sh


生成：

E1-A/run_e1a_eval_step70.sh


评估：

E1-A checkpoint step70


输出：

E1-A/eval_results/run_timestamp/


保存：

- accuracy.json
- benchmark_results.json
- trajectory_metrics.json
- search_round_statistics.json
- parallel_statistics.json
- eval.log


========================
诊断日志
========================

训练和评估必须额外保存：

- reward distribution
- efficiency bonus distribution
- logical_search_batches
- search_queries
- parallel_factor
- group_correct_count
- filter retention statistics


========================
最终报告
========================

自动生成：

E1-A/FINAL_REPORT.md


必须包含：

1.
实验目标

2.
Baseline:

GAP step60→70 original reward


3.
方法:

Success-conditioned group-relative efficiency reward

lambda=0.05

cost=logical_search_batches


4.
训练过程

5.
完整7 benchmark结果


至少比较：

EM

search rounds

search queries

turns

tokens

parallel factor

filter retention


6.
Safety analysis:

query packing

zero-search

malformed output

reward distribution


7.
最终判断：

是否建议进入 E1-B step120


========================
执行方式
========================

你可以一次性完成全部任务。

中途无需向我汇报。

如果遇到问题：

1.
记录问题到：

E1-A/issues.log


2.
自行修复。

3.
继续执行。

完成后只输出：

- 修改内容总结
- 实验运行状态
- FINAL_REPORT.md 路径
- 是否建议进入 E1-B
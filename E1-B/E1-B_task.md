你现在负责完成 GAP 项目的 E1-B 实验。

实验目标：

从已有 E1-A step70 checkpoint 继续进行强化学习训练，到 step120。

唯一实验变量：

success-conditioned group-relative efficiency reward。

保持 GAP/E1-A 其他设置完全不变。

==============================
一、实验定义
==============================

E1-B:

Start:
E1-A step70 checkpoint

End:
step120

Reward:

R_i = A_i * (1 + 0.05 * E_i)

其中：

A_i:
original EM correctness reward

E_i:
group-relative efficiency bonus

cost:
logical_search_batches


禁止修改：

- GRPO algorithm
- advantage estimator
- optimizer
- scheduler
- RL dataset
- SFT model
- rollout configuration
- wiki environment
- evaluation protocol


==============================
二、代码管理
==============================

不要直接修改原 GAP 主代码。

创建独立 E1-B 分支目录。

所有修改必须保存在：

E1-B/

目录结构：

E1-B/
├── code_changes/
├── scripts/
├── configs/
├── checkpoints/
├── logs/
├── evaluations/
├── diagnostics/
└── FINAL_REPORT.md


==============================
三、启动前检查
==============================

执行训练前必须自动检查：

1.
E1-A step70 checkpoint 是否存在。

如果不存在：
停止并报告。

禁止自动重新训练。


2.
检查 E1-B 输出目录。

禁止覆盖已有结果。

如果存在：
创建新的 run_xxx 子目录。


3.
检查 retriever wiki service。

如果已经运行：
复用。

如果未运行：
启动。

记录：

logs/wiki_service.log


4.
进入 parallel-agent 环境前：

source environment.sh


==============================
四、代码修改要求
==============================

优先复用 E1-A reward 实现。

不要重新设计 reward。

仅确认：

compute_score_em_efficiency_batch

正常工作。


如果需要修改：

batch.py:

确保：

RL reward
和
accuracy metric

分离。

reward:
用于GRPO

acc:
保持真实EM


不要污染 validation metric。


==============================
五、训练过程
==============================

继续训练：

step70 -> step120


每10 step保存：

checkpoint-80
checkpoint-90
checkpoint-100
checkpoint-110
checkpoint-120


保存：

reward statistics

efficiency statistics

training logs


每个checkpoint记录：

mean reward

mean EM

mean efficiency bonus

active efficiency groups

retained groups

mean search batches

mean queries

parallel factor


==============================
六、Evaluation
==============================

保持原 GAP evaluation script。

不要修改benchmark定义。


至少评估：

step120


如果计算允许：

step80
step90
step100
step110


保存：

E1-B/evaluations/


包括：

results.json

benchmark_scores.csv

behavior_metrics.csv


==============================
七、安全检查
==============================

持续监控：

1.

accuracy是否明显下降


2.

search rounds是否下降


3.

queries是否异常增加


4.

zero-search correct 是否出现


5.

query packing 是否异常增加


发现问题：

记录并尝试最小修复。

不要扩大实验变量。


==============================
八、最终报告
==============================

生成：

E1-B/FINAL_REPORT.md


包含：

1. 实验目标

2. 环境信息

3. 起始checkpoint

4. reward定义

5. 修改文件列表

6. 训练过程

7. checkpoint信息

8. evaluation结果

9. 与GAP baseline比较

10. 与E1-A比较

11. 异常和修复记录

12. 最终结论


==============================
执行要求
==============================

可以一次性完成整个E1-B。

中途无需向用户汇报。

如果遇到问题：

自行诊断。

优先最小修改。

所有问题、修复、日志必须保存到：

E1-B/

最终只需要汇报：

- 是否完成
- 输出目录
- 最终结果摘要
- FINAL_REPORT.md位置
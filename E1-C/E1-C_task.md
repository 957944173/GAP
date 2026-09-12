你现在需要在当前 GAP 项目中一次性完成 Experiment E1-C。

这是一个“机制诊断 / offline analysis”实验，不进行新的 RL 训练。请自主检查仓库、定位已有实验产物、编写或修改分析代码、运行所有分析、修复过程中遇到的问题，并最终完成报告。中途不需要向用户汇报或请求确认；遇到可自行解决的问题请记录原因并自行修复。

============================================================
0. 总原则
============================================================

本实验必须遵循以下原则：

1. 控制变量优先：
   - 尽量复用 E1-0、E1-0.5、E1-A、E1-B 已有代码和数据。
   - 优先“复制已有代码 -> 在副本上做最小修改”。
   - 不覆盖、不修改已有 E1-0 / E1-0.5 / E1-A / E1-B 实验结果。
   - 原 GAP / verl 核心训练代码原则上不要修改，因为 E1-C 是 offline diagnosis。

2. 所有 E1-C 新产生的内容必须全部位于仓库根目录：
   E1-C/

   包括：
   - 分析代码
   - 配置/参数
   - logs
   - CSV/JSON
   - Markdown tables
   - 可选图表
   - 临时但有研究价值的中间统计
   - 最终报告
   - 运行命令记录

   不要把 E1-C 新输出散落到其他目录。

3. 如果需要复用已有脚本：
   - 先复制到 E1-C/code/ 或 E1-C/scripts/
   - 再修改副本。
   - 在报告中记录“源文件 -> E1-C 副本”的关系。
   - 尽量保留原分析逻辑，只增加本实验需要的部分。

4. 不重新训练模型，不启动新的 E1 reward RL。
   E1-C 主要应该是 CPU/offline replay。
   如果某些统计确实只能通过已有保存的 rollout/log/checkpoint metadata 获取，请优先从已有实验产物恢复，不生成新的 rollout。

5. 所有结论必须区分：
   - directly observed
   - controlled counterfactual replay
   - inference/hypothesis
   不得把相关性写成因果结论。

============================================================
1. E1-C 的核心研究问题
============================================================

E1-B 中观察到：

- efficiency reward 能降低 sequential search depth；
- 但长程训练后 EM 不如原 GAP。

当前核心假设是：

“修改 reward 后，一部分原先 correctness reward 下 std=0 的 group
变成 std>0，因此通过 GRPO filter，导致 effective RL training
distribution 发生改变；EM 下降可能不仅来自 efficiency objective，
也可能来自训练样本选择机制的改变。”

E1-C 要回答以下问题：

RQ1.
在完全相同的 candidate rollout pool 上：

S_GAP = 原 correctness-only reward 下会被 filter 保留的 groups
S_E1  = E1 shaped reward 下会被 filter 保留的 groups

它们究竟是：

- 相等
- S_GAP ⊂ S_E1
- S_E1 ⊂ S_GAP
- 部分重叠
- 基本不同

必须用真实数据验证，不能只依靠理论判断。

RQ2.
如果 S_E1 比 S_GAP 多，新增的是哪些 group？

重点判断：
- all-correct + efficiency variation
- mixed correctness
- 其他类型

RQ3.
即使 candidate-level 是包含关系，
由于 trainer “不断 generate 直到凑够目标 retained groups”，
新的 filter 是否导致实际 optimizer batch 的 prompt composition 改变？

即：

新 reward 是否因为更早凑满 training batch，
导致 E1 使用一些新激活的 easy/all-correct groups，
而原 GAP 会继续生成并使用更多 mixed/harder groups？

这是本实验最关键的问题之一。

RQ4.
λ 的作用到底是什么？

需要区分：

A. λ=0 vs λ>0 是否改变 retained membership
B. 不同正 λ（如 0.01/0.025/0.05/0.1/0.2）
   是否继续改变 retained membership
C. 若 positive λ retained set 相同，
   λ 是否主要只改变 reward/advantage magnitude

不得把“reward 开启效应”和“λ 数值大小效应”混在一起。

RQ5.
E1-B 的 accuracy drop 是否与 effective training distribution shift
具有一致的统计证据？

这里只能分析支持/不支持该 hypothesis，
不能声称已经证明因果。

============================================================
2. 首先进行数据和代码审计
============================================================

请首先自动检查：

E1-0/
E1-0.5/
E1-A/
E1-B/

以及必要的原 GAP training logs。

找到：

- candidate rollout / group records
- uid / prompt index / dataset row id
- data source（如 NQ / HotpotQA）
- correctness / EM
- logical_search_batches
- structured_search_rounds
- search_queries
- response length/token count
- shaped reward
- baseline reward
- generation batch ordering
- optimizer-step association
- filter before/after records
- 实际 retained prompt/group IDs
- trainer 每个 optimizer step 需要的 retained group 数
- n=8 rollout grouping 信息

优先使用已经在 E1-0.5 验证过的 group reconstruction 和 replay 逻辑。

如果 E1-0.5 中已经存在类似 analysis script：
复制到 E1-C/code/ 后扩展，不重新写一套完全不同的 parser。

生成：

E1-C/data_inventory.md

内容至少包括：

- 使用了哪些源文件
- 每个文件的路径
- 样本/group 数
- 可用字段
- 缺失字段
- 哪些分析可以严格完成
- 哪些分析只能做代理统计
- hash 或其他必要 provenance 信息（能方便记录则记录）

============================================================
3. Analysis A：Same-pool retained-set relationship
============================================================

这是最严格的 controlled analysis。

必须在“完全相同的 rollout group pool”上同时计算：

baseline reward:
R_gap = EM ∈ {0,1}

E1 reward:
R_e1 = A * (1 + λ E)

使用与真实 trainer 完全一致的 std/filter 规则。

默认正式 λ：
λ = 0.05

定义：

S_GAP
S_E1

输出：

| Metric | Value |
|---|---|
| total candidate groups | |
| |S_GAP| | |
| |S_E1| | |
| intersection | |
| GAP-only | |
| E1-only | |
| union | |
| Jaccard | |
| containment: intersection / GAP | |
| containment: intersection / E1 | |

必须自动判断集合关系：

EQUAL
GAP_STRICT_SUBSET_OF_E1
E1_STRICT_SUBSET_OF_GAP
PARTIAL_OVERLAP
DISJOINT/NEAR_DISJOINT

把 exact IDs 导出：

E1-C/results/same_pool_gap_only.csv
E1-C/results/same_pool_e1_only.csv
E1-C/results/same_pool_intersection.csv

============================================================
4. Analysis B：新增/删除 group 类型分析
============================================================

重点分析：

E1-only = S_E1 - S_GAP
GAP-only = S_GAP - S_E1

对于每个 group，统计 baseline 8-rollout correctness count：

k = number of correct rollouts ∈ [0,8]

分类：

- all-wrong: k=0
- mixed: 1<=k<=7
- all-correct: k=8

再统计：

- data source
- mean search rounds
- min/max search rounds
- search-round variation
- mean queries
- queries/round
- mean response length/tokens
- efficiency variation
- 若能获得 dataset row/question ID，也保留

重点回答：

1. E1-only 是否几乎全部为：
   all-correct + efficiency variation？

2. GAP-only 是否存在？
   如果存在，为什么会消失？
   检查：
   - metadata invalid
   - numerical threshold
   - scorer fallback
   - 其他实现原因

若同 pool 上理论预期应该是 S_GAP ⊆ S_E1，
但真实结果不是，必须排查原因，不得忽略。

输出：

E1-C/results/group_type_summary.csv
E1-C/results/e1_only_details.csv
E1-C/results/gap_only_details.csv

============================================================
5. Analysis C：Sequential batch-fill counterfactual replay
============================================================

这是 E1-C 最关键的分析。

必须模拟真实 trainer 的逻辑：

“按原 generation 顺序不断接收 candidate groups，
通过 filter 后累计 retained groups，
直到达到该 optimizer step 所需的 target count，
然后停止 generation。”

从真实 config / logs 自动获取 target retained prompt/group count，
不要硬编码，除非无法获取；若只能硬编码必须在报告中说明来源。

在 SAME ordered candidate rollout stream 上分别 replay：

Scenario GAP:
- correctness-only reward
- 原 filter

Scenario E1:
- λ=0.05 shaped reward
- 相同 filter 实现

对每个能严格 replay 的 optimizer step 输出：

- GAP 需要多少 generation batches 才凑满
- E1 需要多少 generation batches才凑满
- GAP optimizer-batch selected group IDs
- E1 optimizer-batch selected group IDs

并计算：

| Metric | Value |
|---|---|
| GAP selected count | |
| E1 selected count | |
| intersection | |
| GAP-selected-only / displaced | |
| E1-selected-only | |
| Jaccard | |

这里特别注意：

candidate-level 可能有：

S_GAP ⊂ S_E1

但 actual fixed-size optimizer batch 仍可能是：

部分重叠 / 部分替换。

必须明确区分这两个结论。

============================================================
6. Analysis D：被“替换/挤出”的样本是否更难？
============================================================

定义：

D_gap_only =
在 sequential batch-fill replay 中，
会进入 GAP optimizer batch、
但因为 E1 更早填满而没有进入 E1 optimizer batch 的 groups

D_e1_only =
进入 E1 optimizer batch、
但没有进入 GAP optimizer batch 的 groups

比较二者：

1. correctness count k/8
2. all-correct / mixed 比例
3. source：
   - NQ
   - HotpotQA
   - 其他实际存在 source
4. search rounds
5. search queries
6. response length/tokens
7. group success rate = k/8
8. 若有其他已有 difficulty proxy，也可使用，但不要自己发明未经验证的复杂 difficulty score

重点判断：

是否出现：

E1-only：
- 更多 all-correct
- 更高 group success rate
- 相对 easy

而 GAP-only/displaced：
- 更多 mixed-correctness
- group success rate 较低
- 相对 hard

如果出现，报告中可以说：

“supports the hypothesis that reward-induced filtering reallocates
optimization capacity from mixed/harder groups toward solved-but-
efficiency-variable groups.”

不能说：
“this proves the EM drop is caused by ...”

============================================================
7. Analysis E：λ sweep —— 分开分析 membership 与 advantage
============================================================

进行 offline λ replay。

至少：

λ ∈ {
0,
0.01,
0.025,
0.05,
0.1,
0.2
}

若现有 E1-0.5 已分析过某些值，可直接复用并扩展。

对每个 λ：

A. Membership层面：
- total retained groups
- same-pool S_lambda
- 与 λ=0 的 overlap
- 与 λ=0.05 的 overlap
- Jaccard
- new groups
- removed groups

B. fixed-size sequential batch-fill：
- generation batches needed
- optimizer batch membership
- overlap with λ=0
- composition

C. Advantage层面：
复用真实 GRPO advantage 逻辑。
统计：
- mean/std advantage
- correct-vs-wrong gap
- efficient-correct vs inefficient-correct advantage gap
- max absolute advantage
- NaN/Inf
- 若不同正 λ membership 完全一样，要明确报告：
  positive λ primarily changes advantage scale/magnitude rather than
  filter membership（如果真实数据支持这一结论）。

不要因为数值非常接近就省略结果。

输出：

E1-C/results/lambda_membership.csv
E1-C/results/lambda_batchfill.csv
E1-C/results/lambda_advantage.csv

============================================================
8. Analysis F：实际 GAP vs E1-B training run composition
============================================================

如果已有日志足以恢复：
原 GAP step70->120
和
E1-B step70->120

实际每个 optimizer step 使用的 prompt/group IDs，
则进一步做 actual-run comparison。

但必须注意：

从 policy 第一次更新以后，
两条训练分支产生的 rollout 已经不同。

因此 actual-run overlap 不是严格 controlled causal comparison。

请把这一部分单独标记为：

OBSERVATIONAL / POLICY-CONFOUNDED ANALYSIS

统计：

- 每 step retained groups
- source distribution
- correctness-count distribution
- all-correct/mixed ratio
- active efficiency groups
- group success rate
- search rounds
- prompt/data row ID overlap（如果可恢复）

尤其绘制/统计 step 70 -> 120 随训练的变化。

如果无法从已有 artifacts 严格恢复实际 optimizer membership：
不要伪造。
明确写明 unavailable，并说明能做到的最佳 proxy。

============================================================
9. Analysis G：定量判断 distribution-shift hypothesis
============================================================

最终需要根据数据对以下 hypotheses 分别给 verdict：

H1:
E1 reward 在同 candidate pool 上增加了新的 retained groups。

verdict:
SUPPORTED / NOT SUPPORTED / INCONCLUSIVE

H2:
新增 retained groups 主要是 all-correct but efficiency-variable groups。

H3:
由于 fixed optimizer batch / early stop，
新增 groups 会挤出 GAP 原本后续生成的 mixed groups，
因此 actual optimizer training composition 改变。

H4:
被 E1 引入的 groups 平均比被挤出的 GAP groups 更“容易”
（以 k/8、all-correct rate 等可观测指标衡量）。

H5:
不同 positive λ 的主要作用是改变 advantage magnitude，
而不是进一步改变 retained membership。

H6:
E1-B 的 EM drop 与 training-distribution-shift hypothesis 一致。

对 H6 最多写：
SUPPORTED BY ASSOCIATION / NOT SUPPORTED / INCONCLUSIVE

不得写“causally proven”。

============================================================
10. 给出下一实验建议，但不要自行启动下一训练实验
============================================================

E1-C 最终报告必须根据分析结果选择下一步建议。

至少考虑以下三个候选：

Candidate A: Filter-Decoupled Reward
------------------------------------
filter 是否保留 group，仍然只使用原 GAP correctness reward：
std(R_EM) > 0

但对于已经被 baseline filter 保留的 group，
GRPO reward/advantage 使用 E1 efficiency-shaped reward。

目的：
- 完全保持 baseline filter distribution；
- 只改变 retained group 内 correct trajectories 的 efficiency ranking；
- 排除 all-correct revived groups 导致的 distribution shift。

缺点：
efficiency signal 可能显著变少。

若 E1-C 显示 distribution shift 很强，
这是优先候选。

Candidate B: Baseline-Preserving + Capped Efficiency Groups
------------------------------------------------------------
保证原 baseline mixed-correctness groups 的 quota，
同时允许一定比例 all-correct efficiency groups，
例如设置固定上限。

只有在数据明确支持时才推荐，
不要现在自行确定比例。

Candidate C: λ adjustment only
-------------------------------
只有在结果显示：
- retained distribution 基本没变
- 但 advantage magnitude 与 accuracy degradation 更吻合

时才优先推荐简单降低 λ。

如果 E1-C 显示：
所有 positive λ retained set 都一样，
则必须明确指出：
“只调 λ 可能无法解决 filter-distribution shift。”

不要启动 Candidate A/B/C 的训练。
E1-C 到诊断和下一步设计建议为止。

============================================================
11. 结果目录结构
============================================================

请最终整理为类似：

E1-C/
├── README.md
├── FINAL_REPORT.md
├── data_inventory.md
├── code/
│   ├── ...
├── configs/
│   └── analysis_config.json
├── logs/
│   ├── run.log
│   └── errors_and_fixes.md
├── results/
│   ├── summary.json
│   ├── same_pool_set_relationship.csv
│   ├── same_pool_gap_only.csv
│   ├── same_pool_e1_only.csv
│   ├── same_pool_intersection.csv
│   ├── group_type_summary.csv
│   ├── gap_only_details.csv
│   ├── e1_only_details.csv
│   ├── batchfill_overlap_by_step.csv
│   ├── batchfill_gap_only.csv
│   ├── batchfill_e1_only.csv
│   ├── difficulty_comparison.csv
│   ├── lambda_membership.csv
│   ├── lambda_batchfill.csv
│   ├── lambda_advantage.csv
│   └── actual_run_comparison.csv  # 若可恢复
└── figures/
    └── ...                       # 有帮助再生成，不强制

文件名可根据现有仓库代码略作调整，
但所有结果必须留在 E1-C/。

============================================================
12. FINAL_REPORT.md 必须包含
============================================================

最终报告要求尽量完整，使之后只看 FINAL_REPORT.md
就能理解 E1-C 的全部结论。

至少包括：

1. Executive Summary
2. Motivation
3. Exact research questions
4. Source artifacts / provenance
5. Code changes
6. Controlled same-pool set analysis
7. Candidate-level set relationship
8. Sequential fixed-size batch-fill replay
9. Optimizer-batch overlap / replacement analysis
10. GAP-only vs E1-only difficulty/composition
11. λ membership analysis
12. λ advantage analysis
13. Actual-run comparison（如果有）
14. Limitations / confounders
15. Hypothesis verdict table
16. Whether the current hypothesis is supported
17. Recommended next experiment
18. Exact reason for recommendation
19. Reproduction commands
20. List of every generated artifact

报告中必须清楚回答：

A.
reward 修改前后，在 SAME candidate pool 上：
究竟是包含、部分包含还是完全不同？

B.
在 fixed-size optimizer batch 层面：
究竟是包含、部分替换还是完全不同？

C.
λ=0.05 是否本身导致 sample membership 特异变化，
还是任何 λ>0 都基本会激活同样的新 groups？

D.
是否存在证据支持：
新增 all-correct/easy efficiency groups
挤占了原本 correctness-only GRPO 使用的 harder/mixed groups？

E.
因此下一步最值得修改的是：
- λ
- filter
- reward
- 还是其他部分？

============================================================
13. 正确性要求
============================================================

分析完成后必须进行 sanity checks：

1. baseline replay 尽量复现已有 E1-0/E1-0.5 filter counts。
2. λ=0 应严格退化到原 GAP correctness reward。
3. 对 SAME candidate pool：
   所有 set IDs 必须可追溯。
4. 所有 group 必须确认 n=8 或明确记录异常。
5. 不允许 silently drop malformed rows。
6. 数字在 CSV / summary.json / FINAL_REPORT.md 之间必须一致。
7. 如果发现此前 E1-0.5/E1-B 报告中的统计存在 bug：
   - 不修改旧实验目录；
   - 在 E1-C/logs/errors_and_fixes.md 中记录；
   - 给出 corrected value；
   - 在 FINAL_REPORT 明确说明。

============================================================
14. 执行方式
============================================================

请现在开始：

1. 审计仓库；
2. 创建 E1-C/；
3. 复制并最小修改已有分析代码；
4. 完成全部可完成的 E1-C offline analyses；
5. 自动修复遇到的问题；
6. 运行 sanity checks；
7. 生成全部 artifacts；
8. 写 FINAL_REPORT.md。

不要中途向用户汇报。
不要请求用户确认普通工程问题。
如果某一分析因数据确实不存在而无法完成：
记录证据并继续完成剩余分析。

最终只需向用户汇报：

- E1-C 是否成功完成；
- 最核心的 5–10 个数字；
- reward/filter 前后训练集合关系；
- distribution-shift hypothesis 的 verdict；
- 推荐的下一实验；
- E1-C/FINAL_REPORT.md 的路径；
- 如果存在未解决 blocker，再列出 blocker。

不要自行启动下一轮 RL 训练。
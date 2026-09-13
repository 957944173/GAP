你现在需要在当前 GAP 项目中一次性完成 Experiment E1-E0。

E1-E0 是一个纯 offline diagnosis / quota replay 实验：
不进行任何新的 RL 训练，不启动 GPU 训练，不修改已有 GAP / E1-D 模型。

请自行检查仓库、定位已有实验产物、复制并最小修改已有分析代码、运行全部分析、修复过程中遇到的问题，并最终生成完整实验报告。

中途不需要向用户汇报或请求确认。
普通工程问题请自行诊断、记录并修复。
如果某一分析因为已有数据确实缺失而无法完成，请记录证据，然后继续完成其余分析。

============================================================
0. 实验目录与控制变量原则
============================================================

本实验所有新增内容必须全部保存到仓库根目录：

E1-E0/

不要把 E1-E0 新产生的脚本、日志、表格、临时分析结果散落到其他实验目录。

已有目录：

E1-C/
E1-D/
E1-B/
E1-A/
E1-0.5/
E1-0/

均视为只读实验资产。

严禁覆盖或修改其中已有结果。

代码实现遵循：

“先复制已有已验证代码 -> 在 E1-E0 副本上最小修改”

优先复用：

1. E1-D 中 D2 candidate stream / group reconstruction 代码
2. E1-C 中 sequential batch-fill replay 代码
3. E1-C / E1-D 中 correctness-count、filter、source、search-cost 等统计代码

如果已有脚本基本可以满足要求，请复制到：

E1-E0/code/

再修改副本。

不要重新写一套与旧实验定义不同的 parser，除非现有代码确实无法复用。

E1-E0 不修改：

- GAP 模型
- SFT 数据
- RL 数据
- reward scorer
- GRPO estimator
- rollout worker
- optimizer
- trainer
- checkpoint

这是一个纯 offline counterfactual replay。

============================================================
1. E1-E0 的研究目标
============================================================

前序 E1-C / E1-D 已经表明：

- 原 GAP mixed-correctness groups 主要提供 correctness supervision；
- all-correct 但效率存在差异的 groups 可以提供 efficiency supervision；
- unrestricted efficiency-group admission 会改变 optimizer-batch composition；
- E1-D 完全使用 EM filter 后，efficiency supervision 明显减少；
- 下一候选方法是：
  在保持 GAP 主体 correctness training 的基础上，
  以有限 quota 引入 all-correct efficiency-variable groups。

E1-E0 不训练模型。

本实验唯一目标是：

在真实已有 candidate stream 上，
离线模拟不同 efficiency-group quota q，
评估它们会如何改变 optimizer batch composition，
从而选择一个最合理的 q 供后续 E1-E 正式训练。

重点测试：

q ∈ {0, 1, 2, 3, 4}

其中 q 表示：

每个固定 32-group optimizer batch 中，
最多允许多少个
“all-correct + efficiency-variable”
group 替换原 GAP mixed-correctness group。

============================================================
2. E1-E0 必须严格采用的 batch 构造原则
============================================================

这是本实验最重要的控制变量。

Baseline GAP candidate / stopping rule 必须保持不变。

具体逻辑：

A. Mixed-correctness group 定义

对于一个 n=8 rollout group：

k = correct rollout count

若：

1 <= k <= 7

则定义为：

mixed-correctness group

这些是原 GAP correctness training 的主要有效 group。

B. All-correct efficiency group 定义

必须同时满足：

k == 8

且：

正确 rollout 之间存在 efficiency variation：

max(search_cost) > min(search_cost)

其中 search_cost 必须沿用前序已经验证的定义，
优先使用前序 E1-C / E1-D 最终采用的：

logical_search_batches

不要自行换成新 cost。

C. generation stopping rule

必须继续按照原 GAP / E1-D 的 correctness criterion：

按原 candidate stream 顺序读取 generation batches，

持续累计 mixed-correctness groups，

直到 mixed-correctness group 数达到真实 trainer
每个 optimizer step 所需 target count。

target count 必须从真实 config / log 自动读取。
若真实值为 32，应验证后使用，不要仅因为提示词中写了 32 就硬编码。

重要：

all-correct efficiency groups
不能使 generation 提前停止。

也就是说：

q=0,1,2,3,4
必须使用完全相同的 generation stopping point。

这是 E1-E0 的关键控制变量。

D. efficiency-group cache

在读取 candidate stream、等待 mixed groups 填满期间，
同时按照原始 candidate 顺序缓存：

all-correct + efficiency-variable groups。

不能因为 q 不同改变 candidate generation stream。

E. fixed-size optimizer batch

假设 target batch size = B。

对于 quota q：

实际加入：

m = min(q, number_of_available_efficiency_groups)

最终 optimizer batch：

前 B-m 个 mixed groups
+
candidate stream 中最早出现的 m 个 efficiency groups

即：

[M1, M2, ..., M(B-m), E1, E2, ..., Em]

不要：

- 按 difficulty 选择 mixed groups
- 按 reward 选择 mixed groups
- 按 efficiency gap 最大挑 efficiency group
- 按数据源挑 group
- 随机挑 group

必须使用稳定、确定性的 stream order。

这样避免引入新的 selection bias。

F. 被替换的 mixed groups

定义为：

原 q=0 optimizer batch 中最后 m 个 mixed groups。

因此 quota 的含义清晰：

q=1：
最多替换 1/B

q=2：
最多替换 2/B

以此类推。

============================================================
3. 首先完成 provenance / inventory
============================================================

自动检查 E1-D 和必要的 E1-C 产物，确定：

- candidate stream 来源文件
- generation batch ordering
- optimizer-step boundary
- uid / prompt id
- source
- rollout correctness
- n=8 grouping
- logical_search_batches
- search_queries
- response length/tokens
- baseline filter state
- candidate group ordering
- target optimizer group count

优先使用 E1-D D2 所使用的 candidate stream。

生成：

E1-E0/data_inventory.md

必须记录：

- 所有源文件路径
- 使用哪些旧脚本
- 复制到 E1-E0 的哪些新脚本
- candidate group 数
- generation batch 数
- optimizer steps 数
- 可用字段
- 缺失字段
- 数据 provenance
- 能严格 replay 的范围
- 任何限制

如果前序报告中的数字与重新解析不一致，
不要强行匹配旧报告。

应：

1. 检查原因
2. 记录在 errors_and_fixes.md
3. 使用经过 sanity check 的真实结果

============================================================
4. Baseline replay：q=0
============================================================

在进行 quota replay 前，
必须首先严格重放 q=0。

q=0 必须等价于：

原 GAP-style mixed-correctness fixed-size optimizer batch。

检查：

- 每 step generation batches needed
- 每 step retained mixed groups
- optimizer batch size
- selected group IDs
- source composition

尽量复现 E1-C / E1-D 中已有 baseline replay 数字。

若不能复现：
先排查，不要继续 quota 分析直到确认原因。

输出：

E1-E0/results/q0_baseline_replay.csv

============================================================
5. Quota sweep
============================================================

对于：

q ∈ {0,1,2,3,4}

在完全相同 candidate stream 上进行 replay。

每个 optimizer step 记录：

- optimizer_step
- q
- generation_batches_consumed
- candidate_groups_seen
- mixed_groups_available
- efficiency_groups_available
- requested_quota
- actual_efficiency_groups_inserted
- mixed_groups_used
- mixed_groups_replaced
- quota_filled (bool)
- optimizer_batch_size
- source distribution
- mean k/8
- mean search rounds
- mean search queries
- mean response length（若已有）
- efficiency-group source distribution

所有 q 的：

generation_batches_consumed

对于同一 optimizer step 理论上必须完全一致。

如果不一致：
视为实现 bug，
必须排查并修复。

============================================================
6. Quota availability / saturation analysis
============================================================

对每个 q 输出：

- total optimizer steps
- steps with full quota
- full-quota rate
- mean actual efficiency groups / step
- median
- p10 / p50 / p90
- total efficiency groups inserted
- total mixed groups replaced
- replacement fraction
- zero-efficiency-group step rate

重点回答：

q=1、q=2、q=3、q=4
是否在大部分 optimizer steps 都能实际填满。

建议重点报告：

FullQuotaRate(q)

以及：

ActualReplacementRate(q)

不要只报告理论 q/B。

============================================================
7. Mixed group displacement analysis
============================================================

对于每个 q：

比较：

A. 被加入的 all-correct efficiency groups
B. 被替换掉的 mixed-correctness groups

统计：

- count
- data source
- k/8
- search rounds
- search queries
- response length/tokens
- parallel factor（若已有）
- question/prompt ID

Mixed groups 必须报告：

k distribution：

1/8
2/8
...
7/8

重点检查：

quota replacement 是否系统性挤出了：

- 更困难的低 k/8 mixed groups
- 某一特定数据源
- 更长链 multi-hop groups

如果存在明显偏差，
报告中说明风险。

不能因为 replacement 按 stream tail 进行就假设其无偏，
必须用数据验证。

============================================================
8. Efficiency-group composition analysis
============================================================

对于实际会进入 optimizer batch 的 efficiency groups：

统计：

- source
- min search rounds
- max search rounds
- efficiency gap
- search queries
- queries/round
- response length/tokens
- efficient-vs-inefficient trajectory 差异

检查：

是否仍存在 query-packing 风险。

特别统计：

efficient correct trajectory 的 search rounds ↓
是否通常伴随：

- query count 不升
- token 不升
- 或至少没有异常 query explosion

不需要修改 reward，
这里只做诊断。

============================================================
9. 估算 efficiency supervision density
============================================================

对于每个 q 计算：

EfficiencyGroupFraction =
actual efficiency groups /
total optimizer groups

例如如果 B=32：

q=1 理论上限约 3.125%
q=2 理论上限约 6.25%
q=4 理论上限约 12.5%

但必须报告真实 actual fraction。

同时和已有 unrestricted E1-B / E1-C
efficiency-group fraction做对比，
前提是旧结果中有严格可比数据。

输出：

E1-E0/results/quota_supervision_density.csv

============================================================
10. Source balance analysis
============================================================

对于每个 q 比较 optimizer batch 中：

- NQ
- HotpotQA
- 其他真实存在的数据源

相对于 q=0 的变化。

重点检查：

q 增大是否使 source distribution
向某一个数据集明显倾斜。

输出：

E1-E0/results/source_shift_by_quota.csv

============================================================
11. Offline batch-level proxy metrics
============================================================

注意：

E1-E0 没有训练模型，
因此不能声称某个 q 会真正产生多少 EM 或 search-round 改善。

但可以构造简单、透明的 batch-level proxy：

- mixed correctness supervision retained %
- efficiency supervision fraction
- mean group success rate
- all-correct share
- mixed share
- source shift
- displaced mixed mean k/8
- inserted efficiency search-gap magnitude

不要构造复杂的自定义总分，
不要把 proxy 伪装成模型效果预测。

============================================================
12. q 的推荐决策规则
============================================================

最终必须根据真实数据推荐：

q ∈ {1,2,3,4}

但不要机械固定 q=2。

推荐时考虑以下原则：

第一优先：
不能大幅损失 correctness supervision。

第二优先：
需要稳定获得 efficiency supervision。

优先候选最好满足：

1. 大多数 optimizer steps 能填满 quota
2. replacement fraction 明显低于 unrestricted E1-B
3. 被替换 mixed groups 不呈现严重 source/difficulty bias
4. efficiency groups 本身具有真实 search-round variation
5. query-packing 风险较低

可以使用如下定性判断：

若 q=2：
- full-quota rate 很高（例如约 80–90% 或更高）
- replacement fraction 约 6%左右
- 没有严重 source/difficulty bias

则 q=2 是优先正式训练候选。

若 q=2 经常无法填满：
考虑 q=1。

若 q=2 非常稳定且 q=3 仍然低风险：
可以把 q=3 作为后续消融候选，
但第一正式训练仍优先更保守配置。

若 q=1/2 都显示明显 bias：
不要启动 E1-E，
报告风险。

============================================================
13. 必须额外模拟 E1-E 正式 reward 语义
============================================================

虽然 E1-E0 不训练，
但请验证正式 E1-E 方法的 reward 逻辑可以在 replay 数据上正确分类。

对于 mixed groups：

1 <= k <= 7

未来 E1-E 应使用：

R_i = A_i

完全恢复原 GAP correctness reward。

不要对 mixed groups使用 efficiency shaping。

对于 selected all-correct efficiency groups：

k = 8

未来 E1-E 使用：

R_i = 1 + lambda * E_i

lambda 暂定：

0.05

这里只需 offline 计算并 sanity check：

- reward finite
- efficiency ranking 正确
- efficient trajectory reward > inefficient trajectory reward
- group std > 0
- GRPO advantage 可正常计算
- 无 NaN / Inf

复用真实 GRPO advantage implementation，
不要自己写近似公式。

无需 sweep lambda。

lambda=0.05 只是为了与前序实验保持一致。

输出：

E1-E0/results/e1e_reward_sanity.csv

============================================================
14. 关键 sanity checks
============================================================

必须自动验证：

1. 所有合法 group n=8。
2. q=0 严格退化为 baseline replay。
3. q>0 不改变 generation stopping point。
4. 所有 q 的 optimizer batch size 完全相同。
5. actual inserted efficiency groups <= q。
6. inserted efficiency group 必须：
   - k=8
   - efficiency variation > 0
7. mixed group 必须：
   - 1<=k<=7
8. 不允许 duplicate uid/group ID 出现在同一 optimizer batch。
9. 不允许 silently drop malformed records。
10. CSV / JSON / FINAL_REPORT 中数字一致。
11. q 增大时 total replacement 不应反常下降，
    除非受到 quota availability 限制；
    若出现需解释。
12. 所有异常均写入：
    E1-E0/logs/errors_and_fixes.md

============================================================
15. 建议目录结构
============================================================

最终整理为：

E1-E0/
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
│   ├── q0_baseline_replay.csv
│   ├── quota_by_step.csv
│   ├── quota_summary.csv
│   ├── quota_availability.csv
│   ├── mixed_displacement_by_quota.csv
│   ├── efficiency_group_details.csv
│   ├── quota_supervision_density.csv
│   ├── source_shift_by_quota.csv
│   ├── group_success_distribution.csv
│   ├── e1e_reward_sanity.csv
│   └── recommended_quota.json
└── figures/
    └── ...

如有更合理文件名可以调整，
但所有新内容必须全部位于 E1-E0/。

============================================================
16. 可选图表
============================================================

如果容易生成，可输出少量有解释力图表：

1. q vs actual efficiency-group fraction
2. q vs full-quota rate
3. q vs mixed-group replacement fraction
4. q vs source distribution
5. displaced mixed k/8 distribution

不要花大量时间美化图表。

表格和最终数字优先。

============================================================
17. FINAL_REPORT.md 必须完整回答
============================================================

最终报告至少包含：

1. Executive Summary
2. Motivation
3. Relationship to E1-C / E1-D
4. Exact E1-E0 design
5. Data provenance
6. Reused code and minimal modifications
7. Baseline q=0 replay validation
8. q=1/2/3/4 quota availability
9. Generation stopping invariance
10. Optimizer-batch replacement analysis
11. Mixed-group displacement analysis
12. Efficiency-group composition
13. Source balance
14. Query-packing risk
15. E1-E reward sanity check
16. Comparison with unrestricted E1-B if strictly comparable
17. Recommended q
18. Exact reason for q recommendation
19. Risks / limitations
20. Whether E1-E formal training should proceed
21. Proposed E1-E formal configuration
22. Reproduction commands
23. Complete artifact list

必须明确回答：

A.
q=1/2/3/4 实际各有多少 efficiency groups 能进入训练？

B.
q=2 是否在绝大多数 optimizer steps 能填满？

C.
不同 q 替换了多少 mixed-correctness supervision？

D.
被替换 mixed groups 是否比平均 mixed group更难，
或存在 source bias？

E.
加入的 all-correct groups 是否真的提供稳定 efficiency signal？

F.
是否存在明显 query-packing 风险？

G.
哪个 q 最适合第一轮正式 E1-E 训练？

H.
是否建议正式进入 E1-E？

============================================================
18. 最终 E1-E 配置建议
============================================================

如果 E1-E0 结果支持正式训练，
FINAL_REPORT 必须给出一个明确的推荐配置，例如：

start checkpoint:
original GAP step60

optimizer batch target:
保持原配置

quota:
q = X

mixed-group reward:
original EM only

all-correct efficiency-group reward:
1 + 0.05 * E

generation stopping:
only mixed-correctness groups count toward baseline stopping target

efficiency groups:
cached during same candidate stream,
maximum q inserted per optimizer batch

selection:
stream order only

batch size:
保持原值

GRPO:
不修改

rollout n:
不修改

其他 trainer / optimizer / dataset：
全部不修改

注意：
只生成建议，不启动 E1-E 正式训练。

============================================================
19. 执行方式
============================================================

现在开始一次性完成整个 E1-E0：

1. 审计已有 E1-C / E1-D 资产
2. 创建 E1-E0/
3. 复制并最小修改已有代码
4. 完成 baseline replay
5. 完成 q=0/1/2/3/4 replay
6. 完成所有 composition / displacement / source 分析
7. 完成 reward sanity checks
8. 自动排查并修复普通 bug
9. 完成 sanity checks
10. 生成所有结果文件
11. 写 FINAL_REPORT.md

中途不要向用户汇报。

不要启动新的 RL 训练。

最终只需向用户简洁汇报：

- E1-E0 是否成功完成
- candidate / optimizer replay 是否通过 sanity checks
- q=1/2/3/4 的核心数字
- 推荐 q
- 是否建议进入正式 E1-E
- 最关键风险
- E1-E0/FINAL_REPORT.md 路径
- 若存在 blocker，再列出 blocker
你现在需要在当前 Graph-Agent-Planning / GAP 项目中，一次性完成正式实验：

Experiment E1-E — q=2 Solved-Group Gated / Capped Efficiency Training

这是 E1-E0 已经离线预注册并验证的正式在线 RL 实验。

你被允许：
- 自主审计仓库和已有 E1-0/E1-A/E1-B/E1-C/E1-D/E1-E0 产物；
- 自主复制并最小修改已有代码；
- 自主运行 preflight、Wiki 服务检查/启动、RL、7 benchmark evaluation、完整性检查、统计分析；
- 遇到普通工程问题自行诊断和修复；
- 存储空间不足时，在严格安全规则下删除 experiments/ 中确认无用的旧模型 checkpoint；
- 一次性完成整个 E1-E，不需要中途向用户汇报或请求普通工程问题确认。

但不要自行改变 E1-E 的科学设计。
不要训练 q=1/q=3/q=4。
不要修改 λ。
不要增加新的 reward 项。
不要修改原 GAP、E1-B、E1-D、E1-E0 的既有结果。

最终只需完成实验、生成 E1-E/FINAL_REPORT.md，并向用户做一次最终汇报。

======================================================================
0. 最高优先级原则
======================================================================

本实验的最高原则：

1. 最大限度复用已有代码。
2. 采用“复制 -> 改副本”的方式建立 E1-E 实验代码分支。
3. 原项目核心源码原则上保持只读。
4. E1-E 和原 GAP / E1-B / E1-D 的科学变量必须尽可能严格控制。
5. 所有轻量实验代码、日志、诊断、评估结果、分析和最终报告统一保存：
   <repo>/E1-E/
6. 模型 checkpoint 因体积巨大，允许继续使用既有框架的：
   <repo>/experiments/<E1E_EXPERIMENT_NAME>/
   但必须：
   - 在 E1-E/manifests/ 下完整记录；
   - 在 E1-E/checkpoints/ 下建立路径说明或 symlink；
   - 不把多 GB checkpoint 重复复制进 E1-E；
   - 除模型 checkpoint / Hydra 强制输出外，E1-E 特有结果不得散落在其他目录。
7. framework 自动写入 verl/outputs 或 verl/logs 的轻量日志/config，实验结束后必须收集/复制或明确索引到 E1-E/。
8. 不允许 silent overwrite。
9. 不允许 silent data deletion。
10. 所有异常、修复和删除动作都必须写入：
    E1-E/logs/issues_and_fixes.md

======================================================================
1. E1-E 的科学设计已经冻结，不得自行改变
======================================================================

E1-E0 的正式设计：

start checkpoint:
    ORIGINAL GAP full-state checkpoint:
    experiments/DAPO-GAP3B-MHQA-Agent-4gpu/global_step_60

不是：
- E1-A step70
- E1-B step120
- E1-D step120

正式 quota：

    q = 2

optimizer group batch size：

    B = 32

rollout：

    n = 8

对一个 n=8 group，定义：

    k = number of correct rollouts

三类 group：

A. all-wrong:
    k = 0
    不进入训练。

B. mixed-correctness:
    1 <= k <= 7
    这是原 GAP correctness supervision。
    Reward 必须完全恢复原 GAP：
        R_i = A_i
    A_i ∈ {0,1}
    不允许 efficiency shaping。

C. all-correct efficiency-variable:
    k = 8
    且：
        max(logical_search_batches) >
        min(logical_search_batches)

    它才是 efficiency group。

    对该 group：
        C_i = logical_search_batches of rollout i

        E_i =
            (C_max - C_i)
            / (C_max - C_min)

        R_i = 1 + 0.05 * E_i

lambda 固定：
    0.05

不得 sweep λ。

----------------------------------------------------------------------
generation stopping rule
----------------------------------------------------------------------

这点最关键。

必须继续 generation，直到：

    收集到 32 个 mixed-correctness groups

只有 mixed groups：

    1 <= k <= 7

可以计入 stopping target。

all-correct efficiency group：
    只能缓存，
    绝对不能帮助提前满足 32-group stopping condition。

all-wrong：
    忽略。

在达到 32 个 mixed groups时停止 generation。

在同一 stopping prefix 内，维护 efficiency cache：
    所有遇到的 all-correct efficiency-variable groups，
    严格按 candidate stream / generation stream 顺序保存。

定义：

    m = min(q, len(efficiency_cache))
      = min(2, len(cache))

最终 optimizer batch：

    mixed[:32-m] + efficiency_cache[:m]

因此：

若 cache >= 2：

    30 mixed + 2 efficiency

若 cache = 1：

    31 mixed + 1 efficiency

若 cache = 0：

    32 mixed

绝对不要为了填满 q=2 而继续额外 generation。

----------------------------------------------------------------------
selection rule
----------------------------------------------------------------------

Efficiency groups：

    按 stream order 取最先出现的 m 个。

不得：

- 按 efficiency gap 选；
- 按 source 选；
- 按 response length 选；
- 按 difficulty 选；
- 按 reward 大小选。

被替换的 mixed groups：

    是原 32 个 mixed groups中最后 m 个。

不得主动挑“最难”“最容易”“reward 最低”等 group。

======================================================================
2. 必须首先阅读和复用的已有实验
======================================================================

正式动代码前，先审计：

E1-D/
E1-E0/

重点阅读：

E1-D/FINAL_REPORT.md
E1-D/scripts/e1d_preflight.sh
E1-D/scripts/run_e1d_train.sh
E1-D/scripts/run_e1d_train_60_70.sh
E1-D/scripts/run_e1d_train_70_120.sh
E1-D/scripts/e1d_eval_common.sh
E1-D/scripts/run_e1d_eval_step.sh
E1-D/scripts/run_e1d_eval_70.sh
E1-D/scripts/run_e1d_eval_120.sh
E1-D/scripts/e1d_integrity.py
E1-D/scripts/run_e1d_analysis.sh
E1-D/scripts/run_e1d_all.sh

E1-D/code/
特别是：
- trainer patch
- reward manager
- scorer
- cost parser
- diagnostics
- sitecustomize
- checkpoint/resume fixes
- watchdog/safety tooling

以及：

E1-E0/FINAL_REPORT.md
E1-E0/code/e1e_common.py
E1-E0/code/e1e_quota_replay.py
E1-E0/results/recommended_quota.json
E1-E0/results/optimizer_batch_members_by_quota.csv
E1-E0/results/e1e_reward_sanity.csv
E1-E0/configs/analysis_config.json

E1-E0 是 quota semantics 的规范定义。

E1-D 是在线运行工程模板。

----------------------------------------------------------------------
代码复用原则
----------------------------------------------------------------------

优先：

复制 E1-D 的工程框架到 E1-E 后重命名为 e1e_*，
再做最小修改。

不要重新写一套完全不同的训练 pipeline。

例如：

E1-D/code/e1d_*.py
    ->
E1-E/code/e1e_*.py

E1-D/scripts/e1d_*.sh/.py
    ->
E1-E/scripts/e1e_*.sh/.py

必须记录：

E1-E/manifests/copied_from.json

说明：
- E1-E 文件
- 来源文件
- 修改目的

======================================================================
3. 不直接修改原 GAP 核心源码
======================================================================

原则上禁止直接修改：

verl/verl/trainer/ppo/ray_trainer.py
verl/verl/workers/reward_manager/batch.py
verl/verl/utils/reward_score/mhqa_train.py
verl/verl/workers/rollout/sglang_rollout.py
以及 E1-D/E1-B 原实验文件。

优先继续沿用 E1-D 的：

PYTHONPATH + sitecustomize / runtime patch

方式注入 E1-E 实验逻辑。

如果确实因框架限制必须修改核心源码：

1. 先复制原文件到 E1-E/backups/；
2. 保存 pre hash；
3. 只做最小 patch；
4. 实验结束必须恢复原文件；
5. post hash 必须和 pre hash 一致；
6. 全过程记录到 issues_and_fixes.md。

尽可能避免走这条路径。

======================================================================
4. E1-E runtime patch 的核心任务
======================================================================

E1-D 使用：

    algorithm.filter_groups.metric=em

因此 all-correct group 原本会被 EM filter 丢弃。

E1-E 必须在 EM filter / batch accumulation 的合适位置：

1. 保留 mixed groups；
2. 同时在被 EM filter 丢弃前观察 all-correct group；
3. 识别：
       k=8 && cost variation > 0
4. 将这些 group 放进 efficiency cache；
5. efficiency cache 不计入 stopping count；
6. 收集到 32 mixed 后：
       m=min(2,len(cache))
       batch=mixed[:32-m]+cache[:m]
7. 再进入 GRPO advantage / actor update。

科学语义必须与 E1-E0 replay 完全一致。

如果能够把 quota/batch-construction 封装成纯函数，
请这样做，以便 offline selftest 与 runtime 共用同一实现。

======================================================================
5. Reward manager/scorer 规则
======================================================================

不要继续直接沿用 E1-D 中：

    对 mixed group 也 efficiency-shape

的语义。

E1-E 必须 group-type-specific：

mixed group：
    score = EM
    即：
        R_i = A_i

selected all-correct efficiency group：
    R_i = 1 + 0.05 E_i

all-correct but no cost variation：
    不进入 efficiency cache。

all-wrong：
    不进入训练。

必须继续独立输出：

    em

作为真实 correctness metric。

绝不能让 shaped reward 污染：

    data.batch["acc"]

必须保持：

    acc = original EM

validation scorer 完全保持原 GAP evaluator，不修改。

======================================================================
6. 建立 E1-E 文件结构
======================================================================

最终至少：

E1-E/
├── README.md
├── FINAL_REPORT.md
├── code/
├── scripts/
├── configs/
├── logs/
│   ├── issues_and_fixes.md
│   ├── disk_cleanup.log
│   └── ...
├── runs/
├── diagnostics/
├── analysis/
├── eval_results/
├── manifests/
├── checkpoints/
└── figures/

模型 checkpoint 可继续位于：

experiments/DAPO-GAP3B-MHQA-Agent-E1E-q2-step60to120-4gpu/

但：

E1-E/checkpoints/experiment_path.txt

必须保存真实路径。

可以建立 symlink：

E1-E/checkpoints/model_dir
    ->
experiments/...E1E...

不要复制 checkpoint 本体。

======================================================================
7. 所有 train/eval 入口必须做严格 input/output preflight
======================================================================

每一个强化学习或 evaluation launcher 在真正执行前必须：

A. 输入检查

确认所有需要的：

- resume checkpoint；
- model shards；
- optimizer shards；
- extra_state；
- data.pt；
- base model；
- RL parquet；
- 7 benchmark 数据；
- wiki tool config；
- reward/scorer code；
- trainer patch；
- environment.sh；

全部存在且非空。

禁止自动用“类似文件”代替缺失输入。

B. checkpoint 完整性

至少验证：

model world-size shards
optimizer shards
extra state
data.pt

以及 resume global step。

D1：
必须从 original GAP step60 full state 恢复。

D2：
必须从 E1-E 自己的 step70 full state 恢复。

C. 输出防覆盖

训练启动前：

- 检查目标 experiment dir；
- 检查 E1-E/runs/...；
- 检查 log；
- 检查 eval output；
- 检查 latest pointer。

禁止覆盖既有完整 run。

如果发现同名完整实验：
    STOP，并记录 BLOCKED_OUTPUT_COLLISION。

如果发现当前 E1-E 自己产生的、可验证的 incomplete interrupted run：
    可以自行判断安全 resume，
    但必须记录：
    - 为什么认定是同一 E1-E；
    - resume checkpoint；
    - tracker；
    - 哪些旧输出会继续使用。

不得覆盖 E1-A/B/C/D/E0。

D. 保存 manifest

pre-run 对核心源文件保存：
path
size
mtime

实验后再次检查。

======================================================================
8. Wiki 服务和环境启动顺序 —— 必须严格执行
======================================================================

所有 RL 或 evaluation 开始前必须保证 Wiki retriever 已健康。

顺序：

------------------------------------------------------------
Step 1：检查 Wiki
------------------------------------------------------------

目标 URL 从原 tool config 读取，
预计为：

http://127.0.0.1:8008/retrieve

先检查：

- listener；
- process；
- 空 query probe；
- 至少一次真实 retrieval probe。

如果已经健康：

    REUSE

不要重复启动。
不要 kill 正常已有 Wiki 服务。

记录：

WIKI_REUSED=true

------------------------------------------------------------
Step 2：Wiki 不健康时启动
------------------------------------------------------------

只有检查失败时：

source conda setup
conda activate retriever

然后使用仓库已验证的 Wiki launcher，
优先完全复用 E1-D 的启动方式：

Agent/tool_servers/wiki_server/launch_rag_server.sh

等待 readiness。
允许等待较长时间，例如最长 900 秒。

必须真实 retrieval probe 成功后才算 ready。

记录：

WIKI_STARTED=true
PID
URL
启动日志

------------------------------------------------------------
Step 3：RL / evaluation 环境
------------------------------------------------------------

Wiki 已 healthy 后：

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate parallel-agent

确认：

CONDA_DEFAULT_ENV=parallel-agent

然后：

cd <repo_root>
source environment.sh

再次确认：

CONDA_DEFAULT_ENV 仍为 parallel-agent

然后 export：

WIKI_RAG_SERVER_URL=<healthy wiki URL>

再次 functional probe。

只有全部通过，才允许启动 RL / evaluation。

注意：
用户明确要求：
先进入 parallel-agent 环境，
再 source environment.sh。

不得把顺序反过来。

======================================================================
9. GPU / Ray / 进程安全
======================================================================

preflight 记录：

nvidia-smi
CUDA_VISIBLE_DEVICES
GPU memory
existing main_ppo
Ray
SGLang
Wiki processes

不要 kill 未确认归属的进程。

如果存在另一个 RL/main_ppo 占用实验 GPU：

- 判断是否是当前 E1-E 自己的 resumable process；
- 若不是，不要擅自 kill；
- 可以等待合理时间或最终 BLOCKED_RESOURCE；
- 记录详情。

Ray cleanup 仅处理明确属于当前 E1-E 的残留。

======================================================================
10. 磁盘空间管理
======================================================================

RL 开始前、D1 完成后、D2 开始前、D2 过程中定期检查：

df -h
df -B1
du -sh experiments/*
du -sh E1-E/*

根据 E1-D 实际 checkpoint 大小估计：

- 新 E1-E step70；
- step80/90/100/110/120；
- logs；
- evaluation artifacts；

所需剩余空间。

建议安全阈值：

available bytes >= 1.1 * estimated remaining requirement

如果空间足够：

    不删除任何模型。

如果预计空间不足：

允许清理 experiments/ 中“明确无用”的旧 checkpoint，
但必须严格遵守：

----------------------------------------------------------------------
永远保护，不得删除
----------------------------------------------------------------------

1. original GAP:
   experiments/DAPO-GAP3B-MHQA-Agent-4gpu/global_step_60

2. original GAP step120 baseline（若存在）

3. base/SFT model:
   experiments/exp_1_lr1e-5_wr0.03_bs1_ga8_4gpu

4. E1-0 / E1-A 用于 step70 comparison 的最终 checkpoint

5. E1-B step120 final checkpoint
   以及其最终分析明确引用的关键 checkpoint

6. E1-D step70 和 step120

7. 当前 E1-E：
   - latest complete checkpoint
   - step70
   - 最终 step120

8. 任何 manifest/FINAL_REPORT 明确标记为：
   start checkpoint
   final checkpoint
   baseline checkpoint
   的目录

9. 无法确认用途的未知目录

----------------------------------------------------------------------
可优先考虑删除
----------------------------------------------------------------------

仅限：
已完成旧实验中的冗余 intermediate checkpoints，
例如 global_step_80/90/100/110，
且同时满足：

- final step120 完整存在；
- 该 intermediate 不再用于 resume；
- FINAL_REPORT 不要求后续复现该 checkpoint；
- 已有必要统计/manifest；
- 明确确认不是当前运行；
- 删除后不会破坏已知比较。

删除前记录：

path
size
mtime
所属实验
删除原因

写入：

E1-E/logs/disk_cleanup.log

删除后再次 df 并记录释放空间。

不得删除：
代码、parquet、eval results、报告、logs。

如果仅删除安全 checkpoint仍不足：
不要冒险删除保护文件。
停止并最终报告 BLOCKED_DISK。

======================================================================
11. 在真正训练前必须做 offline runtime selftest
======================================================================

E1-E runtime quota implementation 完成后，
先不用 GPU 做 selftest。

优先使用 E1-E0 的真实 D2 candidate stream。

要求 runtime quota function 在 offline replay 时：

q=0：
必须复现 E1-E0 q=0 baseline selection。

q=2：
必须逐 step 尽可能精确复现：

E1-E0/results/optimizer_batch_members_by_quota.csv
中 q=2 的：

- stopping points；
- inserted group IDs；
- displaced mixed group IDs；
- batch size；
- quota；
- reward semantics。

至少验证：

1. 每 batch = 32 groups；
2. q <= 2；
3. mixed-only stopping；
4. quota 不改变 generation stopping；
5. inserted groups k=8；
6. inserted groups cost variation >0；
7. mixed group reward == EM；
8. inserted group reward ∈ [1,1.05]；
9. wrong reward=0；
10. all group reward std valid；
11. efficiency rollout ranking correct；
12. no NaN / Inf。

如果 selftest 与 E1-E0 不一致：
必须先修复，不能启动训练。

生成：

E1-E/analysis/preflight_selftest.md
E1-E/analysis/preflight_selftest.json

======================================================================
12. Phase E1-E1：step60 -> step70
======================================================================

Experiment name 建议：

DAPO-GAP3B-MHQA-Agent-E1E-q2-step60to120-4gpu

D1：

original GAP step60
    ->
E1-E step70

保持与 GAP/E1-D 一致：

actor lr
batch sizes
gen batch
n=8
GRPO
advantage estimator
norm_adv_by_std_in_grpo
epochs
tool config
temperature
max turns
save_freq=10
test_freq=10
validation
GPU 数
optimizer
scheduler
RNG/full-state resume

唯一科学变化：

E1-E quota/batch composition
+
group-type-specific reward

训练期间每一个 optimizer step 必须记录：

global_step
generation_batches_consumed
candidate_groups_seen

mixed_groups_seen
mixed_groups_selected

efficiency_groups_seen
efficiency_cache_size

quota_target=2
quota_actual

inserted_efficiency_group_ids
displaced_mixed_group_ids

每个 selected group：
uid
source
k
C_min
C_max
cost_gap
group_type

reward：
mean/min/max
EM
efficiency bonus

advantage：
mean/std/min/max
NaN/Inf count

queries
logical_search_batches
parallel_factor
response length/token proxy

记录到 JSONL/CSV，
全部在：

E1-E/runs/.../diagnostics/
或
E1-E/diagnostics/

======================================================================
13. D1 完整性检查
======================================================================

step70 训练完成后，不要马上进入 D2。

自动进行 integrity：

- exit code 0；
- tracker = 70；
- full checkpoint complete；
- resume 日志确认为 step60；
- optimizer/scheduler state 连续；
- RNG state 存在；
- 无 NaN/Inf；
- 无 OOM；
- 无 NCCL fatal；
- 无未处理 traceback；
- 每 optimizer batch 恰好 32 groups；
- quota_actual <=2；
- all inserted group 都 k=8；
- inserted group 都 efficiency-variable；
- mixed reward == EM；
- all-correct efficiency reward 符合公式；
- all-correct group不计入 stopping；
- 没有为了 quota 额外 generation。

特别统计：

quota actual distribution：
0 / 1 / 2

不要假设 step61–70 必然和 E1-E0 D2 stream一样有 >=2。

正式实现必须：

m=min(2,len(cache))

而不是 assert cache>=2。

======================================================================
14. step70 完整 7 benchmark evaluation
======================================================================

D1 integrity PASS 后：

先确保 Wiki healthy；
parallel-agent activate；
source environment.sh；
然后使用和 GAP/E1-D 完全相同的 evaluation protocol。

跑完整 7 benchmark：

NQ
TriviaQA
PopQA
HotpotQA
2WikiMultiHopQA
Musique
Bamboogle

保存到：

E1-E/eval_results/step70_<timestamp>/

统计至少：

EM / accuracy
micro EM
macro EM
per-benchmark EM

search rounds
search queries
queries / round
parallel sample rate
parallel factor
output tokens/length
total turns
zero-search rate
malformed / no-answer
tool errors

如果题目 ID 可严格对齐：
做 paired comparison。

比较至少：

GAP/E1-0 step70
E1-A step70
E1-D step70
E1-E step70

======================================================================
15. D1 -> D2 自动 gate
======================================================================

本任务允许无人值守，因此不需要向用户询问是否继续。

如果以下硬条件满足：

A. training/integrity 无 fatal；
B. quota semantics 正确；
C. step70 checkpoint 完整；
D. 7 benchmark evaluation 完整；
E. 无 reward hacking / catastrophic anomaly；

则自动进入 D2。

不要因为 step70 efficiency gain 只有 1-3% 就停止。

E1-E 信号仅占约 6.25%，短期效果可能较弱。

只有出现明显失败才停止，例如：

- runtime semantics 和 E1-E0 不一致；
- quota 导致 generation stopping 被改变；
- batch size错误；
- reward formula错误；
- NaN/Inf；
- persistent OOM/NCCL；
- micro EM 相比 GAP/E1-0 step70 出现 >1.0 pp 的明显一致性下降，
  且不是 evaluator noise / provenance 问题；
- query explosion >约15% 且 rounds下降疑似 packing；
- 大量 zero-search shortcut；
- malformed 输出显著爆发。

若出现上述：
停止 D2，
完成失败分析和 FINAL_REPORT，
不要盲目继续烧 GPU。

======================================================================
16. Phase E1-E2：step70 -> step120
======================================================================

如果 D1 gate PASS：

resume：

E1-E 自己的 full-state global_step_70

不是 original GAP70，
不是 E1-D70。

继续到：

step120

保存：

80
90
100
110
120

除 E1-E quota mechanism 外，
所有配置保持 D1 不变。

每 10 step 保存/统计 quota/runtime diagnostics。

======================================================================
17. step120 7 benchmark evaluation
======================================================================

训练到 step120 后：

再次运行完整七 benchmark。

比较：

1. original GAP120
2. E1-B120
3. E1-D120
4. E1-E120

如果可以严格对齐 sample IDs：

做 paired analysis：

- paired accuracy difference
- McNemar 或 paired bootstrap
- rounds paired delta
- query paired delta
- token paired delta

不要只报告 aggregate mean。

======================================================================
18. E1-E 最关键的研究问题
======================================================================

最终必须回答：

RQ1.
q=2 controlled solved-group efficiency supervision
能否保持更多 GAP correctness capacity？

RQ2.
相比 E1-D：

E1-E 是否同时：

    EM 更高
    AND
    search rounds 更低

即 Pareto-dominate E1-D？

RQ3.
相比 GAP120：

是否达到：

    Δ micro-EM >= -0.5 pp

同时：

    Δ search rounds <= -10%

这是预注册的主要工程成功标准。

RQ4.
相比 E1-B：

是否用明显更小 accuracy cost 保留一部分主要 efficiency gain？

RQ5.
正式 runtime 中 q=2 实际 fill rate 是多少？

RQ6.
效率改善是否伴随：

queries 不异常增加
queries/round 不爆炸
zero-search 不增加
malformed 不增加

RQ7.
E1-E0 的 offline quota availability / bias 结论是否在线 policy 下基本成立？

======================================================================
19. Pareto 分析
======================================================================

必须生成至少一张 Pareto 图：

x:
search rounds（越低越好）

y:
micro EM 或 accuracy（越高越好）

包含：

GAP120
E1-B120
E1-D120
E1-E120

必要时添加 step70 points，但和 step120 区分。

同时做表格：

Method | EM | ΔEM vs GAP | Rounds | ΔRounds | Queries | Parallel factor

不要只说“better”。

明确判断：

- dominated
- non-dominated
- Pareto improvement
- trade-off

======================================================================
20. q=3 暂时不要训练
======================================================================

E1-E0 已将 q=3 预注册为后续 ablation。

本次 E1-E：

只训练：

    q=2

无论结果好坏，
不要自动追加 q=1/q=3/q=4 训练。

最终只提出下一步建议。

======================================================================
21. 分析和报告中必须严格区分的概念
======================================================================

必须区分：

training-side cost：
    logical_search_batches

evaluation-side search-round metric：
    当前 evaluator 的既有定义

不要把两个定义无标签混在同一数值列。

必须清楚注明。

还要区分：

candidate groups
mixed retained groups
efficiency cache
optimizer selected groups

不要混为一谈。

======================================================================
22. 必须进行 source-provenance/no-modification 检查
======================================================================

实验前后对：

原 GAP 核心源码
E1-D 源文件
E1-E0 源文件
RL dataset
tool config
evaluation scripts
environment.sh

做 hash manifest。

实验结束：

0 unexpected sha256 mismatch

如果 E1-E 使用 runtime patch，
确认原文件没有被永久修改。

======================================================================
23. E1-E FINAL_REPORT.md
======================================================================

所有实验完成后生成：

E1-E/FINAL_REPORT.md

必须使后续研究者只看这一个文件，
基本可以理解 E1-E 的设计、执行、结果和结论。

至少包括：

1. Executive Summary
2. Motivation from E1-C/E1-D/E1-E0
3. Exact preregistered design
4. Scientific control variables
5. Code reuse / copied-from provenance
6. Exact code modifications
7. Preflight / Wiki / environment
8. Disk-space status and any cleanup actions
9. Runtime quota implementation
10. Offline selftest vs E1-E0
11. D1 step60->70 training integrity
12. Runtime quota statistics step61–70
13. step70 seven-benchmark evaluation
14. D1 gate decision
15. D2 step70->120 training integrity
16. Runtime quota statistics step71–120
17. step120 seven-benchmark evaluation
18. Per-benchmark accuracy table
19. Efficiency metrics table
20. GAP / E1-B / E1-D / E1-E four-way comparison
21. Paired statistical tests
22. Pareto analysis
23. Reward-hacking / safety analysis
24. E1-E0 offline prediction vs online observation
25. Whether E1-E meets preregistered success criteria
26. What E1-E establishes
27. What it does NOT establish
28. Limitations
29. Recommended next experiment
30. Full reproduction commands
31. Complete artifact inventory

======================================================================
24. Success / failure wording
======================================================================

不要为了“实验完成”而把结果写成成功。

请按真实结果判断。

建议：

STRONG PASS：
    E1-E Pareto-dominates E1-D
    并大体满足：
        ΔEM vs GAP >= -0.5 pp
        ΔRounds vs GAP <= -10%

PASS / PROMISING：
    未完全达到上述两个阈值，
    但明显改善 E1-D 或 E1-B frontier。

NEGATIVE BUT INFORMATIVE：
    quota semantics正确，
    但 E1-E不改善 Pareto frontier。

FAILED ENGINEERING：
    runtime / data / checkpoint / environment 无法可靠完成。

这些分类只是报告语言，
不要修改实验结果以适配分类。

======================================================================
25. unattended orchestration
======================================================================

请创建类似：

E1-E/scripts/run_e1e_all.sh

实现一次性无人值守流程：

1. repository audit
2. input/output collision check
3. disk preflight
4. Wiki check/start
5. parallel-agent activate
6. source environment.sh
7. source/provenance manifest
8. offline selftest
9. D1 train 60->70
10. D1 integrity
11. step70 full 7-benchmark eval
12. automatic gate
13. disk recheck / safe cleanup if needed
14. D2 train 70->120
15. D2 integrity
16. step120 full 7-benchmark eval
17. analyses
18. figures
19. post manifest
20. FINAL_REPORT.md

中途不需要向用户汇报。

遇到普通问题：
自主诊断、修复、继续，
并写 issues_and_fixes.md。

遇到无法安全修复的问题：
停止危险操作，
保存已有结果，
写清 blocker，
仍然生成 FINAL_REPORT.md。

======================================================================
26. 防止误覆盖 / 可重入
======================================================================

所有 run 使用 timestamped run dir。

例如：

E1-E/runs/E1_202609xx_xxxxxx/
E1-E/runs/E2_202609xx_xxxxxx/

launcher 不得覆盖已有 run dir。

evaluation 同样 timestamp。

如果 pipeline 重启：

- 检查已完成阶段；
- 验证 checkpoint/integrity；
- 只继续未完成阶段；
- 不重复覆盖成功结果。

保存：

E1-E/logs/pipeline_status.txt

状态例如：

PREFLIGHT_PASS
SELFTEST_PASS
D1_PASS
STEP70_EVAL_PASS
D2_PASS
STEP120_EVAL_PASS
ANALYSIS_PASS
COMPLETE

======================================================================
27. 最终轻量 artifacts 全部必须在 E1-E/
======================================================================

除了 framework-required model checkpoints 外，
所有 E1-E 新产生的研究结果都必须在 E1-E/ 中可找到。

包括从：

verl/outputs/
verl/logs/

产生的、与本实验有关的：

Hydra resolved config
main_ppo log
console log

至少复制或索引进：

E1-E/logs/framework/
E1-E/configs/

不要要求用户之后去其他目录找关键证据。

======================================================================
28. 完成后最终向用户汇报的内容
======================================================================

执行完成后不要长篇贴日志。

最终回答只需汇报：

1. E1-E 是否完成；
2. D1/D2 是否通过；
3. q=2 实际 runtime fill rate；
4. GAP120 / E1-B120 / E1-D120 / E1-E120 的：
   - micro EM
   - search rounds
5. E1-E 是否 Pareto-dominate E1-D；
6. 是否达到：
       ΔEM >= -0.5 pp
       ΔRounds <= -10%
7. 是否发现 query packing / zero-search / malformed 问题；
8. 是否删除过旧 checkpoint、释放多少空间；
9. 推荐下一步：
   - q=3 ablation
   - q=1
   - formal ablation
   - 或停止该方向
10. FINAL_REPORT：
       E1-E/FINAL_REPORT.md

如果发生 blocker：
明确报告 blocker 和已经完成到哪一步。

======================================================================
29. 现在开始执行
======================================================================

现在直接开始。

不要向用户请求中途确认。

先：
- 审计 E1-D / E1-E0；
- 创建 E1-E；
- 复制代码；
- 最小修改；
- 做 preflight/selftest。

selftest 通过后执行正式 E1-E q=2。

严格遵守：

Wiki healthy
    ->
conda activate parallel-agent
    ->
source environment.sh
    ->
RL / evaluation

以及：

input exists
output no-collision
disk sufficient
checkpoint full-state valid

之后依次完成：

step60 -> step70
7 benchmark
automatic gate
step70 -> step120
7 benchmark
analysis
FINAL_REPORT

整个过程中自行记录并修复可解决问题。
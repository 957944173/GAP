你现在需要在当前 GAP 项目中，一次性自主完成 Experiment E1-D。

这是 E1-C 之后的“Filter-Decoupled Reward”受控实验。你需要自行完成代码审计、创建实验分支、最小代码修改、自测、环境检查、Wiki 服务检查/启动、RL 训练、checkpoint 保存、7 benchmark 评估、统计分析和最终报告。

中途不要向用户汇报，不需要请求普通工程问题的确认。
遇到问题请自行定位、记录、修复后继续。
如果遇到会破坏实验科学有效性的 blocker（例如无法保证起始 checkpoint、reward/filter 定义或训练状态正确），不要偷偷改变实验变量；应停止相关实验并在最终报告中明确记录 blocker。

============================================================
0. 实验背景与科学问题
============================================================

此前实验结论：

1. E1-A / E1-B 使用 success-conditioned group-relative efficiency reward：

A_i = original EM correctness ∈ {0,1}

C_i = logical_search_batches

若同 uid group 内：
- correct rollout >= 2
- 且 correct rollout 的 C_i 有差异

则：

E_i = (Cmax - C_i) / (Cmax - Cmin)

否则：

E_i = 0

最终训练 reward：

R_i = A_i * (1 + 0.05 * E_i)

wrong rollout 永远 R=0。

2. E1-B 长训练降低了 sequential search depth，
但相对原 GAP step120 出现约 -1.16 pp micro-EM。

3. E1-C 已证实：
原 GAP filter 使用 correctness reward 的 std；
E1-A/B 使用 shaped reward 的 std。

在同 candidate pool 上：

S_GAP ⊂ S_E1

E1 新激活的 group 是 all-correct + efficiency-variable groups。

固定 optimizer batch 下，这些新增 solved groups 会部分替换
原 GAP 会训练的 mixed-correctness groups。

4. E1-C 推荐下一实验 Candidate A：

Filter-Decoupled Reward

即：

FILTER 使用原 GAP correctness EM：

keep(group) iff std(A_i) > 0
（或沿用 trainer 对 singleton group 的原逻辑）

但进入训练后的 GRPO outcome reward 仍然使用：

R_i = A_i * (1 + 0.05 * E_i)

科学问题：

“E1-B 的 accuracy-efficiency trade-off 中，
有多少来自 efficiency objective 本身，
有多少来自 reward-dependent filter 引起的 training distribution shift？”

============================================================
1. E1-D 唯一核心变量
============================================================

E1-D 必须满足：

训练 reward：
与 E1-B 完全相同。

lambda：
0.05，完全不改。

cost：
logical_search_batches，完全不改。

GRPO：
完全不改。

n：
8，完全不改。

dataset：
完全不改。

rollout/tool/wiki config：
完全不改。

norm_adv_by_std_in_grpo：
保持原值 True。

模型、optimizer、scheduler、dataloader state：
从指定 checkpoint 完整恢复。

VALIDATION / EVALUATION scorer：
继续使用原 GAP evaluation EM，不做 shaping。

唯一核心变化：

E1-A/B：
filter_groups.metric = shaped sequence reward

E1-D：
filter_groups.metric = original EM

目标实现优先采用：

algorithm.filter_groups.metric=em

因为 reward manager 已经输出原始 EM 到 reward_extra_info / non_tensor_batch。

------------------------------------------------------------
非常重要：
------------------------------------------------------------

不要为了实现 E1-D 去修改 ray_trainer.py，
除非经过代码和 runtime 检查后确认当前框架无法直接用 metric=em。

如果 config-only 可以实现，则必须使用 config-only。

如果确实不能：
才允许在 E1-D/code/ 中复制并做最小 runtime patch，
不得直接修改原 verl 文件。

============================================================
2. 控制变量与代码分支原则
============================================================

必须最大限度复用 E1-A / E1-B 已验证代码。

优先来源：

E1-B/code_changes/
E1-A/code/

建议：

复制 E1-B 已验证的：
- reward manager
- scorer
- cost extractor
- state / diagnostics
- sitecustomize / registry injection
- training launcher logic
- evaluation launcher逻辑

到：

E1-D/code/

然后只做：
- E1B -> E1D 命名修改
- 输出路径修改
- filter metric 改为 em
- 本实验需要的额外诊断记录

不要重新设计 reward。

不要清洗数据。

不要修改 RL 算法。

不要修改并行 tool execution。

不要修改 Wiki retriever。

不要修改 evaluation metric。

不要修改原 GAP / E1-A / E1-B / E1-C 文件。

============================================================
3. E1-D 目录要求
============================================================

所有 E1-D 的“可分析实验产物”必须统一保存到：

E1-D/

包括：

E1-D/
├── E1-D_task.md
├── README.md
├── FINAL_REPORT.md
├── code/
├── scripts/
├── configs/
├── logs/
├── diagnostics/
├── analysis/
├── eval_results/
├── manifests/
└── runs/
    └── run_<timestamp>/

如需要保存大型模型 checkpoint：

优先放到：

E1-D/checkpoints/

如果现有 verl/checkpoint 机制强依赖 experiments/ 路径，
允许将实际大型 checkpoint 保存在一个新的、
明确命名且绝不覆盖旧实验的 experiments/...E1D... 目录中，

但必须在：

E1-D/manifests/checkpoints.json

记录：
- checkpoint 实际绝对路径
- global step
- 文件列表
- size
- tracker/global step
- data.pt 是否存在
- actor/optimizer/extra state 是否完整

不得复制已有 step60 checkpoint，仅引用原 checkpoint。

所有：
logs
configs
eval outputs
statistics
diagnostics
final report

必须实际位于 E1-D/。

============================================================
4. 禁止文件覆盖
============================================================

从任何 RL / eval / analysis 脚本入口开始，
第一步必须检查：

A. 输入文件是否存在且完整

B. 输出路径是否已经存在

C. log 是否已存在

D. checkpoint 目标目录是否已存在

E. evaluation result 目录是否已存在

原则：

绝不静默覆盖已有文件。

如果目标目录不存在：
正常创建。

如果发现同名但已经完整完成的 E1-D run：
不要重复覆盖。

如果发现同名 partial run：
检查完整性。

可以安全 resume 时：
使用新的日志文件继续，并明确记录。

不能安全 resume 时：
创建新的 timestamp run dir。

任何 rename / backup / resume 都必须记录在：

E1-D/logs/issues_and_fixes.md

============================================================
5. 运行环境的强制顺序
============================================================

强化学习和评估开始之前必须检查 Wiki retriever 服务。

============================================================
5.1 Wiki service
============================================================

目标 Wiki service 是 GAP 当前 search tool 实际使用的服务。

首先：

1. 检查当前监听端口和进程。
2. 检查 repo 当前 tool config 中配置的 URL / port。
3. 使用现有可靠方式实际发一个最小 retrieval 请求验证服务可用。
4. 不允许仅凭“有进程”判断服务健康。

如果 Wiki 已经健康运行：

REUSE。

不要重复启动第二个 Wiki server。

记录：
- PID
- port
- command
- health check
- time

到：

E1-D/logs/wiki_service.log

如果 Wiki 没启动：

必须先进入 retriever 环境。

非交互 shell 下如果需要：

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate retriever

然后优先复用 repo / E1-A / E1-B 已验证过的 Wiki 启动方式。

不要自己发明新的 server command。

启动后等待直到 retrieval health check 成功。

如果启动失败：
自行排查并记录。

训练结束后：
如果 Wiki 是实验前已经存在的，不得关闭。
如果是 E1-D 自己启动的，可以在整个 E1-D 的训练+评估全部完成后再决定关闭，并记录。

============================================================
5.2 RL / Evaluation environment
============================================================

任何 RL training 或 evaluation command 运行前都必须：

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate parallel-agent

然后：

source environment.sh

environment.sh 的实际位置请先在 repo 中定位，
不要假设错误路径。

每一个独立 shell launcher 都必须满足该顺序，
不能依赖父 shell 偶然遗留环境。

记录：

which python
python --version
conda env
CUDA_VISIBLE_DEVICES
nvidia-smi
必要包版本

到 preflight log。

============================================================
6. GPU / process / Ray preflight
============================================================

训练前检查：

- nvidia-smi
- GPU 数量
- GPU memory occupancy
- 是否已有 GAP training/evaluation process
- Ray 状态
- SGLang worker
- Wiki retriever process

禁止杀掉不属于本实验的未知进程。

如果发现以前 E1-D 自己失败后遗留的 Ray / worker：
确认 PID ownership 后才允许清理。

Wiki retriever 不属于需要清理的 Ray worker。

所有清理操作写入 issues_and_fixes.md。

============================================================
7. 输入完整性检查
============================================================

E1-D 必须从“原始 GAP step60”开始，
不是 E1-A step70，
不是 E1-B step120。

根据已有实验，预期路径类似：

experiments/DAPO-GAP3B-MHQA-Agent-4gpu/global_step_60

但不要盲信字符串。
请结合 E1-A FINAL_REPORT 和磁盘实际情况确认。

必须验证：

- actor/model_world_size_4_rank_{0..3}.pt
- optimizer state
- actor extra state
- lr scheduler state
- RNG state（若框架保存）
- data.pt
- global step tracker
- 文件非空

resume 必须是 full-state resume：

trainer.resume_mode=resume_path
trainer.resume_from_path=<original GAP step60>

日志中必须确认：

Setting global step to 60

并确认 dataloader data.pt 已恢复。

还必须检查：

RL dataset：
Agent/data/mhqa_agent/GAP-MHQA-RL-Dataset/GAP-RL-16w.cleaned.parquet

tool config：
verl/verl/tools/config/search_tool_config/wiki_rag_config.yaml

以及训练/评估需要的所有实际输入。

缺任何关键输入时：
停止训练并报告，不允许换数据集替代。

============================================================
8. E1-D 实现
============================================================

优先复制 E1-B 已验证 reward 实现。

训练 scorer 仍然计算：

A_i
C_i
E_i
R_i=A_i*(1+0.05E_i)

必须继续返回至少：

score = shaped reward R_i
em = original correctness A_i
efficiency = E_i
logical_search_batches
search_queries
parallel_factor
group_correct_count
group_active
metadata_valid

validation：
完全 delegate 到原 GAP scorer。

acc：
必须继续表示 EM，
不能被 shaped reward 污染。

token_level_scores / token_level_rewards：
必须继续是 shaped reward，
供 GRPO advantage 使用。

============================================================
8.1 Filter decoupling
============================================================

E1-D 的关键：

algorithm.filter_groups.metric = em

必须在训练前验证 trainer 的真实数据流：

reward_extra_infos_dict
  -> new_batch.non_tensor_batch["em"]
  -> filter_groups.metric == "em"
  -> std(em) 分组
  -> kept_prompt_uids

而：

new_batch.batch["token_level_rewards"]

仍然是 shaped reward。

GRPO compute_advantage 使用 shaped reward，
不是 EM。

这两个通路必须在 selftest / runtime diagnostics 中分别确认。

============================================================
9. E1-D 自测：必须全部 PASS 才允许训练
============================================================

在真正用 GPU RL 前完成 CPU/offline selftest。

至少：

TEST 1 — reward equivalence
---------------------------
E1-D shaped reward 与 E1-B shaped reward，
在已有真实 E1-0 / E1-B rollout 上逐条比较。

必须：
0 mismatches
或解释任何 mismatch 并修复。

TEST 2 — lambda
---------------
lambda 必须严格是 0.05。

TEST 3 — filter baseline equivalence
------------------------------------
在 E1-C/E1-0.5 已保存的真实 group pool 上：

使用 filter metric=em 后：

E1-D retained set

必须严格等于同 pool 的原 GAP correctness filter retained set。

例如 E1-C 已报告过的 pool 中，
应能复现其 S_GAP 数量。

不要硬编码这些数字作为逻辑；
用它们做 sanity reference。

输出 exact set comparison。

TEST 4 — no all-correct revival
-------------------------------
all-correct efficiency-variable groups：

即使 shaped reward std > 0，

因为 EM std = 0，

E1-D filter 必须不保留。

TEST 5 — mixed group shaping preserved
--------------------------------------
对 baseline filter 保留的 mixed-correctness group：

如果 correct rollouts 存在 cost variation，

进入 optimizer 后：
correct-efficient 与 correct-inefficient rollout
必须获得不同 shaped reward / GRPO advantage。

也就是：

filter channel 被关闭，
objective channel 仍然存在。

TEST 6 — validation unchanged
-----------------------------
validation manager/scorer 必须和原 GAP 一致。

TEST 7 — no NaN/Inf
-------------------
reward / advantage 无 NaN/Inf。

将结果写入：

E1-D/analysis/preflight_selftest.md
E1-D/analysis/preflight_selftest.json

============================================================
10. 训练阶段 D1：step60 -> step70
============================================================

从原 GAP step60 完整恢复。

训练：

global_step 60 -> 70

total_training_steps = 70
save_freq = 10
test_freq 保持原实验协议
rollout.n = 8

唯一科学变化：

filter metric = em
+
shaped reward 与 E1-B 相同

其它超参数必须对照 resolved config，
确保没有无意变化。

启动前再检查：

- Wiki healthy
- parallel-agent active
- environment.sh sourced
- input checkpoint complete
- output dir 不存在/不会覆盖
- GPU available

训练 console log：

E1-D/logs/train_step60_to70_<timestamp>.log

完整 resolved config：

E1-D/configs/resolved_step60_to70.json

保存 rollout/group diagnostics。

============================================================
11. D1 训练侧必须记录的统计
============================================================

每个 generation batch / optimizer step 尽量保存：

- global_step
- call_index
- uid
- group size
- #correct k
- EM reward
- shaped reward
- efficiency E
- logical_search_batches
- search_queries
- queries/round
- parallel_factor
- response length/tokens
- metadata_valid

同时记录：

actual filter:
- filter_metric = em
- baseline/EM kept groups
- all-correct kept
- mixed kept
- all-wrong kept

counterfactual shaped filter:
- 如果仍使用 E1-B filter，会额外保留多少 groups
- 仅做诊断，不用于训练

统计：

- actual retained groups
- shaped-filter hypothetical retained
- all-correct efficiency-active groups excluded by EM filter
- mixed efficiency-active groups actually retained
- efficiency-active supervision ratio
- generation batches needed per optimizer step

特别验证：

actual retained set on current candidate pool
==
same-pool original EM filter set

必须 100% 成立。

============================================================
12. step70 checkpoint 完整性
============================================================

完成后检查：

- exit code
- global_step_70 checkpoint
- model shards
- optimizer
- extra state
- data.pt
- tracker=70
- lr scheduler
- RNG state（如果保存）
- CUDA OOM
- NCCL error
- traceback
- Wiki tool failures
- NaN/Inf

输出：

E1-D/analysis/step70_integrity.md

如果存在科学完整性问题：
不要继续 step120。

如果只是可修复工程问题：
自行修复。

============================================================
13. step70 完整 7 benchmark 评估
============================================================

如果 step70 training integrity PASS：

运行完整 7 benchmark：

- nq
- triviaqa
- popqa
- hotpotqa
- 2wikimultihopqa
- musique
- bamboogle

必须复用原 GAP/E1-A/E1-B 已验证 evaluation entry/protocol。

启动前重新确认：

1. Wiki service healthy
2. conda activate parallel-agent
3. source environment.sh
4. checkpoint input 存在
5. eval output 目录不会覆盖

输出必须位于：

E1-D/eval_results/step70_<timestamp>/

记录：
- EM macro/micro
- per benchmark EM
- search rounds
- search queries
- parallel factor
- parallel sample rate
- assistant turns
- response length/tokens
- zero-search
- no-answer / malformed
- query packing diagnostics

============================================================
14. step70 三方比较
============================================================

比较：

A. GAP original step70
   使用 E1-0 已有 matched baseline

B. E1-A step70
   shaped reward + shaped filter

C. E1-D step70
   shaped reward + EM filter

必须使用已有评估 artifact，
不要重新训练 baseline。

如果已有 baseline eval 可直接解析则复用。

做 paired per-prompt comparison，
优先沿用 E1-A/E1-B 已验证的 pairing 逻辑：

(data_source, prompt)

注意处理 popqa duplicate prompt 与此前协议一致。

重点回答：

1. E1-D vs GAP70：
   在原 filter 下，仅加入 objective shaping 后，
   EM 和 rounds 怎么变化？

2. E1-D vs E1-A70：
   reward 完全相同，
   只改变 filter channel 后，
   EM 和 rounds 怎么变化？

============================================================
15. 是否继续 D2 的规则
============================================================

E1-D 是 causal/mechanism ablation。

因此：

不要因为 step70 efficiency improvement 小
就提前判定方法失败。

只要：

- training integrity PASS
- filter decoupling 确认正确
- shaped reward 确实进入 GRPO
- 无 catastrophic reward hacking
- 无明显工程错误

就自动继续：

step70 -> step120

不需要向用户汇报或等待确认。

如果 step70 EM 有普通波动，
仍继续。

只有以下情况才停止：

- checkpoint / optimizer resume 不可信
- actual filter 并非 EM filter
- shaped reward 没进入 GRPO
- Wiki/tool pipeline 故障
- NaN/Inf / unrecoverable OOM
- reward 定义偏离 E1-B
- 其他会使实验失去控制变量意义的问题

============================================================
16. 训练阶段 D2：step70 -> step120
============================================================

必须从 E1-D 自己的 step70 checkpoint
完整 full-state resume。

不能从 E1-A / E1-B checkpoint 开始。

保持所有 E1-D 设置不变：

filter metric = em
reward = shaped
lambda = 0.05

训练至：

global_step_120

save_freq=10

至少保存：
80
90
100
110
120

如果磁盘空间不足：
不得擅自删除 step70 或 step120。
优先记录后再安全处理中间 checkpoint，
但不要影响训练。

每 10 step 保存相同训练 diagnostics。

训练完整 log：

E1-D/logs/train_step70_to120_<timestamp>.log

============================================================
17. step120 完整 7 benchmark 评估
============================================================

训练 integrity PASS 后：

评估 E1-D step120 的完整 7 benchmark。

输出：

E1-D/eval_results/step120_<timestamp>/

不要求本轮立即完整评估 80/90/100/110，
因为主要研究问题只要求 matched step70 / step120。

中间 checkpoint 保留，方便后续需要时补评估。

============================================================
18. step120 核心三方比较
============================================================

主要比较：

1. Original GAP step120
   filter=EM
   reward=EM

2. E1-B step120
   filter=shaped
   reward=shaped

3. E1-D step120
   filter=EM
   reward=shaped

这构成主要 causal/mechanism matrix：

                       Reward
                EM             Shaped
Filter EM       GAP            E1-D
Filter shaped   --             E1-B

必须至少报告：

macro EM
micro EM
per-benchmark EM

mean search rounds
p90 search rounds（若已有）
search queries
queries/round
parallel factor
parallel sample rate
assistant turns
response tokens
zero-search
no-answer/malformed

paired per-prompt：

E1-D120 vs GAP120
E1-D120 vs E1-B120

计算：
- ΔEM
- up/down pairs
- sign test / 已有 paired significance test
- Δsearch rounds
- decreased/increased prompts
- per benchmark paired result

尽量直接复用 E1-B 已验证 comparison scripts。

============================================================
19. 训练机制分析
============================================================

E1-D 的 FINAL_REPORT 必须回答：

Q1.
EM filter 是否真的消除了 all-correct efficiency groups 的 revival？

Q2.
actual retained candidate set 是否严格等于：
same-pool GAP EM filter set？

Q3.
在 EM filter 下，
还有多少 retained mixed groups 是 efficiency-active？

重点统计：

active efficiency groups /
actual retained groups

并与 E1-C 已知现象比较：

E1-B shaped-filter efficiency supervision 很多；
E1-D 预期大幅减少。

Q4.
shaped reward 是否仍在这些 retained mixed groups 内
产生 efficient-correct > inefficient-correct 的 advantage？

Q5.
E1-D 是否减少了 generation batch keep-rate shift？

Q6.
step120：
EM 是否向 GAP 恢复？

Q7.
step120：
效率改善是否仍存在？

============================================================
20. 预先定义的结果解释矩阵
============================================================

FINAL_REPORT 请按真实结果选择，不要强行套结论。

CASE A:
E1-D EM≈GAP
效率改善大部分消失

解释：
支持 filter-induced distribution shift
是 E1-B trade-off 的重要机制；
大部分效率 signal 来自 revived all-correct groups。

下一步：
优先 Candidate B：
baseline-preserving + capped efficiency groups。

CASE B:
E1-D EM≈GAP
同时仍有明显效率改善

解释：
最理想。
objective shaping 在 baseline filter 内本身有效，
而 shaped-filter distribution shift 是主要副作用。

下一步：
E1-D 可成为主方法，
再做正式 ablation。

CASE C:
E1-D EM 仍明显低于 GAP
效率仍改善

解释：
即便移除 filter shift，
efficiency objective 本身仍和 correctness 有明显 trade-off。

下一步：
优先重新设计 reward/advantage，
不是 quota。

CASE D:
E1-D EM≈GAP
效率≈GAP

解释：
E1-B 的主要行为效应来自 revived all-correct groups；
mixed-group shaping signal 太稀。

下一步：
Candidate B 最有价值。

不要因为单个 benchmark 异常
就擅自重训或改 reward。

============================================================
21. Safety / reward hacking 检查
============================================================

必须继续监控：

- query packing
- queries per round
- total queries
- zero-search
- zero-search correct
- malformed XML/tool output
- missing answer
- max-turn termination
- extreme parallel factor

E1-D reward 只对 correct rollout 生效，
但仍需检查：

rounds下降
是否伴随 queries异常上涨。

如果出现：
search rounds ↓
但 queries ↑ >10%

必须标记为潜在 packing，
不能直接称效率提升。

============================================================
22. 不允许做的事
============================================================

E1-D 期间禁止：

- 调 lambda
- 新加 query penalty
- response length reward
- reflection penalty
- 改 SFT 数据
- 改 RL dataset
- 改 DAPO/GRPO
- 改 advantage estimator
- 改 norm_adv_by_std_in_grpo
- 改 rollout n
- 改 Wiki corpus
- 改 tool parser
- 改 async execution
- 换 backbone
- 自动进入 Candidate B

E1-D 只回答 filter-decoupling 问题。

============================================================
23. 代码与配置 provenance
============================================================

创建：

E1-D/manifests/source_files.json

记录所有关键源文件：

- E1-B reward code
- E1-A code（若使用）
- original batch.py
- ray_trainer.py
- core_algos.py
- main RL launcher
- evaluation launcher
- environment.sh
- RL dataset
- tool config
- starting checkpoint

尽量记录：
absolute path
size
mtime

训练前后再生成：

original_files_manifest_pre.json
original_files_manifest_post.json

检查：
原 GAP / E1-A / E1-B / E1-C 关键文件未被修改。

============================================================
24. 日志和错误修复
============================================================

所有问题记录：

E1-D/logs/issues_and_fixes.md

每条至少写：

- timestamp
- issue
- root cause
- affected stage
- fix
- 是否改变科学变量
- verification

普通工程修复：
允许自行进行。

任何可能影响控制变量的修复：
必须明确说明。

============================================================
25. FINAL_REPORT.md
============================================================

全部 E1-D 完成后自动生成：

E1-D/FINAL_REPORT.md

必须做到：
后续只读 FINAL_REPORT.md
就能理解实验全过程。

至少包含：

1. Executive Summary
2. Scientific question
3. Exact experimental design
4. Why E1-D follows E1-C
5. Environment
6. Wiki service status / reuse or launch
7. Starting checkpoint integrity
8. Source-code provenance
9. Exact code/config modifications
10. Proof that E1-D reward == E1-B reward
11. Proof that filter uses EM
12. Proof validation is unchanged
13. Preflight/selftest results
14. step60→70 training integrity
15. step70 training diagnostics
16. step70 full 7-benchmark results
17. GAP70 vs E1-A70 vs E1-D70
18. step70→120 training integrity
19. step70→120 training diagnostics
20. step120 full 7-benchmark results
21. GAP120 vs E1-B120 vs E1-D120
22. paired per-prompt statistics
23. efficiency supervision density
24. actual filter composition
25. query-packing / zero-search safety analysis
26. interpretation using CASE A/B/C/D
27. What E1-D establishes
28. What it does NOT establish
29. Recommended next experiment
30. Reproduction commands
31. Full generated-artifact inventory
32. Known limitations

============================================================
26. summary.json
============================================================

同时生成：

E1-D/analysis/summary.json

至少结构化保存：

experiment_status
start_checkpoint
step70_checkpoint
step120_checkpoint

filter_metric
reward_formula
lambda

step70:
  macro_em
  micro_em
  search_rounds
  queries
  parallel_factor
  retained_groups
  efficiency_active_retained
  all_correct_retained
  generation_batches

step120:
  同上

comparisons:
  e1d70_vs_gap70
  e1d70_vs_e1a70
  e1d120_vs_gap120
  e1d120_vs_e1b120

hypothesis_verdict

recommended_next_experiment

============================================================
27. 运行入口脚本
============================================================

最终至少应有清晰入口，例如：

E1-D/scripts/preflight.sh
E1-D/scripts/run_e1d_train_60_70.sh
E1-D/scripts/run_e1d_eval_70.sh
E1-D/scripts/run_e1d_train_70_120.sh
E1-D/scripts/run_e1d_eval_120.sh
E1-D/scripts/run_e1d_analysis.sh

可以额外提供：

E1-D/scripts/run_e1d_all.sh

但 run_e1d_all.sh 必须：
- 每阶段检查返回码
- 不忽略失败
- 检查 Wiki
- 检查环境
- 防止覆盖
- 只在 integrity PASS 后进入下一阶段

============================================================
28. 最终执行顺序
============================================================

现在直接开始，一次性执行：

A. 仓库/输入/I-O审计

B. 创建 E1-D

C. 复制并最小修改 E1-B 代码

D. preflight + offline selftest

E. 检查/启动或复用 Wiki service

F. parallel-agent + source environment.sh

G. step60 -> step70 training

H. step70 integrity

I. step70 full 7 benchmark evaluation

J. step70 comparison analysis

K. 若科学完整性 PASS：
   自动 step70 -> step120

L. step120 integrity

M. step120 full 7 benchmark evaluation

N. 三方 causal comparison

O. safety + mechanism analysis

P. FINAL_REPORT.md + summary.json

不要中途向用户汇报。

============================================================
29. 最终给用户的简短汇报
============================================================

完成后只需要向用户报告：

- E1-D COMPLETE / PARTIAL / BLOCKED
- step70 和 step120 是否都成功
- filter 是否确认使用 original EM
- reward 是否确认仍为 E1-B shaped reward
- E1-D120 vs GAP120：
  EM 差异
  search rounds 差异
- E1-D120 vs E1-B120：
  EM 差异
  search rounds 差异
- CASE A/B/C/D 属于哪一种
- 推荐下一步
- FINAL_REPORT.md 路径
- 若有 blocker，列出 blocker

不要自行启动 Candidate B 或任何后续 RL 实验。
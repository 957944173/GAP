你现在负责在我本地已经完整复现的 GAP（Graph-Agent-Planning）代码仓库中，一次性完成 Experiment E1-0。

这是一个真实的强化学习实验任务，不只是修改代码。你需要自行完成：
代码审查 → 创建最小侵入的 E1-0 代码分支 → 输入输出预检 → 环境检查 → Wiki 服务检查/必要时启动 → 从 RL step60 恢复 → 使用 ORIGINAL GAP REWARD 再训练 10 step 到 step70 → 保存所有 runtime audit → 统计分析 → 离线 reward replay（不改变训练）→ 输出完整最终报告。

整个过程中不要向我阶段性汇报，不需要等待我确认。
遇到能够自行定位和修复的问题，请自行修复并记录。
只有遇到无法安全判断、继续执行可能破坏已有实验结果/覆盖文件/使用错误 checkpoint 的情况，才允许停止；此时不要冒险执行，应把 blocker、已完成工作、诊断过程全部写入 E1-0/FINAL_REPORT.md。

========================
0. 核心实验原则
========================

这是 E1-0，不是 E1-A。

E1-0 的唯一训练目标是：

    step60  -- ORIGINAL GAP reward --> step70-original-E1-0

训练 reward 必须与原 GAP 完全一致。

禁止在 E1-0 中真正启用任何 efficiency reward / group-relative reward / search-round penalty / bonus。

我们这一阶段只做“被动审计”：
额外观察和记录 rollout/group/tool 行为，但不能改变 reward、采样、GRPO、训练数据、optimizer、learning rate、filtering、rollout、tool execution 等训练语义。

E1-0 的核心控制变量要求：

    E1-0 reward == original GAP reward

逐 rollout 必须验证：
    reward_original == reward_E1_0

允许浮点误差时也必须实际为 0/1 EM，最终报告中给出：
    mismatch_count
    max_abs_diff
    equality_rate

目标：
    mismatch_count = 0
    max_abs_diff = 0
    equality_rate = 100%

如果不满足，不允许把该 run 认定为有效 E1-0。

========================
1. 以本地仓库为唯一事实来源
========================

首先定位仓库根目录。

从以下本地入口开始向下审查整个实际调用链：

    Agent/train/mhqa_agent/rl/run_gen_bs_supervisor.sh

沿该脚本实际检查：

1. 使用的训练 parquet 数据路径
2. 模型/checkpoint 路径
3. step60 resume 路径
4. trainer total steps / save frequency / test frequency
5. reward manager
6. custom reward function path/name
7. rollout n
8. GRPO / advantage estimator
9. tool config
10. Wiki retriever URL
11. trainer.default_local_dir / checkpoint output
12. stdout/stderr/log 输出
13. SwanLab/W&B 等 experiment name/output
14. Ray 临时目录/日志（如果相关）
15. resume_mode / optimizer / scheduler 的恢复机制

不要根据文件名猜，必须沿真实代码和配置确认。

同时检查当前实际版本：

    verl/verl/workers/reward_manager/batch.py
    verl/verl/utils/reward_score/mhqa_train.py
    verl/verl/workers/rollout/sglang_rollout/sglang_rollout.py

以及它们实际依赖的：
    reward registry
    trainer filtering
    GRPO advantage
    DataProto
    tool metrics schema
    XML parser
    wiki tool


========================
2. 原始代码必须保持不变
========================

原则：
    baseline 原文件不直接修改。

先复制/创建 E1-0 专用版本，然后只修改 E1-0 版本。

优先采用“薄 wrapper / subclass / import 原函数”的方式复用原逻辑，
不要大段复制实现，除非框架加载机制要求必须复制。

建议命名可以根据当前 repo 实际调用机制调整，例如：

    run_gen_bs_supervisor_e10.sh
    batch_e10.py
    mhqa_train_e10.py

但不要机械使用这些名称。
先确认 registry/import/config 如何加载，再选择最小修改方案。

任何原始文件如果确实不得不修改才能注册新模块：
    1. 创建备份
    2. 修改必须极小
    3. 在报告中解释原因
    4. 生成完整 diff
优先避免这种情况。

所有修改前后执行：

    git status
    git diff

保存记录。

========================
3. E1-0 文件夹规则
========================

仓库根目录统一创建：

    E1-0/

E1-0 的所有审计、日志、代码快照、统计和最终报告必须放在这里或其子目录。

不要覆盖已有 E1-0 内容。


推荐结构：

    E1-0/
      FINAL_REPORT.md
      latest_run.txt
      runs/
        <run_id>/
          preflight/
          code/
          configs/
          logs/
          audit/
          analysis/
          checkpoints_manifest/
          patches/
          wiki/
          environment/
          FINAL_REPORT.md

根目录：
    E1-0/FINAL_REPORT.md

应最终复制/汇总最新有效 run 的结论，方便我直接阅读。

禁止覆盖：
    旧 checkpoint
    旧训练 log
    旧 benchmark result
    旧 SwanLab/W&B run
    原 GAP step60/70/120/... 目录

新实验的：
    experiment_name
    local_dir
    log filename
    checkpoint directory
必须使用唯一 E1-0 标识。

========================
4. PRE-FLIGHT：训练前必须完成
========================

不要一上来运行 GPU 训练。

先生成：

    RUN_DIR/preflight/preflight_report.md
    RUN_DIR/preflight/preflight.json

检查并记录：

A. 仓库
    git root
    branch
    commit
    git status
    dirty files
    当前 Python package 实际 import 路径

B. 输入训练文件
对 run_gen_bs_supervisor.sh 所引用的每个输入文件：
    realpath
    exists
    size
    modification time
    parquet rows/schema（如适用）

文件不存在必须先根据本地原复现配置自行定位正确路径。
严禁悄悄换成其他 dataset。

C. step60 checkpoint
必须确认真正恢复的是此前 GAP RL 的 step60。

至少检查：
    checkpoint path
    global step metadata
    model files
    optimizer state
    scheduler state
    trainer state
    是否完整可 resume

如果存在多个 step60 candidate：
优先依据原 supervisor 配置、原 training log、checkpoint metadata 交叉确定。

如果仍无法唯一确定：
停止实验并写 BLOCKED。
不要凭猜测选 checkpoint。

D. 输出覆盖风险
对即将使用的：
    checkpoint output
    log
    results
    generated samples
    experiment tracker
逐项检查是否已存在。

若存在：
绝对不要 rm/rm -rf 原结果。
创建新的唯一 RUN_DIR 和 experiment name。

========================
5. Wiki 服务启动顺序：严格执行
========================

训练和评估的先决条件是 Wiki 服务可用。

必须按以下顺序执行。

第一阶段：retriever 环境

初始化 conda：
    source "$(conda info --base)/etc/profile.d/conda.sh"

然后：
    conda activate retriever

不要马上再启动一个 Wiki server。

先根据本地代码/配置确认：
    Wiki server 启动脚本
    corpus/index/model 路径
    host
    port
    API endpoint
    请求格式

检查现有服务。

至少检查：
    port 是否监听
    process
    HTTP endpoint
    实际 functional retrieval request

只看端口 LISTEN 不算健康。

必须发送一条与 wiki tool 实际格式一致的最小 retrieval 请求，
并确认返回结构可被当前 wiki tool 正常解析。

如果 Wiki 服务已经健康：
    复用现有服务
    记录：
        reused = true
        pid/process（若可获得）
        URL
        health response summary
    不允许启动重复服务。

如果未运行/不健康：
    使用当前本地复现时真实使用的启动方式启动。
    不要自行发明新的 server 实现。

server stdout/stderr 保存到：
    RUN_DIR/wiki/

记录 PID。

等待 functional health check 成功后再继续。

如果启动失败：
自行诊断和修复合理的环境/路径问题。
若最终仍不可用，则停止 E1-0 并完整报告。

禁止为了训练而关闭一个原本已经健康的用户 Wiki server。

========================
6. parallel-agent 环境
========================

Wiki server 健康后：

    conda activate parallel-agent

然后到仓库根目录执行：

    source environment.sh

这一步必须在启动 RL/评估之前完成。

source environment.sh 后再次进行 Wiki functional check。

必须确认：
    当前 WIKI_RAG_SERVER_URL
与刚刚验证的服务一致并可访问。

如果 environment.sh 导致 URL 指向错误 host：
优先理解原复现环境的设计并安全修复 E1-0 启动脚本，
不要随意修改原 environment.sh。

========================
7. E1-0 代码目标
========================

E1-0 只增加 passive audit。

不要修改：
    SGLang tool execution semantics
    asyncio concurrency
    rollout sampling
    n=8
    prompts
    training parquet
    original EM definition
    GRPO
    advantage estimator
    PPO/DAPO parameters
    learning rate
    optimizer
    group filtering
    tool config

sglang_rollout.py 第一版原则上不要改。

只有在经过实际检查后确认必要 metadata 根本无法获得，
才允许增加纯 logging metadata；
且不得改变 rollout 行为。

========================
8. Reward Manager 的 E1-0 最小修改
========================

创建 E1-0 专用 reward manager。

复用原 BatchRewardManager 的绝大部分逻辑。

需要把 scorer 做审计所需的现有 metadata 被动转发，例如当前 DataProto 中实际存在的：

    uid
    detailed_tool_metrics
    messages
    tool_call_info

具体字段必须先检查当前 runtime/schema，不要假设。

对不存在字段：
    使用安全 get
    标记 metadata_valid=false
不要让 audit 字段缺失导致训练失败。

同时把：

    RL score
和
    actual EM accuracy

从代码语义上分开。

即新 manager 中：
    reward = score["score"]
    acc = score.get("em", original_em)

E1-0 因为 score == em，数值仍完全相同。

这么做只是为了未来 E1-A 不需要再次动 manager。

========================
9. E1-0 scorer
========================

不要改原：

    compute_score_em()
    compute_score_em_batch()

创建新的 E1-0 batch scorer。

必须直接复用原 compute_score_em 的 EM 结果，
不要自己重新实现 normalize/extract/EM 规则。

每条 trajectory：

    original = compute_score_em(...)

然后：

    score = original["score"]
    em = original["em"]

E1-0 必须：
    score == em
    不添加任何 shaped reward。

同时只计算 audit metrics。

========================
10. 每条 rollout 必须采集的 audit 字段
========================

尽量从 structured metadata 与 response 同时计算。

至少记录：

身份：
    global_step
    timestamp
    uid
    extra_info.index
    rollout position within uid group
    question identifier（不要丢）
    data_source

正确性：
    em
    reward_score
    answer extracted / parse status（合理情况下）
    completion_reason

文本/长度：
    response token length
    response char length

搜索行为：

A. logical_search_batches
    response 中真实 assistant 生成的非空
    <wiki_search>...</wiki_search>
    逻辑 block 数

注意避免 observation 中出现相同字符串导致误计。
如果只能从 response 解析，要使用与当前 trajectory 格式一致的安全 parser。

B. search_queries
    实际 query 数
    对 "|" 并行 query 的拆分规则必须与当前 XML tool parser / wiki tool 一致。

C. structured_search_rounds
    如果 detailed_tool_metrics/tool sequence 可恢复：
        wiki_search 出现的 unique execution/assistant turn 数

D. structured_search_calls
    structured metadata 中实际 wiki search call 数

E. derived：
    queries_per_round
    parallel_factor
    zero_search
    metadata_valid

如果 structured 字段含义与旧审计不一致：
以当前代码/runtime 为准，并在报告解释。

========================
11. Group-level audit
========================

E1-0 最关键的是 n=8 同 prompt group。

必须在 reward/filter 前拿到完整 group。

确认真实 group key。

预期可能是 uid，但必须运行时验证。

对每个 group 记录：

    group_id
    group_size
    extra_info.index unique values
    correct_count
    incorrect_count

按每种 candidate cost：
    min
    max
    mean
    variance
    correct-only min/max
    correct-only variation

重点统计：

P_group8
    P(group_size == 8)

P2
    P(correct_count >= 2)

Pvar_xml
    P(correct trajectories 有 logical_search_batches variation
      | correct_count >= 2)

Pvar_structured
    同上，但 structured_search_rounds

P_all_correct
    P(correct_count == 8)

P_all_correct_costvar
    all-correct group 中存在 cost variation 的比例

P_zero_search_correct
    P(cost == 0 | correct)

signal_density
    P(correct_count >= 2 且 correct trajectories cost 有 variation)

输出 correct_count=0..8 的完整 histogram。

========================
12. XML batch vs structured round runtime 对齐
========================

这是 E1-0 的核心结论之一。

逐 trajectory 比较：

    logical_search_batches
vs
    structured_search_rounds

统计：

    both_valid_count
    exact_match_count
    exact_match_rate
    mean_abs_difference
    mismatch_count

更重要的是 group ranking：

对于同一 group 的正确 rollout，
比较两种 cost 是否产生相同排序。

统计：

    correct_group_ranking_agreement
    ranking_changed_group_count
    ranking_changed_group_rate

保存至少若干典型 mismatch cases，
但不要无限输出全文：
    question
    response relevant excerpt
    XML batch
    structured rounds
    tool call sequence summary
    reason classification

E1-0 不需要强行决定最终 reward，
但最终报告必须基于结果给出：
    推荐 XML logical batch
或
    推荐 structured search round
及证据。

参考判据，但不要机械执行：

若：
    exact alignment >= 98%
并且：
    correct-group ranking 几乎完全一致
且 mismatch 没有系统性语义问题
则优先推荐 logical XML batch，
因为实现最简单。

若：
    mismatch > 5%
或：
    correct group 的效率排序存在明显变化
或：
    多 XML block 实际属于同一并发 round 的系统性现象
则优先推荐 structured search round。

2%-5% 灰区需要结合 mismatch 类型判断，并在报告解释。

========================
13. Baseline group filtering 的离线审计
========================

不要修改 trainer filter。

检查当前源码实际 filter_groups 逻辑。

在 audit 数据上离线模拟 ORIGINAL GAP filtering。

统计：

    groups_before_filter
    groups_expected_kept_by_original_reward
    groups_expected_removed
    all-correct groups removed
    all-wrong groups removed

然后离线模拟未来 shaped reward 时，
统计可能被“新保留”的 all-correct groups。

这里只做分析：
    不修改 E1-0 training filtering。

========================
14. Offline reward replay（不进行 shaped reward 训练）
========================

E1-0 训练完成后，可在 CPU 上进行 E1-0.5 性质的离线 replay，
结果仍统一放进 E1-0 本次 RUN_DIR。

不要启动新的 GPU RL。

只用已经采集的真实 n=8 rollout。

候选 future reward：

    A_i = EM correctness

若同 group：
    correct_count >= 2
且正确 rollout cost 有 variation：

    E_i = (Cmax - Ci) / (Cmax - Cmin)

否则：
    E_i = 0

候选：
    R_i = A_i * (1 + lambda * E_i)

离线测试：
    lambda = 0.05
    lambda = 0.10
    lambda = 0.20

注意：
这只是 replay，不得用于 E1-0 训练。

尽量直接复用当前代码中的：
    compute_grpo_outcome_advantage()
或真实 GRPO advantage 实现。

对每个 lambda 统计：

    reward range
    correct > incorrect 是否始终成立
    correct-efficient vs correct-inefficient advantage separation
    mixed correctness groups 的变化
    all-correct groups 的变化
    potential newly-retained groups
    NaN/Inf
    advantage distribution

给出下一阶段 E1-A 推荐 lambda，
但不要启动 E1-A。

========================
15. 启动 step60 → step70
========================

只有全部 preflight 和 Wiki health 通过后才训练。

从本地原始：

    Agent/train/mhqa_agent/rl/run_gen_bs_supervisor.sh

复制出 E1-0 专用入口。

保持原所有训练参数不变，
只做必要修改：

    resume = step60
    stop/save at step70
    E1-0 unique output path
    E1-0 unique experiment name
    使用 E1-0 audit reward manager/scorer
    audit enabled
    reward shaping disabled

确认：
    rollout n 不变
    batch size 不变
    seed 不变（如果原本固定）
    LR 不变
    total effective update 数正确
    optimizer/scheduler resume 正确

如果原 supervisor 的 total step/stop 逻辑比较复杂，
安全实现恰好新增 10 个 global steps，
最终必须证明：
    start_global_step = 60
    end_global_step = 70

不要训练到 71。
不要从 0 重训。
不要仅加载 weights 却错误重置 optimizer，
除非原 step60 本来就没有完整 state；
若发生此情况，必须报告，这个 run 不能声称是严格 matched continuation。

========================
16. 训练监控与自动处理
========================

运行过程中无需向我汇报。

保存完整：
    stdout
    stderr
    training log
    Ray errors
    reward metrics
    checkpoint events

遇到普通错误：
    路径
    import
    registry
    JSON serialisation
    audit logger
    文件权限
    端口
等，可自行修复并重新运行。

但修复后：
如果已经发生过 optimizer update，
不能简单在同一个 run 上从头/中途混杂继续却不记录。

应：
    标记失败 attempt
    新建 attempt 子目录
    从干净 step60 重新开始有效 E1-0
或根据 checkpoint 语义明确恢复。

严禁因为 audit logger 出错而改变训练数学逻辑。

========================
17. Audit logging 的可靠性
========================

训练可能涉及多个 process/rank。

不要让多个进程无锁竞争写同一个 JSONL 导致损坏。

可采用：
    rank/pid 分文件
例如：
    groups_rank0_pidXXXX.jsonl

训练后统一 merge。

每条 audit record 要能追溯：
    run_id
    rank
    pid
    global_step
    uid

如果 reward manager 实际只在单一进程执行，
报告中证明后可以用单文件。

========================
18. 训练完成后的完整性检查
========================

step70 完成后检查：

    checkpoint 是否存在
    checkpoint global step == 70
    model state
    optimizer state
    scheduler state
    trainer state

确认 training log 中：
    step61...step70
均存在合理记录。

检查：
    NaN
    Inf
    CUDA OOM
    NCCL error
    Ray crash
    Wiki request failure
    malformed audit log

统计 Wiki tool failure rate。

========================
19. E1-0 输出文件
========================

至少生成：

    E1-0/latest_run.txt

RUN_DIR 下：

    preflight/preflight_report.md
    preflight/preflight.json

    environment/environment.txt
    environment/gpu_before.txt
    environment/gpu_after.txt

    code/original_files_manifest.txt
    code/e10_files_manifest.txt
    patches/e10.patch

    configs/resolved_training_config.yaml
    configs/launch_command.txt

    wiki/wiki_status.json
    wiki/wiki_health_check.txt
    wiki/wiki_server.log   # 仅若本次启动

    logs/training.log
    logs/launcher.log

    audit/rollouts*.jsonl
    audit/groups.jsonl
    audit/audit_summary.json
    audit/audit_summary.csv
    audit/mismatch_cases.jsonl

    analysis/group_statistics.md
    analysis/xml_vs_structured.md
    analysis/reward_identity.md
    analysis/offline_reward_replay.md
    analysis/offline_reward_replay.json

    checkpoints_manifest/step60_source.md
    checkpoints_manifest/step70.md

    FINAL_REPORT.md

然后在：
    E1-0/FINAL_REPORT.md
生成一份方便直接上传/阅读的汇总报告。

不要把大模型 checkpoint 复制到 E1-0 里产生重复占用。
E1-0 中保存 checkpoint 的真实路径和 manifest 即可；
实际 checkpoint 放在新实验专用 output dir。

========================
20. FINAL_REPORT.md 必须回答的问题
========================

报告必须明确区分：
    VERIFIED
    OBSERVED
    INFERRED
    BLOCKER（如有）

必须明确回答：

1. E1-0 是否从正确的 step60 开始？
2. 是否完整训练了恰好 10 step 到 step70？
3. optimizer/scheduler 是否真实 resume？
4. 原 GAP reward 与 E1-0 reward 是否逐 rollout 完全一致？
5. n=8 group 是否运行时得到验证？
6. group key 最终是什么？
7. group size=8 的比例？
8. correct_count 分布？
9. P(correct_count >= 2)？
10. 在 >=2 correct group 中，有效率差异的比例？
11. logical XML batch 与 structured search round 的一致率？
12. 它们是否会改变 correct-rollout 的组内效率排序？
13. 推荐 E1-A 使用哪种 cost：
       logical search batch
       或 structured search round
    为什么？
14. all-correct groups 有多少？
15. all-correct 但存在 efficiency variation 的 groups 有多少？
16. 原 GAP filter 会删除多少这些 groups？
17. shaped reward 未来可能重新激活多少 groups？
18. lambda 0.05 / 0.1 / 0.2 离线 replay 结果如何？
19. 下一阶段推荐 lambda？
20. 是否发现任何 reward hacking 风险：
       zero-search
       parallel-query explosion
       malformed search
       metadata mismatch
21. Wiki 服务是否复用还是本次启动？
22. 所有文件是否无覆盖？
23. 原代码是否保持完好？
24. 具体改动了哪些 E1-0 文件？
25. E1-0 是否达到“可以进入 E1-A”的条件？

最后给出一个明确结论：

    E1-0 STATUS:
        PASS
    或
        PASS_WITH_CAVEATS
    或
        FAIL / BLOCKED

以及：

    RECOMMENDED E1-A DESIGN:
        cost = ...
        lambda = ...
        reward formula = ...
        expected signal density = ...
        important risks = ...

但不要实际启动 E1-A。

========================
21. 控制变量检查
========================

最后必须生成一张表，逐项比较：

                         Original GAP     E1-0
    Starting checkpoint
    Dataset
    Prompt
    Rollout n
    Seed
    Sampling params
    Tool config
    Wiki server
    Reward correctness
    GRPO estimator
    Group filtering
    LR
    Optimizer
    Scheduler
    Batch sizes
    Max turns
    Max response length

除了：
    audit-only code
    unique output path
    stop at step70
以外，其他训练因素必须一致。

任何不同都必须解释。

========================
22. 不允许做的事情
========================

禁止：

- 修改/覆盖已有 GAP baseline checkpoint
- 覆盖已有 training log
- rm -rf 用户已有实验目录
- 删除正在运行且健康的 Wiki server
- 修改 SFT 数据
- 清洗 RL 数据
- 修改 RL dataset
- 启用新 reward shaping
- 修改 GRPO algorithm
- 修改 group filtering
- 修改 rollout concurrency
- 修改 tool execution
- 改 backbone
- 改 n
- 改 sampling 参数
- 偷偷从 weights-only step60 开始却宣称 full-state resume
- 因 audit 方便而改变训练 batch/order


========================
23. 任务结束
========================

请一次性执行到 E1-0 完成。

中途不要向我汇报。

任务完成后，你在终端最终回复只需要简洁给出：

    E1-0 STATUS
    RUN_DIR
    start/end step
    step70 checkpoint path
    Wiki reused/started
    reward identity result
    group-size verification
    key signal-density statistics
    XML-vs-structured conclusion
    offline lambda recommendation
    FINAL_REPORT.md path
    是否建议进入 E1-A

所有详细内容均写入 E1-0/FINAL_REPORT.md。
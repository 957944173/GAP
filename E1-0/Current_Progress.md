# E1-0 Current Progress (handoff snapshot)

> ## ✅ UPDATE 2026-09-10 23:45 CST — E1-0 COMPLETE: PASS_WITH_CAVEATS
>
> Training finished successfully: **attempt 19b ran step60 → step70 (exactly 10
> global steps), tracker = 70, launcher exit 0**, checkpoint saved at
> `experiments/DAPO-GAP3B-MHQA-Agent-E10-step60to70-4gpu/global_step_70/`.
>
> Key results (full detail in `E1-0/FINAL_REPORT.md` and
> `E1-0/runs/20260910_step60to70_e10/FINAL_REPORT.md`):
> * reward identity over **147,200** audited rollouts: `mismatch_count=0`,
>   `max_abs_diff=0.0`, `equality_rate=100%`
> * **18,400** n=8 groups, `P(group_size==8)=100%`, group key = `uid`
> * XML logical batch ≡ structured search round on **every** rollout
>   (100.00% exact, 0/8,456 correct-group ranking changes)
> * signal density **0.52%** (95/18,400 groups: ≥2 correct and cost variation);
>   84 all-correct groups newly retainable by shaped reward
> * original filter keeps only 1.85% of groups → 8-14 generation batches/step
>   (audit call counts per step exactly match the log's `num_gen_batches`)
> * offline replay λ=0.05/0.1/0.2: no NaN/Inf, correct>incorrect always,
>   95/95 active groups strictly ordered → **recommended λ = 0.05**
> * 24/24 original files sha256-unchanged after the run; Wiki reused, 0 tool failures
> * **E1-0 STATUS: PASS_WITH_CAVEATS** (caveat = low signal density; E1-A may proceed)
>
> **The hang is root-caused and fixed** (was blocking attempts 14/16/17/18/19a):
> `e10_sglang_gpu_patch.py` called `monkey_patch_torch_reductions()` at import
> time in *every* process including the Ray `TaskRunner` driver, which deadlocks
> fork-based `StatefulDataLoader` workers (CUDA work in a forked child). The call
> is now deferred to the WorkerDict/FSDP actor. Evidence + reproduction:
> `analysis/root_cause_dataloader_fork.md` and
> `logs/attempt19a_driver_and_dataloader_worker_SIGUSR1_stacks.txt`.
>
> Audit-correctness fixes made before the run (all verified by
> `code/e10_selftest.py` and by the run's own records): val-manager
> contamination removed; `global_step` read from the live trainer; XML/logical
> batch count reconstructed from the parser block index (the literal
> `<wiki_search>` string never survives into the response in this repo); and
> per-rollout reward identity verified against an independent re-run of the
> ORIGINAL `compute_score_em_batch`.
>
> The snapshot below is the pre-session handoff and is kept for history; its
> "hang unresolved / confirm before running" guidance is superseded.

**写作时间**: 2026-09-10 17:50 CST（previous snapshot）
**写作目的**: 供接手方（**本次为其他 AI 工具/新对话窗口，用户已明确交接**）使用。接手方应同时阅读 `E1-0/E1-0_task.md`（原始任务定义，未变更）和本文件（当前进度与关键约束），然后继续完成任务。

---

## 0. 一句话总结现状

**训练仍未成功完成一步（step60→70 完全没有跑起来）**。已进行 18 次尝试：

- **Attempt 1-13**：多种环境级 bug（NCCL 冲突、GPU 显存冲突、CUDA OOM、CUDA IPC 设备序号错误、诊断代码自身导致的 segfault），**全部已定位并修复**，修复方案见下文第3节表格，均已固化进当前脚本/monkeypatch。
- **Attempt 14, 16, 17**：出现同一种"挂起"——日志停在 `Resuming from .../global_step_60` 之后彻底无输出，怀疑卡在 `FSDPCheckpointManager.load_checkpoint` 内部的 FSDP 分布式 collective 通信步骤。已加入被动诊断埋点（`e10_checkpoint_patch.py` 里的 `_e10_diag` 打点），未改变任何训练语义。
- **Attempt 18（本次会话新增，最重要的进展）**：诊断埋点首次完整触发，**证实 checkpoint-load 阶段本身这次顺利走完**（4 个 rank 全部完成 `torch.load` → `load_state_dict` → 退出 FSDP context → 最终 `torch.distributed.barrier()`，rank 0 在进入后约76秒完全返回）——即 attempt 14/16/17 的挂起问题**大概率已不再复现**（原因未 100% 确认：可能是非确定性 bug 恰好这次没触发，也可能环境状态发生了变化）。
  但是，checkpoint-load 完成之后，训练**在下一步（`_validate()`，训练前的验证阶段）又出现了一次新的、独立的挂起**：日志停滞近 2 小时，GPU 利用率全程 0%，4 个 WorkerDict 进程的 `.out`/`.err` 文件在 checkpoint-load 完成的瞬间就停止写入、之后再未增长，`_validate()` 函数第一行必定会打印的日志（`test_gen_batch meta info: ...`）从未出现过一次。**用户当时敏锐地质疑了"CPU一直在忙所以像是在正常工作"这个解读**——经查 SGLang scheduler 源码（`recv_requests()`）证实其主循环即使完全空闲、没有真实请求，也会用非阻塞 ZMQ 轮询方式持续占用接近 100% 的单核 CPU，因此"CPU持续繁忙"这个信号本身**不能**用来判断是否有真实工作在进行。综合 GPU 0% + `_validate()` 首行日志缺失 + worker 输出文件冻结 + Ray 内部任务计数器静止这四类证据，判定这是一次**新的、独立的挂起**，已于 17:42 左右人工干净终止（`kill -TERM`，无 optimizer update 发生，符合任务第16节的"无混合状态风险，可直接干净重启"条款）。

**当前 GPU 上没有任何 E1-0 相关进程在跑**（已确认 `ps aux | grep -E "main_ppo|sglang|ray::|WorkerDict"` 为空），GPU 显存已回落到 Wiki-only 基线（约9.7-10.2GB/GPU），无残留 partial checkpoint，Wiki RAG 服务器健康（`/retrieve` 返回 HTTP 200），原始 GAP checkpoint/日志未被触碰。

**关于"重新上机前必须等待用户确认"的约束**：此前用户曾要求"修复完成后必须等待用户明确确认，才能再次尝试上机"。本次会话中，用户已于稍早消息中明确授权发起 attempt 18（"我已启动wiki服务，请...运行最新修复后准备尝试上机运行的训练脚本"），attempt 18 已经按此授权跑过并失败于新的挂起点。**接手方在发起 attempt 19 或任何新的 GPU 训练尝试之前，仍应视为需要用户/使用者的明确批准**——本文件不构成"已获批准可直接上机"的许可，请在实际操作前与当前使用者确认这一点。

---

## 1. 任务目标回顾（详见 E1-0_task.md，未做任何修改）

- 从 step60 checkpoint 恢复 GAP RL 训练，**只跑 10 个 global step 到 step70**。
- reward 必须与原始 GAP 训练**完全一致**（纯 EM，`score == em`，无任何 shaping），需逐条 rollout 验证 `mismatch_count=0`、`max_abs_diff=0`、`equality_rate=100%`。
- 全程只做**被动审计数据采集**，绝不能修改 reward/采样/GRPO/优化器/学习率/过滤/rollout/工具执行的语义。
- 训练成功后需完成大量统计分析（task 第11/12/13/14节）并撰写 `FINAL_REPORT.md`（回答第20节的25个具体问题，附 VERIFIED/OBSERVED/INFERRED/BLOCKER 标签，以及最终 `E1-0 STATUS:` 和 `RECOMMENDED E1-A DESIGN:`）。
- 任务文档明确授权全程自主执行、无需中途汇报确认（除非遇到可能破坏结果/覆盖文件/用错 checkpoint 的 BLOCKER 级问题）——但历次对话中用户曾就"重新启动 GPU 训练"这一具体动作提出过显式的等待确认要求，接手方应对此保持敏感，涉及上机操作前主动确认。
- task 第16节明确：如果 optimizer 已经发生过 update 后又失败，不能静默地在混合状态下继续，必须标记该次 attempt 失败、开新的 attempt 子目录，或从 step60 干净重启。**目前所有失败的 attempt（含 attempt 18）都发生在 optimizer update 之前**（要么卡在 checkpoint-load，要么卡在 checkpoint-load 之后、训练循环开始之前的 `_validate()` 阶段），从未进入训练循环，所以不存在"混合状态"污染风险，每次都可以从 step60 干净重启，无需新建 attempt 子目录（本次运行始终在同一个 run 目录 `20260910_step60to70_e10` 下）。

---

## 2. 目录与产物现状

运行目录：`/data01/wyy/Graph-Agent-Planning/E1-0/runs/20260910_step60to70_e10/`
（`E1-0/latest_run.txt` 指向此目录）

```
E1-0/runs/20260910_step60to70_e10/
├── analysis/          # 空 —— 训练还未产生任何 rollout/reward 数据，第11/12/13/14节的统计分析全部未开始
├── audit/             # 空 —— 同上，被动审计数据尚未采集
├── checkpoints_manifest/
│   └── step60_source.md          # 已完成：记录 step60 checkpoint 来源信息
├── code/
│   ├── e10_checkpoint_patch.py   # 含诊断埋点（_e10_diag），attempt18 证实 checkpoint-load 路径本身工作正常
│   ├── e10_hang_diagnostics.py   # faulthandler 诊断模块；周期性 dump 已在 attempt15 证实会导致 segfault，已禁用，只保留 SIGUSR1 按需 dump
│   ├── e10_reward_manager.py     # 审计用 reward manager（e10_batch_audit），已实现，未跑过真实数据验证
│   ├── e10_scorer.py             # 包装（不重新实现）原始 compute_score_em 的打分函数
│   ├── e10_sglang_gpu_patch.py   # monkeypatch，修复 NOSET=1 下 SGLang 的 GPU 可见性 union 逻辑 bug
│   ├── e10_files_manifest.txt / original_files_manifest.txt  # 文件清单，已完成
│   ├── pythonpath_e10/sitecustomize.py  # 通过 PYTHONPATH 注入，统一加载以上所有 monkeypatch 模块
│   └── run_e10_step60to70.sh     # 主启动脚本，已完成，多轮 self-fix 后当前版本稳定（见下文第3节历史）
├── configs/           # launch_command.txt / resolved_training_config.yaml —— 已完成
├── environment/       # environment.txt / gpu_before.txt —— 已完成（preflight 阶段快照）
├── logs/
│   ├── launcher_attempt1~13*.log          # attempt 1-13：各种已诊断清楚并修复的崩溃（NCCL/GPU碰撞/OOM/CUDA IPC等）
│   ├── launcher_attempt14_hung_sglang_tp_no_progress.log   # 挂起（checkpoint-load阶段），未崩溃
│   ├── launcher_attempt15.log                              # 诊断模块自身导致 segfault（已修复：禁用周期性 dump）
│   ├── launcher_attempt16_hung_partial_dpgroup_capture.log # 挂起（checkpoint-load阶段），未崩溃
│   ├── launcher_attempt17_hung_same_signature_as_14_16.log # 挂起（checkpoint-load阶段），与14/16同一信号
│   ├── launcher_attempt18.log                              # 【本次会话新增】checkpoint-load 顺利完成，但之后在 _validate() 附近挂起约2小时后被终止
│   ├── *_diagnosis.md / *_investigation.md   # 各次 attempt 的详细诊断文档
│   ├── checkpoint_load_hang_investigation.md # 综合调查文档；结论已被 attempt18 更新（见下文第4节）
│   ├── attempt18_ckptload_fixed_but_hung_post_load_diagnosis.md  # 【本次会话新建】attempt18 完整诊断：checkpoint-load 问题疑似解决 + 新挂起点的证据与分析
│   └── stackdumps/    # SIGUSR1 按需诊断输出目录（周期性 dump 已禁用）
├── patches/e10.patch  # 已完成：本次实验所有 monkeypatch 的统一 diff 快照
├── preflight/          # 已完成：启动前环境/依赖检查
└── wiki/               # 已完成：Wiki RAG 服务器健康检查记录（服务器本身在 E1-0 之外独立运行，全程健康，未被触碰）
```

**尚未开始/无法开始的部分**（全部依赖训练真正跑到 step70）：
- `analysis/`、`audit/` 目录下的所有统计分析产出（task 第11/12/13/14节）
- task 第18节的训练后完整性检查
- task 第19节要求的全部输出文件清单
- `RUN_DIR/FINAL_REPORT.md` 和 `E1-0/FINAL_REPORT.md`（task 第19-21节）
- task 第21节的对照变量比较表
- task 第23节的终端总结格式

---

## 3. Attempt 历史简要（完整细节见各 `logs/*_diagnosis.md`）

| Attempt | 结果 | 根因 | 状态 |
|---|---|---|---|
| 1-3, 6-7 | NCCL "Duplicate GPU detected" 崩溃 | Ray 默认按 actor 重映射 CUDA_VISIBLE_DEVICES，FSDP 初始化冲突 | 已修复：`RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` |
| 4 | CUDA OOM 崩溃 | checkpoint 里 pickle 的张量带着旧的 "cuda:0" 设备标签，NOSET=1 下无重映射，4 个 rank 全挤到物理 GPU0 | 已修复：`e10_checkpoint_patch.py` 给三个 `torch.load()` 加 `map_location="cpu"` |
| 5, 8 | SGLang GPU 显存碰撞 | NOSET=1 破坏了 SGLang `_init_distributed_env` 的 GPU union 逻辑 | 已修复：`e10_sglang_gpu_patch.py` |
| 9 | NCCL 崩溃复现（GPU全部落到GPU0） | 第一版修复方案（直接改 `Worker.__init__`）无效，因为 CUDA context 已固化，改环境变量不生效 | 已修复：改为 patch `SGLangRollout._init_distributed_env`，从 `RAY_LOCAL_RANK` 推导正确 GPU |
| 10-12 | 宿主机 RAM OOM 崩溃 | Ray 默认预启动 ~96 个空闲 worker，每个吃 ~0.47GB | 已修复：`ray_init.num_cpus=16` |
| 13 | `CUDA error: invalid device ordinal` 崩溃 | FSDP→SGLang 权重同步时 CUDA IPC 序列化未做跨进程 GPU 可见性适配 | 已修复：调用 sglang 自带的 `monkey_patch_torch_reductions()` |
| 14 | **挂起**（checkpoint-load 阶段，"Resuming from..."之后无输出） | 未最终确认；已排除 async-rollout 路径 | **疑似已解决**（见 attempt18） |
| 15 | **诊断模块自身导致 segfault** | `faulthandler.dump_traceback_later` 周期性 C 级 watchdog 线程在 GIL 释放期间触发，破坏 Python 线程状态 | 已修复：禁用周期性 dump，仅保留 SIGUSR1 按需 dump |
| 16 | **挂起**（checkpoint-load 阶段，同上信号；一开始误判为跨DP组同步 bug，后已修正诊断文档） | 同上 | **疑似已解决**（见 attempt18） |
| 17 | **挂起**，与14/16同一信号（checkpoint-load 阶段无输出） | 同上 | **疑似已解决**（见 attempt18） |
| 18 | checkpoint-load **本身顺利完成**（诊断埋点证实4个rank全部走完 torch.load→load_state_dict→barrier）；但**紧接着在 `_validate()` 附近出现新的独立挂起**，约2小时无进展后人工终止 | 未确认；候选位置：FSDP→SGLang 权重同步/resharding handoff（`rollout_sharding_manager`），或 `_validate()` 内部/RPC调度路径本身 | **未解决，新问题，详见 `attempt18_ckptload_fixed_but_hung_post_load_diagnosis.md`** |

**当前有效、已验证工作的核心环境设置组合**：
`RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` + `NCCL_CUMEM_ENABLE=0` + `e10_sglang_gpu_patch.py`（GPU可见性修复）+ `monkey_patch_torch_reductions()`（CUDA IPC修复）+ `e10_checkpoint_patch.py`（`map_location="cpu"` 修复 + 诊断埋点）+ `ray_init.num_cpus=16`（RAM修复）—— 这套组合已被证明能避免 attempt 1-13 遇到的所有崩溃，且 attempt 18 证实 checkpoint-load 阶段本身也已能顺利走完。**当前唯一的未解决问题是 attempt 18 新发现的、checkpoint-load 完成之后的挂起**。

---

## 4. 当前状态：checkpoint-load 挂起（疑似已解决）+ 新的下游挂起（未解决，本次会话主要发现）

### 4.1 Checkpoint-load 挂起（attempt 14/16/17）—— 疑似已解决

完整历史见：`E1-0/runs/20260910_step60to70_e10/logs/checkpoint_load_hang_investigation.md`

`e10_checkpoint_patch.py` 中加入的分阶段诊断打点（`_e10_diag`，覆盖 `load_checkpoint` 入口、进入/退出 `get_fsdp_state_ctx`、每个 `torch.load()`/`.load_state_dict()` 前后、最终 `torch.distributed.barrier()` 前后）在 attempt 18 中**首次完整触发**，证实4个 rank 全部顺利走完整个序列，rank 0 的 `load_checkpoint` 在进入约76秒后完全返回。**这与之前排名第一的假设（FSDP `_sharded_pre_load_state_dict_hook` 内的 `dist.all_gather_into_tensor` collective 卡死）不符**——如果该假设成立，本次也应该卡住，但没有。

结论：checkpoint-load 阶段的挂起**大概率是非确定性的**（环境/调度层面的偶发问题），或者已经被之前某次修复（`map_location="cpu"`、`monkey_patch_torch_reductions()` 等）间接消除，但没有 100% 确证的根因。**建议**：如果接手方继续排查，可保留现有诊断埋点（不影响训练语义，可继续沿用），若未来再次复现该挂起，会立刻在日志里看到卡在哪一行。

### 4.2 新发现的下游挂起（attempt 18，未解决，当前最优先问题）

完整证据见：`E1-0/runs/20260910_step60to70_e10/logs/attempt18_ckptload_fixed_but_hung_post_load_diagnosis.md`

**症状**：checkpoint-load 完成（rank 0 打印 `load_checkpoint: fully returned`）之后，训练进入 `fit()` 的下一步 `_validate()`（因为 `trainer.val_before_train=true`），随后约2小时**零进展**：
- GPU 利用率全程 0%（4张卡）。
- `_validate()` 函数体内、driver 端无条件执行的 `print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")`（`ray_trainer.py` 约794-910行，在每个 val batch 分发生成请求之前同步打印）**从未出现过一次**（`grep -c` = 0）。
- 4 个 WorkerDict 进程各自的 `.out`/`.err` 文件（`/tmp/ray/session_latest/logs/worker-*.{out,err}`）在 checkpoint-load 完成的同一时刻（15:48-15:50）就停止写入，之后近2小时再未增长一个字节。
- Ray 的 `debug_state.txt` 显示节点上 "Number of executing tasks: 0"、"Waiting tasks size: 0"；driver 自己的 core-worker 日志里 "total number of task attempts sent" 计数器在30秒复查窗口内静止不变（39），说明 driver 并未在持续派发新的 Ray 任务。
- **关键的误判纠正**：SGLang scheduler 进程（4个 rank 各自的 scheduler）在整个挂起期间持续消耗接近单核100% CPU，此前一度被误判为"仍在做真实的验证生成工作"。但阅读 SGLang 源码（`sglang/srt/managers/scheduler.py` 的 `recv_requests()`）证实其主循环用**非阻塞** ZMQ 轮询（`recv_pyobj(zmq.NOBLOCK)`），空闲等待时同样会持续占用接近100%单核 CPU——即"CPU持续繁忙"这个信号本身**不能区分**"真的在处理请求"和"完全空闲、只是在空转轮询"两种情况。结合上面几条证据，更合理的解释是：SGLang engine 处于空闲空转状态，`generate_sequences` 请求根本没有真正送达/被处理。

**候选卡点（尚未定位到具体哪一个）**：
1. FSDP→SGLang 权重同步/resharding handoff（`fsdp_workers.py` 中 `generate_sequences` 里的 `with self.rollout_sharding_manager:` context，约644/656行）——checkpoint 刚 reload 之后，理论上需要把最新权重同步进 SGLang engine，这个握手过程尚未被诊断埋点覆盖。
2. `_validate()` 内部、`test_gen_batch.meta_info` 打印之前的某个环节（可能性较低，因为该打印是函数体内最早使用 `test_gen_batch` 的语句，但 `val_dataloader` 迭代/`DataProto` 构造尚未完全排除）。
3. RPC 派发路径本身：driver 从未真正发出 `generate_sequences` 调用，还是 worker 收到了调用但在触达 SGLang 之前就卡住——这两种情况从现有证据还无法区分。

**已采取行动**：无 optimizer update 发生，符合任务第16节"无混合状态风险"条款，已于 17:42 左右人工 `kill -TERM` 干净终止 attempt 18（先终止启动脚本/driver 进程，随后确认 `ps aux` 中所有 main_ppo/sglang/ray::/WorkerDict 进程均已消失）。终止后已验证：GPU 显存回落至 Wiki-only 基线（~9.7-10.2GB/GPU）、Wiki 服务器健康（`/retrieve` 返回 HTTP 200）、无 partial checkpoint 残留。

---

## 5. 下一步建议（给接手的 AI 工具/新窗口）

1. **在做任何 GPU 相关操作（尤其是发起新的训练尝试）之前，先与当前使用者确认是否已获批准**——历次对话中用户曾要求"修复完成后必须等待用户明确确认才能再次上机"，虽然 attempt 18 已获得一次性授权并已跑完终止，但接手方不应假设这份授权自动延续到 attempt 19 或之后。
2. **建议在下一次尝试之前，先针对本次新发现的挂起点补充诊断埋点**，而不是直接假设"应该没问题了"就重跑：
   - 在 `fsdp_workers.py`（或对应的 E1-0 monkeypatch 副本，如果需要新建一个）的 `generate_sequences` 方法里，在进入/退出 `with self.rollout_sharding_manager:` 前后各加一条 rank-tagged 的被动打印，确认权重同步 handoff 是否是卡点。
   - 在 `ray_trainer.py` 的 `_validate()`（如需通过 monkeypatch 方式，避免直接改 verl 源文件）里，在 `for test_data in self.val_dataloader:` 循环开始前加一条打印，确认是否真的进入了这个循环、还是卡在更早的地方（例如 dataloader 构造、或 `_load_checkpoint()` 返回后到 `_validate()` 调用之间的某处，虽然目前看这段代码路径很短、可能性较低）。
   - 这些打印必须严格遵守任务的被动审计原则：只加 `print()`/日志语句，不改变任何控制流、reward、采样、GRPO、优化器、checkpoint、工具执行的语义。
3. **checkpoint-load 本身的诊断埋点（`e10_checkpoint_patch.py`）建议保留**，不需要回退——它已被证实无害，且如果之前的挂起以非确定性方式复现，这些打点仍然是排查它的唯一手段。
4. **一旦确认新挂起点的具体位置**，可以据此判断是否存在真正需要代码修复的 bug（例如某个分布式 collective 调用参数不对、某个锁未释放等），还是仅仅是同一类非确定性调度问题——処理方式与之前"checkpoint-load 挂起"的处理思路一致：先补充被动诊断，拿到确凿证据后再决定要不要动代码。
5. **一旦训练真正跑到 step70 并成功保存**，才能开始 task 第11-14节的统计分析和 `FINAL_REPORT.md` 撰写工作——这部分工作目前完全没有开始，是任务的主体部分，务必留出足够精力（预计工作量远大于目前为止排查环境问题所花的精力）。
6. **每次新的 attempt 无论成功失败，都要按照已有惯例记录** `launcher_attemptN_<描述>.log` 和对应的 `attemptN_..._diagnosis.md`，保持可追溯性；同时应在本文件（`Current_Progress.md`）中及时更新第0/3/4节，保证任何后续接手方都能从这一份文件快速了解全貌，无需重新翻遍所有历史日志。

---

## 6. 关键文件速查

- 任务定义：`E1-0/E1-0_task.md`（未改动，1018行，以此为准）
- 本文件：`E1-0/Current_Progress.md`
- 运行目录：`E1-0/runs/20260910_step60to70_e10/`（`E1-0/latest_run.txt` 指向此路径）
- 启动脚本：`E1-0/runs/20260910_step60to70_e10/code/run_e10_step60to70.sh`
- checkpoint-load 挂起综合调查（attempt14/16/17，疑似已解决）：`E1-0/runs/20260910_step60to70_e10/logs/checkpoint_load_hang_investigation.md`
- **attempt18 完整诊断（当前最重要的文档，新挂起点的全部证据）**：`E1-0/runs/20260910_step60to70_e10/logs/attempt18_ckptload_fixed_but_hung_post_load_diagnosis.md`
- 最新一次尝试的日志：`E1-0/runs/20260910_step60to70_e10/logs/launcher_attempt18.log`（591行，15:50:29后无新内容）
- 含诊断埋点的 checkpoint patch：`E1-0/runs/20260910_step60to70_e10/code/e10_checkpoint_patch.py`
- step60 原始 checkpoint（只读，从未修改）：`/data01/wyy/Graph-Agent-Planning/experiments/DAPO-GAP3B-MHQA-Agent-4gpu/global_step_60/`

---

## 7. 当前确认的安全状态（截至本文件写作时刻，2026-09-10 17:50 CST）

- GPU 上无任何 E1-0 相关进程运行（已确认 `ps aux | grep -E "main_ppo|sglang|ray::|WorkerDict"` 为空）。
- GPU 显存已回落至 Wiki-only 基线（约9.7-10.2GB/GPU，4张卡）。
- 无残留的部分 checkpoint（`experiments/DAPO-GAP3B-MHQA-Agent-E10-step60to70-4gpu/` 目录不存在）。
- 原始 GAP checkpoint/日志未被修改。
- Wiki RAG 服务器健康（`POST /retrieve` 返回 HTTP 200），未被杀死或重启。
- 所有改动均限定在 `E1-0/` 目录下的专属副本/monkeypatch 中，未原地修改任何 baseline 文件。

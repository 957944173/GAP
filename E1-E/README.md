# E1-E — q=2 Solved-Group Gated / Capped Efficiency Training

E1-E is the **formal, pre-registered online RL experiment** that E1-E0 froze. It tests whether
*solved-group efficiency supervision* can be injected into GAP without the collateral damage
that reward shaping caused in E1-D: instead of shaping every retained group's reward, E1-E keeps
the mixed-group objective exactly as in E1-B (`R_i = A_i`) and adds a **small, quota-gated,
reward-separated** efficiency signal from groups that are *already solved* but differ in search
cost.  The scientific design was frozen by `E1-E0` before any GPU work; nothing in this
directory changes it.

* Task specification: `E1-E/E1-E_task.md`
* Frozen offline design + prediction: `E1-E0/FINAL_REPORT.md`, `E1-E0/results/`
* This run's report: `E1-E/FINAL_REPORT.md`

## Design (frozen, not swept)

| element | value |
|---|---|
| quota `q` | **2** (chosen by E1-E0: 100 % fill, 6.25 % replacement, no difficulty/source bias) |
| `lambda` | **0.05** (frozen; never swept) |
| filter metric | `e1e_quota_metric` — a published per-rollout metric, `EM` for mixed groups and `1 + 0.05·E` for the gated all-correct efficiency groups |
| stopping rule | only **mixed** groups (`1 ≤ k ≤ 7`) advance the generation stopping counter to `B = 32`; all-correct efficiency groups (`k = 8` with correct-rollout cost variation) are only **cached** in stream order, capped at `q` |
| optimizer batch | at the stopping generation batch: `mixed[:B−m] + efficiency_cache[:m]`, `m = min(q, |cache|)`; the last `m` mixed groups of the q=0 batch are displaced |
| reward, mixed group | `R_i = A_i` — the original EM, **no shaping** |
| reward, gated efficiency group | `R_i = 1 + 0.05·E_i`, `E_i = (Cmax − C_i)/(Cmax − Cmin)` over that group's correct rollouts |
| all other groups | dropped by the filter; never trained on |
| cost definition | `logical_search_batches` |
| `acc` | restored to the **true EM** after the reward manager runs |
| validation | original GAP scorer (`mhqa_eval.py:compute_score_em_batch`), completely unchanged |
| everything else | identical to GAP/E1-B: lr 1e-6, `train_bs` 32, `gen_bs` 160, n=8, GRPO with `norm_adv_by_std_in_grpo`, DAPO clip 0.2/0.28, 2 epochs, temperature 1.0, 8 turns, same tool config, save/test every 10 steps, 4 GPUs, full-state resume |

Phases: **D1** original GAP `global_step_60` → E1-E `global_step_70`; then the automatic
section-15 gate; then **D2** E1-E's *own* `global_step_70` → `global_step_120`
(checkpoints 80/90/100/110/120).

## Implementation strategy (no original source modified)

The original GAP/verl core is read-only. E1-E injects everything through
`PYTHONPATH=E1-E/code/pythonpath_e1e` (a `sitecustomize.py` shim) which:

1. registers the reward manager `e1e_quota_gated` (`E1-E/code/e1e_reward_manager.py`) in the
   unmodified `verl` registry, and
2. loads the E1-0 root-caused runtime fixes (hang diagnostics, FSDP `load_checkpoint`
   `map_location="cpu"`, per-rank SGLang GPU visibility, passive worker prints).

The custom filter metric is produced by the custom scorer
(`E1-E/code/e1e_scorer.py:compute_score_em_efficiency_batch`) which publishes
`e1e_quota_metric` through the trainer's existing `reward_extra_info -> non_tensor_batch`
path, so `algorithm.filter_groups.metric=e1e_quota_metric` works with the stock
`ray_trainer.py`.  Because the trainer drops filtered-out groups before accumulating, the
reward manager also **materialises** the quota batch in place at the stopping generation batch
(`E1-E/code/e1e_quota_plan.py` is the single pure planner shared by the runtime and the offline
selftest).

## Layout

```
E1-E/
  E1-E_task.md              task specification (pre-registered)
  README.md                 this file
  FINAL_REPORT.md           31-section report (RQ1-RQ7, four-way comparison, verdict)
  code/                     runtime code (planner, scorer, reward manager, shims, selftest)
  scripts/                  launchers, preflight, integrity, gate, analysis, report helpers
  configs/                  resolved configs of the two phases
  logs/                     preflight / training / integrity / analysis logs, status files
  runs/                     per-phase run dirs (diagnostics, config, preflight records)
  diagnostics/              per-phase training diagnostics and checkpoint statistics
  analysis/                 selftest, integrity, gate, comparison, summary artifacts
  eval_results/             step70 / step120 full 7-benchmark evaluations
  manifests/                provenance manifests (pre/post original-file hashes, checkpoints)
  checkpoints/              checkpoint pointers/manifests (models stay in experiments/)
  figures/                  report figures
  legacy_e1d_unused/        parked E1-D copies that must not be used (see its README)
```

## Reproduce

```bash
# 0. offline gate (CPU only, no GPU): planner + runtime replay vs E1-E0, 12 mandated invariants
bash E1-E/scripts/e1e_preflight.sh
PYTHONPATH=$PWD/verl:$PWD/E1-E/code/pythonpath_e1e python3 E1-E/code/e1e_runtime_selftest.py

# 1. everything, unattended (preflight -> selftest -> D1 -> integrity -> eval70 -> gate ->
#    D2 -> integrity -> eval120 -> analysis)
bash E1-E/scripts/run_e1e_all.sh all

# or stage by stage
bash E1-E/scripts/run_e1e_all.sh d1
bash E1-E/scripts/run_e1e_all.sh eval70
bash E1-E/scripts/run_e1e_all.sh gate
bash E1-E/scripts/run_e1e_all.sh d2
bash E1-E/scripts/run_e1e_all.sh eval120
bash E1-E/scripts/run_e1e_all.sh analysis
```

Model checkpoints live in
`experiments/DAPO-GAP3B-MHQA-Agent-E1E-q2-step60to120-4gpu/` (recorded in `E1-E/manifests/`
and `E1-E/checkpoints/`); nothing is copied into `E1-E/`.

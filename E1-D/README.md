# E1-D — Filter-Decoupled Reward (controlled causal/mechanism ablation)

E1-D trains with the **E1-B shaped reward** but the **original GAP filter**:

```
filter : algorithm.filter_groups.metric = em        (E1-B used seq_final_reward)
reward : R_i = A_i * (1 + 0.05 * E_i),  A_i = original EM, cost = logical_search_batches
```

Start: original GAP `global_step_60` → D1 to step70 → D2 to step120.
Evaluation: the original 7-benchmark protocol (51,201 samples, greedy) at step70 and step120.

## Status — COMPLETE (2026-09-13 18:49)

| | |
|---|---|
| D1 (step60→70) | exit 0, integrity **PASS**, `global_step_70` complete |
| D2 (step70→120) | exit 0, integrity **PASS**, checkpoints 80/90/100/110/120 complete |
| step70 eval | `E1-D/eval_results/step70_20260913_024024/` — macro EM 0.3976, micro EM 0.4333 |
| step120 eval | `E1-D/eval_results/step120_20260913_165834/` — macro EM 0.4059, micro EM 0.4386 |
| filter decoupling | 100 % confirmed (0 mismatches; 0 all-correct groups retained) |
| verdict | **CASE A** |

## Headline (step120, paired over 49,947 prompts)

| comparison | search rounds | micro EM |
|---|---|---|
| **E1-D120 vs GAP120** (objective channel only) | 1.857 → 1.723 (**−6.99 %**, p ≈ 1e-233) | 0.4450 → 0.4392 (**−0.57 pp**, p = 2.8e-4) |
| **E1-D120 vs E1-B120** (filter channel only) | 1.297 → 1.727 (**+33.13 %**, p ≈ 0) | 0.4332 → 0.4392 (**+0.61 pp**, p = 2.5e-4) |

E1-B's total effect vs GAP120 decomposes as **rounds −30.1 % = −23.2 % objective + −76.8 % filter**
and **EM −1.18 pp = −0.57 pp objective + −0.61 pp filter**.  This is **CASE A**: with the original
filter the accuracy returns to ≈GAP while the efficiency gain largely disappears; the
filter-induced distribution shift was the dominant driver.

## Layout

```
E1-D/
├── FINAL_REPORT.md          # 32 sections
├── README.md
├── code/                    # e1d_*.py (E1-B copies, renamed) + pythonpath_e1d shim
├── scripts/                 # preflight, train (D1/D2), eval (70/120), integrity, analysis, pipeline
├── configs/                 # resolved configs for both phases, reward/filter config
├── logs/                    # train logs, wiki_service.log, preflight.log, issues_and_fixes.md
├── diagnostics/             # per-phase training diagnostics (checkpoint stats)
├── analysis/                # integrity, selftest, comparisons, paired tests, summary.json
├── eval_results/            # step70 / step120 evaluations + flat CSV/JSON artifacts
├── manifests/               # source + original-file pre/post manifests, checkpoint manifest
├── checkpoints/             # per-checkpoint manifests/stats (pointers; weights stay in experiments/)
└── runs/                    # run_<ts> dirs with stdout/diagnostics/preflight per phase
```

## Reproduce

See `FINAL_REPORT.md` §30; the one-shot entry point is
`bash E1-D/scripts/run_e1d_all.sh all`.

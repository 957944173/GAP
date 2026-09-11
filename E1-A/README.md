# E1-A — Success-conditioned group-relative efficiency reward (step60 → step70)

**Goal** (E1-A_task.md): verify, with minimal code change, whether
`R_i = A_i * (1 + 0.05 * E_i)` (cost = `logical_search_batches`) can **reduce sequential
search depth while preserving accuracy**.

**Result snapshot** (details in `FINAL_REPORT.md`):
training 60→70 completed (109 generation batches, checkpoint step70, integrity PASS).
Against the matched baseline (same 10 steps, original reward) the shaped reward
**reduced sequential search depth on 7/7 benchmarks** (macro rounds −1.80%, paired
per-prompt −2.04%, sign test p≈3e-25) while **accuracy stayed statistically unchanged**
(macro EM −0.22 pp, paired −0.08 pp, p=0.56). No reward hacking introduced
(packing-suspect 3.45% vs 4.21%; zero-search-and-correct = 0).
Verdict: **GO to E1-B (step70 → step120)** with the success/stop criteria in §7.

## Layout

```
E1-A/
  E1-A_task.md                  task definition (unchanged)
  README.md                     this file
  issues.log                    problems found + how they were handled
  FINAL_REPORT.md               final report (7 required sections + E1-B recommendation)
  latest_run.txt                newest training run dir
  run_e1a_train.sh              training entry      (step60 -> step70, shaped reward)
  run_e1a_eval_step70.sh        evaluation entry    (7 benchmarks, E1-A step70)
  code/                         E1-A reward implementation + runtime fixes
    e1a_scorer.py               compute_score_em_efficiency_batch()  (R = A*(1+0.05E))
    e1a_cost.py                 logical_search_batches extraction from rollout metadata
    e1a_reward_manager.py       @register("e1a_batch_shaped"): shaped reward,
                                acc-vs-reward separation, diagnostics
    e1a_selftest.py             CPU-only self-test (PASS)
    pythonpath_e1a/sitecustomize.py   registry shim + runtime-fix loader
    e1a_{state,trainer_patch,diag_patch,checkpoint_patch,sglang_gpu_patch,
         hang_diagnostics,stall_watchdog}.py   proven runtime fixes (copies from E1-0)
  scripts/
    e1a_preflight.sh                     wiki + env + input preflight
    e1a_eval_common.sh                   shared eval helper (reuses the ORIGINAL eval code)
    e1a_eval_baseline_step70.sh          matched-baseline eval (E1-0 step70)
    e1a_analyze_eval.py                  7-benchmark aggregation (5 required JSONs)
    e1a_analyze_training.py              training diagnostics + baseline comparison
    e1a_compare_eval.py                  E1-A vs baseline comparison table
    e1a_check_training_integrity.py      post-training integrity (checkpoint/steps/errors)
    e1a_extract_config.py                resolved config extraction from the log
  run_<timestamp>/               training run dir: stdout.log, stderr.log, config/,
                                 preflight/, diagnostics/, analysis/, logs/, checkpoints_manifest/
  eval_results/
    <timestamp>/                 E1-A step70 eval: accuracy.json, benchmark_results.json,
                                 trajectory_metrics.json, search_round_statistics.json,
                                 parallel_statistics.json, eval.log
    baseline_<timestamp>/        matched baseline (E1-0 step70) eval, same artifacts
    step60_reference/            GAP step60 (starting point) eval, same artifacts
    comparison_vs_step60/        E1-A vs step60 comparison
    comparison_vs_baseline/      E1-A vs matched baseline comparison
  preflight/                     wiki_service_status.log/json, manifests, code manifest
```

## What was changed (the ONLY intended change: the reward)

Nothing in the original GAP tree is modified (verified by sha256 before/after:
`preflight/original_files_manifest.txt` + `preflight/original_files_reverify_after_e1a.txt`,
19/19 files unchanged). The reward is replaced through the same mechanism the original
script already uses to select a reward manager:

| | original | E1-A |
|---|---|---|
| `reward_model.reward_manager` | `batch` | `e1a_batch_shaped` (subclass of `batch`) |
| train scorer | `mhqa_train.py:compute_score_em_batch` | `e1a_scorer.py:compute_score_em_efficiency_batch` |
| val scorer | `mhqa_eval.py:compute_score_em_batch` | **unchanged** |
| `data.batch["acc"]` | scorer score | **original EM** (shaped reward excluded, per task) |
| everything else | — | byte-identical (model, optimizer, scheduler, dataset, GRPO/DAPO trainer, rollout n=8, tool config, wiki env, eval protocol) |

## Reproduce

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate parallel-agent && source environment.sh
bash E1-A/scripts/e1a_preflight.sh          # wiki (retriever) + env + inputs
bash E1-A/run_e1a_train.sh                  # ~2h40m on 4 GPUs, step60 -> step70
bash E1-A/run_e1a_eval_step70.sh            # ~1h40m, 7 benchmarks on the E1-A step70
bash E1-A/scripts/e1a_eval_baseline_step70.sh   # matched baseline (E1-0 step70), same protocol
python3 E1-A/scripts/e1a_analyze_training.py --run-dir "$(cat E1-A/latest_run.txt)" \
        --out "$(cat E1-A/latest_run.txt)/analysis"
python3 E1-A/scripts/e1a_compare_eval.py --baseline E1-A/eval_results/baseline_<ts> \
        --e1a E1-A/eval_results/<ts> --out E1-A/eval_results/comparison_vs_baseline
```

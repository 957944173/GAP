# E1-B — reward scaling: step70 → step120

E1-B continues the GAP / E1-A reinforcement-learning run from the **E1-A step70
checkpoint** to **step120** with the *only* changed variable being the reward:

```
R_i = A_i * (1 + 0.05 * E_i)          A_i = original EM (unchanged)
                                      E_i = success-conditioned group-relative efficiency bonus
                                      cost = logical_search_batches
```

Everything else (GRPO, advantage estimator, optimiser, scheduler, RL dataset, SFT model,
rollout configuration, wiki environment, evaluation protocol) is byte-identical to the
E1-A / original GAP configuration.

## Status — COMPLETE (2026-09-12)

| | |
|---|---|
| training | step70 → step120, 50 steps, 11:34 → 23:50 (12:00:46), exit 0, integrity PASS |
| run dir | `E1-B/run_20260911_113432` |
| checkpoints | `global_step_{80,90,100,110,120}` (all complete) in `experiments/DAPO-GAP3B-MHQA-Agent-E1B-step70to120-4gpu/` |
| evaluations | step120, 80, 90, 100, 110 (original 7-benchmark protocol, 51,201 samples each) |
| report | `E1-B/FINAL_REPORT.md` (12 sections, results rendered from the run's artifacts) |

Key results (see `FINAL_REPORT.md` §8–§12 for the full tables):

* **vs the E1-B start (E1-A step70)**: search rounds **−18.95 %**, queries **−10.88 %**,
  paired per-prompt EM **p = 0.829** (statistically unchanged), 7/7 benchmarks with fewer rounds;
* **vs the original GAP trajectory at step120**: rounds **−30.14 %**, queries **−19.99 %**,
  micro EM −1.16 pp (p = 2e-12) — essentially the accuracy gain the original reward earned
  between step60 and step120, traded for a 30 % depth reduction;
* **safety**: zero-search-and-correct = 0, metadata_invalid = 0, packing-suspect 8.25 %
  (< 10 % stop line), no OOM/NCCL/traceback/wiki failures over 641,280 rollouts.


## Directory layout

| path | content |
|---|---|
| `code_changes/` | the runtime reward implementation: `e1b_*.py`, verbatim copy of `E1-A/code/e1a_*.py` (identifier rename only), plus the `pythonpath_e1b/sitecustomize.py` registration shim |
| `scripts/` | preflight, training launcher, safety monitor, analyzers, evaluation launchers |
| `configs/` | resolved run config, reward config, launch command, original-file integrity manifests |
| `checkpoints/` | per-checkpoint statistics (`checkpoint_stats.{json,csv,md}`) and manifests |
| `logs/` | `wiki_service.log`, preflight log, evaluation orchestrator log, waiter/watchdog logs |
| `evaluations/` | step80/90/100/110/120 evaluations + re-aggregated references + comparison tables |
| `diagnostics/` | reward/efficiency/behaviour diagnostics emitted by the reward manager and the safety monitor |
| `run_<ts>/` | run-scoped: stdout/stderr, `diagnostics/*.jsonl`, `preflight/`, `config/`, `analysis/` |
| `FINAL_REPORT.md` | the deliverable report (12 required sections) |

## Run

```bash
# 1. preflight (wiki reuse/start, checkpoint + environment checks, integrity manifest)
bash E1-B/scripts/e1b_preflight.sh

# 2. prove the reward is an unchanged reuse of E1-A
python3 E1-B/scripts/e1b_cross_check_vs_e1a.py

# 3. train step70 -> step120 (~15 h on 4xA800), diagnostics + safety monitor included
bash E1-B/scripts/run_e1b_train.sh

# 4. per-checkpoint statistics
python3 E1-B/scripts/e1b_checkpoint_stats.py --run-dir "$(cat E1-B/latest_run.txt)"

# 5. training-side analysis (vs E1-0 original-reward baseline and vs E1-A)
python3 E1-B/scripts/e1b_analyze_training.py --run-dir "$(cat E1-B/latest_run.txt)" \
    --out "$(cat E1-B/latest_run.txt)/analysis"

# 6. evaluation: step120 first, then 80/90/100/110, then artifact collection + comparison
bash E1-B/scripts/e1b_eval_all_steps.sh 120 80 90 100 110

# 7. integrity re-verification (original GAP files unchanged)
bash E1-B/scripts/e1b_verify_original_unchanged.sh
```

## Invariants

* **No original GAP file is modified** — the reward manager is registered at runtime
  through a `PYTHONPATH` `sitecustomize` shim; sha256 manifests before/after the run are in
  `configs/`.
* **No E1-0 / E1-A artifact is overwritten** — E1-B only reads them (the E1-A reward source
  and the E1-0/E1-A audit files are used as references).
* **`acc` (real EM) and `reward` (shaped) stay separated**: the manager writes the shaped
  reward into the reward tensor and restores `data.batch["acc"]` to the original EM, so no
  accuracy metric can be contaminated by the shaping.
* **Validation is untouched**: the validation scorer remains
  `mhqa_eval.py:compute_score_em_batch`.

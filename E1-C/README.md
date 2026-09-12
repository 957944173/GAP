# E1-C — filter-distribution diagnosis of the E1 efficiency reward

Offline (no training) diagnosis of the mechanism behind the E1-B accuracy gap.  See
`FINAL_REPORT.md` for the full analysis and `data_inventory.md` for provenance.

## Headline numbers

| question | answer |
|---|---|
| same-pool set relationship | `S_GAP` is a **strict subset** of `S_E1` on all three pools (E1-0: 340⊂424, E1-A: 294⊂349, E1-B: 1412⊂1685); GAP-only = **0** |
| what is added | **273/273 = 100 %** are all-correct, efficiency-varying groups (H2 supported) |
| optimizer-batch level | **partial overlap**, not containment: on the uncensored E1-0 pool E1 fills 32 groups in **94** generation batches vs GAP's **115** (−18.3 %), replacing **62/320 (19.4 %)** of the optimizer slots |
| who is displaced | E1-added groups: success rate 1.00 (k=8/8); GAP-displaced: success rate 0.467 (k≈3.7/8), Mann-Whitney p = 7.4e-73 |
| λ | membership is a **step function of λ**: λ=0 → 1412 groups, every λ>0 → the same 1685 groups (H5 supported) |
| advantage | on a fixed batch the efficiency gradient is **negative** under λ=0 (−0.626) and **positive and ≈λ-invariant** for λ>0 (+0.211 … +0.219) |
| hypothesis | H1–H5 SUPPORTED, H6 **SUPPORTED BY ASSOCIATION** (not causal) |
| recommendation | **Candidate A (filter-decoupled reward)**, with Candidate B as fallback |

## Layout

```
E1-C/
├── FINAL_REPORT.md          # 20 sections + answers A-E
├── data_inventory.md        # source files, fields, provenance hashes, limits
├── README.md
├── code/                    # e1c_common.py, e1c_run_all.py, e1c_figures.py, e1c_docs.py
├── configs/                 # analysis_config.json, source_provenance.json
├── logs/                    # run.log, run_all_stderr.log, errors_and_fixes.md
├── results/                 # every CSV/JSON listed in FINAL_REPORT.md §20
└── figures/                 # 5 PNGs
```

## Reproduce

```bash
# 1. main analyses A-G (CPU only, ~7 min, reads ~750k rollout records)
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate parallel-agent
python3 E1-C/code/e1c_run_all.py

# 2. figures (parallel-agent has no matplotlib)
/home/nf5468m6/miniconda3/envs/huatuo/bin/python3 E1-C/code/e1c_figures.py

# 3. documentation
python3 E1-C/code/e1c_docs.py
```

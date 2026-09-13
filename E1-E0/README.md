# E1-E0 — offline quota replay for the E1-E efficiency-group mixture

Pure offline diagnosis on the **real E1-D D2 candidate stream**.  No training, no GPU, no
model/trainer/reward modification.

## Question

If the next experiment keeps GAP's correctness training intact but admits at most **q**
all-correct efficiency-variable groups per 32-group optimizer batch, what does each q do to
the batch composition — and which q is the right one to train with?

## Headline

| q | full-quota rate | mean efficiency groups / step | replacement fraction | efficiency supervision fraction |
|---|---|---|---|---|
| 1 | 1.00 | 1.00 | 0.0312 | 0.0312 |
| 2 | 1.00 | 2.00 | 0.0625 | 0.0625 |
| 3 | 1.00 | 3.00 | 0.0938 | 0.0938 |
| 4 | 0.94 | 3.94 | 0.1231 | 0.1231 |
| unrestricted (reference) | — | 8.06 | 0.2519 | 0.2519 |

**Recommended q = 2** — see `FINAL_REPORT.md` §17/§18.

## Layout

```
E1-E0/
├── FINAL_REPORT.md      # 23 sections + answers A-H
├── data_inventory.md
├── README.md
├── code/                # e1e_common.py, e1e_quota_replay.py, e1e_figures.py, e1e_docs.py
├── configs/             # analysis_config.json, source_provenance.json
├── logs/                # run.log, errors_and_fixes.md
├── results/             # every CSV/JSON listed in FINAL_REPORT.md §23
└── figures/             # 5 PNGs
```

## Reproduce

```bash
cd /data01/wyy/Graph-Agent-Planning
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate parallel-agent
python3 E1-E0/code/e1e_quota_replay.py          # all analyses (~4 min, CPU only)
/home/nf5468m6/miniconda3/envs/huatuo/bin/python3 E1-E0/code/e1e_figures.py   # figures
python3 E1-E0/code/e1e_docs.py                  # inventory / config / error log / README
```

# E1-0 reward identity (original GAP reward vs E1-0 reward)

| metric | value |
|---|---|
| compared | 147200 |
| mismatch_count | 0 |
| max_abs_diff | 0.0 |
| equality_rate | 1.0 |
| equality_rate_pct | 100.0 |
| score_eq_em_mismatch_count | 0 |
| answer_parse_missing | 360 |
| note | reward_e10 is the value actually written into reward_tensor (what GRPO consumes); reward_original is an independent re-computation with the unmodified original mhqa_train.compute_score_em_batch on the same decoded prompt/response/ground-truth. |

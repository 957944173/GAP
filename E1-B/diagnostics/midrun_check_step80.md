# E1-B mid-run check — first checkpoint (global_step_80)

Recorded while training continues (step83 in progress, 2026-09-11 14:4x).

## Checkpoint integrity (global_step_80, saved 14:04)

* `actor/model_world_size_4_rank_{0..3}.pt` — 3,397,345,082 B each ✓
* `actor/optim_world_size_4_rank_{0..3}.pt` — 6,171,911,046 B each ✓
* `actor/extra_state_world_size_4_rank_{0..3}.pt` — 14.6–14.8 KB each ✓
* tokenizer/config files present ✓
* `data.pt` — 1,492 B, `StatefulDataLoader` state keys
  `['_snapshot', '_steps_since_snapshot', '_iterator_finished']` ✓ (the data stream continues)
* tracker `latest_checkpointed_iteration.txt` = 80 ✓

## Training-side behaviour, steps 71–80 (cumulative, reward-manager diagnostics)

| metric | E1-B step71–80 | E1-A step61–70 (its own branch start) | delta |
|---|---|---|---|
| mean EM | 0.4583 | 0.4539 | **+0.44 pp** |
| mean reward (shaped) | 0.4584 | 0.4540 | +0.44 pp |
| mean efficiency bonus E | 0.0021 | 0.0018 | +0.0003 |
| active efficiency groups | 73 | 95 | (10-step window vs 10-step window) |
| retained groups (shaped) | 337 | 349 | −12 |
| mean logical_search_batches | 1.4214 | 1.4407 | **−1.34 %** |
| mean search queries | 1.9727 | 1.9898 | −0.86 % |
| parallel factor | 1.5144 | 1.5091 | +0.35 % |
| zero-search-correct | 0 | 0 | — |
| metadata_invalid | 0 | 0 | — |

Early read: continuing the shaped-reward training from step70 does **not** degrade accuracy;
the sequential search depth stays slightly below the E1-A step70 level while the
per-round parallel factor is unchanged. Full evaluation at step120 will be the verdict.

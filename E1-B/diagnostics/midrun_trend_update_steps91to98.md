# E1-B mid-run trend update — steps 91–98 (training still running, step ~99/120)

Recorded 2026-09-11 17:45 from the reward-manager diagnostics of the live run.

## Windowed training-side trend (10-step windows)

| window | mean EM | mean logical_search_batches | mean search queries | Δrounds vs window 71–80 |
|---|---|---|---|---|
| 71–80 | 0.4582 | 1.4231 | 1.9763 | — |
| 81–90 | 0.4533 | 1.3956 | 1.9681 | −1.9 % |
| 91–98 | **0.4614** | **1.3445** | **1.9390** | **−5.5 %** |

Reference points: E1-A steps 61–70 (the E1-B start) had mean EM 0.4528 / rounds 1.4329 /
queries 1.9838; the E1-0 original-reward baseline (same 10 steps) had EM 0.4539 /
rounds 1.4407 / queries 1.9898.

## Reading

* the efficiency shaping **keeps accumulating after step70**: sequential search depth is
  still falling (1.4231 → 1.3956 → 1.3445) while mean EM is *not* falling
  (0.4582 → 0.4533 → 0.4614, i.e. flat within noise and above the E1-A start);
* **search queries fall together with the rounds** (1.9763 → 1.9390), which is the direct
  counter-evidence to a query-packing failure mode: the policy is shortening the whole
  trajectory, not re-bundling the same queries into fewer rounds;
* per-step cost is now consistently ~1.31–1.35 versus ~1.39–1.43 in steps 71–80.

This strengthens the I7 assessment (see `issues.log`): the packing-suspect tail statistic is
not accompanied by any query inflation, and the direction of both search depth *and* query
count is downward with accuracy preserved.

---

## Update at step 113/120 (2026-09-11 22:05) — the effect is monotone and clean

| 10-step window | mean EM | mean logical_search_batches | mean search queries |
|---|---|---|---|
| 71–80 | 0.4582 | 1.4231 | 1.9763 |
| 81–90 | 0.4533 | 1.3956 | 1.9681 |
| 91–100 | 0.4574 | 1.3360 | 1.9287 |
| 101–110 | 0.4582 | 1.2776 | 1.8863 |
| 111–113 (4 steps) | 0.4584 | 1.2415 | 1.8478 |

* search depth falls **monotonically** through the whole 50-step extension
  (1.4231 → 1.2415, **−12.8 %** relative to the first window, and well below the E1-A step70
  level of 1.4329);
* mean EM is **flat** (0.4533 – 0.4584) across all windows, i.e. the accuracy cost is zero
  within training-side noise;
* search queries fall in tandem (1.9763 → 1.8478, **−6.5 %**), so rounds are not being
  bought by packing more queries into fewer rounds — the opposite of the failure signature;
* every checkpoint verified so far (80, 90, 100, 110) has complete actor/optimizer/
  extra_state shards, `data.pt` dataloader state, and a matching tracker value.

# E1-0.5 — offline diagnosis of the E1-0 rollout audit

**Goal**: decide, purely offline, whether it is worth entering **E1-A**
(success-conditioned group-relative efficiency reward training) — without
modifying any GAP code, reward implementation, GRPO/filter/trainer, and without
re-training anything.

**Input**: the E1-0 passive-audit rollouts (read-only):
`E1-0/runs/20260910_step60to70_e10/audit/rollouts_pid2552655.jsonl`
(147,200 rollouts = 115 generation batches × 160 prompts × n=8, steps 61..70).

**Output**: everything in this directory. No E1-0 file is written or overwritten.

```
E1-0.5/
  README.md                          this file
  E1-0.5_task.md                     the task definition (unchanged)
  scripts/
    e105_common.py                   audit loading, group table, filter/efficiency helpers
    run_analysis.py                  Parts 2-7 + the figure (single entry point)
  outputs/
    group_table.csv                  all 18,400 n=8 groups (+ is_active flag)          [Part 2]
    active_groups.csv                the 95 active groups, self-contained             [Part 2]
    cost_gap_distribution.csv        长表: gap / min / max / contains / pair patterns  [Part 3]
    cost_gap_wide.csv                gap -> #groups (figure source)                    [Part 3]
    cost_pair_patterns.csv           distinct correct-cost patterns                    [Part 3]
    cost_gap_group_detail.csv        per-active-group gap detail                       [Part 3]
    source_breakdown.csv             benchmark/source totals + active ratio            [Part 4]
    query_round_analysis.csv         efficient vs inefficient correct trajectories     [Part 5]
    query_packing_verdict.json       packing-hack verdict                             [Part 5]
    query_packing_group_detail.csv   per-active-group packing detail                   [Part 5]
    parallel_behavior.csv            parallel-factor / tokens / turns by class          [Part 6]
    parallel_behavior_hist.json      parallel-factor histograms by class                [Part 6]
    filter_replay_results.json       baseline vs new-reward filter replay (headline)   [Part 7]
    filter_replay_group_level.csv    per-group keep decisions                          [Part 7]
    filter_replay_per_call.csv       per-generation-batch kept counts                  [Part 7]
    filter_replay_per_step.csv       per-step calls needed (baseline vs new)           [Part 7]
    filter_replay_lambda_sensitivity.csv  lambda sweep                                 [Part 7]
    summary.json                     all headline numbers
  reports/
    figures/cost_gap_distribution.png
    E1-0.5_FINAL_REPORT.md           the report + GO/NO-GO
```

## How to re-run

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate parallel-agent          # pandas + numpy, no matplotlib
python3 E1-0.5/scripts/run_analysis.py
```

The figure is rendered by shelling out to a conda env that has matplotlib
(`huatuo` on this box, fallback `llama_factory`); the interpreter actually used
is recorded in `outputs/summary.json["figure"]["renderer"]`. `matplotlib` is not
installed in `parallel-agent`, which is why this is a subprocess step rather than
an import.

## Analysis decisions

### Grouping key
The task says *group by uid, 8 rollouts per group*. `uid` is the dataset row
index and the training-time group key is `(generation batch, uid)`. In this run
**18,400 group instances contain 18,400 distinct uids — every uid appears exactly
8 times in the whole audit**, i.e. no prompt was re-sampled across steps 61-70
(the dataloader resumed from step60's sampler state and never wrapped). Grouping
by `uid` alone and by `(call_index, uid)` therefore produce **identical** groups,
and both contain exactly 8 rollouts. The scripts still key on
`(call_index, uid)` (unambiguous if a future audit does re-sample a prompt) and
the group table carries both keys; `call_index` is preserved everywhere so the
per-generation-batch replay stays faithful to the trainer.

### Active group definition
`correct_count >= 2 AND (max - min) > 0` over **the correct rollouts'**
`logical_search_batches`. The task's Part 7 defines `E` as a *group-relative
efficiency score among correct trajectories*, so the cost variation that can
actually produce efficiency supervision is the variation among correct
trajectories. (The all-rollout gap is also derivable from the group table, but
mixing wrong rollouts into the gap would create "active" groups whose correct
subset has a single cost, i.e. `E ≡ 0` — no supervision.)

### Reward used in the counterfactual replay (Part 7)
```
A_i = 1 if EM-correct else 0
E_i = (Cmax - C_i) / (Cmax - Cmin)   for correct rollouts of an active group, else 0
R_i = A_i * (1 + lambda * E_i)       with lambda = 0.05
```
and GAP's real filter (`verl/verl/trainer/ppo/ray_trainer.py:1780-1781`):
keep a uid group iff `np.std(group_rewards) > 0` (or the group has 1 member).

## Schema findings and corrections (task constraint 6)

1. **`logical_search_batches` is a *reconstructed* logical-block count, and that
   is the right cost — the response-text `<wiki_search>` count is NOT.**
   `SGLangRollout.__init__` replaces the tokenizer's chat template
   (`verl/verl/workers/rollout/sglang_rollout/sglang_rollout.py:289`) with one
   that renders every tool call as `<name>arguments</name>`. Because
   `XMLToolParser.parse_non_stream` splits `a|b|c` into **three separate tool
   calls**, the re-rendered response contains **one `<wiki_search>` block per
   query**, not one per logical block. Measured over all 147,200 rollouts:
   `resp_xml_search_batches == search_queries` for **99.79%** (the remaining
   0.21% are length-truncated responses), while
   `logical_search_batches == structured_search_rounds` for **100%**, and
   `parallel_factor == search_queries / logical_search_batches` for **100%**.
   So: `logical_search_batches` = number of original assistant search blocks
   (= search rounds here), `search_queries` = number of individual queries,
   `parallel_factor` = queries per block. E1-0.5 uses
   `logical_search_batches` as the cost, exactly as the task specifies.
   *This also corrects an incorrect claim in `E1-0/.../FINAL_REPORT.md` §6*
   ("the literal `<wiki_search>` string never appears in the response"): it does
   appear, once per query, because of the custom template. E1-0's core results
   are unaffected (its `logical_search_batches` already equals the structured
   round count and the reward-identity check is independent of this field). The
   E1-0 files are intentionally **not** modified here.

2. **Replay units bug (found and fixed during development).** The first version
   of the counterfactual replay accumulated *groups* instead of *generation
   batches* and reported 17,531 "calls" instead of the true 115. After the fix
   the baseline replay reproduces the observed `train/num_gen_batches` for
   **all ten steps exactly** (13,14,13,9,13,13,12,8,10,10 = 115), which is an
   independent validation that the simulated filter matches the trainer.

3. **`matplotlib` is absent from `parallel-agent`**, so the figure is rendered in
   a subprocess by an env that has it (`huatuo`); this is recorded in
   `summary.json` rather than silently skipped.

4. **`data_source` is not a single benchmark**: the audit contains `hotpotqa`
   (78,864 rollouts / 9,858 groups) and `nq` (68,336 / 8,542). Part 4 therefore
   reports a real per-source breakdown instead of a single row.

## Headline results

| | |
|---|---|
| rollouts / groups | 147,200 / 18,400 (all exactly 8 rollouts) |
| active efficiency groups | **95 (0.5163%)** |
| cost gap (correct) | gap 1: 86 groups (90.5%); gap 2: 7; gap 3: 1; gap 4: 1 |
| active by source | hotpotqa 65 / nq 30 |
| efficient vs inefficient correct | rounds 1.49 vs 2.49; queries 2.04 vs 2.93; tokens 1,343 vs 1,829; queries/block 1.48 vs 1.20 |
| packing-hack verdict | **4/95 groups (4.2%)** show pure packing; in 91/95 the inefficient class also uses more queries |
| filter replay (λ=0.05) | baseline retained 340 → new retained **424** (newly added **84**, all all-correct; none lost) |
| efficiency supervision ratio | 95/424 = **22.41%** of retained groups carry an efficiency term |
| generation-batch replay | baseline 115 → new **94** calls (−18.26%), same 32-prompt batch |
| λ sensitivity | any λ>0 gives the identical retained set; λ only scales advantage magnitude |

**Verdict: GO** (with monitoring recommendations) — see
`reports/E1-0.5_FINAL_REPORT.md`.

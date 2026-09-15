# E1-E online selection-composition check (RQ7)

- optimizer steps: 60
- inserted efficiency groups: 117 of 465 seen (348 seen but not inserted)
- kept mixed groups: 1803
- actual replacement rate: 0.0609
- every inserted group all-correct (k = 8): True
- inserted cost variation: mean 1.1367521367521367 (min 1.0), present for 117/117 inserted groups
- mean insertion position: generation batch 10.95 (the stopping batch, by construction)

## source composition

- recorded sources for inserted groups: {'hotpotqa': 1, 'nq': 1} (2 of 117; the rest were injected from the cache, where the manager does not carry `source`)
- sources of the kept mixed groups: {'hotpotqa': 1009, 'nq': 794}

## what cannot be checked online (instrumentation gap)

- data source of cached/injected efficiency groups: rows are written from a synthetic group dict with source=None and rollouts_pid has no source column, so the offline inserted-vs-displaced source test (E1-E0 p = 0.920) cannot be reproduced online
- difficulty (k/8) of the displaced mixed groups: they are dropped before any record is written

## available bias evidence

- selection is pure stream order on a shuffled dataset and the planner only ever takes cache[:m]; it never ranks groups by source, difficulty or reward
- every inserted group is k = 8 (all-correct) and cost-varying by construction, so the mechanism cannot prefer easy or hard *mixed* groups
- E1-E0's offline test on E1-D's stream found no difficulty bias (p = 0.943) and no source bias (p = 0.920); the online run's own stream differs, so this remains a transferred (not re-measured) result


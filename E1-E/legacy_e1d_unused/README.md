# Legacy E1-D self-test copies — NOT part of E1-E

These two files were copied from `E1-D/code/` during the E1-E scaffold step (identifiers renamed
`e1d_ → e1e_`) and are **not used by E1-E**. They are parked here instead of deleted so the
audit trail of the scaffold is complete (E1-D originals remain untouched in `E1-D/code/`).

Why they cannot be used for E1-E:

* `e1e_selftest.py` and `e1e_filter_selftest.py` test the **E1-D reward semantics**
  (`R_i = A_i · (1 + 0.05·E_i)` for *every* retained group, i.e. mixed-group shaping) and the
  E1-D filter (`algorithm.filter_groups.metric=em` with a shaped-reward revival channel).
* E1-E's pre-registered semantics (E1-E_task.md §1, frozen by E1-E0) are different:
  mixed groups (`1 ≤ k ≤ 7`) get `R_i = A_i` with **no shaping**, only all-correct efficiency
  groups (`k = 8` with correct-cost variation) get `R_i = 1 + 0.05·E_i`, they enter the batch
  only through the q = 2 quota cache, and the runtime filter metric is `e1e_quota_metric`.

Running them would therefore both mis-assert E1-E semantics and overwrite the real
`E1-E/analysis/preflight_selftest.{md,json}` produced by the actual E1-E gate.

The E1-E §11 preflight gate is `E1-E/code/e1e_runtime_selftest.py` (planner replay vs E1-E0 for
q=0 and q=2, runtime `E1EPQuotaRewardManager` replay through the trainer's own accumulate/
truncate loop, and the 13 named invariants mapped 1:1 onto the 12 §11 requirements).

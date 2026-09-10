# E1-0 cleanup record — 2026-09-11

Scope requested by the user: **remove only category A** ("zero-information-loss"
items: byte-identical duplicates, empty/failed diagnostics, bytecode caches).
Category B (failed-attempt history logs, `*_diagnosis.md`, `stackdumps/`) was
**deliberately kept**. No final artifact, audit record, analysis output or
report was deleted or modified (except one stale filename in
`FINAL_REPORT.md` §14, corrected to match).

Before delete: 295 MB / 646 files. After delete: 272 MB / 618 files.

## Deleted (each verified as redundant first)

| path | size | reason |
|---|---|---|
| `audit/groups_pid2552655.jsonl` | 22,720,574 B | md5-identical to the kept `audit/groups.jsonl` |
| `logs/launcher_attempt19b.log` | 848,853 B | md5-identical to the kept `logs/launcher.log` |
| `logs/launcher_attempt19b_success.log` | 848,853 B | md5-identical to the kept `logs/launcher.log` |
| `logs/launcher_attempt19a_hung_dataloader_fork.stdout.log` | 66,322 B | md5-identical to the kept `logs/launcher_attempt19.log` |
| `logs/launcher_attempt18_live_log.bak` | 58,948 B | stale pre-19b live-log snapshot, superseded by `logs/launcher_attempt18.log` |
| `logs/archive_attempt16/` (9 files) | 261 B total | each file is 29 bytes: `sudo: a password is required` (py-spy/ptrace unavailable, no information) |
| `logs/watchdog_stdout.log` | 842 B | only the watchdog banner; the real log is `logs/watchdog.log` |
| `code/__pycache__/` (8 `.pyc`) + `code/pythonpath_e10/__pycache__/` (1 `.pyc`) | 84 KB | Python bytecode cache |

md5s used to prove duplication (against the kept copy):
`d3c4da7a695ca10ef04ae6034193df76` (launcher*), `35ee823bf20925ac6a77d968546cfdc6`
(19a stdout), `8c89dda5b5275d6fbb72b01a1807c773` (groups).

## Kept and still canonical

* `audit/rollouts_pid2552655.jsonl` (147,200 records), `audit/groups.jsonl`
  (18,400 groups), `audit/calls_pid2552655.jsonl` (115),
  `audit/audit_summary.{json,csv}`, `audit/mismatch_cases.jsonl` (empty: 0 mismatches)
* `logs/training.log`, `logs/launcher.log`
* `logs/attempt19a_driver_and_dataloader_worker_SIGUSR1_stacks.txt` (root-cause evidence)
* all `code/*.py`, `analysis/*`, `preflight/`, `configs/`, `environment/`,
  `wiki/`, `checkpoints_manifest/`, `patches/`, both `FINAL_REPORT.md`,
  `Current_Progress.md`, `latest_run.txt`
* category B history: `logs/launcher_attempt1..19a*.log`, `logs/*_diagnosis.md`,
  `logs/checkpoint_load_hang_investigation.md`, `logs/stackdumps/` (535 files),
  `logs/attempt19a_ps_snapshot_at_stall.txt`, `logs/watchdog*.log`

## Note for external review (e.g. GPT)

The folder is still ~272 MB, of which ~266 MB is raw audit JSONL. A chat model
cannot ingest `audit/rollouts_pid2552655.jsonl` (240 MB) or
`audit/groups.jsonl` (21 MB) directly. For a text-only review, use:
`FINAL_REPORT.md` (both copies), `analysis/` (all `.md`/`.json`),
`audit/audit_summary.json`, `preflight/preflight_report.md`,
`checkpoints_manifest/*`, `configs/*`, and `Current_Progress.md` — about 1 MB
in total.

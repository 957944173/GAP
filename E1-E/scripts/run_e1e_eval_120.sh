#!/usr/bin/env bash
# E1-E: full 7-benchmark evaluation of E1-E step120.
set -Eeuo pipefail
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_e1e_eval_step.sh" 120

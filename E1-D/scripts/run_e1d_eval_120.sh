#!/usr/bin/env bash
# E1-D: full 7-benchmark evaluation of E1-D step120.
set -Eeuo pipefail
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_e1d_eval_step.sh" 120

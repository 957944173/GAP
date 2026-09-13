#!/usr/bin/env bash
# E1-D phase D2: E1-D's own step70 -> step120 (full-state resume inside the same experiment dir).
set -Eeuo pipefail
export E1D_PHASE=D2
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_e1d_train.sh"

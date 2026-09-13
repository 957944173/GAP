#!/usr/bin/env bash
# E1-D phase D1: original GAP step60 -> E1-D step70.
set -Eeuo pipefail
export E1D_PHASE=D1
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_e1d_train.sh"

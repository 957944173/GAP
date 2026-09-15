#!/usr/bin/env bash
# E1-E phase D1: original GAP step60 -> E1-E step70.
set -Eeuo pipefail
export E1E_PHASE=D1
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_e1e_train.sh"

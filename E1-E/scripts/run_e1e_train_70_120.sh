#!/usr/bin/env bash
# E1-E phase D2: E1-E's own step70 -> step120 (full-state resume inside the same experiment dir).
set -Eeuo pipefail
export E1E_PHASE=D2
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_e1e_train.sh"

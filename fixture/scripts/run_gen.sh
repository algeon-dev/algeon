#!/usr/bin/env bash
# Run a BloodHound graph fixture generator from this directory's local .venv.
#
# Usage:
#   ./run_gen.sh github                   # bh_sf1 by default
#   BH_GITHUB_SCALE=bh_sf1 ./run_gen.sh github
#
# Bootstraps `.venv/` with numpy + pyarrow on first run. The .venv directory
# is gitignored.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${HERE}/.venv"

if [[ ! -x "${VENV}/bin/python" ]]; then
  python3 -m venv "${VENV}"
  "${VENV}/bin/pip" install --quiet --upgrade pip
  "${VENV}/bin/pip" install --quiet 'numpy>=1.26' 'pyarrow>=15'
fi

case "${1:-}" in
  github)
    exec "${VENV}/bin/python" "${HERE}/gen_github_bh.py"
    ;;
  ad_entra)
    exec "${VENV}/bin/python" "${HERE}/gen_ad_entra_bh.py"
    ;;
  "")
    echo "usage: $0 <generator>   (available: github, ad_entra)" >&2
    exit 2
    ;;
  *)
    echo "unknown generator: $1" >&2
    exit 2
    ;;
esac

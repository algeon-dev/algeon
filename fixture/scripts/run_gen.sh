#!/usr/bin/env bash
# Run a local fixture generator/preparer.
#
# Usage:
#   ./run_gen.sh github                         # bh_sf1 by default
#   ./run_gen.sh cugraph --oracle               # copy cuGraph data + generate oracle JSON
#   BH_GITHUB_SCALE=bh_sf1 ./run_gen.sh github
#
# Bootstraps `.venv/` with numpy + pyarrow on first run. The .venv directory
# is gitignored.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
VENV="${HERE}/.venv"

rapids_python() {
  if [[ -n "${CUDF_CUGRAPH_PYTHON:-}" ]]; then
    echo "${CUDF_CUGRAPH_PYTHON}"
  elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    echo "${REPO_ROOT}/.venv/bin/python"
  else
    echo "python3"
  fi
}

case "${1:-}" in
  graph500 | graph500_rmat)
    exec "$(rapids_python)" "${HERE}/gen_graph500_rmat.py" "${@:2}"
    ;;
  cugraph)
    exec "$(rapids_python)" "${HERE}/prepare_cugraph.py" "${@:2}"
    ;;
esac

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
    echo "usage: $0 <generator>   (available: github, ad_entra, graph500_rmat, cugraph)" >&2
    exit 2
    ;;
  *)
    echo "unknown generator: $1" >&2
    exit 2
    ;;
esac

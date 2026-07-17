#!/usr/bin/env bash
# Prepare the local cuGraph parity fixture.
#
# Usage: fixture/graph/prepare.sh [prepare_cugraph.py options]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${HERE}" rev-parse --show-toplevel)"

rapids_python() {
  if [[ -n "${CUDF_CUGRAPH_PYTHON:-}" ]]; then
    echo "${CUDF_CUGRAPH_PYTHON}"
  elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    echo "${REPO_ROOT}/.venv/bin/python"
  else
    echo "python3"
  fi
}

exec "$(rapids_python)" "${HERE}/prepare_cugraph.py" "$@"

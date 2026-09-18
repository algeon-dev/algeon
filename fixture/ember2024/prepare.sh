#!/usr/bin/env bash
# Prepare the EMBER2024 forest-inference fixture (models, raw data, vectorization, Parquet export).
#
# Usage: fixture/ember2024/prepare.sh [prepare.py options]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

forest_python() {
  if [[ -n "${ALGEON_EMBER_PYTHON:-}" ]]; then
    echo "${ALGEON_EMBER_PYTHON}"
  elif [[ -x "${HOME}/.venvs/forest/bin/python" ]]; then
    echo "${HOME}/.venvs/forest/bin/python"
  else
    echo "python3"
  fi
}

exec "$(forest_python)" "${HERE}/prepare.py" "$@"

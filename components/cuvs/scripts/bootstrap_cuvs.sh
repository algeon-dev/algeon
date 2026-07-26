#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
git -C "${repo_root}" submodule sync -- cuvs
git -C "${repo_root}" submodule update --init --recursive cuvs
"${repo_root}/scripts/check_cuvs_source.sh"

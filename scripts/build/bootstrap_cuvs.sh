#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
git -C "${repo_root}" submodule sync -- components/cuvs/cuvs
git -C "${repo_root}" submodule update --init --recursive components/cuvs/cuvs
(cd "${repo_root}" && scripts/ci/check_native_sources.sh cuvs)

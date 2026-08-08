#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
git -C "${repo_root}" submodule sync -- cuvs
git -C "${repo_root}" submodule update --init --recursive cuvs
(cd "${repo_root}" && cargo xtask source check)

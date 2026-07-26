#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
list_file="$(mktemp)"
trap 'rm -f "${list_file}"' EXIT

export CUVS_ROOT="${CUVS_ROOT:-${repo_root}/cuvs-install}"
[[ -d "${CUVS_ROOT}" ]] || {
    echo "error: canonical cuVS prefix is missing: ${CUVS_ROOT}" >&2
    echo "Run scripts/build_libcuvs.sh or set CUVS_ROOT to a compatible installed prefix." >&2
    exit 1
}

cargo test \
    --manifest-path "${repo_root}/Cargo.toml" \
    -p cuvs-nexus-sys \
    --test cmake_root_contract \
    -- \
    --list >"${list_file}"

expected_tests=2
actual_tests="$(awk '/: test$/ { count += 1 } END { print count + 0 }' "${list_file}")"
if [[ "${actual_tests}" -ne "${expected_tests}" ]]; then
    echo "error: cmake_root_contract listed ${actual_tests} tests, expected ${expected_tests}" >&2
    cat "${list_file}" >&2
    exit 1
fi

cargo test \
    --manifest-path "${repo_root}/Cargo.toml" \
    -p cuvs-nexus-sys \
    --test cmake_root_contract

echo "CMake integration contract OK (${actual_tests} tests executed)"

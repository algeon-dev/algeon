#!/usr/bin/env bash
set -euo pipefail

mode="write"
if [[ "${1:-}" == "--check" ]]; then
    mode="check"
    shift
fi
[[ $# -eq 0 ]] || {
    echo "usage: $0 [--check]" >&2
    exit 2
}

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
"${repo_root}/scripts/check_cuvs_source.sh"

dlpack_include="${DLPACK_INCLUDE_DIR:-${repo_root}/cuvs/cpp/build/_deps/dlpack-src/include}"
[[ -f "${dlpack_include}/dlpack/dlpack.h" ]] || {
    echo "error: DLPack header is missing at ${dlpack_include}/dlpack/dlpack.h" >&2
    echo "Build the canonical cuVS prefix first or set DLPACK_INCLUDE_DIR." >&2
    exit 1
}

generated_file="$(mktemp)"
trap 'rm -f -- "${generated_file}"' EXIT
CUVS_SOURCE_DIR="${repo_root}/cuvs" \
DLPACK_INCLUDE_DIR="${dlpack_include}" \
    cargo run --manifest-path "${repo_root}/Cargo.toml" -p cuvs-nexus-bindings-generator --quiet -- "${generated_file}"
checked_in="${repo_root}/cuvs-nexus-sys/src/bindings.rs"
[[ -s "${generated_file}" ]] || {
    echo "error: bindings generator did not produce ${generated_file}" >&2
    exit 1
}

if [[ "${mode}" == "check" ]]; then
    cmp -s "${generated_file}" "${checked_in}" || {
        echo "error: checked-in bindings are stale: ${checked_in}" >&2
        diff -u "${checked_in}" "${generated_file}" | sed -n '1,200p' >&2 || true
        echo "Run: scripts/generate-bindings.sh" >&2
        exit 1
    }
    echo "checked-in bindings match pinned cuVS headers: ${checked_in}"
else
    cp "${generated_file}" "${checked_in}"
    echo "updated ${checked_in} from pinned cuVS ${repo_root}/cuvs"
fi

#!/usr/bin/env bash
set -euo pipefail

readonly expected_revision="2bd7cd71d31e39eb9ee00c2a250085a1afb84977"
readonly expected_url="https://github.com/rapidsai/cuvs.git"
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
source_dir="${repo_root}/cuvs"

fail() {
    echo "error: $*" >&2
    exit 1
}

[[ -f "${repo_root}/.gitmodules" ]] || fail ".gitmodules is missing"
[[ "$(git -C "${repo_root}" config -f .gitmodules --get submodule.cuvs.path || true)" == "cuvs" ]] || \
    fail ".gitmodules must define submodule.cuvs.path=cuvs"
[[ "$(git -C "${repo_root}" config -f .gitmodules --get submodule.cuvs.url || true)" == "${expected_url}" ]] || \
    fail ".gitmodules must define submodule.cuvs.url=${expected_url}"

index_entry="$(git -C "${repo_root}" ls-files --stage -- cuvs)"
read -r mode recorded_revision stage path <<<"${index_entry}"
[[ "${mode:-}" == "160000" && "${stage:-}" == "0" && "${path:-}" == "cuvs" ]] || \
    fail "cuvs must be a stage-0 gitlink in the superproject index, found: ${index_entry:-<missing>}"
[[ "${recorded_revision}" == "${expected_revision}" ]] || \
    fail "superproject index records cuvs ${recorded_revision}, expected ${expected_revision}"

head_entry="$(git -C "${repo_root}" ls-tree HEAD -- cuvs 2>/dev/null || true)"
read -r head_mode _head_type head_revision _head_path <<<"${head_entry}"
if [[ "${head_mode:-}" == "160000" ]]; then
    [[ "${head_revision}" == "${expected_revision}" ]] || \
        fail "superproject HEAD records cuvs ${head_revision}, expected ${expected_revision}"
fi

[[ -e "${source_dir}/.git" ]] || {
    echo "error: cuVS submodule is not initialized at '${source_dir}'." >&2
    echo "Run: git submodule update --init --recursive cuvs" >&2
    exit 1
}
[[ ! -L "${source_dir}" ]] || fail "cuvs source path must not be a symlink"

actual_revision="$(git -C "${source_dir}" rev-parse HEAD)"
[[ "${actual_revision}" == "${expected_revision}" ]] || {
    echo "error: cuVS submodule HEAD is ${actual_revision}; expected ${expected_revision}." >&2
    echo "Run: git submodule update --init --recursive cuvs" >&2
    exit 1
}

dirty="$(git -C "${source_dir}" status --porcelain=v1 --untracked-files=all)"
[[ -z "${dirty}" ]] || fail "cuVS source has build-input changes; reset or commit them before bootstrapping:
${dirty}"

echo "cuVS source contract OK: ${expected_revision}"

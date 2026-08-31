#!/usr/bin/env bash
set -euo pipefail

readonly repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly package_target="${repo_root}/target/component-package-check"
readonly packages=(
  rapids-interop
  cuvs-nexus-sys
  cuvs-nexus
)
cd "${repo_root}"
mkdir -p "${package_target}"

metadata=$(cargo metadata --format-version 1 --no-deps --locked --all-features)

add_patch() {
  local name=$1
  local path=$2
  patch_args+=(--config "patch.crates-io.${name}.path='${repo_root}/${path}'")
}

package_patches() {
  patch_args=()
  case "$1" in
    cuvs-nexus)
      add_patch cudf-nexus components/cudf/cudf-nexus
      add_patch cudf-nexus-sys components/cudf/cudf-nexus-sys
      add_patch cuvs-nexus-sys components/cuvs/cuvs-nexus-sys
      add_patch rapids-interop components/cudf/rapids-interop
      ;;
  esac
}

for package in "${packages[@]}"; do
  listing=$(cargo package --locked --allow-dirty -p "${package}" --all-features --list)
  grep -qx LICENSE <<<"${listing}" || {
    printf 'error: %s package list omits the root LICENSE\n' "${package}" >&2
    exit 1
  }

  package_patches "${package}"
  # Only components with an archive contract belong in this lane; workspace
  # clippy/nextest and the separate native-free phase own compilation.
  cargo package \
    --locked \
    --allow-dirty \
    --no-verify \
    -p "${package}" \
    --all-features \
    --target-dir "${package_target}" \
    "${patch_args[@]}"

  version=$(python3 -c \
    'import json,sys; package=sys.argv[1]; data=json.load(sys.stdin); print(next(item["version"] for item in data["packages"] if item["name"] == package))' \
    "${package}" <<<"${metadata}")
  archive="${package_target}/package/${package}-${version}.crate"
  [[ -f "${archive}" ]] || {
    printf 'error: cargo package did not create %s\n' "${archive}" >&2
    exit 1
  }
  archive_listing=$(tar -tzf "${archive}")
  grep -qx "${package}-${version}/LICENSE" <<<"${archive_listing}" || {
    printf 'error: %s archive omits LICENSE\n' "${package}" >&2
    exit 1
  }
done

printf 'component package archive contracts OK\n'

#!/usr/bin/env bash
set -euo pipefail

readonly repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly package_target="${repo_root}/target/component-package-check"
readonly extract_root="$(mktemp -d "${TMPDIR:-/tmp}/datafusion-nexus-component-packages.XXXXXX")"
trap 'rm -rf -- "${extract_root}"' EXIT
readonly packages=(
  rapids-interop
  nexus-build-support
  cudf-nexus-sys
  cugraph-nexus-sys
  cudf-nexus
  cugraph-nexus
  cuvs-nexus-sys
  cuvs-nexus
)
cd "${repo_root}"
mkdir -p "${package_target}"

metadata="$(cargo metadata --format-version 1 --no-deps --locked --all-features)"

package_version() {
  python3 -c \
    'import json,sys; package=sys.argv[1]; data=json.load(sys.stdin); print(next(item["version"] for item in data["packages"] if item["name"] == package))' \
    "$1" <<<"${metadata}"
}

add_patch() {
  local name=$1
  local path=$2
  patch_args+=(--config "patch.crates-io.${name}.path='${path}'")
}

package_patches() {
  patch_args=()
  case "$1" in
    cudf-nexus-sys | cugraph-nexus-sys | cuvs-nexus-sys)
      add_patch nexus-build-support "${repo_root}/components/cudf/nexus-build-support"
      ;;
    cudf-nexus)
      add_patch cudf-nexus-sys "${repo_root}/components/cudf/cudf-nexus-sys"
      add_patch rapids-interop "${repo_root}/components/cudf/rapids-interop"
      ;;
    cugraph-nexus)
      add_patch cudf-nexus "${repo_root}/components/cudf/cudf-nexus"
      add_patch cudf-nexus-sys "${repo_root}/components/cudf/cudf-nexus-sys"
      add_patch cugraph-nexus-sys "${repo_root}/components/cugraph/cugraph-nexus-sys"
      add_patch rapids-interop "${repo_root}/components/cudf/rapids-interop"
      ;;
    cuvs-nexus)
      add_patch cudf-nexus "${repo_root}/components/cudf/cudf-nexus"
      add_patch cudf-nexus-sys "${repo_root}/components/cudf/cudf-nexus-sys"
      add_patch cuvs-nexus-sys "${repo_root}/components/cuvs/cuvs-nexus-sys"
      add_patch rapids-interop "${repo_root}/components/cudf/rapids-interop"
      ;;
  esac
}

for package in "${packages[@]}"; do
  listing="$(cargo package --locked --allow-dirty -p "${package}" --all-features --list)"
  grep -qx LICENSE <<<"${listing}" || {
    printf 'error: %s package list omits the root LICENSE\n' "${package}" >&2
    exit 1
  }

  package_patches "${package}"
  cargo package \
    --locked \
    --allow-dirty \
    --no-verify \
    -p "${package}" \
    --all-features \
    --target-dir "${package_target}" \
    "${patch_args[@]}"

  version="$(package_version "${package}")"
  archive="${package_target}/package/${package}-${version}.crate"
  [[ -f "${archive}" ]] || {
    printf 'error: cargo package did not create %s\n' "${archive}" >&2
    exit 1
  }
  archive_listing="$(tar -tzf "${archive}")"
  grep -qx "${package}-${version}/LICENSE" <<<"${archive_listing}" || {
    printf 'error: %s archive omits LICENSE\n' "${package}" >&2
    exit 1
  }
  package_extract="${extract_root}/${package}-${version}"
  rm -rf -- "${package_extract}"
  tar -xzf "${archive}" -C "${extract_root}"
done

add_extracted_patch() {
  local package=$1
  add_patch "${package}" "${extract_root}/${package}-$(package_version "${package}")"
}

extracted_patches() {
  patch_args=()
  case "$1" in
    cudf-nexus-sys | cugraph-nexus-sys | cuvs-nexus-sys)
      add_extracted_patch nexus-build-support
      ;;
    cudf-nexus)
      add_extracted_patch cudf-nexus-sys
      add_extracted_patch nexus-build-support
      add_extracted_patch rapids-interop
      ;;
    cugraph-nexus)
      add_extracted_patch cudf-nexus
      add_extracted_patch cudf-nexus-sys
      add_extracted_patch cugraph-nexus-sys
      add_extracted_patch nexus-build-support
      add_extracted_patch rapids-interop
      ;;
    cuvs-nexus)
      add_extracted_patch cudf-nexus
      add_extracted_patch cudf-nexus-sys
      add_extracted_patch cuvs-nexus-sys
      add_extracted_patch nexus-build-support
      add_extracted_patch rapids-interop
      ;;
  esac
}

registry_env=(
  env -i
  "PATH=${PATH}"
  "HOME=${HOME}"
  "CARGO_HOME=${CARGO_HOME:-${HOME}/.cargo}"
  "RUSTUP_HOME=${RUSTUP_HOME:-${HOME}/.rustup}"
  "CARGO_TARGET_DIR=${extract_root}/build-target"
  "NEXUS_NATIVE_DIR=${repo_root}/target/native"
)
for name in CUDAToolkit_ROOT CUDA_HOME CUDA_PATH CC CXX CUDACXX CUDAHOSTCXX; do
  if [[ -v "${name}" ]]; then
    registry_env+=("${name}=${!name}")
  fi
done

for package in "${packages[@]}"; do
  version="$(package_version "${package}")"
  manifest="${extract_root}/${package}-${version}/Cargo.toml"
  extracted_patches "${package}"
  "${registry_env[@]}" cargo generate-lockfile --manifest-path "${manifest}" "${patch_args[@]}"
  "${registry_env[@]}" cargo build --manifest-path "${manifest}" --locked "${patch_args[@]}"

  features="$(python3 - "${manifest}" <<'PY'
import sys
import tomllib

with open(sys.argv[1], "rb") as source:
    docs = tomllib.load(source)["package"]["metadata"]["docs"]["rs"]
print(",".join(docs.get("features", [])))
PY
)"
  doc_features=()
  if [[ -n "${features}" ]]; then
    doc_features=(--features "${features}")
  fi
  env -i \
    PATH="${PATH}" \
    HOME="${HOME}" \
    CARGO_HOME="${CARGO_HOME:-${HOME}/.cargo}" \
    RUSTUP_HOME="${RUSTUP_HOME:-${HOME}/.rustup}" \
    DOCS_RS=1 \
    CARGO_TARGET_DIR="${extract_root}/docs-target" \
    cargo doc --manifest-path "${manifest}" --locked --no-deps "${doc_features[@]}" "${patch_args[@]}"
done

check_missing_native_diagnostic() {
  local package=$1
  local library=$2
  local build_script=$3
  local version
  version="$(package_version "${package}")"
  local manifest="${extract_root}/${package}-${version}/Cargo.toml"
  local log="${extract_root}/${package}-missing-native.log"
  extracted_patches "${package}"
  if env -i \
    PATH="${PATH}" \
    HOME="${HOME}" \
    CARGO_HOME="${CARGO_HOME:-${HOME}/.cargo}" \
    RUSTUP_HOME="${RUSTUP_HOME:-${HOME}/.rustup}" \
    CARGO_TARGET_DIR="${extract_root}/missing-native-target-${package}" \
    cargo check --manifest-path "${manifest}" --locked "${patch_args[@]}" >"${log}" 2>&1; then
    printf 'error: %s unexpectedly built without a native prefix or source tree\n' "${package}" >&2
    exit 1
  fi
  for expected in \
    "${package}: no usable ${library}" \
    "Searched:" \
    'NEXUS_NATIVE_DIR unset' \
    "bash ${build_script}" \
    'https://datafusion.nexus/docs/build/from-source'; do
    grep -Fq "${expected}" "${log}" || {
      printf 'error: %s missing-native diagnostic omits %s\n' "${package}" "${expected}" >&2
      cat "${log}" >&2
      exit 1
    }
  done
  if grep -Eq 'CMake Error|Unable to find libclang|linking with .* failed' "${log}"; then
    printf 'error: %s leaked a lower-level native build failure\n' "${package}" >&2
    cat "${log}" >&2
    exit 1
  fi
}

check_missing_native_diagnostic cudf-nexus-sys libcudf_rust.so scripts/build/build_libcudf.sh
check_missing_native_diagnostic cugraph-nexus-sys libcugraph.so scripts/build/build_libcugraph.sh
check_missing_native_diagnostic cuvs-nexus-sys libcuvs.so scripts/build/build_libcuvs.sh

printf 'component package archive, extracted build, diagnostics, and docs.rs contracts OK\n'

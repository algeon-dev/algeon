#!/usr/bin/env bash
set -euo pipefail

readonly repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly package_target="${repo_root}/target/component-package-check"
readonly extract_root="$(mktemp -d "${TMPDIR:-/tmp}/datafusion-nexus-component-packages.XXXXXX")"
trap 'rm -rf -- "${extract_root}"' EXIT
# Far below the 10 MiB crates.io limit: the leak this guards grows an archive by
# orders of magnitude, so a tight ceiling costs nothing and catches it sooner.
readonly max_archive_bytes=$((2 * 1024 * 1024))
# Only these rebuild from their extracted archive below. The other publishable
# crates would each drag the full FFI graph through a from-scratch build, which
# costs more than the leak it would catch.
readonly components=(
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

# Derived rather than hand-listed so adding a crate or moving a dependency needs
# no matching edit here.
readonly workspace_query='
import json, os, sys

data = json.load(sys.stdin)
packages = {item["name"]: item for item in data["packages"]}


def closure(name):
    seen, stack = set(), [name]
    while stack:
        for dependency in packages[stack.pop()]["dependencies"]:
            other = dependency["name"]
            # A dev-dependency on a workspace path is stripped from the archive,
            # so it never needs a patch entry.
            if other in packages and dependency["kind"] in (None, "build") and other not in seen and other != name:
                seen.add(other)
                stack.append(other)
    return sorted(seen)


mode, args = sys.argv[1], sys.argv[2:]
if mode == "publishable":
    print("\n".join(sorted(name for name, item in packages.items() if item["publish"] is None)))
elif mode == "version":
    print(packages[args[0]]["version"])
elif mode == "manifest-dir":
    print(os.path.dirname(packages[args[0]]["manifest_path"]))
elif mode == "closure":
    print("\n".join(closure(args[0])))
'

workspace_fact() {
  python3 -c "${workspace_query}" "$@" <<<"${metadata}"
}

package_version() {
  workspace_fact version "$1"
}

readarray -t publishable < <(workspace_fact publishable)
# `readarray` from a process substitution keeps the enclosing command's exit
# status, so a failed query would leave an empty list and pass the gate
# vacuously rather than fail it.
if ((${#publishable[@]} < ${#components[@]})); then
  printf 'error: cargo metadata reported %s publishable packages, fewer than the %s known components\n' \
    "${#publishable[@]}" "${#components[@]}" >&2
  exit 1
fi

# Callers set `${patch_root}`; `set -u` turns forgetting it into a failure rather
# than a silent patch against the wrong copy of a dependency.
package_patches() {
  local package=$1 dependency version
  patch_args=()
  while read -r dependency; do
    [[ -n "${dependency}" ]] || continue
    if [[ "${patch_root}" == worktree ]]; then
      patch_args+=(--config "patch.crates-io.${dependency}.path='$(workspace_fact manifest-dir "${dependency}")'")
    else
      version="$(package_version "${dependency}")"
      patch_args+=(--config "patch.crates-io.${dependency}.path='${extract_root}/${dependency}-${version}'")
    fi
  done < <(workspace_fact closure "${package}")
}

for package in "${publishable[@]}"; do
  patch_root=worktree
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

  archive_bytes="$(stat -c %s "${archive}")"
  if ((archive_bytes > max_archive_bytes)); then
    printf 'error: %s archive is %s bytes, over the %s byte ceiling.\n' \
      "${package}" "${archive_bytes}" "${max_archive_bytes}" >&2
    printf 'a package that shares its directory with the workspace needs `include`; its largest entries are:\n' >&2
    tar -tzvf "${archive}" | sort -k3 -n -r | head -20 >&2
    exit 1
  fi
done

for package in "${components[@]}"; do
  version="$(package_version "${package}")"
  archive="${package_target}/package/${package}-${version}.crate"
  archive_listing="$(tar -tzf "${archive}")"
  # The components carry the license as a file because they set `license-file`
  # rather than the SPDX `license` key.
  grep -qx "${package}-${version}/LICENSE" <<<"${archive_listing}" || {
    printf 'error: %s archive omits LICENSE\n' "${package}" >&2
    exit 1
  }
  rm -rf -- "${extract_root:?}/${package}-${version}"
  tar -xzf "${archive}" -C "${extract_root}"
done

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

for package in "${components[@]}"; do
  version="$(package_version "${package}")"
  manifest="${extract_root}/${package}-${version}/Cargo.toml"
  patch_root=extracted
  package_patches "${package}"
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
  patch_root=extracted
  package_patches "${package}"
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

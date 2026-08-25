#!/usr/bin/env bash
# Canonical native libcugraph build for cugraph-sys.
#
# This is a thin wrapper around the upstream cuGraph build script
# (cugraph/build.sh). It adds the contracts cugraph-sys needs on top of a
# stock cuGraph build:
#
#   1. One RMM per process: any process that loads libcugraph together with
#      libcudf must bind every rmm symbol to a single librmm.so build. We make
#      the cudf install prefix self-contained (scripts/build/bootstrap_cudf_install.sh)
#      and force cuGraph to resolve RMM and rapids_logger from it via
#      EXTRA_CMAKE_ARGS, instead of letting CPM build private copies.
#   2. A post-build check that no private RMM/rapids_logger copy leaked into the
#      build tree where it could shadow the canonical one through stale rpaths.
#   3. One cuVS per process: configure cuGraph with the canonical external
#      cuVS prefix and prove libcugraph resolves that exact libcuvs.so.
#
# Env:
#   CUDF_INSTALL_DIR          cudf install prefix (default: target/native/cudf-install)
#   CUVS_INSTALL_DIR          canonical cuVS prefix (default: target/native/cuvs-install)
#   CUGRAPH_CMAKE_BUILD_DIR   cuGraph build dir   (default: target/native/cugraph-build)
#   CMAKE_CUDA_ARCHITECTURES  CUDA arch list (default: NATIVE, cugraph/build.sh's default)
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${repo_root}/scripts/build/cuda_architecture_contract.sh"
upstream_build="${repo_root}/components/cugraph/cugraph/build.sh"
build_dir="${CUGRAPH_CMAKE_BUILD_DIR:-${repo_root}/target/native/cugraph-build}"
cudf_install_dir="${CUDF_INSTALL_DIR:-${repo_root}/target/native/cudf-install}"
default_cuvs_install_dir="${repo_root}/target/native/cuvs-install"
cuvs_install_dir="${CUVS_INSTALL_DIR:-${default_cuvs_install_dir}}"
cuda_architectures="$(nexus_cuda_architectures)"

if [[ ! -x "${upstream_build}" ]]; then
    echo "error: upstream cuGraph build script '${upstream_build}' not found." >&2
    echo "Initialize the cugraph submodule: git submodule update --init --recursive components/cugraph/cugraph" >&2
    exit 1
fi

if [[ -d "${cudf_install_dir}" ]]; then
    cudf_install_dir="$(cd "${cudf_install_dir}" && pwd -P)"
fi
if [[ ! -d "${cuvs_install_dir}" ]]; then
    echo "error: canonical CUVS_INSTALL_DIR '${cuvs_install_dir}' does not exist." >&2
    echo "Build/install cuVS first, or point CUVS_INSTALL_DIR at that canonical prefix." >&2
    exit 1
fi
cuvs_install_dir="$(cd "${cuvs_install_dir}" && pwd -P)"
if [[ -d "${default_cuvs_install_dir}" ]]; then
    default_cuvs_install_dir="$(cd "${default_cuvs_install_dir}" && pwd -P)"
fi

if [[ "${cuvs_install_dir}" == "${default_cuvs_install_dir}" ]]; then
    cuvs_source_revision_path="${cuvs_install_dir}/share/cuvs-nexus/source-revision"
    if [[ ! -f "${cuvs_source_revision_path}" ]]; then
        echo "error: default cuVS install '${cuvs_install_dir}' is missing '${cuvs_source_revision_path#"${cuvs_install_dir}"/}'." >&2
        echo "Rebuild/install the pinned cuVS source, or select a compatible external CUVS_INSTALL_DIR." >&2
        exit 1
    fi
fi

cuvs_config_dir=""
for candidate in "${cuvs_install_dir}/lib/cmake/cuvs" "${cuvs_install_dir}/lib64/cmake/cuvs"; do
    if [[ -f "${candidate}/cuvs-config.cmake" ]]; then
        cuvs_config_dir="$(cd "${candidate}" && pwd -P)"
        break
    fi
done
if [[ -z "${cuvs_config_dir}" ]]; then
    echo "error: CUVS_INSTALL_DIR '${cuvs_install_dir}' does not export cuVS's CMake package." >&2
    echo "Expected lib/cmake/cuvs/cuvs-config.cmake or lib64/cmake/cuvs/cuvs-config.cmake." >&2
    exit 1
fi
cuvs_library_path=""
for candidate in "${cuvs_install_dir}/lib/libcuvs.so" "${cuvs_install_dir}/lib64/libcuvs.so"; do
    if [[ -f "${candidate}" ]]; then
        cuvs_library_path="$(readlink -f "${candidate}")"
        cuvs_runtime_lib_dir="$(cd "$(dirname "${candidate}")" && pwd -P)"
        break
    fi
done
if [[ -z "${cuvs_library_path}" ]]; then
    echo "error: CUVS_INSTALL_DIR '${cuvs_install_dir}' does not contain lib/libcuvs.so or lib64/libcuvs.so." >&2
    exit 1
fi
case "${cuvs_library_path}" in
    "${cuvs_install_dir}"/*) ;;
    *)
        echo "error: '${cuvs_library_path}' resolves outside canonical CUVS_INSTALL_DIR '${cuvs_install_dir}'." >&2
        exit 1
        ;;
esac

nexus_cuda_require_contract \
    "cuDF" "${cudf_install_dir}" "${cuda_architectures}" \
    "bash scripts/build/build_libcudf.sh"
nexus_cuda_require_contract \
    "cuVS" "${cuvs_install_dir}" "${cuda_architectures}" \
    "bash scripts/build/build_libcuvs.sh"
nexus_cuda_require_matching_contracts \
    "cuDF" "${cudf_install_dir}" "cuVS" "${cuvs_install_dir}" \
    "bash scripts/build/build_libcudf.sh && bash scripts/build/build_libcuvs.sh"

# Make the cudf prefix export rmm's full CMake dependency closure (idempotent).
CUDF_INSTALL_DIR="${cudf_install_dir}" "${repo_root}/scripts/build/bootstrap_cudf_install.sh"

if [[ ! -f "${cudf_install_dir}/lib/cmake/rmm/rmm-config.cmake" ]]; then
    echo "error: '${cudf_install_dir}' does not export the rmm CMake package." >&2
    echo "Build and install libcudf first, or point CUDF_INSTALL_DIR at the canonical prefix." >&2
    exit 1
fi

# Force cuGraph to resolve RMM and rapids_logger from the cudf prefix, and cuVS
# from the canonical cuVS prefix. These
# args are appended after cugraph/build.sh's own -D flags, so a later
# -DCMAKE_CUDA_ARCHITECTURES overrides the upstream script's NATIVE
# default. -U rmm_DIR / rapids_logger_DIR force re-resolution to *_ROOT when
# re-configuring a tree that previously pulled a private CPM copy.
extra_cmake_args=(
    "-Drmm_ROOT=${cudf_install_dir}"
    "-Drapids_logger_ROOT=${cudf_install_dir}"
    "-DCUGRAPH_USE_CUVS_STATIC=OFF"
    "-U" "rmm_DIR"
    "-U" "rapids_logger_DIR"
    "-U" "cuvs_DIR"
    "-U" "CPM_PACKAGE_cuvs_*"
    "-U" "CPM_cuvs_SOURCE"
    "-U" "CPM_DOWNLOAD_cuvs"
    "-U" "cuvs_SOURCE_DIR"
    "-U" "cuvs_BINARY_DIR"
    "-U" "CUVS_*"
    "-U" "FETCHCONTENT_SOURCE_DIR_CUVS"
    "-U" "FETCHCONTENT_UPDATES_DISCONNECTED_CUVS"
    "-DCMAKE_FIND_PACKAGE_PREFER_CONFIG=ON"
    "-DCMAKE_PREFIX_PATH=${cudf_install_dir};${cuvs_install_dir}"
    "-Dcuvs_ROOT=${cuvs_install_dir}"
)
extra_cmake_args+=("-DCMAKE_CUDA_ARCHITECTURES=${cuda_architectures}")

# Drive the upstream script: build only libcugraph, no install (-n), no C++ tests.
# PREFIX is required by cugraph/build.sh but unused with -n; point it at the build
# dir so nothing is written outside the tree.
LIBCUGRAPH_BUILD_DIR="${build_dir}" \
PREFIX="${build_dir}" \
EXTRA_CMAKE_ARGS="${extra_cmake_args[*]}" \
    "${upstream_build}" libcugraph -n --skip_cpp_tests

# ---- Validate the single-RMM contract -------------------------------------

cmake_cache_value() {
    local key="$1"
    awk -F= -v key="${key}" '$1 == key { print $2; found = 1 } END { if (!found) exit 1 }' \
        "${build_dir}/CMakeCache.txt"
}

cmake_cache_value_any_type() {
    local key="$1"
    awk -F= -v key="${key}" '$1 ~ ("^" key ":[^=]+$") { print $2; found = 1 } END { if (!found) exit 1 }' \
        "${build_dir}/CMakeCache.txt"
}

canonical_dir() { (cd "$1" && pwd -P); }

check_package_dir() {
    # $1 = cache key, $2 = expected package directory, $3 = package label
    local key="$1" expected="$2" label="$3"
    local resolved
    resolved="$(cmake_cache_value "${key}:PATH")" || {
        echo "error: configure did not write ${key}:PATH to ${build_dir}/CMakeCache.txt." >&2
        echo "The cuGraph build did not expose a verifiable external ${label} package; aborting." >&2
        exit 1
    }
    if [[ "$(canonical_dir "${resolved}")" != "$(canonical_dir "${expected}")" ]]; then
        echo "error: configure resolved ${key}='${resolved}' instead of '${expected}'." >&2
        echo "The cuGraph build would carry its own ${label}; aborting before it can poison consumers." >&2
        exit 1
    fi
}

check_package_dir rmm_DIR "${cudf_install_dir}/lib/cmake/rmm" rmm
check_package_dir rapids_logger_DIR "${cudf_install_dir}/lib/cmake/rapids_logger" rapids_logger
check_package_dir cuvs_DIR "${cuvs_config_dir}" cuVS

configured_prefix_path="$(cmake_cache_value_any_type CMAKE_PREFIX_PATH)" || {
    echo "error: configure did not write CMAKE_PREFIX_PATH to ${build_dir}/CMakeCache.txt." >&2
    exit 1
}
expected_prefix_path="${cudf_install_dir};${cuvs_install_dir}"
if [[ "${configured_prefix_path}" != "${expected_prefix_path}" ]]; then
    echo "error: configure retained CMAKE_PREFIX_PATH='${configured_prefix_path}', expected '${expected_prefix_path}'." >&2
    exit 1
fi

stale_copies="$(find "${build_dir}/_deps" -maxdepth 3 \
    \( -name 'librmm.so*' -o -name 'librapids_logger.so*' -o -name 'libcuvs.so*' \) 2>/dev/null || true)"
if [[ -n "${stale_copies}" ]]; then
    echo "error: stale private RMM/rapids_logger/cuVS artifacts remain under ${build_dir}/_deps:" >&2
    echo "${stale_copies}" >&2
    echo "Remove them (they predate the external-rmm configure) and re-run this script." >&2
    exit 1
fi

cugraph_library="${build_dir}/libcugraph.so"
if [[ ! -f "${cugraph_library}" ]]; then
    echo "error: native build did not produce '${cugraph_library}'." >&2
    exit 1
fi
readelf_output="$(readelf -d "${cugraph_library}")"
needed_cuvs_count="$(grep -Ec 'NEEDED.*Shared library: \[libcuvs[.]so' <<<"${readelf_output}" || true)"
if [[ "${needed_cuvs_count}" != "1" ]]; then
    echo "error: libcugraph must declare exactly one libcuvs.so dependency; readelf reported ${needed_cuvs_count}." >&2
    exit 1
fi
if grep -Eq '/_deps/cuvs[-/]' <<<"${readelf_output}"; then
    echo "error: libcugraph contains an RPATH/RUNPATH into a private _deps cuVS build." >&2
    exit 1
fi
controlled_loader_path="${build_dir}:${cuvs_runtime_lib_dir}:${cudf_install_dir}/lib"
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    controlled_loader_path="${controlled_loader_path}:${LD_LIBRARY_PATH}"
fi
if ! ldd_output="$(LD_LIBRARY_PATH="${controlled_loader_path}" ldd "${cugraph_library}" 2>&1)"; then
    echo "error: ldd failed while validating '${cugraph_library}':" >&2
    echo "${ldd_output}" >&2
    exit 1
fi
if grep -q '=> not found' <<<"${ldd_output}"; then
    echo "error: libcugraph has unresolved loader dependencies:" >&2
    grep '=> not found' <<<"${ldd_output}" >&2
    exit 1
fi
if grep -Eq '/_deps/cuvs[-/]' <<<"${ldd_output}"; then
    echo "error: the dynamic loader selected a private _deps libcuvs." >&2
    exit 1
fi
mapfile -t resolved_cuvs_paths < <(awk '$1 ~ /^libcuvs[.]so/ && $2 == "=>" { print $3 }' <<<"${ldd_output}")
if [[ "${#resolved_cuvs_paths[@]}" != "1" ]]; then
    echo "error: ldd must resolve exactly one libcuvs.so; found ${#resolved_cuvs_paths[@]}." >&2
    exit 1
fi
resolved_cuvs_path="$(readlink -f "${resolved_cuvs_paths[0]}")"
if [[ "${resolved_cuvs_path}" != "${cuvs_library_path}" ]]; then
    echo "error: loader resolved libcuvs to '${resolved_cuvs_path}', expected '${cuvs_library_path}'." >&2
    exit 1
fi

if ! ambient_ldd_output="$(ldd "${cugraph_library}" 2>&1)"; then
    echo "error: ambient ldd failed while checking '${cugraph_library}' for libcuvs shadowing:" >&2
    echo "${ambient_ldd_output}" >&2
    exit 1
fi
mapfile -t ambient_cuvs_paths < <(awk '$1 ~ /^libcuvs[.]so/ && $2 == "=>" && $3 != "not" { print $3 }' \
    <<<"${ambient_ldd_output}")
if [[ "${#ambient_cuvs_paths[@]}" -gt "1" ]]; then
    echo "error: ambient loader resolved multiple libcuvs identities." >&2
    exit 1
fi
if [[ "${#ambient_cuvs_paths[@]}" == "1" ]]; then
    ambient_cuvs_path="$(readlink -f "${ambient_cuvs_paths[0]}")"
    if [[ "${ambient_cuvs_path}" != "${cuvs_library_path}" ]]; then
        echo "error: ambient loader shadows canonical libcuvs with '${ambient_cuvs_path}'." >&2
        exit 1
    fi
fi

metadata_path="${build_dir}/cugraph-nexus-native-contract.env"
printf '%s\n' \
    "CUVS_INSTALL_DIR=${cuvs_install_dir}" \
    "CUVS_LIBRARY_PATH=${cuvs_library_path}" \
    "CUVS_RUNTIME_LIB_DIR=${cuvs_runtime_lib_dir}" >"${metadata_path}"

architecture_contract="$(nexus_cuda_write_contract "${build_dir}" "${cugraph_library}" "${cuda_architectures}")"
nexus_cuda_require_matching_contracts \
    "cuDF" "${cudf_install_dir}" "cuGraph" "${build_dir}" \
    "bash scripts/build/build_libcudf.sh && bash scripts/build/build_libcuvs.sh && bash scripts/build/build_libcugraph.sh"

echo "libcugraph native build OK: rmm/rapids_logger=${cudf_install_dir}, cuVS=${cuvs_install_dir}"
echo "resolved libcuvs: ${cuvs_library_path}"
echo "native contract metadata: ${metadata_path}"
echo "CUDA architectures: requested=${cuda_architectures}, real=$(nexus_cuda_read_contract_value "${architecture_contract}" real)"
echo "libraries: $(cd "${build_dir}" && ls libcugraph*.so 2>/dev/null | tr '\n' ' ')"

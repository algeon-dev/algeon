#!/usr/bin/env bash
# Canonical native libcugraph build for cugraph-sys.
#
# This is a thin wrapper around the upstream cuGraph build script
# (cugraph/build.sh). It adds the two things cugraph-sys needs on top of a
# stock cuGraph build:
#
#   1. One RMM per process: any process that loads libcugraph together with
#      libcudf must bind every rmm symbol to a single librmm.so build. We make
#      the cudf install prefix self-contained (scripts/bootstrap_cudf_install.sh)
#      and force cuGraph to resolve RMM and rapids_logger from it via
#      EXTRA_CMAKE_ARGS, instead of letting CPM build private copies.
#   2. A post-build check that no private RMM/rapids_logger copy leaked into the
#      build tree where it could shadow the canonical one through stale rpaths.
#
# Env:
#   CUDF_INSTALL_DIR          cudf install prefix (default: ../cudf-nexus/cudf-install)
#   CUGRAPH_CPP_BUILD_DIR     cuGraph build dir   (default: cugraph/cpp/build)
#   CMAKE_CUDA_ARCHITECTURES  CUDA arch list (default: NATIVE, cugraph/build.sh's default)
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
upstream_build="${repo_root}/cugraph/build.sh"
build_dir="${CUGRAPH_CPP_BUILD_DIR:-${repo_root}/cugraph/cpp/build}"
cudf_install_dir="${CUDF_INSTALL_DIR:-$(cd "${repo_root}/.." && pwd)/cudf-nexus/cudf-install}"

if [[ ! -x "${upstream_build}" ]]; then
    echo "error: upstream cuGraph build script '${upstream_build}' not found." >&2
    echo "Initialize the cugraph submodule: git submodule update --init --recursive cugraph" >&2
    exit 1
fi

if [[ -d "${cudf_install_dir}" ]]; then
    cudf_install_dir="$(cd "${cudf_install_dir}" && pwd -P)"
fi

# Make the cudf prefix export rmm's full CMake dependency closure (idempotent).
CUDF_INSTALL_DIR="${cudf_install_dir}" "${repo_root}/scripts/bootstrap_cudf_install.sh"

if [[ ! -f "${cudf_install_dir}/lib/cmake/rmm/rmm-config.cmake" ]]; then
    echo "error: '${cudf_install_dir}' does not export the rmm CMake package." >&2
    echo "Build and install libcudf first, or point CUDF_INSTALL_DIR at the canonical prefix." >&2
    exit 1
fi

# Force cuGraph to resolve RMM and rapids_logger from the cudf prefix. These
# args are appended after cugraph/build.sh's own -D flags, so a later
# -DCMAKE_CUDA_ARCHITECTURES (if the caller set one) overrides its NATIVE
# default. -U rmm_DIR / rapids_logger_DIR force re-resolution to *_ROOT when
# re-configuring a tree that previously pulled a private CPM copy.
extra_cmake_args=(
    "-Drmm_ROOT=${cudf_install_dir}"
    "-Drapids_logger_ROOT=${cudf_install_dir}"
    "-U" "rmm_DIR"
    "-U" "rapids_logger_DIR"
)
if [[ -n "${CMAKE_CUDA_ARCHITECTURES:-}" ]]; then
    extra_cmake_args+=("-DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
fi

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

canonical_dir() { (cd "$1" && pwd -P); }

check_package_dir() {
    # $1 = cache key, $2 = expected lib/cmake subdir
    local key="$1" expected="${cudf_install_dir}/lib/cmake/$2"
    local resolved
    resolved="$(cmake_cache_value "${key}:PATH")" || {
        echo "error: configure did not write ${key}:PATH to ${build_dir}/CMakeCache.txt." >&2
        echo "The cuGraph build did not expose a verifiable external $2 package; aborting." >&2
        exit 1
    }
    if [[ "$(canonical_dir "${resolved}")" != "$(canonical_dir "${expected}")" ]]; then
        echo "error: configure resolved ${key}='${resolved}' instead of '${expected}'." >&2
        echo "The cuGraph build would carry its own $2; aborting before it can poison consumers." >&2
        exit 1
    fi
}

check_package_dir rmm_DIR rmm
check_package_dir rapids_logger_DIR rapids_logger

stale_copies="$(find "${build_dir}/_deps" -maxdepth 3 \
    \( -name 'librmm.so*' -o -name 'librapids_logger.so*' \) 2>/dev/null || true)"
if [[ -n "${stale_copies}" ]]; then
    echo "error: stale private RMM/rapids_logger artifacts remain under ${build_dir}/_deps:" >&2
    echo "${stale_copies}" >&2
    echo "Remove them (they predate the external-rmm configure) and re-run this script." >&2
    exit 1
fi

echo "libcugraph native build OK: rmm and rapids_logger resolved from ${cudf_install_dir}"
echo "libraries: $(cd "${build_dir}" && ls libcugraph*.so 2>/dev/null | tr '\n' ' ')"

#!/usr/bin/env bash
# Canonical native cuGraph build for cugraph-sys.
#
# One RMM per process: any process that loads libcugraph together with
# libcudf (datafusion-nexus with --features cugraph) must bind every rmm
# symbol to a single librmm.so build. This script therefore resolves RMM and
# rapids_logger from the cudf install prefix (CUDF_INSTALL_DIR) instead of
# letting CPM build private copies, and refuses to finish if a private copy
# is still reachable from the build tree.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src_dir="${repo_root}/cugraph/cpp"
build_dir="${CUGRAPH_CPP_BUILD_DIR:-${src_dir}/build}"
cudf_install_dir="${CUDF_INSTALL_DIR:-$(cd "${repo_root}/.." && pwd)/cudf-rs/cudf-install}"

if [[ -d "${cudf_install_dir}" ]]; then
    cudf_install_dir="$(cd "${cudf_install_dir}" && pwd -P)"
fi

if [[ ! -f "${cudf_install_dir}/lib/cmake/rmm/rmm-config.cmake" ]]; then
    echo "error: '${cudf_install_dir}' does not export the rmm CMake package (lib/cmake/rmm/rmm-config.cmake)." >&2
    echo "Build and install libcudf first (cudf-rs, CUDF_RMM_BUILD_SHARED=ON is the fork default)," >&2
    echo "or point CUDF_INSTALL_DIR at the prefix that ships the canonical shared librmm.so." >&2
    exit 1
fi
if [[ ! -f "${cudf_install_dir}/lib/cmake/rapids_logger/rapids_logger-config.cmake" ]]; then
    echo "error: '${cudf_install_dir}' does not export the rapids_logger CMake package (lib/cmake/rapids_logger/rapids_logger-config.cmake)." >&2
    echo "Build and install libcudf first, or point CUDF_INSTALL_DIR at the prefix that ships the canonical rapids_logger package." >&2
    exit 1
fi

cmake -S "${src_dir}" -B "${build_dir}" \
    -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-NATIVE}" \
    -DBUILD_TESTS=OFF \
    -DBUILD_CUGRAPH_MG_TESTS=OFF \
    -U rmm_DIR -U 'CPM_PACKAGE_rmm_*' -U 'RMM_*' \
    -U FETCHCONTENT_SOURCE_DIR_RMM -U FETCHCONTENT_UPDATES_DISCONNECTED_RMM \
    -U rapids_logger_DIR -U 'CPM_PACKAGE_rapids_logger_*' -U 'RAPIDS_LOGGER_*' \
    -U FETCHCONTENT_SOURCE_DIR_RAPIDS_LOGGER -U FETCHCONTENT_UPDATES_DISCONNECTED_RAPIDS_LOGGER \
    -Drmm_ROOT="${cudf_install_dir}" \
    -Drapids_logger_ROOT="${cudf_install_dir}"

cmake_cache_value() {
    local key="$1"
    awk -F= -v key="${key}" '$1 == key { print $2; found = 1 } END { if (!found) exit 1 }' "${build_dir}/CMakeCache.txt"
}

canonical_dir() {
    local path="$1"
    (cd "${path}" && pwd -P)
}

resolved_rmm_dir="$(cmake_cache_value 'rmm_DIR:PATH')" || {
    echo "error: configure did not write rmm_DIR:PATH to ${build_dir}/CMakeCache.txt." >&2
    echo "The cuGraph build did not expose a verifiable external RMM package; aborting." >&2
    exit 1
}
expected_rmm_dir="${cudf_install_dir}/lib/cmake/rmm"
resolved_rmm_dir_real="$(canonical_dir "${resolved_rmm_dir}")"
expected_rmm_dir_real="$(canonical_dir "${expected_rmm_dir}")"
if [[ "${resolved_rmm_dir_real}" != "${expected_rmm_dir_real}" ]]; then
    echo "error: configure resolved rmm_DIR='${resolved_rmm_dir}' instead of '${expected_rmm_dir}'." >&2
    echo "The cuGraph build would carry its own RMM; aborting before it can poison consumers." >&2
    exit 1
fi

resolved_logger_dir="$(cmake_cache_value 'rapids_logger_DIR:PATH')" || {
    echo "error: configure did not write rapids_logger_DIR:PATH to ${build_dir}/CMakeCache.txt." >&2
    echo "The cuGraph build did not expose a verifiable external rapids_logger package; aborting." >&2
    exit 1
}
expected_logger_dir="${cudf_install_dir}/lib/cmake/rapids_logger"
resolved_logger_dir_real="$(canonical_dir "${resolved_logger_dir}")"
expected_logger_dir_real="$(canonical_dir "${expected_logger_dir}")"
if [[ "${resolved_logger_dir_real}" != "${expected_logger_dir_real}" ]]; then
    echo "error: configure resolved rapids_logger_DIR='${resolved_logger_dir}' instead of '${expected_logger_dir}'." >&2
    echo "The cuGraph build would carry its own rapids_logger; aborting before it can poison consumers." >&2
    exit 1
fi

cmake --build "${build_dir}"

# A leftover CPM-built copy in the build tree can still shadow the canonical
# librmm.so through stale rpaths; fail loudly instead of leaving the landmine.
stale_copies="$(find "${build_dir}/_deps" -maxdepth 3 \( -name 'librmm.so*' -o -name 'librapids_logger.so*' \) 2>/dev/null || true)"
if [[ -n "${stale_copies}" ]]; then
    echo "error: stale private RMM/rapids_logger artifacts remain under ${build_dir}/_deps:" >&2
    echo "${stale_copies}" >&2
    echo "Remove them (they predate the external-rmm configure) and re-run this script." >&2
    exit 1
fi

echo "cuGraph native build OK: rmm and rapids_logger resolved from ${cudf_install_dir}"

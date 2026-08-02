#!/usr/bin/env bash
# Make a libcudf install prefix self-contained for cugraph-sys.
#
# cuGraph configures against the cudf install prefix and does a real
# `find_package(rmm)`. rmm-dependencies.cmake then `find_dependency(nvtx3)` and
# `find_dependency(rapids_logger)` (which in turn needs fmt + spdlog). The
# upstream libcudf install only ships cudf + rmm + cccl + gtest; it does NOT
# install nvtx3 / rapids_logger / fmt / spdlog, so a downstream consumer using
# the installed prefix cannot satisfy rmm's dependency closure. cudf never
# hits this because cudf-sys links the .so files directly without a CMake
# configure.
#
# This script installs those missing CMake packages into the cudf install
# prefix from the cudf C++ build tree (`_deps/*-build`). It is idempotent:
# packages already present are left alone. Run it once per machine after
# building libcudf and before building cugraph-sys.
#
# Env:
#   CUDF_INSTALL_DIR    cudf install prefix (default: ../cudf-nexus/cudf-install)
#   CUDF_CPP_BUILD_DIR  cudf C++ build dir   (default: <prefix>/../cudf/cpp/build)
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cudf_install_dir="${CUDF_INSTALL_DIR:-$(cd "${repo_root}/.." && pwd)/cudf-nexus/cudf-install}"

if [[ ! -d "${cudf_install_dir}" ]]; then
    echo "error: cudf install prefix '${cudf_install_dir}' does not exist." >&2
    echo "Build and install libcudf first, or set CUDF_INSTALL_DIR." >&2
    exit 1
fi
cudf_install_dir="$(cd "${cudf_install_dir}" && pwd -P)"

# The cudf build tree (with its CPM _deps) sits next to the install prefix by
# convention: <prefix>/../cudf/cpp/build. Allow an explicit override.
cudf_build_dir="${CUDF_CPP_BUILD_DIR:-$(cd "${cudf_install_dir}/.." && pwd)/cudf/cpp/build}"

deps_dir="${cudf_build_dir}/_deps"

package_present() {
    # $1 = lib/cmake subdir, $2 = config file name
    [[ -f "${cudf_install_dir}/lib/cmake/$1/$2" ]]
}

require_deps_dir() {
    if [[ ! -d "${deps_dir}" ]]; then
        echo "error: cudf build tree '${deps_dir}' not found, but the cudf install prefix is" >&2
        echo "missing required CMake packages. Rebuild libcudf (keeping cudf/cpp/build intact)" >&2
        echo "or set CUDF_CPP_BUILD_DIR to the cudf C++ build directory, then re-run." >&2
        exit 1
    fi
}

install_from_build() {
    # $1 = _deps build subdir
    local build="${deps_dir}/$1"
    if [[ ! -f "${build}/cmake_install.cmake" ]]; then
        echo "error: '${build}/cmake_install.cmake' not found; cannot install '$1'." >&2
        echo "Rebuild libcudf so its CPM dependency build trees are present." >&2
        exit 1
    fi
    cmake --install "${build}" --prefix "${cudf_install_dir}" >/dev/null
}

# nvtx3: header-only; rmm::rmm needs the nvtx3::nvtx3-cpp target.
if package_present nvtx3 nvtx3-config.cmake; then
    echo "nvtx3: already present"
else
    require_deps_dir
    echo "nvtx3: installing into ${cudf_install_dir}"
    install_from_build nvtx3-build
fi

# rapids_logger (+ fmt + spdlog): rapids_logger's install script also installs
# fmt and spdlog (its CMake config find_dependency()s both). A clean cudf build
# may have removed the small static archives; rebuild any that are missing so the
# install step does not fail on a missing libfmt.a / libspdlog.a.
if package_present rapids_logger rapids_logger-config.cmake; then
    echo "rapids_logger: already present"
else
    require_deps_dir
    for static_lib in fmt-build/libfmt.a spdlog-build/libspdlog.a rapids_logger-build/librapids_logger.a; do
        if [[ ! -f "${deps_dir}/${static_lib}" && -f "${deps_dir}/$(dirname "${static_lib}")/cmake_install.cmake" ]]; then
            echo "rapids_logger: rebuilding missing ${static_lib}"
            cmake --build "${deps_dir}/$(dirname "${static_lib}")" >/dev/null
        fi
    done
    echo "rapids_logger (+ fmt, spdlog): installing into ${cudf_install_dir}"
    install_from_build rapids_logger-build
fi

# Verify the closure cuGraph's configure needs.
missing=0
for entry in \
    "rmm/rmm-config.cmake" \
    "nvtx3/nvtx3-config.cmake" \
    "rapids_logger/rapids_logger-config.cmake" \
    "fmt/fmt-config.cmake" \
    "spdlog/spdlogConfig.cmake"; do
    if [[ ! -f "${cudf_install_dir}/lib/cmake/${entry}" ]]; then
        echo "error: expected CMake package missing after bootstrap: lib/cmake/${entry}" >&2
        missing=1
    fi
done
[[ "${missing}" -eq 0 ]] || exit 1

echo "cudf install prefix is self-contained for cugraph-sys: ${cudf_install_dir}"

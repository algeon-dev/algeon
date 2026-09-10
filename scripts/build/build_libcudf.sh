#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
source "${repo_root}/scripts/build/cuda_architecture_contract.sh"
upstream_build="${repo_root}/components/cudf/cudf/build.sh"
build_dir="${CUDF_CPP_BUILD_DIR:-${repo_root}/target/native/cudf-build}"
install_dir="${CUDF_INSTALL_DIR:-${repo_root}/target/native/cudf-install}"

build_dir="$(realpath -m -- "${build_dir}")"
install_dir="$(realpath -m -- "${install_dir}")"
cache_file="${build_dir}/CMakeCache.txt"
cuda_architectures="$(nexus_cuda_architectures)"

validate_cache_identity() {
    local cached_install_dir
    local dependency_dir
    local dependency_key

    cached_install_dir="$(sed -n 's/^CMAKE_INSTALL_PREFIX:PATH=//p' "${cache_file}")"
    if [[ -n "${cached_install_dir}" && "${cached_install_dir}" != "${install_dir}" ]]; then
        echo "error: cuDF build tree '${build_dir}' is configured for '${cached_install_dir}'," >&2
        echo "not the requested install prefix '${install_dir}'." >&2
        echo "Use a matching CUDF_CPP_BUILD_DIR or move the stale build tree aside before rebuilding." >&2
        exit 1
    fi

    while IFS='=' read -r dependency_key dependency_dir; do
        case "${dependency_dir}" in
            *-NOTFOUND | "${build_dir}"/* | "${install_dir}"/*) ;;
            *)
                echo "error: cuDF build tree '${build_dir}' resolves ${dependency_key%%:*} from" >&2
                echo "'${dependency_dir}', outside its build/install identity." >&2
                echo "Move the stale build tree aside before rebuilding." >&2
                exit 1
                ;;
        esac
    done < <(sed -n -E '/^(CCCL_DIR|fmt_DIR|nvtx3_DIR|rapids_logger_DIR|rmm_DIR|spdlog_DIR):PATH=/p' "${cache_file}")
}

if [[ -f "${cache_file}" ]]; then
    validate_cache_identity
fi

if [[ ! -x "${upstream_build}" ]]; then
    echo "error: upstream cuDF build script '${upstream_build}' not found." >&2
    echo "Initialize the cuDF submodule: git submodule update --init --recursive components/cudf/cudf" >&2
    exit 1
fi

LIB_BUILD_DIR="${build_dir}" \
INSTALL_PREFIX="${install_dir}" \
CUDF_CMAKE_CUDA_ARCHITECTURES="${cuda_architectures}" \
    "${upstream_build}" libcudf

validate_cache_identity

CUDF_INSTALL_DIR="${install_dir}" \
CUDF_CPP_BUILD_DIR="${build_dir}" \
    "${repo_root}/scripts/build/bootstrap_cudf_install.sh"

cudf_library=""
for candidate in "${install_dir}/lib/libcudf.so" "${install_dir}/lib64/libcudf.so"; do
    if [[ -f "${candidate}" ]]; then
        cudf_library="${candidate}"
        break
    fi
done
[[ -n "${cudf_library}" ]] || {
    echo "error: canonical cuDF install does not contain libcudf.so under ${install_dir}/{lib,lib64}" >&2
    exit 1
}
architecture_contract="$(nexus_cuda_write_contract "${install_dir}" "${cudf_library}" "${cuda_architectures}")"

echo "canonical cuDF/RMM prefix ready: ${install_dir}"
echo "native build: ${build_dir}"
echo "CUDA architectures: requested=${cuda_architectures}, real=$(nexus_cuda_read_contract_value "${architecture_contract}" real)"

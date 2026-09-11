#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
source "${repo_root}/scripts/build/cuda_architecture_contract.sh"
source_dir="${repo_root}/components/cuvs/cuvs"
build_dir="${CUVS_CPP_BUILD_DIR:-${repo_root}/target/native/cuvs-build}"
install_dir="${CUVS_INSTALL_DIR:-${repo_root}/target/native/cuvs-install}"
cudf_install_dir="${CUDF_INSTALL_DIR:-${repo_root}/target/native/cudf-install}"
cuda_architectures="$(algeon_cuda_architectures)"
(cd "${repo_root}" && scripts/ci/check_native_sources.sh cuvs)

[[ -f "${source_dir}/VERSION" ]] || {
    echo "error: cuVS source does not contain VERSION: ${source_dir}" >&2
    exit 1
}
source_version="$(<"${source_dir}/VERSION")"
if [[ "${source_version}" =~ ^([0-9][0-9])\.([0-9][0-9])\.([0-9][0-9])$ ]]; then
    native_version="$((10#${BASH_REMATCH[1]})).$((10#${BASH_REMATCH[2]})).$((10#${BASH_REMATCH[3]}))"
else
    echo "error: unsupported cuVS VERSION format: ${source_version}" >&2
    exit 1
fi
if ! source_revision="$(git -C "${source_dir}" rev-parse HEAD 2>/dev/null)"; then
    echo "error: cuVS source is not a git checkout: ${source_dir}" >&2
    exit 1
fi

cuda_toolkit_root="${CUDAToolkit_ROOT:-${CUDA_HOME:-${CUDA_PATH:-/usr/local/cuda}}}"
cuda_runtime_lib=""
for candidate in "${cuda_toolkit_root}/lib64/libcudart.so" "${cuda_toolkit_root}/lib/libcudart.so"; do
    if [[ -e "${candidate}" ]]; then
        [[ -z "${cuda_runtime_lib}" ]] || {
            echo "error: CUDA toolkit exposes libcudart.so in more than one library directory: ${cuda_toolkit_root}" >&2
            exit 1
        }
        cuda_runtime_lib="$(realpath -e -- "${candidate}")"
    fi
done
[[ -n "${cuda_runtime_lib}" ]] || {
    echo "error: CUDA toolkit does not expose dynamic libcudart.so under ${cuda_toolkit_root}/{lib64,lib}" >&2
    exit 1
}

[[ -f "${cudf_install_dir}/lib/cmake/rmm/rmm-config.cmake" ]] || {
    echo "error: canonical cuDF/RMM prefix does not export rmm: ${cudf_install_dir}" >&2
    echo "Build/install cuDF from components/cudf/cudf first or set CUDF_INSTALL_DIR." >&2
    exit 1
}
[[ -f "${cudf_install_dir}/lib/cmake/rapids_logger/rapids_logger-config.cmake" ]] || {
    echo "error: canonical cuDF/RMM prefix does not export rapids_logger: ${cudf_install_dir}" >&2
    echo "Run scripts/build/bootstrap_cudf_install.sh for that prefix, then retry." >&2
    exit 1
}
algeon_cuda_require_contract \
    "cuDF" "${cudf_install_dir}" "${cuda_architectures}" \
    "bash scripts/build/build_libcudf.sh"

# cuVS's header-only CUTLASS setup reads the static-named cache entry
# even for shared-only builds, so bind that entry to the selected shared file.
cmake -S "${source_dir}/cpp" -B "${build_dir}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${install_dir}" \
    -DCMAKE_CUDA_ARCHITECTURES="${cuda_architectures}" \
    -DCMAKE_PREFIX_PATH="${cudf_install_dir}" \
    -Drmm_ROOT="${cudf_install_dir}" \
    -Drapids_logger_ROOT="${cudf_install_dir}" \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_C_LIBRARY=ON \
    -DBUILD_TESTS=OFF \
    -DBUILD_C_TESTS=OFF \
    -DBUILD_CUVS_BENCH=OFF \
    -DBUILD_MG_ALGOS=OFF \
    -DCMAKE_CUDA_RUNTIME_LIBRARY=SHARED \
    -DCUDA_cudart_static_LIBRARY="${cuda_runtime_lib}" \
    -DCUVS_STATIC_RAPIDS_LIBRARIES=OFF \
    -DCUVSC_STATIC_CUVS_LIBRARY=OFF \
    -DCUDA_STATIC_MATH_LIBRARIES=OFF \
    -DCUVS_COMPILE_DYNAMIC_ONLY=ON
cmake --build "${build_dir}" --parallel "${PARALLEL_LEVEL:-$(nproc)}" --target cuvs cuvs_c
cmake --install "${build_dir}"

dlpack_source_dir="${build_dir}/_deps/dlpack-src/include/dlpack"
[[ -f "${dlpack_source_dir}/dlpack.h" ]] || {
    echo "error: cuVS build did not materialize the required DLPack public headers: ${dlpack_source_dir}" >&2
    exit 1
}
cmake -E copy_directory "${dlpack_source_dir}" "${install_dir}/include/dlpack"
[[ -f "${install_dir}/include/dlpack/dlpack.h" ]] || {
    echo "error: canonical install is missing the public DLPack header required by cuVS C headers" >&2
    exit 1
}

lib_dir=""
for candidate in "${install_dir}/lib" "${install_dir}/lib64"; do
    if compgen -G "${candidate}/libcuvs.so*" >/dev/null && compgen -G "${candidate}/libcuvs_c.so*" >/dev/null; then
        lib_dir="${candidate}"
        break
    fi
done
[[ -n "${lib_dir}" ]] || {
    echo "error: install did not produce both libcuvs.so and libcuvs_c.so under ${install_dir}/{lib,lib64}" >&2
    exit 1
}
[[ -f "${install_dir}/lib/cmake/cuvs/cuvs-config.cmake" || -f "${install_dir}/lib64/cmake/cuvs/cuvs-config.cmake" ]] || {
    echo "error: install did not produce the cuVS CMake package under ${install_dir}" >&2
    exit 1
}

cudf_lib_dir=""
for candidate in "${cudf_install_dir}/lib" "${cudf_install_dir}/lib64"; do
    if [[ -e "${candidate}/librmm.so" ]]; then
        [[ -z "${cudf_lib_dir}" ]] || {
            echo "error: canonical cuDF/RMM prefix exposes librmm.so in more than one library directory" >&2
            exit 1
        }
        cudf_lib_dir="${candidate}"
    fi
done
[[ -n "${cudf_lib_dir}" ]] || {
    echo "error: canonical cuDF/RMM prefix does not contain librmm.so under ${cudf_install_dir}/{lib,lib64}" >&2
    exit 1
}
canonical_rmm="$(realpath -e -- "${cudf_lib_dir}/librmm.so")"
canonical_cuvs="$(realpath -e -- "${lib_dir}/libcuvs.so")"

dynamic_entries="$(readelf -d "${canonical_cuvs}")"
grep -Eq 'Shared library: \[libcudart\.so' <<<"${dynamic_entries}" || {
    echo "error: canonical libcuvs.so does not use the shared CUDA runtime" >&2
    exit 1
}
static_runtime_symbol="$(nm --defined-only "${canonical_cuvs}" 2>/dev/null | awk '$2 ~ /^[Tt]$/ && $3 == "cudaMalloc" { found = $0 } END { print found }')"
[[ -z "${static_runtime_symbol}" ]] || {
    echo "error: canonical libcuvs.so embeds the static CUDA runtime: ${static_runtime_symbol}" >&2
    exit 1
}

command -v patchelf >/dev/null || {
    echo "error: patchelf is required to bind the canonical cuVS install to its selected RMM runtime" >&2
    exit 1
}
for library in "${lib_dir}/libcuvs.so" "${lib_dir}/libcuvs_c.so"; do
    patchelf --set-rpath "\$ORIGIN:${cudf_lib_dir}" "${library}"
done

marker_dir="${install_dir}/share/algeon-cuvs"
mkdir -p "${marker_dir}"
printf '%s\n' "${source_revision}" >"${marker_dir}/source-revision"
printf '%s\n' "${native_version}" >"${marker_dir}/native-version"
printf '%s\n' "${canonical_rmm}" >"${marker_dir}/rmm-runtime-lib"

loader_path="${lib_dir}:${cudf_lib_dir}"
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    loader_path="${loader_path}:${LD_LIBRARY_PATH}"
fi
for library in "${lib_dir}/libcuvs.so" "${lib_dir}/libcuvs_c.so"; do
    [[ -e "${library}" ]] || continue
    ldd_output="$(LD_LIBRARY_PATH="${loader_path}" ldd "${library}")"
    if grep -q 'not found' <<<"${ldd_output}"; then
        echo "error: unresolved dependency in ${library}:" >&2
        printf '%s\n' "${ldd_output}" >&2
        exit 1
    fi
    resolved_rmm="$(awk '$1 == "librmm.so" && $2 == "=>" { print $3 }' <<<"${ldd_output}")"
    [[ -n "${resolved_rmm}" && "$(realpath -e -- "${resolved_rmm}")" == "${canonical_rmm}" ]] || {
        echo "error: ${library} did not resolve the canonical RMM ${canonical_rmm}: ${resolved_rmm:-<missing>}" >&2
        exit 1
    }
    if [[ "${library}" == "${lib_dir}/libcuvs_c.so" ]]; then
        resolved_cuvs="$(awk '$1 == "libcuvs.so" && $2 == "=>" { print $3 }' <<<"${ldd_output}")"
        [[ -n "${resolved_cuvs}" && "$(realpath -e -- "${resolved_cuvs}")" == "${canonical_cuvs}" ]] || {
            echo "error: libcuvs_c.so did not resolve the canonical cuVS ${canonical_cuvs}: ${resolved_cuvs:-<missing>}" >&2
            exit 1
        }
    fi
done

architecture_contract="$(algeon_cuda_write_contract "${install_dir}" "${canonical_cuvs}" "${cuda_architectures}")"
algeon_cuda_require_matching_contracts \
    "cuDF" "${cudf_install_dir}" "cuVS" "${install_dir}" \
    "bash scripts/build/build_libcudf.sh && bash scripts/build/build_libcuvs.sh"

echo "canonical cuVS prefix ready: ${install_dir}"
echo "source revision: ${source_revision}"
echo "native version: ${native_version}"
echo "CUDA architectures: requested=${cuda_architectures}, real=$(algeon_cuda_read_contract_value "${architecture_contract}" real)"
echo "Cargo: CUVS_ROOT=${install_dir} cargo check --workspace --all-targets --all-features"

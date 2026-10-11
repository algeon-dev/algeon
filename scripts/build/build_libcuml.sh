#!/usr/bin/env bash
# Build and install the pinned cuML C++ library against the canonical cuDF/RMM
# and cuVS prefixes.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
source "${repo_root}/scripts/build/cuda_architecture_contract.sh"
source_tree="${repo_root}/components/cuml/cuml"
upstream_build="${source_tree}/build.sh"
build_dir="$(realpath -m -- "${CUML_CPP_BUILD_DIR:-${repo_root}/target/native/cuml-build}")"
install_dir="$(realpath -m -- "${CUML_INSTALL_DIR:-${repo_root}/target/native/cuml-install}")"
cudf_install_dir="$(realpath -m -- "${CUDF_INSTALL_DIR:-${repo_root}/target/native/cudf-install}")"
cuvs_install_dir="$(realpath -m -- "${CUVS_INSTALL_DIR:-${repo_root}/target/native/cuvs-install}")"
cuda_architectures="$(algeon_cuda_architectures)"

"${repo_root}/scripts/ci/check_native_sources.sh" cuml
if [[ ! -x "${upstream_build}" ]]; then
    echo "error: upstream cuML build script '${upstream_build}' not found." >&2
    echo "Initialize components/cuml/cuml at the root-registered v26.08.00 pin." >&2
    exit 1
fi

cuvs_config_dir=""
for candidate in "${cuvs_install_dir}/lib/cmake/cuvs" "${cuvs_install_dir}/lib64/cmake/cuvs"; do
    if [[ -f "${candidate}/cuvs-config.cmake" ]]; then
        cuvs_config_dir="$(cd "${candidate}" && pwd -P)"
        break
    fi
done
if [[ -z "${cuvs_config_dir}" ]]; then
    echo "error: '${cuvs_install_dir}' does not export the cuVS CMake package." >&2
    exit 1
fi
cuvs_library_path=""
cuvs_runtime_lib_dir=""
for candidate in "${cuvs_install_dir}/lib/libcuvs.so" "${cuvs_install_dir}/lib64/libcuvs.so"; do
    if [[ -f "${candidate}" ]]; then
        cuvs_library_path="$(readlink -f "${candidate}")"
        cuvs_runtime_lib_dir="$(cd "$(dirname "${candidate}")" && pwd -P)"
        break
    fi
done
if [[ -z "${cuvs_library_path}" || "${cuvs_library_path}" != "${cuvs_install_dir}"/* ]]; then
    echo "error: canonical '${cuvs_install_dir}' does not contain its own libcuvs.so." >&2
    exit 1
fi

algeon_cuda_require_contract \
    "cuDF" "${cudf_install_dir}" "${cuda_architectures}" \
    "bash scripts/build/build_libcudf.sh"
algeon_cuda_require_contract \
    "cuVS" "${cuvs_install_dir}" "${cuda_architectures}" \
    "bash scripts/build/build_libcuvs.sh"
algeon_cuda_require_matching_contracts \
    "cuDF" "${cudf_install_dir}" "cuVS" "${cuvs_install_dir}" \
    "bash scripts/build/build_libcudf.sh && bash scripts/build/build_libcuvs.sh"
CUDF_INSTALL_DIR="${cudf_install_dir}" "${repo_root}/scripts/build/bootstrap_cudf_install.sh"

if [[ ! -f "${cudf_install_dir}/lib/cmake/rmm/rmm-config.cmake" ]]; then
    echo "error: '${cudf_install_dir}' does not export rmm's CMake package." >&2
    exit 1
fi

# build.sh tokenizes this variable on whitespace. Keep CMake values free of
# spaces, and quote semicolon lists as single shell tokens in the argv passed on.
extra_cmake_args=(
    "-DCUML_ALGORITHMS=hdbscan;pca;tsvd;randomforest;logisticregression;umap;spectralclustering"
    "-DBUILD_CUML_TESTS=OFF"
    "-DBUILD_PRIMS_TESTS=OFF"
    "-DBUILD_CUML_EXAMPLES=OFF"
    "-DBUILD_CUML_BENCH=OFF"
    "-DDETECT_CONDA_ENV=OFF"
    "-Drmm_ROOT=${cudf_install_dir}"
    "-Drapids_logger_ROOT=${cudf_install_dir}"
    "-Dcuvs_ROOT=${cuvs_install_dir}"
    "-DCMAKE_FIND_PACKAGE_PREFER_CONFIG=ON"
    "-DCUML_USE_CUVS_STATIC=OFF"
    "-DCUML_USE_TREELITE_STATIC=OFF"
    "-DDISABLE_OPENMP=ON"
    "-DCMAKE_PROJECT_treelite_INCLUDE=${repo_root}/scripts/build/cmake/treelite_disable_openmp.cmake"
    "-DCMAKE_PROJECT_NVForest_INCLUDE=${repo_root}/scripts/build/cmake/nvforest_backport_196.cmake"
    "-DCPM_DOWNLOAD_Treelite=ON"
    "-DCPM_DOWNLOAD_nvforest=ON"
    "-U" "Treelite_DIR"
    "-U" "nvforest_DIR"
    "-DCMAKE_CUDA_FLAGS=-Xcompiler=-Wno-unknown-pragmas"
    "-DCMAKE_CUDA_ARCHITECTURES=${cuda_architectures}"
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
    "-DCMAKE_PREFIX_PATH=${cudf_install_dir};${cuvs_install_dir}"
)
# Avoid an old build tree retaining a different source/install identity.
if [[ -f "${build_dir}/CMakeCache.txt" ]]; then
    cached_source="$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "${build_dir}/CMakeCache.txt")"
    if [[ -n "${cached_source}" && "$(realpath -m -- "${cached_source}")" != "${source_tree}/cpp" ]]; then
        echo "error: cuML build tree '${build_dir}' belongs to '${cached_source}', not '${source_tree}/cpp'." >&2
        echo "Select a matching CUML_CPP_BUILD_DIR or move the stale build tree." >&2
        exit 1
    fi
fi

LIBCUML_BUILD_DIR="${build_dir}" \
INSTALL_PREFIX="${install_dir}" \
CUML_EXTRA_CMAKE_ARGS="${extra_cmake_args[*]}" \
    "${upstream_build}" libcuml -n --singlegpu --nolibcumltest --configure-only
cmake --build "${build_dir}" --parallel "${PARALLEL_LEVEL:-$(nproc)}"

openmp_cache_value="$(awk -F= '$1 == "DISABLE_OPENMP:BOOL" { print $2; found = 1 } END { if (!found) exit 1 }' "${build_dir}/CMakeCache.txt")" || {
    echo "error: configure did not record DISABLE_OPENMP:BOOL." >&2
    exit 1
}
if [[ "${openmp_cache_value}" != ON ]]; then
    echo "error: cuML was configured with DISABLE_OPENMP=${openmp_cache_value}, expected ON." >&2
    exit 1
fi

cmake_cache_value() {
    awk -F= -v key="$1" '$1 == key { print $2; found = 1 } END { if (!found) exit 1 }' \
        "${build_dir}/CMakeCache.txt"
}
check_package_dir() {
    local key="$1" expected="$2" label="$3" resolved
    resolved="$(cmake_cache_value "${key}:PATH")" || {
        echo "error: configure did not record ${key}:PATH." >&2
        exit 1
    }
    if [[ "$(realpath -m -- "${resolved}")" != "$(realpath -m -- "${expected}")" ]]; then
        echo "error: ${key} resolved to '${resolved}', expected '${expected}' (${label})." >&2
        exit 1
    fi
}
check_package_dir rmm_DIR "${cudf_install_dir}/lib/cmake/rmm" rmm
check_package_dir rapids_logger_DIR "${cudf_install_dir}/lib/cmake/rapids_logger" rapids_logger
check_package_dir cuvs_DIR "${cuvs_config_dir}" cuVS
# Treelite and nvForest must come from the pinned sources in this build tree:
# a find_package hit on an earlier install would bypass the OpenMP override.
for package in Treelite nvforest; do
    resolved="$(cmake_cache_value "CPM_PACKAGE_${package}_SOURCE_DIR:INTERNAL")" || {
        echo "error: configure did not build ${package} from source." >&2
        exit 1
    }
    expected="${build_dir}/_deps/${package,,}-src"
    if [[ "$(realpath -m -- "${resolved}")" != "$(realpath -m -- "${expected}")" ]]; then
        echo "error: ${package} source resolved to '${resolved}', expected '${expected}'." >&2
        exit 1
    fi
done
configured_prefix_path="$(awk -F= '$1 == "CMAKE_PREFIX_PATH:UNINITIALIZED" || $1 == "CMAKE_PREFIX_PATH:STRING" { print $2; found = 1 } END { if (!found) exit 1 }' "${build_dir}/CMakeCache.txt")" || {
    echo "error: configure did not record CMAKE_PREFIX_PATH." >&2
    exit 1
}
if [[ "${configured_prefix_path}" != "${cudf_install_dir};${cuvs_install_dir}" ]]; then
    echo "error: configure retained CMAKE_PREFIX_PATH='${configured_prefix_path}', expected '${cudf_install_dir};${cuvs_install_dir}'." >&2
    exit 1
fi

stale_copies="$(find "${build_dir}/_deps" -maxdepth 4 \
    \( -name 'librmm.so*' -o -name 'librapids_logger.so*' -o -name 'libcuvs.so*' \) 2>/dev/null || true)"
if [[ -n "${stale_copies}" ]]; then
    echo "error: private RMM/rapids_logger/cuVS libraries remain under '${build_dir}/_deps':" >&2
    echo "${stale_copies}" >&2
    exit 1
fi

cmake --install "${build_dir}" --prefix "${install_dir}"
cuml_library="${install_dir}/lib/libcuml.so"
if [[ ! -f "${cuml_library}" ]]; then
    echo "error: native build did not produce '${cuml_library}'." >&2
    exit 1
fi
# The installed libraries carry only $ORIGIN, and RUNPATH is not transitive,
# so a consumer that links libcuml without its own cuVS edge could not
# resolve libcuvs.so. Bind the install to the selected prefixes, as the cuVS
# install binds to its RMM prefix.
algeon_bind_install_runpath "${cuml_library}" "\$ORIGIN:${cuvs_runtime_lib_dir}:${cudf_install_dir}/lib"
algeon_bind_install_runpath "${install_dir}/lib/libnvforest.so" "\$ORIGIN:${cudf_install_dir}/lib"
if ! nm -DC --defined-only "${cuml_library}" | grep 'ML::SpectralEmbedding::transform' >/dev/null; then
    echo "error: the restricted cuML build omitted spectral embedding required by UMAP." >&2
    exit 1
fi
openmp_symbols="$(nm -D --undefined-only "${cuml_library}" | grep -E '[[:space:]](GOMP_|omp_)' || true)"
if [[ -n "${openmp_symbols}" ]]; then
    echo "error: libcuml.so imports OpenMP symbols despite DISABLE_OPENMP=ON:" >&2
    echo "${openmp_symbols}" >&2
    exit 1
fi
readelf_output="$(readelf -d "${cuml_library}")"
needed_cuvs_count="$(grep -Ec 'NEEDED.*Shared library: \[libcuvs[.]so' <<<"${readelf_output}" || true)"
if [[ "${needed_cuvs_count}" != 1 ]]; then
    echo "error: libcuml.so must declare exactly one libcuvs.so dependency; found ${needed_cuvs_count}." >&2
    exit 1
fi
if grep -Eq '/_deps/(cuvs|treelite|nvforest)[-/]' <<<"${readelf_output}"; then
    echo "error: libcuml.so contains a RUNPATH/RPATH into a private dependency build." >&2
    exit 1
fi
controlled_loader_path="${build_dir}:${cuvs_runtime_lib_dir}:${cudf_install_dir}/lib:${install_dir}/lib"
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    controlled_loader_path="${controlled_loader_path}:${LD_LIBRARY_PATH}"
fi
ldd_output="$(LD_LIBRARY_PATH="${controlled_loader_path}" ldd "${cuml_library}")" || {
    echo "error: ldd failed for '${cuml_library}'." >&2
    exit 1
}
if grep -q '=> not found' <<<"${ldd_output}"; then
    echo "error: libcuml.so has unresolved loader dependencies:" >&2
    grep '=> not found' <<<"${ldd_output}" >&2
    exit 1
fi
mapfile -t resolved_cuvs_paths < <(awk '$1 ~ /^libcuvs[.]so/ && $2 == "=>" { print $3 }' <<<"${ldd_output}")
if [[ "${#resolved_cuvs_paths[@]}" != 1 || "$(readlink -f "${resolved_cuvs_paths[0]}")" != "${cuvs_library_path}" ]]; then
    echo "error: libcuml.so did not resolve the canonical libcuvs.so '${cuvs_library_path}'." >&2
    printf '%s\n' "${ldd_output}" >&2
    exit 1
fi
rmm_library_path="$(readlink -f "${cudf_install_dir}/lib/librmm.so")"
mapfile -t resolved_rmm_paths < <(awk '$1 ~ /^librmm[.]so/ && $2 == "=>" { print $3 }' <<<"${ldd_output}")
if [[ "${#resolved_rmm_paths[@]}" != 1 || "$(readlink -f "${resolved_rmm_paths[0]}")" != "${rmm_library_path}" ]]; then
    echo "error: libcuml.so did not resolve the canonical RMM '${rmm_library_path}'." >&2
    printf '%s\n' "${ldd_output}" >&2
    exit 1
fi

system_needed='libstdc[+][+][.]so[.]6|libm[.]so[.]6|libgcc_s[.]so[.]1|libc[.]so[.]6|ld-linux-[a-z0-9_-]+[.]so[.][0-9]+'
# Forest inference loads libnvforest.so and libtreelite.so through the shim.
# Each library's NEEDED set is closed so OpenMP (libgomp) or a private
# dependency copy cannot enter the process through the cuML closure.
check_needed_allowlist() {
    local library="$1" allowed="$2" dynamic needed name openmp
    dynamic="$(readelf -d "${library}")"
    needed="$(sed -n 's/.*(NEEDED).*Shared library: \[\(.*\)\]$/\1/p' <<<"${dynamic}")"
    while IFS= read -r name; do
        if [[ -n "${name}" ]] && ! grep -Eqx "${allowed}|${system_needed}" <<<"${name}"; then
            echo "error: $(basename "${library}") declares NEEDED '${name}' outside its allowlist." >&2
            exit 1
        fi
    done <<<"${needed}"
    openmp="$(nm -D --undefined-only "${library}" | grep -E '[[:space:]](GOMP_|omp_)' || true)"
    if [[ -n "${openmp}" ]]; then
        echo "error: $(basename "${library}") imports OpenMP symbols:" >&2
        echo "${openmp}" >&2
        exit 1
    fi
}
check_needed_allowlist "${install_dir}/lib/libtreelite.so" ''
# The shim instantiates nvForest's importer from these headers; a copy in these
# initializers allocates on the legacy default stream, which the query domain
# refuses (nvForest #196 backport, scripts/build/patches/).
for expected in \
    "include/nvforest/decision_forest.hpp:nodes_{std::move(nodes)}" \
    "include/nvforest/forest_model.hpp:decision_forest_{std::move(forest)}"; do
    if ! grep -Fq "${expected#*:}" "${install_dir}/${expected%%:*}"; then
        echo "error: ${expected%%:*} copies forest buffers instead of moving them (nvForest #196 backport missing)." >&2
        exit 1
    fi
done
check_needed_allowlist "${install_dir}/lib/libnvforest.so" 'librmm[.]so'

crate_version="$(python3 - "${repo_root}/Cargo.toml" <<'PY'
import sys
import tomllib
with open(sys.argv[1], "rb") as source:
    print(tomllib.load(source)["workspace"]["package"]["version"])
PY
)"
metadata_dir="${install_dir}/share/algeon-cuml"
install -d --mode=0755 "${metadata_dir}"
printf '%s\n' "${crate_version}" >"${metadata_dir}/crate-version"
printf '%s\n' "DISABLE_OPENMP=ON" "CMAKE_CUDA_FLAGS=-Xcompiler=-Wno-unknown-pragmas" >"${metadata_dir}/openmp-disabled"
git -C "${source_tree}" rev-parse HEAD >"${metadata_dir}/source-revision"
for dependency in nvforest treelite; do
    git -C "${build_dir}/_deps/${dependency}-src" rev-parse HEAD >"${metadata_dir}/${dependency}-source-revision"
done

shim_source="${repo_root}/components/cuml/algeon-cuml-sys/native"
shim_build_dir="${repo_root}/target/native/cuml-rust-build"
shim_library="${install_dir}/lib/libcuml_rust.so"
shim_manifest="${install_dir}/share/cuml_rust/cargo-link.txt"
shim_cmake_args=(
    "-DCMAKE_INSTALL_PREFIX=${install_dir}"
    "-DCUML_INSTALL_DIR=${install_dir}"
    "-DCUDF_INSTALL_DIR=${cudf_install_dir}"
    "-DCUVS_INSTALL_DIR=${cuvs_install_dir}"
    "-DCUML_RUST_ENABLE_TEST_HOOKS=OFF"
)
if [[ -f "${shim_build_dir}/CMakeCache.txt" ]]; then
    cached_shim_source="$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "${shim_build_dir}/CMakeCache.txt")"
    if [[ -n "${cached_shim_source}" && "$(realpath -m -- "${cached_shim_source}")" != "$(realpath -m -- "${shim_source}")" ]]; then
        echo "error: cuML Rust shim build tree '${shim_build_dir}' belongs to '${cached_shim_source}', not '${shim_source}'." >&2
        echo "Select a matching build tree or move the stale build tree." >&2
        exit 1
    fi
fi
# Remove outputs that an earlier test-hooks build could have installed here.
rm -f -- "${shim_library}" "${install_dir}/include/cuml_rust/test_support.h"
cmake -S "${shim_source}" -B "${shim_build_dir}" "${shim_cmake_args[@]}"
shim_hooks_value="$(awk -F= '$1 == "CUML_RUST_ENABLE_TEST_HOOKS:BOOL" { print $2; found = 1 } END { if (!found) exit 1 }' "${shim_build_dir}/CMakeCache.txt")" || {
    echo "error: shim configure did not record CUML_RUST_ENABLE_TEST_HOOKS:BOOL." >&2
    exit 1
}
if [[ "${shim_hooks_value}" != OFF ]]; then
    echo "error: shim was configured with CUML_RUST_ENABLE_TEST_HOOKS=${shim_hooks_value}, expected OFF." >&2
    exit 1
fi
cmake --build "${shim_build_dir}" --parallel "${PARALLEL_LEVEL:-$(nproc)}"
cmake --install "${shim_build_dir}" --prefix "${install_dir}"
for required in "${shim_library}" "${shim_manifest}" "${install_dir}/include/cuml_rust/cuml_rust.h"; do
    [[ -f "${required}" ]] || {
        echo "error: production shim install is missing '${required}'." >&2
        exit 1
    }
done
if [[ -e "${install_dir}/include/cuml_rust/test_support.h" ]]; then
    echo "error: production shim prefix contains test-hooks header '${install_dir}/include/cuml_rust/test_support.h'." >&2
    exit 1
fi
shim_test_symbols="$(nm -D --defined-only "${shim_library}" | grep -E '[[:space:]]cuml_test_' || true)"
if [[ -n "${shim_test_symbols}" ]]; then
    echo "error: production libcuml_rust.so exports test-hooks symbols:" >&2
    echo "${shim_test_symbols}" >&2
    exit 1
fi
shim_readelf_output="$(readelf -d "${shim_library}")"
for dependency in libcuml.so libcudf_rust_rmm_interop.so libcuvs.so; do
    dependency_count="$(grep -Ec "NEEDED.*Shared library: \\[${dependency//./[.]}\\]" <<<"${shim_readelf_output}" || true)"
    if [[ "${dependency_count}" != 1 ]]; then
        echo "error: libcuml_rust.so must declare exactly one ${dependency} dependency; found ${dependency_count}." >&2
        exit 1
    fi
done
if grep -Eq '/_deps/(cuvs|treelite|nvforest)[-/]' <<<"${shim_readelf_output}"; then
    echo "error: libcuml_rust.so contains a RUNPATH/RPATH into a private dependency build." >&2
    exit 1
fi
shim_loader_path="${shim_build_dir}:${cuvs_runtime_lib_dir}:${cudf_install_dir}/lib:${install_dir}/lib"
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    shim_loader_path="${shim_loader_path}:${LD_LIBRARY_PATH}"
fi
shim_ldd_output="$(LD_LIBRARY_PATH="${shim_loader_path}" ldd "${shim_library}")" || {
    echo "error: ldd failed for '${shim_library}'." >&2
    exit 1
}
if grep -q '=> not found' <<<"${shim_ldd_output}"; then
    echo "error: libcuml_rust.so has unresolved loader dependencies:" >&2
    grep '=> not found' <<<"${shim_ldd_output}" >&2
    exit 1
fi
check_resolved_dependency() {
    local soname="$1" expected="$2" label="$3"
    mapfile -t resolved_paths < <(awk -v soname="${soname}" '$1 ~ ("^" soname "([.]|$)") && $2 == "=>" { print $3 }' <<<"${shim_ldd_output}")
    if [[ "${#resolved_paths[@]}" != 1 || "$(readlink -f "${resolved_paths[0]}")" != "$(readlink -f "${expected}")" ]]; then
        echo "error: libcuml_rust.so did not resolve canonical ${label} '${expected}'." >&2
        printf '%s\n' "${shim_ldd_output}" >&2
        exit 1
    fi
}
check_resolved_dependency libcuml.so "${cuml_library}" cuML
check_resolved_dependency libcudf_rust_rmm_interop.so "${cudf_install_dir}/lib/libcudf_rust_rmm_interop.so" "cuDF RMM interop"
check_resolved_dependency librmm.so "${cudf_install_dir}/lib/librmm.so" RMM
check_resolved_dependency libcuvs.so "${cuvs_library_path}" cuVS
check_resolved_dependency libnvforest.so "${install_dir}/lib/libnvforest.so" nvForest
check_resolved_dependency libtreelite.so "${install_dir}/lib/libtreelite.so" Treelite
check_needed_allowlist "${shim_library}" \
    'libcuml[.]so|libcuvs[.]so|libcudart[.]so[.][0-9]+|libcudf_rust_rmm_interop[.]so|librmm[.]so|libnvforest[.]so|libtreelite[.]so'
# nvForest and RMM export CUDA::cudart_static in their link interfaces; the
# shim must keep resolving the CUDA runtime from the shared libcudart.
shim_cuda_runtime="$(nm -D --defined-only "${shim_library}" | grep -E '[[:space:]]T cuda' || true)"
if [[ -n "${shim_cuda_runtime}" ]]; then
    echo "error: libcuml_rust.so defines CUDA runtime symbols from a static runtime:" >&2
    echo "${shim_cuda_runtime}" >&2
    exit 1
fi
# RMM's per-device resource map is a STB_GNU_UNIQUE header singleton. A private
# copy in a DSO that allocates through the default resource bypasses the query
# domain's memory manager.
for library in "${shim_library}" "${install_dir}/lib/libnvforest.so"; do
    dynamic_symbols="$(readelf --dyn-syms -W "${library}")"
    if ! grep -Eq 'OBJECT[[:space:]]+UNIQUE[[:space:]]+DEFAULT.*_ZGVZN3rmm[^ ]*get_ref_mapEvE' <<<"${dynamic_symbols}"; then
        echo "error: $(basename "${library}") does not export RMM's per-device resource map as STB_GNU_UNIQUE." >&2
        exit 1
    fi
    local_symbols="$(nm --defined-only "${library}" 2>/dev/null || true)"
    if [[ -z "${local_symbols}" ]]; then
        echo "error: $(basename "${library}") has no symbol table for the RMM singleton check." >&2
        exit 1
    fi
    private_rmm="$(grep -E '[[:space:]][bdt][[:space:]]_Z(Z|GVZ)N3rmm[^ ]*(get_ref_map|ref_map_lock|initial_resource)EvE' <<<"${local_symbols}" || true)"
    if [[ -n "${private_rmm}" ]]; then
        echo "error: $(basename "${library}") holds a private copy of RMM's per-device resource singleton:" >&2
        echo "${private_rmm}" >&2
        exit 1
    fi
done
if ! grep -Fxq "cargo:rustc-link-search=native=${install_dir}/lib" "${shim_manifest}" ||
   ! grep -Fxq "cargo:rustc-link-lib=dylib=cuml_rust" "${shim_manifest}" ||
   grep -Eq '@(CMAKE_INSTALL_PREFIX|CUML_INSTALL_DIR|CUDF_INSTALL_DIR|CUDAToolkit_LIBRARY_DIR)@' "${shim_manifest}"; then
    echo "error: shim cargo-link manifest is missing production link directives or contains unresolved paths." >&2
    while IFS= read -r line; do printf '%s\n' "${line}" >&2; done <"${shim_manifest}"
    exit 1
fi
readelf -d "${cuml_library}" | awk '/NEEDED/ { print }' >"${metadata_dir}/needed-libraries.txt"
architecture_contract="$(algeon_cuda_write_contract "${install_dir}" "${install_dir}/lib/libcuml.so" "${cuda_architectures}")"
for required in \
    "lib/libcuml.so" \
    "lib/libnvforest.so" \
    "lib/libtreelite.so" \
    "include/raft/core/handle.hpp" \
    "include/nvforest/decision_forest.hpp" \
    "include/nvforest/version_config.hpp" \
    "include/treelite/c_api.h" \
    "share/algeon-cuml/crate-version" \
    "share/algeon-cuml/openmp-disabled" \
    "share/algeon-cuml/source-revision" \
    "share/algeon-cuml/nvforest-source-revision" \
    "share/algeon-cuml/treelite-source-revision" \
    "share/algeon-cuml/needed-libraries.txt"; do
    [[ -f "${install_dir}/${required}" ]] || {
        echo "error: cuML install is missing '${required}'." >&2
        exit 1
    }
done
algeon_cuda_require_matching_contracts \
    "cuDF" "${cudf_install_dir}" "cuML" "${install_dir}" \
    "bash scripts/build/build_libcudf.sh && bash scripts/build/build_libcuvs.sh && bash scripts/build/build_libcuml.sh"

echo "canonical cuML prefix: ${install_dir}"
echo "RMM/rapids_logger: ${cudf_install_dir}; cuVS: ${cuvs_install_dir}"
echo "NEEDED libraries recorded at ${metadata_dir}/needed-libraries.txt"
echo "CUDA architectures: requested=${cuda_architectures}, real=$(algeon_cuda_read_contract_value "${architecture_contract}" real)"

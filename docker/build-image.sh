#!/usr/bin/env bash
# Build algeon_server, stage it with the native libraries it links into
# docker/dist/, validate the payload, and build the Docker image from it.
#
# The staged payload (the docker build context) holds:
#   bin/algeon_server                  the compiled server
#   lib/libcudf_rust*.so               cudf-sys CMake FFI bridges (OUT_DIR)
#   lib/libcudf.so librmm.so ...       pinned cuDF/RMM installation
#   lib/libcugraph*.so                 pinned cuGraph installation
#   lib/libcuvs.so* libcuvs_c.so*      canonical cuVS install (C API only for `cuvs`)
#   lib/libcuml.so* libcuml_rust.so*   canonical cuML install (for `cuml`)
#   lib/libtreelite.so* libnvforest.so* cuML native dependencies
#   licenses/                          license texts and INVENTORY.tsv (stage_licenses.py)
#   validate-loader-closure.sh         the closure check the Dockerfile reruns
#   payload-manifest.json SHA256SUMS   provenance and file hashes
#
# The Dockerfile harvests the CUDA toolkit libraries (libcudart, libcublas,
# libnvrtc, ...) and their license files from an nvidia/cuda runtime image, so
# they are not staged.
#
# Usage:
#   bash docker/build-image.sh                     # server.features in .config/policies/cpu-packaging.toml
#   FEATURES='' bash docker/build-image.sh         # lean cuDF-only image, no optional Cargo features
#   FEATURES=nvml bash docker/build-image.sh       # cuDF-only image with NVML device diagnostics
#   NO_STRIP=1 bash docker/build-image.sh          # keep debug symbols
#   SKIP_CARGO_BUILD=1 bash docker/build-image.sh  # reuse only a verified canonical artifact
#
# Other inputs: ALGEON_NATIVE_DIR (native install and build root, default
# target/native), the native build-tree selectors CUDF_CPP_BUILD_DIR,
# CUVS_CPP_BUILD_DIR, CUGRAPH_CMAKE_BUILD_DIR and CUML_CPP_BUILD_DIR,
# CPU_PACKAGE_TARGET_ROOT, CPU_PACKAGE_DIST_DIR, IMAGE_REPOSITORY,
# CUDA_RUNTIME_IMAGE and RUNTIME_BASE_IMAGE.
#
# Prerequisites: native artifacts and their build trees under
# ALGEON_NATIVE_DIR, docker, python3, readelf and ldd; patchelf and strip are
# recommended.

set -euo pipefail

if [[ "$#" -ne 0 ]]; then
    echo "ERROR: docker/build-image.sh accepts no positional or Cargo arguments; select only declared policy inputs via environment." >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DIST_DIR="${CPU_PACKAGE_DIST_DIR:-$SCRIPT_DIR/dist}"
POLICY_HELPER="$REPO_ROOT/scripts/build/cpu_packaging_policy.py"
CPU_PACKAGE_TARGET_ROOT="${CPU_PACKAGE_TARGET_ROOT:-$REPO_ROOT/target/cpu-packaging}"
SHIPPED_FEATURES="$(python3 "$POLICY_HELPER" shipped-features)"
# An unset FEATURES selects the shipped list; FEATURES='' selects no optional feature.
FEATURES="$(python3 "$POLICY_HELPER" normalize-features --features "${FEATURES-$SHIPPED_FEATURES}")"

eval "$(python3 "$POLICY_HELPER" describe)"

case "${SKIP_CARGO_BUILD-0}" in
    0 | 1) SKIP_CARGO_BUILD="${SKIP_CARGO_BUILD-0}" ;;
    *) echo "ERROR: SKIP_CARGO_BUILD must be 0 or 1, got '${SKIP_CARGO_BUILD}'." >&2; exit 2 ;;
esac

command -v docker >/dev/null 2>&1 || {
    echo "ERROR: docker is required to build the image." >&2
    exit 1
}

ALGEON_NATIVE_DIR="${ALGEON_NATIVE_DIR:-$REPO_ROOT/target/native}"
CUDF_INSTALL_DIR="$ALGEON_NATIVE_DIR/cudf-install"
CUVS_INSTALL_DIR="$ALGEON_NATIVE_DIR/cuvs-install"
CUML_INSTALL_DIR="$ALGEON_NATIVE_DIR/cuml-install"
CUGRAPH_INSTALL_DIR="$ALGEON_NATIVE_DIR/cugraph-install"
# The license sources of the fetched native dependencies live in the build
# trees; the selectors match scripts/build/build_lib*.sh.
CUDF_BUILD_DIR="${CUDF_CPP_BUILD_DIR:-$ALGEON_NATIVE_DIR/cudf-build}"
CUVS_BUILD_DIR="${CUVS_CPP_BUILD_DIR:-$ALGEON_NATIVE_DIR/cuvs-build}"
CUGRAPH_BUILD_DIR="${CUGRAPH_CMAKE_BUILD_DIR:-$ALGEON_NATIVE_DIR/cugraph-build}"
CUML_BUILD_DIR="${CUML_CPP_BUILD_DIR:-$ALGEON_NATIVE_DIR/cuml-build}"

# /app/lib is where the Dockerfile places the runtime libraries.
RUNTIME_LIB_DIR="/app/lib"

want_cugraph=0
want_cuvs=0
want_cuml=0
[[ ",$FEATURES," == *",cugraph,"* ]] && want_cugraph=1
[[ ",$FEATURES," == *",cuvs,"* ]] && want_cuvs=1
[[ ",$FEATURES," == *",cuml,"* ]] && want_cuml=1
want_cuvs_runtime=0
[[ $want_cugraph -eq 1 || $want_cuvs -eq 1 || $want_cuml -eq 1 ]] && want_cuvs_runtime=1

echo "==> FEATURES            : ${FEATURES:-(none)}   (cugraph=$want_cugraph cuvs=$want_cuvs cuml=$want_cuml)"
echo "==> ALGEON_NATIVE_DIR    : $ALGEON_NATIVE_DIR"

cd "$REPO_ROOT"

# 1. The helper rejects every ambient codegen/config override, starts Cargo
# from an isolated cwd and CARGO_HOME, and supplies the policy target and
# rustflags on the command line.
CARGO_BUILD_ROOT="$CPU_PACKAGE_TARGET_ROOT/$CPU_ARTIFACT_DIR"
BINARY="$CARGO_BUILD_ROOT/$CPU_TARGET_TRIPLE/$CPU_CARGO_PROFILE/algeon_server"
BUILD_RECORD="$BINARY.build-record.json"
CARGO_METADATA="$BINARY.cargo-metadata.json"

if [[ "$SKIP_CARGO_BUILD" == "0" ]]; then
    echo "==> Building canonical algeon_server artifact ..."
    python3 "$POLICY_HELPER" build \
        --features "$FEATURES" \
        --target-root "$CPU_PACKAGE_TARGET_ROOT" \
        --manifest "$REPO_ROOT/crates/algeon-server/Cargo.toml" \
        --explicit-env "ALGEON_NATIVE_DIR=$ALGEON_NATIVE_DIR"
else
    echo "==> SKIP_CARGO_BUILD=1; verifying canonical artifact before reuse"
    python3 "$POLICY_HELPER" verify-build \
        --features "$FEATURES" \
        --binary "$BINARY" \
        --record "$BUILD_RECORD"
fi

[[ -f "$BINARY" ]] || { echo "ERROR: canonical artifact $BINARY not found." >&2; exit 1; }
[[ -f "$BUILD_RECORD" ]] || { echo "ERROR: build record $BUILD_RECORD not found." >&2; exit 1; }

echo "==> Resetting $DIST_DIR ..."
rm -rf "$DIST_DIR"
mkdir -p "$DIST_DIR/bin" "$DIST_DIR/lib"
cp "$BINARY" "$DIST_DIR/bin/algeon_server"

# 2. The cudf-sys shared objects live in the CMake OUT_DIR that the binary's
# RUNPATH names.
RUNPATH=$(patchelf --print-rpath "$BINARY" 2>/dev/null \
    || readelf -d "$BINARY" 2>/dev/null \
        | awk -F'[][]+' '/\(RUNPATH\)|\(RPATH\)/{print $3}' \
    || true)

CUDF_RUST_SO=""
IFS=: read -ra RPATH_DIRS <<< "${RUNPATH:-}"
for dir in "${RPATH_DIRS[@]+"${RPATH_DIRS[@]}"}"; do
    [[ -f "$dir/libcudf_rust.so" ]] && { CUDF_RUST_SO="$dir/libcudf_rust.so"; break; }
done
if [[ -z "$CUDF_RUST_SO" ]]; then
    CUDF_RUST_SO=$(find "$CARGO_BUILD_ROOT/$CPU_TARGET_TRIPLE/$CPU_CARGO_PROFILE/build" \
        -name libcudf_rust.so -path "*/cudf-sys-*/out/lib/*" 2>/dev/null \
        | head -1)
fi
[[ -n "$CUDF_RUST_SO" ]] || { echo "ERROR: cannot locate libcudf_rust.so." >&2; exit 1; }
CUDF_RUST_LIB_DIR=$(dirname "$CUDF_RUST_SO")
CUDF_RUST_RMM_INTEROP_SO="$CUDF_RUST_LIB_DIR/libcudf_rust_rmm_interop.so"
[[ -f "$CUDF_RUST_RMM_INTEROP_SO" ]] || {
    echo "ERROR: cannot locate libcudf_rust_rmm_interop.so beside $CUDF_RUST_SO." >&2
    exit 1
}
echo "==> libcudf_rust.so     : $CUDF_RUST_SO"
cp -L "$CUDF_RUST_SO" "$DIST_DIR/lib/libcudf_rust.so"
# This private DT_NEEDED child must stay beside libcudf_rust.so for its $ORIGIN RUNPATH.
echo "==> libcudf_rust_rmm_interop.so: $CUDF_RUST_RMM_INTEROP_SO"
cp -L "$CUDF_RUST_RMM_INTEROP_SO" "$DIST_DIR/lib/libcudf_rust_rmm_interop.so"

# copy_lib <src-glob...> : copy each matching real file and preserve symlinks.
copy_lib() {
    local pat found=0 f
    for pat in "$@"; do
        for f in $pat; do
            [[ -e "$f" ]] || continue
            cp -a "$f" "$DIST_DIR/lib/"
            found=1
        done
    done
    return $((1 - found))
}

CUDF_LIB="$CUDF_INSTALL_DIR/lib"
[[ -d "$CUDF_LIB" ]] || { echo "ERROR: $CUDF_LIB not found: build cudf first." >&2; exit 1; }

# The cuVS, cuGraph and cuML build scripts reject a prefix whose record differs
# from cuDF's, so cuDF's record describes every staged layer.
source "$REPO_ROOT/scripts/build/cuda_architecture_contract.sh"
CUDA_CONTRACT="$(algeon_cuda_contract_path "$CUDF_INSTALL_DIR")"
[[ -f "$CUDA_CONTRACT" ]] || {
    echo "ERROR: $CUDA_CONTRACT not found; rebuild cuDF with scripts/build/build_libcudf.sh so the image can record its CUDA architectures." >&2
    exit 1
}
CUDA_ARCHITECTURES_REQUESTED="$(algeon_cuda_read_contract_value "$CUDA_CONTRACT" requested)"
CUDA_ARCHITECTURES_REAL="$(algeon_cuda_read_contract_value "$CUDA_CONTRACT" real)"
echo "==> CUDA architectures   : requested=$CUDA_ARCHITECTURES_REQUESTED real=$CUDA_ARCHITECTURES_REAL"
echo "==> Staging forked cuDF libs from $CUDF_LIB"
copy_lib "$CUDF_LIB/libcudf.so"          || { echo "ERROR: libcudf.so missing" >&2; exit 1; }
copy_lib "$CUDF_LIB/librmm.so"           || { echo "ERROR: librmm.so missing" >&2; exit 1; }
# The canonical cuDF build links KvikIO and nvCOMP statically: libcudf.so has
# no NEEDED entry for either and exports their symbols itself. A build that
# links them dynamically lists them as NEEDED, which the check below enforces.
copy_lib "$CUDF_LIB/libkvikio.so"*       || true
copy_lib "$CUDF_LIB/libnvcomp.so"*       || true
copy_lib "$CUDF_LIB/libnvcomp_cpu.so"*   || true
while IFS= read -r needed; do
    [[ -e "$DIST_DIR/lib/$needed" ]] || {
        echo "ERROR: $CUDF_LIB/libcudf.so needs $needed, but $CUDF_LIB does not provide it." >&2
        exit 1
    }
done < <(readelf -d "$CUDF_LIB/libcudf.so" \
    | awk -F'[][]' '/\(NEEDED\)/ && $2 ~ /^lib(kvikio|nvcomp)/ { print $2 }')

CUVS_LIB=""
if [[ $want_cuvs_runtime -eq 1 ]]; then
    for candidate in "$CUVS_INSTALL_DIR/lib" "$CUVS_INSTALL_DIR/lib64"; do
        if [[ -f "$candidate/libcuvs.so" ]]; then
            CUVS_LIB="$candidate"
            break
        fi
    done
    [[ -n "$CUVS_LIB" ]] || {
        echo "ERROR: canonical libcuvs.so not found under $CUVS_INSTALL_DIR/lib or lib64." >&2
        exit 1
    }
    echo "==> Staging canonical cuVS libs from $CUVS_LIB"
    copy_lib "$CUVS_LIB/libcuvs.so"* || { echo "ERROR: canonical libcuvs.so missing" >&2; exit 1; }
    if [[ $want_cuvs -eq 1 ]]; then
        copy_lib "$CUVS_LIB/libcuvs_c.so"* || { echo "ERROR: canonical libcuvs_c.so missing" >&2; exit 1; }
    fi
fi

resolve_needed_realpath() {
    local elf="$1" soname_prefix="$2" resolved
    resolved=$(ldd "$elf" 2>/dev/null | awk -v prefix="$soname_prefix" \
        '$1 ~ ("^" prefix "(\\.|$)") && $2 == "=>" && $3 ~ "^/" { print $3; exit }')
    [[ -n "$resolved" ]] || return 1
    readlink -f "$resolved"
}

if [[ $want_cugraph -eq 1 ]]; then
    CUGRAPH_LIB="$CUGRAPH_INSTALL_DIR/lib"
    [[ -d "$CUGRAPH_LIB" ]] || {
        echo "ERROR: $CUGRAPH_LIB not found; build cugraph first." >&2
        exit 1
    }
    # The installed library's RUNPATH names the cuVS and RMM prefixes it was
    # built against (scripts/build/build_libcugraph.sh), so this resolves
    # without a caller-provided LD_LIBRARY_PATH.
    GRAPH_CUVS_REALPATH=$(resolve_needed_realpath "$CUGRAPH_LIB/libcugraph.so" libcuvs) || {
        echo "ERROR: ldd could not resolve libcugraph.so to a canonical libcuvs.so; its RUNPATH must name the cuVS prefix (rebuild with scripts/build/build_libcugraph.sh)." >&2
        exit 1
    }
    CANONICAL_CUVS_REALPATH=$(readlink -f "$CUVS_LIB/libcuvs.so")
    [[ "$GRAPH_CUVS_REALPATH" != */_deps/* ]] || {
        echo "ERROR: libcugraph selected private cuVS artifact $GRAPH_CUVS_REALPATH." >&2
        exit 1
    }
    [[ "$GRAPH_CUVS_REALPATH" == "$CANONICAL_CUVS_REALPATH" ]] || {
        echo "ERROR: libcugraph and packaging selected different libcuvs.so identities:" >&2
        echo "  cuGraph:   $GRAPH_CUVS_REALPATH" >&2
        echo "  canonical: $CANONICAL_CUVS_REALPATH" >&2
        exit 1
    }
    echo "==> Staging forked cuGraph libs from $CUGRAPH_LIB"
    copy_lib "$CUGRAPH_LIB/libcugraph.so"        || { echo "ERROR: libcugraph.so missing" >&2; exit 1; }
    copy_lib "$CUGRAPH_LIB/libcugraph_c.so"      || { echo "ERROR: libcugraph_c.so missing" >&2; exit 1; }
    copy_lib "$CUGRAPH_LIB/libcugraph_common.so" || { echo "ERROR: libcugraph_common.so missing" >&2; exit 1; }
    copy_lib "$CUGRAPH_LIB/libcugraph_mg.so"     || { echo "ERROR: libcugraph_mg.so missing" >&2; exit 1; }
fi

if [[ $want_cuml -eq 1 ]]; then
    CUML_LIB="$CUML_INSTALL_DIR/lib"
    [[ -d "$CUML_LIB" ]] || { echo "ERROR: $CUML_LIB not found; build cuml first." >&2; exit 1; }
    ML_CUVS_REALPATH=$(resolve_needed_realpath "$CUML_LIB/libcuml.so" libcuvs) || {
        echo "ERROR: ldd could not resolve libcuml.so to a canonical libcuvs.so; its RUNPATH must name the cuVS prefix (rebuild with scripts/build/build_libcuml.sh)." >&2
        exit 1
    }
    CANONICAL_CUVS_REALPATH=$(readlink -f "$CUVS_LIB/libcuvs.so")
    [[ "$ML_CUVS_REALPATH" != */_deps/* && "$ML_CUVS_REALPATH" == "$CANONICAL_CUVS_REALPATH" ]] || {
        echo "ERROR: libcuml.so and packaging selected different libcuvs.so identities:" >&2
        echo "  cuML:      $ML_CUVS_REALPATH" >&2
        echo "  canonical: $CANONICAL_CUVS_REALPATH" >&2
        exit 1
    }
    echo "==> Staging canonical cuML libs from $CUML_LIB"
    for name in libcuml libcuml_rust libtreelite libnvforest; do
        copy_lib "$CUML_LIB/$name.so"* || { echo "ERROR: $name.so missing from $CUML_LIB" >&2; exit 1; }
    done
fi

# rapids_logger may be linked as a shared library, or absorbed into the
# canonical cuDF/RMM build. Stage it only when an artifact in this package has
# a real DT_NEEDED edge; an absorbed build must not acquire a stale second
# logger merely because one happens to exist in the install prefix.
first_elf_needing() {
    local soname_prefix="$1" elf
    while IFS= read -r elf; do
        if readelf -d "$elf" 2>/dev/null \
            | awk -F'[][]' -v prefix="$soname_prefix" \
                '/\(NEEDED\)/ && index($2, prefix) == 1 { found=1 } END { exit !found }'; then
            printf '%s\n' "$elf"
            return 0
        fi
    done < <(find "$DIST_DIR/bin" "$DIST_DIR/lib" -type f -print)
    return 1
}

RAPIDS_LOGGER_NEEDED_BY=$(first_elf_needing librapids_logger.so || true)
if [[ -n "$RAPIDS_LOGGER_NEEDED_BY" ]]; then
    echo "==> Staging shared rapids_logger required by $RAPIDS_LOGGER_NEEDED_BY"
    copy_lib "$CUDF_LIB/librapids_logger.so"* || {
        echo "ERROR: $RAPIDS_LOGGER_NEEDED_BY needs a shared librapids_logger.so, but the canonical cuDF install $CUDF_LIB does not ship it." >&2
        exit 1
    }
else
    echo "    note: rapids_logger has no shared DT_NEEDED edge (static/absorbed); not staging librapids_logger.so"
fi

# 3. RUNPATHs are rewritten to the runtime lib dir and $ORIGIN so no host build
# path reaches the image.
HAVE_PATCHELF=0; command -v patchelf >/dev/null 2>&1 && HAVE_PATCHELF=1
HAVE_STRIP=0;    command -v strip    >/dev/null 2>&1 && HAVE_STRIP=1

if [[ "${NO_STRIP:-0}" != "1" && $HAVE_STRIP -eq 1 ]]; then
    echo "==> Stripping staged artifacts ..."
    strip "$DIST_DIR/bin/algeon_server" 2>/dev/null || true
    for so in "$DIST_DIR"/lib/*.so*; do
        [[ -f "$so" && ! -L "$so" ]] && strip --strip-unneeded "$so" 2>/dev/null || true
    done
fi

if [[ $HAVE_PATCHELF -eq 1 ]]; then
    echo "==> Rewriting RUNPATHs -> $RUNTIME_LIB_DIR (binary), \$ORIGIN (libs)"
    patchelf --set-rpath "$RUNTIME_LIB_DIR" "$DIST_DIR/bin/algeon_server"
    for so in "$DIST_DIR"/lib/*.so*; do
        [[ -f "$so" && ! -L "$so" ]] && patchelf --set-rpath '$ORIGIN' "$so" 2>/dev/null || true
    done
else
    echo "warning: patchelf absent; relying on ENV LD_LIBRARY_PATH=$RUNTIME_LIB_DIR set by the image"
fi

# 4. The validator also ships in the payload so both Dockerfile stages rerun
# the same checks.
echo "==> Validating staged loader closure with readelf + ldd"
bash "$SCRIPT_DIR/validate-loader-closure.sh" \
    --bin "$DIST_DIR/bin/algeon_server" --lib "$DIST_DIR/lib" --features "$FEATURES"
cp "$SCRIPT_DIR/validate-loader-closure.sh" "$DIST_DIR/validate-loader-closure.sh"

# 5. The license directory covers every staged file except the CUDA toolkit
# libraries, whose licenses the Dockerfile copies from the runtime image.
license_layers=(--layer "cudf=$CUDF_BUILD_DIR")
[[ $want_cuvs_runtime -eq 1 ]] && license_layers+=(--layer "cuvs=$CUVS_BUILD_DIR")
[[ $want_cugraph -eq 1 ]] && license_layers+=(--layer "cugraph=$CUGRAPH_BUILD_DIR")
[[ $want_cuml -eq 1 ]] && license_layers+=(--layer "cuml=$CUML_BUILD_DIR")
echo "==> Staging license texts"
python3 "$SCRIPT_DIR/stage_licenses.py" \
    --repo-root "$REPO_ROOT" --dist-dir "$DIST_DIR" \
    --cargo-metadata "$CARGO_METADATA" --root-package algeon-server \
    "${license_layers[@]}"

python3 "$POLICY_HELPER" create-manifest --dist-dir "$DIST_DIR" --build-record "$BUILD_RECORD" \
    --cuda-architectures-requested "$CUDA_ARCHITECTURES_REQUESTED" \
    --cuda-architectures-real "$CUDA_ARCHITECTURES_REAL"
# COPY keeps context modes, and the image's root-owned /app must not inherit a
# group-writable host umask.
chmod -R u=rwX,go=rX "$DIST_DIR"
eval "$(python3 "$POLICY_HELPER" verify-manifest --dist-dir "$DIST_DIR")"
[[ "$PAYLOAD_WORKTREE_DIRTY" == "false" ]] \
    || echo "warning: the payload was built from a worktree with uncommitted changes; revision $PAYLOAD_REVISION does not describe it exactly."

echo ""
echo "==> Staged payload ($(du -sh "$DIST_DIR" | cut -f1) total):"
( cd "$DIST_DIR" && find . -path ./licenses -prune -o \( -type f -o -type l \) -print | sort | while read -r f; do
    printf '    %-46s %s\n' "$f" "$(du -h "$f" 2>/dev/null | cut -f1)"
  done
  printf '    %-46s %s\n' "./licenses/ ($(find licenses -type f | wc -l) files)" "$(du -sh licenses | cut -f1)" )
echo ""

# 6. Only the shipped feature set gets the plain <version>-<baseline> tag, so
# a lean build never overwrites the shipped image. An empty feature list is
# tagged -cudf.
IMAGE_TAG_SUFFIX="$PAYLOAD_VERSION-$CPU_BASELINE"
if [[ -z "$PAYLOAD_FEATURES" ]]; then
    IMAGE_TAG_SUFFIX="$IMAGE_TAG_SUFFIX-cudf"
elif [[ "$PAYLOAD_FEATURES" != "$SHIPPED_FEATURES" ]]; then
    IMAGE_TAG_SUFFIX="$IMAGE_TAG_SUFFIX-${PAYLOAD_FEATURES//,/-}"
fi
IMAGE_TAG="${IMAGE_REPOSITORY:-algeon-server}:$IMAGE_TAG_SUFFIX"
docker_args=(
    build
    -f "$SCRIPT_DIR/Dockerfile"
    --build-arg "CPU_BASELINE=$CPU_BASELINE"
    --build-arg "CPU_ARCH=$CPU_IMAGE_ARCH"
    --build-arg "CPU_TARGET=$CPU_TARGET_TRIPLE"
    --build-arg "CPU_BINARY_SHA256=$CPU_BINARY_SHA256"
    --build-arg "CPU_MANIFEST_SHA256=$CPU_MANIFEST_SHA256"
    --build-arg "CPU_CHECKSUMS_SHA256=$CPU_CHECKSUMS_SHA256"
    --build-arg "ALGEON_VERSION=$PAYLOAD_VERSION"
    --build-arg "ALGEON_REVISION=$PAYLOAD_REVISION"
    --build-arg "ALGEON_WORKTREE_DIRTY=$PAYLOAD_WORKTREE_DIRTY"
    --build-arg "ALGEON_SOURCE=$PAYLOAD_SOURCE"
    --build-arg "ALGEON_CREATED=$PAYLOAD_CREATED"
    --build-arg "ALGEON_FEATURES=$PAYLOAD_FEATURES"
    --build-arg "CUDA_TOOLKIT_VERSION=$PAYLOAD_CUDA_TOOLKIT"
    --build-arg "CUDA_ARCHITECTURES=$PAYLOAD_CUDA_ARCHITECTURES"
)
[[ -n "${CUDA_RUNTIME_IMAGE:-}" ]] && docker_args+=(--build-arg "CUDA_RUNTIME_IMAGE=$CUDA_RUNTIME_IMAGE")
[[ -n "${RUNTIME_BASE_IMAGE:-}" ]] && docker_args+=(--build-arg "RUNTIME_BASE_IMAGE=$RUNTIME_BASE_IMAGE")
docker_args+=(-t "$IMAGE_TAG" "$DIST_DIR")

echo "==> Building image: $IMAGE_TAG"
docker "${docker_args[@]}"
echo "==> SUCCESS: Docker image built: $IMAGE_TAG"

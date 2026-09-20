#!/usr/bin/env bash
# Build algeon_server and stage every runtime artifact the Docker
# image needs into docker/dist/.
#
# WHY THIS SCRIPT EXISTS
# ----------------------
# algeon-datafusion links against FORKED RAPIDS C++ libraries (cuDF, cuGraph,
# RMM, cuVS) that are built from the pinned monorepo sources. They are NOT the
# stock libraries shipped in the official `rapidsai/*` or `nvidia/cuda` images,
# so the image cannot pull them from a base layer: we must bundle the exact
# `.so` files this binary was linked against. This script collects them.
#
# It stages, into docker/dist/:
#   bin/algeon_server        the compiled server
#   lib/libcudf_rust*.so               cudf-sys CMake FFI bridges (OUT_DIR)
#   lib/libcudf.so librmm.so ...       forked RAPIDS C++ libs from cudf
#   lib/libcugraph*.so                 forked RAPIDS C++ libs from cugraph
#   lib/libcuvs.so* libcuvs_c.so*      canonical cuVS install (C API only for `cuvs`)
#
# The CUDA *toolkit* libraries (libcudart, libcublas, libnvrtc, ...) are NOT
# staged here: they are harvested from an official nvidia/cuda `-runtime`
# image inside the Dockerfile, since they are unmodified and version-matched to
# the base image.
#
# Prerequisites:
#   - Monorepo native artifacts built under target/native, or selected through
#     the explicit native environment variables below.
#   - `cargo build --release` succeeds for the chosen FEATURES
#   - patchelf (recommended; rewrites RUNPATHs so the container is hermetic)
#
# Usage:
#   bash docker/stage-docker-payload.sh                  # cugraph,cuvs,nvml
#   FEATURES=nvml bash docker/stage-docker-payload.sh    # lean cudf-only (~1/3 size)
#   NO_STRIP=1 bash docker/stage-docker-payload.sh            # keep debug symbols
#   SKIP_CARGO_BUILD=1 bash docker/stage-docker-payload.sh    # reuse only verified canonical artifact
#
# Then:
#   bash docker/build-image.sh
#
# The image is feature-agnostic: it bundles whatever this script staged and
# harvests only the CUDA libs the binary actually links. A lean cuDF-only
# build therefore yields a ~3x smaller image automatically: no extra flag.

set -euo pipefail

if [[ "$#" -ne 0 ]]; then
    echo "ERROR: docker/stage-docker-payload.sh accepts no positional or Cargo arguments; select only declared policy inputs via environment." >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DIST_DIR="${CPU_PACKAGE_DIST_DIR:-$SCRIPT_DIR/dist}"
POLICY_HELPER="$REPO_ROOT/scripts/build/cpu_packaging_policy.py"
CPU_PACKAGE_TARGET_ROOT="${CPU_PACKAGE_TARGET_ROOT:-$REPO_ROOT/target/cpu-packaging}"
FEATURES="$(python3 "$POLICY_HELPER" normalize-features \
    --features "${FEATURES:-cugraph,cuvs,nvml}")"

eval "$(python3 "$POLICY_HELPER" describe)"

case "${SKIP_CARGO_BUILD-0}" in
    0 | 1) SKIP_CARGO_BUILD="${SKIP_CARGO_BUILD-0}" ;;
    *) echo "ERROR: SKIP_CARGO_BUILD must be 0 or 1, got '${SKIP_CARGO_BUILD}'." >&2; exit 2 ;;
esac

ALGEON_NATIVE_DIR="${ALGEON_NATIVE_DIR:-$REPO_ROOT/target/native}"
CUDF_INSTALL_DIR="$ALGEON_NATIVE_DIR/cudf-install"
CUVS_INSTALL_DIR="$ALGEON_NATIVE_DIR/cuvs-install"
CUGRAPH_INSTALL_DIR="$ALGEON_NATIVE_DIR/cugraph-install"

# /app/lib is where the Dockerfile places the runtime libraries.
RUNTIME_LIB_DIR="/app/lib"

want_cugraph=0
want_cuvs=0
[[ ",$FEATURES," == *",cugraph,"* ]] && want_cugraph=1
[[ ",$FEATURES," == *",cuvs,"* ]] && want_cuvs=1
want_cuvs_runtime=0
[[ $want_cugraph -eq 1 || $want_cuvs -eq 1 ]] && want_cuvs_runtime=1

echo "==> FEATURES            : $FEATURES   (cugraph=$want_cugraph cuvs=$want_cuvs)"
echo "==> ALGEON_NATIVE_DIR    : $ALGEON_NATIVE_DIR"

cd "$REPO_ROOT"

# ── 0. Compile or verify canonical artifact ─────────────────────────────────
# The helper rejects every ambient codegen/config override, starts Cargo from
# an isolated cwd/CARGO_HOME, and supplies the policy target + rustflags on the
# command line.
CARGO_BUILD_ROOT="$CPU_PACKAGE_TARGET_ROOT/$CPU_ARTIFACT_DIR"
BINARY="$CARGO_BUILD_ROOT/$CPU_TARGET_TRIPLE/$CPU_CARGO_PROFILE/algeon_server"
BUILD_RECORD="$BINARY.build-record.json"

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

# ── 2. Reset staging dir ────────────────────────────────────────────────────
echo "==> Resetting $DIST_DIR ..."
rm -rf "$DIST_DIR"
mkdir -p "$DIST_DIR/bin" "$DIST_DIR/lib"

# ── 3. Stage the binary ─────────────────────────────────────────────────────
cp "$BINARY" "$DIST_DIR/bin/algeon_server"

# ── 4. Locate + stage cudf-sys runtime shared objects ───────────────────────
# They live in the cudf-sys CMake OUT_DIR, referenced by the binary's RUNPATH.
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

# ── 5. Stage forked cuDF / RMM libs ─────────────────────────────────────────
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
echo "==> Staging forked cuDF libs from $CUDF_LIB"
copy_lib "$CUDF_LIB/libcudf.so"          || { echo "ERROR: libcudf.so missing" >&2; exit 1; }
copy_lib "$CUDF_LIB/librmm.so"           || { echo "ERROR: librmm.so missing" >&2; exit 1; }
# Optional-but-bundled: kvikio (remote/GDS reads) + nvcomp (parquet codecs).
# These are dlopen'd by libcudf, so they never show up in ldd: bundle eagerly.
copy_lib "$CUDF_LIB/libkvikio.so"        || echo "    note: libkvikio.so absent (remote-direct reads unavailable)"
copy_lib "$CUDF_LIB/libnvcomp.so"*       || echo "    note: libnvcomp.so absent (GPU parquet codecs may be unavailable)"
copy_lib "$CUDF_LIB/libnvcomp_cpu.so"*   || true

# ── 6. Stage canonical cuVS and optional forked cuGraph libraries ────────────
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
    GRAPH_CUVS_REALPATH=$(resolve_needed_realpath "$CUGRAPH_LIB/libcugraph.so" libcuvs) || {
        echo "ERROR: ldd could not resolve libcugraph.so to a canonical libcuvs.so." >&2
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

# ── 7. Strip + rehome RUNPATHs ──────────────────────────────────────────────
# Strip removes ~35-40% of each artifact's size; safe for release .so/bin.
# Rehome every RUNPATH to point only at the runtime lib dir + $ORIGIN, erasing
# the stale host build paths so the image is hermetic and LD_LIBRARY_PATH-free.
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

# ── 8. Fail-closed native loader closure validation ─────────────────────────
validate_loader_closure() {
    command -v readelf >/dev/null 2>&1 || { echo "ERROR: readelf is required for loader validation." >&2; return 1; }
    command -v ldd >/dev/null 2>&1 || { echo "ERROR: ldd is required for loader validation." >&2; return 1; }

    local elf dynamic needed ldd_output unresolved dependency resolved realpath
    local -A cuvs_realpaths=() rmm_realpaths=() logger_realpaths=() cudart_realpaths=()
    while IFS= read -r elf; do
        readelf -h "$elf" >/dev/null 2>&1 || {
            echo "ERROR: staged runtime artifact is not a readable ELF file: $elf" >&2
            return 1
        }
        dynamic=$(readelf -d "$elf" 2>&1) || {
            echo "ERROR: cannot inspect ELF dynamic section: $elf" >&2
            echo "$dynamic" >&2
            return 1
        }
        while IFS= read -r needed; do
            case "$needed" in
                libraft.so*)
                    echo "ERROR: unexpected shared RAFT dependency $needed in $elf; RAFT must remain header-only." >&2
                    return 1
                    ;;
                libcudart.so*)
                    [[ "$needed" == "libcudart.so.13" ]] || {
                        echo "ERROR: unexpected CUDA runtime dependency $needed in $elf; this package targets CUDA 13." >&2
                        return 1
                    }
                    ;;
            esac
        done < <(awk -F'[][]' '/\(NEEDED\)/ { print $2 }' <<< "$dynamic")

        ldd_output=$(LD_LIBRARY_PATH="$DIST_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ldd "$elf" 2>&1) || {
            echo "ERROR: ldd failed for staged ELF $elf" >&2
            echo "$ldd_output" >&2
            return 1
        }
        # NVIDIA's libnvtx3interop uses DT_AUXILIARY [visibility=hidden].
        # glibc ldd renders that marker as a fake `=> not found` row even
        # though it is not a DT_NEEDED soname. Keep every real missing edge.
        unresolved=$(awk \
            '$1 != "visibility=hidden" && $2 == "=>" && $3 == "not" && $4 == "found"' \
            <<< "$ldd_output")
        if [[ -n "$unresolved" ]]; then
            echo "ERROR: unresolved loader dependency for staged ELF $elf" >&2
            echo "$unresolved" >&2
            return 1
        fi
        while read -r dependency _ resolved _; do
            [[ "$resolved" == /* ]] || continue
            realpath=$(readlink -f "$resolved") || {
                echo "ERROR: cannot canonicalize resolved dependency $resolved from $elf" >&2
                return 1
            }
            case "$dependency" in
                libcuvs.so*) cuvs_realpaths["$realpath"]=1 ;;
                librmm.so*) rmm_realpaths["$realpath"]=1 ;;
                librapids_logger.so*) logger_realpaths["$realpath"]=1 ;;
                libcudart.so*) cudart_realpaths["$realpath"]=1 ;;
                libraft.so*)
                    echo "ERROR: unexpected shared RAFT dependency resolved for $elf: $realpath" >&2
                    return 1
                    ;;
            esac
        done <<< "$ldd_output"
    done < <(find "$DIST_DIR/bin" "$DIST_DIR/lib" -type f -print)

    if (( ${#rmm_realpaths[@]} != 1 )); then
        echo "ERROR: loader closure must resolve exactly one librmm.so identity; found ${#rmm_realpaths[@]}." >&2
        printf '  %s\n' "${!rmm_realpaths[@]}" >&2
        return 1
    fi
    local staged_rmm
    staged_rmm=$(readlink -f "$DIST_DIR/lib/librmm.so")
    [[ -n "${rmm_realpaths[$staged_rmm]+present}" ]] || {
        echo "ERROR: loader closure selected librmm.so outside the staged canonical identity $staged_rmm." >&2
        return 1
    }

    if (( ${#logger_realpaths[@]} > 1 )); then
        echo "ERROR: loader closure resolves multiple librapids_logger.so identities." >&2
        printf '  %s\n' "${!logger_realpaths[@]}" >&2
        return 1
    fi
    local -A staged_logger_realpaths=()
    while IFS= read -r dependency; do
        realpath=$(readlink -f "$dependency") || {
            echo "ERROR: cannot canonicalize staged rapids_logger $dependency" >&2
            return 1
        }
        staged_logger_realpaths["$realpath"]=1
    done < <(find "$DIST_DIR/lib" -maxdepth 1 -name 'librapids_logger.so*' -print)
    if (( ${#logger_realpaths[@]} == 1 )); then
        (( ${#staged_logger_realpaths[@]} == 1 )) || {
            echo "ERROR: loader closure resolved a shared librapids_logger.so without one canonical staged identity." >&2
            return 1
        }
        for realpath in "${!logger_realpaths[@]}"; do
            [[ -n "${staged_logger_realpaths[$realpath]+present}" ]] || {
                echo "ERROR: loader closure selected librapids_logger.so outside the staged canonical identity." >&2
                echo "  resolved: $realpath" >&2
                printf '  staged:   %s\n' "${!staged_logger_realpaths[@]}" >&2
                return 1
            }
        done
    elif (( ${#staged_logger_realpaths[@]} != 0 )); then
        echo "ERROR: staged librapids_logger.so is absent from the loader closure; refusing an unneeded or stale shared logger." >&2
        return 1
    fi

    if [[ $want_cuvs_runtime -eq 1 ]]; then
        if (( ${#cuvs_realpaths[@]} != 1 )); then
            echo "ERROR: loader closure must resolve exactly one libcuvs.so identity; found ${#cuvs_realpaths[@]}." >&2
            printf '  %s\n' "${!cuvs_realpaths[@]}" >&2
            return 1
        fi
        local staged_cuvs
        staged_cuvs=$(readlink -f "$DIST_DIR/lib/libcuvs.so")
        [[ -n "${cuvs_realpaths[$staged_cuvs]+present}" ]] || {
            echo "ERROR: loader closure selected libcuvs.so outside the staged canonical identity $staged_cuvs." >&2
            return 1
        }
    elif (( ${#cuvs_realpaths[@]} != 0 )); then
        echo "ERROR: libcuvs.so entered a package with neither cugraph nor cuvs enabled." >&2
        return 1
    fi

    if [[ $want_cuvs -eq 1 && ! -e "$DIST_DIR/lib/libcuvs_c.so" ]]; then
        echo "ERROR: cuvs feature package is missing staged libcuvs_c.so." >&2
        return 1
    fi
    if (( ${#cudart_realpaths[@]} > 1 )); then
        echo "ERROR: loader closure resolves multiple CUDA runtime identities." >&2
        printf '  %s\n' "${!cudart_realpaths[@]}" >&2
        return 1
    fi
}

echo "==> Validating staged loader closure with readelf + ldd"
validate_loader_closure
echo "OK: staged loader closure passed."

# ── 9. Manifest ─────────────────────────────────────────────────────────────
python3 "$POLICY_HELPER" create-manifest \
    --dist-dir "$DIST_DIR" \
    --build-record "$BUILD_RECORD" \
    --output "$DIST_DIR/payload-manifest.json"

python3 "$POLICY_HELPER" verify-manifest \
    --dist-dir "$DIST_DIR" \
    --manifest "$DIST_DIR/payload-manifest.json" >/dev/null

echo ""
echo "==> Staged payload ($(du -sh "$DIST_DIR" | cut -f1) total):"
( cd "$DIST_DIR" && find . -type f -o -type l | sort | while read -r f; do
    printf '    %-46s %s\n' "$f" "$(du -h "$f" 2>/dev/null | cut -f1)"
  done )
echo ""
IMAGE_REPOSITORY="${IMAGE_REPOSITORY:-algeon-server}"
IMAGE_TAG="$IMAGE_REPOSITORY:$CPU_BASELINE"
echo "==> SUCCESS: staging and validation passed."
echo "==> NEXT: bash docker/build-image.sh  ($IMAGE_TAG)"

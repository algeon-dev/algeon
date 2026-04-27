#!/usr/bin/env bash
# Build datafusion_nexus_server and stage every runtime artifact the Docker
# image needs into docker/dist/.
#
# WHY THIS SCRIPT EXISTS
# ----------------------
# datafusion-nexus links against FORKED RAPIDS C++ libraries (cuDF, cuGraph,
# RMM, cuVS) that are built locally from sibling checkouts. They are NOT the
# stock libraries shipped in the official `rapidsai/*` or `nvidia/cuda` images,
# so the image cannot pull them from a base layer — we must bundle the exact
# `.so` files this binary was linked against. This script collects them.
#
# It stages, into docker/dist/:
#   bin/datafusion_nexus_server        the compiled server
#   lib/libcudf_rust.so                cudf-nexus-sys CMake FFI bridge (OUT_DIR)
#   lib/libcudf.so librmm.so ...       forked RAPIDS C++ libs from cudf-nexus
#   lib/libcugraph*.so libcuvs.so      forked RAPIDS C++ libs from cugraph-nexus
#                                      (only when FEATURES contains `cugraph`)
#
# The CUDA *toolkit* libraries (libcudart, libcublas, libnvrtc, ...) are NOT
# staged here — they are harvested from an official nvidia/cuda `-runtime`
# image inside the Dockerfile, since they are unmodified and version-matched to
# the base image.
#
# Prerequisites:
#   - Sibling path deps resolved and BUILT:
#       ../cudf-nexus/cudf-install/lib   (forked libcudf + friends)
#       ../cugraph-nexus/cugraph/cpp/build (forked libcugraph + cuvs, if cugraph)
#   - `cargo build --release` succeeds for the chosen FEATURES
#   - patchelf (recommended; rewrites RUNPATHs so the container is hermetic)
#
# Usage:
#   bash docker/build.sh                       # server,iceberg,cugraph,nvml (full)
#   FEATURES=server,iceberg,nvml bash docker/build.sh # lean cudf-only (~1/3 size)
#   NO_STRIP=1 bash docker/build.sh            # keep debug symbols
#   SKIP_CARGO_BUILD=1 bash docker/build.sh    # reuse only verified canonical artifact
#   CPU_PACKAGE_PROFILE=tuned-local-v3 bash docker/build.sh # separate v3 artifact/tag
#
# Then:
#   Run the exact docker build command printed by this script. Its build args
#   bind the image labels to the staged provenance.
#
# The image is feature-agnostic: it bundles whatever this script staged and
# harvests only the CUDA libs the binary actually links. A lean cuDF-only
# build therefore yields a ~3x smaller image automatically — no extra flag.

set -euo pipefail

if [[ "$#" -ne 0 ]]; then
    echo "ERROR: docker/build.sh accepts no positional or Cargo arguments; select only declared policy inputs via environment." >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DIST_DIR="${CPU_PACKAGE_DIST_DIR:-$SCRIPT_DIR/dist}"
POLICY_FILE="$REPO_ROOT/.config/policies/cpu-packaging.toml"
POLICY_HELPER="$REPO_ROOT/scripts/build/cpu_packaging_policy.py"
CPU_PACKAGE_PROFILE="${CPU_PACKAGE_PROFILE:-portable}"
CPU_PACKAGE_TARGET_ROOT="${CPU_PACKAGE_TARGET_ROOT:-$REPO_ROOT/target/cpu-packaging}"
FEATURES="$(python3 "$POLICY_HELPER" --policy "$POLICY_FILE" normalize-features \
    --features "${FEATURES:-server,iceberg,cugraph,nvml}")"

eval "$(python3 "$POLICY_HELPER" --policy "$POLICY_FILE" describe --profile "$CPU_PACKAGE_PROFILE")"

case "${SKIP_CARGO_BUILD-0}" in
    0 | 1) SKIP_CARGO_BUILD="${SKIP_CARGO_BUILD-0}" ;;
    *) echo "ERROR: SKIP_CARGO_BUILD must be 0 or 1, got '${SKIP_CARGO_BUILD}'." >&2; exit 2 ;;
esac

python3 "$POLICY_HELPER" --policy "$POLICY_FILE" check-env --profile "$CPU_PACKAGE_PROFILE"

# Sibling forked-lib locations (override via env if your layout differs).
CUDF_INSTALL_DIR="${CUDF_INSTALL_DIR:-$REPO_ROOT/../cudf-nexus/cudf-install}"
CUGRAPH_BUILD_DIR="${CUGRAPH_BUILD_DIR:-$REPO_ROOT/../cugraph-nexus/cugraph/cpp/build}"

# /app/lib is where the Dockerfile places the runtime libraries.
RUNTIME_LIB_DIR="/app/lib"

want_cugraph=0
[[ ",$FEATURES," == *",cugraph,"* ]] && want_cugraph=1

echo "==> FEATURES            : $FEATURES   (cugraph=$want_cugraph)"
echo "==> CPU package profile : $CPU_POLICY_PROFILE ($CPU_PROFILE_KIND)"
echo "==> CPU baseline        : $CPU_BASELINE"
echo "==> CPU target          : $CPU_TARGET_TRIPLE"
echo "==> CPU rustflags       : $CPU_RUSTFLAGS_JSON"
echo "==> Deployment requires : $CPU_DEPLOYMENT_REQUIREMENT"
echo "==> CUDF_INSTALL_DIR    : $CUDF_INSTALL_DIR"
[[ $want_cugraph -eq 1 ]] && echo "==> CUGRAPH_BUILD_DIR   : $CUGRAPH_BUILD_DIR"

cd "$REPO_ROOT"

# ── 0. Compile or verify canonical artifact ─────────────────────────────────
# The helper rejects every ambient codegen/config override, starts Cargo from
# an isolated cwd/CARGO_HOME, and supplies the policy target + rustflags on the
# command line. Profiles have separate target roots, so tuned artifacts cannot
# masquerade as portable ones.
CARGO_BUILD_ROOT="$CPU_PACKAGE_TARGET_ROOT/$CPU_ARTIFACT_DIR"
BINARY="$CARGO_BUILD_ROOT/$CPU_TARGET_TRIPLE/$CPU_CARGO_PROFILE/datafusion_nexus_server"
BUILD_PROVENANCE="$BINARY.cpu-provenance.json"

if [[ "$SKIP_CARGO_BUILD" == "0" ]]; then
    echo "==> Building canonical datafusion_nexus_server artifact ..."
    python3 "$POLICY_HELPER" --policy "$POLICY_FILE" build \
        --profile "$CPU_PACKAGE_PROFILE" \
        --features "$FEATURES" \
        --target-root "$CPU_PACKAGE_TARGET_ROOT" \
        --manifest "$REPO_ROOT/Cargo.toml" \
        --explicit-env "CUDF_INSTALL_DIR=$CUDF_INSTALL_DIR" \
        --explicit-env "CUGRAPH_BUILD_DIR=$CUGRAPH_BUILD_DIR"
else
    echo "==> SKIP_CARGO_BUILD=1 — verifying canonical artifact before reuse"
    python3 "$POLICY_HELPER" --policy "$POLICY_FILE" verify-build \
        --profile "$CPU_PACKAGE_PROFILE" \
        --features "$FEATURES" \
        --binary "$BINARY" \
        --provenance "$BUILD_PROVENANCE"
fi

[[ -f "$BINARY" ]] || { echo "ERROR: canonical artifact $BINARY not found." >&2; exit 1; }
[[ -f "$BUILD_PROVENANCE" ]] || { echo "ERROR: build provenance $BUILD_PROVENANCE not found." >&2; exit 1; }

# ── 2. Reset staging dir ────────────────────────────────────────────────────
echo "==> Resetting $DIST_DIR ..."
rm -rf "$DIST_DIR"
mkdir -p "$DIST_DIR/bin" "$DIST_DIR/lib"

# ── 3. Stage the binary ─────────────────────────────────────────────────────
cp "$BINARY" "$DIST_DIR/bin/datafusion_nexus_server"

# ── 4. Locate + stage libcudf_rust.so (cudf-nexus-sys CMake FFI bridge) ───────────
# It lives in the cudf-nexus-sys CMake OUT_DIR, referenced by the binary's RUNPATH.
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
        -name libcudf_rust.so -path "*/cudf-nexus-sys-*/out/lib/*" 2>/dev/null \
        | head -1)
fi
[[ -n "$CUDF_RUST_SO" ]] || { echo "ERROR: cannot locate libcudf_rust.so." >&2; exit 1; }
echo "==> libcudf_rust.so     : $CUDF_RUST_SO"
cp -L "$CUDF_RUST_SO" "$DIST_DIR/lib/libcudf_rust.so"

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
[[ -d "$CUDF_LIB" ]] || { echo "ERROR: $CUDF_LIB not found — build cudf-nexus first." >&2; exit 1; }
echo "==> Staging forked cuDF libs from $CUDF_LIB"
copy_lib "$CUDF_LIB/libcudf.so"          || { echo "ERROR: libcudf.so missing" >&2; exit 1; }
copy_lib "$CUDF_LIB/librmm.so"           || { echo "ERROR: librmm.so missing" >&2; exit 1; }
copy_lib "$CUDF_LIB/librapids_logger.so" || { echo "ERROR: librapids_logger.so missing" >&2; exit 1; }
# Optional-but-bundled: kvikio (remote/GDS reads) + nvcomp (parquet codecs).
# These are dlopen'd by libcudf, so they never show up in ldd — bundle eagerly.
copy_lib "$CUDF_LIB/libkvikio.so"        || echo "    note: libkvikio.so absent (remote-direct reads unavailable)"
copy_lib "$CUDF_LIB/libnvcomp.so"*       || echo "    note: libnvcomp.so absent (GPU parquet codecs may be unavailable)"
copy_lib "$CUDF_LIB/libnvcomp_cpu.so"*   || true

# ── 6. Stage forked cuGraph / cuVS libs (only with cugraph feature) ─────────
if [[ $want_cugraph -eq 1 ]]; then
    [[ -d "$CUGRAPH_BUILD_DIR" ]] || { echo "ERROR: $CUGRAPH_BUILD_DIR not found — build cugraph-nexus first." >&2; exit 1; }
    echo "==> Staging forked cuGraph libs from $CUGRAPH_BUILD_DIR"
    copy_lib "$CUGRAPH_BUILD_DIR/libcugraph.so"        || { echo "ERROR: libcugraph.so missing" >&2; exit 1; }
    copy_lib "$CUGRAPH_BUILD_DIR/libcugraph_c.so"      || { echo "ERROR: libcugraph_c.so missing" >&2; exit 1; }
    copy_lib "$CUGRAPH_BUILD_DIR/libcugraph_common.so" || { echo "ERROR: libcugraph_common.so missing" >&2; exit 1; }
    copy_lib "$CUGRAPH_BUILD_DIR/libcugraph_mg.so"     || { echo "ERROR: libcugraph_mg.so missing" >&2; exit 1; }
    copy_lib "$CUGRAPH_BUILD_DIR/_deps/cuvs-build/libcuvs.so" \
        || copy_lib "$CUGRAPH_BUILD_DIR/libcuvs.so" \
        || { echo "ERROR: libcuvs.so missing" >&2; exit 1; }
fi

# ── 7. Strip + rehome RUNPATHs ──────────────────────────────────────────────
# Strip removes ~35-40% of each artifact's size; safe for release .so/bin.
# Rehome every RUNPATH to point only at the runtime lib dir + $ORIGIN, erasing
# the stale host build paths so the image is hermetic and LD_LIBRARY_PATH-free.
HAVE_PATCHELF=0; command -v patchelf >/dev/null 2>&1 && HAVE_PATCHELF=1
HAVE_STRIP=0;    command -v strip    >/dev/null 2>&1 && HAVE_STRIP=1

if [[ "${NO_STRIP:-0}" != "1" && $HAVE_STRIP -eq 1 ]]; then
    echo "==> Stripping staged artifacts ..."
    strip "$DIST_DIR/bin/datafusion_nexus_server" 2>/dev/null || true
    for so in "$DIST_DIR"/lib/*.so*; do
        [[ -f "$so" && ! -L "$so" ]] && strip --strip-unneeded "$so" 2>/dev/null || true
    done
fi

if [[ $HAVE_PATCHELF -eq 1 ]]; then
    echo "==> Rewriting RUNPATHs -> $RUNTIME_LIB_DIR (binary), \$ORIGIN (libs)"
    patchelf --set-rpath "$RUNTIME_LIB_DIR" "$DIST_DIR/bin/datafusion_nexus_server"
    for so in "$DIST_DIR"/lib/*.so*; do
        [[ -f "$so" && ! -L "$so" ]] && patchelf --set-rpath '$ORIGIN' "$so" 2>/dev/null || true
    done
else
    echo "warning: patchelf absent — relying on ENV LD_LIBRARY_PATH=$RUNTIME_LIB_DIR set by the image"
fi

# ── 8. Provenance + manifest ────────────────────────────────────────────────
python3 "$POLICY_HELPER" --policy "$POLICY_FILE" stage \
    --profile "$CPU_PACKAGE_PROFILE" \
    --features "$FEATURES" \
    --source-binary "$BINARY" \
    --staged-binary "$DIST_DIR/bin/datafusion_nexus_server" \
    --build-provenance "$BUILD_PROVENANCE" \
    --output "$DIST_DIR/cpu-provenance.json" \
    --labels-output "$DIST_DIR/cpu-provenance.labels"

CPU_BINARY_SHA256=$(awk -F= '$1 == "binary_sha256" { print $2 }' "$DIST_DIR/cpu-provenance.labels")
CPU_PROVENANCE_SHA256=$(sha256sum "$DIST_DIR/cpu-provenance.json" | awk '{print $1}')
python3 "$POLICY_HELPER" verify-image-inputs \
    --provenance "$DIST_DIR/cpu-provenance.json" \
    --labels "$DIST_DIR/cpu-provenance.labels" \
    --baseline "$CPU_BASELINE" \
    --arch "$CPU_IMAGE_ARCH" \
    --profile "$CPU_POLICY_PROFILE" \
    --profile-kind "$CPU_PROFILE_KIND" \
    --target "$CPU_TARGET_TRIPLE" \
    --binary-sha256 "$CPU_BINARY_SHA256" \
    --provenance-sha256 "$CPU_PROVENANCE_SHA256"

echo ""
echo "==> Staged payload ($(du -sh "$DIST_DIR" | cut -f1) total):"
( cd "$DIST_DIR" && find . -type f -o -type l | sort | while read -r f; do
    printf '    %-46s %s\n' "$f" "$(du -h "$f" 2>/dev/null | cut -f1)"
  done )
echo ""
IMAGE_REPOSITORY="${IMAGE_REPOSITORY:-datafusion-nexus-server}"
IMAGE_TAG="$IMAGE_REPOSITORY:$CPU_IMAGE_TAG_SUFFIX"
echo "==> Canonical Docker command (labels and hashes must match provenance):"
printf '    docker build'
printf ' --build-arg %q' "CPU_BASELINE=$CPU_BASELINE"
printf ' --build-arg %q' "CPU_ARCH=$CPU_IMAGE_ARCH"
printf ' --build-arg %q' "CPU_PROFILE=$CPU_POLICY_PROFILE"
printf ' --build-arg %q' "CPU_PROFILE_KIND=$CPU_PROFILE_KIND"
printf ' --build-arg %q' "CPU_TARGET=$CPU_TARGET_TRIPLE"
printf ' --build-arg %q' "CPU_BINARY_SHA256=$CPU_BINARY_SHA256"
printf ' --build-arg %q' "CPU_PROVENANCE_SHA256=$CPU_PROVENANCE_SHA256"
printf ' -t %q %q\n' "$IMAGE_TAG" "docker/"

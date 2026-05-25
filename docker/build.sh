#!/usr/bin/env bash
# Build datafusion_cudf_server with server+iceberg+cugraph features and stage
# the runtime artifacts needed by the Docker image into docker/dist/.
#
# The RAPIDS C++ libraries (libcudf.so, libcugraph*.so, librmm.so, ...) are
# provided by the RAPIDS base image at runtime, so we only need to stage:
#   - the compiled server binary
#   - libcudf_rust.so  (the CMake bridge built by cudf-sys; not in RAPIDS)
#
# Prerequisites:
#   - All sibling path deps resolved: ../cudf-rs, ../datafusion,
#     ../iceberg-rust, ../cugraph-rs
#   - libcudf already built (cudf-rs/cudf-install/ populated)
#   - patchelf (optional; rewrites RUNPATH so the container needs no
#     LD_LIBRARY_PATH override — the Dockerfile has a fallback ENV if absent)
#
# Usage:
#   bash docker/build.sh
#   docker build -t datafusion-cudf-server docker/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DIST_DIR="$SCRIPT_DIR/dist"
TARGET_ROOT="${CARGO_TARGET_DIR:-$REPO_ROOT/target}"
FEATURES="${FEATURES:-server,iceberg,cugraph}"

cd "$REPO_ROOT"

echo "==> Building datafusion_cudf_server (features: $FEATURES) ..."
cargo build --release \
    --features "$FEATURES" \
    --bin datafusion_cudf_server

BINARY="$TARGET_ROOT/release/datafusion_cudf_server"

echo "==> Staging artifacts into $DIST_DIR ..."
mkdir -p "$DIST_DIR/bin" "$DIST_DIR/lib"

# ── Binary ────────────────────────────────────────────────────────────────
cp "$BINARY" "$DIST_DIR/bin/datafusion_cudf_server"

# ── libcudf_rust.so ───────────────────────────────────────────────────────
# Locate via the RUNPATH baked into the binary by build.rs, which points at
# the cudf-sys CMake OUT_DIR.  Fall back to a glob search if patchelf/readelf
# are unavailable.
RUNPATH=$(patchelf --print-rpath "$BINARY" 2>/dev/null \
    || readelf -d "$BINARY" 2>/dev/null \
        | awk -F'[][]+' '/\(RUNPATH\)|\(RPATH\)/{print $3}' \
    || true)

CUDF_RUST_SO=""
IFS=: read -ra RPATH_DIRS <<< "${RUNPATH:-}"
for dir in "${RPATH_DIRS[@]+"${RPATH_DIRS[@]}"}"; do
    [[ -f "$dir/libcudf_rust.so" ]] && { CUDF_RUST_SO="$dir/libcudf_rust.so"; break; }
done

# Fallback: newest libcudf_rust.so under the release build dir
if [[ -z "$CUDF_RUST_SO" ]]; then
    CUDF_RUST_SO=$(find "$TARGET_ROOT/release/build" \
        -name "libcudf_rust.so" -path "*/cudf-sys-*/out/lib/*" 2>/dev/null \
        | head -1)
fi

if [[ -z "$CUDF_RUST_SO" ]]; then
    echo "ERROR: cannot locate libcudf_rust.so — run 'cargo build --release' first." >&2
    exit 1
fi

echo "==> Copying libcudf_rust.so from $CUDF_RUST_SO"
cp "$CUDF_RUST_SO" "$DIST_DIR/lib/libcudf_rust.so"

# ── Rewrite RUNPATH ───────────────────────────────────────────────────────
# Point the staged binary at /usr/local/lib (where the Dockerfile installs
# libcudf_rust.so) so the container does not need LD_LIBRARY_PATH at all.
# The Dockerfile still sets ENV LD_LIBRARY_PATH as a belt-and-suspenders
# fallback for hosts where patchelf is absent.
if command -v patchelf &>/dev/null; then
    patchelf --set-rpath /usr/local/lib "$DIST_DIR/bin/datafusion_cudf_server"
    echo "==> RUNPATH rewritten to /usr/local/lib"
else
    echo "warning: patchelf not found; container will use ENV LD_LIBRARY_PATH=/usr/local/lib"
fi

echo ""
echo "==> Staged artifacts:"
find "$DIST_DIR" -type f | sort
echo ""
echo "==> Build the image with:"
echo "    docker build -t datafusion-cudf-server docker/"

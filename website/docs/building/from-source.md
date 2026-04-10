---
sidebar_position: 4
description: Clone the repositories and build the stack bottom-up — libcudf, cudf-rs, libcugraph, cugraph-rs, then datafusion-nexus.
---

# Building from Source

All commands assume the [prerequisites](./prerequisites.md) are installed and
`env.sh` from [Environment](./environment.md) is sourced. Steps must run in
this order — each layer consumes the previous one's artifacts.

## 1. Clone the repositories

Three repositories are cloned; everything else (Apache DataFusion and the rest
of the Rust dependency graph) is fetched by Cargo, not cloned.

```bash
export NEXUS_ROOT=~/nexus # pick any root
mkdir -p "$NEXUS_ROOT" && cd "$NEXUS_ROOT"

git clone https://github.com/morristai/cudf-rs.git
git -C cudf-rs submodule update --init cudf          # libcudf fork sources

git clone https://github.com/morristai/cugraph-rs.git
git -C cugraph-rs submodule update --init cugraph    # libcugraph fork sources

git clone https://github.com/morristai/datafusion-nexus.git
```

The `cudf` and `cugraph` submodules are RAPIDS forks; `submodule update` fetches
the fork and checks out the pinned commit. A shallow `--depth 1` submodule
fetch does **not** work here — GitHub refuses to serve the pinned commit as a
bare object, so use the default (full) fetch shown above.

`datafusion-nexus/Cargo.toml` resolves `cudf` and `cugraph` through `../`
sibling paths — keep the directory names exactly as cloned. Apache DataFusion
is an ordinary published crate (`datafusion = "54"`), so there is nothing to
clone for it — Cargo fetches it and the rest of the dependency graph
automatically.

## 2. Build libcudf

This is the long step — a full CUDA compilation of libcudf. Wall-clock scales
with core count; `PARALLEL_LEVEL` defaults to `nproc`, and peak memory scales
with that parallelism, so cap it (`PARALLEL_LEVEL=<n> ./build.sh libcudf`) on a
memory-constrained box.

```bash
cd "$NEXUS_ROOT/cudf-rs/cudf"
./build.sh libcudf
```

`INSTALL_PREFIX` (from `env.sh`) sends the install to
`cudf-rs/cudf-install/`. The fork builds RMM as a **shared** library by
default (`CUDF_RMM_BUILD_SHARED=ON`) — downstream layers depend on that.

By default the build targets your GPU's own architecture
(`CMAKE_CUDA_ARCHITECTURES=NATIVE`); the configure step prints the detected
arch (e.g. `Building for the architecture of the GPU in the system...`). If the
binaries must also run on a *different* GPU model, list every target
architecture explicitly, e.g.
`CUDF_CMAKE_CUDA_ARCHITECTURES="80-real;90-real;120-real" ./build.sh libcudf`.

Verify the install tree. `./build.sh libcudf` installs **cudf and rmm only** —
`libcudf.so`, the shared `librmm.so`, and their CMake packages. The
`rapids_logger`/`nvtx3` packages that cuGraph's `find_package(rmm)` needs are
**not** installed here; the cuGraph build adds them in
[step 4](#4-build-libcugraph). `cudf-sys` links the `.so` files directly
without a CMake configure, so cudf + rmm is all it needs:

```bash
ls "$CUDF_INSTALL_DIR"/lib/libcudf.so "$CUDF_INSTALL_DIR"/lib/librmm.so
ls -d "$CUDF_INSTALL_DIR"/lib/cmake/{cudf,rmm}

# Full kernel coverage for your GPU architecture (hundreds of entries,
# not ~a dozen stubs):
cuobjdump --list-elf "$CUDF_INSTALL_DIR/lib/libcudf.so" \
  | grep -oE 'sm_[0-9]+' | sort | uniq -c
```

## 3. Build and test cudf-rs

`cudf-sys` compiles a C++ shim with CMake and links it against the installed
libcudf; the pure-Rust crates build on top.

```bash
cd "$NEXUS_ROOT/cudf-rs"
cargo build -p cudf-sys      # smoke-test shim compile + libcudf link first
cargo build
flock /tmp/cudf-gpu.lock cargo test --all-features
```

Tests execute on the GPU, so they take the global GPU lock (see
[Testing](./testing.md#gpu-serialization)).

## 4. Build libcugraph

```bash
cd "$NEXUS_ROOT/cugraph-rs"
bash scripts/build_libcugraph.sh
```

Like libcudf, this defaults to your GPU's architecture (`NATIVE`). To build for
other GPU models, pass the same architecture list through
`CMAKE_CUDA_ARCHITECTURES` (the variable the script forwards to CMake):

```bash
CMAKE_CUDA_ARCHITECTURES="80-real;90-real;120-real" bash scripts/build_libcugraph.sh
```

The script first makes the cudf install prefix export RMM's full CMake closure
(`scripts/bootstrap_cudf_install.sh`, run automatically), then configures
`cugraph/cpp` into `cugraph/cpp/build` (override with `CUGRAPH_CPP_BUILD_DIR`),
forcing `rmm` and `rapids_logger` to resolve from `CUDF_INSTALL_DIR`. It fails
loudly if the configure resolved a private RMM or if a CPM-built copy is left
in the build tree — if it aborts, fix the cause (usually a stale build dir from
a pre-external-RMM configure: remove `cugraph/cpp/build` and re-run); do not
bypass the check.

On success it prints:

```text
libcugraph native build OK: rmm and rapids_logger resolved from <CUDF_INSTALL_DIR>
```

After this step the cudf prefix is self-contained — the bootstrap has added the
`rapids_logger`, `nvtx3`, `fmt`, and `spdlog` CMake packages:

```bash
ls -d "$CUDF_INSTALL_DIR"/lib/cmake/{cudf,rmm,rapids_logger,nvtx3,fmt,spdlog}
```

## 5. Build and test cugraph-rs

```bash
cd "$NEXUS_ROOT/cugraph-rs"
cargo build
flock /tmp/cudf-gpu.lock cargo test --all-features
```

### Blackwell (sm_120) and strongly connected components

On Blackwell (`sm_120`, e.g. RTX 50-series / RTX PRO 6000) GPUs, a clean
source-built libcugraph v26.06 has a known upstream defect in
`cugraph_strongly_connected_components`
([rapidsai/cugraph#5543](https://github.com/rapidsai/cugraph/issues/5543)):
CCCL 3.4.0's CUB select tuning has no policy for compute capability 12.0. The
pinned fork routes SCC through a host-side fallback on single-GPU `sm_120`, so
the test suite passes on Blackwell — this is why the submodule pin matters and
the fork, not upstream cuGraph, must be built.

## 6. Build and test datafusion-nexus

```bash
cd "$NEXUS_ROOT/datafusion-nexus"
cargo build --features cugraph,server
flock /tmp/cudf-gpu.lock cargo nextest run --features cugraph,server --no-fail-fast
```

`--features cugraph,server` builds the full GPU stack plus the Flight SQL
server. The nextest invocation is the canonical local check — one flocked run
covers the CPU and GPU lanes together. See [Testing](./testing.md) for the lane
model, `scripts/check_all.sh`, and the feature matrix.

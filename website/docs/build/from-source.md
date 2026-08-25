---
sidebar_position: 4
description: Clone one recursive repository and build the stack bottom-up — libcudf/RMM, cuVS, cuGraph, then the root workspace.
---

# Building from Source

All commands assume the [prerequisites](./prerequisites.md) are installed and
`env.sh` from [Environment](./environment.md) is sourced. Steps must run in
this order — each layer consumes the previous one's artifacts.

## 1. Clone the repository

Clone one repository recursively. The root `.gitmodules` pins the forked cuDF,
forked cuGraph, and official cuVS native sources; Apache DataFusion and the rest
of the Rust dependency graph are fetched by Cargo.

```bash
git clone --recurse-submodules \
  https://github.com/DataFusion-Nexus/datafusion-nexus.git
export NEXUS_ROOT="$PWD/datafusion-nexus"
cd "$NEXUS_ROOT"
```

The recursive checkout places native sources at `components/cudf/cudf`,
`components/cugraph/cugraph`, and `components/cuvs/cuvs`. Do not move those
directories or add separate Rust checkouts: the root workspace resolves all
component crates through `components/*`.

## 2. Build libcudf

This is the long step — a full CUDA compilation of libcudf. Wall-clock scales
with core count; `PARALLEL_LEVEL` defaults to `nproc`, and peak memory scales
with that parallelism, so cap it (`PARALLEL_LEVEL=<n> bash
scripts/build/build_libcudf.sh`) on a memory-constrained box.

```bash
cd "$NEXUS_ROOT"
bash scripts/build/build_libcudf.sh
```

The root helper drives the pinned source at `components/cudf/cudf/`, keeps its
CMake cache in `$NEXUS_ROOT/target/native/cudf-build/`, and sends the install to
`$NEXUS_ROOT/target/native/cudf-install/`. `CUDF_CPP_BUILD_DIR` and
`CUDF_INSTALL_DIR` are the matching selectors; the helper rejects a stale cache
whose `CMAKE_INSTALL_PREFIX` points elsewhere. The fork builds RMM as a
**shared** library by default (`CUDF_RMM_BUILD_SHARED=ON`) — downstream layers
depend on that.

By default the build targets your GPU's own architecture
(`CMAKE_CUDA_ARCHITECTURES=NATIVE`); the configure step prints the detected
arch (e.g. `Building for the architecture of the GPU in the system...`). If the
binaries must also run on a *different* GPU model, list every target
architecture explicitly, e.g.
`CUDF_CMAKE_CUDA_ARCHITECTURES="80-real;90-real;120-real" bash scripts/build/build_libcudf.sh`.

Verify the install tree. The helper installs **cudf and rmm** and then prepares
the shared prefix —
`libcudf.so`, the shared `librmm.so`, and their CMake packages. The
`rapids_logger`/`nvtx3` packages that cuGraph's `find_package(rmm)` needs are
added by the helper's idempotent bootstrap and are also refreshed by
[step 4](#4-prepare-the-shared-cudf-prefix). `cudf-nexus-sys` links the `.so`
files directly without a CMake configure, so cudf + rmm is all it needs:

```bash
ls "$CUDF_INSTALL_DIR"/lib/libcudf.so "$CUDF_INSTALL_DIR"/lib/librmm.so
ls -d "$CUDF_INSTALL_DIR"/lib/cmake/{cudf,rmm}

# Full kernel coverage for your GPU architecture
cuobjdump --list-elf "$CUDF_INSTALL_DIR/lib/libcudf.so" \
  | grep -oE 'sm_[0-9]+' | sort | uniq -c
```

## 3. Build and test cudf-nexus

`cudf-nexus-sys` compiles a C++ shim with CMake and links it against the installed
libcudf; the pure-Rust crates build on top. These Cargo builds need the mold
linker on the host (install mold and clang in
[Prerequisites](./prerequisites.md#rust-linker-mold)).

```bash
cd "$NEXUS_ROOT"
cargo build -p cudf-nexus-sys --all-features  # smoke-test shim + libcudf link
cargo build --all-features
flock /tmp/cudf-gpu.lock cargo test -p cudf-nexus --all-features --no-fail-fast
```

`cudf-nexus-sys` produces `libcudf_rust.so` and its private
`libcudf_rust_rmm_interop.so` under Cargo's `OUT_DIR/lib`; keep both together
for runtime and Docker packaging.

Tests execute on the GPU, so they take the global GPU lock (see
[Tests](./tests.md#gpu-serialization)).

## 4. Prepare the shared cuDF prefix

```bash
cd "$NEXUS_ROOT"
bash scripts/build/bootstrap_cudf_install.sh
```

This prepares the cuDF install so cuVS and cuGraph can use its one shared
`librmm.so`; it does not build another RMM.

## 5. Build libcuvs

```bash
cd "$NEXUS_ROOT"
bash scripts/build/build_libcuvs.sh
```

cuVS uses the shared `librmm.so` produced by libcudf and installs its native
library into `$NEXUS_ROOT/target/native/cuvs-install/`. It must be built before
libcugraph.

## 6. Build and test cuvs-nexus

```bash
cd "$NEXUS_ROOT"
cargo build -p cuvs-nexus --all-features
flock /tmp/cudf-gpu.lock cargo test -p cuvs-nexus --all-features --no-fail-fast
```

## 7. Build libcugraph

```bash
cd "$NEXUS_ROOT"
bash scripts/build/build_libcugraph.sh
```

Like libcudf, this defaults to your GPU's architecture (`NATIVE`). To build for
other GPU models, pass the same architecture list through
`CMAKE_CUDA_ARCHITECTURES` (the variable the script forwards to CMake):

```bash
CMAKE_CUDA_ARCHITECTURES="80-real;90-real;120-real" \
  bash scripts/build/build_libcugraph.sh
```

The script first makes the cudf install prefix export RMM's full CMake closure
(`scripts/build/bootstrap_cudf_install.sh`, run automatically), then configures
`components/cugraph/cugraph/cpp` into `target/native/cugraph-build` (override
with `CUGRAPH_CMAKE_BUILD_DIR`),
forcing `rmm` and `rapids_logger` to resolve from `CUDF_INSTALL_DIR`. It fails
loudly if the configure resolved a private RMM or if a CPM-built copy is left
in the build tree — if it aborts, fix the cause (usually a stale build dir from
a pre-external-RMM configure: remove `target/native/cugraph-build` and re-run); do not
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

## 8. Build and test cugraph-nexus

```bash
cd "$NEXUS_ROOT"
cargo build -p cugraph-nexus --all-features
flock /tmp/cudf-gpu.lock cargo test -p cugraph-nexus --all-features --no-fail-fast
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

## 9. Build and test datafusion-nexus

`datafusion-nexus` selects mold via its `.cargo/config.toml`; mold and clang
must already be installed (see
[Prerequisites](./prerequisites.md#rust-linker-mold)).

```bash
cd "$NEXUS_ROOT"
cargo build -p nexus-server --all-features
bash scripts/check_all.sh
```

The `nexus-server` package owns the Flight SQL server and forwards the adapter's
GPU features. A bare `check_all.sh` is the canonical full pre-merge check; it
assigns CPU and GPU lanes and acquires the cross-consumer GPU lock. See
[Tests](./tests.md) for partial edit loops and the lane model.

To package the standalone Flight SQL server after this source build, continue
with [Building with Docker](./docker.md).

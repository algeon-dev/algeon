---
sidebar_position: 4
description: Clone the five repositories and build the stack bottom-up — libcudf, cudf-rs, libcugraph, cugraph-rs, then datafusion-nexus.
---

# Building from Source

All commands assume the [prerequisites](./prerequisites.md) are installed and
`env.sh` from [Environment](./environment.md) is sourced. Steps must run in
this order — each layer consumes the previous one's artifacts.

## 1. Clone the repositories

```bash
export NEXUS_ROOT=~/nexus          # pick any root
mkdir -p "$NEXUS_ROOT" && cd "$NEXUS_ROOT"

git clone https://github.com/morristai/cudf-rs.git
git -C cudf-rs submodule update --init cudf          # libcudf fork sources

git clone https://github.com/morristai/cugraph-rs.git
git -C cugraph-rs submodule update --init cugraph    # libcugraph fork sources

git clone https://github.com/apache/datafusion.git
git clone -b feat/upgrade-datafusion-v54 https://github.com/morristai/iceberg-rust.git
git clone https://github.com/morristai/datafusion-nexus.git
```

The `cudf` and `cugraph` submodules are full RAPIDS histories; add
`--depth 1` to the `submodule update` calls to fetch only the pinned commit.

:::note[Path dependencies, not crates.io]
`datafusion-nexus/Cargo.toml` resolves `cudf`, `cugraph`, `datafusion`, and
`iceberg` through `../` sibling paths. Do not substitute crates.io versions,
and keep the directory names exactly as cloned. `iceberg-rust` must stay on
the `feat/upgrade-datafusion-v54` branch until upstream releases DataFusion
54 support.
:::

## 2. Build libcudf

This is the long step — a full CUDA compilation of libcudf (measured at
~35 minutes on a 32-core machine; peak memory scales with parallelism,
`PARALLEL_LEVEL` defaults to `nproc`).

```bash
cd "$NEXUS_ROOT/cudf-rs/cudf"
./build.sh libcudf
```

`INSTALL_PREFIX` (from `env.sh`) sends the install to
`cudf-rs/cudf-install/`. The fork builds RMM as a **shared** library by
default (`CUDF_RMM_BUILD_SHARED=ON`) — downstream layers depend on that.

By default the build targets your GPU's own architecture
(`CMAKE_CUDA_ARCHITECTURES=NATIVE`). If the binaries must also run on a
*different* GPU model, list every target architecture explicitly, e.g.
`CUDF_CMAKE_CUDA_ARCHITECTURES="89-real;120-real" ./build.sh libcudf`.

Verify the install tree — the prefix must be a self-contained package root:
the shared libraries **and** the CMake packages the cuGraph build later
resolves from it:

```bash
ls "$CUDF_INSTALL_DIR"/lib/libcudf.so \
   "$CUDF_INSTALL_DIR"/lib/librmm.so \
   "$CUDF_INSTALL_DIR"/lib/librapids_logger.so
ls -d "$CUDF_INSTALL_DIR"/lib/cmake/{cudf,rmm,rapids_logger,nvtx3}

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
bash scripts/build_cpp.sh
```

The script configures `cugraph/cpp` into `cugraph/cpp/build` (override with
`CUGRAPH_CPP_BUILD_DIR`), forcing `rmm` and `rapids_logger` to resolve from
`CUDF_INSTALL_DIR`. It fails loudly if the configure resolved a private RMM
or if a CPM-built copy is left in the build tree — if it aborts, fix the
cause (usually a stale build dir from a pre-external-RMM configure: remove
`cugraph/cpp/build` and re-run); do not bypass the check.

On success it prints:

```text
cuGraph native build OK: rmm and rapids_logger resolved from <CUDF_INSTALL_DIR>
```

## 5. Build and test cugraph-rs

```bash
cd "$NEXUS_ROOT/cugraph-rs"
cargo build
flock /tmp/cudf-gpu.lock cargo test --all-features
```

:::info[Blackwell (sm_120) and strongly connected components]
On RTX 50-series GPUs, a clean source-built libcugraph v26.06 has a known
upstream defect in `cugraph_strongly_connected_components`
([rapidsai/cugraph#5543](https://github.com/rapidsai/cugraph/issues/5543)).
The pinned fork routes SCC through a host-side fallback on single-GPU
`sm_120`, so the test suite passes on Blackwell — this is why the submodule
pin matters and the fork, not upstream cuGraph, must be built.
:::

## 6. Build and test datafusion-nexus

```bash
cd "$NEXUS_ROOT/datafusion-nexus"
cargo build --all-features
flock /tmp/cudf-gpu.lock cargo nextest run --all-features --no-fail-fast
```

The nextest invocation is the canonical local check — one flocked run covers
the CPU and GPU lanes together. See [Testing](./testing.md) for the lane
model, `scripts/check_all.sh`, and the feature matrix.

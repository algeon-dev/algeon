---
sidebar_position: 3
description: Clone one recursive repository, configure one checkout-local environment, and build cuDF/RMM, cuVS, cuGraph, and the Rust workspace.
---

# Building from Source

This is the complete source-build path. It starts with one recursive
`datafusion-nexus` checkout and keeps every source, native build tree, install
prefix, Rust crate, and validation command inside that checkout.

Complete the [prerequisites](./prerequisites.md) first, then run the steps below
in order: cuDF/RMM → cuVS → cuGraph → the Rust workspace.

## 1. Clone and initialize the repository

Clone the root repository recursively into any working directory:

```bash
git clone --recurse-submodules https://github.com/DataFusion-Nexus/datafusion-nexus.git
cd datafusion-nexus
```

For an existing non-recursive clone, initialize it in place:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

The root commit pins the exact native-source revisions as Git submodule
gitlinks:

```text
components/cudf/cudf/       # DataFusion-Nexus cuDF fork
components/cugraph/cugraph/ # DataFusion-Nexus cuGraph fork
components/cuvs/cuvs/       # Temp official RAPIDS cuVS (WIP)
```

Verify that all three are initialized and match the root commit:

```bash
git submodule status --recursive
```

## 2. Configure the build shell

Run this block from the checkout root in every new shell used to build, test,
run, or package the project:

```bash
export NEXUS_ROOT="$(pwd -P)"
test -f "$NEXUS_ROOT/Cargo.toml"
test -f "$NEXUS_ROOT/.gitmodules"

# CUDA toolkit. nvcc must be on PATH for CMake language detection.
export CUDA_HOME=/usr/local/cuda
export CUDA_PATH="$CUDA_HOME"
export CUDAToolkit_ROOT="$CUDA_HOME"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"

# Native host compiler required by the pinned sources.
export CC=/usr/bin/gcc-14
export CXX=/usr/bin/g++-14
export CUDAHOSTCXX=/usr/bin/g++-14

# One CUDA target contract for cuDF, cuVS, and cuGraph. NATIVE is host-local.
export CMAKE_CUDA_ARCHITECTURES=NATIVE
unset CUDF_CMAKE_CUDA_ARCHITECTURES

# Checkout-local native build and install identity.
export CUDF_CPP_BUILD_DIR="$NEXUS_ROOT/target/native/cudf-build"
export CUDF_INSTALL_DIR="$NEXUS_ROOT/target/native/cudf-install"
export INSTALL_PREFIX="$CUDF_INSTALL_DIR"
export CUVS_INSTALL_DIR="$NEXUS_ROOT/target/native/cuvs-install"
export CUVS_ROOT="$CUVS_INSTALL_DIR"
export CUGRAPH_CMAKE_BUILD_DIR="$NEXUS_ROOT/target/native/cugraph-build"

# Use only this checkout's RAPIDS libraries at runtime.
export LD_LIBRARY_PATH="$CUGRAPH_CMAKE_BUILD_DIR:$CUDF_INSTALL_DIR/lib:$CUVS_INSTALL_DIR/lib:$CUDA_HOME/lib64"
```

These assignments deliberately replace inherited native selectors. Preserving
an older `CUDF_INSTALL_DIR`, `CUVS_INSTALL_DIR`, `CUVS_ROOT`, or
`CUGRAPH_CMAKE_BUILD_DIR` can silently mix libraries from another checkout.
Edit this block explicitly if your CUDA toolkit or gcc 14 installation is in a
different location.

The selectors have one shared identity:

| Variable | Owner and purpose |
| --- | --- |
| `NEXUS_ROOT` | Root of the one recursive checkout. |
| `CMAKE_CUDA_ARCHITECTURES` | One CUDA target list shared by cuDF, cuVS, and cuGraph. Defaults to `NATIVE`. |
| `CUDF_CPP_BUILD_DIR` | cuDF CMake cache, normally `target/native/cudf-build`. |
| `CUDF_INSTALL_DIR` / `INSTALL_PREFIX` | cuDF install containing the stack's one shared `librmm.so`. |
| `CUVS_INSTALL_DIR` / `CUVS_ROOT` | One cuVS prefix used by its build, Rust bindings, and cuGraph. |
| `CUGRAPH_CMAKE_BUILD_DIR` | cuGraph CMake build and runtime-library directory. |
| `LD_LIBRARY_PATH` | Runtime resolution in dependency order: cuGraph, cuDF/RMM, cuVS, CUDA. |

The native dependency relationship is:

```text
cuDF/RMM → cuVS → cuGraph → DataFusion Nexus
```

Before a long build, confirm that no selector escapes the checkout:

```bash
env | sort | grep -E '^(NEXUS_ROOT|INSTALL_PREFIX|LD_LIBRARY_PATH|CMAKE_CUDA_ARCHITECTURES|CUDF_[^=]*|CUVS_[^=]*|CUGRAPH_[^=]*)='
```

Every native path in the output must resolve under
`$NEXUS_ROOT/target/native/`. If one does not, rerun the block above from the
current repository root. `scripts/check_all.sh` intentionally inherits the
configured native identity; it does not guess that an exported path is stale.

Choose the architecture contract before starting the first native build. Keep
`NATIVE` for artifacts that will run only on this build host. For artifacts
that move between known GPU models, list their targets explicitly, for example:

```bash
export CMAKE_CUDA_ARCHITECTURES="89-real;120a-real"
```

The `a` in `120a` selects the architecture-specific compute capability 12.0
target and the `f` in a value such as `100f` selects the family-specific one;
`-real` asks CMake for real device code. `CMAKE_CUDA_ARCHITECTURES` is recorded
verbatim as the contract's `requested` value, so the CMake caches keep `NATIVE`
when that is what you set. The `real` value is read back from the built library
and does name the concrete targets, such as `120a`.

## 3. Build the native stack

### Build cuDF and the shared RMM prefix

This is the longest step. `PARALLEL_LEVEL` defaults to `nproc`; cap it on a
memory-constrained host.

```bash
bash scripts/build/build_libcudf.sh
```

The helper drives `components/cudf/cudf`, keeps its CMake cache under
`target/native/cudf-build`, installs cuDF and shared RMM under
`target/native/cudf-install`, and bootstraps the exported RMM dependency
closure needed downstream. It rejects a cache configured for another install
prefix.

```bash
ls "$CUDF_INSTALL_DIR"/lib/libcudf.so "$CUDF_INSTALL_DIR"/lib/librmm.so
ls -d "$CUDF_INSTALL_DIR"/lib/cmake/{cudf,rmm,rapids_logger,nvtx3,fmt,spdlog}
```

The helper maps the shared `CMAKE_CUDA_ARCHITECTURES` value to cuDF's upstream
build interface and records both the requested value and the full-kernel real
cubin set found in `libcudf.so`; small embedded dependency stub sets are not
treated as cuDF targets.

### Build cuVS

```bash
bash scripts/build/build_libcuvs.sh
```

The helper builds the pinned source at `components/cuvs/cuvs`, uses RMM from
`CUDF_INSTALL_DIR`, and installs both the C++ and C libraries into the one
`CUVS_INSTALL_DIR` selected above. It rejects a cuDF prefix built with another
requested selector before configuring cuVS, then compares both libraries'
full-kernel real cubin sets after installation.

```bash
ls "$CUVS_INSTALL_DIR"/lib/libcuvs.so \
   "$CUVS_INSTALL_DIR"/lib/libcuvs_c.so \
   "$CUVS_INSTALL_DIR"/lib/cmake/cuvs/cuvs-config.cmake
```

### Build cuGraph

```bash
bash scripts/build/build_libcugraph.sh
```

The helper builds `components/cugraph/cugraph` into
`target/native/cugraph-build`. It binds cuGraph to the selected cuDF/RMM and
cuVS prefixes and fails if CMake or the dynamic loader resolves a private or
different copy. It also checks the cuDF and cuVS architecture contracts before
the expensive build, then records the real cubins found in `libcugraph.so` and
compares them against cuDF's.

```bash
ls "$CUGRAPH_CMAKE_BUILD_DIR"/libcugraph.so \
   "$CUGRAPH_CMAKE_BUILD_DIR"/libcugraph_c.so
```

Do not change `CMAKE_CUDA_ARCHITECTURES` between these three commands. If a
prefix was built with another target, the next script stops with the exact
component and dependency-order rebuild command instead of leaving a
link-compatible but kernel-incompatible stack.

On Blackwell (`sm_120`), the pinned cuGraph fork carries the required
single-GPU strongly-connected-components compatibility path. Building a
different upstream checkout bypasses that contract.

## 4. Build the Rust workspace

After all three native layers exist, build the unified workspace with one
feature selection:

```bash
cargo build --workspace --all-features
```

Cargo resolves every first-party Rust crate from this checkout's root
workspace and lockfile.

For a smaller edit loop, select a package while retaining the same feature
contract:

```bash
cargo build -p datafusion-nexus --all-features
cargo build -p nexus-query-engine --all-features
cargo build -p cudf-nexus --all-features
```

## 5. Validate the checkout

Run the canonical pre-merge surface bare:

```bash
bash scripts/check_all.sh
```

Do not wrap it in `flock`; the script acquires `/tmp/cudf-gpu.lock` internally
for every GPU phase. It covers policy checks, nightly formatting, CPU and GPU
nextest lanes, doctests, rustdoc, public API snapshots, and clippy. See
[Tests](./tests.md) for focused package and test-harness commands.

To package the standalone Flight SQL server after this source build, continue
with [Building with Docker](./docker.md).

---
sidebar_position: 3
description: Clone one recursive repository, configure one checkout-local environment, and build cuDF/RMM, cuVS, cuGraph, and the Rust workspace.
---

# Building from Source

This is the complete source-build path. It starts with one recursive
`datafusion-nexus` checkout and keeps every source, native build tree, install
prefix, Rust crate, and validation command inside that checkout.

Before you start:

- [Prerequisites](./prerequisites.md) are installed: CUDA toolkit with `nvcc`,
  a RAPIDS-compatible host compiler, CMake, Ninja, `clang`, `mold`, Rust, and
  the pinned `cargo-nextest` and `cargo-hack` versions.
- The build host has the GPU the artifacts will run on, or you have chosen an
  explicit architecture list ([Choosing CUDA architectures](#choosing-cuda-architectures)).

Run the steps in order: cuDF/RMM → cuVS → cuGraph → the Rust workspace. Each
native step prints where it installed and fails early if the previous layer
is missing or was built for a different CUDA target. When a step fails, the
message names the component and the rebuild command; the failures whose text
does not point at the cause are in [Troubleshooting](./troubleshooting.md).

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
components/cuvs/cuvs/       # official RAPIDS cuVS (not a fork)
```

Verify that all three are initialized and match the root commit; every line
must start with a space, not `-` (uninitialized) or `+` (wrong revision):

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

# Native host compiler. Substitute your RAPIDS-compatible compiler paths.
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

Edit the CUDA and compiler lines if your toolkit or compiler lives elsewhere.
The remaining assignments deliberately replace inherited native selectors:
preserving an older `CUDF_INSTALL_DIR`, `CUVS_INSTALL_DIR`, `CUVS_ROOT`, or
`CUGRAPH_CMAKE_BUILD_DIR` can silently mix libraries from another checkout.

Confirm that no selector escapes the checkout:

```bash
env | sort | grep -E '^(NEXUS_ROOT|INSTALL_PREFIX|LD_LIBRARY_PATH|CMAKE_CUDA_ARCHITECTURES|CUDF_[^=]*|CUVS_[^=]*|CUGRAPH_[^=]*)='
```

Every native path in the output must resolve under
`$NEXUS_ROOT/target/native/`. If one does not, rerun the block above from the
current repository root. The build scripts and `scripts/check_all.sh` inherit
the configured identity as-is; they do not guess that an exported path is stale.
What each selector means is listed under
[Build-shell selectors](#build-shell-selectors).

## 3. Build the native stack

Do not change `CMAKE_CUDA_ARCHITECTURES` between the three commands. Each
script records the value in the library it builds and refuses to build on top
of a prefix that carries a different one.

### Build cuDF and the shared RMM prefix

This is the longest step. `PARALLEL_LEVEL` defaults to `nproc`; cap it on a
memory-constrained host.

```bash
bash scripts/build/build_libcudf.sh
```

The script drives `components/cudf/cudf`, keeps its CMake cache under
`target/native/cudf-build`, installs cuDF and shared RMM under
`target/native/cudf-install`, and completes the exported RMM dependency
closure (`rapids_logger`, `nvtx3`, `fmt`, `spdlog`) that cuVS and cuGraph
need. Success looks like:

```bash
ls "$CUDF_INSTALL_DIR"/lib/libcudf.so "$CUDF_INSTALL_DIR"/lib/librmm.so
ls -d "$CUDF_INSTALL_DIR"/lib/cmake/{cudf,rmm,rapids_logger,nvtx3,fmt,spdlog}
```

### Build cuVS

```bash
bash scripts/build/build_libcuvs.sh
```

The script builds `components/cuvs/cuvs` against the RMM in
`CUDF_INSTALL_DIR` and installs the C++ and C libraries into
`CUVS_INSTALL_DIR`. Success looks like:

```bash
ls "$CUVS_INSTALL_DIR"/lib/libcuvs.so \
   "$CUVS_INSTALL_DIR"/lib/libcuvs_c.so \
   "$CUVS_INSTALL_DIR"/lib/cmake/cuvs/cuvs-config.cmake
```

### Build cuGraph

```bash
bash scripts/build/build_libcugraph.sh
```

The script builds `components/cugraph/cugraph` into
`target/native/cugraph-build`, bound to the cuDF/RMM and cuVS prefixes
selected above. It fails if CMake or the dynamic loader resolves a private or
different copy of either. Success looks like:

```bash
ls "$CUGRAPH_CMAKE_BUILD_DIR"/libcugraph.so \
   "$CUGRAPH_CMAKE_BUILD_DIR"/libcugraph_c.so
```

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
workspace and lockfile. For a smaller edit loop, select a package while
retaining the same feature contract:

```bash
cargo build -p datafusion-nexus --all-features
cargo build -p nexus-query-engine --all-features
cargo build -p cudf-nexus --all-features
```

## 5. Validate the checkout

Run the single full pre-merge gate bare, from the checkout root:

```bash
bash scripts/check_all.sh
```

Do not wrap it in `flock`; the script acquires `/tmp/cudf-gpu.lock` internally
for every GPU phase, and an outer hold deadlocks it. It covers policy checks,
nightly formatting, MSRV and feature-isolation compile checks, the CPU and GPU
nextest lanes, doctests, rustdoc, public API snapshots, and clippy. A passing
run ends with a phase table and:

```text
OK: canonical full pre-merge surface passed.
```

[Tests](./tests.md) explains the phases, the partial edit loops, and where the
logs go when a phase fails.

To package the standalone Flight SQL server after this source build, continue
with [Building with Docker](./docker.md).

## Reference

### Build-shell selectors

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

### Choosing CUDA architectures

Choose the architecture contract before starting the first native build. Keep
`NATIVE` for artifacts that will run only on this build host. For artifacts
that move between known GPU models, list their targets explicitly, for example:

```bash
export CMAKE_CUDA_ARCHITECTURES="89-real;120a-real"
```

The `a` in `120a` selects the architecture-specific compute capability 12.0
target, the `f` in a value such as `100f` selects the family-specific one, and
`-real` asks CMake for real device code. The old cuDF-only
`CUDF_CMAKE_CUDA_ARCHITECTURES` selector is rejected; the cuDF script maps the
shared value to cuDF's upstream build interface internally.

### The architecture contract record

Each native script records the requested `CMAKE_CUDA_ARCHITECTURES` value
verbatim, plus the full-kernel real cubin set it finds in the built library, at
`share/nexus-native/cuda-architectures.env` under that layer's prefix. The
cuVS and cuGraph scripts read the upstream record before configuring and
compare real cubin sets after building; small embedded dependency stub sets are
not treated as targets. An explicitly selected external prefix is exempt,
since nexus scripts did not build it and it carries no record. To inspect the
records:

```bash
for root in "$CUDF_INSTALL_DIR" "$CUVS_INSTALL_DIR" "$CUGRAPH_CMAKE_BUILD_DIR"; do
  echo "== $root =="
  cat "$root/share/nexus-native/cuda-architectures.env"
done
```

The check compares the native layers against each other only; it does not
compare them against the GPU installed in this host. A binary that links but
aborts on its first kernel launch is covered in
[Troubleshooting](./troubleshooting.md#gpu-op-aborts-cudaerrornokernelimagefordevice).

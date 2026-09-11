---
sidebar_position: 3
description: Clone one recursive repository, configure one checkout-local environment, and build cuDF/RMM, cuVS, cuGraph, and the Rust workspace.
---

# Building from Source

This is the complete source-build path. It starts with one recursive
`algeon` checkout and keeps every source, native build tree, install
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
git clone --recurse-submodules https://github.com/algeon-dev/algeon.git
cd algeon
```

For an existing non-recursive clone, initialize it in place:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

The root commit pins the exact native-source revisions as Git submodule
gitlinks:

```text
components/cudf/cudf/       # Algeon cuDF fork
components/cugraph/cugraph/ # Algeon cuGraph fork
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
export ALGEON_ROOT="$(pwd -P)"
test -f "$ALGEON_ROOT/Cargo.toml"
test -f "$ALGEON_ROOT/.gitmodules"

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

# Native selectors default to this checkout's target/native/ prefixes. Clearing
# them keeps another checkout's libraries out and gives every shell the same
# build fingerprint. Cargo-built binaries find the native libraries through
# their rpath, so this block does not set LD_LIBRARY_PATH.
unset ALGEON_NATIVE_DIR CUDF_CPP_BUILD_DIR CUDF_INSTALL_DIR INSTALL_PREFIX \
  CUVS_INSTALL_DIR CUVS_ROOT CUGRAPH_CMAKE_BUILD_DIR CUGRAPH_INSTALL_DIR
```

Edit the CUDA and compiler lines if your toolkit or compiler lives elsewhere.
The `unset` clears any native selectors inherited from a previous shell, such
as an older `CUDF_INSTALL_DIR`, `CUVS_INSTALL_DIR`, `CUVS_ROOT`, or
`CUGRAPH_CMAKE_BUILD_DIR`, so another checkout's libraries cannot leak in.
Every Cargo build script and `scripts/build/build_lib*.sh` already default
each cleared selector to this checkout's `target/native/` prefixes, so leaving
them unset gives every shell the same build fingerprint.

Also write the CUDA and compiler lines once into a Cargo `[env]` table outside
the checkout, as described in
[Native build environment](./prerequisites.md#native-build-environment), so a
plain shell that never ran this block still builds with the same toolchain.

Confirm that no selector escapes the checkout:

```bash
env | sort | grep -E '^(ALGEON_ROOT|CMAKE_CUDA_ARCHITECTURES)='
env | sort | grep -E '^(INSTALL_PREFIX|ALGEON_NATIVE_DIR|CUDF_[^=]*|CUVS_[^=]*|CUGRAPH_[^=]*)=' && echo "selector leaked" || echo "clean"
```

The first command must show `ALGEON_ROOT` at the checkout root and
`CMAKE_CUDA_ARCHITECTURES` at the chosen target list. The second must print
`clean`; if it prints `selector leaked`, rerun the block above from the
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
need. It then builds and installs the production Rust shim and its crate-version
marker into the same prefix. Success looks like:

```bash
ls "$ALGEON_ROOT"/target/native/cudf-install/lib/libcudf.so \
   "$ALGEON_ROOT"/target/native/cudf-install/lib/librmm.so
ls "$ALGEON_ROOT"/target/native/cudf-install/lib/libcudf_rust.so \
   "$ALGEON_ROOT"/target/native/cudf-install/share/cudf_rust/{cargo-link.txt,crate-version}
ls -d "$ALGEON_ROOT"/target/native/cudf-install/lib/cmake/{cudf,rmm,rapids_logger,nvtx3,fmt,spdlog}
```

### Build cuVS

```bash
bash scripts/build/build_libcuvs.sh
```

The script builds `components/cuvs/cuvs` against the RMM in
`target/native/cudf-install` and installs the C++ and C libraries into
`target/native/cuvs-install`. Success looks like:

```bash
ls "$ALGEON_ROOT"/target/native/cuvs-install/lib/libcuvs.so \
   "$ALGEON_ROOT"/target/native/cuvs-install/lib/libcuvs_c.so \
   "$ALGEON_ROOT"/target/native/cuvs-install/lib/cmake/cuvs/cuvs-config.cmake
```

### Build cuGraph

```bash
bash scripts/build/build_libcugraph.sh
```

The script builds `components/cugraph/cugraph` into
`target/native/cugraph-build`, bound to the cuDF/RMM and cuVS prefixes
selected above, then installs the registry-consumer artifacts into
`target/native/cugraph-install`. It fails if CMake or the dynamic loader
resolves a private or different copy of either. Success looks like:

```bash
ls "$ALGEON_ROOT"/target/native/cugraph-build/libcugraph.so \
   "$ALGEON_ROOT"/target/native/cugraph-build/libcugraph_c.so
ls "$ALGEON_ROOT"/target/native/cugraph-install/lib/libcugraph.so \
   "$ALGEON_ROOT"/target/native/cugraph-install/include/cugraph_c/algorithms.h \
   "$ALGEON_ROOT"/target/native/cugraph-install/share/cugraph/crate-version
```

On Blackwell (`sm_120`), the pinned cuGraph fork carries the required
single-GPU strongly-connected-components compatibility path. Building a
different upstream checkout bypasses that contract.

### Use the native stack from registry crates

The three build scripts create a native root that registry-installed Algeon
crates can consume without a source checkout inside Cargo's package archive:

```bash
export ALGEON_NATIVE_DIR="$ALGEON_ROOT/target/native"
cargo build --all-features
```

`ALGEON_NATIVE_DIR` must contain `cudf-install`, `cuvs-install`, and
`cugraph-install` from the matching Algeon tag. The sys crates verify their
required library before invoking any native tool; the cuDF and cuGraph
prefixes also carry a `crate-version` marker and are rejected when it differs
from the Rust crate version. A registry build still needs `cmake` for cuVS
package discovery, `readelf` and `ldd` from binutils for cuGraph loader
validation, and a C compiler for the cuVS ABI probe. It does not run bindgen or
build cuDF or cuGraph with CMake.

`CUDF_INSTALL_DIR`, `CUVS_ROOT`, `CUVS_INSTALL_DIR`, and the three
`CUGRAPH_NATIVE_*` variables remain lower-level developer selectors. Prefer
`ALGEON_NATIVE_DIR` when consuming the complete stack.

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
cargo build -p algeon-datafusion --all-features
cargo build -p query-engine --all-features
cargo build -p cudf --all-features
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

Every selector below is an optional override. Left unset, as the build shell
block leaves them, each defaults to this checkout's `target/native/` prefix;
set one only to point a step at a prefix outside the checkout.

| Variable | Owner and purpose |
| --- | --- |
| `ALGEON_ROOT` | Root of the one recursive checkout. |
| `ALGEON_NATIVE_DIR` | Root containing the three matching registry-consumer install prefixes. Defaults to `target/native`. |
| `CMAKE_CUDA_ARCHITECTURES` | One CUDA target list shared by cuDF, cuVS, and cuGraph. Defaults to `NATIVE`. |
| `CUDF_CPP_BUILD_DIR` | cuDF CMake cache. Defaults to `target/native/cudf-build`. |
| `CUDF_INSTALL_DIR` / `INSTALL_PREFIX` | cuDF install containing the stack's one shared `librmm.so`. Defaults to `target/native/cudf-install`. |
| `CUVS_INSTALL_DIR` / `CUVS_ROOT` | One cuVS prefix used by its build, Rust bindings, and cuGraph. Defaults to `target/native/cuvs-install`. |
| `CUGRAPH_CMAKE_BUILD_DIR` | cuGraph CMake build and runtime-library directory. Defaults to `target/native/cugraph-build`. |
| `CUGRAPH_INSTALL_DIR` | Installed cuGraph libraries, headers, and crate-version marker. Defaults to `target/native/cugraph-install`. |
| `LD_LIBRARY_PATH` | Runtime resolution in dependency order: cuGraph, cuDF/RMM, cuVS, CUDA. Not needed for Cargo-built binaries, which resolve native libraries through their rpath; only a caller that runs a non-Cargo loader, such as a rustdoc doctest binary, needs to compose it. |

The native dependency relationship is:

```text
cuDF/RMM → cuVS → cuGraph → Algeon
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
`share/algeon-native/cuda-architectures.env` under that layer's prefix. The
cuVS and cuGraph scripts read the upstream record before configuring and
compare real cubin sets after building; small embedded dependency stub sets are
not treated as targets. An explicitly selected external prefix is exempt,
since algeon scripts did not build it and it carries no record. To inspect the
records:

```bash
for root in "$ALGEON_ROOT"/target/native/{cudf,cuvs,cugraph}-install; do
  echo "== $root =="
  cat "$root/share/algeon-native/cuda-architectures.env"
done
```

The check compares the native layers against each other only; it does not
compare them against the GPU installed in this host. A binary that links but
aborts on its first kernel launch is covered in
[Troubleshooting](./troubleshooting.md#gpu-op-aborts-cudaerrornokernelimagefordevice).

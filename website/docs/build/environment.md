---
sidebar_position: 3
description: The complete environment-variable contract for building and running the stack, and how it was validated against shell-state leakage.
---

# Environment

The whole stack builds and runs with **one small environment file**. Save it
as `env.sh` in your checkout root (`$NEXUS_ROOT`) and source it in every
shell you build or test from.

```bash title="$NEXUS_ROOT/env.sh"
# Root of the recursive datafusion-nexus checkout.
nexus_env_source="${BASH_SOURCE[0]:-$0}"
nexus_env_dir=$(dirname "$nexus_env_source")
nexus_env_dir=$(cd "$nexus_env_dir" && pwd)
NEXUS_ROOT="${NEXUS_ROOT:-$nexus_env_dir}"
unset nexus_env_source nexus_env_dir
export NEXUS_ROOT

# --- CUDA toolkit -----------------------------------------------------------
# CUDACXX / CUDAToolkit_ROOT are read by CMake (libcudf, libcuvs, libcugraph, the
# cudf-nexus-sys shim); nvcc must ALSO be on PATH for CMake language detection.
export CUDA_HOME=/usr/local/cuda
export CUDA_PATH="$CUDA_HOME"
export CUDAToolkit_ROOT="$CUDA_HOME"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"

# --- Host compiler ----------------------------------------------------------
# RAPIDS 26.06 pins gcc 14. Ubuntu 24.04's default is gcc 13, so point the
# native builds at gcc-14 explicitly instead of changing the system default.
# Prefer absolute paths: cudf-nexus-sys fingerprints CC/CXX (and clang tool overrides)
# rather than ambient PATH, so a venv or agent PATH prepend must not force a
# native rebuild when source and selected compilers are unchanged.
export CC=/usr/bin/gcc-14
export CXX=/usr/bin/g++-14
export CUDAHOSTCXX=/usr/bin/g++-14

# --- RAPIDS native libraries -------------------------------------------------
export CUDF_CPP_BUILD_DIR="${CUDF_CPP_BUILD_DIR:-$NEXUS_ROOT/target/native/cudf-build}"
export CUDF_INSTALL_DIR="${CUDF_INSTALL_DIR:-$NEXUS_ROOT/target/native/cudf-install}"
export INSTALL_PREFIX="$CUDF_INSTALL_DIR"
export CUVS_INSTALL_DIR="${CUVS_INSTALL_DIR:-$NEXUS_ROOT/target/native/cuvs-install}"
export CUGRAPH_CMAKE_BUILD_DIR="${CUGRAPH_CMAKE_BUILD_DIR:-$NEXUS_ROOT/target/native/cugraph-build}"

# --- Runtime library resolution ----------------------------------------------
export LD_LIBRARY_PATH="$CUGRAPH_CMAKE_BUILD_DIR:$CUDF_INSTALL_DIR/lib:$CUVS_INSTALL_DIR/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

## What each variable is for

| Variable | Consumed by | Purpose |
| --- | --- | --- |
| `CUDA_HOME` / `CUDA_PATH` / `CUDAToolkit_ROOT` | CMake toolkit discovery, `cudf-nexus-sys` fingerprint | Pin which installed CUDA toolkit compiles the stack. |
| `CUDACXX` | CMake, `cudf-nexus-sys` fingerprint | The CUDA compiler executable. |
| `CC` / `CXX` / `CUDAHOSTCXX` | CMake, `nvcc`, `cudf-nexus-sys` fingerprint | Pin the host compiler to gcc 14 (the RAPIDS 26.06 toolchain) without changing the system default. Prefer **absolute** paths. `cudf-nexus-sys` tracks these selections as rebuild inputs and resolves `CC`/`CXX` basenames once; ambient PATH is no longer a fingerprint. `CUDAHOSTCXX` is the host compiler `nvcc` drives for device compilation. |
| `CUDF_RUST_CLANG_FORMAT` / `CUDF_RUST_CLANG_TIDY` | `cudf-nexus-sys` when `CUDF_RUST_CLANG_LINT=ON` | Optional absolute pins for clang tools. When unset, tools are discovered once from PATH and tracked via these variables / lint mode rather than re-fingerprinting every PATH change. |
| `CUDF_RUST_CMAKE` | `cudf-nexus-sys`, `cmake-rs` | Optional absolute pin for the CMake executable. The build script forwards it through cmake-rs's standard `CMAKE` input and tracks the pin as rebuild identity. |
| `PATH` (+`$CUDA_HOME/bin`) | CMake language detection (`nvcc`), one-shot tool discovery | The `cudf-nexus-sys` shim declares CUDA as a project language; setting only `CUDA_HOME` is not enough — `nvcc` must be on `PATH`. PATH is **not** a `cargo:rerun-if-env-changed` input for `cudf-nexus-sys`. |
| `CUDF_CPP_BUILD_DIR` | `scripts/build/build_libcudf.sh`, `cudf-nexus-sys` | Canonical cuDF CMake build/cache tree; defaults to `target/native/cudf-build`. Its `CMAKE_INSTALL_PREFIX` must match `CUDF_INSTALL_DIR` when both selectors are used. |
| `INSTALL_PREFIX` | `scripts/build/build_libcudf.sh` → `components/cudf/cudf/build.sh` | Install-prefix value forwarded by the root helper to the pinned RAPIDS build; keep it aligned with `CUDF_INSTALL_DIR` rather than selecting a second install tree. |
| `CUDF_INSTALL_DIR` | cuDF, cuVS, and cuGraph builds | The canonical cuDF install, including the one shared `librmm.so` used by the rest of the stack. |
| `CUVS_INSTALL_DIR` | cuVS and cuGraph builds | Where the cuVS build installs `libcuvs.so`; cuGraph uses this library rather than building its own copy. |
| `CUGRAPH_CMAKE_BUILD_DIR` | cuGraph build and downstream runtime | Where the cuGraph build outputs `libcugraph_c.so` and related runtime libraries. |
| `LD_LIBRARY_PATH` (+`$CUGRAPH_CMAKE_BUILD_DIR`, `$CUDF_INSTALL_DIR/lib`, `$CUVS_INSTALL_DIR/lib`, `$CUDA_HOME/lib64`) | test binaries, benchmarks, servers | Runtime resolution of the native cuDF/RMM, cuVS, and cuGraph libraries. |

:::warning[Keep the cuDF prefix views consistent]
`CUDF_CPP_BUILD_DIR` must be the matching CMake cache and
`CUDF_INSTALL_DIR`, `INSTALL_PREFIX`, and the cuDF entry in `LD_LIBRARY_PATH`
must all point at the **same** install tree. If the cache's install prefix or
the environment selectors drift (for example after moving a checkout), the
root helper stops before rebuilding and `cudf-nexus-sys` refuses stale native
metadata.
:::

## Dependency relationship

The `libcudf` build in `components/cudf` produces the stack's one shared
`librmm.so`. The `libcuvs` build in `components/cuvs` uses it, and the
`libcugraph` build in `components/cugraph` uses both `librmm.so` and
`libcuvs.so`. Build in that
order:

```text
cuDF/RMM → cuVS → cuGraph → DataFusion Nexus
```

## Validated against shell-state leakage

Preset variables in your `~/.bashrc`/`~/.zshrc` (an old `CUDF_INSTALL_DIR`, a
different `CUDA_HOME`, a stray `CUDAHOSTCXX`, conda hooks) can silently
redirect a build. This guide's env file was validated by running **every build
and test step** under `env -i` — an empty environment plus a stock system
`PATH` and `env.sh` — so that nothing outside the file influences the result:

```bash
env -i HOME="$HOME" USER="$USER" TERM=xterm \
  PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  bash -c "source $NEXUS_ROOT/env.sh && <build command>"
```

If a step fails for you in a normal shell but works under `env -i`, an
inherited variable is interfering; `env | grep -iE 'cuda|cudf|conda|prefix|hostcxx|^cc=|^cxx='`
usually finds it.

Continue with [Building from Source](./from-source.md).

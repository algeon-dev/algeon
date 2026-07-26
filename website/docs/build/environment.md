---
sidebar_position: 3
description: The complete environment-variable contract for building and running the stack, and how it was validated against shell-state leakage.
---

# Environment

The whole stack builds and runs with **one small environment file**. Save it
as `env.sh` in your checkout root (`$NEXUS_ROOT`) and source it in every
shell you build or test from.

```bash title="$NEXUS_ROOT/env.sh"
# Root that holds the three sibling checkouts.
nexus_env_source="${BASH_SOURCE[0]:-$0}"
nexus_env_dir=$(dirname "$nexus_env_source")
nexus_env_dir=$(cd "$nexus_env_dir" && pwd)
NEXUS_ROOT="${NEXUS_ROOT:-$nexus_env_dir}"
unset nexus_env_source nexus_env_dir
export NEXUS_ROOT

# --- Rust toolchain ---------------------------------------------------------
source "$HOME/.cargo/env"

# --- CUDA toolkit -----------------------------------------------------------
# CUDACXX / CUDAToolkit_ROOT are read by CMake (libcudf, libcugraph, the
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

# --- RAPIDS install prefix ---------------------------------------------------
# One prefix shared by the whole stack (see below).
export CUDF_INSTALL_DIR="$NEXUS_ROOT/cudf-nexus/cudf-install"
export INSTALL_PREFIX="$CUDF_INSTALL_DIR"

# --- Runtime library resolution ----------------------------------------------
export LD_LIBRARY_PATH="$CUDF_INSTALL_DIR/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
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
| `INSTALL_PREFIX` | `cudf/build.sh` | Where `./build.sh libcudf` installs the libcudf tree. The RAPIDS build script defaults to `$CONDA_PREFIX`, which is unset outside conda — this must point at `cudf-nexus/cudf-install`. |
| `CUDF_INSTALL_DIR` | `cudf-nexus-sys/build.rs`, `cugraph-nexus/scripts/build_libcugraph.sh` | How the Rust shim discovers the prebuilt libcudf, and how the cuGraph build resolves the **shared** RMM / rapids_logger packages. |
| `LD_LIBRARY_PATH` (+`$CUDF_INSTALL_DIR/lib`, `$CUDA_HOME/lib64`) | test binaries, benchmarks, servers | Runtime resolution of `libcudf.so`, `librmm.so`, `libcugraph.so`. `cudf-nexus-sys` bakes a `DT_RUNPATH` for its own consumers, but the install prefix on `LD_LIBRARY_PATH` is the documented contract for everything else. |

:::warning[Keep the three prefix views consistent]
`CUDF_INSTALL_DIR`, `INSTALL_PREFIX`, and the `LD_LIBRARY_PATH` entry must
all point at the **same** install tree. If they drift (for example after
moving a checkout), `cudf-nexus-sys` links one libcudf while tests load another —
the staleness check in `cudf-nexus-sys/build.rs` warns about exactly this.
:::

## Why one shared install prefix

`librmm.so` must exist **exactly once per process**. `datafusion-nexus
--features cugraph` loads libcudf and libcugraph together; if the cuGraph
build let CPM compile its own private RMM, both copies would bind RMM symbols
in the same process and corrupt memory-resource state.
`cugraph-nexus/scripts/build_libcugraph.sh` therefore resolves `rmm` and
`rapids_logger` from `CUDF_INSTALL_DIR` and **aborts** if a private copy is
still reachable — this is enforced, not advisory.

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

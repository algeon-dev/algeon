---
sidebar_position: 2
description: Install the upstream native-build prerequisites, then add the small set of tools required by this checkout.
---

# Prerequisites

Start with the [RAPIDS install guide](https://docs.rapids.ai/install/) for your
Linux distribution. It owns NVIDIA driver, CUDA toolkit, GPU support,
host-compiler, CMake, and Ninja compatibility. The upstream dependency
manifests are [cuDF](https://github.com/NVIDIA/cudf/blob/main/dependencies.yaml),
[cuVS](https://github.com/NVIDIA/cuvs/blob/main/dependencies.yaml), and
[cuGraph](https://github.com/rapidsai/cugraph/blob/main/dependencies.yaml).
The exact native sources in this checkout are the authority for this build;
their `dependencies.yaml` and `CMakeLists.txt` files take precedence if they
differ from those upstream references.

This page covers only what connects that upstream environment to DataFusion
Nexus. Linux is the only supported platform.

## Native build environment

The recursive checkout builds its pinned `libcudf`, `libcuvs`, and
`libcugraph` sources locally. Before continuing, use the RAPIDS guidance to
install a compatible NVIDIA driver, CUDA toolkit, host compiler, CMake, and
Ninja. `nvcc` must be available from the CUDA installation you intend to use.

Install the checkout-specific host tools:

```bash
sudo apt-get install -y build-essential git clang mold
```

The root [`.cargo/config.toml`](https://github.com/DataFusion-Nexus/datafusion-nexus/blob/main/.cargo/config.toml)
selects `clang` and `mold` for normal source builds.

The [build shell](./from-source.md#2-configure-the-build-shell) on the next
page sets `CC`, `CXX`, and `CUDAHOSTCXX`; the native build scripts and CMake
read those variables. The example uses `/usr/bin/gcc-14` and
`/usr/bin/g++-14`, so confirm both paths exist before starting, or decide now
which RAPIDS-compatible host compiler you will substitute in that block:

```bash
ls /usr/bin/gcc-14 /usr/bin/g++-14
```

### Pin the toolchain in Cargo `[env]`

The build shell exports these machine-specific toolchain variables into the
current shell only. Write them once into a Cargo `[env]` table outside the
checkout so every shell, and `scripts/check_all.sh`, build with the same
toolchain whether or not the build shell block was run: use
`<parent-of-checkout>/.cargo/config.toml` to cover every checkout and worktree
that [`scripts/dev/new_worktree.sh`](https://github.com/DataFusion-Nexus/datafusion-nexus/blob/main/scripts/dev/new_worktree.sh)
creates as siblings, or `~/.cargo/config.toml` to apply it to all of this
user's Cargo projects:

```toml
[env]
CC = "/usr/bin/gcc-14"
CXX = "/usr/bin/g++-14"
CUDACXX = "/usr/local/cuda/bin/nvcc"
CUDAHOSTCXX = "/usr/bin/g++-14"
CUDA_HOME = "/usr/local/cuda"
CUDA_PATH = "/usr/local/cuda"
CUDAToolkit_ROOT = "/usr/local/cuda"
CMAKE_CUDA_ARCHITECTURES = "NATIVE"
```

Match these values to the build shell block; do not set `force`, since a shell
that legitimately overrides one of these (for example, to select a different
CUDA architecture list) must still win.

Native binaries target the build GPU by default. When they must run on a
different GPU, choose the architecture list before the first native build as
described in [Building from Source](./from-source.md#choosing-cuda-architectures).

## Rust toolchain and test runner

Install Rust with the [official Rustup instructions](https://rustup.rs/). The
workspace declares its MSRV in `Cargo.toml` (currently Rust 1.95); nightly is
needed only for formatting. `scripts/check_all.sh` requires these exact
versions of the test runner and feature checker and exits before any work if
they differ:

```bash
rustup toolchain install nightly
cargo install --locked --version 0.9.144 cargo-nextest
cargo install --locked --version 0.6.45 cargo-hack
```

Enable nextest's run store once per machine, in its own user config (not this
repository's), so `-R latest` reruns and `cargo nextest replay` have runs to
work from:

```toml
# ~/.config/nextest/config.toml
[experimental]
record = true

[record]
enabled = true
```

Retention defaults to 30 days, 100 runs, or 1 GB, whichever comes first.

## Optional fixture dependency

Some `components/cudf` Parquet fixture tests use Python with PyArrow. It is
not needed to build or link the native libraries or Rust workspace. Install
PyArrow from its [official installation guide](https://arrow.apache.org/docs/python/install.html)
only when running that component's full fixture suite.

Continue with [Building from Source](./from-source.md).

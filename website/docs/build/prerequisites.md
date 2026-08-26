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
Nexus.

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
selects `clang` and `mold` for normal source builds. If the pinned native
sources require a compiler that is not your distribution default, install that
compiler as directed by their checked-out dependency files, then set it in the
[build shell](./from-source.md#2-configure-the-build-shell).

Native binaries target the build GPU by default. When they must run on a
different GPU, build the required architectures explicitly as described in
[Building from Source](./from-source.md#3-build-the-native-stack).

## Rust toolchain and test runner

Install Rust with the [official Rustup instructions](https://rustup.rs/). The
workspace declares its MSRV in `Cargo.toml`; nightly is needed only for
formatting. Then install the repository's canonical test runner:

```bash
rustup toolchain install nightly
cargo install --locked --version 0.9.140 cargo-nextest
cargo install --locked --version 0.6.45 cargo-hack
```

## Optional fixture dependency

Some `components/cudf` Parquet fixture tests use Python with PyArrow. It is
not needed to build or link the native libraries or Rust workspace. Install
PyArrow from its [official installation guide](https://arrow.apache.org/docs/python/install.html)
only when running that component's full fixture suite.

Continue with [Building from Source](./from-source.md).

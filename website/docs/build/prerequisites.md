---
sidebar_position: 2
description: System packages, CUDA toolkit, host compiler, and Rust toolchain needed before cloning anything — with each tool mapped to the layer that requires it.
---

# Prerequisites

Everything below installs system-wide, once. Commands are written for Ubuntu
24.04 (the RAPIDS-supported LTS); adapt package names for other distributions.

The two native libraries in the stack — **libcudf** and **libcugraph** — are
built from the RAPIDS 26.06 sources, so their toolchain requirements are the
upstream RAPIDS ones. Where a version matters, this page cites the upstream
requirement rather than pinning the exact build this guide happened to run on;
follow the [RAPIDS install guide](https://docs.rapids.ai/install/) and the
`dependencies.yaml` in [rapidsai/cudf](https://github.com/rapidsai/cudf) /
[rapidsai/cugraph](https://github.com/rapidsai/cugraph) as the source of truth.

## What each tool is for

| Tool | Required by | Notes |
| --- | --- | --- |
| NVIDIA driver + CUDA toolkit | libcudf build, libcugraph build (nvcc); every layer at runtime (GPU) | CUDA 12.2 through 13.2 |
| gcc/g++ **14** | libcudf build, libcugraph build (host compiler) | RAPIDS 26.06 pins gcc 14 |
| CMake ≥ 3.30.4 | libcudf build, libcugraph build, the `cudf-sys` C++ shim | hard floor in the C++ projects |
| Ninja | libcudf build, libcugraph build | RAPIDS build scripts default to it |
| Rust (stable + nightly) | `cudf-rs`, `cugraph-rs`, `datafusion-nexus` | nightly only for `rustfmt` |
| Python + pyarrow | a few `cudf-rs` Parquet **fixture tests** | **not** a build dependency |

## NVIDIA driver and CUDA toolkit

RAPIDS 26.06 supports **CUDA 12.2 through 13.2** (either the CUDA 12 or CUDA 13
line), with a driver new enough for the toolkit you install
([RAPIDS install](https://docs.rapids.ai/install/)):

- CUDA 12.2–12.9 → driver ≥ 525.60.13
- CUDA 13.0–13.2 → driver ≥ 580.65.06

Install the toolkit from NVIDIA's apt repository (the distro archive lags).
Pick a `cuda-toolkit-<major>-<minor>` package within the supported range that
matches your installed driver — for example a CUDA 13 toolkit:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-13-0    # 13-0/13-1/13-2 or a CUDA 12.2+ toolkit
```

This installs under `/usr/local/cuda-<version>` and maintains the
`/usr/local/cuda` alternatives symlink, which the
[environment file](./environment.md) points at. Verify:

```bash
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
/usr/local/cuda/bin/nvcc --version
readlink -f /usr/local/cuda    # confirm the symlink resolves to the toolkit you want
```

### GPU architecture

RAPIDS 26.06 supports NVIDIA **Volta or newer (compute capability 7.0+)**;
Pascal support was dropped in 24.02. The native libraries build for your own
GPU's architecture by default (`CMAKE_CUDA_ARCHITECTURES=NATIVE`) — you do not
need to know your compute capability up front, only to list it explicitly when
building binaries that must run on a *different* GPU model
([Building from Source](./from-source.md#2-build-libcudf)).

## Host compiler: gcc/g++ 14

```bash
sudo apt-get install -y build-essential git ninja-build gcc-14 g++-14 protobuf-compiler
```

This line also installs **ninja** (the generator every CMake build in the stack
uses; RAPIDS build scripts default to it), **git**, and **protoc**
(`protobuf-compiler`, required by the opt-in `substrait` feature that
`--all-features` builds — including `scripts/check_all.sh` and the canonical
`cargo nextest run --all-features` — enable).

RAPIDS 26.06 **pins gcc 14** as the host compiler for both native builds
(`gcc_linux-64=14.*` in the cudf and cugraph `dependencies.yaml`). Ubuntu
24.04's default is gcc 13, so install `gcc-14`/`g++-14` explicitly; the
[environment file](./environment.md) points `CC`/`CXX`/`CUDAHOSTCXX` at it.
There is no need to change the system default compiler.

### CUDA and host-compiler compatibility

`nvcc` rejects host compilers newer than the CUDA toolkit supports. CUDA 13.0
added support up to gcc 15, so the recommended website build pairing is CUDA
13.x with gcc 14. The supported CUDA range does not imply that one host
compiler works across every point release: CUDA 12.2, for example, supports
gcc only through 12.2. If using CUDA 12.x, choose a host compiler from that
toolkit's NVIDIA compatibility table and point `CC`, `CXX`, and `CUDAHOSTCXX`
at it. If several CUDA toolkits are installed, confirm which one
`/usr/local/cuda` resolves to before building — a stale symlink to an older
toolkit is a classic source of inscrutable `nvcc` errors.

### CMake ≥ 3.30.4

Both native builds (`libcudf`, `libcugraph`) and the `cudf-sys` shim declare
`cmake_minimum_required(VERSION 3.30.4)`. Ubuntu 24.04 ships 3.28, which fails
immediately — install a newer CMake from the
[Kitware apt repository](https://apt.kitware.com/):

```bash
test -f /usr/share/doc/kitware-archive-keyring/copyright ||
  wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null \
  | gpg --dearmor - | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ noble main' \
  | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null
sudo apt-get update && sudo apt-get install -y cmake
cmake --version   # ≥ 3.30.4
```

## Rust toolchain

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
rustup toolchain install nightly      # rustfmt.toml uses unstable options
cargo install --locked cargo-nextest  # canonical test runner
```

The Rust crates declare an MSRV in `Cargo.toml`; any current stable satisfies
it. The nightly toolchain is only used for `cargo +nightly fmt`.

## Python (test fixtures only)

A few `cudf-rs` Parquet fixture tests shell out to Python with pyarrow. This is
**not** a build dependency — libcudf, libcugraph, and every Rust crate compile
and link without it — so install it only if you intend to run the full
`cudf-rs` test suite:

```bash
python3 -m pip install --user pyarrow
```

Continue with [Environment](./environment.md).

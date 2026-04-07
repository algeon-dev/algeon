---
sidebar_position: 2
description: System packages, CUDA toolkit, host compiler, and Rust toolchain needed before cloning anything.
---

# Prerequisites

Everything below installs system-wide, once. Commands are for Ubuntu 24.04;
adapt package names for other distributions.

## NVIDIA driver and CUDA toolkit

You need a driver new enough for CUDA 13 and the CUDA 13.2 toolkit. Install
the toolkit from NVIDIA's apt repository (the distro archive lags):

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-13-2
```

This installs to `/usr/local/cuda-13.2` and maintains the
`/usr/local/cuda` alternatives symlink, which the
[environment file](./environment.md) points at. Verify:

```bash
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
/usr/local/cuda/bin/nvcc --version   # release 13.2
```

:::info[CUDA ↔ host-compiler compatibility]
`nvcc` rejects host compilers newer than it supports. CUDA 13.2 compiles
cleanly with gcc 13 (the Ubuntu 24.04 default). If your box has several CUDA
toolkits installed, confirm which one `/usr/local/cuda` resolves to with
`readlink -f /usr/local/cuda` before building — a stale symlink to an older
toolkit is a classic source of inscrutable `nvcc` errors.
:::

## Build tools

```bash
sudo apt-get install -y build-essential git ninja-build
```

- **gcc/g++ 13** — comes with `build-essential` on 24.04; this is the host
  compiler for all CUDA compilation.
- **ninja** — generator used by every CMake build in the stack.

### CMake ≥ 3.30.4

Every native build in the stack (`libcudf`, `libcugraph`, the `cudf-sys`
shim) declares `cmake_minimum_required(VERSION 3.30.4)`. Ubuntu 24.04 ships
3.28, which fails immediately — install from the
[Kitware apt repository](https://apt.kitware.com/) instead:

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
rustup toolchain install nightly   # rustfmt.toml uses unstable options
cargo install cargo-nextest        # canonical test runner
```

The Rust crates declare MSRV 1.94; any current stable works.

## Python (test fixtures only)

Some `cudf-rs` Parquet fixture tests shell out to Python with pyarrow:

```bash
python3 -m pip install --user pyarrow
```

## Network access

The CMake builds fetch pinned third-party sources at configure time via CPM
(rapids-cmake, kvikio, nvcomp, zstd, …), and Cargo fetches crates.io
dependencies. The first build of each layer needs outbound access to GitHub
and crates.io.

Continue with [Environment](./environment.md).

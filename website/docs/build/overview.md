---
sidebar_position: 1
slug: /build
title: Build & Test
description: What you are building, how the repositories fit together, and the upstream-defined requirements the guide targets.
---

# Overview

DataFusion Nexus sits above a feature-selected GPU stack. Relational execution
uses RAPIDS cuDF, and graph SQL adds cuGraph. The cuVS SQL surface can be
discovered and planned, but cannot run under bounded admission yet. A
full-feature source build still includes libcuvs because cuGraph links it, so
the native libraries must be built in a fixed, bottom-up order.

This guide walks the whole path: system prerequisites, environment variables,
cloning, building each layer, optionally packaging the Flight SQL server into a
Docker image, and running the verified test surface. The source-build path was
executed end-to-end in a clean sandbox with no inherited shell state (see
[Environment](./environment.md#validated-against-shell-state-leakage)); the
Docker path is documented separately in [Building with Docker](./docker.md).

## The stack

| Component | Role | Language | How it is consumed |
| --- | --- | --- | --- |
| Root adapter | DataFusion adapter, native engine (`nexus-query-engine`), workload contracts, benchmarks, Flight SQL server | Rust | `src/`, `crates/` |
| `components/cudf` | Safe Rust API over libcudf (`cudf-nexus`, `cudf-nexus-sys`, `rapids-interop`); pins the [`cudf` fork](https://github.com/DataFusion-Nexus/cudf) | Rust + C++/CUDA | one component + submodule |
| `components/cuvs` | Safe Rust API over libcuvs (`cuvs-nexus`, `cuvs-nexus-sys`); pins official [`cuVS`](https://github.com/rapidsai/cuvs) | Rust + C++/CUDA | one component + submodule |
| `components/cugraph` | Safe Rust API over libcugraph (`cugraph-nexus`, `cugraph-nexus-sys`); pins the [`cugraph` fork](https://github.com/DataFusion-Nexus/cugraph) | Rust + C++/CUDA | one component + submodule |

The root `.gitmodules` pins the three native sources and the root Cargo
workspace resolves all Rust components through `components/*`. Cargo fetches
Apache DataFusion and the remaining registry or Git dependencies automatically;
there is one product checkout and one recursive submodule update.

```text
datafusion-nexus/
├── components/cudf/cudf/       # pinned cuDF fork submodule
├── components/cugraph/cugraph/ # pinned cuGraph fork submodule
├── components/cuvs/cuvs/       # pinned official cuVS submodule
├── crates/ and src/             # root Rust workspace
└── target/native/
    ├── cudf-build/             # canonical cuDF CMake cache
    ├── cudf-install/           # canonical cuDF/RMM install prefix
    ├── cuvs-install/           # canonical cuVS install prefix
    └── cugraph-build/          # canonical cuGraph CMake build tree
```

## Build order

Keep this dependency model in mind:

```text
libcudf ── produces ──> librmm.so
                            ├──> libcuvs
                            └──> libcugraph <── libcuvs

datafusion-nexus uses the enabled cuDF, cuVS, and cuGraph Rust bindings.
```

The full-feature build order follows those dependencies:

1. **libcudf** — `bash scripts/build/build_libcudf.sh` from the root. It drives
   the pinned source at `components/cudf/cudf/`, keeps the CMake cache in
   `target/native/cudf-build/`, and installs `libcudf.so` plus the shared
   `librmm.so` into `target/native/cudf-install/`.
2. **cudf-nexus** — root-workspace `cargo build --all-features` /
   `cargo test --all-features`.
   `cudf-nexus-sys` compiles its C++
   shim against that native install.
3. **Shared cuDF prefix** — `scripts/build/bootstrap_cudf_install.sh`.
   Prepares the cuDF install for the cuVS and cuGraph builds without creating a
   second RMM.
4. **libcuvs** — `scripts/build/build_libcuvs.sh`. Uses the `librmm.so`
   produced by libcudf and installs `libcuvs.so` into `target/native/cuvs-install/`.
5. **cuvs-nexus** — root-workspace `cargo build --all-features` /
   `cargo test --all-features` against that
   canonical prefix.
6. **libcugraph** — `scripts/build/build_libcugraph.sh`. Uses the same
   `librmm.so` from libcudf and the `libcuvs.so` built in step 4.
7. **cugraph-nexus** — root-workspace `cargo build --all-features` /
   `cargo test --all-features` against
   `target/native/cugraph-build/`.
8. **datafusion-nexus** — `cargo build --all-features` for a compile check; bare
   `bash scripts/check_all.sh` for the canonical full pre-merge surface.
9. **Docker image** — optional packaging for the standalone Flight SQL server.
   `docker/stage-docker-payload.sh` stages the release binary and the locally built forked
   RAPIDS libraries; `docker/build-image.sh` then harvests the CUDA runtime
   libraries into a runnable image ([Building with Docker](./docker.md)).

## Requirements

The three native libraries are built from RAPIDS 26.06 sources, so their
toolchain requirements are the upstream RAPIDS ones. Select a mutually
compatible driver, CUDA toolkit, and host compiler; the supported ranges below
do not imply that every cross-product is valid.

| Component | Requirement | Source |
| --- | --- | --- |
| OS | A RAPIDS-supported Linux (this guide uses Ubuntu 24.04 LTS) | [RAPIDS install](https://docs.rapids.ai/install/) |
| GPU | NVIDIA Volta or newer — compute capability **7.0+** (`sm_70`) | RAPIDS install |
| NVIDIA driver | ≥ 525.60.13 (CUDA 12) / ≥ 580.65.06 (CUDA 13) | RAPIDS install |
| CUDA toolkit | 12.2 through 13.2 | cudf/cuvs/cugraph `dependencies.yaml` |
| Host compiler | gcc/g++ **14** | RAPIDS 26.06 `dependencies.yaml` |
| CMake | ≥ **3.30.4** | `cmake_minimum_required` in all three C++ projects |
| Ninja | build dependency (RAPIDS scripts default to it) | cudf/cuvs/cugraph `dependencies.yaml` |
| mold + clang | required linker for normal source Cargo builds | `datafusion-nexus/.cargo/config.toml` (+ host packages) |
| Rust | MSRV 1.94; nightly for `rustfmt` | `Cargo.toml` and rustup |

The libcudf/libcuvs/libcugraph builds target **your own GPU's architecture** by default
(`CMAKE_CUDA_ARCHITECTURES=NATIVE`), so nothing here is tied to a specific GPU
model. Build for additional architectures explicitly only when the binaries
must run on a different GPU ([Building from Source](./from-source.md#2-build-libcudf)).

Continue with [Prerequisites](./prerequisites.md).

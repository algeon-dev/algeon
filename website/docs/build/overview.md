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
| [`datafusion-nexus`](https://github.com/DataFusion-Nexus/datafusion-nexus) | DataFusion adapter, native engine (`nexus-query-engine`), workload contracts, benchmarks, Flight SQL server | Rust | cloned (this repo) |
| [`cudf-nexus`](https://github.com/DataFusion-Nexus/cudf-nexus) | Safe Rust API over libcudf (`cudf-nexus`, `cudf-nexus-sys`, `rapids-interop`); vendors the [`cudf` fork](https://github.com/DataFusion-Nexus/cudf) as a submodule | Rust + C++/CUDA | cloned (+ submodule) |
| [`cuvs-nexus`](https://github.com/DataFusion-Nexus/cuvs-nexus) | Safe Rust API over libcuvs (`cuvs-nexus`, `cuvs-nexus-sys`); vendors upstream [`cuVS`](https://github.com/rapidsai/cuvs) as a submodule | Rust + C++/CUDA | cloned (+ submodule) |
| [`cugraph-nexus`](https://github.com/DataFusion-Nexus/cugraph-nexus) | Safe Rust API over libcugraph (`cugraph-nexus`, `cugraph-nexus-sys`); vendors the [`cugraph` fork](https://github.com/DataFusion-Nexus/cugraph) as a submodule | Rust + C++/CUDA | cloned (+ submodule) |

`datafusion-nexus` pins `cudf`, `cuvs`, and `cugraph` to **sibling path
checkouts**, so these four repositories must sit next to each other in one root
directory. Cargo fetches Apache DataFusion and the remaining registry or Git
dependencies automatically; there is nothing else to clone.

```text
<NEXUS_ROOT>/
├── cudf-nexus/            # + cudf/ submodule (libcudf sources)
│   └── cudf-install/   # created by the libcudf build (step 2)
├── cuvs-nexus/            # + cuvs/ submodule (libcuvs sources)
│   └── cuvs-install/   # created by the libcuvs build (step 5)
├── cugraph-nexus/         # + cugraph/ submodule (libcugraph sources)
└── datafusion-nexus/
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

1. **libcudf** — `./build.sh libcudf` inside the `cudf` submodule. Installs
   `libcudf.so` and the shared `librmm.so` into `cudf-nexus/cudf-install/`.
2. **cudf-nexus** — `cargo build` / `cargo test`. `cudf-nexus-sys` compiles its C++
   shim against that native install.
3. **Shared cuDF prefix** — `cugraph-nexus/scripts/bootstrap_cudf_install.sh`.
   Prepares the cuDF install for the cuVS and cuGraph builds without creating a
   second RMM.
4. **libcuvs** — `cuvs-nexus/scripts/build_libcuvs.sh`. Uses the `librmm.so`
   produced by libcudf and installs `libcuvs.so` into `cuvs-nexus/cuvs-install/`.
5. **cuvs-nexus** — `cargo build` / `cargo test` against that build tree.
6. **libcugraph** — `cugraph-nexus/scripts/build_libcugraph.sh`. Uses the same
   `librmm.so` from libcudf and the `libcuvs.so` built in step 4.
7. **cugraph-nexus** — `cargo build` / `cargo test` against that build tree.
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

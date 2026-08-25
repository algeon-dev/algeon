---
sidebar_position: 1
slug: /build
title: Build & Test
description: What you are building, how the repository fits together, and the upstream-defined requirements the guide targets.
---

# Overview

DataFusion Nexus sits above a feature-selected GPU stack. Relational execution
uses RAPIDS cuDF, and graph SQL adds cuGraph. The cuVS SQL surface can be
discovered and planned, but cannot run under bounded admission yet. A
full-feature source build still includes libcuvs because cuGraph links it, so
the native libraries must be built in a fixed, bottom-up order.

This guide walks the whole path: system prerequisites, one recursive clone,
checkout-local environment selection, building each layer, running the verified
test surface, and optionally packaging the Flight SQL server into a Docker
image. [Building from Source](./from-source.md) owns that complete linear path;
the Docker packaging path follows it in
[Building with Docker](./docker.md).

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

1. **libcudf and shared RMM** — `bash scripts/build/build_libcudf.sh` drives
   `components/cudf/cudf`, installs into `target/native/cudf-install`, and
   prepares the one shared dependency prefix.
2. **libcuvs** — `bash scripts/build/build_libcuvs.sh` consumes that RMM and
   installs into `target/native/cuvs-install`.
3. **libcugraph** — `bash scripts/build/build_libcugraph.sh` consumes the same
   RMM and cuVS installs and builds into `target/native/cugraph-build`.
4. **Unified Rust workspace** — `cargo build --workspace --all-features`, then
   bare `bash scripts/check_all.sh` for the canonical full pre-merge surface.
5. **Docker image** — optional packaging for the standalone Flight SQL server.
   `docker/stage-docker-payload.sh` stages the release binary and the locally built forked
   RAPIDS libraries; `docker/build-image.sh` then harvests the CUDA runtime
   libraries into a runnable image ([Building with Docker](./docker.md)).

## Requirements

The checked-out submodule gitlinks are the native-version source of truth, and
their toolchain requirements are the upstream RAPIDS ones. Select a mutually
compatible driver, CUDA toolkit, and host compiler; the supported ranges below
do not imply that every cross-product is valid. To inspect the versions in a
checkout, read `components/{cudf,cugraph,cuvs}/*/VERSION`.

| Component | Requirement | Source |
| --- | --- | --- |
| OS | A RAPIDS-supported Linux (this guide uses Ubuntu 24.04 LTS) | [RAPIDS install](https://docs.rapids.ai/install/) |
| GPU | NVIDIA Volta or newer — compute capability **7.0+** (`sm_70`) | RAPIDS install |
| NVIDIA driver | ≥ 525.60.13 (CUDA 12) / ≥ 580.65.06 (CUDA 13) | RAPIDS install |
| CUDA toolkit | 12.2, 12.5, 12.8, 12.9, or 13.0 through 13.3 | pinned cudf/cuvs/cugraph `dependencies.yaml` files |
| Host compiler | gcc/g++ **14** | pinned native `dependencies.yaml` files |
| CMake | ≥ **4.0** | `cmake_minimum_required` in all three pinned C++ projects |
| Ninja | build dependency (RAPIDS scripts default to it) | cudf/cuvs/cugraph `dependencies.yaml` |
| mold + clang | required linker for normal source Cargo builds | `datafusion-nexus/.cargo/config.toml` (+ host packages) |
| Rust | MSRV 1.94; nightly for `rustfmt` | `Cargo.toml` and rustup |

The libcudf/libcuvs/libcugraph builds target **your own GPU's architecture** by default
(`CMAKE_CUDA_ARCHITECTURES=NATIVE`), so nothing here is tied to a specific GPU
model. Build for additional architectures explicitly only when the binaries
must run on a different GPU
([Building from Source](./from-source.md#build-cudf-and-the-shared-rmm-prefix)).

Continue with [Prerequisites](./prerequisites.md).

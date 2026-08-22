---
sidebar_position: 1
slug: /build
title: Build & Test
description: What you are building, how the repository fits together, and where native requirements are defined.
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
| Root adapter | DataFusion lowering, execution wrappers, report projection, workload contracts, benchmarks, and Flight SQL server | Rust | `src/`, adapter-support crates |
| `nexus-query-planning` | DataFusion-free IR, optimizer, capability, proof, admission contracts, and compiled pipeline | Rust | `crates/nexus-query-planning/` |
| `nexus-query-runtime` | Attempt lifecycle, query service, device ledger, immutable grants, cache, observation, and metrics | Rust | `crates/nexus-query-runtime/` |
| `nexus-query-engine` | Native execution composition and the compatible public facade over planning and runtime | Rust | `crates/nexus-query-engine/` |
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
├── crates/                      # planning, runtime, engine, and support crates
├── src/                         # DataFusion adapter
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

nexus-query-runtime depends on nexus-query-planning.
nexus-query-engine depends on both and preserves the public facade.
datafusion-nexus depends on that facade and uses the enabled cuDF, cuVS, and cuGraph Rust bindings.
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

The checked-out submodule gitlinks are the native-version source of truth.
Install the compatible GPU and native-build environment using the
[RAPIDS install guide](https://docs.rapids.ai/install/), then follow
[Prerequisites](./prerequisites.md) for this checkout's Rust, linker, and
test-runner requirements. The source-build guide owns the checkout-local
environment and cross-GPU build instructions.

Continue with [Prerequisites](./prerequisites.md).

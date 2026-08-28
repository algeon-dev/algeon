---
sidebar_position: 1
slug: /build
title: Build & Test
description: What you are building, how the repository fits together, and the order in which the pages of this guide are meant to be followed.
---

# Build & Test

DataFusion Nexus sits above a feature-selected GPU stack. Relational execution
uses RAPIDS cuDF, graph SQL adds cuGraph, and vector SQL adds cuVS. cuGraph
links libcuvs, so a full-feature source build always includes it, and the
native libraries must be built in a fixed, bottom-up order.

Follow the pages in this order:

1. [Prerequisites](./prerequisites.md) — the GPU, compiler, Rust, and
   test-runner environment.
2. [Building from Source](./from-source.md) — one recursive clone, one build
   shell, the three native builds, the Rust workspace, and the full gate.
3. [Tests](./tests.md) — the full gate in detail, partial edit loops, and
   maintainer diagnostics.
4. [Building with Docker](./docker.md) — optional packaging of the Flight SQL
   server after a source build.
5. [Troubleshooting](./troubleshooting.md) — native failures whose error text
   does not point at the cause.

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

```text
libcudf ── produces ──> librmm.so
                            ├──> libcuvs
                            └──> libcugraph <── libcuvs

nexus-query-runtime depends on nexus-query-planning.
nexus-query-engine depends on both and preserves the public facade.
datafusion-nexus depends on that facade and uses the enabled cuDF, cuVS, and cuGraph Rust bindings.
```

The full-feature build follows those dependencies, and each step below is
one command in [Building from Source](./from-source.md):

1. **libcudf and shared RMM** — `scripts/build/build_libcudf.sh` installs into
   `target/native/cudf-install`, the one shared dependency prefix.
2. **libcuvs** — `scripts/build/build_libcuvs.sh` consumes that RMM and installs
   into `target/native/cuvs-install`.
3. **libcugraph** — `scripts/build/build_libcugraph.sh` consumes the same RMM
   and cuVS installs and builds into `target/native/cugraph-build`.
4. **Rust workspace** — `cargo build --workspace --all-features`, then the
   single full gate `bash scripts/check_all.sh`, run bare.
5. **Docker image** — optional. `docker/stage-docker-payload.sh` stages the
   release binary and the locally built RAPIDS libraries;
   `docker/build-image.sh` produces the runnable image.

All three native layers share one `CMAKE_CUDA_ARCHITECTURES` value; the build
scripts record it in each library and reject a mismatched downstream build.
The source-build page explains how to choose it.

Continue with [Prerequisites](./prerequisites.md).

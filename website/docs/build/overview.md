---
sidebar_position: 1
slug: /build
title: Build & Test
description: What you are building, how the repository fits together, and the order in which the pages of this guide are meant to be followed.
---

# Build & Test

Algeon sits above a feature-selected GPU stack. Relational execution
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
| DataFusion adapter | DataFusion lowering, execution wrappers, report projection, workload contracts, benchmarks, and Flight SQL server | Rust | `crates/algeon-datafusion/src/`, adapter-support crates |
| `query-planning` | DataFusion-free IR, optimizer, capability, proof, admission contracts, and compiled pipeline | Rust | `crates/algeon-query-planning/` |
| `query-runtime` | Attempt lifecycle, query service, device ledger, immutable grants, cache, observation, and metrics | Rust | `crates/algeon-query-runtime/` |
| `query-engine` | Native execution composition and the compatible public facade over planning and runtime | Rust | `crates/algeon-query-engine/` |
| `components/cudf` | Safe Rust API over libcudf (`cudf`, `cudf-sys`, `rapids-interop`); pins the [`cudf` fork](https://github.com/algeon-dev/cudf) | Rust + C++/CUDA | one component + submodule |
| `components/cuvs` | Safe Rust API over libcuvs (`cuvs`, `cuvs-sys`); pins official [`cuVS`](https://github.com/rapidsai/cuvs) | Rust + C++/CUDA | one component + submodule |
| `components/cugraph` | Safe Rust API over libcugraph (`cugraph`, `cugraph-sys`); pins the [`cugraph` fork](https://github.com/algeon-dev/cugraph) | Rust + C++/CUDA | one component + submodule |

The root `.gitmodules` pins the three native sources and the root Cargo
workspace resolves all Rust components through `components/*`. Cargo fetches
Apache DataFusion and the remaining registry or Git dependencies automatically;
there is one product checkout and one recursive submodule update.

```text
algeon/
├── components/cudf/cudf/       # pinned cuDF fork submodule
├── components/cugraph/cugraph/ # pinned cuGraph fork submodule
├── components/cuvs/cuvs/       # pinned official cuVS submodule
├── crates/                      # planning, runtime, engine, and support crates
├── crates/algeon-datafusion/src/ # DataFusion adapter
└── target/native/
    ├── cudf-build/             # canonical cuDF CMake cache
    ├── cudf-install/           # canonical cuDF/RMM install prefix
    ├── cuvs-install/           # canonical cuVS install prefix
    ├── cugraph-build/          # canonical cuGraph CMake build tree
    └── cugraph-install/        # canonical cuGraph install prefix
```

## Build order

```text
libcudf ── produces ──> librmm.so
                            ├──> libcuvs
                            └──> libcugraph <── libcuvs

query-runtime depends on query-planning.
query-engine depends on both and preserves the public facade.
algeon-datafusion depends on that facade and uses the enabled cuDF, cuVS, and cuGraph Rust bindings.
```

The full-feature build follows those dependencies, and each step below is
one command in [Building from Source](./from-source.md):

1. **libcudf and shared RMM** — `scripts/build/build_libcudf.sh` installs into
   `target/native/cudf-install`, the one shared dependency prefix.
2. **libcuvs** — `scripts/build/build_libcuvs.sh` consumes that RMM and installs
   into `target/native/cuvs-install`.
3. **libcugraph** — `scripts/build/build_libcugraph.sh` consumes the same RMM
   and cuVS installs, builds in `target/native/cugraph-build`, and installs into
   `target/native/cugraph-install`.
4. **Rust workspace** — `cargo build --workspace --all-features`, then the
   single full gate `bash scripts/check_all.sh`, run bare.
5. **Docker image** — optional. `docker/stage-docker-payload.sh` stages the
   release binary and the locally built RAPIDS libraries;
   `docker/build-image.sh` produces the runnable image.

All three native layers share one `CMAKE_CUDA_ARCHITECTURES` value; the build
scripts record it in each library and reject a mismatched downstream build.
The source-build page explains how to choose it.

Continue with [Prerequisites](./prerequisites.md).

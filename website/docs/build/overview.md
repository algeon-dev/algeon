---
sidebar_position: 1
slug: /build
title: Build & Test
description: What you are building, how the repository fits together, and the order in which the pages of this guide are meant to be followed.
---

# Build & Test

Algeon sits above a feature-selected GPU stack. Relational execution
uses RAPIDS cuDF, graph SQL adds cuGraph, vector SQL adds cuVS, and cuML SQL
adds dimensionality reduction. cuGraph and cuML depend on the shared cuVS
install. A full-feature source build includes all four native layers.

Follow the pages in this order:

1. [Prerequisites](./prerequisites.md): the GPU, compiler, Rust, and
   test-runner environment.
2. [Building from Source](./from-source.md): one recursive clone, one build
   shell, the four native builds, the Rust workspace, and the full gate.
3. [Tests](./tests.md): focused checks, the local pre-merge gate, and
   acceptance runs.
4. [Building with Docker](./docker.md): optional packaging of the Flight SQL
   server after a source build.

## Run the server only {/* #run-the-server-only */}

To run the Flight SQL server without contributing, follow these steps and
skip the rest of this guide:

1. Check the [CPU requirement](./prerequisites.md#cpu-requirement) and
   install the CUDA toolkit, host compiler, and stable Rust from
   [Prerequisites](./prerequisites.md). Nightly Rust, `cargo-nextest`, and
   `cargo-hack` are not needed.
2. Clone the repository and configure the build shell as in steps 1 and 2 of
   [Building from Source](./from-source.md#1-clone-and-initialize-the-repository).
3. Build the native libraries in order:
   [cuDF](./from-source.md#build-cudf-and-the-shared-rmm-prefix),
   [cuVS](./from-source.md#build-cuvs),
   [cuGraph](./from-source.md#build-cugraph), and
   [cuML](./from-source.md#build-cuml).
4. Build the server:

   ```bash
   cargo build --release -p algeon-server --features cugraph,cuvs,cuml,nvml
   ```

5. Start `target/release/algeon_server`. It needs no configuration file and
   listens on `127.0.0.1:50051`.
6. Run a first query with the
   [Flight SQL quickstart](../integration/flight-sql-server.mdx#quickstart).

The workspace build with `--all-features` and the full gate in
[step 5 of Building from Source](./from-source.md#5-validate-the-checkout) are
for contributors.

## The stack

The [module ownership map](../design/index.mdx#crates-and-ownership) defines the
adapter, planning, runtime and engine boundaries. Their native dependencies are:

| Component | Role | Language | How it is consumed |
| --- | --- | --- | --- |
| `components/cudf` | Safe Rust API over libcudf (`cudf`, `cudf-sys`, `rapids-interop`); pins the [`cudf` fork](https://github.com/algeon-dev/cudf) | Rust + C++/CUDA | one component + submodule |
| `components/cuvs` | Safe Rust API over libcuvs (`cuvs`, `cuvs-sys`); pins the [`cuvs` fork](https://github.com/algeon-dev/cuvs) | Rust + C++/CUDA | one component + submodule |
| `components/cuml` | Safe Rust API over libcuml (`cuml`, `cuml-sys`); pins the cuML submodule | Rust + C++/CUDA | one component + submodule |
| `components/cugraph` | Safe Rust API over libcugraph (`cugraph`, `cugraph-sys`); pins the [`cugraph` fork](https://github.com/algeon-dev/cugraph) | Rust + C++/CUDA | one component + submodule |

The root `.gitmodules` pins the four native sources and the root Cargo
workspace resolves all Rust components through `components/*`. Cargo fetches
Apache DataFusion and the remaining registry or Git dependencies automatically;
there is one product checkout and one recursive submodule update.

```text
algeon/
├── components/cudf/cudf/       # pinned cuDF fork submodule
├── components/cugraph/cugraph/ # pinned cuGraph fork submodule
├── components/cuvs/cuvs/       # pinned cuVS fork submodule
├── components/cuml/cuml/       # pinned cuML submodule
├── crates/                      # planning, runtime, engine, and support crates
├── crates/algeon-datafusion/src/ # DataFusion adapter
└── target/native/
    ├── cudf-build/             # canonical cuDF CMake cache
    ├── cudf-install/           # canonical cuDF/RMM install prefix
    ├── cuvs-install/           # canonical cuVS install prefix
    ├── cuml-install/           # canonical cuML install prefix
    ├── cugraph-build/          # canonical cuGraph CMake build tree
    └── cugraph-install/        # canonical cuGraph install prefix
```

## Workspace and publication contracts

The root `.gitmodules`, `Cargo.toml` workspace and `Cargo.lock` are the only
native source registry, Rust workspace and lockfile. Components share them;
there are no component-local registries or workspaces. The root manifest's
`default-members` lists the default build surface; bench, tools and examples
are explicit non-default members. Platform and toolchain requirements are in
[Prerequisites](./prerequisites.md).

Published packages use the `algeon-` prefix, and crate directory names match
package names. `[workspace.dependencies]` retains bare dependency aliases
through `package = "algeon-…"`; Rust imports and `dep:` features keep those
identities. Refer to layers by their bare alias. Every renamed crate pins its
`[lib] name` explicitly to preserve imports and public API snapshot names.

Each crate declares `publish` directly: `publish = ["crates-io"]` for the
release closure and `publish = false` otherwise. Do not set
`[workspace.package].publish`. The allowlist and its validation belong to
[`check_workspace_publication.py`](https://github.com/algeon-dev/algeon/blob/main/scripts/ci/check_workspace_publication.py),
which rejects undeclared or mismatched publication settings.

The release closure includes the normal and build dependencies needed to
install `algeon-datafusion` from crates.io. Published implementation packages
keep their owner APIs for workspace consumers. The
[application API and documentation boundary](../design/index.mdx#api-documentation)
determines which interfaces require application-facing rustdoc.

[`algeon-build-support`](https://github.com/algeon-dev/algeon/blob/main/components/algeon-build-support/Cargo.toml)
(library `rapids_build_support`) is the shared build-dependency for packages
whose build scripts link RAPIDS libraries. It remains std-only and cannot
depend on crates built after it. Its validation is described in the
[harness maintenance reference](https://github.com/algeon-dev/algeon/blob/main/scripts/test-harness.md#doctests-and-build-support).

## Build order

```text
libcudf ── produces ──> librmm.so
                            ├──> libcuvs ──> libcuml
                            └──> libcugraph <── libcuvs

query-runtime depends on query-planning.
query-engine depends on both.
algeon-datafusion depends on all three and uses the enabled cuDF, cuVS, cuML, and cuGraph Rust bindings.
```

The full-feature build follows those dependencies, and each step below is
one command in [Building from Source](./from-source.md):

1. **libcudf and shared RMM**: `scripts/build/build_libcudf.sh` installs into
   `target/native/cudf-install`, the one shared dependency prefix.
2. **libcuvs**: `scripts/build/build_libcuvs.sh` consumes that RMM and installs
   into `target/native/cuvs-install`.
3. **libcugraph**: `scripts/build/build_libcugraph.sh` consumes the same RMM
   and cuVS installs, builds in `target/native/cugraph-build`, and installs into
   `target/native/cugraph-install`.
4. **libcuml**: `scripts/build/build_libcuml.sh` installs cuML and its native dependencies into `target/native/cuml-install` against the shared cuVS and RMM installs.
5. **Rust workspace**: `cargo build --workspace --all-features`, then the
   single full gate `bash scripts/check_all.sh`, run bare.
6. **Docker image**: optional. `docker/stage-docker-payload.sh` stages the
   release binary and the locally built RAPIDS libraries;
   `docker/build-image.sh` produces the runnable image.

The native layers share one `CMAKE_CUDA_ARCHITECTURES` value; the build
scripts record it in each library and reject a mismatched downstream build.
The source-build page explains how to choose it.

Continue with [Prerequisites](./prerequisites.md).

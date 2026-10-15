---
sidebar_position: 1
slug: /build
title: Build & Test
description: What you are building, how the repository fits together, and the order in which the pages of this guide are meant to be followed.
---

# Build & Test

Algeon sits above a feature-selected GPU stack. Relational execution
uses RAPIDS cuDF, graph SQL adds cuGraph, vector SQL adds cuVS, and cuML SQL
adds machine learning algorithms and forest inference. cuGraph and cuML depend
on the shared cuVS install. A full-feature source build includes all four
native layers.

For an application that embeds Algeon, start with
[DataFusion Session](../integration/datafusion-session.mdx). It links to
[Building for an Application](./rust-dependency.md) for the native libraries
the application's Cargo features need and for the loader environment of its
binary.

To run the Flight SQL server without contributing, follow
[Run as a Flight SQL server](#run-as-a-flight-sql-server). For repository
development, follow these pages in order:

1. [Prerequisites](./prerequisites.md): the GPU, compiler, Rust, and
   test-runner environment.
2. [Building from Source](./from-source.md): one recursive clone, one build
   shell, the four native builds, the Rust workspace, and the full gate.
3. [Tests](./tests.md): focused checks, the local pre-merge gate, and
   acceptance runs.
4. [Building with Docker](./docker.md): optional packaging of the Flight SQL
   server after a source build.

## Run as a Flight SQL server {/* #run-as-a-flight-sql-server */}

Follow these steps and skip the rest of this guide:

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
   cargo build --release -p algeon-server --features cugraph,cuvs,cuml
   ```

   For device diagnostics, build with `--features cugraph,cuvs,cuml,nvml`
   instead ([NVML diagnostics](./rust-dependency.md#nvml-diagnostics)).

5. Start `target/release/algeon_server`. It needs no configuration file and
   listens on `127.0.0.1:50051`.
6. Run a first query with the
   [Flight SQL quickstart](../integration/flight-sql-server.mdx#quickstart).

The workspace build with `--all-features` and the full gate in
[step 5 of Building from Source](./from-source.md#5-validate-the-checkout) are
for contributors.

## The stack

The [module ownership map](../design/index.mdx#crates-and-ownership) defines the
adapter, planning, runtime and engine boundaries. Their native dependencies are
[`components/cudf`](https://github.com/algeon-dev/algeon/blob/main/components/cudf/README.md),
`components/cuvs`, `components/cuml` and `components/cugraph`. These hold the
safe Rust APIs `cudf`/`cudf-sys` (with `rapids-interop`), `cuvs`/`cuvs-sys`,
`cuml`/`cuml-sys` and `cugraph`/`cugraph-sys`, and pin the
[cuDF](https://github.com/algeon-dev/cudf),
[cuVS](https://github.com/algeon-dev/cuvs) and
[cuGraph](https://github.com/algeon-dev/cugraph) forks and the cuML source as
submodules.

The root Cargo workspace resolves all Rust components through `components/*`.
Cargo fetches Apache DataFusion and the remaining registry or Git dependencies
automatically. The submodule paths are listed in
[step 1 of Building from Source](./from-source.md#1-clone-and-initialize-the-repository).

## Workspace

One root `Cargo.toml` workspace, `Cargo.lock` and `.gitmodules` cover every
component; there are no component-local workspaces, lockfiles or source
registries. The root manifest's `default-members` key specifies the primary
workspace build surface, while the `bench`, `tools`, and `examples` packages
remain explicit non-default members. Package naming, `publish` fields, the release closure and the shared
build-support crate follow the
[workspace and publication contracts](https://github.com/algeon-dev/algeon/blob/main/scripts/test-harness.md#workspace-and-publication-contracts).
Platform and toolchain requirements are in [Prerequisites](./prerequisites.md).

## Build order

Each step of the full-feature build is one command in
[Building from Source](./from-source.md):

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
6. **Docker image**: optional. `docker/build-image.sh` stages the release
   binary and the locally built RAPIDS libraries, then builds the runnable
   image from them.

The native layers share one `CMAKE_CUDA_ARCHITECTURES` value. The build
scripts record it in each library and reject a mismatched downstream build.
The source-build page explains how to choose it.

Continue with [Prerequisites](./prerequisites.md).

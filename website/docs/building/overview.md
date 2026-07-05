---
sidebar_position: 1
description: What you are building, how the repositories fit together, and the upstream-defined requirements the guide targets.
---

# Overview

DataFusion Nexus is the top of a GPU query stack. The adapter crate lowers
DataFusion physical plans into a native IR and executes them on RAPIDS cuDF
(and, optionally, cuGraph) through Rust bindings — so building it from source
means building **two native CUDA libraries** (libcudf, libcugraph) and **three
Rust workspaces** in a fixed, bottom-up order.

This guide walks the whole path: system prerequisites, environment variables,
cloning, building each layer, and running the verified test surface. Every
command in it was executed end-to-end in a clean sandbox with no inherited
shell state (see [Environment](./environment.md#validated-against-shell-state-leakage)).

## The stack

| Component | Role | Language | How it is consumed |
| --- | --- | --- | --- |
| [`datafusion-nexus`](https://github.com/morristai/datafusion-nexus) | DataFusion physical-plan adapter, native IR (`nexus-query-engine`), benchmarks, Flight SQL server | Rust | cloned (this repo) |
| [`cudf-rs`](https://github.com/morristai/cudf-rs) | Safe Rust API over libcudf (`cudf`, `cudf-sys`, `rapids-interop`); vendors the [`cudf` fork](https://github.com/DataFusion-Nexus/cudf) as a submodule | Rust + C++/CUDA | cloned (+ submodule) |
| [`cugraph-rs`](https://github.com/morristai/cugraph-rs) | Safe Rust API over libcugraph (`cugraph`, `cugraph-sys`); vendors the [`cugraph` fork](https://github.com/DataFusion-Nexus/cugraph) as a submodule | Rust + C++/CUDA | cloned (+ submodule) |

`datafusion-nexus` pins `cudf` and `cugraph` to **sibling path checkouts**, so
these three repositories must sit next to each other in one root directory.
Everything else — Apache DataFusion and the rest of the Rust dependency graph —
is an ordinary published crate that Cargo fetches automatically; there is
nothing else to clone.

```text
<NEXUS_ROOT>/
├── cudf-rs/            # + cudf/ submodule (libcudf sources)
│   └── cudf-install/   # created by the libcudf build (step 2)
├── cugraph-rs/         # + cugraph/ submodule (libcugraph sources)
└── datafusion-nexus/
```

## Build order

The layers are built bottom-up; each step consumes the artifacts of the
previous one.

1. **libcudf** — `./build.sh libcudf` inside the `cudf` submodule. Installs
   `libcudf.so`, a shared `librmm.so`, and their CMake packages into
   `cudf-rs/cudf-install/`. (It does **not** install `rapids_logger`/`nvtx3`;
   those are added in step 3 — see below.)
2. **cudf-rs** — `cargo build` / `cargo test`. `cudf-sys` compiles its C++
   shim against the install tree discovered through `CUDF_INSTALL_DIR`, linking
   the `.so` files directly (no CMake `find_package`), so cudf + rmm is all it
   needs.
3. **libcugraph** — `cugraph-rs/scripts/build_libcugraph.sh`. First runs
   `bootstrap_cudf_install.sh`, which augments the cudf prefix with the
   `nvtx3`, `rapids_logger`, `fmt`, and `spdlog` CMake packages that RMM's
   dependency closure needs (a `find_package(rmm)` from cuGraph requires them,
   but the libcudf install omits them). Then builds `libcugraph.so`, resolving
   RMM and rapids_logger **from the libcudf install prefix** so that exactly one
   `librmm.so` exists per process.
4. **cugraph-rs** — `cargo build` / `cargo test` against that build tree.
5. **datafusion-nexus** — `cargo build` plus the nextest suite. Cargo pulls the
   remaining dependencies (including Apache DataFusion) from crates.io.

## Requirements

The two native libraries are built from RAPIDS 26.06 sources, so their
toolchain requirements are the upstream RAPIDS ones. These are **requirements**,
not the single configuration the guide ran on — any combination within these
ranges is expected to behave the same.

| Component | Requirement | Source |
| --- | --- | --- |
| OS | A RAPIDS-supported Linux (this guide uses Ubuntu 24.04 LTS) | [RAPIDS install](https://docs.rapids.ai/install/) |
| GPU | NVIDIA Volta or newer — compute capability **7.0+** (`sm_70`) | RAPIDS install |
| NVIDIA driver | ≥ 525.60.13 (CUDA 12) / ≥ 580.65.06 (CUDA 13) | RAPIDS install |
| CUDA toolkit | 12.2 through 13.2 | cudf/cugraph `dependencies.yaml` |
| Host compiler | gcc/g++ **14** | RAPIDS 26.06 `dependencies.yaml` |
| CMake | ≥ **3.30.4** | `cmake_minimum_required` in both C++ projects |
| Ninja | build dependency (RAPIDS scripts default to it) | cudf/cugraph `dependencies.yaml` |
| Rust | current stable (MSRV in `Cargo.toml`) + nightly for `rustfmt` | this repo |

The libcudf/libcugraph builds target **your own GPU's architecture** by default
(`CMAKE_CUDA_ARCHITECTURES=NATIVE`), so nothing here is tied to a specific GPU
model. Build for additional architectures explicitly only when the binaries
must run on a different GPU ([Building from Source](./from-source.md#2-build-libcudf)).

Continue with [Prerequisites](./prerequisites.md).

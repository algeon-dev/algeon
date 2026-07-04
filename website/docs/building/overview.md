---
sidebar_position: 1
description: What you are building, how the five repositories fit together, and the exact configuration this guide was verified on.
---

# Overview

DataFusion Nexus is the top of a five-repository stack. The adapter crate
lowers DataFusion physical plans into a native IR and executes them on RAPIDS
cuDF (and, optionally, cuGraph) through Rust bindings — so building it from
source means building two native CUDA libraries and three Rust workspaces in a
fixed order.

This guide walks the whole path: system prerequisites, environment variables,
cloning, building each layer, and running the verified test surface. Every
command in it was executed end-to-end on the configuration below, in a clean
sandbox with no inherited shell state.

## The stack

| Repository | Role | Language |
| --- | --- | --- |
| [`datafusion-nexus`](https://github.com/morristai/datafusion-nexus) | DataFusion physical-plan adapter, native IR (`nexus-query-engine`), benchmarks, Flight SQL server | Rust |
| [`cudf-rs`](https://github.com/morristai/cudf-rs) | Safe Rust API over libcudf (`cudf`, `cudf-sys`, `rapids-interop`); vendors the [`cudf` fork](https://github.com/DataFusion-Nexus/cudf) as a submodule | Rust + C++/CUDA |
| [`cugraph-rs`](https://github.com/morristai/cugraph-rs) | Safe Rust API over libcugraph (`cugraph`, `cugraph-sys`); vendors the [`cugraph` fork](https://github.com/DataFusion-Nexus/cugraph) as a submodule | Rust + C++/CUDA |
| [`datafusion`](https://github.com/apache/datafusion) | Upstream Apache DataFusion, consumed as a path dependency | Rust |
| [`iceberg-rust`](https://github.com/morristai/iceberg-rust) | Iceberg catalog/storage crates (fork branch with DataFusion 54 support), consumed as a path dependency | Rust |

`datafusion-nexus` pins its heavyweight dependencies to **sibling path
checkouts**, not crates.io releases. The five repositories must sit next to
each other in one root directory:

```text
<NEXUS_ROOT>/
├── cudf-rs/            # + cudf/ submodule (libcudf sources)
│   └── cudf-install/   # created by the libcudf build (step 2)
├── cugraph-rs/         # + cugraph/ submodule (libcugraph sources)
├── datafusion/
├── iceberg-rust/
└── datafusion-nexus/
```

## Build order

The layers must be built bottom-up; each step consumes the artifacts of the
previous one.

1. **libcudf** — `./build.sh libcudf` inside the `cudf` submodule. Installs
   `libcudf.so`, a shared `librmm.so`, a shared `librapids_logger.so`, and
   their CMake packages into `cudf-rs/cudf-install/`.
2. **cudf-rs** — `cargo build` / `cargo test`. `cudf-sys` compiles its C++
   shim against the install tree discovered through `CUDF_INSTALL_DIR`.
3. **libcugraph** — `cugraph-rs/scripts/build_cpp.sh`. Builds
   `libcugraph.so`, resolving RMM and rapids_logger **from the libcudf
   install prefix** so that exactly one `librmm.so` exists per process.
4. **cugraph-rs** — `cargo build` / `cargo test` against that build tree.
5. **datafusion-nexus** — `cargo build --all-features` plus the nextest
   suite; pulls in the `datafusion` and `iceberg-rust` path checkouts.

## Verified configuration

| Component | Verified with |
| --- | --- |
| OS | Ubuntu 24.04.4 LTS (kernel 6.17) |
| GPU | NVIDIA GeForce RTX 5070 Ti, 16 GB (compute capability 12.0 / `sm_120`) |
| NVIDIA driver | 595.71 |
| CUDA toolkit | 13.2 (`nvcc` V13.2.78) |
| Host compiler | gcc/g++ 13.3 (Ubuntu 24.04 default) |
| CMake | 4.3.4 (Kitware apt) — the stack requires ≥ 3.30.4 |
| Ninja | 1.11.1 (`ninja-build`) |
| Rust | stable 1.96.1 (MSRV 1.94) + a nightly toolchain for `rustfmt` |
| Python | 3.12 + pyarrow 23 (test fixtures only) |

Any recent Linux with a CUDA-13-capable driver should behave the same; the
version columns above are the combination this guide was actually validated
on, not minimums (except where a minimum is called out).

Continue with [Prerequisites](./prerequisites.md).

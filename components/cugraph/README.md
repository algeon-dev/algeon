# cugraph-nexus

`cugraph-nexus` is a Rust GPU graph layer built on NVIDIA [cuGraph](https://github.com/DataFusion-Nexus/cugraph).

This repository is not meant to be only a thin set of `libcugraph` bindings. The
raw C API bindings live in `cugraph-nexus-sys/` and publish as `cugraph-nexus-sys`, but the main goal is to provide the
service-facing pieces needed by backend systems, including DataFusion-Nexus:

- safe Rust ownership around cuGraph resources and GPU execution context
- cudf-backed graph ingestion and result conversion
- graph algorithm APIs that can be embedded in query and service runtimes
- structured errors that preserve machine-readable failure context

## Workspace

- `cugraph-nexus-sys/` - raw bindgen output for the cuGraph C API, published as `cugraph-nexus-sys` (import `cugraph_sys`)
- `cugraph-nexus/` - safe Rust crate published as `cugraph-nexus` (import `cugraph`)
- `cugraph/` - upstream NVIDIA cuGraph source submodule used for local native builds

## Scope

This project focuses on backend integration rather than wrapping every cuGraph
surface immediately. Work that clarifies resource ownership, cudf interop,
algorithm contracts, error semantics, or DataFusion-Nexus integration belongs
here.

This project is not a standalone graph database, a replacement for Python
cuGraph, or a way to hide the native CUDA/RAPIDS deployment requirements.

## Building libcugraph locally

`cugraph-nexus-sys` builds native cuGraph (`cugraph/cpp`) and resolves RMM and
`rapids_logger` from the cudf install prefix (`CUDF_INSTALL_DIR`) 
so cudf and cugraph share **one** RMM build.

Because `cugraph-nexus-sys` does a real `find_package(rmm)` against that prefix, the
prefix must export RMM's full CMake dependency closure. The upstream libcudf
install ships only `cudf` + `rmm` (+ `cccl`, `gtest`); it does **not** install
`nvtx3`, `rapids_logger`, `fmt`, or `spdlog`, which `rmm-dependencies.cmake`
`find_dependency()`s. (`cudf-nexus` itself never trips on this because `cudf-nexus-sys` links
the `.so` files directly without a CMake configure.)

Build native libcugraph with one command (needs `cmake`, `ninja`, `nvcc`):

```bash
CUDF_INSTALL_DIR=/path/to/cudf-nexus/cudf-install scripts/build_libcugraph.sh
```

`scripts/build_libcugraph.sh` is a thin wrapper around the upstream
`cugraph/build.sh`: it makes the cudf prefix self-contained, forces cuGraph to
resolve RMM and rapids_logger from it (one shared RMM), builds `libcugraph` into
`cugraph/cpp/build`, and verifies no private RMM copy leaked. Set
`CMAKE_CUDA_ARCHITECTURES` to override the default `NATIVE` arch.

The self-contained step is also available standalone and is idempotent — it
installs the missing CMake packages from the cudf C++ build tree
(`cudf/cpp/build/_deps/*-build`):

```bash
CUDF_INSTALL_DIR=/path/to/cudf-nexus/cudf-install scripts/bootstrap_cudf_install.sh
```

Run the tests with both env vars set (`libcudf.so` is a transitive dependency,
and `DT_RUNPATH` does not cascade to dependencies-of-dependencies):

```bash
CUDF_INSTALL_DIR=/path/to/cudf-nexus/cudf-install \
LD_LIBRARY_PATH=/path/to/cudf-nexus/cudf-install/lib:/path/to/cugraph-nexus/cugraph/cpp/build \
cargo test --all-features --no-fail-fast
```

## Status

This project is in early development. API stability is less important than
getting the service-facing ownership, execution, memory, and interop contracts
right. Contributions are welcome. Please start every PR with an issue that explains
why the change is needed, and include the test environment and exact checks you
ran in the PR description. See [CONTRIBUTING.md](CONTRIBUTING.md) for the
short contribution policy.

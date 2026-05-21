# cugraph-rs

`cugraph-rs` is a Rust GPU graph layer built on NVIDIA [cuGraph](https://github.com/DataFusion-Nexus/cugraph).

This repository is not meant to be only a thin set of `libcugraph` bindings. The
raw C API bindings exist in `cugraph-sys`, but the main goal is to provide the
service-facing pieces needed by backend systems, including DataFusion-Nexus:

- safe Rust ownership around cuGraph resources and GPU execution context
- cudf-backed graph ingestion and result conversion
- graph algorithm APIs that can be embedded in query and service runtimes
- structured errors that preserve machine-readable failure context

## Workspace

- `cugraph-sys` - raw bindgen output for the cuGraph C API
- `cugraph-rs` - safe Rust crate published as `cugraph`
- `cugraph/` - upstream NVIDIA cuGraph source submodule used for local native builds

## Scope

This project focuses on backend integration rather than wrapping every cuGraph
surface immediately. Work that clarifies resource ownership, cudf interop,
algorithm contracts, error semantics, or DataFusion-Nexus integration belongs
here.

This project is not a standalone graph database, a replacement for Python
cuGraph, or a way to hide the native CUDA/RAPIDS deployment requirements.

## Status

This project is in early development. API stability is less important than
getting the service-facing ownership, execution, memory, and interop contracts
right. Contributions are welcome. Please start every PR with an issue that explains
why the change is needed, and include the test environment and exact checks you
ran in the PR description. See [CONTRIBUTING.md](CONTRIBUTING.md) for the
short contribution policy.

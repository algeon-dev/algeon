# cugraph-nexus

> [!IMPORTANT]
> Development has moved to
> [`DataFusion-Nexus/datafusion-nexus`](https://github.com/DataFusion-Nexus/datafusion-nexus/tree/main/components/cugraph).
> This repository is retained as a read-only historical archive. Clone the
> monorepo recursively for the current source, build tooling, and tests.

`cugraph-nexus` is a Rust GPU graph analytics workspace built around [`libcugraph`](https://github.com/DataFusion-Nexus/cugraph).
Its main direction is to make cuGraph usable as part of backend services and query execution systems for
[DataFusion-Nexus](https://github.com/DataFusion-Nexus/DataFusion-Nexus). It focuses on safe Rust APIs for GPU graph
processing while keeping ownership, memory movement, execution context, synchronization, and interop
behavior explicit enough for service-side integration.

## Focus

- Safe, idiomatic Rust wrappers around `libcugraph`.
- Backend-service and query-engine integration, rather than just direct API bindings.
- cudf-backed graph ingestion and result conversion.
- Graph algorithm APIs that can be embedded in query and service runtimes.
- Structured errors that preserve machine-readable failure context across the GPU boundary.

## Workspace

- `cugraph-nexus/` - the safe public Rust API crate, published as `cugraph-nexus` (import `cugraph`).
- `cugraph-nexus-sys/` - low-level FFI bindings around the cuGraph C API (import `cugraph_sys`).
- `cugraph/` - the pinned upstream **[forked cuGraph](https://github.com/DataFusion-Nexus/cugraph)** source submodule used to build `libcugraph`.

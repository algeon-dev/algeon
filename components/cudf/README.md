# cudf-nexus

> [!IMPORTANT]
> Development has moved to
> [`DataFusion-Nexus/datafusion-nexus`](https://github.com/DataFusion-Nexus/datafusion-nexus/tree/main/components/cudf).
> This repository is retained as a read-only historical archive. Clone the
> monorepo recursively for the current source, build tooling, and tests.

`cudf-nexus` is a Rust GPU DataFrame workspace built around [`libcudf`](https://github.com/DataFusion-Nexus/cudf).
Its main direction is to make cuDF usable as part of backend services and query execution systems for 
[DataFusion-Nexus](https://github.com/DataFusion-Nexus/DataFusion-Nexus). It focuses on safe Rust APIs for GPU data
processing while keeping ownership, memory movement, execution context, synchronization, and interop
behavior explicit enough for service-side integration.

## Focus

- Safe, idiomatic Rust wrappers around `libcudf`.
- Backend-service and query-engine integration, rather than just direct API bindings.
- Runtime and execution-context abstractions that can be honored by real compute and I/O paths.
- Clear separation between borrowing, shared ownership, device copies, host materialization, and external interoperability contracts.
- A shared RAPIDS Rust interoperability vocabulary for DataFusion-Nexus and other service components.

## Workspace

- `cudf-nexus/` - the safe public Rust API crate, published as `cudf-nexus` (import `cudf`).
- `cudf-nexus-sys/` - low-level FFI bindings and the native C shim around `libcudf` (import `cudf_sys`).
- `rapids-interop/` - shared RAPIDS Rust interop contract types and traits.
- `cudf/` - the pinned upstream **[forked cuDF](https://github.com/DataFusion-Nexus/cudf)** source submodule used to build `libcudf`.

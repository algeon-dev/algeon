# cuvs-nexus

> [!IMPORTANT]
> Development has moved to
> [`DataFusion-Nexus/datafusion-nexus`](https://github.com/DataFusion-Nexus/datafusion-nexus/tree/main/components/cuvs).
> This repository is retained as a read-only historical archive. Clone the
> monorepo recursively for the current source, build tooling, and tests.

`cuvs-nexus` is a Rust GPU vector search and clustering workspace built around [`libcuvs`](https://github.com/rapidsai/cuvs).
Its main direction is to make cuVS usable as part of backend services and query execution systems for
[DataFusion-Nexus](https://github.com/DataFusion-Nexus/DataFusion-Nexus). It focuses on safe Rust APIs for GPU vector
search and clustering while keeping ownership, memory movement, execution context, synchronization, and interop
behavior explicit enough for service-side integration.

## Focus

- Safe, idiomatic Rust wrappers around `libcuvs` for brute-force k-NN, k-means, and PCA.
- Backend-service and query-engine integration, rather than just direct API bindings.
- Exact binding to a `cudf::Runtime` device, stream, and memory resource.
- Device-resident cuDF interop with explicit ownership and synchronization.
- Checked FFI, ABI, tensor layout, and algorithm-output contracts.

## Workspace

- `cuvs-nexus/` - the safe public Rust API crate, published as `cuvs-nexus` (import `cuvs`).
- `cuvs-nexus-sys/` - generated low-level bindings and native build discovery for `libcuvs` (import `cuvs_sys`).
- `cuvs/` - the pinned official [cuVS](https://github.com/rapidsai/cuvs) source submodule used to build the native libraries.
- `scripts/` - operational entry points for source bootstrap and the canonical native build.
- `tools/xtask/` - unpublished binding and verification maintenance tooling.

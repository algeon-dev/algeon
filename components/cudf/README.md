# cudf-nexus

`cudf-nexus` is a Rust GPU DataFrame workspace built around [`libcudf`](https://github.com/DataFusion-Nexus/cudf), but it is
not intended to be only a thin Rust binding layer. Its main direction is to
make cuDF usable as part of backend services and query execution systems,
including DataFusion-Nexus.

The project focuses on safe Rust APIs for GPU data processing while keeping
ownership, memory movement, execution context, synchronization, and interop
behavior explicit enough for service-side integration.

## Focus

- Safe, idiomatic Rust wrappers over `libcudf`.
- Backend-service and query-engine integration, not just direct API binding.
- Runtime and execution-context surfaces that can be honored by real compute
  and IO paths.
- Clear separation between borrowing, shared ownership, device copies,
  host materialization, and external interop contracts.
- Shared RAPIDS Rust interop vocabulary for DataFusion-Nexus and other
  service components.

## Workspace

- `cudf-nexus/` - the safe public Rust API crate, published as `cudf-nexus` (import `cudf`).
- `cudf-nexus-sys/` - low-level FFI bindings and the native C shim around `libcudf` (import `cudf_sys`).
- `rapids-interop/` - shared RAPIDS Rust interop contract types and traits.
- `cudf/` - the pinned upstream **[forked cuDF](https://github.com/DataFusion-Nexus/cudf)** source submodule used to build `libcudf`.

## Status

This project is in early development. API stability is less important than
getting the service-facing ownership, execution, memory, and interop contracts
right. Contributions are welcome. Please start every PR with an issue that explains
why the change is needed, and include the test environment and exact checks you
ran in the PR description. See [CONTRIBUTING.md](CONTRIBUTING.md) for the
short contribution policy.

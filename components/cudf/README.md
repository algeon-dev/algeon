# cuDF component

`components/cudf` is the DataFusion Nexus Rust integration component built
around [`libcudf`](https://github.com/DataFusion-Nexus/cudf). It is part of the
root Cargo workspace; it is not a second repository or workspace.
Its main direction is to make cuDF usable as part of backend services and query execution systems for
[DataFusion Nexus](https://github.com/DataFusion-Nexus/datafusion-nexus/tree/main/components/cudf). It focuses on safe Rust APIs for GPU data
processing while keeping ownership, memory movement, execution context, synchronization, and interop
behavior explicit enough for service-side integration.

## Focus

- Safe, idiomatic Rust wrappers around `libcudf`.
- Backend-service and query-engine integration, rather than just direct API bindings.
- Runtime and execution-context abstractions that can be honored by real compute and I/O paths.
- Clear separation between borrowing, shared ownership, device copies, host materialization, and external interoperability contracts.
- A shared RAPIDS Rust interoperability vocabulary for DataFusion-Nexus and other service components.

## Workspace

- `components/cudf/cudf-nexus/` - the safe public Rust API crate, published as `cudf-nexus` (import `cudf`).
- `components/cudf/cudf-nexus-sys/` - low-level FFI bindings and the native C shim around `libcudf` (import `cudf_sys`).
- `components/cudf/rapids-interop/` - shared RAPIDS Rust interop contract types and traits.
- `components/cudf/cudf/` - the pinned **[forked cuDF](https://github.com/DataFusion-Nexus/cudf)** source submodule used to build `libcudf`.

Native build defaults are repo-relative: `target/native/cudf-build` for the
cuDF CMake cache and `target/native/cudf-install` for the installed cuDF/RMM
prefix. From the monorepo root, use `scripts/build/build_libcudf.sh`; it drives
the pinned source and keeps the cache/install identity aligned. Override the
two locations only with matching `CUDF_CPP_BUILD_DIR` and `CUDF_INSTALL_DIR`
selectors. The root `Cargo.lock`, `.gitmodules`, and `scripts/check_all.sh`
remain canonical for every component.

## Migration provenance

- Wrapper repository and archive: <https://github.com/DataFusion-Nexus/cudf-nexus>
- Frozen imported tip: `a2e893dc7193964feeb4a045d47229d8a2a830ee`
- Import merge: `f6f9046adae3ed6077efde97add417cbc5a11627`
- Final path: `components/cudf`

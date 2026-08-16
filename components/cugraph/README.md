# cuGraph component

`components/cugraph` is the DataFusion Nexus Rust graph integration component
built around [`libcugraph`](https://github.com/DataFusion-Nexus/cugraph). It is
part of the root Cargo workspace, not a second repository or workspace.
Its main direction is to make cuGraph usable as part of backend services and query execution systems for
[DataFusion Nexus](https://github.com/DataFusion-Nexus/datafusion-nexus/tree/main/components/cugraph). It focuses on safe Rust APIs for GPU graph
processing while keeping ownership, memory movement, execution context, synchronization, and interop
behavior explicit enough for service-side integration.

## Focus

- Safe, idiomatic Rust wrappers around `libcugraph`.
- Backend-service and query-engine integration, rather than just direct API bindings.
- cudf-backed graph ingestion and result conversion.
- Graph algorithm APIs that can be embedded in query and service runtimes.
- Structured errors that preserve machine-readable failure context across the GPU boundary.

## Workspace

- `components/cugraph/cugraph-nexus/` - the safe public Rust API crate, published as `cugraph-nexus` (import `cugraph`).
- `components/cugraph/cugraph-nexus-sys/` - low-level FFI bindings around the cuGraph C API (import `cugraph_sys`).
- `components/cugraph/cugraph/` - the pinned **[forked cuGraph](https://github.com/DataFusion-Nexus/cugraph)** source submodule used to build `libcugraph`.

Native defaults are repo-relative: `target/native/cudf-install` and
`target/native/cuvs-install` provide the shared cuDF/RMM and cuVS prefixes, and
`target/native/cugraph-build` is the cuGraph CMake build tree. Use the root
`scripts/build/build_libcugraph.sh` helper and root `scripts/check_all.sh` gate.

## validation

```bash
cargo clippy -p cugraph-nexus -p cugraph-nexus-sys --all-targets --all-features -- -D warnings
cargo +nightly fmt --all --check
flock /tmp/cudf-gpu.lock cargo nextest run -p cugraph-nexus -p cugraph-nexus-sys --all-features --no-fail-fast
```

`cugraph-nexus-sys` intentionally generates bindings from the selected native
headers in `build.rs`; the selected headers and include paths are part of its
package/native contract.

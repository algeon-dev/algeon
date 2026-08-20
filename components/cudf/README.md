# cuDF component

`components/cudf` is the DataFusion Nexus Rust integration component built
around [`libcudf`](https://github.com/DataFusion-Nexus/cudf). It is part of the
root Cargo workspace; it is not a second repository or workspace.
It exists only to support the native backend of
[DataFusion Nexus](https://github.com/DataFusion-Nexus/datafusion-nexus/tree/main/components/cudf).
Its Rust package and import names are workspace contracts, not an external
semantic-versioning or publication promise.

## Focus

- Explicit device, stream, memory-resource, and owner provenance.
- Prepared transfer and completion ordering across RAPIDS components.
- Metadata-only preflight facts and structured errors for bounded admission.
- Safe ownership of libcudf objects and validated host/device materialization.

## Workspace

- `components/cudf/cudf-nexus/` - the internal safe execution crate, packaged as `cudf-nexus` (import `cudf`).
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

## validation

```bash
cargo clippy -p cudf-nexus -p cudf-nexus-sys -p rapids-interop --all-targets --all-features -- -D warnings
cargo +nightly fmt --all --check
flock /tmp/cudf-gpu.lock cargo nextest run \
  -p cudf-nexus -p cudf-nexus-sys -p rapids-interop --all-features --no-fail-fast
```

Use `scripts/dev/bindings.sh cudf [--check]` for the two checked-in binding
surfaces; consumer builds never run bindgen.

# cuvs-nexus

`cuvs-nexus` is a Rust GPU vector-search and clustering workspace built around
NVIDIA [cuVS](https://github.com/rapidsai/cuvs) for DataFusion Nexus. It exposes
safe Rust APIs over the native library and integrates cuVS execution with the
device, stream, memory, and ownership contracts from `cudf-nexus`.

The project is an integration layer, not a query engine: SQL planning and query
execution remain in DataFusion Nexus.

## Focus

- Safe, idiomatic Rust APIs for brute-force k-NN, k-means, and PCA.
- Exact binding to a `cudf::Runtime` device, stream, and memory resource.
- Device-resident cuDF interop with explicit ownership and synchronization.
- Reproducible native builds from a pinned upstream cuVS source revision.
- Checked FFI, ABI, tensor layout, and algorithm-output contracts.

## Workspace

- `cuvs-nexus/` - the safe public Rust API crate, published as `cuvs-nexus`
  (import `cuvs`).
- `cuvs-nexus-sys/` - generated low-level bindings and native build discovery,
  published as `cuvs-nexus-sys` (import `cuvs_sys`).
- `cuvs/` - the pinned official
  [cuVS](https://github.com/rapidsai/cuvs) source submodule used to build the
  native libraries.
- `scripts/` - source bootstrap, native build, binding generation, and contract
  checks.
- `tools/generate-bindings/` - the unpublished maintenance tool that refreshes
  checked-in bindings without changing the library feature set.

## Status

This project is in early development. It currently targets Linux, Rust 1.94 or
newer, and cuVS 26.6.0 from the pinned upstream source. API stability is less
important than getting execution, memory, ownership, and interop contracts
right for DataFusion Nexus.

## Testing

Cargo features are reserved for additive library capabilities; documentation
and binding-generation maintenance modes do not alter native linkage. After
building the canonical native prefix, the complete workspace test surface is:

```bash
cargo test --workspace --all-features --no-fail-fast
```

Within this checkout, native discovery defaults to the canonical
`cuvs-install/` prefix produced by `scripts/build_libcuvs.sh`. `CUVS_ROOT`
remains an explicit override for external prefixes and downstream builds.

## Downstream runtime loading

`CUVS_ROOT` selects the native package at build time; it does not register that
directory with the host dynamic loader. Cargo also does not propagate an
rlib's linker arguments into a downstream executable. A consuming executable
must therefore bundle/install the shared libraries in a loader-visible
location, set `LD_LIBRARY_PATH` when launching, or depend directly on
`cuvs-nexus-sys` and use `DEP_CUVS_C_RUNTIME_LIB_DIRS` in its build script to
emit its own RUNPATH. Other native dependencies, including the separately
built cuDF shim, need the same explicit treatment.

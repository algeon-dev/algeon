# Algeon for DataFusion

`algeon-datafusion` is Algeon's Apache DataFusion adapter. It installs physical
optimizer rules that lower supported DataFusion plans into Algeon's native GPU
runtime, while preserving the executable DataFusion plan for candidates that
are not selected.

The adapter also provides SQL table functions for cuGraph and cuVS algorithms,
structured planning and runtime diagnostics, and Parquet source integration.
See the [Algeon documentation](https://algeon.dev) for setup,
supported SQL, configuration, and operational contracts.

## Use as a Rust dependency

Add the adapter and DataFusion to the application's dependencies:

```toml
[dependencies]
algeon-datafusion = "0.1.0"
datafusion = "55.1"
```

The Algeon dependency permits `0.1.x` patch updates. Build native libraries
from the source tag matching the resolved Algeon version recorded in the
application's `Cargo.lock`, such as `v0.1.0` for `0.1.0`. An Algeon version
update requires native libraries from the corresponding release.

The adapter has no default features. Relational execution needs cuDF/RMM;
`cuvs` adds cuVS, `cugraph` adds cuVS and cuGraph, and `cuml` adds cuVS and
cuML. Rust package installation does not build these native libraries.
Follow the [Rust dependency quickstart](https://algeon.dev/build/rust-dependency)
for native build order, compiler and header requirements, `ALGEON_NATIVE_DIR`,
`LD_LIBRARY_PATH`, and a complete async application with result assertions and
backend shutdown. The guide pins `0.1.0` and uses the release consumer fixture
as its example; update its source tag and adapter version together when using
another Algeon release. Fixture validation does not establish that the tag or
packages have been published.

## Native source batching

Native source batching uses the immutable admission grant when no source-byte
target is configured. Individual row groups remain retryable units; compatible
units are coalesced into reads. Source-adjacent aggregates size later reads from
observed decoded and staged bytes per row, including retained allocation-domain
usage. Explicit `native.max_source_chunk_bytes` and
`native.max_row_groups_per_chunk` settings bound the configured grouping targets.
The query-domain allocator enforces the grant, and allocation failures retain
the bounded split-retry path.

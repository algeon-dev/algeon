# Algeon for DataFusion

`algeon-datafusion` is Algeon's Apache DataFusion adapter. It installs physical
optimizer rules that lower supported DataFusion plans into Algeon's native GPU
runtime, while preserving the executable DataFusion plan for candidates that
are not selected.

The adapter also provides SQL table functions for cuGraph and cuVS algorithms,
structured planning and runtime diagnostics, and Parquet source integration.
See the [Algeon documentation](https://algeon.dev) for setup,
supported SQL, configuration, and operational contracts.

Native source batching uses the immutable admission grant when no source-byte
target is configured. Individual row groups remain retryable units; compatible
units are coalesced into reads. Source-adjacent aggregates size later reads from
observed decoded and staged bytes per row, including retained allocation-domain
usage. Explicit `native.max_source_chunk_bytes` and
`native.max_row_groups_per_chunk` settings bound the configured grouping targets.
The query-domain allocator enforces the grant, and allocation failures retain
the bounded split-retry path.

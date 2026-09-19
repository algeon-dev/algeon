# Algeon for DataFusion

`algeon-datafusion` is Algeon's Apache DataFusion adapter. It installs physical
optimizer rules that lower supported DataFusion plans into Algeon's native GPU
runtime, while preserving the executable DataFusion plan for candidates that
are not selected.

The adapter also provides SQL table functions for cuGraph and cuVS algorithms,
structured planning and runtime diagnostics, and Parquet and Iceberg source
integration. See the [Algeon documentation](https://algeon.dev) for setup,
supported SQL, configuration, and operational contracts.

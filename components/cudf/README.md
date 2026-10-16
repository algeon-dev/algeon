# algeon-cudf

`algeon-cudf` (library `cudf`) is the safe Rust API over libcudf that Algeon's
query engine executes on: `DataFrame`, `Series`, `GroupBy`, Parquet reads and
the `Runtime` that binds them to one CUDA device, stream and memory resource.
A call that submits GPU work, allocates or materializes names its execution
context through `Exec`; nothing falls back to ambient CUDA state. Applications
depend on [`algeon-datafusion`](https://docs.rs/algeon-datafusion) instead.

[DESIGN.md](https://github.com/algeon-dev/algeon/blob/main/components/cudf/DESIGN.md)
describes the layers, handle ownership, execution provenance and memory
domains, and
[SAFETY.md](https://github.com/algeon-dev/algeon/blob/main/components/cudf/algeon-cudf-sys/SAFETY.md)
holds the rules for code that touches raw handles.

The crate builds against Algeon's fork of cuDF, upstream `v26.08.00` plus the
changes listed under [Fork changes](https://algeon.dev/reference/forks#cudf).
`algeon-cudf-sys` holds the raw C ABI.

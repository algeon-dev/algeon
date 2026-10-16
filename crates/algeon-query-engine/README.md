# algeon-query-engine

`algeon-query-engine` executes a planned query on the GPU. It takes one
planning proof from `algeon-query-planning` and one runtime grant from
`algeon-query-runtime`, reads Parquet sources in admitted chunks, and runs
relational, graph, vector and machine-learning operators through the cuDF,
cuGraph, cuVS and cuML components. `algeon-datafusion` is its intended caller;
applications depend on that crate.

[GPU execution](https://algeon.dev/design/gpu-execution) describes the
operator pipeline, and
[module boundaries](https://algeon.dev/design#crates-and-ownership) shows
which query crate owns each stage.

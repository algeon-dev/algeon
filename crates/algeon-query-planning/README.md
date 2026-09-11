# query-planning

`query-planning` is the backend-neutral owner of native planning, proof,
and admission contracts. Its resolved dependency closure, under the union of
every feature, contains neither DataFusion nor any RAPIDS component, nor
`query-runtime` or `query-engine`. That is enforced by
[`scripts/ci/check_dependency_direction.py`](../../scripts/ci/check_dependency_direction.py),
which walks the closure rather than the declared dependencies, because a helper
crate is how such a dependency comes back.

Backend-neutral means the contracts are expressed in semantics, types, data
shape, and lifetime — not that the crate is ignorant of join builds, resident
frames, or graph layouts. Arrow and pure-Rust Parquet metadata types stay.

## Main areas

- `plan/`, `expr/`, `source/`, `chunk/`, and `datatypes.rs` own the native IR,
  source facts, and the Arrow type and cast semantics the query language admits.
- `backend.rs` states the questions planning cannot answer itself — whether a
  backend represents a type, binds an expression or aggregate, or infers a
  kernel's output type. The engine implements it and passes it in explicitly as
  an `AnalysisContext`; there is no registry and no default.
- `optimizer/` owns native IR rewrite rules and diagnostics.
- `capability/`, `admission/`, and `graph_memory/` own capability-first proof,
  allocation-owner evidence, and structured rejection.
- `pipeline/` compiles the single validated schedule consumed downstream.
- `error.rs` owns the shared native `QueryError` taxonomy used by planning,
  runtime, and execution. Projecting a component's error *into* that taxonomy
  belongs to the layer that calls the component: `query_runtime::native_error`
  for cuDF, `query_engine::native_error` for cuVS.

Planning performs no file or network I/O. It may *consume* Parquet footers and
decoded rows, but whoever read them supplies them: `source::SourceFooterFacts`
for footers the engine already loaded, and a `LocalParquetPruningResolver` (or
the async `LakehouseFooterPruningResolver` for object storage) for the facts
dimension-derived pruning needs. A missing fact makes the optimization skip; it
never makes it guess.

Runtime resource ownership does not live here. Planning produces proof and
requirements but does not select a device, mint a grant, maintain a ledger, or
observe an attempt.

## Validation

```bash
cargo nextest run -p query-planning --all-features --lib
cargo clippy -p query-planning --all-targets --all-features
bash scripts/check_all.sh
```

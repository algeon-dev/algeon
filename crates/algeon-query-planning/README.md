# query-planning

`algeon-query-planning` is a published implementation dependency of Algeon.
Applications use [`algeon-datafusion`](https://docs.rs/algeon-datafusion)
and its re-exported types. This page describes the workspace owner API; see
the [API documentation boundary](https://algeon.dev/design#api-documentation).

`query-planning` is the backend-neutral owner of native planning, proof,
and admission contracts. Its resolved dependency closure, under the union of
every feature, contains neither DataFusion nor any RAPIDS component, nor
`query-runtime` or `query-engine`.
[`scripts/ci/check_dependency_direction.py`](../../scripts/ci/check_dependency_direction.py)
enforces this by walking the resolved closure, because a helper crate can carry
such a dependency without declaring it.

Backend-neutral means the contracts are expressed in semantics, types, data
shape, and lifetime. The crate still models join builds, resident frames, and
graph layouts at that level, and Arrow and pure-Rust Parquet metadata types
stay.

## Main areas

- `plan/`, `expr/`, `source/`, `chunk/`, and `datatypes.rs` own the native IR,
  source facts, and the Arrow type and cast semantics the query language admits.
- `backend.rs` states the questions planning cannot answer itself: whether a
  backend represents a type, binds an expression or aggregate, or infers a
  kernel's output type. The engine implements it and passes it in as an
  `AnalysisContext`, with no registry or default.
- `optimizer/` owns native IR rewrite rules and diagnostics.
- `capability/` and `admission/` own capability-first proof and structured
  rejection.
- `pipeline/` compiles the single validated schedule consumed downstream.
- `error.rs` owns the shared native `QueryError` taxonomy used by planning,
  runtime, and execution. Projecting a component's error *into* that taxonomy
  belongs to the layer that calls the component: `query_runtime::native_error`
  for cuDF, `query_engine::native_error` for cuVS.

Planning performs no file or network I/O. It consumes Parquet footers and
decoded rows that the caller supplies: `source::SourceFooterFacts` for footers
the engine already loaded, and a `LocalParquetPruningResolver` (or the async
`LakehouseFooterPruningResolver` for object storage) for the facts
dimension-derived pruning needs. The optimizer skips a rule when a required
fact is unavailable.

Runtime resource ownership does not live here. Planning produces proof and
requirements but does not select a device, mint a grant, maintain a ledger, or
observe an attempt.

## Validation

```bash
cargo nt -E 'package(algeon-query-planning)'
cargo clippy --workspace --all-targets --all-features -- -D warnings
bash scripts/check_all.sh
```

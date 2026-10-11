# query-runtime

`algeon-query-runtime` is a published implementation dependency of Algeon.
Applications use [`algeon-datafusion`](https://docs.rs/algeon-datafusion)
and its re-exported types. This page describes the workspace owner API; see
the [API documentation boundary](https://algeon.dev/design#api-documentation).

`query-runtime` is the native owner of attempt lifecycle
and GPU resource admission. It depends on `query-planning` contracts and
does not depend on `query-engine` execution.

## Main areas

- `attempt/` owns cancellation, lifecycle, queue state, and attempt ownership.
- `runtime/query_service/` and `runtime/gpu/` own device placement, the single
  per-device ledger, immutable grants, transactions, and cleanup.
- `runtime/cache/` and `runtime/source_identity/` own resident cache lifecycle
  and cache identity.
- `observability/` and `metrics/` project the single runtime observation
  snapshot and sealed attempt evidence.
- `native_error.rs` owns the one cuDF-to-`QueryError` projection, so the
  runtime and the engine classify a native failure identically.

Source format parsing and chunk preparation are not here: Parquet footers and
source chunk plans belong to `query-engine`. The runtime owns the
resources those readers use; the object-store client registry, credential
profiles, the I/O executor, queues, and cancellation; and does not need to
understand row groups to own them.

Runtime consumes the planning proof and refines deferred evidence inside the
same immutable grant. It does not define another sizing model, scheduler ledger,
grant, observation store, or error taxonomy.

## Validation

```bash
cargo nextest run -p query-runtime --all-features --lib
cargo clippy -p query-runtime --all-targets --all-features
bash scripts/check_all.sh
```

# nexus-query-runtime

`nexus-query-runtime` is the DataFusion-free owner of native attempt lifecycle
and GPU resource admission. It depends on `nexus-query-planning` contracts and
does not depend on `nexus-query-engine` execution.

## Main areas

- `attempt/` owns cancellation, lifecycle, queue state, and attempt ownership.
- `runtime/query_service/` and `runtime/gpu/` own device placement, the single
  per-device ledger, immutable grants, transactions, and cleanup.
- `runtime/cache/` and `runtime/source_identity/` own resident cache lifecycle
  and cache identity.
- `observability/` and `metrics/` project the single runtime observation
  snapshot and sealed attempt evidence.

Runtime consumes the planning proof and refines deferred evidence inside the
same immutable grant. It does not define another proof model, scheduler ledger,
grant, observation store, or error taxonomy.

## Validation

```bash
cargo nextest run -p nexus-query-runtime --all-features --lib
cargo clippy -p nexus-query-runtime --all-targets --all-features
bash scripts/check_all.sh
```

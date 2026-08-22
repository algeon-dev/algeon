# nexus-query-planning

`nexus-query-planning` is the DataFusion-free owner of native planning, proof,
and admission contracts. It has no dependency on DataFusion,
`nexus-query-runtime`, or `nexus-query-engine` execution.

## Main areas

- `plan/`, `expr/`, `source/`, `chunk/`, and `datatypes.rs` own the native IR
  and source facts.
- `optimizer/` owns native IR rewrite rules and diagnostics.
- `capability/`, `admission/`, and `graph_memory/` own capability-first proof,
  allocation-owner evidence, and structured rejection.
- `pipeline/` compiles the single validated schedule consumed downstream.
- `error.rs` owns the shared native `QueryError` taxonomy used by planning,
  runtime, and execution.

Runtime resource ownership does not live here. Planning produces proof and
requirements but does not select a device, mint a grant, maintain a ledger, or
observe an attempt.

## Validation

```bash
cargo nextest run -p nexus-query-planning --all-features --lib
cargo clippy -p nexus-query-planning --all-targets --all-features
bash scripts/check_all.sh
```

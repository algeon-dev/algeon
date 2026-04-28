# nexus-query-engine

`nexus-query-engine` is the DataFusion-free native engine used by
`datafusion-nexus`. It owns the native query IR, capability admission,
strict/adaptive runtime contracts, source/chunk planning, GPU execution, native
metrics, and native error facts.

This crate must not depend on DataFusion. DataFusion-specific recognition,
lowering, table-format integration, and `ExecutionPlan` wrappers belong in the
root adapter crate.

## Main Areas

- `plan/`, `expr/`, `source.rs`, `datatypes.rs` - native IR and source facts.
- `capability/` - capability-first admission and structured rejection reasons.
- `runtime/`, `mode.rs` - execution context, reservation policy, strict/adaptive
  mode contracts.
- `pipeline/` - compiled DAG tasks, edges, gates, and resident resources.
- `exec/` - GPU execution, bounded execution, source runtime, joins,
  aggregates, final sort/top-k, and graph execution.
- `optimizer/` - native IR rewrite rules and rewrite diagnostics.
- `adaptive_report.rs`, `contract_report.rs`, `metrics.rs`, `error.rs` -
  runtime evidence, metrics, and native errors.

## Validation

```bash
cargo test -p nexus-query-engine
cargo test -p nexus-query-engine --test native_contract_tests
flock /tmp/cudf-gpu.lock cargo test -p nexus-query-engine --test native_execution_tests
```

Do not merge strict and adaptive APIs, and do not reintroduce retired bounded
family names. The contract tests pin those boundaries.

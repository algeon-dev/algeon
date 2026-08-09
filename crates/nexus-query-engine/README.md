# nexus-query-engine

`nexus-query-engine` is the DataFusion-free native engine used by
`datafusion-nexus`. It owns the native query IR, capability admission,
the bounded per-device resource controller, source/chunk planning, GPU
execution, native metrics, and native error facts.

This crate must not depend on DataFusion. DataFusion-specific recognition,
lowering, table-format integration, and `ExecutionPlan` wrappers belong in the
root adapter crate.

## Main Areas

- `plan/`, `expr/`, `source.rs`, `datatypes.rs` - native IR and source facts.
- `capability/` - capability-first admission and structured rejection reasons.
- `runtime/`, `mode.rs` - execution context, immutable grant and reservation
  policy, and bounded report evidence.
- `pipeline/` - compiled DAG tasks, edges, gates, and resident resources.
- `exec/` - GPU execution, bounded execution, source runtime, joins,
  aggregates, final sort/top-k, and graph execution.
- `optimizer/` - native IR rewrite rules and rewrite diagnostics.
- `contract_report.rs`, `metrics.rs`, `error.rs` - runtime evidence, metrics,
  and native errors.

## Validation

The engine's full test surface requires cargo-nextest per-test-process
isolation. Allocator-lifecycle cases intentionally install a process-global
`DeviceMemoryManager`, and native-execution cases retain one backend policy per
process. Do not use broad `cargo test -p nexus-query-engine --all-features`,
`cargo test --all-features --lib`, or serialized libtest threads as
package-level validation.

```bash
cargo nextest run -p nexus-query-engine --all-features --test native_contract_tests
cargo nextest run -p nexus-query-engine --all-features --test public_api_boundary
flock /tmp/cudf-gpu.lock cargo nextest run -p nexus-query-engine --all-features --lib
flock /tmp/cudf-gpu.lock cargo nextest run -p nexus-query-engine --all-features --test query_service_allocator_lifecycle
bash scripts/check_all.sh
```

Do not reintroduce tenant policy, grant bands, mutable grants, or strict/adaptive
product APIs. The contract tests pin the one-controller boundary and bounded
capability evidence.

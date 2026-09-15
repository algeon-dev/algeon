# nexus-query-engine

`nexus-query-engine` is the DataFusion-free native execution owner and public
facade used by `datafusion-nexus`. Its package and Rust import identity remain
the compatible entry point for callers that used the native engine before the
planning/runtime split.

The crate depends on `nexus-query-planning` and `nexus-query-runtime`, re-exports
the established planning and runtime module paths, and composes one planning
proof with one runtime grant for native execution. It does not define another
proof, ledger, grant, observation store, or error taxonomy.

DataFusion-specific recognition, lowering, table-format integration, and
`ExecutionPlan` wrappers belong in the root adapter crate.

## Main areas

- `exec/` owns bounded relational, graph, vector, and source execution plus
  host/GPU-resident result handling.
- `chunk/` composes planning chunk contracts with execution behavior.
- `lib.rs` is the compatible facade over planning and runtime owners.
- `execution_integration_tests/` covers planning/runtime/execution composition.

See [`nexus-query-planning`](../nexus-query-planning/README.md) for IR,
capability, proof, admission contracts, and pipelines. See
[`nexus-query-runtime`](../nexus-query-runtime/README.md) for lifecycle, query
service, device resources, caches, observation, and metrics.

## Validation

The engine's full test surface requires cargo-nextest per-test-process
isolation. Allocator-lifecycle cases intentionally install a process-global
`DeviceMemoryManager`, and native-execution cases retain one backend policy per
process. Do not use broad `cargo test -p nexus-query-engine --all-features`,
`cargo test --all-features --lib`, or serialized libtest threads as
package-level validation.

```bash
cargo nt -E 'binary_id(nexus-query-engine::native_contract_tests)'
cargo nt -E 'binary_id(nexus-query-engine::public_api_boundary)'
cargo nt -E 'package(nexus-query-engine) & kind(lib)'
cargo nt -E 'binary_id(nexus-query-engine::gpu_allocator_event_correlation_gpu_exclusive)'
bash scripts/check_all.sh
```

The contract tests pin facade compatibility, dependency direction, the single
controller boundary, and bounded execution composition.

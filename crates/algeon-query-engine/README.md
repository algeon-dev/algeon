# query-engine

`query-engine` is the native execution owner. It depends on
`query-planning` and `query-runtime` and composes one planning proof with one
runtime grant for native execution. It exports only its own modules: callers,
including `algeon-datafusion`, import planning and runtime items from those
crates. It does not define another proof, ledger, grant, observation store, or
error taxonomy.

DataFusion-specific recognition, lowering, and `ExecutionPlan` wrappers belong
in the DataFusion adapter crate.

## Main areas

- `exec/` owns bounded relational, graph, vector, and source execution plus
  host/GPU-resident result handling.
- `rapids_backend.rs` implements planning's backend capability contract. The
  checks *are* the lowering, so capability and execution binding cannot drift.
- `prepared/` and `datatypes.rs` bind query IR to cuDF expressions, aggregates,
  and dtypes; the dtype conversion table itself belongs to `cudf`.
- `parquet_metadata.rs`, `source_chunk/`, and `chunk/` own source preparation:
  footer fetch, parsing, sidecars, the footer metadata cache, and chunk plans.
- `native_error.rs` projects cuVS and cuML errors; the shared cuDF projection
  belongs to the runtime's native boundary.
- `execution_integration_tests/` covers planning/runtime/execution composition.

See [`query-planning`](../query-planning/README.md) for IR,
capability, proof, admission contracts, and pipelines. See
[`query-runtime`](../query-runtime/README.md) for lifecycle, query
service, device resources, caches, observation, and metrics.

## Validation

The engine's full test surface requires cargo-nextest per-test-process
isolation. Allocator-lifecycle cases intentionally install a process-global
`DeviceMemoryManager`, and native-execution cases retain one backend policy per
process. Do not use broad `cargo test -p query-engine --all-features`,
`cargo test --all-features --lib`, or serialized libtest threads as
package-level validation.

```bash
cargo nt -E 'binary_id(query-engine::native_contract_tests)'
cargo nt -E 'binary_id(query-engine::public_api_boundary)'
cargo nt -E 'package(query-engine) & kind(lib)'
cargo nt -E 'binary_id(query-engine::gpu_allocator_event_correlation_gpu_exclusive)'
bash scripts/check_all.sh
```

The contract tests pin dependency direction, the single controller boundary,
and bounded execution composition.

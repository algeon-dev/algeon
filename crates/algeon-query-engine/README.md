# query-engine

`algeon-query-engine` is a published implementation dependency of Algeon.
Applications use [`algeon-datafusion`](https://docs.rs/algeon-datafusion)
and its re-exported types. This page describes the workspace owner API; see
the [API documentation boundary](https://algeon.dev/design#api-documentation).

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
- `tests/runtime_execution_integration_tests_gpu/` covers planning/runtime/execution composition.

See [`query-planning`](https://github.com/algeon-dev/algeon/blob/main/crates/algeon-query-planning/README.md) for IR,
capability, proof, admission contracts, and pipelines. See
[`query-runtime`](https://github.com/algeon-dev/algeon/blob/main/crates/algeon-query-runtime/README.md) for lifecycle, query
service, device resources, caches, observation, and metrics.

## Validation

The engine's full test surface requires cargo-nextest per-test-process
isolation. Allocator-lifecycle cases intentionally install a process-global
`DeviceMemoryManager`, and native-execution cases retain one backend policy per
process. Do not use broad `cargo test -p query-engine --all-features`,
`cargo test --all-features --lib`, or serialized libtest threads as
package-level validation.

```bash
cargo nt -E 'binary_id(algeon-query-engine::native_contract_tests_cpu)'
cargo nt -E 'binary_id(algeon-query-engine::public_api_boundary_cpu)'
cargo nt -E 'package(algeon-query-engine) & kind(lib)'
cargo nt -E 'binary_id(algeon-query-engine::gpu_allocator_event_correlation_gpu_exclusive)'
bash scripts/check_all.sh
```

The contract tests pin dependency direction, the single controller boundary,
and bounded execution composition.

# AGENTS.md

## Working principles

- Fix the owning abstraction or contract, not a failing query with an adapter
  workaround. Extend `cudf-rs` or `cugraph-rs` when the gap belongs there.
- Keep changes scoped. Preserve unrelated worktree edits and avoid opportunistic
  cleanup.
- Preserve public behavior and contract tests unless the task changes them. Add
  tests for behavior or regressions; comment only on non-obvious constraints.

## Project boundaries

`datafusion-nexus` lowers supported DataFusion physical plans to
`nexus-query-engine` IR and executes them with `cudf-rs` and `cugraph-rs`.

- DataFusion-dependent code belongs in the root adapter; DataFusion-independent
  IR and execution belong in `crates/nexus-query-engine`.
- `cudf` and `cugraph` are sibling path dependencies; DataFusion is pinned in
  `Cargo.toml`.
- Default members are the adapter, `nexus-workloads`, and
  `nexus-query-engine`. Bench, tools, and examples are explicit non-default
  members.
- Linux only. MSRV is Rust 1.94; formatting requires nightly.

## Architecture contracts

- `NexusNativePhysicalOptimizerRule` lowers supported plans to
  `nexus_query_engine::QueryPlan` and emits structured `PlanningReport`s. Assert
  reports, not printed plans.
- Admission is capability-first: reject locally with structured reasons.
- Strict and adaptive bounded execution have separate admission, evidence, and
  report APIs. Do not project one mode's evidence into the other.
- Model aggregate partial/final state explicitly. AVG is SUM + COUNT followed by
  final SUM/SUM, never average-of-averages.

## Build and test

```bash
# Default members
cargo build
cargo clippy --all-targets --all-features
cargo +nightly fmt --all

# Targeted edit loops
cargo nextest run -p datafusion-nexus --no-default-features --test cpu_contract_tests
cargo nextest run -p datafusion-nexus --no-default-features --test native_planning_tests
cargo nextest run -p nexus-query-engine --no-default-features --test native_contract_tests

# Non-default bench package, when touched
cargo check -p nexus-bench --all-targets --all-features
cargo clippy -p nexus-bench --all-targets --all-features
cargo nextest run -p nexus-bench --all-features

# Only full pre-merge surface
bash scripts/check_all.sh
```

Run focused checks first, then broaden validation with the change's blast
radius. `bash scripts/check_all.sh --fast` and every explicit `--lane` are
partial and must not be reported as full validation.

## Testing and repository policy

- Serialize GPU work with `flock /tmp/cudf-gpu.lock`.
- `.config/policies/test-lanes.toml` is the only lane-admission source. After
  changing it, run
  `python3 scripts/ci/test_lane_policy.py generate-nextest --write`.
- Do not add external-service E2E tests to Cargo test targets; invoke them
  explicitly through `cargo run -p nexus-tools`.
- Use `bash scripts/dev/clean_target_artifacts.sh` only to report removable build
  artifacts; cleanup is never implicit.
- `scripts/check_all.sh` is the only top-level script entry. Build tooling lives
  in `scripts/build/`, CI helpers in `scripts/ci/`, manual helpers in
  `scripts/dev/`, and policy data in `.config/policies/`.
- Keep `scripts/build/rapids.rs` std-only and shared by the root, native engine,
  bench, and examples packages.

## Public contracts

- Preserve report row schemas, TSV ordering, and `NATIVE_*_TSV_HEADER` constants.
- Preserve `NATIVE_NEXUS_DATAFUSION_METRIC_NAMES` and native execution-plan
  wrappers.
- TSV, DataFusion metrics, and `QueryMetrics` are projections of the runtime
  observation snapshot, not a second event store.
- Preserve workload environment variables and item shapes.
- Keep `src/error.rs` errors structured; convert to `DataFusionError` only at
  the DataFusion boundary.
- `datafusion_nexus_native` (`NEXUS_NATIVE_PHYSICAL_OPTIMIZER_RULE_NAME`) is
  public and is the rule deduplication key.

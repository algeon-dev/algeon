# AGENTS.md

## Working principles

- Fix the owning abstraction or contract, not a failing query with an adapter
  workaround. Extend `cudf-nexus` or `cugraph-nexus` when the gap belongs there.
- Keep changes scoped. Preserve unrelated worktree edits and avoid opportunistic
  cleanup.
- Preserve public behavior and contract tests unless the task changes them. Add
  tests for behavior or regressions; comment only on non-obvious constraints.
- An explicit `drop()` in production code needs a one-line comment explaining
  why the value cannot wait until end of scope (such as a lock window,
  re-entrancy, or `RefCell` re-borrow); otherwise delete it rather than add it.
  Test code is exempt.

## Project boundaries

`datafusion-nexus` lowers supported DataFusion physical plans to
`nexus-query-engine` IR and executes them with `cudf-nexus` and `cugraph-nexus`.

- DataFusion-dependent code belongs in the root adapter; DataFusion-independent
  IR and execution belong in `crates/nexus-query-engine`.
- `cudf-nexus` and `cugraph-nexus` are sibling path dependencies; DataFusion is pinned in
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
- There is one bounded admission contract: the per-device ledger, immutable
  grants, and transactional envelopes. There is no strict/adaptive product
  mode, tenant policy, grant band, or slot guarantee; preserved strict-named
  and adaptive-named wire fields are canonical projections of the one
  contract, not separate producers.
- The reservation ledger is integer accounting, not a residency oracle; dropping
  a frame earlier does not change what it reports.
- Model aggregate partial/final state explicitly. AVG is SUM + COUNT followed by
  final SUM/SUM, never average-of-averages.

## Build and test

Use `--all-features` everywhere, in every inner-loop command. Cargo keys build
artifacts by feature unification, so alternating between
`--no-default-features` and `--all-features` across commands in the same edit
loop does not reuse the other set's artifacts and pays a fresh build of the
~400-unit dependency graph each way. Staying in one feature set keeps every
command below on the same artifacts.

Package selection is not part of this: `-p <crate> --all-features` and
`--workspace --all-features` were measured to leave each other's artifacts
valid (0 units rebuilt in both directions), so the targeted `-p` loops below
cost nothing extra against the workspace-wide gate.

```bash
# Build and lint, any package
cargo build --all-features
cargo clippy --workspace --all-targets --all-features
cargo +nightly fmt --all

# Targeted edit loops
cargo nextest run -p datafusion-nexus --all-features --test cpu_contract_tests
cargo nextest run -p datafusion-nexus --all-features --test native_planning_tests
cargo nextest run -p nexus-query-engine --all-features --test native_contract_tests

# Non-default bench package, when touched
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

# AGENTS.md

## Working principles

- Fix the owning abstraction or contract, not a failing query with an adapter
  workaround. Extend `cudf-nexus` or `cugraph-nexus` when the gap belongs there.
- Keep changes scoped: preserve unrelated worktree edits, public behavior, and
  contract tests unless the task changes them; avoid opportunistic cleanup.
- Grow the system in layers: start from the smallest version that works end to
  end, and add each capability on top of a product that already works; never
  trade a working product for unfinished complexity.

## Subtractive development

- Prefer extending existing code over adding parallel paths and the smallest
  change that fully solves the current requirement; net line count should
  trend down or stay flat. About to add a second way to do something? Stop and
  propose removing or consolidating the first.
- Before writing new code for a non-trivial task, list the related existing
  files and functions; for each, state whether it should be extended, replaced,
  or left alone. Report before implementing.
- Never add speculative abstractions, configuration, or "just in case"
  fallbacks; every defensive branch must name a concrete scenario that actually
  happens in this codebase, and delete any that cannot. Remove obsolete or
  backward-compatibility paths not required by a contract in this file instead
  of stacking new code on top.
- Before committing, re-read your own diff for anything now duplicated,
  superseded, unused, or overly defensive. Propose deletions as a separate
  cleanup commit, or fold them in when the scope is small and safe.

## Comments

- Default to no comments. Never explain what the code does; if code needs a
  comment to be understandable, rename things instead.
- Comment only non-obvious rationale, invariants, safety constraints, external
  quirks, or business rules the code cannot express.
- Public API documentation describes observable contracts, not incidental
  implementation details.
- An explicit `drop()` in production code needs a one-line comment on why the
  value cannot wait until end of scope (lock window, re-entrancy, `RefCell`
  re-borrow); otherwise delete the `drop()`. Test code is exempt.

## Project boundaries

`datafusion-nexus` lowers DataFusion physical plans to `nexus-query-engine` IR
and executes them via `cudf-nexus` and `cugraph-nexus`. DataFusion-dependent
code belongs in the root adapter; DataFusion-independent IR and execution
belong in `crates/nexus-query-engine`.

- `cudf-nexus` and `cugraph-nexus` are sibling path dependencies; DataFusion is
  pinned in `Cargo.toml`.
- Default members are the adapter, `nexus-workloads`, and `nexus-query-engine`.
  Bench, tools, and examples are explicit non-default members.
- Linux only. MSRV is Rust 1.94; formatting requires nightly.

## Architecture contracts

- `NexusNativePhysicalOptimizerRule` lowers supported plans to
  `nexus_query_engine::QueryPlan` and emits structured `PlanningReport`s. Assert
  reports, not printed plans.
- Admission is capability-first: reject locally with structured reasons.
- There is one bounded admission contract: the per-device ledger, immutable
  grants, and transactional envelopes. There is no strict/adaptive product
  mode, tenant policy, grant band, or slot guarantee; strict- and adaptive-named
  wire fields are canonical projections of the one contract, not separate
  producers.
- The reservation ledger is integer accounting, not a residency oracle; dropping
  a frame earlier does not change what it reports.
- Model aggregate partial/final state explicitly. AVG is SUM + COUNT followed by
  final SUM/SUM, never average-of-averages.

## Build and test

Use `--all-features` in every command: cargo keys build artifacts by feature
unification, so mixing feature sets rebuilds the ~400-unit dependency graph
each way. `-p <crate>` and `--workspace` share artifacts, so the targeted `-p`
loops below cost nothing extra against the workspace gate.

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

## Writing tests

- Prefer extending existing coverage at the behavior boundary before creating
  new tests.
- Add tests for realistic observable regressions, non-trivial invariants or
  boundaries, and concrete bugs. Code changing or coverage increasing is not
  sufficient justification by itself.
- Do not test what the type system guarantees or mirror literals, mappings,
  obvious control flow, implementation details, or removed features unless
  absence is itself a contract.
- For concurrency, prefer deterministic coordination or controlled scheduling
  over sleeps when practical.
- A new test must fail when the code under test is reverted; if it does not,
  delete it.

## Testing and repository policy

- Serialize direct GPU commands with `flock /tmp/cudf-gpu.lock`. Never wrap
  `scripts/check_all.sh` in that flock: it acquires the same lock per GPU
  phase internally, so an outer hold self-deadlocks the gate. Run the gate bare.
- After a full gate begins, a late repair may re-run only phases whose
  source-state hash changed, plus every GPU-locked phase unconditionally;
  rerunning only the changed subset never establishes full validation.
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
  server, bench, tools, and examples packages.

## Public contracts

- Preserve report row schemas, TSV ordering, `NATIVE_*_TSV_HEADER` constants,
  `NATIVE_NEXUS_DATAFUSION_METRIC_NAMES`, and native execution-plan wrappers.
- TSV, DataFusion metrics, and `QueryMetrics` are projections of the runtime
  observation snapshot, not a second event store.
- Preserve workload environment variables and item shapes.
- Keep `src/error.rs` errors structured; convert to `DataFusionError` only at
  the DataFusion boundary.
- `datafusion_nexus_native` (`NEXUS_NATIVE_PHYSICAL_OPTIMIZER_RULE_NAME`) is
  public and is the rule deduplication key.

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
  change that fully solves the current requirement. About to add a second way
  to do something? Stop and propose removing or consolidating the first.
- Never add speculative abstractions, configuration, or "just in case"
  fallbacks; every defensive branch should protect a supported input, failure
  mode, or invariant. Remove obsolete or backward-compatibility paths not
  required by a contract in this file instead of stacking new code on top.

## Comments

- Omit comments that only narrate code. Comment non-obvious rationale,
  invariants, safety constraints, external quirks, or business rules the code
  cannot express.
- Public API documentation describes observable contracts, not incidental
  implementation details.
- An explicit `drop()` in production code needs a one-line comment on why the
  value cannot wait until end of scope (lock window, re-entrancy, `RefCell`
  re-borrow); otherwise delete the `drop()`. Test code is exempt.

## Project boundaries

`datafusion-nexus` lowers DataFusion physical plans through the compatible
`nexus-query-engine` facade and executes them via the Rust components under
`components/`. DataFusion-dependent code belongs in the root adapter. IR,
optimizer, capability, proof, admission contracts, and pipeline belong in
`crates/nexus-query-planning`; attempt lifecycle, query service, device ledger,
immutable grants, cache, observation, and metrics belong in
`crates/nexus-query-runtime`; native execution composition and facade
compatibility belong in `crates/nexus-query-engine`.

- `cudf-nexus`, `cugraph-nexus`, and `cuvs-nexus` are internal members of the
  root workspace. Their package and Rust import identities remain unchanged.
- Native sources are the root-registered submodules at
  `components/cudf/cudf`, `components/cugraph/cugraph`, and
  `components/cuvs/cuvs`; native defaults live under `target/native/`.
- The root `.gitmodules`, `Cargo.lock`, workspace, and `scripts/check_all.sh`
  are the only source registry, lockfile, workspace, and full validation gate.
- Default members are the adapter, `nexus-workloads`, and `nexus-query-engine`.
  Bench, tools, and examples are explicit non-default members.
- Linux only. MSRV is Rust 1.94; formatting requires nightly.

## Native GPU engine core

- Support requires a correct kernel, types, and semantics; an operator lacking one is structured unsupported. A missing memory estimate is never a reason to reject.
- Admission issues a service-configured cap, not a predicted requirement: `nexus-query-planning` proves the plan is executable, `nexus-query-runtime` admits it against the device ledger, and the query domain's allocator enforces the cap while `nexus-query-engine` executes.
- The query domain's allocator is the single arbiter of a query's allocation limit. Never add a second Rust-side reservation, per-operator subgrant, or estimate-derived veto.
- Use one compiled schedule, per-device ledger, and immutable grant; deferred refinement stays inside that grant.
- Admit by operator capability, never customer, SQL text, names, one-off algorithm allowlists, or plan-shape matching.
- Keep one truth source; consolidate and delete special constructors, flags, matchers, or ledgers.
- Output and resident facts carry scope, occurrence, schema, and provenance; allocation failures are typed at their source and reported with their cleanup disposition.
- `PlanningReport` distinguishes accepted, deferred, and rejected candidates with structured reasons.
- Keep backend memory models in component owners and IR, facts, schedules, and admission backend neutral.
- Before adding an operator, identify its allocation, facts, lifetime, diagnostics, and test owners; otherwise stop.
- The ledger is integer accounting, not a residency oracle; model aggregate partial/final state explicitly.

## Build and test

Use `--all-features` in every Cargo build, check, lint, and test command except
this one repository-owned cargo-hack feature-isolation command:

```bash
cargo hack check --workspace --each-feature --exclude-features default --locked --keep-going
```

That exact compile-only command is the sole exception; it proves package-local
no-feature and non-default feature compilation. Do not generalize the
exception to tests, Clippy, or ad-hoc feature combinations.

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
- `.config/policies/test-lanes.toml` is the only lane-admission exception source. Validate it with
  `python3 scripts/ci/test_lane_policy.py check-catalog`.
- Do not add external-service E2E tests to Cargo test targets; invoke them
  explicitly through `cargo run -p nexus-tools --all-features`.
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

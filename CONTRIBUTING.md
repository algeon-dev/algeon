# Contributing

Contributions are very welcome. Algeon brings specialized GPU algorithms into
composable SQL workflows. Its first integration lowers Apache DataFusion
physical plans to native IR and executes them via cuDF, cuGraph, cuVS, and cuML. The most
valuable contributions make physical-to-native lowering, capability admission,
bounded resource management, and GPU execution clearer, faster, and more
reliable.

This repository is Linux-only. Concrete toolchain requirements, system
prerequisites, build environments, and operational specifications are
maintained on the [documentation website](website/docs/build/overview.md).

## SQL Function Terminology

Call `cugraph_*`, `cuvs_*`, and `cuml_*` SQL calls GPU UDTFs (user-defined
table functions): each appears in `FROM` and returns rows. Call
`gpu_list_functions`, `gpu_describe_function`, `gpu_validate_call`, and
`algeon_explain_coverage` metadata UDTFs. Use DataFusion's `TableFunction`
name when discussing its API and UDF for scalar, aggregate, or window APIs. Keep
existing SQL names, Rust identifiers, and catalog URLs unchanged.

## Start With an Issue

Every pull request must start with an issue. Before opening a PR, open an issue
that explains why the change is needed, what problem it solves, and what
behavior or API contract it affects.

This keeps design context visible and reduces review churn, especially for
changes that touch physical-plan lowering, capability admission, native IR,
memory ledgers, or cross-crate boundaries.

For a change whose design or scope is unclear, discuss the proposed solution in
the issue before writing code:

- Solve a demonstrated problem or concrete query pattern. Speculative
  abstractions, unverified performance fixes, and mechanical refactors without
  an agreed technical goal are not useful starting points.
- Identify the owning layer early. If a capability gap belongs in `cudf` or
  `cugraph`, open the issue and implement the fix in its component under
  `components/` rather than introducing an adapter-level workaround.
- For bug reports, specify the affected crate, physical plan or SQL query,
  minimal reproduction steps, sanitized configuration or data, actual result,
  and expected result. Source inspection alone is not confirmation of a bug; if
  the behavior is not reproduced, describe it as a hypothesis and state what
  remains unknown.
- Add a regression test for every bug fix that fails when the fix is reverted.

## AI-Assisted Contributions

AI tools may help with investigation, implementation, and review, but a human
contributor must guide the work and remain accountable for the result.

- Participate meaningfully in every issue and pull request conversation. Do not
  delegate project communication entirely to an AI tool.
- Review, edit, and verify generated analysis, lowering rules, and IR
  transformations before submitting them. You should understand every claim and
  change well enough to explain it during review.
- Disclose material AI assistance and any assumptions or unknowns that could
  affect review.
- Treat an AI-generated hypothesis as a lead, not as evidence. Reproduce the
  behavior against real physical plans or native IR before calling it a
  confirmed bug.
- Do not include credentials, private data, sanitized cloud endpoints, or
  unredacted logs in prompts, issues, or pull requests.

The contributor, not the AI tool, is responsible for the correctness,
relevance, and maintainability of the submission.

## Architecture & Project Boundaries

Algeon maintains strict separation of concerns across workspace
crates:

```text
                 Apache DataFusion Physical Plan
                               │
                               ▼
┌───────────────────────────────────────────────────────────────┐
│ algeon-datafusion (DataFusion Adapter)                        │
│ - Physical optimizer rules (AlgeonNativePhysicalOptimizerRule)│
│ - Table providers, Iceberg integration, Parquet scans         │
│ - NativeExecutionPlan wrappers & DataFusion metric bridges    │
└──────────────────────────────┬────────────────────────────────┘
                               │ uses the compatible facade
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ crates/algeon-query-engine (Facade + Native Execution)      │
│ - Public planning/runtime re-exports                        │
│ - Bounded relational, graph, vector, ML, and source execution│
│ - Host/GPU-resident result composition                      │
└──────────────────────────────┬──────────────────────────────┘
                               │ depends on
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ query-runtime -> query-planning                             │
│ - Runtime: lifecycle, query service, ledger, grant, cache   │
│ - Planning: IR, optimizer, capability, proof, pipeline      │
└──────────────────────────────┬──────────────────────────────┘
                               │ execution uses
                               ▼
        components/cudf      components/cugraph
        cuDF / RMM          cuGraph algorithms
        components/cuvs      components/cuml
        cuVS vector         cuML machine learning
```

### Key Invariants

- **Fix the owning abstraction**: DataFusion-dependent lowering belongs in the
  DataFusion adapter; IR, optimizer, capability, proof, admission contracts, and
  pipeline belong in `crates/algeon-query-planning`; lifecycle, query service,
  ledger, grant, cache, observation, and metrics belong in
  `crates/algeon-query-runtime`; native execution and public facade compatibility
  belong in `crates/algeon-query-engine`. Native cuDF, cuGraph, cuVS, and cuML
  operations belong in their respective component crates. Never paper over an owner or
  RAPIDS limitation with an adapter workaround.
- **One bounded admission contract**: Every native attempt is admitted against
  a single per-device ledger, immutable grants, and transactional envelopes.
  The reservation ledger is integer accounting, not a residency oracle.
- **Structured planning diagnostics**: `AlgeonNativePhysicalOptimizerRule` emits
  structured `PlanningReport`s. Assert reports and structured diagnostics in
  tests, not printed string plans.
- **Explicit aggregation state**: Model partial and final aggregate states
  explicitly (e.g., `AVG` is `SUM + COUNT` followed by final `SUM / SUM`).
- **Structured errors**: Keep `crates/algeon-datafusion/src/error.rs` errors structured and typed;
  convert to `DataFusionError` only at the DataFusion boundary.

For complete architectural details and reference designs, consult the
[Architecture Guides](website/docs/design/index.mdx).

## Subtractive Development

- **Net line count should trend down or stay flat**: Prefer extending existing
  abstractions over adding parallel paths. About to add a second way to do
  something? Consolidate or remove the first.
- **Audit before implementing**: For non-trivial changes, list related existing
  files and functions, and state whether each should be extended, replaced, or
  left alone.
- **No speculative code**: Never add "just in case" configuration, fallbacks, or
  defensive branches without a concrete, reproducible scenario in this
  codebase.
- **Minimal comments**: Default to no comments. Never explain what the code
  does; rename things instead. Comment only non-obvious rationale, invariants,
  safety constraints, external quirks, or business rules. An explicit `drop()`
  in production code requires a one-line comment explaining why it cannot wait
  until the end of scope.

## Workflow

### Branches

Use a descriptive topic branch for a pull request. Keep unrelated edits,
generated files, caches, and test artifacts out of the diff. Keep changes small
and scoped so that ownership and validation are straightforward to review.

### Pull Requests

Use a concise, conventional-commit-style title that makes the affected layer and
intent clear:

```text
feat(lowering): support decimal cast expressions in native projection
fix(query-engine): release memory grant on cancelled pipeline tasks
test(iceberg): add partition-pruning regression coverage
docs: clarify GPU test isolation and lock requirements
```

In the PR description, include:

- The linked issue.
- A concise summary of the change and its user-facing or technical value.
- The behavior, API, or contract that changed.
- The layer that owns the change (DataFusion adapter, `query-planning`,
  `query-runtime`, `query-engine`, `server`, `iceberg-catalog`,
  or a component crate under `components/`).
- The test environment used: CUDA driver and toolkit setup, GPU model, Rust
  toolchain, host OS, and relevant environment variables.
- The exact test commands or verification steps executed.
- Any known gaps, unverified paths, or deliberate non-goals.

Do not describe a compile-only check as native or GPU qualification.

### Lint Exceptions

Workspace lint levels are owned by `[workspace.lints]` in the root `Cargo.toml`
and inherited by every workspace package; the canonical Clippy gate is
`cargo clippy --workspace --all-targets --all-features -- -D warnings`.

- Fix the owning abstraction before adding a lint exception. When a lint
  fires, first ask whether the code repeats shared state across signatures or
  is missing a real owner; suppressions must not become a substitute for that
  repair.
- Clippy exceptions must be narrow and justified. Prefer a per-item
  `#[expect(clippy::..., reason = "...")]` with a concrete, verifiable
  justification. Do not add crate-, module-, or workspace-wide
  `#[allow]` attributes, and never use `#[allow]` as a migration baseline;
  it silently survives the refactor it was meant to be temporary for.

## Local Setup

Algeon is a single recursive checkout. The root workspace owns the
adapter, planning/runtime/engine stack, server, tests, and all four Rust
integration components:

```text
algeon/
├── components/cudf/             # cudf, cudf-sys, rapids-interop
│   └── cudf/                    # pinned cuDF fork submodule
├── components/cugraph/          # cugraph and cugraph-sys
│   └── cugraph/                 # pinned cuGraph fork submodule
├── components/cuvs/             # cuvs and cuvs-sys
│   └── cuvs/                    # pinned cuVS submodule
├── components/cuml/             # cuml and cuml-sys
│   └── cuml/                    # pinned cuML submodule
├── crates/algeon-datafusion/    # DataFusion adapter
│   └── src/
├── crates/                      # Planning, runtime, engine, and support crates
```

Clone the repository with its pinned native sources:

```bash
git clone --recurse-submodules \
  https://github.com/algeon-dev/algeon.git
```

## Native Components

`components/cudf`, `components/cuvs`, `components/cuml`, and `components/cugraph` are members of
this workspace, not independent projects. Their crate boundaries and native
defaults are documented in their respective READMEs. The pinned submodules and
native build order are defined in [Building from Source](website/docs/build/from-source.md).

### Toolchain & Environment Specifications

All toolchain specifications (Rust compiler channels, linker setup with mold
and clang, CMake and host compiler requirements) and environment variables are
defined and maintained in the website documentation:

- [Prerequisites](website/docs/build/prerequisites.md): System dependencies,
  CUDA requirements, compiler pairings, and linker setup.
- [Building from Source](website/docs/build/from-source.md): Recursive clone,
  checkout-local native selectors (`CUDF_INSTALL_DIR`, `ALGEON_NATIVE_DIR`,
  `CUGRAPH_CMAKE_BUILD_DIR`, `CUVS_INSTALL_DIR`, `CUML_INSTALL_DIR`), native build order, and Rust
  workspace compilation.

Always use `--workspace --all-features` in cargo commands, and select tests
with nextest filtersets rather than `-p`. Feature unification ensures that the
entire workspace dependency graph builds with a unified feature set and avoids
unnecessary artifact rebuilds. `cargo nt` is the workspace alias for
`cargo nextest run --workspace --all-features`.

Focused component validation uses the same commands for every native wrapper:

```bash
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo +nightly fmt --all --check
cargo nt -E 'package(<component-packages...>)'
```

The package groups are `package(cudf) | package(cudf-sys) |
package(rapids-interop)`, `package(cugraph) | package(cugraph-sys)`,
`package(cuvs) | package(cuvs-sys)`, and
`package(algeon-cuml) | package(algeon-cuml-sys)`.

## Testing & Validation

For comprehensive testing policies, execution lanes, and test harness details,
refer to the [Tests Documentation](website/docs/build/tests.md).

### Test Isolation & Runners

`query-engine` tests require process isolation because allocator lifecycle
cases install a process-global `DeviceMemoryManager`, and native execution cases
retain backend policy per process. Broad `cargo test --all-features` runs without process
isolation fail closed or corrupt global state. Test execution must use
`cargo-nextest` as specified in the testing guide.

### GPU Serialization Lock

GPU tests take `/tmp/cudf-gpu.lock` automatically through nextest's
run-wrapper (`.config/nextest.toml`, `scripts/ci/nextest_gpu_lock.sh`): shared
for the `gpu` group, exclusive for `gpu-exclusive`. A bare `cargo nt` already
serializes correctly. Non-test GPU commands (benchmarks, servers, tools) still
take the lock themselves:

```shell
flock /tmp/cudf-gpu.lock cargo run -p algeon-server --all-features --bin algeon_server
```

> [!IMPORTANT]
> **Never wrap `cargo nt` or `scripts/check_all.sh` in `flock /tmp/cudf-gpu.lock`**.
> They already take the lock through the nextest run-wrapper; an outer hold
> self-deadlocks them. Run them bare.

### Targeted Edit Loops

Run focused checks first, then expand validation according to the blast radius:

```shell
# Format and clippy
cargo +nightly fmt --all
cargo clippy --workspace --all-targets --all-features

# What you changed, then everything downstream
cargo nt -E 'package(algeon-datafusion) | package(algeon-cuml)'
cargo nt -E 'rdeps(algeon-datafusion)'

# Rerun only the failures (repeat the same -E, or skipped binaries rerun as new)
cargo nt -E 'package(algeon-datafusion)' -R latest
```

### Full Pre-Merge Validation

`scripts/check_all.sh` is the canonical pre-merge gate:

```shell
bash scripts/check_all.sh
```

- `bash scripts/check_all.sh --fast` is a partial edit-loop helper (policy,
  rustfmt, CPU tests) and does not constitute full pre-merge validation. The
  full gate outlives a single terminal session: start it with `nohup` and poll
  `target/check-all/latest/summary.tsv`.
- `.config/nextest.toml` is the sole source of test-lane admission: CPU tests
  are opt-in by a `_cpu` target suffix or a listed binary, `_gpu_exclusive`
  targets run alone and last, and every other test runs in the `gpu` group.
- Do not add external-service E2E tests (such as live Iceberg REST/Glue catalogs)
  to default Cargo test targets; run them explicitly via
  `cargo run -p tools --all-features`
  or fixture scripts under `fixture/`.

# Contributing

Contributions are very welcome. DataFusion Nexus is an experimental GPU query
engine for Apache DataFusion that lowers physical plans to native IR and
executes them via cuDF and cuGraph. The most valuable contributions make
physical-to-native lowering, capability admission, bounded resource management,
and GPU execution clearer, faster, and more reliable.

This repository is Linux-only. Concrete toolchain requirements, system
prerequisites, build environments, and operational specifications are
maintained on the [documentation website](website/docs/build/overview.md).

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
- Identify the owning layer early. If a capability gap belongs in `cudf-nexus` or
  `cugraph-nexus`, open the issue and implement the fix in its component under
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

DataFusion Nexus maintains strict separation of concerns across workspace
crates:

```text
Apache DataFusion Physical Plan
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ datafusion-nexus (Root Adapter)                             │
│ - Physical optimizer rules (NexusNativePhysicalOptimizerRule)│
│ - Table providers, Iceberg integration, Parquet scans       │
│ - NativeExecutionPlan wrappers & DataFusion metric bridges  │
└──────────────────────────────┬──────────────────────────────┘
                               │ lowers to
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ crates/nexus-query-engine (Native Engine)                   │
│ - DataFusion-independent QueryPlan IR                       │
│ - Capability-first admission & structured rejection facts   │
│ - Per-device reservation ledger & immutable memory grants   │
│ - Pipeline compilation & DAG execution                      │
└──────────────────────────────┬──────────────────────────────┘
                               │ executes via
                               ▼
        ┌──────────────────────┴──────────────────────┐
        ▼                                             ▼
┌──────────────────────────────┐        ┌─────────────────────────────┐
│ components/cudf              │        │ components/cugraph          │
│ cuDF / RMM / KvikIO          │        │ cuGraph C++ / C API          │
└──────────────────────────────┘        └─────────────────────────────┘
```

### Key Invariants

- **Fix the owning abstraction**: DataFusion-dependent lowering belongs in the
  root adapter; DataFusion-independent IR, capability admission, and execution
  belong in `crates/nexus-query-engine`; native cuDF and cuGraph operations
  belong in their respective component crates. Never paper over an engine or
  RAPIDS limitation with an adapter workaround.
- **One bounded admission contract**: Every native attempt is admitted against
  a single per-device ledger, immutable grants, and transactional envelopes.
  There is no strict/adaptive product mode, tenant policy, grant band, or slot
  guarantee. The reservation ledger is integer accounting, not a residency
  oracle.
- **Structured planning diagnostics**: `NexusNativePhysicalOptimizerRule` emits
  structured `PlanningReport`s. Assert reports and structured diagnostics in
  tests, not printed string plans.
- **Explicit aggregation state**: Model partial and final aggregate states
  explicitly (e.g., `AVG` is `SUM + COUNT` followed by final `SUM / SUM`).
- **Structured errors**: Keep `src/error.rs` errors structured and typed;
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
- The layer that owns the change (root adapter, `nexus-query-engine`,
  `nexus-server`, `nexus-iceberg`, or a component crate under `components/`).
- The test environment used: CUDA driver and toolkit setup, GPU model, Rust
  toolchain, host OS, and relevant environment variables.
- The exact test commands or verification steps executed.
- Any known gaps, unverified paths, or deliberate non-goals.

Do not describe a compile-only check as native or GPU qualification.

## Local Setup

DataFusion Nexus is a single recursive checkout. The root workspace owns the
adapter, engine, server, tests, and all three Rust integration components:

```text
datafusion-nexus/
├── components/cudf/             # cudf-nexus, cudf-nexus-sys, rapids-interop
│   └── cudf/                    # pinned cuDF fork submodule
├── components/cugraph/          # cugraph-nexus and cugraph-nexus-sys
│   └── cugraph/                 # pinned cuGraph fork submodule
├── components/cuvs/             # cuvs-nexus and cuvs-nexus-sys
│   └── cuvs/                    # pinned official cuVS submodule
├── crates/                      # DataFusion-free engine and support crates
└── src/                         # DataFusion adapter
```

Clone the repository with its pinned native sources:

```bash
git clone --recurse-submodules \
  https://github.com/DataFusion-Nexus/datafusion-nexus.git
```

## Native Components

`components/cudf`, `components/cuvs`, and `components/cugraph` are members of
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
  checkout-local native selectors (`CUDF_INSTALL_DIR`,
  `CUGRAPH_CMAKE_BUILD_DIR`, `CUVS_INSTALL_DIR`), native build order, and Rust
  workspace compilation.

Always use `--all-features` in cargo commands. Feature unification ensures that
the entire workspace dependency graph builds with a unified feature set and
avoids unnecessary artifact rebuilds.

Focused component validation uses the same commands for every native wrapper:

```bash
cargo clippy -p <component-packages...> --all-targets --all-features -- -D warnings
cargo +nightly fmt --all --check
flock /tmp/cudf-gpu.lock cargo nextest run \
  -p <component-packages...> --all-features --no-fail-fast
```

The package groups are `cudf-nexus cudf-nexus-sys rapids-interop`,
`cugraph-nexus cugraph-nexus-sys`, and `cuvs-nexus cuvs-nexus-sys`.

## Testing & Validation

For comprehensive testing policies, execution lanes, and test harness details,
refer to the [Tests Documentation](website/docs/build/tests.md).

### Test Isolation & Runners

`nexus-query-engine` tests require process isolation because allocator lifecycle
cases install a process-global `DeviceMemoryManager`, and native execution cases
retain backend policy per process. Broad `cargo test --all-features` runs without process
isolation fail closed or corrupt global state. Test execution must use
`cargo-nextest` as specified in the testing guide.

### GPU Serialization Lock

Direct GPU commands must be serialized with `flock /tmp/cudf-gpu.lock`:

```shell
flock /tmp/cudf-gpu.lock cargo nextest run -p nexus-query-engine --all-features --lib
flock /tmp/cudf-gpu.lock cargo nextest run -p datafusion-nexus --all-features --test root_gpu_tests
```

> [!IMPORTANT]
> **Never wrap `scripts/check_all.sh` in `flock /tmp/cudf-gpu.lock`**.
> `scripts/check_all.sh` manages GPU locks internally per phase. Wrapping the
> script externally causes a self-deadlock. Run the script bare.

### Targeted Edit Loops

Run focused checks first, then expand validation according to the blast radius:

```shell
# Format and clippy
cargo +nightly fmt --all
cargo clippy --workspace --all-targets --all-features

# Fast CPU and planning loops
cargo nextest run -p datafusion-nexus --all-features --test cpu_contract_tests
cargo nextest run -p datafusion-nexus --all-features --test native_planning_tests
cargo nextest run -p nexus-query-engine --all-features --test native_contract_tests

# Targeted GPU tests (must be serialized)
flock /tmp/cudf-gpu.lock cargo nextest run -p datafusion-nexus --all-features --test native_exec_tests
flock /tmp/cudf-gpu.lock cargo nextest run -p nexus-query-engine --all-features --test query_service_allocator_lifecycle
```

### Full Pre-Merge Validation

`scripts/check_all.sh` is the canonical pre-merge gate:

```shell
bash scripts/check_all.sh
```

- `bash scripts/check_all.sh --fast` and `--lane <LANE>` are partial edit-loop
  helpers and do not constitute full pre-merge validation.
- `.config/policies/test-lanes.toml` is the sole source of lane admission. After
  modifying it, regenerate nextest configuration:
  `python3 scripts/ci/test_lane_policy.py generate-nextest --write`.
- Do not add external-service E2E tests (such as live Iceberg REST/Glue catalogs)
  to default Cargo test targets; run them explicitly via
  `cargo run -p nexus-tools --all-features`
  or fixture scripts under `fixture/`.

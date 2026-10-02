# AGENTS.md

## Working principles

- Fix the owning abstraction or contract, not a failing query with an adapter
  workaround. Extend `cudf`, `cugraph`, or `cuvs` when the gap belongs there.
- Keep changes scoped: preserve unrelated worktree edits, public behavior, and
  contract tests unless the task changes them; avoid opportunistic cleanup.
- Grow the system in layers: start from the smallest version that works end to
  end, and add each capability on top of a product that already works; never
  trade a working product for unfinished complexity.

## Subtractive development

- Prefer extending existing code over adding parallel paths and the smallest
  change that fully solves the current requirement. Reuse or consolidate
  existing paths within the task; ask only when doing so requires an
  unrequested public-contract change or expansion of scope.
- Never add speculative abstractions, configuration, or "just in case"
  fallbacks; every defensive branch should protect a supported input, failure
  mode, or invariant. Remove redundant paths involved in the requested change
  only after confirming that no supported caller or contract requires them.

## Comments

- Omit comments that only narrate code. Comment non-obvious rationale,
  invariants, safety constraints, external quirks, or business rules the code
  cannot express.
- Public API documentation describes observable contracts, not incidental
  implementation details.
- An explicit `drop()` in production code needs a one-line comment on why the
  value cannot wait until end of scope (lock window, re-entrancy, `RefCell`
  re-borrow); otherwise delete the `drop()`. Test code is exempt.

## Technical writing and anti-slop

- **No false agency**: Never attribute human actions, consciousness, or feelings
  to algorithms, data, or compilers (e.g. do not write `the planner decides`,
  `the data tells us`, or `the engine understands`). Name the specific
  condition, hardware constraint, or user input.
- **No em-dashes**: The em-dash character (`—`) is banned across code comments
  and technical documentation. Use commas, semicolons, parentheses, or separate
  sentences.
- **No rule of three or buzzword triads**: Prohibit listing three generic
  adjectives (e.g. `fast, reliable, and scalable`, `flexible, robust, and extensible`).
- **No binary contrasts and colon reveals**: Avoid theatrical structures such as
  `It is not about X; it is about Y` or `The reality: X`.
- **Burstiness and sentence length**: Vary sentence length and shape; runs of
  same-length, subject-verb-object sentences read as generated text.
- **Fact-dense, source-locked technical claims**: Every paragraph must advance
  a concrete invariant, benchmark number, architectural boundary, or API contract.
  Delete any sentence that functions purely as ungrounded promotional uplift.
- **Direct reporting and interaction**: Omit conversational filler, flattery, and
  apologies. State changed files, line ranges, non-obvious rationale, verification
  commands and outcomes, and open technical decisions directly.
- **Domain anti-slop policy**: prose about the GPU query engine, physical
  optimization, or CUDA memory allocation follows
  [`.config/policies/algeon_domain_anti_slop.md`](.config/policies/algeon_domain_anti_slop.md).

## Project boundaries

`algeon-datafusion` lowers DataFusion physical plans through the compatible
`query-engine` facade and executes them via the Rust components under
`components/`. DataFusion-dependent code belongs in
`crates/algeon-datafusion`; IR, optimizer, capability, proof, admission
contracts, and pipeline belong in
`crates/algeon-query-planning`; attempt lifecycle, query service, device ledger,
immutable grants, cache, observation, and metrics belong in
`crates/algeon-query-runtime`; native execution composition and facade
compatibility belong in `crates/algeon-query-engine`.

- `cudf`, `cugraph`, `cuvs`, and `cuml` are members of the root workspace. Every
  published package carries an `algeon-` prefix, because the bare RAPIDS names
  are taken on crates.io (`cudf`/`cudf-sys` by a third party, `cuvs`/`cuvs-sys`
  by NVIDIA's rapidsai/cuvs). `[workspace.dependencies]` maps each one through
  a `package = "algeon-…"` alias under its bare key, so Rust imports stay
  `cudf`, `cugraph`, `cuvs`, and `cuml`, and `dep:` feature strings keep
  resolving.
  Directory names match package names; refer to layers by their bare alias.
  Every renamed crate pins `[lib] name` explicitly: without it Cargo derives the
  library name from the package name, so a rename silently moves `use
  query_engine::…` and the `public-api` snapshots, which record library names.
- Each crate declares `publish` directly: the release closure as
  `publish = ["crates-io"]`, everything else as `publish = false`. Never set
  `[workspace.package].publish`; inheriting it once silently reverted the
  whole closure to unpublishable. `scripts/ci/check_workspace_publication.py`
  holds the allowlist and fails closed.
- Native sources are the root-registered submodules at
  `components/cudf/cudf`, `components/cugraph/cugraph`, and
  `components/cuvs/cuvs`, and `components/cuml/cuml`; native defaults live
  under `target/native/`.
- The root `.gitmodules`, `Cargo.lock`, workspace, and `scripts/check_all.sh`
  are the only source registry, lockfile, workspace, and full validation gate.
- Default members are the adapter, `workloads`, and `query-engine`.
  Bench, tools, and examples are explicit non-default members.
- Linux only. MSRV is Rust 1.95; formatting requires nightly.

## Native GPU engine core

- Support requires a correct kernel, types, and semantics; an operator lacking one is structured unsupported. A missing memory estimate is never a reason to reject.
- Admission issues a service-configured cap, not a predicted requirement: `query-planning` proves the plan is executable, `query-runtime` admits it against the device ledger, and the query domain's allocator enforces the cap while `query-engine` executes.
- The query domain's allocator is the single arbiter of a query's allocation limit. Never add a second Rust-side reservation, per-operator subgrant, or estimate-derived veto.
- Every production query-engine execution belongs to one attempt. A plain
  DataFusion execution creates an internal attempt at its execution boundary;
  component runtimes may be used independently.
- Each native fragment runs one compiled schedule. An attempt owns one
  admission claim and one immutable grant. Every native fragment the attempt
  executes shares that grant, including sequential executions such as a CTAS
  input and its result, and deferred refinement stays inside it. A fragment
  planned after the claim joins the grant when the granted device provides
  every capability the fragment requires; otherwise it fails with a typed
  error. The grant is released once the attempt is terminal and no fragment is
  running. A second admission claim returns a typed error before queue or
  ledger mutation. Keep one authoritative ledger per device.
- Admit by operator capability, never customer, SQL text, names, one-off algorithm allowlists, or plan-shape matching.
- Keep one truth source; consolidate redundant constructors, flags, matchers, or ledgers within the requested change, preserving supported callers and contracts.
- Output and resident facts carry scope, occurrence, schema, and provenance; allocation failures are typed at their source and reported with their cleanup disposition.
- `PlanningReport` records each candidate once as selected, not supported
  (with the first contract it misses), or not selected by cost, with
  structured reasons.
- Keep backend memory models in component owners and IR, facts, schedules, and admission backend neutral.
- Before adding an operator, identify its allocation, facts, lifetime, diagnostics, and test owners from existing code. Ask only if unresolved ownership affects the contract.
- The ledger is integer accounting, not a residency oracle; model aggregate partial/final state explicitly.

## Imports

- Import items in the file's top `use` block, or in `mod tests` when only tests
  use them. Reach variants and associated functions through the imported type.
- Use `super::` only for the direct parent; use `crate::` above it.
- On name collisions, the local item keeps the bare name and foreign items use
  an origin prefix, such as `CudfDataType`, `ArrowDataType`, or
  `DataFusionResult`. Prefix both sides when both are foreign. Keep the local
  `Result` bare; use `std::result::Result` inline only when its arity differs.
- Inline paths are otherwise limited to visibility qualifiers, attributes,
  doc links, calls through imported modules, sibling private modules, and enum
  translation matches. Gate imports with the same `cfg` as their uses.

## Build and test

Use `--workspace --all-features` for ordinary Cargo build, check, lint, and test
commands and select tests with nextest filtersets, never `-p`: one build
configuration keeps those commands warm. Keep two explicit feature-coverage
exceptions in `scripts/check_all.sh`: the existing
`cargo hack check --workspace --each-feature --exclude-features default --all-targets --locked --keep-going`
package shards, whose union must match that command's `--print-command-list`
output; and a shipped server clippy lane that selects only the `algeon_server`
binary through its manifest, with `--no-default-features`, the packaging feature
set, `--locked`, and `-D warnings`. The shipped lane does not select tests,
examples, or all targets, and uses its own persistent target directory.
Packaging and this lane read the shipped feature set from one source. The full
gate requires both exceptions to pass; focused and `--fast` runs remain partial
validation.

`cargo nt` is `cargo nextest run --workspace --all-features`.

```bash
cargo +nightly fmt --all
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo nt -E 'package(<package>)'           # what you changed
cargo nt -E 'rdeps(<package>)'             # everything downstream
cargo nt -E '<same filterset>' -R latest   # rerun only failures
bash scripts/check_all.sh --fast           # policy, fmt, CPU tests (partial)
bash scripts/check_all.sh                  # full pre-merge gate
```

Select commands by the change's blast radius; they are not a checklist for
every task. The full gate outlives a single tool call: start it with `nohup`
and poll `target/check-all/latest/summary.tsv`. Full validation is one
complete gate run on the final source state; `--fast`, focused runs, and
`-R latest` reruns are partial.

Run focused checks first and broaden when the affected contracts or failures
warrant it. Finish when the requested work and relevant checks are complete,
with no unresolved task-related failures. For ordinary implementation work,
passing the relevant focused lint and behavior tests is the stopping
condition. Run the full gate only for pre-merge validation or when the task
explicitly requests it. Report a focused run as partial; do not automatically
expand it into the full gate. Read-only reviews and prose-only edits do not
require Cargo or GPU checks unless requested. The non-default bench package
uses the same commands: `cargo nt -E 'package(bench)'`.

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

- GPU tests take `/tmp/cudf-gpu.lock` through nextest's run-wrapper: shared for
  the `gpu` group, exclusive for `gpu-exclusive`. Do not wrap nextest or
  `scripts/check_all.sh` in `flock`; non-test GPU commands (benchmarks, servers,
  tools) still take the lock themselves.
- `.config/nextest.toml` is the only test-lane source: CPU tests are opt-in by a
  `_cpu` target suffix or a listed binary, `_gpu_exclusive` targets run alone,
  and every other test runs in the `gpu` group.
- Do not add external-service E2E tests to Cargo test targets; invoke them
  explicitly through `cargo run -p tools --all-features`.
- Use `bash scripts/dev/clean_target_artifacts.sh` only to report removable build
  artifacts; cleanup is never implicit.
- `scripts/check_all.sh` is the only top-level script entry. Build tooling lives
  in `scripts/build/`, CI helpers in `scripts/ci/`, manual helpers in
  `scripts/dev/`, and policy data in `.config/policies/`.
- Keep `components/algeon-build-support` (library `rapids_build_support`)
  std-only: it is the build-dependency shared by every package whose build
  script links RAPIDS native libraries, so it must not depend on anything that
  is built after it.

## Public contracts

- Treat reports, TSV families, metrics, and native execution-plan wrappers as
  explicit contracts. Before publication, change a contract atomically with all
  workspace consumers and fixtures; do not retain deprecated paths or fabricate
  values for removed behavior. Version persisted report and TSV schemas when
  fields, units, or defined value semantics change. Each changed TSV family
  carries an explicit schema field and a matching header constant. Rename a
  metric in `NATIVE_ALGEON_DATAFUSION_METRIC_NAMES` when its meaning or unit
  changes; preserve its name otherwise. Internal Rust refactors and fixes that
  restore documented semantics do not by themselves require a wire-schema
  version change.
- TSV, DataFusion metrics, and `QueryMetrics` are projections of the runtime
  observation snapshot, not a second event store.
- Preserve workload environment variables and item shapes.
- Keep `crates/algeon-datafusion/src/error.rs` errors structured; convert to `DataFusionError` only at
  the DataFusion boundary.
- `algeon_datafusion_native` (`ALGEON_NATIVE_PHYSICAL_OPTIMIZER_RULE_NAME`) is
  public and is the rule deduplication key.

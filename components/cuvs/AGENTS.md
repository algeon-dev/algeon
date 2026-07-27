# AGENTS.md

## Working principles

- Fix the owning runtime, interop, FFI, or native-discovery contract instead of
  hiding a failure in an algorithm wrapper. Extend `cudf-nexus` or
  `rapids-interop` when the missing ownership or execution primitive belongs
  there.
- Keep changes scoped. Preserve unrelated worktree edits and avoid
  opportunistic cleanup, generated-file churn, or native rebuilds outside the
  task.
- Preserve public behavior and contract tests unless the task changes them.
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

`cuvs-nexus` is the runtime-bound Rust integration for the scoped cuVS surface
used by DataFusion Nexus. SQL planning and query execution belong in
`datafusion-nexus`, not this repository.

- Safe algorithm, tensor, ownership, and completion APIs belong in
  `cuvs-nexus/`; raw generated C bindings and native discovery belong in
  `cuvs-nexus-sys/`.
- `cudf-nexus` and `rapids-interop` are sibling path dependencies. The pinned
  official cuVS source is the `cuvs/` submodule.
- Default members are the two library crates. `tools/xtask` is unpublished
  maintenance tooling and remains an explicit non-default member.
- Linux only. MSRV is Rust 1.94; formatting requires nightly.

## Architecture contracts

- `CuvsResources` binds one exact `cudf::Runtime` device, CUDA stream, and
  process-lifetime current memory resource. Reject binding drift and arbitrary
  custom memory resources before calling cuVS.
- Runtime-bound resources, indexes, models, and prepared tensor leases remain
  `!Send` and `!Sync`. Do not weaken those markers without a demonstrated native
  concurrency contract.
- Prepared cuDF reads carry explicit producer ownership, completion witnesses,
  and incomplete-drop behavior. Complete reads only after cuVS work is enqueued;
  never release producer storage before the consumer fence permits it.
- Checked-in bindings come only from the pinned cuVS revision through
  `cargo xtask bindings generate`. Normal library builds, docs, and lints must
  not regenerate bindings or gain a bindgen build dependency.
- Native discovery accepts a canonical installed cuVS package, never a private
  `_deps` copy. Preserve the selected cuDF/RMM identity, shared CUDA runtime,
  source/version markers, and dynamic-loader checks.
- Validate native output ranges and floating-point validity before constructing
  or returning safe Rust values. A successful FFI status alone is not proof of
  a valid output.

## Build and test

```bash
# Native-free repository, package, workspace, and checked-binding contracts
cargo xtask verify
cargo clippy -p xtask --all-features --all-targets -- -D warnings
DOCS_RS=1 RUSTDOCFLAGS="-D warnings" cargo doc -p cuvs-nexus-sys --all-features --no-deps
cargo +nightly fmt --all

# Source and canonical native prefix setup
scripts/bootstrap_cuvs.sh
scripts/build_libcuvs.sh
cargo xtask bindings check

# Targeted native edit loops
flock /tmp/cudf-gpu.lock cargo test -p cuvs-nexus-sys --test abi_contract
flock /tmp/cudf-gpu.lock cargo test -p cuvs-nexus --test gpu

# Full native workspace surface
cargo check --workspace --all-features --all-targets
cargo clippy --workspace --all-features --all-targets -- -D warnings
flock /tmp/cudf-gpu.lock cargo test --workspace --all-features --no-fail-fast
```

Run focused checks first, then broaden validation with the change's blast
radius. `cargo xtask verify` is native-free maintenance validation and must not
be reported as the full native or GPU test surface.

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

- Serialize every GPU test with `flock /tmp/cudf-gpu.lock`.
- `cargo xtask verify` is the canonical maintenance-contract entry point.
  Binding refresh and reproducibility checks live under
  `cargo xtask bindings`; do not add replacement one-off check scripts.
- `scripts/bootstrap_cuvs.sh` and `scripts/build_libcuvs.sh` are the only
  top-level operational scripts. Keep source, workspace, package, and binding
  policy in `tools/xtask`.
- Keep `CUVS_NEXUS_SYS_SKIP_NATIVE` and `DOCS_RS` as maintenance environments,
  not additive `cuvs-nexus-sys` Cargo features. The sys crate must remain
  usable for native-free linting, documentation, and package verification.
- Rebuild the canonical prefix and rerun binding plus ABI checks after changing
  the source pin, native build flags, headers, discovery, loader metadata, or
  generated bindings.
- `CUVS_ROOT` is an explicit build-time package selector, not a runtime loader
  configuration. Do not add ambient sibling, cuGraph build-tree, or private
  dependency fallbacks.

## Public contracts

- Preserve package/import identities: `cuvs-nexus` imports as `cuvs`,
  `cuvs-nexus-sys` imports as `cuvs_sys`, and the sys crate keeps
  `links = "cuvs_c"`.
- Preserve `CuvsResources` execution-binding identity and the ownership and
  completion semantics of `PreparedTensorLease`,
  `PreparedValidatedTensorLease`, and `PreparedReadCompletion`.
- Keep `ErrorKind`, `ErrorStatus`, operation, context, and source errors
  structured. Consumers must not need to parse display-message text.
- Preserve native metadata consumed through `DEP_CUVS_C_*`, especially runtime
  library directories. Downstream executables remain responsible for their own
  loader-visible deployment or RUNPATH.

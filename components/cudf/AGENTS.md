# AGENTS.md

## Working principles

- Fix ownership, execution, memory, or FFI gaps in the layer that owns them;
  do not hide them behind a DataFrame-specific workaround.
- Keep changes scoped and preserve unrelated worktree edits, generated output,
  caches, and native build trees.
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

`components/cudf` provides service-facing Rust DataFrame and runtime integration
over libcudf. Query planning belongs in the root `datafusion-nexus` adapter.

- Safe APIs belong in `cudf-nexus/`; the C++ shim and raw FFI belong in
  `cudf-nexus-sys/`; cross-library ownership and synchronization vocabulary
  belongs in `rapids-interop/`.
- `components/cudf/cudf/` is the pinned DataFusion-Nexus cuDF fork used for
  native builds.
- The three component crates are members of the root workspace; preserve their
  dependency direction and package/import identities.
- Default native output is `target/native/cudf-install`; use the explicit
  `CUDF_INSTALL_DIR` or `CUDF_CPP_BUILD_DIR` selectors when needed.
- Linux only. MSRV is Rust 1.94; formatting requires nightly.

## Architecture contracts

- Keep runtime device, stream, memory-resource, and execution provenance
  explicit across every GPU boundary. Do not silently fall back to ambient
  state where a runtime-bound API exists.
- Distinguish borrowed views, retained shared ownership, owned transfers,
  device copies, and host materialization. Preserve prepared-read completion
  and release ordering.
- Preserve device-aware destruction and the existing `Send`/`Sync` contract for
  each owned, borrowed, runtime, and interop type. Re-audit `docs/SAFETY.md` when
  changing unsafe code, lifetimes, raw handles, or thread traits.
- Keep host-copy Arrow interop separate from Arrow C Device and RAPIDS
  device-resident interop paths.

## Build and test

```bash
cargo check --workspace --all-features --all-targets
cargo clippy --workspace --all-features --all-targets -- -D warnings
cargo +nightly fmt --all

# Focused contract loops
cargo test -p rapids-interop --all-features
flock /tmp/cudf-gpu.lock cargo test -p cudf-nexus --all-features --test semantic_contracts

# Full workspace surface
flock /tmp/cudf-gpu.lock cargo test --workspace --all-features --no-fail-fast
```

Run focused checks first, then broaden validation with the change's blast
radius. Report exact commands and selected native prefix or GPU environment.

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
- Set at least one native discovery path explicitly: `CUDF_INSTALL_DIR` selects
  an installed libcudf prefix, while `CUDF_CPP_BUILD_DIR` names a preconfigured
  build tree. When both are set, the install prefix takes precedence; preserve
  build fingerprints.
- Native source and component checks run from the monorepo root. The root
  `scripts/check_all.sh` is the only full gate; component crates do not carry a
  second workspace gate or lockfile.
- Keep the cuDF submodule pin and C++ shim ABI synchronized. Test both Rust
  contracts and native failure/cleanup paths when changing FFI.

## Public contracts

- Preserve package/import identities: `cudf-nexus` imports as `cudf`,
  `cudf-nexus-sys` imports as `cudf_sys`, `rapids-interop` imports as
  `rapids_interop`, and the sys crate keeps `links = "cudf_rust"`.
- Preserve `DEP_CUDF_RUST_*` native metadata and exact libcudf/RMM runtime
  identity for downstream consumers.
- Keep structured errors, execution-boundary evidence, ownership state
  machines, and prepared-transfer item shapes stable unless explicitly changed.
- Preserve output validation before safe Rust adoption and cleanup of partially
  constructed native objects on failure.

# AGENTS.md

## Working principles

- Fix ownership, execution, memory, or FFI gaps in the layer that owns them;
  do not hide them behind a DataFrame-specific workaround.
- Keep changes scoped and preserve unrelated worktree edits, generated output,
  caches, and native build trees.

## Project boundaries

`components/cudf` provides service-facing Rust DataFrame and runtime integration
over libcudf. Query planning and runtime ownership follow the root
`datafusion-nexus` ownership map.

- Safe APIs belong in `cudf-nexus/`; the C++ shim and raw FFI belong in
  `cudf-nexus-sys/`; cross-library ownership and synchronization vocabulary
  belongs in `rapids-interop/`.
- `components/cudf/cudf/` is the pinned DataFusion-Nexus cuDF fork used for
  native builds.
- The three component crates are members of the root workspace; preserve their
  dependency direction and package/import identities.
- Default native output is `target/native/cudf-install`; use the explicit
  `CUDF_INSTALL_DIR` or `CUDF_CPP_BUILD_DIR` selectors when needed.

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
cargo clippy -p cudf-nexus -p cudf-nexus-sys -p rapids-interop \
  --all-targets --all-features -- -D warnings
cargo +nightly fmt --all --check
flock /tmp/cudf-gpu.lock cargo nextest run \
  -p cudf-nexus -p cudf-nexus-sys -p rapids-interop \
  --all-features --no-fail-fast
```

Run focused checks first, then broaden validation with the change's blast
radius. Report exact commands and selected native prefix or GPU environment for
native work.

## Testing and repository policy

- Set at least one native discovery path explicitly: `CUDF_INSTALL_DIR` selects
  an installed libcudf prefix, while `CUDF_CPP_BUILD_DIR` names a preconfigured
  build tree. When both are set, the install prefix takes precedence; preserve
  build fingerprints.
- Native source and component checks run from the monorepo root. The root
  `scripts/check_all.sh` is the only full gate.
- Checked-in production and test-support bindings are maintained only through
  `scripts/dev/bindings.sh cudf [--check]`.
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

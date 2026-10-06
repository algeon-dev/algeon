# AGENTS.md

Follow the root [AGENTS.md](../../AGENTS.md) for shared development and
validation rules.

## Project boundaries

`components/cudf` provides service-facing Rust DataFrame and runtime integration
over libcudf. Query planning and runtime ownership follow the
[module boundaries](../../website/docs/design/index.mdx#crates-and-ownership).

- Safe APIs belong in `algeon-cudf/`; the C++ shim and raw FFI belong in
  `algeon-cudf-sys/`; cross-library ownership and synchronization vocabulary
  belongs in `components/algeon-interop/`. The public `cudf::interop::contract`
  entry re-exports that owner crate for consumers that already depend on `cudf`.
- `components/cudf/cudf/` is the pinned Algeon cuDF fork used for
  native builds.
- Both component crates and `components/algeon-interop` are members of the
  root workspace; preserve their dependency direction and package/import
  identities.
- Default native output is `target/native/cudf-install`; use the explicit
  `CUDF_INSTALL_DIR` or `CUDF_CPP_BUILD_DIR` selectors when needed.

## Architecture contracts

- Keep runtime device, stream, memory-resource, and execution provenance
  explicit across every GPU boundary. Do not silently fall back to ambient
  state where a runtime-bound API exists.
- `DataFrame`, `Series`, and `GroupBy` methods that submit GPU work, allocate,
  or materialize run only through `.on(Exec)`; `Exec::RETAINED` selects the
  inputs' own stream and memory resource. Reusable handles take the `Exec` as
  their last argument. A handle whose native state binds its execution context
  at construction, such as the mixed left join state, runs on that context and
  takes no override. Metadata access and `Expr` construction take no execution
  scope.
- Distinguish prepared exports, retained shared ownership, device copies, and
  host materialization. Preserve prepared-read completion ordering.
- Preserve device-aware destruction and the existing `Send`/`Sync` contract for
  each owned, borrowed, runtime, and interop type. Re-audit `algeon-cudf-sys/SAFETY.md` when
  changing unsafe code, lifetimes, raw handles, or thread traits.
- Keep host-copy Arrow interop separate from supported RAPIDS device-resident
  interop. Add an external device interchange API only with a supported
  producer or consumer and end-to-end ownership, synchronization, and dtype
  tests. Prefer the Arrow C Device interface for Arrow-compatible device data;
  document required conversions and do not promise universal zero-copy.

## Build and test

```bash
cargo nt -E 'package(algeon-cudf) | package(algeon-cudf-sys) | package(algeon-interop)'
```

## Testing and repository policy

- Set at least one native discovery path explicitly: `CUDF_INSTALL_DIR` selects
  an installed libcudf prefix, while `CUDF_CPP_BUILD_DIR` names a preconfigured
  build tree. When both are set, the install prefix takes precedence; preserve
  build fingerprints.
- Checked-in production and test-support bindings are maintained only through
  `scripts/dev/bindings.sh cudf [--check]`.
- Keep the cuDF submodule pin and C++ shim ABI synchronized. Test both Rust
  contracts and native failure/cleanup paths when changing FFI.

## Public contracts

- Preserve package/import identities: `algeon-cudf` imports as `cudf`,
  `algeon-cudf-sys` imports as `cudf_sys`, `algeon-interop` imports as
  `rapids_interop`, and the sys crate keeps `links = "cudf_rust"`.
- Keep shared interop types owned by `algeon-interop`; expose them to `cudf`
  consumers through `cudf::interop::contract` without wrappers or copies.
- Preserve `DEP_CUDF_RUST_*` native metadata and exact libcudf/RMM runtime
  identity for downstream consumers.
- Keep structured errors, ownership state machines, and the item shapes and
  completion contracts of supported prepared transfers stable unless explicitly
  changed. Validate observable transfer and allocation behavior at their owning
  boundaries. Query-wide observations belong to query-runtime; cuDF operations
  do not fail solely because diagnostic evidence could not be recorded.
- Preserve output validation before safe Rust adoption and cleanup of partially
  constructed native objects on failure.

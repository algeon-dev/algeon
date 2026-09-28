# AGENTS.md

## Working principles

- Fix graph ownership, execution, interop, or native-discovery gaps in the
  owning layer; do not add algorithm-specific fallback glue.
- Keep changes scoped and preserve unrelated worktree edits, submodule state,
  generated build output, and native caches.
- Before adding a dependency or replacement implementation, check existing
  capabilities and types.

## Project boundaries

`components/cugraph` provides service-facing Rust graph APIs over libcugraph.
SQL planning, runtime admission, and query execution follow the root
`algeon-datafusion` ownership map.

- Safe graph, algorithm, result, and cuDF interop APIs belong in
  `algeon-cugraph/`; bindgen and native contract resolution belong in
  `algeon-cugraph-sys/`.
- `cudf` and `cuvs` are root-workspace component dependencies. Shared interop
  contract types are imported from `cudf::interop::contract`, owned by
  `components/algeon-interop`. `components/cugraph/cugraph/` is the pinned
  native source; the default cuVS prefix is `target/native/cuvs-install`,
  while an explicitly selected compatible external prefix is supported.
- Preserve the two component crates and their dependency direction inside the
  root workspace.

## Architecture contracts

- cuDF, cuGraph, and compatible cuVS 26.08.x must resolve one RMM runtime. The
  default root cuVS prefix requires its pinned source marker; an explicitly
  selected external prefix may omit that Algeon marker. Never accept a private
  `_deps` RMM or libcuvs copy.
- Resolve the native cuGraph contract from exactly one source: the formal CMake
  export or all three explicit include, link, and runtime path lists.
- Runtime-bound handles bind the exact cuDF device, stream, and memory resource
  without a second RAFT stream pool. Public active GPU wrappers preserve the
  `!Send + !Sync` assertions in `type_threading_contract_cpu`; `ParkedGraph` remains
  `Send + !Sync`.
- Preserve prepared and shared-owner cuDF interop paths, including their
  completion ordering.
- Preserve native ownership, graph type-state, null checks, and result dtype
  decoding across FFI boundaries.
- Checked-in bindings are generated only by `scripts/dev/bindings.sh cugraph`;
  wrapper or upstream header changes require regeneration, binding coverage,
  and ABI checks.

## Build and test

```bash
# Canonical native build
CUDF_INSTALL_DIR="$PWD/target/native/cudf-install" \
CUVS_INSTALL_DIR="$PWD/target/native/cuvs-install" \
CUGRAPH_CMAKE_BUILD_DIR="$PWD/target/native/cugraph-build" \
scripts/build/build_libcugraph.sh

cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo +nightly fmt --all --check
cargo nt -E 'package(algeon-cugraph) | package(algeon-cugraph-sys)'
```

Run focused checks first, then broaden validation with the change's blast
radius. Report exact native prefixes, loader paths, and GPU environment for
native work.

## Testing and repository policy

- Cargo auto-build is enabled by default and may reconfigure
  `target/native/cugraph-build`; set `CUGRAPH_SYS_AUTO_BUILD=0` when only an existing
  native contract may be used. Keep the manual and Cargo paths aligned.
- Set `CUGRAPH_NATIVE_INCLUDE_DIRS`, `CUGRAPH_NATIVE_LINK_DIRS`, and
  `CUGRAPH_RUNTIME_LIB_DIRS` together or not at all. Use
  `CUGRAPH_CMAKE_BUILD_DIR` for a formal prebuilt export.
- `scripts/build/build_libcugraph.sh` also installs the registry-consumer
  prefix under `target/native/cugraph-install`; validate checked bindings with
  `scripts/dev/bindings.sh cugraph --check`.
- Do not suppress graph validation, synchronization, or dtype errors to make an
  algorithm test pass.

## Public contracts

- Preserve package/import identities: `algeon-cugraph` imports as `cugraph`,
  `algeon-cugraph-sys` imports as `cugraph_sys`, and the sys crate keeps
  `links = "cugraph"`.
- Preserve `DEP_CUGRAPH_*` native include, link, and runtime metadata for
  downstream consumers.
- Keep structured `ErrorKind`/`ErrorStatus`, graph type-state, ownership
  transfer, result dtype, and ordering stable unless explicitly changed.

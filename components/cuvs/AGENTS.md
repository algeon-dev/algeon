# AGENTS.md

Follow the root [AGENTS.md](../../AGENTS.md) for shared development and
validation rules.

## Project boundaries

`components/cuvs` is the runtime-bound Rust integration for the scoped cuVS
surface used by Algeon. SQL planning, runtime admission, and query execution
follow the [module boundaries](../../website/docs/design/index.mdx#crates-and-ownership).

- Safe algorithm, tensor, ownership, and completion APIs belong in
  `algeon-cuvs/`; raw generated C bindings and native discovery belong in
  `algeon-cuvs-sys/`.
- `cudf` is a root-workspace component dependency, and the interop contract
  types are imported through `cudf::interop::contract`.
  `components/cuvs/cuvs/` is the pinned Algeon cuVS fork used for
  native builds.
- The canonical install defaults are `target/native/cuvs-install` and the
  shared `target/native/cudf-install`; override them only with explicit
  selectors such as `CUVS_ROOT` and `CUVS_INSTALL_DIR`.

## Architecture contracts

- `CuvsResources` binds one exact `cudf::Runtime` device, CUDA stream, and
  process-lifetime current memory resource. Reject binding drift and arbitrary
  custom memory resources before calling cuVS.
- Runtime-bound resources, indexes, models, and prepared tensor leases remain
  `!Send` and `!Sync`. Do not weaken those markers without a demonstrated native
  concurrency contract.
- Prepared cuDF reads carry explicit producer ownership, completion witnesses,
  and a drop-time fallback for leases that are not completed. Complete reads
  only after cuVS work is enqueued; never release producer storage before the
  consumer fence permits it.
- Checked-in bindings come only from the pinned cuVS revision through
  `scripts/dev/bindings.sh cuvs`. Normal library builds, docs, and lints must
  not regenerate bindings or gain a bindgen build dependency.
- Native discovery accepts a canonical installed cuVS package, never a private
  `_deps` copy. Preserve the selected cuDF/RMM identity, shared CUDA runtime,
  source/version markers, and dynamic-loader checks.
- Validate native output ranges and floating-point validity before constructing
  or returning safe Rust values. A successful FFI status alone is not proof of
  a valid output.

## Build and test

```bash
cargo nt -E 'package(algeon-cuvs) | package(algeon-cuvs-sys)'

# Native-free workspace and checked-binding contracts
DOCS_RS=1 cargo check --workspace --all-targets --all-features --locked \
  --target-dir target/native-free-check
scripts/dev/bindings.sh cuvs --check
```

Native-free maintenance checks do not establish full native or GPU validation.

## Testing and repository policy

- `scripts/ci/check_native_sources.sh`, `scripts/ci/check_workspace_publication.py`,
  and `scripts/dev/bindings.sh` are the root owners for source, package, and
  checked-binding maintenance. Do not add a component-local umbrella command.
- `scripts/build/bootstrap_cuvs.sh` and `scripts/build/build_libcuvs.sh` are the
  root-owned cuVS operational scripts.
- `DOCS_RS` is the single native-free maintenance environment. Keep the sys
  crate usable for native-free linting, documentation, and package verification.
- Rebuild the canonical prefix and rerun binding plus ABI checks after changing
  the source pin, native build flags, headers, discovery, loader metadata, or
  generated bindings.
- `CUVS_ROOT` is an explicit build-time package selector, not a runtime loader
  configuration. Do not add ambient component, cuGraph build-tree, or private
  dependency fallbacks.

## Public contracts

- Preserve package/import identities: `algeon-cuvs` imports as `cuvs`,
  `algeon-cuvs-sys` imports as `cuvs_sys`, and the sys crate keeps
  `links = "cuvs_c"`.
- Preserve `CuvsResources` execution-binding identity and the ownership and
  completion semantics of `PreparedTensorLease` and
  `PreparedValidatedTensorLease`. Their `complete` returns the
  `algeon-interop` `LeaseCompletion`, which owns the retained-owner rule.
- Keep `ErrorKind`, `ErrorStatus`, operation, context, and source errors
  structured. Consumers must not need to parse display-message text.
- Preserve native metadata consumed through `DEP_CUVS_C_*`, especially runtime
  library directories. Downstream executables remain responsible for their own
  loader-visible deployment or RUNPATH.

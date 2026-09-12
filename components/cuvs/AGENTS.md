# AGENTS.md

## Working principles

- Fix the owning runtime, interop, FFI, or native-discovery contract instead of
  hiding a failure in an algorithm wrapper. Extend `cudf-nexus` or
  `rapids-interop` when the missing ownership or execution primitive belongs
  there.
- Keep changes scoped. Preserve unrelated worktree edits and avoid
  opportunistic cleanup, generated-file churn, or native rebuilds outside the
  task.

## Project boundaries

`components/cuvs` is the runtime-bound Rust integration for the scoped cuVS
surface used by DataFusion Nexus. SQL planning, runtime admission, and query
execution follow the root `datafusion-nexus` ownership map.

- Safe algorithm, tensor, ownership, and completion APIs belong in
  `cuvs-nexus/`; raw generated C bindings and native discovery belong in
  `cuvs-nexus-sys/`.
- `cudf-nexus` and `rapids-interop` are root-workspace component dependencies.
  The pinned official cuVS source is `components/cuvs/cuvs`.
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
  and incomplete-drop behavior. Complete reads only after cuVS work is enqueued;
  never release producer storage before the consumer fence permits it.
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
cargo clippy -p cuvs-nexus -p cuvs-nexus-sys \
  --all-targets --all-features -- -D warnings
cargo +nightly fmt --all --check
flock /tmp/cudf-gpu.lock cargo nextest run \
  -p cuvs-nexus -p cuvs-nexus-sys --all-features --no-fail-fast

# Native-free workspace and checked-binding contracts
DOCS_RS=1 cargo check --workspace --all-targets --all-features --locked \
  --target-dir target/native-free-check
scripts/dev/bindings.sh cuvs --check
```

Run focused checks first, then broaden validation with the change's blast
radius. Native-free maintenance checks do not establish full native or GPU
validation.

## Testing and repository policy

- `scripts/ci/check_native_sources.sh`, `scripts/ci/check_component_packages.sh`,
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

# cuVS component

`components/cuvs` is the Algeon Rust vector integration component
built around [`libcuvs`](https://github.com/rapidsai/cuvs). It is part of the
root Cargo workspace, not a second repository or workspace.
Its main direction is to make cuVS usable as part of backend services and query execution systems for
[Algeon](https://github.com/algeon-dev/algeon/tree/main/components/cuvs). It focuses on safe Rust APIs for GPU vector
search and clustering while keeping ownership, memory movement, execution context, synchronization, and interop
behavior explicit enough for service-side integration.

## Focus

- Safe, idiomatic Rust wrappers around `libcuvs` for brute-force k-NN, pairwise distance, vector Top-K, k-means, and PCA.
- Backend-service and query-engine integration, rather than just direct API bindings.
- Exact binding to a `cudf::Runtime` device, stream, and memory resource.
- Device-resident cuDF interop with explicit ownership and synchronization.
- Checked FFI, ABI, tensor layout, and algorithm-output contracts.

## Workspace

- `components/cuvs/algeon-cuvs/` - the safe public Rust API crate, published as `cuvs` (import `cuvs`).
- `components/cuvs/algeon-cuvs-sys/` - generated low-level bindings and native build discovery for `libcuvs` (import `cuvs_sys`).
- `components/cuvs/cuvs/` - the pinned [algeon-dev cuVS fork](https://github.com/algeon-dev/cuvs) source submodule used to build the native libraries.
- `scripts/build/bootstrap_cuvs.sh` and `scripts/build/build_libcuvs.sh` - root-owned native build entry points.
- `scripts/dev/bindings.sh cuvs` - root-owned checked-binding maintenance.

Native defaults are repo-relative: `target/native/cudf-install` supplies the
shared cuDF/RMM prefix and `target/native/cuvs-install` is the canonical cuVS
install. The root `.gitmodules`, `Cargo.lock`, and `scripts/check_all.sh` are
the only source, lockfile, and full-gate authorities.

## validation

```bash
cargo clippy -p cuvs -p cuvs-sys --all-targets --all-features -- -D warnings
cargo +nightly fmt --all --check
cargo nt -E 'package(cuvs) | package(cuvs-sys)'
```

Use `scripts/dev/bindings.sh cuvs --check` for checked bindings and
`DOCS_RS=1 cargo check --workspace --all-targets --all-features --locked
--target-dir target/native-free-check` for the native-free workspace contract.

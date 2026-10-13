# cuVS component

`algeon-cuvs` and `algeon-cuvs-sys` are published implementation
dependencies of Algeon. Applications use
[`algeon-datafusion`](https://docs.rs/algeon-datafusion); the APIs below serve
workspace component consumers. See the
[API documentation boundary](https://algeon.dev/design#api-documentation).

`components/cuvs` is the Algeon Rust vector integration component
built around [`libcuvs`](https://github.com/algeon-dev/cuvs), and it lives in
the root Cargo workspace. It provides safe Rust APIs for GPU vector search and
clustering with explicit ownership, memory movement, execution context,
synchronization, and cuDF interop contracts.

## Focus

- Safe Rust wrappers around `libcuvs` for brute-force k-NN, query-local IVF-Flat, IVF-PQ, and CAGRA search, pairwise distance, vector Top-K, k-means, and PCA.
- Binding to one `cudf::Runtime` device, stream, and memory resource.
- Device-resident cuDF interop with explicit ownership and synchronization.
- Checked FFI, ABI, tensor layout, and algorithm-output contracts.

## Workspace

- `components/cuvs/algeon-cuvs/` - the safe public Rust API crate, published as `algeon-cuvs` (import `cuvs`).
- `components/cuvs/algeon-cuvs-sys/` - generated low-level bindings and native build discovery for `libcuvs` (import `cuvs_sys`).
- `components/cuvs/cuvs/` - the pinned **[forked cuVS](https://github.com/algeon-dev/cuvs)** source submodule used to build `libcuvs`.
- `scripts/build/bootstrap_cuvs.sh` and `scripts/build/build_libcuvs.sh` - root-owned native build entry points.
- `scripts/dev/bindings.sh cuvs` - root-owned checked-binding maintenance.

Native defaults are repo-relative: `target/native/cudf-install` supplies the
shared cuDF/RMM prefix and `target/native/cuvs-install` is the canonical cuVS
install. The root `.gitmodules`, `Cargo.lock`, and `scripts/check_all.sh` are
the only source, lockfile, and full-gate authorities.

## Fork changes

The pinned fork is upstream `v26.08.01` plus the commits below on branch
`algeon-26.08`. Upstream `v26.10.00` contains none of them, so all five carry
forward; the stream pool getters and the Lanczos patch need rework against the
26.10 RAFT headers. Each is proposed for upstream so the fork can shrink.

Features (C API):

- Batched `select_k` with min or max selection, `cuvsSelectKBatched`. SQL
  top-k needs both directions; upstream `cuvsSelectK` selects smallest only.
- Stream pool accessors: `cuvsResourcesSetStreamPool`,
  `cuvsResourcesGetStreamPoolSize` and `cuvsResourcesGetStreamPoolStream`.
  `CuvsResources` attaches each pool stream to the query's stream
  reservation, so RAFT allocations on pool streams are charged to the query
  cap. Upstream `main` after 26.10 adds the setter only, so the fork keeps
  the two getters from 26.12 on.
- Typed out-of-memory errors, `cuvsGetLastErrorKind` and `cuvsErrorKind_t`.
  Device OOM maps to `ErrorKind::OutOfMemory` without parsing message text.

Bug fixes:

- L2 expanded distance: the self-pair clamp is bounded by `k * eps * norms`.
  The upstream clamp zeroed real distances up to 1e-3 between distinct rows
  with equal norms, which misranked near-unit-norm embeddings in brute-force
  kNN, pairwise distance and fused L2 kNN.
- RAFT Lanczos restart at an invariant-subspace breakdown, carried as
  `cpp/cmake/patches/raft-lanczos-invariant-subspace.diff` and applied by
  `get_raft.cmake` at configure time. Without it, spectral clustering through
  cuGraph returned arbitrary partitions or failed to converge on graphs with
  few distinct eigenvalues.

## Validation

```bash
cargo clippy -p cuvs -p cuvs-sys --all-targets --all-features -- -D warnings
cargo +nightly fmt --all --check
cargo nt -E 'package(cuvs) | package(cuvs-sys)'
```

Use `scripts/dev/bindings.sh cuvs --check` for checked bindings and
`DOCS_RS=1 cargo check --workspace --all-targets --all-features --locked
--target-dir target/native-free-check` for the native-free workspace contract.

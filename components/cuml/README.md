# cuML component

`algeon-cuml` and `algeon-cuml-sys` are published implementation
dependencies of Algeon. Applications use
[`algeon-datafusion`](https://docs.rs/algeon-datafusion); the APIs below serve
workspace component consumers. See the
[API documentation boundary](https://algeon.dev/design#api-documentation).

`components/cuml` provides Rust bindings to the pinned NVIDIA cuML `v26.08.00`
source submodule. The component contains a native C++ shim, checked-in C ABI
bindings, a safe Rust API, and integration tests. Normal Cargo builds do not run
bindgen.

## Packages and supported surface

- `algeon-cuml-sys` (`cuml_sys`) declares the C ABI and builds/links the native
  shim in `algeon-cuml-sys/native`.
- `algeon-cuml` (`cuml`) provides resource-bound APIs for HDBSCAN, PCA, TSVD,
  QN logistic regression, random-forest classification/regression, UMAP, and
  nvForest inference over imported XGBoost, LightGBM, and Treelite models.
  Fitted random forests predict through the same nvForest path.
  The Rust crate exposes only these algorithms, not the full cuML upstream API.
- `cuml/` is the root-registered upstream source submodule.

Native build configuration enables only HDBSCAN, PCA, TSVD, random forest,
logistic regression, and UMAP. `scripts/build/build_libcuml.sh` builds
`libcuml.so` against the canonical workspace cuDF/RMM and cuVS installs, then
builds and installs the Rust shim. The build checks the configured and linked
RMM/cuVS identities, loader dependencies, OpenMP imports, and production shim
contents. OpenMP is disabled for cuML and for the Treelite build that nvForest
and the shim load. RF parameters explicitly use `n_streams=1`; the RF Rust API
separately owns a RAFT stream pool for fit.

nvForest and Treelite come from cuML's CPM build and are always compiled from
their pinned sources. The shim instantiates nvForest's header-only importer, so
two build rules protect query-domain accounting: the shim's version script
keeps RMM's per-device resource map shared with the rest of the process, and
`scripts/build/patches/nvforest-196-move-buffers.patch` backports upstream
nvForest #196 so imported buffers are not copied on the legacy default stream.
Remove that backport when cuML moves to 26.10.

## Runtime and memory ownership

`CumlResources::for_runtime` binds a native handle to the exact cuDF runtime
device, CUDA stream, and current device-memory-manager RMM resource. Each native
call revalidates that binding and routes the manager's resource on the runtime
stream. The runtime and resource owner must outlive the handle. Runtime-bound
resources, buffers, models, and stream pools are not `Send` or `Sync`.

When RF fit runs under a query domain, pass its runtime-stream
`StreamReservationLease`; every native pool stream receives a sibling lease
with that domain's allocation limit. For standalone runtimes, pass `None` to
route the manager's backend resource to each pool stream without a query
domain. If the runtime stream has a query-domain reservation, the caller must
pass it rather than `None`; the API cannot infer the intended grant from the
stream. The pool synchronizes on success and failure. Rust removes routes or
member leases before destroying their native CUDA streams.

The shim catches native exceptions and keeps thread-local error detail,
including the native message, CUDA status, allocation failure source, and
reservation identity/request fields when available. Rust copies this detail
immediately after a failed call. Allocation attribution distinguishes physical
allocation failure from reservation-limit failure; unknown source remains
unknown rather than being inferred from status alone.

TSVD `fit` also computes a temporary `rows * n_components` score matrix to
populate explained variance and ratio; the pinned native metadata-only fit
entry point does not produce those statistics. Its allocation can exhaust a
query grant even when model buffers alone would fit.

## Verification

Run source and generated-binding checks, then build cuML after the canonical
cuDF and cuVS prefixes are available. `env.sh` is a local ignored file; if
used, set `CUML_INSTALL_DIR="$ALGEON_NATIVE_DIR/cuml-install"` and include
`$CUML_INSTALL_DIR/lib` in its `LD_LIBRARY_PATH`. The build script also has a
checkout-local default when the variable is absent.

```bash
bash scripts/ci/check_native_sources.sh cuml
bash scripts/dev/bindings.sh cuml --check
bash scripts/build/build_libcuml.sh
```

The focused Rust checks use the workspace-wide feature configuration:

```bash
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo nt -E 'package(algeon-cuml) | package(algeon-cuml-sys)'
```

The full pre-merge gate is `bash scripts/check_all.sh`. A command listed here is
not evidence that it passed; record validation results from the actual run.

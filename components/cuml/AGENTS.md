# cuML component guidance

Follow the root [AGENTS.md](../../AGENTS.md) for shared development and
validation rules.

## Boundaries and supported surface

- `algeon-cuml` owns safe algorithm and runtime-resource APIs. `algeon-cuml-sys`
  owns the C ABI, native shim, and native linking. `rapids-interop` owns shared
  runtime-binding vocabulary; cuDF owns shared memory and reservation
  contracts.
- The upstream source is `components/cuml/cuml`, pinned at `v26.08.00`.
  Supported Rust algorithms are HDBSCAN, PCA, TSVD, QN logistic regression,
  random-forest classification/regression, UMAP, and nvForest inference
  (`ForestInferenceModel`) over imported models. Do not imply that the
  broader upstream cuML surface is exposed.
- Root scripts own native source validation and build entry points. The
  `scripts/build/build_libcuml.sh` script configures a limited algorithm set,
  disables OpenMP, links against canonical cuDF/RMM and cuVS installs, and
  builds the C++ shim. Checked-in bindings come from the shim headers. Normal
  builds must not invoke bindgen.

## Runtime and memory contracts

- Bind each cuML native handle to one exact `cudf::Runtime` device, stream, and
  current device-memory-manager RMM resource. Validate drift before a native
  call and route the manager's resource on the runtime stream.
- Keep runtime-bound resources, buffers, models, and native-state owners
  `!Send` and `!Sync`. Retain borrowed owners and device allocations through
  native stream completion.
- Under a query domain, RF fit requires the runtime-stream reservation and a
  sibling reservation for every pool stream using that domain's allocation
  limit. For standalone runtimes without a domain, route the manager's backend
  resource to every pool stream. Keep routes or member leases through fit and
  remove them before destroying pool streams; native fit synchronizes the pool
  on both success and failure. `n_streams=1` does not remove the native pool.
- Preserve native error messages and structured error details. Allocation
  attribution must use the captured failure source and reservation identity,
  not guess from a generic out-of-memory status.
- Validate native outputs before exposing safe Rust values. Cleanup across the
  C ABI must catch exceptions and return a status.

## Validation

```bash
bash scripts/ci/check_native_sources.sh cuml
bash scripts/dev/bindings.sh cuml --check
bash scripts/build/build_libcuml.sh
cargo nt -E 'package(algeon-cuml) | package(algeon-cuml-sys)'
```

The native build requires canonical cuDF/RMM and cuVS install prefixes. Select
the relevant checks under the root validation policy.

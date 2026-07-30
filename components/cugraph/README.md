# cugraph-rs

`cugraph-rs` is a Rust GPU graph layer built on NVIDIA [cuGraph](https://github.com/DataFusion-Nexus/cugraph).

This repository is not meant to be only a thin set of `libcugraph` bindings. The
raw C API bindings exist in `cugraph-sys`, but the main goal is to provide the
service-facing pieces needed by backend systems, including DataFusion-Nexus:

- safe Rust ownership around cuGraph resources and GPU execution context
- cudf-backed graph ingestion and result conversion
- graph algorithm APIs that can be embedded in query and service runtimes
- structured errors that preserve machine-readable failure context

## Workspace

- `cugraph-sys` - raw bindgen output for the cuGraph C API
- `cugraph-rs` - safe Rust crate published as `cugraph`
- `cugraph/` - upstream NVIDIA cuGraph source submodule used for local native builds

## Scope

This project focuses on backend integration rather than wrapping every cuGraph
surface immediately. Work that clarifies resource ownership, cudf interop,
algorithm contracts, error semantics, or DataFusion-Nexus integration belongs
here.

This project is not a standalone graph database, a replacement for Python
cuGraph, or a way to hide the native CUDA/RAPIDS deployment requirements.

## Native cuGraph on Blackwell (sm_120 / RTX 50-series): known SCC failure

`cugraph-sys` links against a pre-built cuGraph CMake build dir (default:
`cugraph/cpp/build`). On an sm_120 GPU, a **clean locally source-built** cuGraph
v26.06.00 without the local fallback fails inside
`cugraph_strongly_connected_components`:

```
cugraph_strongly_connected_components failed:
  copy_if failed on 2nd step: cudaErrorInvalidDeviceFunction: invalid device function
```

The failing kernel is CUB `DeviceSelect` (`thrust::copy_if`/`remove_if`) inside
`intersect_reachable_sets`. CCCL 3.4.0's CUB select tuning chain tops out at
`Policy1000` (sm_100), with no policy for CC 12.0.

### What we verified (and what we did NOT)

- **Official prebuilt works.** `rapidsai/base:26.06-cuda13` runs SCC correctly on
  this sm_120 card (Python `cugraph.strongly_connected_components`, which calls
  the same `cugraph_strongly_connected_components` C API).
- **Clean local source builds fail — regardless of arch list.** Both `NATIVE`
  (`sm_120a` only) and the full RAPIDS list (`sm_75;80;86;90a;100f;120a;120` +
  PTX) fail with the identical error. **So the CUDA arch list is NOT the fix on
  its own.**
- Same CUDA 13.2.78 and same CCCL 3.4.0 (no rapids-cmake patch), with equivalent
  `sm_120` code-object coverage. The sharpest split we found is artifact-level:
  official `libcugraph.so` passes, while clean local-built `libcugraph.so` fails.
  The exact build/root-cause difference is still open.
- Official CI never caught this: its test GPU matrix tops out at H100 / sm_90,
  with no Blackwell.

### Recommendation

For non-forked workloads, linking against the official prebuilt cuGraph
libraries is a valid workaround. This fork needs the local BFS predicate
extension, so it carries a narrow native fallback: on single-GPU sm_120 only,
`cugraph_strongly_connected_components` routes through a host-side SCC fallback
and keeps the normal C API result shape.

Tracked upstream as [cugraph#5543](https://github.com/rapidsai/cugraph/issues/5543).

## Status

This project is in early development. API stability is less important than
getting the service-facing ownership, execution, memory, and interop contracts
right. Contributions are welcome. Please start every PR with an issue that explains
why the change is needed, and include the test environment and exact checks you
ran in the PR description. See [CONTRIBUTING.md](CONTRIBUTING.md) for the
short contribution policy.

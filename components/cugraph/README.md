# cuGraph component

`algeon-cugraph` and `algeon-cugraph-sys` are published implementation
dependencies of Algeon. Applications use
[`algeon-datafusion`](https://docs.rs/algeon-datafusion); the APIs below serve
workspace component consumers. See the
[API documentation boundary](https://algeon.dev/design#api-documentation).

`components/cugraph` is the Algeon Rust graph integration component
built around [`libcugraph`](https://github.com/algeon-dev/cugraph). It is
part of the root Cargo workspace, not a second repository or workspace.
It exists only to support the native backend of
[Algeon](https://github.com/algeon-dev/algeon/tree/main/components/cugraph).

## Focus

- Runtime-bound graph ownership and algorithm execution.
- Canonical prepared cuDF ingress and result egress with completion ordering.
- Allocation preflight facts for bounded Algeon admission.
- Structured errors that preserve machine-readable native failure context.

## Workspace

- `components/cugraph/algeon-cugraph/` - the internal safe execution crate, packaged as `algeon-cugraph` (import `cugraph`).
- `components/cugraph/algeon-cugraph-sys/` - low-level FFI bindings around the cuGraph C API (import `cugraph_sys`).
- `components/cugraph/cugraph/` - the pinned **[forked cuGraph](https://github.com/algeon-dev/cugraph)** source submodule used to build `libcugraph`.

Native defaults are repo-relative: `target/native/cudf-install` and
`target/native/cuvs-install` provide the shared cuDF/RMM and cuVS prefixes, and
`target/native/cugraph-build` is the cuGraph CMake build tree. Use the root
`scripts/build/build_libcugraph.sh` helper and root `scripts/check_all.sh` gate.

## Fork changes

The pinned fork is upstream `v26.08.00` plus the changes below on branch
`algeon-26.08`. Upstream `v26.10.00` supersedes none of them; every change
cherry-picks onto 26.10 without conflicts, and only the fork's BFS test needs
rework for the RMM 26.10 `device_scalar` constructor. The generic bug fixes
are proposed upstream so the fork can shrink.

Features:

- Predicate-aware single-GPU BFS, `cugraph_bfs_with_predicates` and
  `bfs_predicate_result_t`, with vertex include and exclude lists, edge-ID
  filters lowered on the device, and target early exit. Backs SQL
  `cugraph_bfs`.
- Error contract for the C API: typed allocation source
  (`cugraph_error_allocation_source`), query-domain attribution of allocation
  failures (`cugraph_error_failure_source`, `cugraph_error_domain_id`,
  `cugraph_error_requested_bytes` and related accessors), exception-safe
  result construction, and precondition or out-of-range vertex failures
  reported as `CUGRAPH_INVALID_INPUT` with a reason. The Rust `ErrorKind`
  needs no message-text parsing.
- Effective device memory provider, `cugraph::detail::effective_device_memory`.
  Internal temporary-buffer sizing reads the query budget supplied by the
  cuDF memory-resource owner instead of `totalGlobalMem`.
- Non-convergence reporting for eigenvector, Katz and HITS centrality
  (`*_allow_nonconvergence`, `cugraph_hits_result_converged`), matching the
  existing PageRank option.

Bug fixes:

- SSSP on an unweighted graph uses unit weights instead of dereferencing a
  null weight property.
- Cosine similarity normalizes over full neighborhoods and guards a zero
  denominator.
- Multisource betweenness stays exact under source batching and accumulates
  deterministically.
- Leiden returns only connected communities and the modularity of the
  returned partition.
- BFS `depth_limit` counts hops; oversized limits are clamped instead of
  wrapping.
- `cugraph_k_core` reads the caller's `degree_type`.
- Strongly connected components rejects symmetric graphs as invalid input.
- Weighted graph construction uses less peak device memory.

Build:

- `build.sh` no longer passes an empty argument to `cmake --build`.
- CMake discovers the cuDF `cudf_rust/error_boundary.hpp` header from the
  RMM prefix and registers the fork's contract tests.

## Validation

```bash
cargo clippy -p cugraph -p cugraph-sys --all-targets --all-features -- -D warnings
cargo +nightly fmt --all --check
cargo nt -E 'package(cugraph) | package(cugraph-sys)'
```

`cugraph-sys` intentionally generates bindings from the selected native
headers in `build.rs`; the selected headers and include paths are part of its
package/native contract.

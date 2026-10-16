---
unlisted: true
title: Fork changes
description: What Algeon's pinned cuDF, cuGraph and cuVS forks add to upstream 26.08, and which of those changes upstream 26.10 already contains.
---

# Fork changes

Algeon pins cuDF, cuGraph and cuVS as Git submodules of its own forks. The
generic bug fixes are proposed upstream so each fork can shrink, and each
section records which changes upstream 26.10 already contains. cuML has no
fork. Its submodule pins upstream `v26.08.00`, and the native build applies
`scripts/build/patches/nvforest-196-move-buffers.patch` to the nvForest sources
that cuML's CPM build fetches.

## cuDF

The pinned fork is upstream `v26.08.00` plus the changes below on branch
`algeon-26.08`. Upstream `v26.10.00` supersedes the memory-resource parameters
on the device-view factories, row preprocessing, `sizes_to_offsets` and gather
temporaries, and the pinned-pool stream change; every other change is still
absent upstream and is re-applied on the 26.10 HashCSR hash join, per-thread
stream pool and reworked Parquet pruning code. Five preflight commits that the
fork later reverted are not carried.

Features:

- Retained join state with exact output sizing and delayed materialization:
  `hash_join::pre_build_reservation_size`, `filtered_join` pre-build sizing
  with `begin_left_semi_probe` and `begin_left_anti_probe`,
  `conditional_full_join_size`, `mixed_left_semi_join_size`,
  `mixed_left_anti_join_size`, `mixed_full_join_size`, and
  `mixed_left_join_state` for a chunked right side. Admission reserves the
  exact bytes before a build and materializes at most the admitted size.
- An explicit memory resource reaches join temporaries, gather, lexicographic
  sort scratch and row-operator preprocessing; their scratch allocations are
  charged to the query reservation.
- Prepared regex backreference replacement,
  `replace_with_backrefs_output_size` and `replace_with_backrefs_prepared`:
  size first, then execute from retained state.
- Stream routing: `cudf::detail::install_stream_provider` lets libcudf's
  internal `fork_streams` use attempt-owned streams, and the device-buffer
  datasource stops using a global pool stream.
- Worker future drain guard (`drain_futures`, `future_drain_guard`) in the
  Parquet, CSV and compression paths. A failing worker leaves no future
  unjoined.
- Remote Parquet byte ranges are coalesced across gaps up to
  `LIBCUDF_PARQUET_REMOTE_COALESCE_GAP_BYTES` (64 KiB by default) to reduce
  the number of remote reads.

Bug fixes:

- Parquet predicate pushdown: a bare boolean column or literal used as the
  reader filter no longer crashes, the stats survivor count is reported only
  when the stats stage ran, the decimal literal type check honors scale, and
  decimal fixed-length dictionary values are decoded big-endian.
- cuco exception safety, applied as a build-time patch: temporary storage is
  neither leaked nor double-freed when an allocation throws.
- A GCC 14.2 internal compiler error in `make_datasources` is avoided by
  naming the lambda parameter type.

Build:

- `CUDF_RMM_BUILD_SHARED` builds and installs RMM as a shared library and
  installs nvtx3 and rapids_logger; cuDF, cuGraph, cuVS and cuML then bind
  one `librmm.so` in a process.

## cuGraph

The pinned fork is upstream `v26.08.00` plus the changes below on branch
`algeon-26.08`. Upstream `v26.10.00` supersedes none of them; every change
cherry-picks onto 26.10 without conflicts, and only the fork's BFS test needs
rework for the RMM 26.10 `device_scalar` constructor.

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

## cuVS

The pinned fork is upstream `v26.08.01` plus the commits below on branch
`algeon-26.08`. Upstream `v26.10.00` contains none of them. All five carry
forward; the stream pool getters and the Lanczos patch need rework against the
26.10 RAFT headers.

Features (C API):

- Batched `select_k` with min or max selection, `cuvsSelectKBatched`. SQL
  top-k needs both directions; upstream `cuvsSelectK` selects smallest only.
- Stream pool accessors: `cuvsResourcesSetStreamPool`,
  `cuvsResourcesGetStreamPoolSize` and `cuvsResourcesGetStreamPoolStream`.
  `CuvsResources` attaches each pool stream to the query's stream
  reservation; RAFT allocations on pool streams are then charged to the query
  cap. Upstream `main` after 26.10 adds the setter only; the fork keeps the
  two getters from 26.12 on.
- Typed out-of-memory errors, `cuvsGetLastErrorKind` and `cuvsErrorKind_t`.
  Device OOM maps to `ErrorKind::OutOfMemory` without parsing message text.

Bug fixes:

- L2 expanded distance: the self-pair clamp is bounded by `k * eps * norms`.
  The upstream clamp zeroed distances up to 1e-3 between distinct rows with
  equal norms and misranked near-unit-norm embeddings in brute-force kNN,
  pairwise distance and fused L2 kNN.
- RAFT Lanczos restart at an invariant-subspace breakdown, carried as
  `cpp/cmake/patches/raft-lanczos-invariant-subspace.diff` and applied by
  `get_raft.cmake` at configure time. Before the patch, spectral clustering
  through cuGraph returned arbitrary partitions or failed to converge on
  graphs with few distinct eigenvalues.

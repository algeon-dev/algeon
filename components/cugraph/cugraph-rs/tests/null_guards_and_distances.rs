//! Tests for Finding 2 (null-pointer guards) and Finding 3 (Distances enum).
//!
//! Finding 2: All FFI result accessors must return `Result` errors rather than
//! wrapping null pointers in `RawDeviceArrayView` (latent UB). We verify the
//! existing happy-path still works and that the `Distances` type is usable.
//!
//! Finding 3: `BfsResult::distances()` returns a `Distances` enum (not
//! `VertexIds`) to correctly model integer hop counts (BFS) and future float
//! distances (SSSP).

use cugraph::{
    BfsParams, DeviceArray, Distances, Forward, GraphBuilder, PageRankParams, ResourceHandle, ScoreValues, Transposed,
    VertexIds,
};

// ═══════════════════════════════════════════════════════════════════════════
// Finding 3 — Distances enum
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn bfs_distances_returns_distances_enum_not_vertex_ids() {
    // BFS distances must return the `Distances` enum, not `VertexIds`.
    let handle = ResourceHandle::new().expect("handle");
    let src = DeviceArray::from_slice(&handle, &[0i32, 1, 2]).expect("src");
    let dst = DeviceArray::from_slice(&handle, &[1i32, 2, 3]).expect("dst");

    let mut graph = GraphBuilder::new(&src.view(), &dst.view()).build::<Forward>(&handle).expect("graph");
    let sources = DeviceArray::from_slice(&handle, &[0i32]).expect("sources");
    let result = graph.bfs(&handle, &sources.view(), &BfsParams::default()).expect("bfs");

    let distances = result.distances().expect("distances");

    // BFS on an i32 graph should produce I32 distances (integer hop counts)
    assert!(matches!(distances, Distances::I32(_)), "BFS i32 graph should produce Distances::I32");
    assert_eq!(distances.len(), 4);
    assert!(!distances.is_empty());
}

#[test]
fn bfs_distances_i32_to_vec_i64_widens() {
    let handle = ResourceHandle::new().expect("handle");
    let src = DeviceArray::from_slice(&handle, &[0i32, 1]).expect("src");
    let dst = DeviceArray::from_slice(&handle, &[1i32, 2]).expect("dst");

    let mut graph = GraphBuilder::new(&src.view(), &dst.view()).build::<Forward>(&handle).expect("graph");
    let sources = DeviceArray::from_slice(&handle, &[0i32]).expect("sources");
    let result = graph.bfs(&handle, &sources.view(), &BfsParams::default()).expect("bfs");

    let distances = result.distances().expect("distances");
    // Widening to i64 should always work
    let as_i64 = distances.to_vec_i64(&handle).expect("to_vec_i64");
    assert_eq!(as_i64.len(), 3);
}

#[test]
fn bfs_distances_i32_to_vec_i32_succeeds() {
    let handle = ResourceHandle::new().expect("handle");
    let src = DeviceArray::from_slice(&handle, &[0i32, 1]).expect("src");
    let dst = DeviceArray::from_slice(&handle, &[1i32, 2]).expect("dst");

    let mut graph = GraphBuilder::new(&src.view(), &dst.view()).build::<Forward>(&handle).expect("graph");
    let sources = DeviceArray::from_slice(&handle, &[0i32]).expect("sources");
    let result = graph.bfs(&handle, &sources.view(), &BfsParams::default()).expect("bfs");

    let distances = result.distances().expect("distances");
    let as_i32 = distances.to_vec_i32(&handle).expect("to_vec_i32");
    assert_eq!(as_i32.len(), 3);
    // Source vertex distance should be 0
    assert!(as_i32.contains(&0), "source distance should be 0");
}

#[test]
fn bfs_distances_i64_graph_returns_i64() {
    let handle = ResourceHandle::new().expect("handle");
    let src = DeviceArray::from_slice(&handle, &[0i64, 1, 2]).expect("src");
    let dst = DeviceArray::from_slice(&handle, &[1i64, 2, 3]).expect("dst");

    let mut graph = GraphBuilder::new(&src.view(), &dst.view()).renumber().build::<Forward>(&handle).expect("graph");
    let sources = DeviceArray::from_slice(&handle, &[0i64]).expect("sources");
    let result = graph.bfs(&handle, &sources.view(), &BfsParams::default()).expect("bfs");

    let distances = result.distances().expect("distances");
    assert!(matches!(distances, Distances::I64(_)), "BFS i64 graph should produce Distances::I64");
}

#[test]
fn distances_to_vec_f64_succeeds_for_integer_distances() {
    // Distances::to_vec_f64 should widen integer values to f64
    let handle = ResourceHandle::new().expect("handle");
    let src = DeviceArray::from_slice(&handle, &[0i32, 1]).expect("src");
    let dst = DeviceArray::from_slice(&handle, &[1i32, 2]).expect("dst");

    let mut graph = GraphBuilder::new(&src.view(), &dst.view()).build::<Forward>(&handle).expect("graph");
    let sources = DeviceArray::from_slice(&handle, &[0i32]).expect("sources");
    let result = graph.bfs(&handle, &sources.view(), &BfsParams::default()).expect("bfs");

    let distances = result.distances().expect("distances");
    let as_f64 = distances.to_vec_f64(&handle).expect("to_vec_f64");
    assert_eq!(as_f64.len(), 3);
    assert!(as_f64.contains(&0.0), "source distance should be 0.0");
}

// ═══════════════════════════════════════════════════════════════════════════
// Finding 2 — Null-pointer guards (happy-path regression tests)
// ═══════════════════════════════════════════════════════════════════════════
//
// We cannot easily inject null pointers from the C API in integration tests.
// Instead we verify that all result accessors return `Result` (not raw values)
// and that the happy-path still works without panic.

#[test]
fn centrality_result_accessors_return_result() {
    // PageRank accessors should return Result, not panic
    let handle = ResourceHandle::new().expect("handle");
    let src = DeviceArray::from_slice(&handle, &[1i32, 2, 3, 0, 0, 0]).expect("src");
    let dst = DeviceArray::from_slice(&handle, &[0i32, 0, 0, 1, 2, 3]).expect("dst");

    let mut graph = GraphBuilder::new(&src.view(), &dst.view()).renumber().build::<Transposed>(&handle).expect("graph");

    let result = graph.pagerank_simple(&handle, &PageRankParams::default()).expect("pagerank");

    // These calls exercise the null-guard code paths (happy path — should succeed)
    let vids = result.vertex_ids().expect("vertex_ids should not be null");
    let scores = result.scores().expect("scores should not be null");
    assert!(matches!(vids, VertexIds::I32(_)));
    assert!(matches!(scores, ScoreValues::F32(_)));
    assert_eq!(vids.len(), 4);
    assert_eq!(scores.len(), 4);
}

#[test]
fn bfs_result_accessors_return_result() {
    // BFS vertex_ids and distances should return Result, not panic
    let handle = ResourceHandle::new().expect("handle");
    let src = DeviceArray::from_slice(&handle, &[0i32, 1, 2]).expect("src");
    let dst = DeviceArray::from_slice(&handle, &[1i32, 2, 3]).expect("dst");

    let mut graph = GraphBuilder::new(&src.view(), &dst.view()).build::<Forward>(&handle).expect("graph");
    let sources = DeviceArray::from_slice(&handle, &[0i32]).expect("sources");
    let result = graph.bfs(&handle, &sources.view(), &BfsParams::default().with_predecessors()).expect("bfs");

    let vids = result.vertex_ids().expect("vertex_ids should not be null");
    let dists = result.distances().expect("distances should not be null");
    let preds = result.predecessors().expect("predecessors result").expect("predecessors should exist");

    assert_eq!(vids.len(), 4);
    assert_eq!(dists.len(), 4);
    assert_eq!(preds.len(), 4);
}

#[test]
fn layout_result_accessors_return_result() {
    // ForceAtlas2 vertex_ids, x, y should return Result, not panic
    use cugraph::{ForceAtlas2Options, ForceAtlas2Params, IterationCount, RngState, types::GraphProperties};

    let handle = ResourceHandle::new().expect("handle");
    let mut rng = RngState::new(&handle, 42).expect("rng");

    let src = DeviceArray::from_slice(&handle, &[0i32, 1, 2, 1, 2, 0]).expect("src");
    let dst = DeviceArray::from_slice(&handle, &[1i32, 2, 0, 0, 1, 2]).expect("dst");

    let mut graph = GraphBuilder::new(&src.view(), &dst.view())
        .properties(GraphProperties::undirected())
        .renumber()
        .build::<Forward>(&handle)
        .expect("graph");

    let options: ForceAtlas2Options<'_> = ForceAtlas2Options {
        params: ForceAtlas2Params { max_iter: IterationCount::new(10).unwrap(), ..Default::default() },
        ..Default::default()
    };
    let result = graph.force_atlas2(&handle, &mut rng, &options).expect("fa2");

    let vids = result.vertex_ids().expect("vertex_ids should not be null");
    let x = result.x().expect("x should not be null");
    let y = result.y().expect("y should not be null");

    assert_eq!(vids.len(), 3);
    assert_eq!(x.len(), 3);
    assert_eq!(y.len(), 3);
}

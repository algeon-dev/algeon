//! Integration tests for the cudf interop API surface:
//!
//! - `device_array_view_from_slice()` — checked zero-copy DeviceSlice → DeviceArrayView
//! - `graph_from_series()` — zero-copy graph construction from cudf Series
//! - `IntoSeries` trait — result arrays → cudf Series conversion
//! - Full round-trip: cudf DataFrame → Graph → algorithm → cudf DataFrame

use std::collections::HashMap;

use cugraph::{
    array::DeviceArray,
    centrality::PageRankParams,
    cudf_interop::{
        DataFrameGraphOptions, IntoSeries, SeriesGraphOptions, device_array_view_from_slice, graph_from_dataframe,
        graph_from_series,
    },
    error::ErrorKind,
    graph::{Forward, GraphBuilder, Transposed},
    handle::ResourceHandle,
    traversal::BfsParams,
    types::GraphProperties,
};

// ═══════════════════════════════════════════════════════════════════════════
// device_array_view_from_slice — checked zero-copy bridge
// ═══════════════════════════════════════════════════════════════════════════

/// Verify that `device_array_view_from_slice::<i32>` produces a view with the
/// correct length and a non-null device pointer when the slice dtype matches.
#[test]
fn device_array_view_from_slice_i32() {
    let series = cudf::Series::new("v", &[10i32, 20, 30]).expect("series");
    let col_view = series.as_column_view().expect("column view");
    let slice = col_view.device_slice::<i32>().expect("device_slice");

    assert!(!slice.as_ptr().is_null(), "device pointer should be non-null");

    let view = device_array_view_from_slice::<i32>(&slice).expect("type should match");
    assert_eq!(view.len(), 3);
}

/// Same test with `i64` vertex dtype.
#[test]
fn device_array_view_from_slice_i64() {
    let series = cudf::Series::new("v", &[100i64, 200, 300]).expect("series");
    let col_view = series.as_column_view().expect("column view");
    let slice = col_view.device_slice::<i64>().expect("device_slice");

    let view = device_array_view_from_slice::<i64>(&slice).expect("type should match");
    assert_eq!(view.len(), 3);
}

/// Verify that a dtype mismatch is rejected: an i32 slice cannot be viewed as f32.
#[test]
fn device_array_view_from_slice_rejects_mismatch() {
    let series = cudf::Series::new("v", &[1i32, 2, 3]).expect("series");
    let col_view = series.as_column_view().expect("column view");
    let slice = col_view.device_slice::<i32>().expect("device_slice");

    let err = device_array_view_from_slice::<f32>(&slice).expect_err("should reject i32 → f32 mismatch");
    assert_eq!(err.kind(), cugraph::ErrorKind::UnsupportedType);
}

// ═══════════════════════════════════════════════════════════════════════════
// graph_from_series
// ═══════════════════════════════════════════════════════════════════════════

/// Build a directed graph from i32 Series, run BFS from vertex 0, verify
/// expected distances match the known graph topology.
#[test]
fn graph_from_series_i32_bfs() {
    let handle = ResourceHandle::new().expect("handle");

    // Triangle + tail: 0->1, 1->2, 2->0, 2->3
    let src = cudf::Series::new("src", &[0i32, 1, 2, 2]).expect("src");
    let dst = cudf::Series::new("dst", &[1i32, 2, 0, 3]).expect("dst");

    let opts = SeriesGraphOptions::default().renumber();
    let mut graph = graph_from_series::<Forward>(&handle, &src, &dst, None, &opts).expect("graph_from_series");

    let sources = DeviceArray::from_slice(&handle, &[0i32]).expect("sources");
    let result = graph.bfs(&handle, &sources.view(), &BfsParams::default()).expect("bfs");

    let vertices: Vec<i32> = result.vertex_ids().expect("vids").to_vec_i32(&handle).expect("copy");
    let distances: Vec<i32> = result.distances().expect("dists").to_vec_i32(&handle).expect("copy");

    let dist_map: HashMap<i32, i32> = vertices.into_iter().zip(distances).collect();
    assert_eq!(dist_map[&0], 0, "source distance");
    assert_eq!(dist_map[&1], 1, "vertex 1");
    assert_eq!(dist_map[&2], 2, "vertex 2");
    assert_eq!(dist_map[&3], 3, "vertex 3");
}

/// Build a directed graph from i64 Series, run BFS, verify correct results.
#[test]
fn graph_from_series_i64_bfs() {
    let handle = ResourceHandle::new().expect("handle");

    // Simple chain: 0->1->2
    let src = cudf::Series::new("src", &[0i64, 1]).expect("src");
    let dst = cudf::Series::new("dst", &[1i64, 2]).expect("dst");

    let opts = SeriesGraphOptions::default().renumber();
    let mut graph = graph_from_series::<Forward>(&handle, &src, &dst, None, &opts).expect("graph_from_series");

    let sources = DeviceArray::from_slice(&handle, &[0i64]).expect("sources");
    let result = graph.bfs(&handle, &sources.view(), &BfsParams::default()).expect("bfs");

    let vertices: Vec<i64> = result.vertex_ids().expect("vids").to_vec_i64(&handle).expect("copy");
    let distances: Vec<i64> = result.distances().expect("dists").to_vec_i64(&handle).expect("copy");

    let dist_map: HashMap<i64, i64> = vertices.into_iter().zip(distances).collect();
    assert_eq!(dist_map[&0], 0, "source distance");
    assert_eq!(dist_map[&1], 1, "vertex 1");
    assert_eq!(dist_map[&2], 2, "vertex 2");
}

/// Build a weighted graph from Series, run PageRank, verify scores are positive.
#[test]
fn graph_from_series_weighted_pagerank() {
    let handle = ResourceHandle::new().expect("handle");

    // Undirected triangle with weights
    let src = cudf::Series::new("src", &[0i32, 1, 2, 1, 2, 0]).expect("src");
    let dst = cudf::Series::new("dst", &[1i32, 2, 0, 0, 1, 2]).expect("dst");
    let wgt = cudf::Series::new("wgt", &[1.0f32, 2.0, 3.0, 1.0, 2.0, 3.0]).expect("wgt");

    let opts = SeriesGraphOptions::default().properties(GraphProperties::undirected()).renumber();

    let mut graph = graph_from_series::<Transposed>(&handle, &src, &dst, Some(&wgt), &opts).expect("graph_from_series");

    let result = graph.pagerank_simple(&handle, &PageRankParams::default()).expect("pagerank");

    let scores: Vec<f32> = result.scores().expect("scores").to_vec_f32(&handle).expect("copy");
    assert_eq!(scores.len(), 3, "should have 3 vertices");
    for (i, &s) in scores.iter().enumerate() {
        assert!(s > 0.0 && s.is_finite(), "score[{}] = {} should be positive finite", i, s);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// IntoSeries trait
// ═══════════════════════════════════════════════════════════════════════════

/// Convert BFS VertexIds and Distances to cudf Series, verify names and lengths.
#[test]
fn into_series_bfs_results() {
    let handle = ResourceHandle::new().expect("handle");

    let src = DeviceArray::from_slice(&handle, &[0i32, 1]).expect("src");
    let dst = DeviceArray::from_slice(&handle, &[1i32, 2]).expect("dst");

    let mut graph = GraphBuilder::new(&src.view(), &dst.view()).renumber().build::<Forward>(&handle).expect("graph");

    let sources = DeviceArray::from_slice(&handle, &[0i32]).expect("sources");
    let result = graph.bfs(&handle, &sources.view(), &BfsParams::default()).expect("bfs");

    let vids = result.vertex_ids().expect("vids");
    let dists = result.distances().expect("dists");

    // Convert to Series
    let v_series = vids.to_series("my_vertex").expect("VertexIds::to_series");
    let d_series = dists.to_series("my_distance").expect("Distances::to_series");

    assert_eq!(v_series.name(), Some("my_vertex"));
    assert_eq!(d_series.name(), Some("my_distance"));

    let v_host: Vec<i32> = v_series.view().expect("view").to_vec().expect("to_vec");
    let d_host: Vec<i32> = d_series.view().expect("view").to_vec().expect("to_vec");

    assert_eq!(v_host.len(), 3, "3 vertices in 0->1->2");
    assert_eq!(d_host.len(), 3);
}

/// Convert PageRank ScoreValues to cudf Series, verify values are positive.
#[test]
fn into_series_pagerank_scores() {
    let handle = ResourceHandle::new().expect("handle");

    let src = DeviceArray::from_slice(&handle, &[0i32, 1, 2, 1, 2, 0]).expect("src");
    let dst = DeviceArray::from_slice(&handle, &[1i32, 2, 0, 0, 1, 2]).expect("dst");

    let mut graph = GraphBuilder::new(&src.view(), &dst.view())
        .properties(GraphProperties::undirected())
        .renumber()
        .build::<Transposed>(&handle)
        .expect("graph");

    let result = graph.pagerank_simple(&handle, &PageRankParams::default()).expect("pagerank");
    let scores = result.scores().expect("scores");

    let series = scores.to_series("pr_score").expect("ScoreValues::to_series");
    assert_eq!(series.name(), Some("pr_score"));

    let host: Vec<f32> = series.view().expect("view").to_vec().expect("to_vec");
    assert_eq!(host.len(), 3);
    for &v in &host {
        assert!(v > 0.0_f32 && v.is_finite());
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Full round-trip: DataFrame → Graph → algorithm → DataFrame
// ═══════════════════════════════════════════════════════════════════════════

/// Build a graph from a cudf DataFrame using the zero-copy path
/// (`graph_from_dataframe` now reads Series views directly), run BFS,
/// convert the result back to a DataFrame, and verify values.
#[test]
fn roundtrip_dataframe_to_graph_to_dataframe() {
    let handle = ResourceHandle::new().expect("handle");

    // Build input DataFrame
    let src = cudf::Column::from_slice(&[0i32, 1, 1, 2, 2, 2, 3, 4]).expect("src");
    let dst = cudf::Column::from_slice(&[1i32, 3, 4, 0, 1, 3, 5, 5]).expect("dst");
    let wgt = cudf::Column::from_slice(&[0.1f32, 2.1, 1.1, 5.1, 3.1, 4.1, 7.2, 3.2]).expect("wgt");

    let df = cudf::DataFrame::new(vec!["src".into(), "dst".into(), "weight".into()], vec![src, dst, wgt])
        .expect("dataframe");

    // Build graph via zero-copy path
    let opts = DataFrameGraphOptions::new("src", "dst")
        .weight_col("weight")
        .properties(GraphProperties::directed())
        .renumber();
    let mut graph = graph_from_dataframe::<Forward>(&handle, &df, &opts).expect("graph_from_dataframe");

    // Run BFS with predecessors
    let sources = DeviceArray::from_slice(&handle, &[0i32]).expect("sources");
    let params = BfsParams::default().with_predecessors();
    let result = graph.bfs(&handle, &sources.view(), &params).expect("bfs");

    // Convert result to DataFrame
    let result_df = result.to_dataframe().expect("to_dataframe");

    // Verify schema
    assert_eq!(result_df.num_columns(), 3, "vertex + distance + predecessor");
    let names = result_df.column_names();
    assert_eq!(names[0], "vertex");
    assert_eq!(names[1], "distance");
    assert_eq!(names[2], "predecessor");

    // Verify BFS distances
    let vertices: Vec<i32> = result_df.series_view("vertex").expect("vertex").to_vec().expect("v");
    let distances: Vec<i32> = result_df.series_view("distance").expect("distance").to_vec().expect("d");

    let dist_map: HashMap<i32, i32> = vertices.into_iter().zip(distances).collect();
    assert_eq!(dist_map[&0], 0, "source vertex 0");
    assert_eq!(dist_map[&1], 1, "vertex 1");
    assert_eq!(dist_map[&2], i32::MAX, "vertex 2 unreachable from 0");
    assert_eq!(dist_map[&3], 2, "vertex 3");
    assert_eq!(dist_map[&4], 2, "vertex 4");
    assert_eq!(dist_map[&5], 3, "vertex 5");
}

/// Build a graph from cudf Series, run PageRank, convert result to DataFrame,
/// verify the full pipeline works end-to-end.
#[test]
fn roundtrip_series_to_graph_to_dataframe() {
    let handle = ResourceHandle::new().expect("handle");

    let src = cudf::Series::new("src", &[0i32, 1, 2, 1, 2, 0]).expect("src");
    let dst = cudf::Series::new("dst", &[1i32, 2, 0, 0, 1, 2]).expect("dst");

    let opts = SeriesGraphOptions::default().properties(GraphProperties::undirected()).renumber();

    let mut graph = graph_from_series::<Transposed>(&handle, &src, &dst, None, &opts).expect("graph_from_series");

    let result = graph.pagerank_simple(&handle, &PageRankParams::default()).expect("pagerank");
    let df = result.to_dataframe().expect("to_dataframe");

    // Verify schema
    assert_eq!(df.num_columns(), 2);
    let names = df.column_names();
    assert_eq!(names[0], "vertex");
    assert_eq!(names[1], "value");

    // Verify values
    let vertices: Vec<i32> = df.series_view("vertex").expect("vertex").to_vec().expect("v");
    let values: Vec<f32> = df.series_view("value").expect("value").to_vec().expect("v");

    assert_eq!(vertices.len(), 3, "triangle has 3 vertices");
    let mut sorted = vertices.clone();
    sorted.sort();
    assert_eq!(sorted, vec![0, 1, 2]);

    for &v in &values {
        assert!(v > 0.0 && v.is_finite());
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Provenance validation in graph_from_series
// ═══════════════════════════════════════════════════════════════════════════

/// `graph_from_series` should reject src and dst from different Runtimes.
#[test]
fn graph_from_series_rejects_mixed_provenance() {
    let handle = ResourceHandle::new().expect("handle");

    let rt_a = cudf::Runtime::builder().build().expect("runtime A");
    let rt_b = cudf::Runtime::builder().build().expect("runtime B");

    let src = cudf::Series::new_with_runtime(&rt_a, "src", &[0i32, 1, 2]).expect("src");
    let dst = cudf::Series::new_with_runtime(&rt_b, "dst", &[1i32, 2, 0]).expect("dst");

    let opts = SeriesGraphOptions::default().renumber();
    let err =
        graph_from_series::<Forward>(&handle, &src, &dst, None, &opts).expect_err("should reject mixed provenance");

    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert!(err.message().contains("provenance"), "error message should mention provenance, got: {}", err.message());
}

/// `graph_from_series` should reject weights from a different Runtime than src/dst.
#[test]
fn graph_from_series_rejects_mixed_provenance_weights() {
    let handle = ResourceHandle::new().expect("handle");

    let rt_a = cudf::Runtime::builder().build().expect("runtime A");
    let rt_b = cudf::Runtime::builder().build().expect("runtime B");

    let src = cudf::Series::new_with_runtime(&rt_a, "src", &[0i32, 1, 2]).expect("src");
    let dst = cudf::Series::new_with_runtime(&rt_a, "dst", &[1i32, 2, 0]).expect("dst");
    let wgt = cudf::Series::new_with_runtime(&rt_b, "wgt", &[1.0f32, 2.0, 3.0]).expect("wgt");

    let opts = SeriesGraphOptions::default().renumber();
    let err =
        graph_from_series::<Forward>(&handle, &src, &dst, Some(&wgt), &opts).expect_err("should reject mixed weights");

    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert!(err.message().contains("provenance"), "error message should mention provenance, got: {}", err.message());
}

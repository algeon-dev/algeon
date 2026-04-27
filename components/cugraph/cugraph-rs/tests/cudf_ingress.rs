//! Integration tests for cuDF-backed graph construction ingress paths.
mod common;

use std::collections::{BTreeMap, HashMap};

use common::{
    cudf::{
        TempParquetFixture, assert_score_maps_close_f32, explicit_runtime, python_undirected_multigraph_dataframe,
        python_weighted_reciprocal_dataframe, python_weighted_reciprocal_expected_pagerank_map, read_parquet_dataframe,
        unique_temp_dir, write_parquet_dataframe,
    },
    support::{
        BFS_SOURCE_VERTEX_I32, LOUVAIN_GRAPH_DST_I32, LOUVAIN_GRAPH_SRC_I32, LOUVAIN_GRAPH_WEIGHT_F32,
        bfs_graph_dataframe,
    },
};
use cudf::core::Table;
use cugraph::{
    array::DeviceArray,
    centrality::{BetweennessCentralityParams, PageRankParams},
    community::LouvainParams,
    cudf_interop::{
        DataFrameGraphOptions, GraphConstructionPolicy, ParquetGraphOptions, TableGraphOptions,
        graph_from_borrowed_table_view, graph_from_dataframe, graph_from_parquet,
    },
    graph::{Forward, GraphBuilder, Transposed},
    handle::{EnteredContext, ResourceHandle},
    traversal::BfsParams,
    types::GraphProperties,
};

fn test_handle() -> ResourceHandle {
    let runtime = std::sync::Arc::new(explicit_runtime());
    ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).expect("handle")
}

#[derive(Debug, Clone, PartialEq)]
struct LouvainShapeSummary {
    edge_count: usize,
    community_count: usize,
    community_sizes_sorted: Vec<usize>,
    modularity: f64,
}

fn louvain_fixture_dataframe(runtime: &cudf::Runtime) -> cudf::DataFrame {
    let src = runtime.series(LOUVAIN_GRAPH_SRC_I32).expect("src");
    let dst = runtime.series(LOUVAIN_GRAPH_DST_I32).expect("dst");
    let weight = runtime.series(LOUVAIN_GRAPH_WEIGHT_F32).expect("weight");
    cudf::DataFrame::from_columns([("src", src), ("dst", dst), ("weight", weight)]).expect("louvain graph dataframe")
}

fn assert_python_weighted_ingress_scores_match_expected(
    handle: &EnteredContext<'_>,
    label: &str,
    graph: &mut cugraph::Graph<Transposed>,
) {
    let actual = common::cudf::pagerank_score_map(handle, graph);
    let expected = python_weighted_reciprocal_expected_pagerank_map(handle);
    assert_score_maps_close_f32(&actual, &expected, 1.0e-6, label);
}

fn summarize_louvain_shape(graph: &mut cugraph::Graph<Forward>, handle: &EnteredContext<'_>) -> LouvainShapeSummary {
    let edge_count = graph.edge_count(handle).expect("edge_count");
    let result = graph.louvain(handle, &LouvainParams::default()).expect("louvain");
    let clusters = result.clusters().expect("clusters").to_vec_i32(handle).expect("cluster copy");

    let mut counts = BTreeMap::<i32, usize>::new();
    for cluster in clusters {
        *counts.entry(cluster).or_insert(0) += 1;
    }

    let mut community_sizes_sorted = counts.into_values().collect::<Vec<_>>();
    community_sizes_sorted.sort_unstable();

    LouvainShapeSummary {
        edge_count,
        community_count: community_sizes_sorted.len(),
        community_sizes_sorted,
        modularity: result.modularity(),
    }
}

fn summarize_pagerank_scores(
    graph: &mut cugraph::Graph<Transposed>,
    handle: &EnteredContext<'_>,
) -> BTreeMap<i32, f64> {
    let result = graph.pagerank(handle, &PageRankParams::default()).expect("pagerank");
    let vertices = result.vertex_ids().expect("vertex_ids").to_vec_i32(handle).expect("vertex copy");
    let scores = result.scores().expect("scores").to_vec_f64(handle).expect("score copy");
    vertices.into_iter().zip(scores).collect()
}

fn summarize_betweenness_scores(
    graph: &mut cugraph::Graph<Forward>,
    handle: &EnteredContext<'_>,
) -> BTreeMap<i32, f64> {
    let result =
        graph.betweenness_centrality_all(handle, &BetweennessCentralityParams::default()).expect("betweenness");
    let vertices = result.vertex_ids().expect("vertex_ids").to_vec_i32(handle).expect("vertex copy");
    let scores = result.scores().expect("scores").to_vec_f64(handle).expect("score copy");
    vertices.into_iter().zip(scores).collect()
}

fn assert_score_maps_close(actual: &BTreeMap<i32, f64>, expected: &BTreeMap<i32, f64>, tolerance: f64, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label} vertex count mismatch");
    assert_eq!(
        actual.keys().copied().collect::<Vec<_>>(),
        expected.keys().copied().collect::<Vec<_>>(),
        "{label} vertex ids mismatch"
    );

    for (vertex, expected_score) in expected {
        let actual_score = actual.get(vertex).expect("score entry");
        assert!(
            (actual_score - expected_score).abs() <= tolerance,
            "{label} score mismatch for vertex {vertex}: expected {expected_score}, got {actual_score}"
        );
    }
}

#[test]
fn graph_from_parquet_then_bfs() {
    let handle = test_handle();
    let entered = handle.enter("graph_from_parquet_then_bfs").expect("entered");
    let fixture = TempParquetFixture::bfs_graph();

    let opts = ParquetGraphOptions::new(fixture.path_str(), 0, 1)
        .weight_col(2)
        .properties(GraphProperties::directed())
        .renumber();
    let loaded = graph_from_parquet::<Forward>(&entered, &opts).expect("graph_from_parquet");
    let mut graph = loaded.graph;

    let sources = DeviceArray::from_slice(&entered, &[BFS_SOURCE_VERTEX_I32]).expect("sources");
    let sources_view = sources.view().expect("sources_view");
    let result = graph.bfs(&entered, &sources_view, &BfsParams::default()).expect("bfs");

    let vertices: Vec<i32> = result.vertex_ids().expect("vertex_ids").to_vec_i32(&entered).expect("vertices");
    let distances: Vec<i32> = result.distances().expect("distances").to_vec_i32(&entered).expect("distances");
    let dist_map: HashMap<i32, i32> = vertices.into_iter().zip(distances).collect();

    assert!(!dist_map.is_empty(), "graph_from_parquet should produce a non-empty graph");
    assert_eq!(dist_map[&0], 0, "source vertex 0 should have distance 0");
}

#[test]
fn graph_from_parquet_rejects_nonexistent_file() {
    let handle = test_handle();
    let entered = handle.enter("graph_from_parquet_rejects_nonexistent_file").expect("entered");
    let opts = ParquetGraphOptions::new("/tmp/this_file_definitely_does_not_exist_12345.parquet", 0, 1)
        .properties(GraphProperties::directed())
        .renumber();

    let err = graph_from_parquet::<Forward>(&entered, &opts).expect_err("should reject non-existent file");
    assert_eq!(err.kind(), cugraph::error::ErrorKind::IoError);
    assert_eq!(err.operation(), "graph_from_parquet");
}

#[test]
fn graph_from_parquet_rejects_nulls_in_src_column() {
    let runtime = explicit_runtime();
    let src = runtime.column_from_slice_nullable(&[0i32, 1, 2], &[true, false, true]).expect("src");
    let dst = runtime.column_from_slice(&[1i32, 2, 0]).expect("dst");
    let wgt = runtime.column_from_slice(&[1.0f32, 2.0, 3.0]).expect("wgt");

    let table = Table::from_columns(vec![src, dst, wgt]).expect("table");
    let tmp_dir = unique_temp_dir("parquet_nulls");
    std::fs::create_dir_all(&tmp_dir).expect("create tmp dir");
    let path = tmp_dir.join("edges_with_null_src.parquet");

    let df = cudf::DataFrame::from_table(table, vec!["0".into(), "1".into(), "2".into()]).expect("dataframe");
    write_parquet_dataframe(&df, &path);

    let handle = test_handle();
    let entered = handle.enter("graph_from_parquet_rejects_nulls_in_src_column").expect("entered");
    let opts = ParquetGraphOptions::new(path.to_str().expect("utf8 path"), 0, 1)
        .weight_col(2)
        .properties(GraphProperties::directed())
        .renumber();
    let err = graph_from_parquet::<Forward>(&entered, &opts).expect_err("should reject nullable src column");

    assert_eq!(err.kind(), cugraph::error::ErrorKind::InvalidInput);
    assert_eq!(err.operation(), "graph_from_parquet");
    assert!(err.message().contains("null"), "unexpected error message: {}", err.message());

    let _ = std::fs::remove_dir_all(&tmp_dir);
}

#[test]
fn graph_from_borrowed_table_view_then_bfs() {
    let handle = test_handle();
    let entered = handle.enter("graph_from_borrowed_table_view_then_bfs").expect("entered");
    let runtime = explicit_runtime();
    let fixture = TempParquetFixture::bfs_graph();

    let df = read_parquet_dataframe(fixture.path(), &runtime);
    let borrowed = df.try_borrow_table_view().expect("borrowed table view");

    let opts = TableGraphOptions::new(0, 1).weight_col(2).properties(GraphProperties::directed()).renumber();
    let mut graph =
        graph_from_borrowed_table_view::<Forward>(&entered, &borrowed, &opts).expect("graph_from_borrowed_table_view");

    let sources = DeviceArray::from_slice(&entered, &[BFS_SOURCE_VERTEX_I32]).expect("sources");
    let sources_view = sources.view().expect("sources_view");
    let result = graph.bfs(&entered, &sources_view, &BfsParams::default()).expect("bfs");

    let vertices: Vec<i32> = result.vertex_ids().expect("vertex_ids").to_vec_i32(&entered).expect("vertices");
    let distances: Vec<i32> = result.distances().expect("distances").to_vec_i32(&entered).expect("distances");
    let dist_map: HashMap<i32, i32> = vertices.into_iter().zip(distances).collect();

    assert!(!dist_map.is_empty(), "should produce a non-empty graph");
    assert_eq!(dist_map[&0], 0, "source vertex 0 should have distance 0");
}

#[test]
fn graph_from_borrowed_table_view_without_weights() {
    let handle = test_handle();
    let entered = handle.enter("graph_from_borrowed_table_view_without_weights").expect("entered");
    let runtime = explicit_runtime();
    let fixture = TempParquetFixture::bfs_graph();

    let df = read_parquet_dataframe(fixture.path(), &runtime);
    let borrowed = df.try_borrow_table_view().expect("borrowed table view");

    let opts = TableGraphOptions::new(0, 1).properties(GraphProperties::directed()).renumber();
    let mut graph = graph_from_borrowed_table_view::<Forward>(&entered, &borrowed, &opts)
        .expect("graph_from_borrowed_table_view without weights");

    let sources = DeviceArray::from_slice(&entered, &[BFS_SOURCE_VERTEX_I32]).expect("sources");
    let sources_view = sources.view().expect("sources_view");
    let result = graph.bfs(&entered, &sources_view, &BfsParams::default()).expect("bfs");
    let vertices: Vec<i32> = result.vertex_ids().expect("vertex_ids").to_vec_i32(&entered).expect("vertices");

    assert!(!vertices.is_empty(), "graph should have vertices");
}

#[test]
fn graph_from_borrowed_table_view_rejects_out_of_bounds_column() {
    let handle = test_handle();
    let entered = handle.enter("graph_from_borrowed_table_view_rejects_out_of_bounds_column").expect("entered");
    let runtime = explicit_runtime();
    let fixture = TempParquetFixture::bfs_graph();

    let df = read_parquet_dataframe(fixture.path(), &runtime);
    let borrowed = df.try_borrow_table_view().expect("borrowed table view");

    let opts = TableGraphOptions::new(0, 99).properties(GraphProperties::directed()).renumber();
    let err = graph_from_borrowed_table_view::<Forward>(&entered, &borrowed, &opts)
        .expect_err("should reject out-of-bounds column");
    assert_eq!(err.kind(), cugraph::error::ErrorKind::InvalidInput);
}

#[test]
fn graph_from_borrowed_table_view_rejects_nullable_column() {
    let runtime = explicit_runtime();
    let src = runtime.column_from_slice_nullable(&[0i32, 1, 2], &[true, false, true]).expect("src");
    let dst = runtime.column_from_slice(&[1i32, 2, 0]).expect("dst");
    let wgt = runtime.column_from_slice(&[1.0f32, 2.0, 3.0]).expect("wgt");

    let table = Table::from_columns(vec![src, dst, wgt]).expect("table");
    let df = cudf::DataFrame::from_table(table, vec!["src".into(), "dst".into(), "weight".into()]).expect("dataframe");
    let borrowed = df.try_borrow_table_view().expect("borrowed table view");

    let handle = test_handle();
    let entered = handle.enter("graph_from_borrowed_table_view_rejects_nullable_column").expect("entered");
    let opts = TableGraphOptions::new(0, 1).weight_col(2).properties(GraphProperties::directed()).renumber();
    let err = graph_from_borrowed_table_view::<Forward>(&entered, &borrowed, &opts)
        .expect_err("should reject nullable column");

    assert_eq!(err.kind(), cugraph::error::ErrorKind::InvalidInput);
    assert!(err.message().contains("null"), "error message should mention null, got: {}", err.message());
}

#[test]
fn graph_from_dataframe_by_name() {
    let handle = test_handle();
    let entered = handle.enter("graph_from_dataframe_by_name").expect("entered");
    let runtime = explicit_runtime();
    let df = bfs_graph_dataframe(&runtime);

    let opts = DataFrameGraphOptions::new("src", "dst")
        .weight_col("weight")
        .properties(GraphProperties::directed())
        .renumber();
    let mut graph = graph_from_dataframe::<Forward>(&entered, &df, &opts).expect("graph_from_dataframe");

    let sources = DeviceArray::from_slice(&entered, &[BFS_SOURCE_VERTEX_I32]).expect("sources");
    let sources_view = sources.view().expect("sources_view");
    let result = graph.bfs(&entered, &sources_view, &BfsParams::default()).expect("bfs");

    let vertices: Vec<i32> = result.vertex_ids().expect("vertex_ids").to_vec_i32(&entered).expect("vertices");
    let distances: Vec<i32> = result.distances().expect("distances").to_vec_i32(&entered).expect("distances");
    let dist_map: HashMap<i32, i32> = vertices.into_iter().zip(distances).collect();

    assert_eq!(dist_map[&0], 0, "source vertex 0 should have distance 0");
    assert_eq!(dist_map[&1], 1, "vertex 1 should have distance 1");
}

#[test]
fn graph_from_dataframe_rejects_unknown_column_name() {
    let runtime = explicit_runtime();
    let src = runtime.series(&[0i32, 1]).expect("src");
    let dst = runtime.series(&[1i32, 0]).expect("dst");
    let df = cudf::DataFrame::from_columns([("src", src), ("dst", dst)]).expect("dataframe");

    let handle = test_handle();
    let entered = handle.enter("graph_from_dataframe_rejects_unknown_column_name").expect("entered");
    let opts = DataFrameGraphOptions::new("nonexistent", "dst").properties(GraphProperties::directed()).renumber();
    let err = graph_from_dataframe::<Forward>(&entered, &df, &opts).expect_err("should reject unknown column name");
    assert_eq!(err.kind(), cugraph::error::ErrorKind::InvalidInput);
}

#[test]
fn graph_from_dataframe_undirected_louvain_defaults_to_python_cugraph_policy() {
    let handle = test_handle();
    let entered =
        handle.enter("graph_from_dataframe_undirected_louvain_defaults_to_python_cugraph_policy").expect("entered");
    let runtime = explicit_runtime();
    let df = louvain_fixture_dataframe(&runtime);

    let src = DeviceArray::from_slice(&entered, LOUVAIN_GRAPH_SRC_I32).expect("src");
    let dst = DeviceArray::from_slice(&entered, LOUVAIN_GRAPH_DST_I32).expect("dst");
    let weights = DeviceArray::from_slice(&entered, LOUVAIN_GRAPH_WEIGHT_F32).expect("weights");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let weights_view = weights.view().expect("weights_view");

    let mut expected = GraphBuilder::new(&src_view, &dst_view)
        .weights(&weights_view)
        .properties(GraphProperties::undirected())
        .renumber()
        .drop_multi_edges()
        .symmetrize()
        .build::<Forward>(&entered)
        .expect("python-compatible graph");
    let expected = summarize_louvain_shape(&mut expected, &entered);

    let opts = DataFrameGraphOptions::new("src", "dst")
        .weight_col("weight")
        .properties(GraphProperties::undirected())
        .renumber();
    let mut actual = graph_from_dataframe::<Forward>(&entered, &df, &opts).expect("graph_from_dataframe");
    let actual = summarize_louvain_shape(&mut actual, &entered);

    assert_eq!(actual.edge_count, expected.edge_count);
    assert_eq!(actual.community_count, expected.community_count);
    assert_eq!(actual.community_sizes_sorted, expected.community_sizes_sorted);
    assert!(
        (actual.modularity - expected.modularity).abs() < 1e-9,
        "expected modularity {} but got {}",
        expected.modularity,
        actual.modularity
    );
}

#[test]
fn graph_from_dataframe_can_explicitly_request_raw_libcugraph_policy() {
    let handle = test_handle();
    let entered = handle.enter("graph_from_dataframe_can_explicitly_request_raw_libcugraph_policy").expect("entered");
    let runtime = explicit_runtime();
    let df = louvain_fixture_dataframe(&runtime);

    let src = DeviceArray::from_slice(&entered, LOUVAIN_GRAPH_SRC_I32).expect("src");
    let dst = DeviceArray::from_slice(&entered, LOUVAIN_GRAPH_DST_I32).expect("dst");
    let weights = DeviceArray::from_slice(&entered, LOUVAIN_GRAPH_WEIGHT_F32).expect("weights");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let weights_view = weights.view().expect("weights_view");

    let mut expected = GraphBuilder::new(&src_view, &dst_view)
        .weights(&weights_view)
        .properties(GraphProperties::undirected())
        .renumber()
        .build::<Forward>(&entered)
        .expect("raw graph");
    let expected = summarize_louvain_shape(&mut expected, &entered);

    let opts = DataFrameGraphOptions::new("src", "dst")
        .weight_col("weight")
        .properties(GraphProperties::undirected())
        .renumber()
        .construction_policy(GraphConstructionPolicy::RawLibcugraph);
    let mut actual = graph_from_dataframe::<Forward>(&entered, &df, &opts).expect("graph_from_dataframe");
    let actual = summarize_louvain_shape(&mut actual, &entered);

    assert_eq!(actual.edge_count, expected.edge_count);
    assert_eq!(actual.community_count, expected.community_count);
    assert_eq!(actual.community_sizes_sorted, expected.community_sizes_sorted);
    assert!(
        (actual.modularity - expected.modularity).abs() < 1e-9,
        "expected modularity {} but got {}",
        expected.modularity,
        actual.modularity
    );
}

#[test]
fn graph_from_dataframe_python_cugraph_weighted_undirected_simple_normalization_matches_expected_across_ingress_paths()
{
    let handle = test_handle();
    let entered = handle.enter("graph_from_dataframe_python_cugraph_weighted_undirected_simple_normalization_matches_expected_across_ingress_paths").expect("entered");
    let runtime = explicit_runtime();
    let df = python_weighted_reciprocal_dataframe(&runtime);

    let df_opts = DataFrameGraphOptions::new("src", "dst")
        .weight_col("weight")
        .properties(GraphProperties::undirected())
        .renumber();
    let mut dataframe_graph =
        graph_from_dataframe::<Transposed>(&entered, &df, &df_opts).expect("graph_from_dataframe");
    assert_python_weighted_ingress_scores_match_expected(&entered, "dataframe", &mut dataframe_graph);

    let borrowed = df.try_borrow_table_view().expect("borrowed table view");
    let table_opts = TableGraphOptions::new(0, 1).weight_col(2).properties(GraphProperties::undirected()).renumber();
    let mut borrowed_graph = graph_from_borrowed_table_view::<Transposed>(&entered, &borrowed, &table_opts)
        .expect("graph_from_borrowed_table_view");
    assert_python_weighted_ingress_scores_match_expected(&entered, "borrowed_table", &mut borrowed_graph);

    let tmp_dir = unique_temp_dir("python_weighted_ingress");
    std::fs::create_dir_all(&tmp_dir).expect("create tmp dir");
    let parquet_path = tmp_dir.join("python_weighted_ingress.parquet");
    write_parquet_dataframe(&df, &parquet_path);

    let parquet_opts = ParquetGraphOptions::new(parquet_path.to_str().expect("utf8 path"), 0, 1)
        .weight_col(2)
        .properties(GraphProperties::undirected())
        .renumber();
    let loaded = graph_from_parquet::<Transposed>(&entered, &parquet_opts).expect("graph_from_parquet");
    let mut parquet_graph = loaded.graph;
    assert_python_weighted_ingress_scores_match_expected(&entered, "parquet", &mut parquet_graph);

    let _ = std::fs::remove_dir_all(&tmp_dir);
}

#[test]
fn graph_from_dataframe_python_cugraph_accepts_undirected_multigraph_properties_across_ingress_paths() {
    let handle = test_handle();
    let entered = handle
        .enter("graph_from_dataframe_python_cugraph_accepts_undirected_multigraph_properties_across_ingress_paths")
        .expect("entered");
    let runtime = explicit_runtime();
    let df = python_undirected_multigraph_dataframe(&runtime);
    let properties = GraphProperties { is_symmetric: true, is_multigraph: true };

    let df_opts = DataFrameGraphOptions::new("src", "dst").properties(properties).renumber();
    let mut dataframe_graph = graph_from_dataframe::<Forward>(&entered, &df, &df_opts).expect("graph_from_dataframe");
    assert_eq!(dataframe_graph.edge_count(&entered).expect("edge_count"), 3);

    let borrowed = df.try_borrow_table_view().expect("borrowed table view");
    let table_opts = TableGraphOptions::new(0, 1).properties(properties).renumber();
    let mut borrowed_graph = graph_from_borrowed_table_view::<Forward>(&entered, &borrowed, &table_opts)
        .expect("graph_from_borrowed_table_view");
    assert_eq!(borrowed_graph.edge_count(&entered).expect("edge_count"), 3);

    let tmp_dir = unique_temp_dir("python_multigraph_ingress");
    std::fs::create_dir_all(&tmp_dir).expect("create tmp dir");
    let parquet_path = tmp_dir.join("python_multigraph_ingress.parquet");
    write_parquet_dataframe(&df, &parquet_path);

    let parquet_opts =
        ParquetGraphOptions::new(parquet_path.to_str().expect("utf8 path"), 0, 1).properties(properties).renumber();
    let loaded = graph_from_parquet::<Forward>(&entered, &parquet_opts).expect("graph_from_parquet");
    let mut parquet_graph = loaded.graph;
    assert_eq!(parquet_graph.edge_count(&entered).expect("edge_count"), 3);

    let _ = std::fs::remove_dir_all(&tmp_dir);
}

#[test]
fn graph_from_dataframe_undirected_pagerank_defaults_to_python_cugraph_policy() {
    let handle = test_handle();
    let entered =
        handle.enter("graph_from_dataframe_undirected_pagerank_defaults_to_python_cugraph_policy").expect("entered");
    let runtime = explicit_runtime();
    let df = louvain_fixture_dataframe(&runtime);

    let src = DeviceArray::from_slice(&entered, LOUVAIN_GRAPH_SRC_I32).expect("src");
    let dst = DeviceArray::from_slice(&entered, LOUVAIN_GRAPH_DST_I32).expect("dst");
    let weights = DeviceArray::from_slice(&entered, LOUVAIN_GRAPH_WEIGHT_F32).expect("weights");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let weights_view = weights.view().expect("weights_view");

    let mut expected = GraphBuilder::new(&src_view, &dst_view)
        .weights(&weights_view)
        .properties(GraphProperties::undirected())
        .renumber()
        .drop_multi_edges()
        .symmetrize()
        .build::<Transposed>(&entered)
        .expect("python-compatible graph");
    let expected = summarize_pagerank_scores(&mut expected, &entered);

    let opts = DataFrameGraphOptions::new("src", "dst")
        .weight_col("weight")
        .properties(GraphProperties::undirected())
        .renumber();
    let mut actual = graph_from_dataframe::<Transposed>(&entered, &df, &opts).expect("graph_from_dataframe");
    let actual = summarize_pagerank_scores(&mut actual, &entered);

    assert_score_maps_close(&actual, &expected, 1e-6, "pagerank");
}

#[test]
fn graph_from_dataframe_undirected_betweenness_defaults_to_python_cugraph_policy() {
    let handle = test_handle();
    let entered =
        handle.enter("graph_from_dataframe_undirected_betweenness_defaults_to_python_cugraph_policy").expect("entered");
    let runtime = explicit_runtime();
    let df = louvain_fixture_dataframe(&runtime);

    let src = DeviceArray::from_slice(&entered, LOUVAIN_GRAPH_SRC_I32).expect("src");
    let dst = DeviceArray::from_slice(&entered, LOUVAIN_GRAPH_DST_I32).expect("dst");
    let weights = DeviceArray::from_slice(&entered, LOUVAIN_GRAPH_WEIGHT_F32).expect("weights");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let weights_view = weights.view().expect("weights_view");

    let mut expected = GraphBuilder::new(&src_view, &dst_view)
        .weights(&weights_view)
        .properties(GraphProperties::undirected())
        .renumber()
        .drop_multi_edges()
        .symmetrize()
        .build::<Forward>(&entered)
        .expect("python-compatible graph");
    let expected = summarize_betweenness_scores(&mut expected, &entered);

    let opts = DataFrameGraphOptions::new("src", "dst")
        .weight_col("weight")
        .properties(GraphProperties::undirected())
        .renumber();
    let mut actual = graph_from_dataframe::<Forward>(&entered, &df, &opts).expect("graph_from_dataframe");
    let actual = summarize_betweenness_scores(&mut actual, &entered);

    assert_score_maps_close(&actual, &expected, 1e-6, "betweenness");
}

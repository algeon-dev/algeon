//! Integration tests for cuDF-backed graph construction ingress paths.
mod common;

use std::collections::HashMap;

use common::{
    cudf::{TempParquetFixture, explicit_runtime, read_parquet_table, unique_temp_dir, write_parquet_table},
    support::{BFS_SOURCE_VERTEX_I32, bfs_graph_dataframe},
};
use cudf::core::Table;
use cugraph::{
    array::DeviceArray,
    cudf_interop::{
        DataFrameGraphOptions, ParquetGraphOptions, TableGraphOptions, graph_from_dataframe, graph_from_parquet,
        graph_from_table_view,
    },
    graph::Forward,
    handle::ResourceHandle,
    traversal::BfsParams,
    types::GraphProperties,
};

#[test]
fn graph_from_parquet_then_bfs() {
    let handle = ResourceHandle::new().expect("handle");
    let fixture = TempParquetFixture::bfs_graph();

    let opts = ParquetGraphOptions::new(fixture.path_str(), 0, 1)
        .weight_col(2)
        .properties(GraphProperties::directed())
        .renumber();
    let loaded = graph_from_parquet::<Forward>(&handle, &opts).expect("graph_from_parquet");
    let mut graph = loaded.graph;

    let sources = DeviceArray::from_slice(&handle, &[BFS_SOURCE_VERTEX_I32]).expect("sources");
    let sources_view = sources.view().expect("sources_view");
    let result = graph.bfs(&handle, &sources_view, &BfsParams::default()).expect("bfs");

    let vertices: Vec<i32> = result.vertex_ids().expect("vertex_ids").to_vec_i32(&handle).expect("vertices");
    let distances: Vec<i32> = result.distances().expect("distances").to_vec_i32(&handle).expect("distances");
    let dist_map: HashMap<i32, i32> = vertices.into_iter().zip(distances).collect();

    assert!(!dist_map.is_empty(), "graph_from_parquet should produce a non-empty graph");
    assert_eq!(dist_map[&0], 0, "source vertex 0 should have distance 0");
}

#[test]
fn graph_from_parquet_rejects_nonexistent_file() {
    let handle = ResourceHandle::new().expect("handle");
    let opts = ParquetGraphOptions::new("/tmp/this_file_definitely_does_not_exist_12345.parquet", 0, 1)
        .properties(GraphProperties::directed())
        .renumber();

    let err = graph_from_parquet::<Forward>(&handle, &opts).expect_err("should reject non-existent file");
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

    let table_view = table.view().expect("table view");
    write_parquet_table(&table_view, &path, &["0", "1", "2"], &runtime);

    let handle = ResourceHandle::new().expect("handle");
    let opts = ParquetGraphOptions::new(path.to_str().expect("utf8 path"), 0, 1)
        .weight_col(2)
        .properties(GraphProperties::directed())
        .renumber();
    let err = graph_from_parquet::<Forward>(&handle, &opts).expect_err("should reject nullable src column");

    assert_eq!(err.kind(), cugraph::error::ErrorKind::InvalidInput);
    assert_eq!(err.operation(), "graph_from_prepared_table_export");
    assert!(err.message().contains("null"), "unexpected error message: {}", err.message());

    let _ = std::fs::remove_dir_all(&tmp_dir);
}

#[test]
fn graph_from_table_view_then_bfs() {
    let handle = ResourceHandle::new().expect("handle");
    let runtime = explicit_runtime();
    let fixture = TempParquetFixture::bfs_graph();

    let table = read_parquet_table(fixture.path(), &runtime);
    let view = table.view().expect("table view");

    let opts = TableGraphOptions::new(0, 1).weight_col(2).properties(GraphProperties::directed()).renumber();
    let mut graph = graph_from_table_view::<Forward>(&handle, &view, &opts).expect("graph_from_table_view");

    let sources = DeviceArray::from_slice(&handle, &[BFS_SOURCE_VERTEX_I32]).expect("sources");
    let sources_view = sources.view().expect("sources_view");
    let result = graph.bfs(&handle, &sources_view, &BfsParams::default()).expect("bfs");

    let vertices: Vec<i32> = result.vertex_ids().expect("vertex_ids").to_vec_i32(&handle).expect("vertices");
    let distances: Vec<i32> = result.distances().expect("distances").to_vec_i32(&handle).expect("distances");
    let dist_map: HashMap<i32, i32> = vertices.into_iter().zip(distances).collect();

    assert!(!dist_map.is_empty(), "should produce a non-empty graph");
    assert_eq!(dist_map[&0], 0, "source vertex 0 should have distance 0");
}

#[test]
fn graph_from_table_view_without_weights() {
    let handle = ResourceHandle::new().expect("handle");
    let runtime = explicit_runtime();
    let fixture = TempParquetFixture::bfs_graph();

    let table = read_parquet_table(fixture.path(), &runtime);
    let view = table.view().expect("table view");

    let opts = TableGraphOptions::new(0, 1).properties(GraphProperties::directed()).renumber();
    let mut graph =
        graph_from_table_view::<Forward>(&handle, &view, &opts).expect("graph_from_table_view without weights");

    let sources = DeviceArray::from_slice(&handle, &[BFS_SOURCE_VERTEX_I32]).expect("sources");
    let sources_view = sources.view().expect("sources_view");
    let result = graph.bfs(&handle, &sources_view, &BfsParams::default()).expect("bfs");
    let vertices: Vec<i32> = result.vertex_ids().expect("vertex_ids").to_vec_i32(&handle).expect("vertices");

    assert!(!vertices.is_empty(), "graph should have vertices");
}

#[test]
fn graph_from_table_view_rejects_out_of_bounds_column() {
    let handle = ResourceHandle::new().expect("handle");
    let runtime = explicit_runtime();
    let fixture = TempParquetFixture::bfs_graph();

    let table = read_parquet_table(fixture.path(), &runtime);
    let view = table.view().expect("table view");

    let opts = TableGraphOptions::new(0, 99).properties(GraphProperties::directed()).renumber();
    let err = graph_from_table_view::<Forward>(&handle, &view, &opts).expect_err("should reject out-of-bounds column");
    assert_eq!(err.kind(), cugraph::error::ErrorKind::InvalidInput);
}

#[test]
fn graph_from_table_view_rejects_nullable_column() {
    let runtime = explicit_runtime();
    let src = runtime.column_from_slice_nullable(&[0i32, 1, 2], &[true, false, true]).expect("src");
    let dst = runtime.column_from_slice(&[1i32, 2, 0]).expect("dst");
    let wgt = runtime.column_from_slice(&[1.0f32, 2.0, 3.0]).expect("wgt");

    let table = Table::from_columns(vec![src, dst, wgt]).expect("table");
    let view = table.view().expect("table view");

    let handle = ResourceHandle::new().expect("handle");
    let opts = TableGraphOptions::new(0, 1).weight_col(2).properties(GraphProperties::directed()).renumber();
    let err = graph_from_table_view::<Forward>(&handle, &view, &opts).expect_err("should reject nullable column");

    assert_eq!(err.kind(), cugraph::error::ErrorKind::InvalidInput);
    assert!(err.message().contains("null"), "error message should mention null, got: {}", err.message());
}

#[test]
fn graph_from_dataframe_by_name() {
    let handle = ResourceHandle::new().expect("handle");
    let runtime = explicit_runtime();
    let df = bfs_graph_dataframe(&runtime);

    let opts = DataFrameGraphOptions::new("src", "dst")
        .weight_col("weight")
        .properties(GraphProperties::directed())
        .renumber();
    let mut graph = graph_from_dataframe::<Forward>(&handle, &df, &opts).expect("graph_from_dataframe");

    let sources = DeviceArray::from_slice(&handle, &[BFS_SOURCE_VERTEX_I32]).expect("sources");
    let sources_view = sources.view().expect("sources_view");
    let result = graph.bfs(&handle, &sources_view, &BfsParams::default()).expect("bfs");

    let vertices: Vec<i32> = result.vertex_ids().expect("vertex_ids").to_vec_i32(&handle).expect("vertices");
    let distances: Vec<i32> = result.distances().expect("distances").to_vec_i32(&handle).expect("distances");
    let dist_map: HashMap<i32, i32> = vertices.into_iter().zip(distances).collect();

    assert_eq!(dist_map[&0], 0, "source vertex 0 should have distance 0");
    assert_eq!(dist_map[&1], 1, "vertex 1 should have distance 1");
}

#[test]
fn graph_from_dataframe_rejects_unknown_column_name() {
    let runtime = explicit_runtime();
    let src = runtime.series("src", &[0i32, 1]).expect("src");
    let dst = runtime.series("dst", &[1i32, 0]).expect("dst");
    let df = cudf::DataFrame::from_series([src, dst]).expect("dataframe");

    let handle = ResourceHandle::new().expect("handle");
    let opts = DataFrameGraphOptions::new("nonexistent", "dst").properties(GraphProperties::directed()).renumber();
    let err = graph_from_dataframe::<Forward>(&handle, &df, &opts).expect_err("should reject unknown column name");
    assert_eq!(err.kind(), cugraph::error::ErrorKind::InvalidInput);
}

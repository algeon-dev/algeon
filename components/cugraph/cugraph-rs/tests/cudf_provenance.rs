//! Integration tests for cuDF producer provenance and borrowed-column
//! compatibility requirements.
mod common;

use common::{
    cudf::{
        assert_score_maps_close_f32, explicit_runtime, python_undirected_multigraph_dataframe,
        python_weighted_reciprocal_expected_pagerank_map,
    },
    support::{
        BFS_SOURCE_VERTEX_I32, SIMPLE_CHAIN_DST_I64, SIMPLE_CHAIN_SRC_I64, TRIANGLE_TAIL_DST_I32,
        TRIANGLE_TAIL_SRC_I32, TRIANGLE_TAIL_WEIGHT_F32, bfs_distance_map_i32_result, bfs_distance_map_i64_result,
    },
};
use cugraph::{
    array::DeviceArray,
    cudf_interop::{
        SeriesGraphOptions, TableGraphOptions, graph_from_borrowed_columns, graph_from_borrowed_table_view,
        graph_from_series,
    },
    graph::{Forward, Transposed},
    handle::ResourceHandle,
    traversal::BfsParams,
    types::GraphProperties,
};

fn test_handle() -> ResourceHandle {
    let runtime = explicit_runtime();
    ResourceHandle::for_runtime_device(&runtime).expect("handle")
}

#[test]
fn graph_from_series_allows_mixed_explicit_runtimes_on_same_device() {
    let handle = test_handle();
    let rt_a = explicit_runtime();
    let rt_b = explicit_runtime();

    let src = rt_a.series("src", SIMPLE_CHAIN_SRC_I64).expect("src");
    let dst = rt_b.series("dst", SIMPLE_CHAIN_DST_I64).expect("dst");

    let opts = SeriesGraphOptions::default().renumber();
    let mut graph =
        graph_from_series::<Forward>(&handle, &src, &dst, None, &opts).expect("same-device mixed runtimes should work");

    let sources = DeviceArray::from_slice(&handle, &[0i64]).expect("sources");
    let sources_view = sources.view().expect("sources_view");
    let result = graph.bfs(&handle, &sources_view, &BfsParams::default()).expect("bfs");

    let dist_map = bfs_distance_map_i64_result(&handle, &result);
    assert_eq!(dist_map[&2], 2);
}

#[test]
fn graph_from_borrowed_columns_allows_mixed_explicit_runtimes_on_same_device() {
    let handle = test_handle();
    let rt_a = explicit_runtime();
    let rt_b = explicit_runtime();
    let rt_c = explicit_runtime();

    let src = rt_a.series("src", TRIANGLE_TAIL_SRC_I32).expect("src");
    let dst = rt_b.series("dst", TRIANGLE_TAIL_DST_I32).expect("dst");
    let wgt = rt_c.series("wgt", TRIANGLE_TAIL_WEIGHT_F32).expect("wgt");

    let src_borrowed = src.try_borrow_column_view().expect("src borrowed");
    let dst_borrowed = dst.try_borrow_column_view().expect("dst borrowed");
    let wgt_borrowed = wgt.try_borrow_column_view().expect("wgt borrowed");

    let opts = SeriesGraphOptions::default().renumber();
    let mut graph =
        graph_from_borrowed_columns::<Forward>(&handle, &src_borrowed, &dst_borrowed, Some(&wgt_borrowed), &opts)
            .expect("borrowed mixed runtimes should work");

    let sources = DeviceArray::from_slice(&handle, &[BFS_SOURCE_VERTEX_I32]).expect("sources");
    let sources_view = sources.view().expect("sources_view");
    let result = graph.bfs(&handle, &sources_view, &BfsParams::default()).expect("bfs");

    let dist_map = bfs_distance_map_i32_result(&handle, &result);
    assert_eq!(dist_map[&3], 3);
}

#[test]
fn graph_from_borrowed_columns_python_cugraph_weighted_undirected_simple_keeps_minimum_reciprocal_weight() {
    let handle = test_handle();
    let runtime = explicit_runtime();

    let src = runtime.series("src", &[0i32, 1, 1, 2]).expect("src");
    let dst = runtime.series("dst", &[1i32, 0, 2, 1]).expect("dst");
    let wgt = runtime.series("wgt", &[5.0f32, 2.0, 1.0, 1.0]).expect("wgt");

    let src_borrowed = src.try_borrow_column_view().expect("src borrowed");
    let dst_borrowed = dst.try_borrow_column_view().expect("dst borrowed");
    let wgt_borrowed = wgt.try_borrow_column_view().expect("wgt borrowed");

    let opts = SeriesGraphOptions::default().properties(GraphProperties::undirected()).renumber();
    let mut graph =
        graph_from_borrowed_columns::<Transposed>(&handle, &src_borrowed, &dst_borrowed, Some(&wgt_borrowed), &opts)
            .expect("graph_from_borrowed_columns");

    let actual = common::cudf::pagerank_score_map(&handle, &mut graph);
    let expected = python_weighted_reciprocal_expected_pagerank_map(&handle);
    assert_score_maps_close_f32(&actual, &expected, 1.0e-6, "graph_from_borrowed_columns");
}

#[test]
fn graph_from_borrowed_table_view_python_cugraph_accepts_undirected_multigraph() {
    let handle = test_handle();
    let runtime = explicit_runtime();
    let df = python_undirected_multigraph_dataframe(&runtime);
    let table = df.to_table_copy().expect("to_table_copy");
    let view = table.view().expect("table view");
    let borrowed = view.try_borrow_table_view().expect("borrowed table view");
    let props = GraphProperties { is_symmetric: true, is_multigraph: true };

    let opts = TableGraphOptions::new(0, 1).properties(props).renumber();
    let mut graph =
        graph_from_borrowed_table_view::<Forward>(&handle, &borrowed, &opts).expect("graph_from_borrowed_table_view");

    assert_eq!(graph.edge_count(&handle).expect("edge_count"), 3);
}

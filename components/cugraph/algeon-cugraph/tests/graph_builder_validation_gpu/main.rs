mod builder_owner;

use std::collections::HashMap;

use common::cudf::explicit_runtime;
use cudf::testing::run_named_cases;
use cugraph::{
    DeviceArray, EnteredContext, Forward, GraphBuilder, ResourceHandle, Transposed,
    error::ErrorKind,
    types::{DataType, GraphProperties},
};
#[path = "../support/common/mod.rs"]
mod common;
use common::support::{
    DIRECTED_TRIANGLE_CYCLE_DST_I32, DIRECTED_TRIANGLE_CYCLE_SRC_I32, MULTI_EDGE_DROP_DST_I32, MULTI_EDGE_DROP_SRC_I32,
    SELF_LOOP_DROP_DST_I32, SELF_LOOP_DROP_SRC_I32, TRIANGLE_TAIL_DST_I32, TRIANGLE_TAIL_SRC_I32,
    TRIANGLE_VERTEX_IDS_I32, TRIANGLE_VERTEX_IDS_I64, TWO_CYCLE_DST_I32, TWO_CYCLE_SRC_I32,
    UNDIRECTED_TRIANGLE_DST_I32, UNDIRECTED_TRIANGLE_SRC_I32,
};

fn test_handle() -> ResourceHandle {
    let runtime = std::sync::Arc::new(explicit_runtime());
    ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).expect("handle")
}

fn degree_map(
    graph: &mut cugraph::Graph<Transposed>,
    handle: &EnteredContext<'_>,
    query: &[i32],
) -> HashMap<i32, (i64, i64)> {
    let result = graph.degrees(handle, query).expect("degrees");
    let vertices = result.vertex_ids().expect("vertex_ids").to_vec_i32(handle).expect("vertex copy");
    let in_degrees = result.in_degrees().expect("in_degrees").to_vec_i64(handle).expect("in copy");
    let out_degrees = result.out_degrees().expect("out_degrees").to_vec_i64(handle).expect("out copy");

    vertices.into_iter().zip(in_degrees.into_iter().zip(out_degrees)).collect()
}

#[test]
fn builder_cases_1() {
    run_named_cases(&[
        ("builder_rejects_mismatched_src_dst_lengths", builder_rejects_mismatched_src_dst_lengths),
        ("builder_rejects_mismatched_weights_length", builder_rejects_mismatched_weights_length),
        ("builder_rejects_symmetrize_with_edge_ids", builder_rejects_symmetrize_with_edge_ids),
        ("builder_rejects_mismatched_temporal_dtypes", builder_rejects_mismatched_temporal_dtypes),
        ("builder_rejects_vertices_dtype_mismatch_with_src_dst", builder_rejects_vertices_dtype_mismatch_with_src_dst),
        ("builder_accepts_vertices_matching_src_dst_dtype", builder_accepts_vertices_matching_src_dst_dtype),
        ("builder_chainable_setters_preserve_build_path", builder_chainable_setters_preserve_build_path),
        ("graph_properties_are_preserved_on_built_graph", graph_properties_are_preserved_on_built_graph),
        ("drop_self_loops_removes_loop_degree_contribution", drop_self_loops_removes_loop_degree_contribution),
        ("drop_multi_edges_collapses_duplicate_edges", drop_multi_edges_collapses_duplicate_edges),
    ]);
}

#[test]
fn builder_cases_2() {
    run_named_cases(&[
        ("edge_count_matches_directed_input_rows", edge_count_matches_directed_input_rows),
        ("edge_count_respects_drop_self_loops", edge_count_respects_drop_self_loops),
        ("edge_count_respects_drop_multi_edges", edge_count_respects_drop_multi_edges),
        ("edge_count_matches_python_style_undirected_edges", edge_count_matches_python_style_undirected_edges),
        ("edge_count_counts_undirected_self_loops_once", edge_count_counts_undirected_self_loops_once),
        (
            "graph::tests::test_graph_creation_rejects_mismatched_edge_ids_dtype",
            builder_owner::test_graph_creation_rejects_mismatched_edge_ids_dtype,
        ),
        (
            "graph::tests::test_graph_creation_rejects_mismatched_edge_type_ids_length",
            builder_owner::test_graph_creation_rejects_mismatched_edge_type_ids_length,
        ),
        (
            "graph::tests::test_graph_creation_rejects_mismatched_edge_ids_length",
            builder_owner::test_graph_creation_rejects_mismatched_edge_ids_length,
        ),
        (
            "graph::tests::test_rejects_symmetrize_with_edge_type_ids",
            builder_owner::test_rejects_symmetrize_with_edge_type_ids,
        ),
        (
            "graph::tests::test_rejects_start_time_without_end_time",
            builder_owner::test_rejects_start_time_without_end_time,
        ),
    ]);
}

fn builder_rejects_mismatched_src_dst_lengths() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let handle = test_handle();
    let handle = handle.enter("builder_rejects_mismatched_src_dst_lengths").expect("entered");
    let src = DeviceArray::from_slice(&handle, &[0i32, 1, 2]).expect("src");
    let dst = DeviceArray::from_slice(&handle, &[1i32, 2]).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");

    let err =
        GraphBuilder::new(&src_view, &dst_view).build::<Forward>(&handle).expect_err("must reject mismatched lengths");

    assert_eq!(err.kind(), ErrorKind::InvalidInput);
}

fn builder_rejects_mismatched_weights_length() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let handle = test_handle();
    let handle = handle.enter("builder_rejects_mismatched_weights_length").expect("entered");
    let src = DeviceArray::from_slice(&handle, TRIANGLE_TAIL_SRC_I32).expect("src");
    let dst = DeviceArray::from_slice(&handle, TRIANGLE_TAIL_DST_I32).expect("dst");
    let wgt = DeviceArray::from_slice(&handle, &[1.0f32, 2.0]).expect("wgt");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let wgt_view = wgt.view().expect("wgt_view");

    let err = GraphBuilder::new(&src_view, &dst_view)
        .weights(&wgt_view)
        .build::<Forward>(&handle)
        .expect_err("must reject mismatched weights length");

    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert!(err.message().contains("weights"));
}

fn builder_rejects_symmetrize_with_edge_ids() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let handle = test_handle();
    let handle = handle.enter("builder_rejects_symmetrize_with_edge_ids").expect("entered");
    let src = DeviceArray::from_slice(&handle, DIRECTED_TRIANGLE_CYCLE_SRC_I32).expect("src");
    let dst = DeviceArray::from_slice(&handle, DIRECTED_TRIANGLE_CYCLE_DST_I32).expect("dst");
    let edge_ids = DeviceArray::from_slice(&handle, &[10i32, 20, 30]).expect("edge_ids");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let edge_ids_view = edge_ids.view().expect("edge_ids_view");

    let err = GraphBuilder::new(&src_view, &dst_view)
        .edge_ids(&edge_ids_view)
        .symmetrize()
        .build::<Forward>(&handle)
        .expect_err("must reject symmetrize plus edge ids");

    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert!(err.message().contains("symmetrize"));
    assert!(err.message().contains("edge_ids"));
    assert!(err.message().contains("symmetrize"));
}

fn builder_rejects_mismatched_temporal_dtypes() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let handle = test_handle();
    let handle = handle.enter("builder_rejects_mismatched_temporal_dtypes").expect("entered");
    let src = DeviceArray::from_slice(&handle, DIRECTED_TRIANGLE_CYCLE_SRC_I32).expect("src");
    let dst = DeviceArray::from_slice(&handle, DIRECTED_TRIANGLE_CYCLE_DST_I32).expect("dst");
    let start = DeviceArray::from_slice(&handle, TRIANGLE_VERTEX_IDS_I32).expect("start");
    let end = DeviceArray::from_slice(&handle, TRIANGLE_VERTEX_IDS_I64).expect("end");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let start_view = start.view().expect("start_view");
    let end_view = end.view().expect("end_view");

    let err = GraphBuilder::new(&src_view, &dst_view)
        .edge_start_time_ids(&start_view)
        .edge_end_time_ids(&end_view)
        .build::<Forward>(&handle)
        .expect_err("must reject mismatched temporal dtypes");

    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert!(err.message().contains("same dtype"));
}

fn builder_rejects_vertices_dtype_mismatch_with_src_dst() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let handle = test_handle();
    let handle = handle.enter("builder_rejects_vertices_dtype_mismatch_with_src_dst").expect("entered");
    let src = DeviceArray::from_slice(&handle, DIRECTED_TRIANGLE_CYCLE_SRC_I32).expect("src");
    let dst = DeviceArray::from_slice(&handle, DIRECTED_TRIANGLE_CYCLE_DST_I32).expect("dst");
    // vertices are i64 but src/dst are i32; upstream would truncate without an error
    let verts = DeviceArray::from_slice(&handle, TRIANGLE_VERTEX_IDS_I64).expect("verts");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let verts_view = verts.view().expect("verts_view");

    let err = GraphBuilder::new(&src_view, &dst_view)
        .vertices(&verts_view)
        .build::<Forward>(&handle)
        .expect_err("must reject vertices dtype != src/dst dtype");

    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert!(err.message().contains("vertices"));
    assert!(err.message().contains("same dtype"));
}

fn builder_accepts_vertices_matching_src_dst_dtype() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let handle = test_handle();
    let handle = handle.enter("builder_accepts_vertices_matching_src_dst_dtype").expect("entered");
    let src = DeviceArray::from_slice(&handle, DIRECTED_TRIANGLE_CYCLE_SRC_I32).expect("src");
    let dst = DeviceArray::from_slice(&handle, DIRECTED_TRIANGLE_CYCLE_DST_I32).expect("dst");
    let verts = DeviceArray::from_slice(&handle, TRIANGLE_VERTEX_IDS_I32).expect("verts");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let verts_view = verts.view().expect("verts_view");

    // Same dtype (i32); should succeed
    GraphBuilder::new(&src_view, &dst_view)
        .vertices(&verts_view)
        .renumber()
        .build::<Forward>(&handle)
        .expect("vertices with matching dtype should build");
}

fn builder_chainable_setters_preserve_build_path() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let handle = test_handle();
    let handle = handle.enter("builder_chainable_setters_preserve_build_path").expect("entered");
    let src = DeviceArray::from_slice(&handle, TWO_CYCLE_SRC_I32).expect("src");
    let dst = DeviceArray::from_slice(&handle, TWO_CYCLE_DST_I32).expect("dst");
    let wgt = DeviceArray::from_slice(&handle, &[1.0f32, 2.0]).expect("wgt");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let wgt_view = wgt.view().expect("wgt_view");

    let graph = GraphBuilder::new(&src_view, &dst_view)
        .properties(GraphProperties::directed())
        .weights(&wgt_view)
        .renumber()
        .drop_self_loops()
        .drop_multi_edges()
        .do_expensive_check()
        .build::<Forward>(&handle)
        .expect("graph");

    assert_eq!(graph.vertex_dtype(), DataType::Int32);
}

fn graph_properties_are_preserved_on_built_graph() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let handle = test_handle();
    let handle = handle.enter("graph_properties_are_preserved_on_built_graph").expect("entered");
    let src = DeviceArray::from_slice(&handle, MULTI_EDGE_DROP_SRC_I32).expect("src");
    let dst = DeviceArray::from_slice(&handle, MULTI_EDGE_DROP_DST_I32).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let properties = GraphProperties { is_symmetric: true, is_multigraph: true };

    let graph =
        GraphBuilder::new(&src_view, &dst_view).properties(properties).build::<Forward>(&handle).expect("graph");

    assert_eq!(graph.properties(), properties);
}

fn drop_self_loops_removes_loop_degree_contribution() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let handle = test_handle();
    let handle = handle.enter("drop_self_loops_removes_loop_degree_contribution").expect("entered");
    let src = DeviceArray::from_slice(&handle, SELF_LOOP_DROP_SRC_I32).expect("src");
    let dst = DeviceArray::from_slice(&handle, SELF_LOOP_DROP_DST_I32).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .properties(GraphProperties { is_symmetric: false, is_multigraph: true })
        .drop_self_loops()
        .build::<Transposed>(&handle)
        .expect("graph");

    let degrees = degree_map(&mut graph, &handle, &[0, 1, 2]);

    assert_eq!(degrees[&0], (0, 1));
    assert_eq!(degrees[&1], (1, 1));
    assert_eq!(degrees[&2], (1, 0));
}

fn drop_multi_edges_collapses_duplicate_edges() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let handle = test_handle();
    let handle = handle.enter("drop_multi_edges_collapses_duplicate_edges").expect("entered");
    let src = DeviceArray::from_slice(&handle, MULTI_EDGE_DROP_SRC_I32).expect("src");
    let dst = DeviceArray::from_slice(&handle, MULTI_EDGE_DROP_DST_I32).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .properties(GraphProperties { is_symmetric: false, is_multigraph: true })
        .drop_multi_edges()
        .build::<Transposed>(&handle)
        .expect("graph");

    let degrees = degree_map(&mut graph, &handle, &[0, 1, 2]);

    assert_eq!(degrees[&0], (0, 1));
    assert_eq!(degrees[&1], (1, 1));
    assert_eq!(degrees[&2], (1, 0));
}

fn edge_count_matches_directed_input_rows() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let handle = test_handle();
    let handle = handle.enter("edge_count_matches_directed_input_rows").expect("entered");
    let src = DeviceArray::from_slice(&handle, TRIANGLE_TAIL_SRC_I32).expect("src");
    let dst = DeviceArray::from_slice(&handle, TRIANGLE_TAIL_DST_I32).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view).build::<Forward>(&handle).expect("graph");

    assert_eq!(graph.edge_count(&handle).expect("edge_count"), TRIANGLE_TAIL_SRC_I32.len());

    // SAFETY: the entered runtime retains this native handle and its exact stream/MR binding
    // throughout the borrowed wrapper's use on the current thread.
    let borrowed =
        unsafe { ResourceHandle::from_raw_resource_handle(handle.handle_ptr()) }.expect("borrow external handle");
    let entered = borrowed.enter("external_edge_count").expect("entered external handle");
    let src = DeviceArray::from_slice(&entered, &[0i64, 0, 1, 1]).expect("external sources");
    let dst = DeviceArray::from_slice(&entered, &[1i64, 1, 0, 1]).expect("external destinations");
    let (src_view, dst_view) = (src.view().expect("sources view"), dst.view().expect("destinations view"));
    let mut graph = GraphBuilder::new(&src_view, &dst_view).build::<Forward>(&entered).expect("external graph");
    assert_eq!(graph.edge_count(&entered).expect("external logical edge count"), 3);
}

fn edge_count_respects_drop_self_loops() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let handle = test_handle();
    let handle = handle.enter("edge_count_respects_drop_self_loops").expect("entered");
    let src = DeviceArray::from_slice(&handle, SELF_LOOP_DROP_SRC_I32).expect("src");
    let dst = DeviceArray::from_slice(&handle, SELF_LOOP_DROP_DST_I32).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .properties(GraphProperties { is_symmetric: false, is_multigraph: true })
        .drop_self_loops()
        .build::<Forward>(&handle)
        .expect("graph");

    assert_eq!(graph.edge_count(&handle).expect("edge_count"), 2);
}

fn edge_count_respects_drop_multi_edges() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let handle = test_handle();
    let handle = handle.enter("edge_count_respects_drop_multi_edges").expect("entered");
    let src = DeviceArray::from_slice(&handle, MULTI_EDGE_DROP_SRC_I32).expect("src");
    let dst = DeviceArray::from_slice(&handle, MULTI_EDGE_DROP_DST_I32).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .properties(GraphProperties { is_symmetric: false, is_multigraph: true })
        .drop_multi_edges()
        .build::<Forward>(&handle)
        .expect("graph");

    assert_eq!(graph.edge_count(&handle).expect("edge_count"), 2);
}

fn edge_count_matches_python_style_undirected_edges() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let handle = test_handle();
    let handle = handle.enter("edge_count_matches_python_style_undirected_edges").expect("entered");
    let src = DeviceArray::from_slice(&handle, UNDIRECTED_TRIANGLE_SRC_I32).expect("src");
    let dst = DeviceArray::from_slice(&handle, UNDIRECTED_TRIANGLE_DST_I32).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .properties(GraphProperties::undirected())
        .build::<Forward>(&handle)
        .expect("graph");

    assert_eq!(graph.edge_count(&handle).expect("edge_count"), 3);
}

fn edge_count_counts_undirected_self_loops_once() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let handle = test_handle();
    let handle = handle.enter("edge_count_counts_undirected_self_loops_once").expect("entered");
    let src = DeviceArray::from_slice(&handle, &[0i32, 0, 1]).expect("src");
    let dst = DeviceArray::from_slice(&handle, &[0i32, 1, 0]).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .properties(GraphProperties::undirected())
        .build::<Forward>(&handle)
        .expect("graph");

    assert_eq!(graph.edge_count(&handle).expect("edge_count"), 2);

    let src = DeviceArray::from_slice(&handle, &[0i64, 0, 0, 1, 0, 1]).expect("src");
    let dst = DeviceArray::from_slice(&handle, &[0i64, 0, 1, 0, 1, 0]).expect("dst");
    let (src_view, dst_view) = (src.view().expect("src_view"), dst.view().expect("dst_view"));
    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .properties(GraphProperties { is_symmetric: true, is_multigraph: true })
        .build::<Forward>(&handle)
        .expect("symmetric multigraph");
    assert_eq!(graph.edge_count(&handle).expect("multigraph edge count"), 4);

    let src = DeviceArray::from_slice(&handle, &[0i64, 0, 1]).expect("src");
    let dst = DeviceArray::from_slice(&handle, &[1i64, 1, 0]).expect("dst");
    let (src_view, dst_view) = (src.view().expect("src_view"), dst.view().expect("dst_view"));
    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .properties(GraphProperties { is_symmetric: true, is_multigraph: true })
        .build::<Forward>(&handle)
        .expect("unchecked odd-multiplicity graph");
    let error = graph.edge_count(&handle).expect_err("odd normalized multiplicity");
    assert_eq!(error.kind(), ErrorKind::InvalidState);
    assert_eq!(error.context_value("edge"), Some("(0, 1)"));
    assert_eq!(error.context_value("run_len"), Some("3"));
}

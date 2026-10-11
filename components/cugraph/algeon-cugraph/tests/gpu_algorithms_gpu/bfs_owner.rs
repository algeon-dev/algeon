use cudf::testing::run_named_cases;
use cugraph::{
    array::DeviceArray,
    error::ErrorKind,
    graph::{Forward, GraphBuilder},
    handle::EnteredContext,
    params::BfsDepthLimit,
    traversal::{BfsParams, BfsPredicateParams, BfsResult},
};

use crate::test_handle;

fn bfs_distance_map(handle: &EnteredContext<'_>, result: &BfsResult) -> std::collections::HashMap<i32, i32> {
    let vertices = result.vertex_ids().expect("vertex_ids").to_vec_i32(handle).expect("vertices");
    let distances = result.distances().expect("distances").to_vec_i32(handle).expect("distances");
    vertices.into_iter().zip(distances).collect()
}

#[test]
fn gpu_cases_1() {
    run_named_cases(&[
        ("traversal::bfs::tests::test_bfs_directed_graph", test_bfs_directed_graph),
        ("traversal::bfs::tests::test_bfs_depth_limit_works", test_bfs_depth_limit_works),
        (
            "traversal::bfs::tests::test_bfs_depth_limit_zero_reaches_only_the_sources",
            test_bfs_depth_limit_zero_reaches_only_the_sources,
        ),
        (
            "traversal::bfs::tests::test_bfs_depth_limit_beyond_the_vertex_type_is_unlimited",
            test_bfs_depth_limit_beyond_the_vertex_type_is_unlimited,
        ),
        ("traversal::bfs::tests::test_bfs_with_excluded_vertices", test_bfs_with_excluded_vertices),
        ("traversal::bfs::tests::test_bfs_with_included_vertices", test_bfs_with_included_vertices),
        (
            "traversal::bfs::tests::test_bfs_rejects_include_and_exclude_vertices_together",
            test_bfs_rejects_include_and_exclude_vertices_together,
        ),
        ("traversal::bfs::tests::test_bfs_with_include_edge_ids", test_bfs_with_include_edge_ids),
        (
            "traversal::bfs::tests::test_bfs_include_edge_ids_rejects_graph_without_edge_ids",
            test_bfs_include_edge_ids_rejects_graph_without_edge_ids,
        ),
        (
            "traversal::bfs::tests::test_bfs_include_edge_ids_rejects_mismatched_dtype",
            test_bfs_include_edge_ids_rejects_mismatched_dtype,
        ),
        ("traversal::bfs::tests::test_bfs_target_vertices_return_metadata", test_bfs_target_vertices_return_metadata),
    ]);
}

fn test_bfs_directed_graph() {
    let handle = test_handle();
    let entered = handle.enter("test_bfs_directed_graph").expect("entered");

    // Directed graph matching the C API test pattern:
    // 0->1, 1->3, 1->4, 2->0, 2->1, 2->3, 3->5, 4->5
    let src_data: Vec<i32> = vec![0, 1, 1, 2, 2, 2, 3, 4];
    let dst_data: Vec<i32> = vec![1, 3, 4, 0, 1, 3, 5, 5];
    let wgt_data: Vec<f32> = vec![0.1, 2.1, 1.1, 5.1, 3.1, 4.1, 7.2, 3.2];

    let src = DeviceArray::from_slice(&entered, &src_data).expect("src");
    let dst = DeviceArray::from_slice(&entered, &dst_data).expect("dst");
    let wgt = DeviceArray::from_slice(&entered, &wgt_data).expect("wgt");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let wgt_view = wgt.view().expect("wgt_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .weights(&wgt_view)
        .renumber()
        .build::<Forward>(&entered)
        .expect("graph");

    // BFS from vertex 0
    let sources_arr = DeviceArray::from_slice(&entered, &[0i32]).expect("sources");
    let sources_view = sources_arr.view().expect("sources_view");

    let params = BfsParams::default().with_predecessors();
    let result = graph.bfs(&entered, &sources_view, &params).expect("bfs");

    let vids = result.vertex_ids().expect("vertex_ids");
    let dists = result.distances().expect("distances");
    assert_eq!(vids.len(), 6, "should have 6 vertices");
    assert_eq!(dists.len(), 6, "should have 6 distances");

    let dist_map = bfs_distance_map(&entered, &result);

    // Expected: {0: 0, 1: 1, 2: unreachable, 3: 2, 4: 2, 5: 3}
    assert_eq!(dist_map[&0], 0, "source should have distance 0");
    assert_eq!(dist_map[&1], 1, "vertex 1 should have distance 1");
    assert_eq!(dist_map[&2], i32::MAX, "vertex 2 should be unreachable from 0");
    assert_eq!(dist_map[&3], 2, "vertex 3 should have distance 2");
    assert_eq!(dist_map[&4], 2, "vertex 4 should have distance 2");
    assert_eq!(dist_map[&5], 3, "vertex 5 should have distance 3");

    let preds = result.predecessors().expect("predecessors");
    assert!(preds.is_some(), "predecessors should be computed");
}

fn test_bfs_depth_limit_works() {
    let handle = test_handle();
    let entered = handle.enter("test_bfs_depth_limit_works").expect("entered");

    // Path graph 0->1->2->3->4
    let src_data: Vec<i32> = vec![0, 1, 2, 3];
    let dst_data: Vec<i32> = vec![1, 2, 3, 4];

    let src = DeviceArray::from_slice(&entered, &src_data).expect("src");
    let dst = DeviceArray::from_slice(&entered, &dst_data).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view).build::<Forward>(&entered).expect("graph");

    let sources_arr = DeviceArray::from_slice(&entered, &[0i32]).expect("sources");
    let sources_view = sources_arr.view().expect("sources_view");

    let params = BfsParams::default().with_depth_limit(BfsDepthLimit::new(2));
    let result = graph.bfs(&entered, &sources_view, &params).expect("bfs");

    let dist_map = bfs_distance_map(&entered, &result);

    assert_eq!(dist_map[&0], 0);
    assert_eq!(dist_map[&1], 1);
    assert_eq!(dist_map[&2], 2);
    assert_eq!(dist_map[&3], i32::MAX);
    assert_eq!(dist_map[&4], i32::MAX);
}

fn test_bfs_depth_limit_zero_reaches_only_the_sources() {
    let handle = test_handle();
    let entered = handle.enter("test_bfs_depth_limit_zero_reaches_only_the_sources").expect("entered");

    // Path graph 0->1->2
    let src = DeviceArray::from_slice(&entered, &[0i32, 1]).expect("src");
    let dst = DeviceArray::from_slice(&entered, &[1i32, 2]).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let mut graph = GraphBuilder::new(&src_view, &dst_view).build::<Forward>(&entered).expect("graph");
    let sources_arr = DeviceArray::from_slice(&entered, &[0i32]).expect("sources");
    let sources_view = sources_arr.view().expect("sources_view");

    let params = BfsParams::default().with_depth_limit(BfsDepthLimit::new(0));
    let result = graph.bfs(&entered, &sources_view, &params).expect("bfs");
    let dist_map = bfs_distance_map(&entered, &result);

    assert_eq!(dist_map[&0], 0);
    assert_eq!(dist_map[&1], i32::MAX);
    assert_eq!(dist_map[&2], i32::MAX);
}

fn test_bfs_depth_limit_beyond_the_vertex_type_is_unlimited() {
    let handle = test_handle();
    let entered = handle.enter("test_bfs_depth_limit_beyond_the_vertex_type_is_unlimited").expect("entered");

    let src = DeviceArray::from_slice(&entered, &[0i32, 1, 2]).expect("src");
    let dst = DeviceArray::from_slice(&entered, &[1i32, 2, 3]).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let mut graph = GraphBuilder::new(&src_view, &dst_view).build::<Forward>(&entered).expect("graph");
    let sources_arr = DeviceArray::from_slice(&entered, &[0i32]).expect("sources");
    let sources_view = sources_arr.view().expect("sources_view");

    // 2^32 truncates to 0 as an Int32 depth limit, which would reach no vertex but the source.
    let params = BfsParams::default().with_depth_limit(BfsDepthLimit::new(1usize << 32));
    let result = graph.bfs(&entered, &sources_view, &params).expect("bfs");
    let dist_map = bfs_distance_map(&entered, &result);

    assert_eq!(dist_map[&3], 3);
}

fn test_bfs_with_excluded_vertices() {
    let handle = test_handle();
    let entered = handle.enter("test_bfs_with_excluded_vertices").expect("entered");

    let src_data = vec![0i32, 0, 1, 3];
    let dst_data = vec![1i32, 3, 2, 2];
    let src = DeviceArray::from_slice(&entered, &src_data).expect("src");
    let dst = DeviceArray::from_slice(&entered, &dst_data).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let mut graph = GraphBuilder::new(&src_view, &dst_view).build::<Forward>(&entered).expect("graph");

    let sources = DeviceArray::from_slice(&entered, &[0i32]).expect("sources");
    let excluded = DeviceArray::from_slice(&entered, &[1i32]).expect("excluded");
    let sources_view = sources.view().expect("sources_view");
    let excluded_view = excluded.view().expect("excluded_view");
    let predicates: BfsPredicateParams<'_, i32> =
        BfsPredicateParams { exclude_vertices: Some(&excluded_view), ..BfsPredicateParams::default() };

    let result = graph.bfs_with_predicates(&entered, &sources_view, &BfsParams::default(), &predicates).expect("bfs");
    let dist_map = bfs_distance_map(&entered, &result);

    assert_eq!(dist_map[&0], 0);
    assert_eq!(dist_map[&1], i32::MAX);
    assert_eq!(dist_map[&3], 1);
    assert_eq!(dist_map[&2], 2);
}

fn test_bfs_with_included_vertices() {
    let handle = test_handle();
    let entered = handle.enter("test_bfs_with_included_vertices").expect("entered");

    let src_data = vec![0i32, 0, 1, 3];
    let dst_data = vec![1i32, 3, 2, 2];
    let src = DeviceArray::from_slice(&entered, &src_data).expect("src");
    let dst = DeviceArray::from_slice(&entered, &dst_data).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let mut graph = GraphBuilder::new(&src_view, &dst_view).build::<Forward>(&entered).expect("graph");

    let sources = DeviceArray::from_slice(&entered, &[0i32]).expect("sources");
    let included = DeviceArray::from_slice(&entered, &[0i32, 1, 2]).expect("included");
    let sources_view = sources.view().expect("sources_view");
    let included_view = included.view().expect("included_view");
    let predicates: BfsPredicateParams<'_, i32> =
        BfsPredicateParams { include_vertices: Some(&included_view), ..BfsPredicateParams::default() };

    let result = graph.bfs_with_predicates(&entered, &sources_view, &BfsParams::default(), &predicates).expect("bfs");
    let dist_map = bfs_distance_map(&entered, &result);

    assert_eq!(dist_map[&0], 0);
    assert_eq!(dist_map[&1], 1);
    assert_eq!(dist_map[&3], i32::MAX);
    assert_eq!(dist_map[&2], 2);
}

fn test_bfs_rejects_include_and_exclude_vertices_together() {
    let handle = test_handle();
    let entered = handle.enter("test_bfs_rejects_include_and_exclude_vertices_together").expect("entered");

    let src = DeviceArray::from_slice(&entered, &[0i32, 1]).expect("src");
    let dst = DeviceArray::from_slice(&entered, &[1i32, 2]).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let mut graph = GraphBuilder::new(&src_view, &dst_view).build::<Forward>(&entered).expect("graph");

    let sources = DeviceArray::from_slice(&entered, &[0i32]).expect("sources");
    let included = DeviceArray::from_slice(&entered, &[0i32, 1]).expect("included");
    let excluded = DeviceArray::from_slice(&entered, &[1i32]).expect("excluded");
    let sources_view = sources.view().expect("sources_view");
    let included_view = included.view().expect("included_view");
    let excluded_view = excluded.view().expect("excluded_view");
    let predicates: BfsPredicateParams<'_, i32> = BfsPredicateParams {
        include_vertices: Some(&included_view),
        exclude_vertices: Some(&excluded_view),
        ..BfsPredicateParams::default()
    };

    let err = graph
        .bfs_with_predicates(&entered, &sources_view, &BfsParams::default(), &predicates)
        .expect_err("include/exclude vertex predicates are mutually exclusive");
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.operation(), "Graph::bfs_with_predicates");
    assert!(err.message().contains("include_vertices and exclude_vertices are mutually exclusive"));
}

fn test_bfs_with_include_edge_ids() {
    let handle = test_handle();
    let entered = handle.enter("test_bfs_with_include_edge_ids").expect("entered");

    let src_data = vec![0i32, 0, 1, 2];
    let dst_data = vec![1i32, 2, 3, 3];
    let edge_id_data = vec![10i32, 11, 12, 13];
    let src = DeviceArray::from_slice(&entered, &src_data).expect("src");
    let dst = DeviceArray::from_slice(&entered, &dst_data).expect("dst");
    let edge_ids = DeviceArray::from_slice(&entered, &edge_id_data).expect("edge_ids");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let edge_ids_view = edge_ids.view().expect("edge_ids_view");
    let mut graph =
        GraphBuilder::new(&src_view, &dst_view).edge_ids(&edge_ids_view).build::<Forward>(&entered).expect("graph");

    let sources = DeviceArray::from_slice(&entered, &[0i32]).expect("sources");
    let allowed_edge_ids = DeviceArray::from_slice(&entered, &[11i32, 13]).expect("allowed_edge_ids");
    let sources_view = sources.view().expect("sources_view");
    let allowed_edge_ids_view = allowed_edge_ids.view().expect("allowed_edge_ids_view");
    let predicates =
        BfsPredicateParams { include_edge_ids: Some(&allowed_edge_ids_view), ..BfsPredicateParams::default() };

    let result = graph.bfs_with_predicates(&entered, &sources_view, &BfsParams::default(), &predicates).expect("bfs");
    let dist_map = bfs_distance_map(&entered, &result);

    assert_eq!(dist_map[&0], 0);
    assert_eq!(dist_map[&1], i32::MAX);
    assert_eq!(dist_map[&2], 1);
    assert_eq!(dist_map[&3], 2);
}

fn test_bfs_include_edge_ids_rejects_graph_without_edge_ids() {
    let handle = test_handle();
    let entered = handle.enter("test_bfs_include_edge_ids_rejects_graph_without_edge_ids").expect("entered");

    let src = DeviceArray::from_slice(&entered, &[0i32, 1]).expect("src");
    let dst = DeviceArray::from_slice(&entered, &[1i32, 2]).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let mut graph = GraphBuilder::new(&src_view, &dst_view).build::<Forward>(&entered).expect("graph");

    let sources = DeviceArray::from_slice(&entered, &[0i32]).expect("sources");
    let allowed_edge_ids = DeviceArray::from_slice(&entered, &[0i32]).expect("allowed_edge_ids");
    let sources_view = sources.view().expect("sources_view");
    let allowed_edge_ids_view = allowed_edge_ids.view().expect("allowed_edge_ids_view");
    let predicates =
        BfsPredicateParams { include_edge_ids: Some(&allowed_edge_ids_view), ..BfsPredicateParams::default() };

    let err = graph
        .bfs_with_predicates(&entered, &sources_view, &BfsParams::default(), &predicates)
        .expect_err("edge-id predicates require graph edge ids");
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.operation(), "Graph::bfs_with_predicates");
    assert!(err.message().contains("include_edge_ids requires a graph with edge IDs"));
}

fn test_bfs_include_edge_ids_rejects_mismatched_dtype() {
    let handle = test_handle();
    let entered = handle.enter("test_bfs_include_edge_ids_rejects_mismatched_dtype").expect("entered");

    let src = DeviceArray::from_slice(&entered, &[0i32, 1]).expect("src");
    let dst = DeviceArray::from_slice(&entered, &[1i32, 2]).expect("dst");
    let edge_ids = DeviceArray::from_slice(&entered, &[10i32, 11]).expect("edge_ids");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let edge_ids_view = edge_ids.view().expect("edge_ids_view");
    let mut graph =
        GraphBuilder::new(&src_view, &dst_view).edge_ids(&edge_ids_view).build::<Forward>(&entered).expect("graph");

    let sources = DeviceArray::from_slice(&entered, &[0i32]).expect("sources");
    let allowed_edge_ids = DeviceArray::from_slice(&entered, &[10i64]).expect("allowed_edge_ids");
    let sources_view = sources.view().expect("sources_view");
    let allowed_edge_ids_view = allowed_edge_ids.view().expect("allowed_edge_ids_view");
    let predicates =
        BfsPredicateParams { include_edge_ids: Some(&allowed_edge_ids_view), ..BfsPredicateParams::default() };

    let err = graph
        .bfs_with_predicates(&entered, &sources_view, &BfsParams::default(), &predicates)
        .expect_err("include_edge_ids dtype should match graph edge-id dtype");
    assert_eq!(err.kind(), ErrorKind::UnsupportedType);
    assert_eq!(err.operation(), "Graph::bfs_with_predicates");
    assert!(err.message().contains("include_edge_ids dtype Int64 does not match graph edge-id dtype Int32"));
}

fn test_bfs_target_vertices_return_metadata() {
    let handle = test_handle();
    let entered = handle.enter("test_bfs_target_vertices_return_metadata").expect("entered");

    let src = DeviceArray::from_slice(&entered, &[0i32, 1, 2]).expect("src");
    let dst = DeviceArray::from_slice(&entered, &[1i32, 2, 3]).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let mut graph = GraphBuilder::new(&src_view, &dst_view).build::<Forward>(&entered).expect("graph");

    let sources = DeviceArray::from_slice(&entered, &[0i32]).expect("sources");
    let targets = DeviceArray::from_slice(&entered, &[2i32]).expect("targets");
    let sources_view = sources.view().expect("sources_view");
    let targets_view = targets.view().expect("targets_view");
    let predicates: BfsPredicateParams<'_, i32> = BfsPredicateParams {
        target_vertices: Some(&targets_view),
        stop_when_any_target_found: true,
        ..BfsPredicateParams::default()
    };

    let result = graph.bfs_with_predicates(&entered, &sources_view, &BfsParams::default(), &predicates).expect("bfs");
    let target_info = result.target_info().expect("target metadata should be returned");

    assert!(target_info.target_found);
    assert_eq!(target_info.target_distance, Some(2));
    let dist_map = bfs_distance_map(&entered, &result);
    assert_eq!(dist_map[&3], i32::MAX, "early stop should avoid depth after target discovery");
}

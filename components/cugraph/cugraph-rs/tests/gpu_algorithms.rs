mod common;

use std::{collections::HashMap, num::NonZeroUsize};

use common::{
    cudf::explicit_runtime,
    support::{
        DEGREE_QUERY_I64, DEGREE_SUBSET_DST_I64, DEGREE_SUBSET_SRC_I64, DIRECTED_STAR_DST_I64, DIRECTED_STAR_SRC_I64,
        DISCONNECTED_TWO_CYCLES_DST_I32, DISCONNECTED_TWO_CYCLES_SRC_I32, FOUR_VERTEX_CHAIN_DST_I32,
        FOUR_VERTEX_CHAIN_DST_I64, FOUR_VERTEX_CHAIN_SRC_I32, FOUR_VERTEX_CHAIN_SRC_I64, LOUVAIN_GRAPH_DST_I32,
        LOUVAIN_GRAPH_SRC_I32, LOUVAIN_GRAPH_WEIGHT_F32, SSSP_GRAPH_DST_I32, SSSP_GRAPH_SRC_I32, SSSP_GRAPH_WEIGHT_F32,
        SSSP_GRAPH_WEIGHT_F64, TEST_LAYOUT_SECONDARY_SEED, TEST_LAYOUT_SMOKE_ITERATIONS, TRIANGLE_VERTEX_IDS_I64,
        UNDIRECTED_TRIANGLE_DST_I32, UNDIRECTED_TRIANGLE_DST_I64, UNDIRECTED_TRIANGLE_SRC_I32,
        UNDIRECTED_TRIANGLE_SRC_I64, UNDIRECTED_TRIANGLE_UNIT_WEIGHT_F64, bfs_distance_map_i64_result,
    },
};
use cugraph::{
    DeviceArray, Forward, GraphBuilder, ResourceHandle, Transposed, VertexSelection,
    centrality::{BetweennessCentralityParams, PageRankParams},
    community::{ExtractEgoParams, KTrussSubgraphParams, LouvainParams, TriangleCountParams},
    labeling::ConnectedComponentsParams,
    layout::{ForceAtlas2Params, ForceAtlas2VertexAttrs, InitialPositions},
    params::{IterationCount, NonNegativeF64},
    random::RngState,
    structure::{CoreNumberParams, Distances, IntegerValues, KCoreDegreeType, KCoreParams, ScoreValues, VertexIds},
    traversal::{BfsParams, SsspParams},
    tree::MinimumSpanningTreeParams,
    types::GraphProperties,
};

fn test_handle() -> ResourceHandle {
    let runtime = std::sync::Arc::new(explicit_runtime());
    ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).expect("handle")
}

#[test]
fn bfs_accepts_i64_sources_for_i64_graphs() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let handle = test_handle();
    let entered = handle.enter("bfs_accepts_i64_sources_for_i64_graphs").expect("entered");
    let src = DeviceArray::from_slice(&entered, FOUR_VERTEX_CHAIN_SRC_I64).expect("src");
    let dst = DeviceArray::from_slice(&entered, FOUR_VERTEX_CHAIN_DST_I64).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view).renumber().build::<Forward>(&entered).expect("graph");
    let sources = DeviceArray::from_slice(&entered, &[0i64]).expect("sources");
    let sources_view = sources.view().expect("sources_view");

    let result = graph.bfs(&entered, &sources_view, &BfsParams::default().with_predecessors()).expect("bfs");

    let vertex_ids = result.vertex_ids().expect("vertex_ids");
    assert!(matches!(vertex_ids, VertexIds::I64(_)));
    let distances = result.distances().expect("distances");
    assert!(matches!(distances, Distances::I64(_)));
    let predecessors = result.predecessors().expect("predecessors").expect("predecessor values");
    assert!(matches!(predecessors, VertexIds::I64(_)));

    let distance_map = bfs_distance_map_i64_result(&entered, &result);

    assert_eq!(distance_map[&0], 0);
    assert_eq!(distance_map[&1], 1);
    assert_eq!(distance_map[&2], 2);
    assert_eq!(distance_map[&3], 3);
}

#[test]
fn betweenness_accepts_i64_seed_vertices() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let handle = test_handle();
    let entered = handle.enter("betweenness_accepts_i64_seed_vertices").expect("entered");
    let src = DeviceArray::from_slice(&entered, DIRECTED_STAR_SRC_I64).expect("src");
    let dst = DeviceArray::from_slice(&entered, DIRECTED_STAR_DST_I64).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view).renumber().build::<Forward>(&entered).expect("graph");
    let seeds = DeviceArray::from_slice(&entered, &[0i64, 1]).expect("seeds");
    let seeds_view = seeds.view().expect("seeds_view");

    let result = graph
        .betweenness_centrality(&entered, &seeds_view, &BetweennessCentralityParams::default())
        .expect("betweenness");

    let vertex_ids = result.vertex_ids().expect("vertex_ids");
    assert!(matches!(vertex_ids, VertexIds::I64(_)));
    assert_eq!(vertex_ids.len(), 4);

    let scores = result.scores().expect("scores");
    let values = scores.to_vec_f64(&entered).expect("score copy");
    assert_eq!(values.len(), 4);
    assert!(values.iter().all(|v| v.is_finite() && *v >= 0.0));
}

#[test]
fn degrees_accept_i64_vertex_subsets() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let handle = test_handle();
    let entered = handle.enter("degrees_accept_i64_vertex_subsets").expect("entered");
    let src = DeviceArray::from_slice(&entered, DEGREE_SUBSET_SRC_I64).expect("src");
    let dst = DeviceArray::from_slice(&entered, DEGREE_SUBSET_DST_I64).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view).renumber().build::<Transposed>(&entered).expect("graph");
    let query = DeviceArray::from_slice(&entered, DEGREE_QUERY_I64).expect("query");
    let query_view = query.view().expect("query_view");

    let result = graph.degrees(&entered, &query_view).expect("degrees");

    let vertices = result.vertex_ids().expect("vertex_ids").to_vec_i64(&entered).expect("vertex copy");
    let in_degrees = result.in_degrees().expect("in_degrees").to_vec_i64(&entered).expect("in copy");
    let out_degrees = result.out_degrees().expect("out_degrees").to_vec_i64(&entered).expect("out copy");

    let degree_map: HashMap<i64, (i64, i64)> =
        vertices.into_iter().zip(in_degrees.into_iter().zip(out_degrees)).collect();

    assert_eq!(degree_map.len(), 2);
    assert_eq!(degree_map[&0], (1, 1));
    assert_eq!(degree_map[&2], (1, 2));
}

#[test]
fn in_degrees_checked_accepts_all_vertices() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let handle = test_handle();
    let entered = handle.enter("in_degrees_checked_accepts_all_vertices").expect("entered");
    let src = DeviceArray::from_slice(&entered, DIRECTED_STAR_SRC_I64).expect("src");
    let dst = DeviceArray::from_slice(&entered, DIRECTED_STAR_DST_I64).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view).renumber().build::<Transposed>(&entered).expect("graph");

    let result = graph.in_degrees_checked(&entered, VertexSelection::<i64>::All, true).expect("in_degrees_checked");

    let vertices = result.vertex_ids().expect("vertex_ids").to_vec_i64(&entered).expect("vertex copy");
    let in_degrees = result.in_degrees().expect("in_degrees").to_vec_i64(&entered).expect("in copy");
    let degree_map: HashMap<i64, i64> = vertices.into_iter().zip(in_degrees).collect();

    assert_eq!(degree_map[&0], 3);
    for v in [1_i64, 2, 3] {
        assert_eq!(degree_map[&v], 1, "leaf vertex {v} should have in-degree 1");
    }
}

#[test]
fn out_degrees_checked_accepts_all_vertices() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let handle = test_handle();
    let entered = handle.enter("out_degrees_checked_accepts_all_vertices").expect("entered");
    let src = DeviceArray::from_slice(&entered, DIRECTED_STAR_SRC_I64).expect("src");
    let dst = DeviceArray::from_slice(&entered, DIRECTED_STAR_DST_I64).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view).renumber().build::<Forward>(&entered).expect("graph");

    let result = graph.out_degrees_checked(&entered, VertexSelection::<i64>::All, true).expect("out_degrees_checked");

    let vertices = result.vertex_ids().expect("vertex_ids").to_vec_i64(&entered).expect("vertex copy");
    let out_degrees = result.out_degrees().expect("out_degrees").to_vec_i64(&entered).expect("out copy");
    let degree_map: HashMap<i64, i64> = vertices.into_iter().zip(out_degrees).collect();

    assert_eq!(degree_map[&0], 3);
    for v in [1_i64, 2, 3] {
        assert_eq!(degree_map[&v], 1, "leaf vertex {v} should have out-degree 1");
    }
}

#[test]
fn bfs_rejects_vertex_selection_all() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let handle = test_handle();
    let entered = handle.enter("bfs_rejects_vertex_selection_all").expect("entered");
    let src = DeviceArray::from_slice(&entered, FOUR_VERTEX_CHAIN_SRC_I32).expect("src");
    let dst = DeviceArray::from_slice(&entered, FOUR_VERTEX_CHAIN_DST_I32).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view).renumber().build::<Forward>(&entered).expect("graph");
    let err =
        graph.bfs(&entered, VertexSelection::<i32>::All, &BfsParams::default()).expect_err("bfs should reject All");

    assert_eq!(err.kind(), cugraph::ErrorKind::InvalidInput);
    assert_eq!(err.operation(), "Graph::bfs");
}

#[test]
fn sssp_weighted_graph_returns_float_distances_and_predecessors() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let handle = test_handle();
    let entered = handle.enter("sssp_weighted_graph_returns_float_distances_and_predecessors").expect("entered");
    let src = DeviceArray::from_slice(&entered, SSSP_GRAPH_SRC_I32).expect("src");
    let dst = DeviceArray::from_slice(&entered, SSSP_GRAPH_DST_I32).expect("dst");
    let wgt = DeviceArray::from_slice(&entered, SSSP_GRAPH_WEIGHT_F32).expect("wgt");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let wgt_view = wgt.view().expect("wgt_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .weights(&wgt_view)
        .renumber()
        .build::<Forward>(&entered)
        .expect("graph");

    let params = SsspParams::default().with_cutoff(NonNegativeF64::new(10.0).expect("cutoff")).with_predecessors();
    let result = graph.sssp(&entered, 0i32, &params).expect("sssp");

    assert!(matches!(result.distances().expect("distances"), Distances::F32(_)));
    let vertices = result.vertex_ids().expect("vertex_ids").to_vec_i32(&entered).expect("vertices");
    let distances = result.distances().expect("distances").to_vec_f32(&entered).expect("distances");
    let predecessors = result
        .predecessors()
        .expect("predecessors")
        .expect("predecessors should exist")
        .to_vec_i32(&entered)
        .expect("predecessors");

    let distance_map: HashMap<i32, f32> = vertices.iter().copied().zip(distances).collect();
    let predecessor_map: HashMap<i32, i32> = vertices.into_iter().zip(predecessors).collect();

    assert!((distance_map[&0] - 0.0).abs() < 1.0e-5);
    assert!((distance_map[&1] - 0.1).abs() < 1.0e-5);
    assert!(distance_map[&2] > 1.0e30, "unreachable vertex should use f32::MAX sentinel");
    assert!((distance_map[&3] - 2.2).abs() < 1.0e-5);
    assert!((distance_map[&4] - 1.2).abs() < 1.0e-5);
    assert!((distance_map[&5] - 4.4).abs() < 1.0e-5);
    assert_eq!(predecessor_map[&0], -1);
    assert_eq!(predecessor_map[&1], 0);
    assert_eq!(predecessor_map[&2], -1);
    assert_eq!(predecessor_map[&3], 1);
    assert_eq!(predecessor_map[&4], 1);
    assert_eq!(predecessor_map[&5], 4);
}

#[test]
fn sssp_without_predecessors_omits_predecessor_column() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let handle = test_handle();
    let entered = handle.enter("sssp_without_predecessors_omits_predecessor_column").expect("entered");
    let src = DeviceArray::from_slice(&entered, SSSP_GRAPH_SRC_I32).expect("src");
    let dst = DeviceArray::from_slice(&entered, SSSP_GRAPH_DST_I32).expect("dst");
    let wgt = DeviceArray::from_slice(&entered, SSSP_GRAPH_WEIGHT_F64).expect("wgt");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let wgt_view = wgt.view().expect("wgt_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .weights(&wgt_view)
        .renumber()
        .build::<Forward>(&entered)
        .expect("graph");

    let result = graph.sssp(&entered, 0i32, &SsspParams::default()).expect("sssp");

    assert!(matches!(result.distances().expect("distances"), Distances::F64(_)));
    assert!(result.predecessors().expect("predecessors").is_none());
}

#[test]
fn extract_paths_from_bfs_and_sssp_results() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let handle = test_handle();
    let entered = handle.enter("extract_paths_from_bfs_and_sssp_results").expect("entered");
    let src = DeviceArray::from_slice(&entered, SSSP_GRAPH_SRC_I32).expect("src");
    let dst = DeviceArray::from_slice(&entered, SSSP_GRAPH_DST_I32).expect("dst");
    let wgt = DeviceArray::from_slice(&entered, SSSP_GRAPH_WEIGHT_F32).expect("wgt");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let wgt_view = wgt.view().expect("wgt_view");

    let mut graph =
        GraphBuilder::new(&src_view, &dst_view).weights(&wgt_view).build::<Forward>(&entered).expect("graph");
    let sources = DeviceArray::from_slice(&entered, &[0i32]).expect("sources");
    let destinations = DeviceArray::from_slice(&entered, &[5i32]).expect("destinations");
    let sources_view = sources.view().expect("sources_view");
    let destinations_view = destinations.view().expect("destinations_view");

    let bfs_result = graph.bfs(&entered, &sources_view, &BfsParams::default().with_predecessors()).expect("bfs");
    let bfs_paths =
        bfs_result.extract_paths(&entered, &mut graph, &sources_view, &destinations_view).expect("bfs extract paths");
    assert_eq!(bfs_paths.max_path_length(), 4);
    let bfs_path_values = bfs_paths.paths().expect("paths").to_vec_i32(&entered).expect("paths");
    assert_eq!(bfs_path_values, vec![0, 1, 3, 5]);

    let sssp_result = graph.sssp(&entered, 0i32, &SsspParams::default().with_predecessors()).expect("sssp");
    let sssp_paths =
        sssp_result.extract_paths(&entered, &mut graph, &sources_view, &destinations_view).expect("sssp extract paths");
    assert_eq!(sssp_paths.max_path_length(), 4);
    let sssp_path_values = sssp_paths.paths().expect("paths").to_vec_i32(&entered).expect("paths");
    assert_eq!(sssp_path_values, vec![0, 1, 4, 5]);
}

#[test]
fn extract_paths_rejects_results_without_predecessors() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let handle = test_handle();
    let entered = handle.enter("extract_paths_rejects_results_without_predecessors").expect("entered");
    let src = DeviceArray::from_slice(&entered, FOUR_VERTEX_CHAIN_SRC_I32).expect("src");
    let dst = DeviceArray::from_slice(&entered, FOUR_VERTEX_CHAIN_DST_I32).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view).build::<Forward>(&entered).expect("graph");
    let sources = DeviceArray::from_slice(&entered, &[0i32]).expect("sources");
    let destinations = DeviceArray::from_slice(&entered, &[3i32]).expect("destinations");
    let sources_view = sources.view().expect("sources_view");
    let destinations_view = destinations.view().expect("destinations_view");
    let result = graph.bfs(&entered, &sources_view, &BfsParams::default()).expect("bfs");

    let err = result
        .extract_paths(&entered, &mut graph, &sources_view, &destinations_view)
        .expect_err("extract paths should require predecessors");

    assert_eq!(err.kind(), cugraph::ErrorKind::InvalidInput);
    assert_eq!(err.operation(), "BfsResult::extract_paths");
}

#[test]
fn sssp_rejects_source_dtype_mismatch() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let handle = test_handle();
    let entered = handle.enter("sssp_rejects_source_dtype_mismatch").expect("entered");
    let src = DeviceArray::from_slice(&entered, FOUR_VERTEX_CHAIN_SRC_I32).expect("src");
    let dst = DeviceArray::from_slice(&entered, FOUR_VERTEX_CHAIN_DST_I32).expect("dst");
    let wgt = DeviceArray::from_slice(&entered, &[1.0f32, 1.0, 1.0]).expect("wgt");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let wgt_view = wgt.view().expect("wgt_view");

    let mut graph =
        GraphBuilder::new(&src_view, &dst_view).weights(&wgt_view).build::<Forward>(&entered).expect("graph");

    let err = graph.sssp(&entered, 0i64, &SsspParams::default()).expect_err("i64 source should not match i32 graph");

    assert_eq!(err.kind(), cugraph::ErrorKind::UnsupportedType);
    assert_eq!(err.operation(), "Graph::sssp");
}

#[test]
fn force_atlas2_accepts_i64_vertices_and_f32_helpers() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let handle = test_handle();
    let entered = handle.enter("force_atlas2_accepts_i64_vertices_and_f32_helpers").expect("entered");
    let mut rng = RngState::new(&entered, TEST_LAYOUT_SECONDARY_SEED).expect("rng");

    let src = DeviceArray::from_slice(&entered, UNDIRECTED_TRIANGLE_SRC_I64).expect("src");
    let dst = DeviceArray::from_slice(&entered, UNDIRECTED_TRIANGLE_DST_I64).expect("dst");
    let wgt = DeviceArray::from_slice(&entered, UNDIRECTED_TRIANGLE_UNIT_WEIGHT_F64).expect("wgt");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let wgt_view = wgt.view().expect("wgt_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .weights(&wgt_view)
        .properties(GraphProperties::undirected())
        .renumber()
        .build::<Forward>(&entered)
        .expect("graph");

    let pos_vertices = DeviceArray::from_slice(&entered, TRIANGLE_VERTEX_IDS_I64).expect("pos vertices");
    let x = DeviceArray::from_slice(&entered, &[0.0f32, 1.0, -1.0]).expect("x");
    let y = DeviceArray::from_slice(&entered, &[0.0f32, 0.5, -0.5]).expect("y");
    let mass_values = DeviceArray::from_slice(&entered, &[1.0f32, 1.5, 2.0]).expect("mass");
    let pos_vertices_view = pos_vertices.view().expect("pos_vertices_view");
    let x_view = x.view().expect("x_view");
    let y_view = y.view().expect("y_view");
    let mass_values_view = mass_values.view().expect("mass_values_view");

    let params = ForceAtlas2Params {
        max_iter: IterationCount::new(TEST_LAYOUT_SMOKE_ITERATIONS).expect("iter"),
        ..Default::default()
    };
    let init_pos = InitialPositions { vertices: &pos_vertices_view, x: &x_view, y: &y_view };
    let attrs = ForceAtlas2VertexAttrs { mass: Some((&pos_vertices_view, &mass_values_view)), ..Default::default() };

    let result =
        graph.force_atlas2_with(&entered, &mut rng, &params, Some(&attrs), Some(&init_pos)).expect("force_atlas2");

    let vertex_ids = result.vertex_ids().expect("vertex_ids");
    assert!(matches!(vertex_ids, VertexIds::I64(_)));

    let x_coords = result.x().expect("x");
    let y_coords = result.y().expect("y");
    assert!(matches!(x_coords, ScoreValues::F32(_)));
    assert!(matches!(y_coords, ScoreValues::F32(_)));

    let x_values = x_coords.to_vec_f32(&entered).expect("x copy");
    let y_values = y_coords.to_vec_f32(&entered).expect("y copy");
    assert_eq!(x_values.len(), 3);
    assert_eq!(y_values.len(), 3);
    assert!(x_values.iter().all(|v| v.is_finite()));
    assert!(y_values.iter().all(|v| v.is_finite()));
}

#[test]
fn force_atlas2_rejects_sparse_mass_helpers() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let handle = test_handle();
    let entered = handle.enter("force_atlas2_rejects_sparse_mass_helpers").expect("entered");
    let mut rng = RngState::new(&entered, TEST_LAYOUT_SECONDARY_SEED).expect("rng");

    let vertex_count = 128i32;
    let mut src_data = Vec::with_capacity(((vertex_count - 1) * 2) as usize);
    let mut dst_data = Vec::with_capacity(((vertex_count - 1) * 2) as usize);
    for v in 0..(vertex_count - 1) {
        src_data.push(v);
        dst_data.push(v + 1);
        src_data.push(v + 1);
        dst_data.push(v);
    }

    let src = DeviceArray::from_slice(&entered, &src_data).expect("src");
    let dst = DeviceArray::from_slice(&entered, &dst_data).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .properties(GraphProperties::undirected())
        .renumber()
        .build::<Forward>(&entered)
        .expect("graph");

    let sparse_mass_vertices = DeviceArray::from_slice(&entered, &[0i32, 1, 2]).expect("mass vertices");
    let sparse_mass_values = DeviceArray::from_slice(&entered, &[1.0f32, 2.0, 3.0]).expect("mass values");
    let sparse_mass_vertices_view = sparse_mass_vertices.view().expect("mass vertices view");
    let sparse_mass_values_view = sparse_mass_values.view().expect("mass values view");

    let params = ForceAtlas2Params { max_iter: IterationCount::new(1).expect("iter"), ..Default::default() };
    let attrs = ForceAtlas2VertexAttrs {
        mass: Some((&sparse_mass_vertices_view, &sparse_mass_values_view)),
        ..Default::default()
    };

    let err = graph
        .force_atlas2_with(&entered, &mut rng, &params, Some(&attrs), None)
        .expect_err("sparse mass helper should be rejected before reaching CUDA");

    assert_eq!(err.kind(), cugraph::ErrorKind::InvalidInput);
    assert_eq!(err.operation(), "force_atlas2");
    assert!(
        err.to_string().contains("vertex mass must provide one value for each graph vertex"),
        "unexpected error: {err}"
    );
}

#[test]
fn louvain_returns_vertex_partitions_and_finite_modularity() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let handle = test_handle();
    let entered = handle.enter("louvain_returns_vertex_partitions_and_finite_modularity").expect("entered");
    let src = DeviceArray::from_slice(&entered, LOUVAIN_GRAPH_SRC_I32).expect("src");
    let dst = DeviceArray::from_slice(&entered, LOUVAIN_GRAPH_DST_I32).expect("dst");
    let weights = DeviceArray::from_slice(&entered, LOUVAIN_GRAPH_WEIGHT_F32).expect("weights");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let weights_view = weights.view().expect("weights_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .weights(&weights_view)
        .properties(GraphProperties::undirected())
        .renumber()
        .build::<Forward>(&entered)
        .expect("graph");

    let result = graph.louvain(&entered, &LouvainParams::default()).expect("louvain");

    let vertex_ids = result.vertex_ids().expect("vertex_ids");
    assert!(matches!(vertex_ids, VertexIds::I32(_)));
    let clusters = result.clusters().expect("clusters");
    assert!(matches!(clusters, VertexIds::I32(_)));
    assert_eq!(vertex_ids.len(), clusters.len());
    assert!(!vertex_ids.is_empty());
    assert!(result.modularity().is_finite());
}

#[test]
fn weakly_connected_components_labels_two_symmetric_components() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let handle = test_handle();
    let entered = handle.enter("weakly_connected_components_labels_two_symmetric_components").expect("entered");
    let src = DeviceArray::from_slice(&entered, DISCONNECTED_TWO_CYCLES_SRC_I32).expect("src");
    let dst = DeviceArray::from_slice(&entered, DISCONNECTED_TWO_CYCLES_DST_I32).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .properties(GraphProperties::undirected())
        .renumber()
        .build::<Forward>(&entered)
        .expect("graph");

    let result = graph.weakly_connected_components(&entered, &ConnectedComponentsParams::default()).expect("wcc");

    let vertex_ids = result.vertex_ids().expect("vertex_ids");
    assert!(matches!(vertex_ids, VertexIds::I32(_)));
    let labels = result.labels().expect("labels");
    assert!(matches!(labels, IntegerValues::I32(_)));

    let vertices = vertex_ids.to_vec_i32(&entered).expect("vertices");
    let labels = labels.to_vec_i32(&entered).expect("labels");
    let label_map: HashMap<i32, i32> = vertices.into_iter().zip(labels).collect();

    assert_eq!(label_map[&0], label_map[&1]);
    assert_eq!(label_map[&2], label_map[&3]);
    assert_ne!(label_map[&0], label_map[&2]);
}

#[test]
fn strongly_connected_components_labels_directed_components() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let handle = test_handle();
    let entered = handle.enter("strongly_connected_components_labels_directed_components").expect("entered");
    let src = DeviceArray::from_slice(&entered, &[0i32, 1, 1, 2, 3]).expect("src");
    let dst = DeviceArray::from_slice(&entered, &[1i32, 0, 2, 3, 2]).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view).renumber().build::<Forward>(&entered).expect("graph");

    let result = graph.strongly_connected_components(&entered, &ConnectedComponentsParams::default()).expect("scc");

    let vertices = result.vertex_ids().expect("vertex_ids").to_vec_i32(&entered).expect("vertices");
    let labels = result.labels().expect("labels").to_vec_i32(&entered).expect("labels");
    let label_map: HashMap<i32, i32> = vertices.into_iter().zip(labels).collect();

    assert_eq!(label_map[&0], label_map[&1]);
    assert_eq!(label_map[&2], label_map[&3]);
    assert_ne!(label_map[&0], label_map[&2]);
}

#[test]
fn core_number_on_triangle_tail_returns_stable_core_values() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let handle = test_handle();
    let entered = handle.enter("core_number_on_triangle_tail_returns_stable_core_values").expect("entered");
    let src = DeviceArray::from_slice(&entered, &[0i32, 1, 2, 1, 2, 0, 2, 3]).expect("src");
    let dst = DeviceArray::from_slice(&entered, &[1i32, 2, 0, 0, 1, 2, 3, 2]).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .properties(GraphProperties::undirected())
        .renumber()
        .build::<Forward>(&entered)
        .expect("graph");

    let params = CoreNumberParams { degree_type: KCoreDegreeType::Out, do_expensive_check: false };
    let result = graph.core_number(&entered, &params).expect("core_number");

    let vertices = result.vertex_ids().expect("vertex_ids").to_vec_i32(&entered).expect("vertices");
    let core_numbers = result.core_numbers().expect("core_numbers").to_vec_i32(&entered).expect("core numbers");
    let core_map: HashMap<i32, i32> = vertices.into_iter().zip(core_numbers).collect();

    assert_eq!(core_map[&0], 2);
    assert_eq!(core_map[&1], 2);
    assert_eq!(core_map[&2], 2);
    assert_eq!(core_map[&3], 1);
}

#[test]
fn triangle_count_all_and_start_vertices_return_expected_counts() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let handle = test_handle();
    let entered = handle.enter("triangle_count_all_and_start_vertices_return_expected_counts").expect("entered");
    let src = DeviceArray::from_slice(&entered, UNDIRECTED_TRIANGLE_SRC_I32).expect("src");
    let dst = DeviceArray::from_slice(&entered, UNDIRECTED_TRIANGLE_DST_I32).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .properties(GraphProperties::undirected())
        .renumber()
        .build::<Forward>(&entered)
        .expect("graph");

    let params = TriangleCountParams::default();
    let all = graph.triangle_count_all(&entered, &params).expect("triangle_count all");
    let all_vertices = all.vertex_ids().expect("vertices").to_vec_i32(&entered).expect("vertices");
    let all_counts = all.counts().expect("counts").to_vec_i32(&entered).expect("counts");
    let all_map: HashMap<i32, i32> = all_vertices.into_iter().zip(all_counts).collect();

    assert_eq!(all_map.len(), 3);
    assert_eq!(all_map[&0], 1);
    assert_eq!(all_map[&1], 1);
    assert_eq!(all_map[&2], 1);

    let start = DeviceArray::from_slice(&entered, &[0i32, 2]).expect("start");
    let start_view = start.view().expect("start_view");
    let subset = graph.triangle_count(&entered, &start_view, &params).expect("triangle_count subset");
    let subset_vertices = subset.vertex_ids().expect("vertices").to_vec_i32(&entered).expect("vertices");
    let subset_counts = subset.counts().expect("counts").to_vec_i32(&entered).expect("counts");
    let subset_map: HashMap<i32, i32> = subset_vertices.into_iter().zip(subset_counts).collect();

    assert_eq!(subset_map.len(), 2);
    assert_eq!(subset_map[&0], 1);
    assert_eq!(subset_map[&2], 1);
}

#[test]
fn k_core_with_and_without_precomputed_core_numbers_returns_triangle_edges() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let handle = test_handle();
    let entered =
        handle.enter("k_core_with_and_without_precomputed_core_numbers_returns_triangle_edges").expect("entered");
    let src = DeviceArray::from_slice(&entered, &[0i32, 1, 2, 1, 2, 0, 2, 3]).expect("src");
    let dst = DeviceArray::from_slice(&entered, &[1i32, 2, 0, 0, 1, 2, 3, 2]).expect("dst");
    let wgt = DeviceArray::from_slice(&entered, &[1.0f32; 8]).expect("wgt");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let wgt_view = wgt.view().expect("wgt_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .weights(&wgt_view)
        .properties(GraphProperties::undirected())
        .renumber()
        .build::<Forward>(&entered)
        .expect("graph");

    let params = KCoreParams {
        k: NonZeroUsize::new(2).expect("k"),
        degree_type: KCoreDegreeType::Out,
        do_expensive_check: false,
    };
    let result = graph.k_core(&entered, &params).expect("k_core");
    assert!(result.weights().expect("weights").is_some());
    let sources = result.sources().expect("sources").to_vec_i32(&entered).expect("sources");
    let destinations = result.destinations().expect("destinations").to_vec_i32(&entered).expect("destinations");
    assert_eq!(sources.len(), destinations.len());
    assert!(sources.iter().zip(&destinations).all(|(src, dst)| *src < 3 && *dst < 3));

    let core_numbers = graph
        .core_number(&entered, &CoreNumberParams { degree_type: KCoreDegreeType::Out, do_expensive_check: false })
        .expect("core_number");
    let precomputed =
        graph.k_core_with_core_numbers(&entered, &params, &core_numbers).expect("k_core with core numbers");
    assert_eq!(precomputed.sources().expect("sources").len(), result.sources().expect("sources").len());
}

#[test]
fn minimum_spanning_tree_returns_weighted_edge_result() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let handle = test_handle();
    let entered = handle.enter("minimum_spanning_tree_returns_weighted_edge_result").expect("entered");
    let src = DeviceArray::from_slice(&entered, LOUVAIN_GRAPH_SRC_I32).expect("src");
    let dst = DeviceArray::from_slice(&entered, LOUVAIN_GRAPH_DST_I32).expect("dst");
    let weights = DeviceArray::from_slice(&entered, LOUVAIN_GRAPH_WEIGHT_F32).expect("weights");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let weights_view = weights.view().expect("weights_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .weights(&weights_view)
        .properties(GraphProperties::undirected())
        .renumber()
        .build::<Forward>(&entered)
        .expect("graph");

    let result =
        graph.minimum_spanning_tree(&entered, &MinimumSpanningTreeParams::default()).expect("minimum_spanning_tree");
    let sources = result.sources().expect("sources").to_vec_i32(&entered).expect("sources");
    let destinations = result.destinations().expect("destinations").to_vec_i32(&entered).expect("destinations");
    let weights = result.weights().expect("weights").expect("weight column").to_vec_f32(&entered).expect("weights");
    let offsets =
        result.subgraph_offsets().expect("offsets").expect("offset column").to_vec_usize(&entered).expect("offsets");

    assert_eq!(sources.len(), destinations.len());
    assert_eq!(sources.len(), weights.len());
    assert_eq!(sources.len(), 10);
    assert_eq!(offsets, vec![0, sources.len()]);
    assert!(result.edge_ids().expect("edge ids").is_none());
    assert!(result.edge_type_ids().expect("edge type ids").is_none());
}

#[test]
fn k_truss_subgraph_returns_expected_offsets_and_edges() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let handle = test_handle();
    let entered = handle.enter("k_truss_subgraph_returns_expected_offsets_and_edges").expect("entered");
    let src = DeviceArray::from_slice(&entered, LOUVAIN_GRAPH_SRC_I32).expect("src");
    let dst = DeviceArray::from_slice(&entered, LOUVAIN_GRAPH_DST_I32).expect("dst");
    let weights = DeviceArray::from_slice(&entered, LOUVAIN_GRAPH_WEIGHT_F32).expect("weights");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let weights_view = weights.view().expect("weights_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .weights(&weights_view)
        .properties(GraphProperties::undirected())
        .renumber()
        .build::<Forward>(&entered)
        .expect("graph");

    let params = KTrussSubgraphParams { k: NonZeroUsize::new(3).expect("k"), do_expensive_check: false };
    let result = graph.k_truss_subgraph(&entered, &params).expect("k_truss_subgraph");
    let sources = result.sources().expect("sources").to_vec_i32(&entered).expect("sources");
    let destinations = result.destinations().expect("destinations").to_vec_i32(&entered).expect("destinations");
    let offsets =
        result.subgraph_offsets().expect("offsets").expect("offset column").to_vec_usize(&entered).expect("offsets");

    assert_eq!(sources.len(), destinations.len());
    assert_eq!(sources.len(), 10);
    assert_eq!(offsets, vec![0, sources.len()]);
    assert!(result.weights().expect("weights").is_some());
}

#[test]
fn extract_ego_returns_per_seed_offsets() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let handle = test_handle();
    let entered = handle.enter("extract_ego_returns_per_seed_offsets").expect("entered");
    let src = DeviceArray::from_slice(&entered, &[0i32, 1, 1, 2, 2, 2, 3, 3, 4]).expect("src");
    let dst = DeviceArray::from_slice(&entered, &[1i32, 3, 4, 0, 1, 3, 4, 5, 5]).expect("dst");
    let weights =
        DeviceArray::from_slice(&entered, &[0.1f32, 2.1, 1.1, 5.1, 3.1, 4.1, 7.2, 3.2, 6.1]).expect("weights");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let weights_view = weights.view().expect("weights_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .weights(&weights_view)
        .renumber()
        .build::<Forward>(&entered)
        .expect("graph");
    let seeds = DeviceArray::from_slice(&entered, &[0i32, 1]).expect("seeds");
    let seeds_view = seeds.view().expect("seeds_view");
    let params = ExtractEgoParams { radius: NonZeroUsize::new(2).expect("radius"), do_expensive_check: false };

    let result = graph.extract_ego(&entered, &seeds_view, &params).expect("extract_ego");
    let sources = result.sources().expect("sources").to_vec_i32(&entered).expect("sources");
    let destinations = result.destinations().expect("destinations").to_vec_i32(&entered).expect("destinations");
    let offsets =
        result.subgraph_offsets().expect("offsets").expect("offset column").to_vec_usize(&entered).expect("offsets");

    assert_eq!(sources.len(), destinations.len());
    assert_eq!(offsets, vec![0, 4, 9]);
    assert_eq!(sources.len(), *offsets.last().expect("last offset"));
}

#[test]
fn pagerank_on_single_vertex_graph_returns_score_one() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let handle = test_handle();
    let entered = handle.enter("pagerank_on_single_vertex_graph_returns_score_one").expect("entered");
    let empty: [i32; 0] = [];
    let src = DeviceArray::from_slice(&entered, &empty).expect("src");
    let dst = DeviceArray::from_slice(&entered, &empty).expect("dst");
    let vertices = DeviceArray::from_slice(&entered, &[0i32]).expect("vertices");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let vertices_view = vertices.view().expect("vertices_view");

    let mut graph =
        GraphBuilder::new(&src_view, &dst_view).vertices(&vertices_view).build::<Transposed>(&entered).expect("graph");

    let result = graph.pagerank(&entered, &PageRankParams::default()).expect("pagerank");
    let returned_vertices = result.vertex_ids().expect("vertex_ids").to_vec_i32(&entered).expect("vertex copy");
    let scores = result.scores().expect("scores").to_vec_f32(&entered).expect("score copy");

    assert_eq!(returned_vertices, vec![0]);
    assert_eq!(scores.len(), 1);
    assert!((scores[0] - 1.0).abs() < 1e-6, "single-vertex PageRank should be 1.0, got {}", scores[0]);
    assert!(result.converged().expect("converged"));
}

#[test]
fn pagerank_on_disconnected_graph_returns_normalized_positive_scores() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let handle = test_handle();
    let entered = handle.enter("pagerank_on_disconnected_graph_returns_normalized_positive_scores").expect("entered");
    let src = DeviceArray::from_slice(&entered, DISCONNECTED_TWO_CYCLES_SRC_I32).expect("src");
    let dst = DeviceArray::from_slice(&entered, DISCONNECTED_TWO_CYCLES_DST_I32).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view).build::<Transposed>(&entered).expect("graph");

    let result = graph.pagerank(&entered, &PageRankParams::default()).expect("pagerank");
    let vertices = result.vertex_ids().expect("vertex_ids").to_vec_i32(&entered).expect("vertex copy");
    let scores = result.scores().expect("scores").to_vec_f32(&entered).expect("score copy");

    let score_map: HashMap<i32, f32> = vertices.into_iter().zip(scores.iter().copied()).collect();
    let total_score: f32 = scores.iter().sum();

    assert_eq!(score_map.len(), 4);
    assert!(scores.iter().all(|score| score.is_finite() && *score > 0.0));
    assert!((total_score - 1.0).abs() < 1e-5, "scores should be normalized, got sum {}", total_score);
    assert!((score_map[&0] - score_map[&1]).abs() < 1e-6);
    assert!((score_map[&2] - score_map[&3]).abs() < 1e-6);
}

#[test]
fn degrees_all_on_empty_graph_returns_empty_results() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let handle = test_handle();
    let entered = handle.enter("degrees_all_on_empty_graph_returns_empty_results").expect("entered");
    let empty: [i32; 0] = [];
    let src = DeviceArray::from_slice(&entered, &empty).expect("src");
    let dst = DeviceArray::from_slice(&entered, &empty).expect("dst");
    let vertices = DeviceArray::from_slice(&entered, &empty).expect("vertices");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let vertices_view = vertices.view().expect("vertices_view");

    let mut graph =
        GraphBuilder::new(&src_view, &dst_view).vertices(&vertices_view).build::<Transposed>(&entered).expect("graph");

    let result = graph.degrees_all(&entered).expect("degrees");
    let returned_vertices = result.vertex_ids().expect("vertex_ids").to_vec_i32(&entered).expect("vertex copy");
    let in_degrees = result.in_degrees().expect("in_degrees").to_vec_i64(&entered).expect("in copy");
    let out_degrees = result.out_degrees().expect("out_degrees").to_vec_i64(&entered).expect("out copy");

    assert!(returned_vertices.is_empty());
    assert!(in_degrees.is_empty());
    assert!(out_degrees.is_empty());
}

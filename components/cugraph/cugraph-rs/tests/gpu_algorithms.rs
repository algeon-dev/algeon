mod common;

use std::collections::HashMap;

use common::support::{
    DEGREE_QUERY_I64, DEGREE_SUBSET_DST_I64, DEGREE_SUBSET_SRC_I64, DIRECTED_STAR_DST_I64, DIRECTED_STAR_SRC_I64,
    DISCONNECTED_TWO_CYCLES_DST_I32, DISCONNECTED_TWO_CYCLES_SRC_I32, FOUR_VERTEX_CHAIN_DST_I32,
    FOUR_VERTEX_CHAIN_DST_I64, FOUR_VERTEX_CHAIN_SRC_I32, FOUR_VERTEX_CHAIN_SRC_I64, TEST_LAYOUT_SECONDARY_SEED,
    TEST_LAYOUT_SMOKE_ITERATIONS, TRIANGLE_VERTEX_IDS_I64, UNDIRECTED_TRIANGLE_DST_I64, UNDIRECTED_TRIANGLE_SRC_I64,
    UNDIRECTED_TRIANGLE_UNIT_WEIGHT_F64, bfs_distance_map_i64_result,
};
use cugraph::{
    DeviceArray, Forward, GraphBuilder, ResourceHandle, Transposed, VertexSelection,
    centrality::{BetweennessCentralityParams, PageRankOptions},
    layout::{ForceAtlas2Options, ForceAtlas2Params, ForceAtlas2VertexAttrs, InitialPositions},
    params::IterationCount,
    random::RngState,
    structure::{Distances, ScoreValues, VertexIds},
    traversal::BfsParams,
    types::GraphProperties,
};

#[test]
fn bfs_accepts_i64_sources_for_i64_graphs() {
    let handle = ResourceHandle::new().expect("handle");
    let src = DeviceArray::from_slice(&handle, FOUR_VERTEX_CHAIN_SRC_I64).expect("src");
    let dst = DeviceArray::from_slice(&handle, FOUR_VERTEX_CHAIN_DST_I64).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view).renumber().build::<Forward>(&handle).expect("graph");
    let sources = DeviceArray::from_slice(&handle, &[0i64]).expect("sources");
    let sources_view = sources.view().expect("sources_view");

    let result = graph.bfs(&handle, &sources_view, &BfsParams::default().with_predecessors()).expect("bfs");

    let vertex_ids = result.vertex_ids().expect("vertex_ids");
    assert!(matches!(vertex_ids, VertexIds::I64(_)));
    let distances = result.distances().expect("distances");
    assert!(matches!(distances, Distances::I64(_)));
    let predecessors = result.predecessors().expect("predecessors").expect("predecessor values");
    assert!(matches!(predecessors, VertexIds::I64(_)));

    let distance_map = bfs_distance_map_i64_result(&handle, &result);

    assert_eq!(distance_map[&0], 0);
    assert_eq!(distance_map[&1], 1);
    assert_eq!(distance_map[&2], 2);
    assert_eq!(distance_map[&3], 3);
}

#[test]
fn betweenness_accepts_i64_seed_vertices() {
    let handle = ResourceHandle::new().expect("handle");
    let src = DeviceArray::from_slice(&handle, DIRECTED_STAR_SRC_I64).expect("src");
    let dst = DeviceArray::from_slice(&handle, DIRECTED_STAR_DST_I64).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view).renumber().build::<Forward>(&handle).expect("graph");
    let seeds = DeviceArray::from_slice(&handle, &[0i64, 1]).expect("seeds");
    let seeds_view = seeds.view().expect("seeds_view");

    let result = graph
        .betweenness_centrality(&handle, &seeds_view, &BetweennessCentralityParams::default())
        .expect("betweenness");

    let vertex_ids = result.vertex_ids().expect("vertex_ids");
    assert!(matches!(vertex_ids, VertexIds::I64(_)));
    assert_eq!(vertex_ids.len(), 4);

    let scores = result.scores().expect("scores");
    let values = scores.to_vec_f64(&handle).expect("score copy");
    assert_eq!(values.len(), 4);
    assert!(values.iter().all(|v| v.is_finite() && *v >= 0.0));
}

#[test]
fn degrees_accept_i64_vertex_subsets() {
    let handle = ResourceHandle::new().expect("handle");
    let src = DeviceArray::from_slice(&handle, DEGREE_SUBSET_SRC_I64).expect("src");
    let dst = DeviceArray::from_slice(&handle, DEGREE_SUBSET_DST_I64).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view).renumber().build::<Transposed>(&handle).expect("graph");
    let query = DeviceArray::from_slice(&handle, DEGREE_QUERY_I64).expect("query");
    let query_view = query.view().expect("query_view");

    let result = graph.degrees(&handle, &query_view).expect("degrees");

    let vertices = result.vertex_ids().expect("vertex_ids").to_vec_i64(&handle).expect("vertex copy");
    let in_degrees = result.in_degrees().expect("in_degrees").to_vec_i64(&handle).expect("in copy");
    let out_degrees = result.out_degrees().expect("out_degrees").to_vec_i64(&handle).expect("out copy");

    let degree_map: HashMap<i64, (i64, i64)> =
        vertices.into_iter().zip(in_degrees.into_iter().zip(out_degrees)).collect();

    assert_eq!(degree_map.len(), 2);
    assert_eq!(degree_map[&0], (1, 1));
    assert_eq!(degree_map[&2], (1, 2));
}

#[test]
fn in_degrees_checked_accepts_all_vertices() {
    let handle = ResourceHandle::new().expect("handle");
    let src = DeviceArray::from_slice(&handle, DIRECTED_STAR_SRC_I64).expect("src");
    let dst = DeviceArray::from_slice(&handle, DIRECTED_STAR_DST_I64).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view).renumber().build::<Transposed>(&handle).expect("graph");

    let result = graph.in_degrees_checked(&handle, VertexSelection::<i64>::All, true).expect("in_degrees_checked");

    let vertices = result.vertex_ids().expect("vertex_ids").to_vec_i64(&handle).expect("vertex copy");
    let in_degrees = result.in_degrees().expect("in_degrees").to_vec_i64(&handle).expect("in copy");
    let degree_map: HashMap<i64, i64> = vertices.into_iter().zip(in_degrees).collect();

    assert_eq!(degree_map[&0], 3);
    for v in [1_i64, 2, 3] {
        assert_eq!(degree_map[&v], 1, "leaf vertex {v} should have in-degree 1");
    }
}

#[test]
fn out_degrees_checked_accepts_all_vertices() {
    let handle = ResourceHandle::new().expect("handle");
    let src = DeviceArray::from_slice(&handle, DIRECTED_STAR_SRC_I64).expect("src");
    let dst = DeviceArray::from_slice(&handle, DIRECTED_STAR_DST_I64).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view).renumber().build::<Forward>(&handle).expect("graph");

    let result = graph.out_degrees_checked(&handle, VertexSelection::<i64>::All, true).expect("out_degrees_checked");

    let vertices = result.vertex_ids().expect("vertex_ids").to_vec_i64(&handle).expect("vertex copy");
    let out_degrees = result.out_degrees().expect("out_degrees").to_vec_i64(&handle).expect("out copy");
    let degree_map: HashMap<i64, i64> = vertices.into_iter().zip(out_degrees).collect();

    assert_eq!(degree_map[&0], 3);
    for v in [1_i64, 2, 3] {
        assert_eq!(degree_map[&v], 1, "leaf vertex {v} should have out-degree 1");
    }
}

#[test]
fn bfs_rejects_vertex_selection_all() {
    let handle = ResourceHandle::new().expect("handle");
    let src = DeviceArray::from_slice(&handle, FOUR_VERTEX_CHAIN_SRC_I32).expect("src");
    let dst = DeviceArray::from_slice(&handle, FOUR_VERTEX_CHAIN_DST_I32).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view).renumber().build::<Forward>(&handle).expect("graph");
    let err =
        graph.bfs(&handle, VertexSelection::<i32>::All, &BfsParams::default()).expect_err("bfs should reject All");

    assert_eq!(err.kind(), cugraph::ErrorKind::InvalidInput);
    assert_eq!(err.operation(), "Graph::bfs");
}

#[test]
fn force_atlas2_accepts_i64_vertices_and_f32_helpers() {
    let handle = ResourceHandle::new().expect("handle");
    let mut rng = RngState::new(&handle, TEST_LAYOUT_SECONDARY_SEED).expect("rng");

    let src = DeviceArray::from_slice(&handle, UNDIRECTED_TRIANGLE_SRC_I64).expect("src");
    let dst = DeviceArray::from_slice(&handle, UNDIRECTED_TRIANGLE_DST_I64).expect("dst");
    let wgt = DeviceArray::from_slice(&handle, UNDIRECTED_TRIANGLE_UNIT_WEIGHT_F64).expect("wgt");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let wgt_view = wgt.view().expect("wgt_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .weights(&wgt_view)
        .properties(GraphProperties::undirected())
        .renumber()
        .build::<Forward>(&handle)
        .expect("graph");

    let pos_vertices = DeviceArray::from_slice(&handle, TRIANGLE_VERTEX_IDS_I64).expect("pos vertices");
    let x = DeviceArray::from_slice(&handle, &[0.0f32, 1.0, -1.0]).expect("x");
    let y = DeviceArray::from_slice(&handle, &[0.0f32, 0.5, -0.5]).expect("y");
    let mass_values = DeviceArray::from_slice(&handle, &[1.0f32, 1.5, 2.0]).expect("mass");
    let pos_vertices_view = pos_vertices.view().expect("pos_vertices_view");
    let x_view = x.view().expect("x_view");
    let y_view = y.view().expect("y_view");
    let mass_values_view = mass_values.view().expect("mass_values_view");

    let options = ForceAtlas2Options {
        params: ForceAtlas2Params {
            max_iter: IterationCount::new(TEST_LAYOUT_SMOKE_ITERATIONS).expect("iter"),
            ..Default::default()
        },
    };
    let init_pos = InitialPositions { vertices: &pos_vertices_view, x: &x_view, y: &y_view };
    let attrs = ForceAtlas2VertexAttrs { mass: Some((&pos_vertices_view, &mass_values_view)), ..Default::default() };

    let result =
        graph.force_atlas2_with(&handle, &mut rng, &options, Some(&attrs), Some(&init_pos)).expect("force_atlas2");

    let vertex_ids = result.vertex_ids().expect("vertex_ids");
    assert!(matches!(vertex_ids, VertexIds::I64(_)));

    let x_coords = result.x().expect("x");
    let y_coords = result.y().expect("y");
    assert!(matches!(x_coords, ScoreValues::F32(_)));
    assert!(matches!(y_coords, ScoreValues::F32(_)));

    let x_values = x_coords.to_vec_f32(&handle).expect("x copy");
    let y_values = y_coords.to_vec_f32(&handle).expect("y copy");
    assert_eq!(x_values.len(), 3);
    assert_eq!(y_values.len(), 3);
    assert!(x_values.iter().all(|v| v.is_finite()));
    assert!(y_values.iter().all(|v| v.is_finite()));
}

#[test]
fn pagerank_on_single_vertex_graph_returns_score_one() {
    let handle = ResourceHandle::new().expect("handle");
    let empty: [i32; 0] = [];
    let src = DeviceArray::from_slice(&handle, &empty).expect("src");
    let dst = DeviceArray::from_slice(&handle, &empty).expect("dst");
    let vertices = DeviceArray::from_slice(&handle, &[0i32]).expect("vertices");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let vertices_view = vertices.view().expect("vertices_view");

    let mut graph =
        GraphBuilder::new(&src_view, &dst_view).vertices(&vertices_view).build::<Transposed>(&handle).expect("graph");

    let result = graph.pagerank(&handle, &PageRankOptions::default()).expect("pagerank");
    let returned_vertices = result.vertex_ids().expect("vertex_ids").to_vec_i32(&handle).expect("vertex copy");
    let scores = result.scores().expect("scores").to_vec_f32(&handle).expect("score copy");

    assert_eq!(returned_vertices, vec![0]);
    assert_eq!(scores.len(), 1);
    assert!((scores[0] - 1.0).abs() < 1e-6, "single-vertex PageRank should be 1.0, got {}", scores[0]);
    assert!(result.converged().expect("converged"));
}

#[test]
fn pagerank_on_disconnected_graph_returns_normalized_positive_scores() {
    let handle = ResourceHandle::new().expect("handle");
    let src = DeviceArray::from_slice(&handle, DISCONNECTED_TWO_CYCLES_SRC_I32).expect("src");
    let dst = DeviceArray::from_slice(&handle, DISCONNECTED_TWO_CYCLES_DST_I32).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view).build::<Transposed>(&handle).expect("graph");

    let result = graph.pagerank(&handle, &PageRankOptions::default()).expect("pagerank");
    let vertices = result.vertex_ids().expect("vertex_ids").to_vec_i32(&handle).expect("vertex copy");
    let scores = result.scores().expect("scores").to_vec_f32(&handle).expect("score copy");

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
    let handle = ResourceHandle::new().expect("handle");
    let empty: [i32; 0] = [];
    let src = DeviceArray::from_slice(&handle, &empty).expect("src");
    let dst = DeviceArray::from_slice(&handle, &empty).expect("dst");
    let vertices = DeviceArray::from_slice(&handle, &empty).expect("vertices");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let vertices_view = vertices.view().expect("vertices_view");

    let mut graph =
        GraphBuilder::new(&src_view, &dst_view).vertices(&vertices_view).build::<Transposed>(&handle).expect("graph");

    let result = graph.degrees_all(&handle).expect("degrees");
    let returned_vertices = result.vertex_ids().expect("vertex_ids").to_vec_i32(&handle).expect("vertex copy");
    let in_degrees = result.in_degrees().expect("in_degrees").to_vec_i64(&handle).expect("in copy");
    let out_degrees = result.out_degrees().expect("out_degrees").to_vec_i64(&handle).expect("out copy");

    assert!(returned_vertices.is_empty());
    assert!(in_degrees.is_empty());
    assert!(out_degrees.is_empty());
}

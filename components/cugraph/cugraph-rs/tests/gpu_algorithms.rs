use std::collections::HashMap;

use cugraph::{
    BetweennessCentralityParams, BfsParams, DataType, DegreeVertices, DeviceArray, Distances, ForceAtlas2Options,
    ForceAtlas2Params, ForceAtlas2VertexAttrs, Forward, GraphBuilder, InitialPositions, IterationCount, PageRankParams,
    ResourceHandle, RngState, ScoreValues, Transposed, VertexIds, types::GraphProperties,
};

#[test]
fn pagerank_supports_i64_vertices_and_f64_weights() {
    let handle = ResourceHandle::new().expect("handle");
    let src = DeviceArray::from_slice(&handle, &[1i64, 2, 3, 0, 0, 0]).expect("src");
    let dst = DeviceArray::from_slice(&handle, &[0i64, 0, 0, 1, 2, 3]).expect("dst");
    let wgt = DeviceArray::from_slice(&handle, &[1.0f64; 6]).expect("wgt");

    let mut graph = GraphBuilder::new(&src.view(), &dst.view())
        .weights(&wgt.view())
        .renumber()
        .build::<Transposed>(&handle)
        .expect("graph");

    assert_eq!(graph.vertex_dtype(), DataType::Int64);
    assert_eq!(graph.weight_dtype(), DataType::Float64);

    let result = graph.pagerank_simple(&handle, &PageRankParams::default()).expect("pagerank");

    let vertex_ids = result.vertex_ids().expect("vertex_ids");
    assert!(matches!(vertex_ids, VertexIds::I64(_)));
    let vertices = vertex_ids.to_vec_i64(&handle).expect("vertex copy");
    assert_eq!(vertices.len(), 4);

    let scores = result.scores().expect("scores");
    assert!(matches!(scores, ScoreValues::F64(_)));
    let values = scores.to_vec_f64(&handle).expect("score copy");
    assert_eq!(values.len(), 4);
    assert!(values.iter().all(|v| v.is_finite() && *v > 0.0));
}

#[test]
fn bfs_accepts_i64_sources_for_i64_graphs() {
    let handle = ResourceHandle::new().expect("handle");
    let src = DeviceArray::from_slice(&handle, &[0i64, 1, 2]).expect("src");
    let dst = DeviceArray::from_slice(&handle, &[1i64, 2, 3]).expect("dst");

    let mut graph = GraphBuilder::new(&src.view(), &dst.view()).renumber().build::<Forward>(&handle).expect("graph");
    let sources = DeviceArray::from_slice(&handle, &[0i64]).expect("sources");

    let result = graph.bfs(&handle, &sources.view(), &BfsParams::default().with_predecessors()).expect("bfs");

    let vertex_ids = result.vertex_ids().expect("vertex_ids");
    assert!(matches!(vertex_ids, VertexIds::I64(_)));
    let distances = result.distances().expect("distances");
    assert!(matches!(distances, Distances::I64(_)));
    let predecessors = result.predecessors().expect("predecessors").expect("predecessor values");
    assert!(matches!(predecessors, VertexIds::I64(_)));

    let distance_map: HashMap<i64, i64> = vertex_ids
        .to_vec_i64(&handle)
        .expect("vertex copy")
        .into_iter()
        .zip(distances.to_vec_i64(&handle).expect("distance copy"))
        .collect();

    assert_eq!(distance_map[&0], 0);
    assert_eq!(distance_map[&1], 1);
    assert_eq!(distance_map[&2], 2);
    assert_eq!(distance_map[&3], 3);
}

#[test]
fn betweenness_accepts_i64_seed_vertices() {
    let handle = ResourceHandle::new().expect("handle");
    let src = DeviceArray::from_slice(&handle, &[1i64, 2, 3, 0, 0, 0]).expect("src");
    let dst = DeviceArray::from_slice(&handle, &[0i64, 0, 0, 1, 2, 3]).expect("dst");

    let mut graph = GraphBuilder::new(&src.view(), &dst.view()).renumber().build::<Forward>(&handle).expect("graph");
    let seeds = DeviceArray::from_slice(&handle, &[0i64, 1]).expect("seeds");

    let result = graph
        .betweenness_centrality(&handle, Some(&seeds.view()), &BetweennessCentralityParams::default())
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
    let src = DeviceArray::from_slice(&handle, &[0i64, 1, 2, 2]).expect("src");
    let dst = DeviceArray::from_slice(&handle, &[1i64, 2, 0, 3]).expect("dst");

    let mut graph = GraphBuilder::new(&src.view(), &dst.view()).renumber().build::<Transposed>(&handle).expect("graph");
    let query = DeviceArray::from_slice(&handle, &[0i64, 2]).expect("query");

    let result = graph.degrees(&handle, &DegreeVertices::from(&query.view())).expect("degrees");

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
fn force_atlas2_accepts_i64_vertices_and_f32_helpers() {
    let handle = ResourceHandle::new().expect("handle");
    let mut rng = RngState::new(&handle, 7).expect("rng");

    let src = DeviceArray::from_slice(&handle, &[0i64, 1, 2, 1, 2, 0]).expect("src");
    let dst = DeviceArray::from_slice(&handle, &[1i64, 2, 0, 0, 1, 2]).expect("dst");
    let wgt = DeviceArray::from_slice(&handle, &[1.0f64; 6]).expect("wgt");

    let mut graph = GraphBuilder::new(&src.view(), &dst.view())
        .weights(&wgt.view())
        .properties(GraphProperties::undirected())
        .renumber()
        .build::<Forward>(&handle)
        .expect("graph");

    let pos_vertices = DeviceArray::from_slice(&handle, &[0i64, 1, 2]).expect("pos vertices");
    let x = DeviceArray::from_slice(&handle, &[0.0f32, 1.0, -1.0]).expect("x");
    let y = DeviceArray::from_slice(&handle, &[0.0f32, 0.5, -0.5]).expect("y");
    let mass_values = DeviceArray::from_slice(&handle, &[1.0f32, 1.5, 2.0]).expect("mass");

    let options = ForceAtlas2Options {
        params: ForceAtlas2Params { max_iter: IterationCount::new(10).expect("iter"), ..Default::default() },
        initial_positions: Some(InitialPositions { vertices: &pos_vertices.view(), x: &x.view(), y: &y.view() }),
        vertex_attrs: Some(ForceAtlas2VertexAttrs {
            mass: Some((&pos_vertices.view(), &mass_values.view())),
            ..Default::default()
        }),
    };

    let result = graph.force_atlas2(&handle, &mut rng, &options).expect("force_atlas2");

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

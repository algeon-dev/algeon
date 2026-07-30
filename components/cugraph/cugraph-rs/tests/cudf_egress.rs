//! Integration tests for graph/result exports back into cuDF.
mod common;

use std::{collections::HashMap, num::NonZeroUsize};

use common::{
    cudf::{explicit_runtime, read_parquet_dataframe, unique_temp_dir, write_parquet_dataframe},
    support,
    support::{
        BFS_GRAPH_DST_I32, BFS_GRAPH_SRC_I32, BFS_SOURCE_VERTEX_I32, DIRECTED_STAR_DST_I32, DIRECTED_STAR_SRC_I32,
        LOUVAIN_GRAPH_DST_I32, LOUVAIN_GRAPH_SRC_I32, LOUVAIN_GRAPH_WEIGHT_F32, SIMILARITY_JACCARD_F32,
        SIMILARITY_PAIR_FIRST_I32, SIMILARITY_PAIR_SECOND_I32, SSSP_GRAPH_DST_I32, SSSP_GRAPH_SRC_I32,
        SSSP_GRAPH_WEIGHT_F32, TEST_LAYOUT_PRIMARY_SEED, TEST_LAYOUT_SETTLE_ITERATIONS,
    },
};
use cugraph::{
    array::DeviceArray,
    centrality::{BetweennessCentralityParams, EdgeBetweennessCentralityParams, HitsParams, PageRankParams},
    community::{
        EcgParams, ExtractEgoParams, KTrussSubgraphParams, LeidenParams, LouvainParams, SpectralClusteringParams,
        TriangleCountParams,
    },
    graph::{Forward, GraphBuilder, Transposed},
    handle::ResourceHandle,
    labeling::ConnectedComponentsParams,
    random::RngState,
    similarity::{SimilarityParams, VertexPairs},
    structure::{CoreNumberParams, KCoreDegreeType, KCoreParams},
    traversal::{BfsParams, SsspParams},
    tree::MinimumSpanningTreeParams,
    types::GraphProperties,
};

#[test]
fn layout_result_to_dataframe() {
    use cugraph::{layout::ForceAtlas2Params, params::IterationCount, random::RngState};

    let _guard = common::cudf::cuda_test_serial_guard();
    let runtime = std::sync::Arc::new(explicit_runtime());
    let handle = ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).expect("handle");
    let handle = handle.enter("layout_result_to_dataframe").expect("entered");
    let mut rng = RngState::new(&handle, TEST_LAYOUT_PRIMARY_SEED).expect("rng");

    let src = DeviceArray::from_slice(&handle, support::UNDIRECTED_TRIANGLE_SRC_I32).expect("src");
    let dst = DeviceArray::from_slice(&handle, support::UNDIRECTED_TRIANGLE_DST_I32).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .properties(GraphProperties::undirected())
        .renumber()
        .build::<Forward>(&handle)
        .expect("graph");

    let params = ForceAtlas2Params {
        max_iter: IterationCount::new(TEST_LAYOUT_SETTLE_ITERATIONS).unwrap(),
        ..Default::default()
    };

    let result = graph.force_atlas2(&handle, &mut rng, &params).expect("fa2");
    let expected_vertices = result.vertex_ids().expect("vertex_ids").to_vec_i32(&handle).expect("vertex copy");
    let expected_x = result.x().expect("x").to_vec_f32(&handle).expect("x copy");
    let expected_y = result.y().expect("y").to_vec_f32(&handle).expect("y copy");
    let df = result.to_dataframe(&runtime).expect("to_dataframe");

    assert_eq!(df.num_columns(), 3);
    assert_eq!(df.column_names()[0], "vertex");
    assert_eq!(df.column_names()[1], "x");
    assert_eq!(df.column_names()[2], "y");
    assert_eq!(df.num_rows(), 3);

    let vertices: Vec<i32> = df.series("vertex").expect("vertex column").to_vec_strict().expect("vertex to_vec");
    let x_vals: Vec<f32> = df.series("x").expect("x column").to_vec_strict().expect("x to_vec");
    let y_vals: Vec<f32> = df.series("y").expect("y column").to_vec_strict().expect("y to_vec");
    assert_eq!(vertices, expected_vertices);
    assert_eq!(x_vals, expected_x);
    assert_eq!(y_vals, expected_y);
    for (i, (&x, &y)) in x_vals.iter().zip(y_vals.iter()).enumerate() {
        assert!(x.is_finite(), "x[{i}] = {x} should be finite");
        assert!(y.is_finite(), "y[{i}] = {y} should be finite");
    }
}

#[test]
fn bfs_result_to_dataframe_with_predecessors() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let runtime = std::sync::Arc::new(explicit_runtime());
    let handle = ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).expect("handle");
    let handle = handle.enter("bfs_result_to_dataframe_with_predecessors").expect("entered");

    let src = DeviceArray::from_slice(&handle, BFS_GRAPH_SRC_I32).expect("src");
    let dst = DeviceArray::from_slice(&handle, BFS_GRAPH_DST_I32).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let mut graph = GraphBuilder::new(&src_view, &dst_view).renumber().build::<Forward>(&handle).expect("graph");

    let sources = DeviceArray::from_slice(&handle, &[BFS_SOURCE_VERTEX_I32]).expect("sources");
    let params = BfsParams::default().with_predecessors();
    let sources_view = sources.view().expect("sources_view");
    let result = graph.bfs(&handle, &sources_view, &params).expect("bfs");
    let expected_vertices = result.vertex_ids().expect("vertex_ids").to_vec_i32(&handle).expect("vertex copy");
    let expected_distances = result.distances().expect("distances").to_vec_i32(&handle).expect("distance copy");
    let expected_predecessors = result
        .predecessors()
        .expect("predecessors")
        .expect("predecessors should exist")
        .to_vec_i32(&handle)
        .expect("predecessor copy");
    let df = result.to_dataframe(&runtime).expect("to_dataframe");

    assert_eq!(df.num_columns(), 3, "should have 3 columns with predecessors");
    assert_eq!(df.column_names()[0], "vertex");
    assert_eq!(df.column_names()[1], "distance");
    assert_eq!(df.column_names()[2], "predecessor");

    let vertices: Vec<i32> = df.series("vertex").expect("vertex column").to_vec_strict().expect("vertex to_vec");
    let distances: Vec<i32> = df.series("distance").expect("distance column").to_vec_strict().expect("distance to_vec");
    let predecessors: Vec<i32> =
        df.series("predecessor").expect("predecessor column").to_vec_strict().expect("predecessor to_vec");
    assert_eq!(vertices, expected_vertices);
    assert_eq!(distances, expected_distances);
    assert_eq!(predecessors, expected_predecessors);
    let dist_map: HashMap<i32, i32> = vertices.into_iter().zip(distances).collect();
    assert_eq!(dist_map[&0], 0, "source vertex should have distance 0");
}

#[test]
fn sssp_result_to_dataframe_with_predecessors() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let runtime = std::sync::Arc::new(explicit_runtime());
    let handle = ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).expect("handle");
    let handle = handle.enter("sssp_result_to_dataframe_with_predecessors").expect("entered");

    let src = DeviceArray::from_slice(&handle, SSSP_GRAPH_SRC_I32).expect("src");
    let dst = DeviceArray::from_slice(&handle, SSSP_GRAPH_DST_I32).expect("dst");
    let weights = DeviceArray::from_slice(&handle, SSSP_GRAPH_WEIGHT_F32).expect("weights");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let weights_view = weights.view().expect("weights_view");
    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .weights(&weights_view)
        .renumber()
        .build::<Forward>(&handle)
        .expect("graph");

    let result = graph.sssp(&handle, 0i32, &SsspParams::default().with_predecessors()).expect("sssp");
    let expected_vertices = result.vertex_ids().expect("vertex_ids").to_vec_i32(&handle).expect("vertex copy");
    let expected_distances = result.distances().expect("distances").to_vec_f32(&handle).expect("distance copy");
    let expected_predecessors = result
        .predecessors()
        .expect("predecessors")
        .expect("predecessors should exist")
        .to_vec_i32(&handle)
        .expect("predecessor copy");
    let df = result.to_dataframe(&runtime).expect("to_dataframe");

    assert_eq!(df.num_columns(), 3, "should have 3 columns with predecessors");
    assert_eq!(df.column_names()[0], "vertex");
    assert_eq!(df.column_names()[1], "distance");
    assert_eq!(df.column_names()[2], "predecessor");

    let vertices: Vec<i32> = df.series("vertex").expect("vertex column").to_vec_strict().expect("vertex to_vec");
    let distances: Vec<f32> = df.series("distance").expect("distance column").to_vec_strict().expect("distance to_vec");
    let predecessors: Vec<i32> =
        df.series("predecessor").expect("predecessor column").to_vec_strict().expect("predecessor to_vec");
    assert_eq!(vertices, expected_vertices);
    assert_eq!(distances, expected_distances);
    assert_eq!(predecessors, expected_predecessors);
}

#[test]
fn centrality_to_dataframe_write_parquet_roundtrip() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let runtime = std::sync::Arc::new(explicit_runtime());
    let handle = ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).expect("handle");
    let handle = handle.enter("centrality_to_dataframe_write_parquet_roundtrip").expect("entered");

    let src = DeviceArray::from_slice(&handle, support::UNDIRECTED_TRIANGLE_SRC_I32).expect("src");
    let dst = DeviceArray::from_slice(&handle, support::UNDIRECTED_TRIANGLE_DST_I32).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .properties(GraphProperties::undirected())
        .renumber()
        .build::<Transposed>(&handle)
        .expect("graph");

    let df = graph
        .pagerank(&handle, &PageRankParams::default())
        .expect("pagerank")
        .to_dataframe(&runtime)
        .expect("to_dataframe");
    let expected_vertices: Vec<i32> = df.series("vertex").expect("v").to_vec_strict().expect("v vec");
    let expected_values: Vec<f32> = df.series("value").expect("val").to_vec_strict().expect("val vec");

    let tmp_dir = unique_temp_dir("centrality_df_roundtrip");
    std::fs::create_dir_all(&tmp_dir).expect("create tmp dir");
    let out_path = tmp_dir.join("centrality.parquet");

    write_parquet_dataframe(&df, &out_path);

    let read_df = read_parquet_dataframe(&out_path, &explicit_runtime());

    assert_eq!(read_df.num_columns(), 2);
    let read_verts: Vec<i32> = read_df.series("vertex").expect("vertex").to_vec_strict().expect("vertex vec");
    let read_vals: Vec<f32> = read_df.series("value").expect("value").to_vec_strict().expect("value vec");

    assert_eq!(read_verts, expected_vertices, "vertex column mismatch after Parquet round-trip");
    assert_eq!(read_vals, expected_values, "value column mismatch after Parquet round-trip");

    let _ = std::fs::remove_dir_all(&tmp_dir);
}

#[test]
fn betweenness_result_to_dataframe() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let runtime = std::sync::Arc::new(explicit_runtime());
    let handle = ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).expect("handle");
    let handle = handle.enter("betweenness_result_to_dataframe").expect("entered");

    let src = DeviceArray::from_slice(&handle, DIRECTED_STAR_SRC_I32).expect("src");
    let dst = DeviceArray::from_slice(&handle, DIRECTED_STAR_DST_I32).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view).renumber().build::<Forward>(&handle).expect("graph");

    let df = graph
        .betweenness_centrality_all(&handle, &BetweennessCentralityParams::default())
        .expect("betweenness")
        .to_dataframe(&runtime)
        .expect("to_dataframe");

    assert_eq!(df.num_columns(), 2);
    assert_eq!(df.column_names()[0], "vertex");
    assert_eq!(df.column_names()[1], "value");
    assert_eq!(df.num_rows(), 4);

    let vertices: Vec<i32> = df.series("vertex").expect("vertex column").to_vec_strict().expect("vertex to_vec");
    let values: Vec<f32> = df.series("value").expect("value column").to_vec_strict().expect("value to_vec");

    assert_eq!(vertices.len(), 4);
    assert_eq!(values.len(), 4);
    assert!(values.iter().all(|value| value.is_finite() && *value >= 0.0));
}

#[test]
fn hits_result_to_dataframe() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let runtime = std::sync::Arc::new(explicit_runtime());
    let handle = ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).expect("handle");
    let handle = handle.enter("hits_result_to_dataframe").expect("entered");

    let src = DeviceArray::from_slice(&handle, &[0i32, 0, 1, 2]).expect("src");
    let dst = DeviceArray::from_slice(&handle, &[1i32, 2, 2, 3]).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let mut graph = GraphBuilder::new(&src_view, &dst_view).renumber().build::<Transposed>(&handle).expect("graph");

    let df = graph.hits(&handle, &HitsParams::default()).expect("hits").to_dataframe(&runtime).expect("df");

    assert_eq!(df.num_columns(), 3);
    assert_eq!(df.column_names()[0], "vertex");
    assert_eq!(df.column_names()[1], "hub");
    assert_eq!(df.column_names()[2], "authority");
    assert_eq!(df.num_rows(), 4);
}

#[test]
fn edge_betweenness_result_to_dataframe() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let runtime = std::sync::Arc::new(explicit_runtime());
    let handle = ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).expect("handle");
    let handle = handle.enter("edge_betweenness_result_to_dataframe").expect("entered");

    let src = DeviceArray::from_slice(&handle, &[0i32, 1, 2]).expect("src");
    let dst = DeviceArray::from_slice(&handle, &[1i32, 2, 3]).expect("dst");
    let edge_ids = DeviceArray::from_slice(&handle, &[10i32, 11, 12]).expect("edge_ids");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let edge_ids_view = edge_ids.view().expect("edge_ids_view");
    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .edge_ids(&edge_ids_view)
        .renumber()
        .build::<Forward>(&handle)
        .expect("graph");

    let df = graph
        .edge_betweenness_centrality_all(&handle, &EdgeBetweennessCentralityParams::default())
        .expect("edge betweenness")
        .to_dataframe(&runtime)
        .expect("df");

    assert_eq!(df.num_columns(), 4);
    assert_eq!(df.column_names()[0], "source");
    assert_eq!(df.column_names()[1], "destination");
    assert_eq!(df.column_names()[2], "edge_id");
    assert_eq!(df.column_names()[3], "score");
    assert_eq!(df.num_rows(), 3);
}

#[test]
fn similarity_result_to_dataframe() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let runtime = std::sync::Arc::new(explicit_runtime());
    let handle = ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).expect("handle");
    let handle = handle.enter("similarity_result_to_dataframe").expect("entered");

    let src = DeviceArray::from_slice(&handle, LOUVAIN_GRAPH_SRC_I32).expect("src");
    let dst = DeviceArray::from_slice(&handle, LOUVAIN_GRAPH_DST_I32).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .properties(GraphProperties::undirected())
        .renumber()
        .build::<Forward>(&handle)
        .expect("graph");

    let pairs =
        graph.vertex_pairs(&handle, SIMILARITY_PAIR_FIRST_I32, SIMILARITY_PAIR_SECOND_I32).expect("vertex_pairs");
    assert_vertex_pairs_dataframe(&runtime, &handle, &pairs, SIMILARITY_JACCARD_F32.len());

    let pairs =
        graph.vertex_pairs(&handle, SIMILARITY_PAIR_FIRST_I32, SIMILARITY_PAIR_SECOND_I32).expect("vertex_pairs");
    let result = graph.jaccard_coefficients(&handle, pairs, &SimilarityParams::default()).expect("jaccard");
    let df = assert_similarity_dataframe_shape(
        result.to_dataframe(&runtime).expect("jaccard df"),
        SIMILARITY_JACCARD_F32.len(),
    );

    let first: Vec<i32> = df.series("first").expect("first column").to_vec_strict().expect("first to_vec");
    let second: Vec<i32> = df.series("second").expect("second column").to_vec_strict().expect("second to_vec");
    let scores: Vec<f32> =
        df.series("similarity").expect("similarity column").to_vec_strict().expect("similarity to_vec");

    assert_eq!(first, SIMILARITY_PAIR_FIRST_I32);
    assert_eq!(second, SIMILARITY_PAIR_SECOND_I32);
    for (index, (&actual, &expected)) in scores.iter().zip(SIMILARITY_JACCARD_F32).enumerate() {
        assert!((actual - expected).abs() < 1.0e-4, "similarity[{index}] expected {expected}, got {actual}");
    }

    let pairs =
        graph.vertex_pairs(&handle, SIMILARITY_PAIR_FIRST_I32, SIMILARITY_PAIR_SECOND_I32).expect("vertex_pairs");
    let sorensen = graph.sorensen_coefficients(&handle, pairs, &SimilarityParams::default()).expect("sorensen");
    assert_similarity_dataframe_shape(
        sorensen.to_dataframe(&runtime).expect("sorensen df"),
        SIMILARITY_JACCARD_F32.len(),
    );

    let pairs =
        graph.vertex_pairs(&handle, SIMILARITY_PAIR_FIRST_I32, SIMILARITY_PAIR_SECOND_I32).expect("vertex_pairs");
    let overlap = graph.overlap_coefficients(&handle, pairs, &SimilarityParams::default()).expect("overlap");
    assert_similarity_dataframe_shape(
        overlap.to_dataframe(&runtime).expect("overlap df"),
        SIMILARITY_JACCARD_F32.len(),
    );

    let pairs =
        graph.vertex_pairs(&handle, SIMILARITY_PAIR_FIRST_I32, SIMILARITY_PAIR_SECOND_I32).expect("vertex_pairs");
    let cosine = graph.cosine_similarity_coefficients(&handle, pairs, &SimilarityParams::default()).expect("cosine");
    assert_similarity_dataframe_shape(cosine.to_dataframe(&runtime).expect("cosine df"), SIMILARITY_JACCARD_F32.len());
}

fn assert_vertex_pairs_dataframe(
    runtime: &cudf::Runtime,
    handle: &cugraph::EnteredContext<'_>,
    pairs: &VertexPairs,
    expected_rows: usize,
) {
    let df = pairs.to_dataframe(runtime).expect("vertex pairs dataframe");
    assert_eq!(df.num_columns(), 2);
    assert_eq!(df.column_names()[0], "first");
    assert_eq!(df.column_names()[1], "second");
    assert_eq!(df.num_rows(), expected_rows);
    assert_eq!(
        df.series("first").expect("first column").to_vec_strict::<i32>().expect("first to_vec"),
        pairs.first().expect("first").to_vec_i32(handle).expect("first copy")
    );
    assert_eq!(
        df.series("second").expect("second column").to_vec_strict::<i32>().expect("second to_vec"),
        pairs.second().expect("second").to_vec_i32(handle).expect("second copy")
    );
}

fn assert_similarity_dataframe_shape(df: cudf::DataFrame, expected_rows: usize) -> cudf::DataFrame {
    assert_eq!(df.num_columns(), 3);
    assert_eq!(df.column_names()[0], "first");
    assert_eq!(df.column_names()[1], "second");
    assert_eq!(df.column_names()[2], "similarity");
    assert_eq!(df.num_rows(), expected_rows);
    df
}

#[test]
fn louvain_result_to_dataframe() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let runtime = std::sync::Arc::new(explicit_runtime());
    let handle = ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).expect("handle");
    let handle = handle.enter("louvain_result_to_dataframe").expect("entered");

    let src = DeviceArray::from_slice(&handle, LOUVAIN_GRAPH_SRC_I32).expect("src");
    let dst = DeviceArray::from_slice(&handle, LOUVAIN_GRAPH_DST_I32).expect("dst");
    let weights = DeviceArray::from_slice(&handle, LOUVAIN_GRAPH_WEIGHT_F32).expect("weights");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let weights_view = weights.view().expect("weights_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .weights(&weights_view)
        .properties(GraphProperties::undirected())
        .renumber()
        .build::<Forward>(&handle)
        .expect("graph");

    let result = graph.louvain(&handle, &LouvainParams::default()).expect("louvain");
    let expected_rows = result.vertex_ids().expect("vertex_ids").len();
    let df = result.to_dataframe(&runtime).expect("to_dataframe");

    assert_eq!(df.num_columns(), 2);
    assert_eq!(df.column_names()[0], "vertex");
    assert_eq!(df.column_names()[1], "partition");
    assert_eq!(df.num_rows(), expected_rows);
}

#[test]
fn leiden_result_to_dataframe() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let runtime = std::sync::Arc::new(explicit_runtime());
    let handle = ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).expect("handle");
    let handle = handle.enter("leiden_result_to_dataframe").expect("entered");
    let mut rng = RngState::new(&handle, 0).expect("rng");

    let src = DeviceArray::from_slice(&handle, LOUVAIN_GRAPH_SRC_I32).expect("src");
    let dst = DeviceArray::from_slice(&handle, LOUVAIN_GRAPH_DST_I32).expect("dst");
    let weights = DeviceArray::from_slice(&handle, LOUVAIN_GRAPH_WEIGHT_F32).expect("weights");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let weights_view = weights.view().expect("weights_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .weights(&weights_view)
        .properties(GraphProperties::undirected())
        .renumber()
        .build::<Forward>(&handle)
        .expect("graph");

    let result = graph.leiden(&handle, &mut rng, &LeidenParams::default()).expect("leiden");
    let expected_rows = result.vertex_ids().expect("vertex_ids").len();
    let df = result.to_dataframe(&runtime).expect("to_dataframe");

    assert_eq!(df.num_columns(), 2);
    assert_eq!(df.column_names()[0], "vertex");
    assert_eq!(df.column_names()[1], "partition");
    assert_eq!(df.num_rows(), expected_rows);
}

#[test]
fn ecg_result_to_dataframe() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let runtime = std::sync::Arc::new(explicit_runtime());
    let handle = ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).expect("handle");
    let handle = handle.enter("ecg_result_to_dataframe").expect("entered");
    let mut rng = RngState::new(&handle, 0).expect("rng");

    let src = DeviceArray::from_slice(&handle, LOUVAIN_GRAPH_SRC_I32).expect("src");
    let dst = DeviceArray::from_slice(&handle, LOUVAIN_GRAPH_DST_I32).expect("dst");
    let weights = DeviceArray::from_slice(&handle, LOUVAIN_GRAPH_WEIGHT_F32).expect("weights");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let weights_view = weights.view().expect("weights_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .weights(&weights_view)
        .properties(GraphProperties::undirected())
        .renumber()
        .build::<Forward>(&handle)
        .expect("graph");

    let result = graph.ecg(&handle, &mut rng, &EcgParams::default()).expect("ecg");
    let expected_rows = result.vertex_ids().expect("vertex_ids").len();
    let df = result.to_dataframe(&runtime).expect("to_dataframe");

    assert_eq!(df.num_columns(), 2);
    assert_eq!(df.column_names()[0], "vertex");
    assert_eq!(df.column_names()[1], "partition");
    assert_eq!(df.num_rows(), expected_rows);
}

#[test]
fn spectral_clustering_result_to_dataframe() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let runtime = std::sync::Arc::new(explicit_runtime());
    let handle = ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).expect("handle");
    let handle = handle.enter("spectral_clustering_result_to_dataframe").expect("entered");
    let src_data = [0i32, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5];
    let dst_data = [1i32, 2, 0, 2, 0, 1, 3, 2, 4, 5, 3, 5, 3, 4];
    let weights_data = [0.1f32, 0.2, 0.1, 1.2, 0.2, 1.2, 2.3, 2.3, 3.4, 3.5, 3.4, 4.5, 3.5, 4.5];
    let src = DeviceArray::from_slice(&handle, &src_data).expect("src");
    let dst = DeviceArray::from_slice(&handle, &dst_data).expect("dst");
    let weights = DeviceArray::from_slice(&handle, &weights_data).expect("weights");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let weights_view = weights.view().expect("weights_view");
    let params = SpectralClusteringParams::default();
    let mut last_error = None;

    for seed in 10..20 {
        let mut graph = GraphBuilder::new(&src_view, &dst_view)
            .weights(&weights_view)
            .properties(GraphProperties::undirected())
            .renumber()
            .build::<Forward>(&handle)
            .expect("graph");
        let mut rng = RngState::new(&handle, seed).expect("rng");

        match graph.spectral_modularity_maximization(&handle, &mut rng, &params) {
            Ok(result) => {
                let expected_rows = result.vertex_ids().expect("vertex_ids").len();
                let df = result.to_dataframe(&runtime).expect("to_dataframe");

                assert_eq!(df.num_columns(), 2);
                assert_eq!(df.column_names()[0], "vertex");
                assert_eq!(df.column_names()[1], "partition");
                assert_eq!(df.num_rows(), expected_rows);
                return;
            },
            Err(err) => last_error = Some(err),
        }
    }

    panic!("spectral clustering did not converge in retry window: {last_error:?}");
}

#[test]
fn connected_components_results_to_dataframe() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let runtime = std::sync::Arc::new(explicit_runtime());
    let handle = ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).expect("handle");
    let handle = handle.enter("connected_components_results_to_dataframe").expect("entered");

    let src = DeviceArray::from_slice(&handle, support::DISCONNECTED_TWO_CYCLES_SRC_I32).expect("src");
    let dst = DeviceArray::from_slice(&handle, support::DISCONNECTED_TWO_CYCLES_DST_I32).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .properties(GraphProperties::undirected())
        .renumber()
        .build::<Forward>(&handle)
        .expect("graph");

    let params = ConnectedComponentsParams::default();
    let wcc = graph.weakly_connected_components(&handle, &params).expect("wcc");
    let expected_vertices = wcc.vertex_ids().expect("vertices").to_vec_i32(&handle).expect("vertices");
    let expected_labels = wcc.labels().expect("labels").to_vec_i32(&handle).expect("labels");
    let df = wcc.to_dataframe(&runtime).expect("wcc dataframe");

    assert_eq!(df.num_columns(), 2);
    assert_eq!(df.column_names()[0], "vertex");
    assert_eq!(df.column_names()[1], "label");
    assert_eq!(df.series("vertex").expect("vertex").to_vec_strict::<i32>().expect("vertex"), expected_vertices);
    assert_eq!(df.series("label").expect("label").to_vec_strict::<i32>().expect("label"), expected_labels);

    let scc = graph.strongly_connected_components(&handle, &params).expect("scc");
    let scc_df = scc.to_dataframe(&runtime).expect("scc dataframe");
    assert_eq!(scc_df.num_columns(), 2);
    assert_eq!(scc_df.column_names()[0], "vertex");
    assert_eq!(scc_df.column_names()[1], "label");
}

#[test]
fn core_number_result_to_dataframe() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let runtime = std::sync::Arc::new(explicit_runtime());
    let handle = ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).expect("handle");
    let handle = handle.enter("core_number_result_to_dataframe").expect("entered");

    let src = DeviceArray::from_slice(&handle, &[0i32, 1, 2, 1, 2, 0, 2, 3]).expect("src");
    let dst = DeviceArray::from_slice(&handle, &[1i32, 2, 0, 0, 1, 2, 3, 2]).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .properties(GraphProperties::undirected())
        .renumber()
        .build::<Forward>(&handle)
        .expect("graph");

    let params = CoreNumberParams { degree_type: KCoreDegreeType::Out, do_expensive_check: false };
    let result = graph.core_number(&handle, &params).expect("core_number");
    let expected_vertices = result.vertex_ids().expect("vertices").to_vec_i32(&handle).expect("vertices");
    let expected_core_numbers = result.core_numbers().expect("core_numbers").to_vec_i32(&handle).expect("core numbers");
    let df = result.to_dataframe(&runtime).expect("to_dataframe");

    assert_eq!(df.num_columns(), 2);
    assert_eq!(df.column_names()[0], "vertex");
    assert_eq!(df.column_names()[1], "core_number");
    assert_eq!(df.series("vertex").expect("vertex").to_vec_strict::<i32>().expect("vertex"), expected_vertices);
    assert_eq!(
        df.series("core_number").expect("core_number").to_vec_strict::<i32>().expect("core_number"),
        expected_core_numbers
    );
}

#[test]
fn triangle_count_result_to_dataframe() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let runtime = std::sync::Arc::new(explicit_runtime());
    let handle = ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).expect("handle");
    let handle = handle.enter("triangle_count_result_to_dataframe").expect("entered");

    let src = DeviceArray::from_slice(&handle, support::UNDIRECTED_TRIANGLE_SRC_I32).expect("src");
    let dst = DeviceArray::from_slice(&handle, support::UNDIRECTED_TRIANGLE_DST_I32).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .properties(GraphProperties::undirected())
        .renumber()
        .build::<Forward>(&handle)
        .expect("graph");

    let result = graph.triangle_count_all(&handle, &TriangleCountParams::default()).expect("triangle_count");
    let expected_vertices = result.vertex_ids().expect("vertices").to_vec_i32(&handle).expect("vertices");
    let expected_counts = result.counts().expect("counts").to_vec_i32(&handle).expect("counts");
    let df = result.to_dataframe(&runtime).expect("to_dataframe");

    assert_eq!(df.num_columns(), 2);
    assert_eq!(df.column_names()[0], "vertex");
    assert_eq!(df.column_names()[1], "triangle_count");
    assert_eq!(df.series("vertex").expect("vertex").to_vec_strict::<i32>().expect("vertex"), expected_vertices);
    assert_eq!(
        df.series("triangle_count").expect("triangle_count").to_vec_strict::<i32>().expect("triangle_count"),
        expected_counts
    );
}

#[test]
fn subgraph_edge_results_to_dataframe() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let runtime = std::sync::Arc::new(explicit_runtime());
    let handle = ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).expect("handle");
    let handle = handle.enter("subgraph_edge_results_to_dataframe").expect("entered");

    let src = DeviceArray::from_slice(&handle, LOUVAIN_GRAPH_SRC_I32).expect("src");
    let dst = DeviceArray::from_slice(&handle, LOUVAIN_GRAPH_DST_I32).expect("dst");
    let weights = DeviceArray::from_slice(&handle, LOUVAIN_GRAPH_WEIGHT_F32).expect("weights");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let weights_view = weights.view().expect("weights_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .weights(&weights_view)
        .properties(GraphProperties::undirected())
        .renumber()
        .build::<Forward>(&handle)
        .expect("graph");

    let k_core_params = KCoreParams {
        k: NonZeroUsize::new(2).expect("k"),
        degree_type: KCoreDegreeType::Out,
        do_expensive_check: false,
    };
    let k_core_df = graph.k_core(&handle, &k_core_params).expect("k_core").to_dataframe(&runtime).expect("k_core df");
    assert_eq!(k_core_df.column_names()[0], "source");
    assert_eq!(k_core_df.column_names()[1], "destination");
    assert_eq!(k_core_df.column_names()[2], "weight");

    let mst_df = graph
        .minimum_spanning_tree(&handle, &MinimumSpanningTreeParams::default())
        .expect("mst")
        .to_dataframe(&runtime)
        .expect("mst df");
    assert_eq!(mst_df.column_names()[0], "source");
    assert_eq!(mst_df.column_names()[1], "destination");
    assert_eq!(mst_df.column_names()[2], "weight");

    let k_truss_df = graph
        .k_truss_subgraph(
            &handle,
            &KTrussSubgraphParams { k: NonZeroUsize::new(3).expect("k"), do_expensive_check: false },
        )
        .expect("k_truss")
        .to_dataframe(&runtime)
        .expect("k_truss df");
    assert_eq!(k_truss_df.column_names()[0], "source");
    assert_eq!(k_truss_df.column_names()[1], "destination");
    assert_eq!(k_truss_df.column_names()[2], "weight");
}

#[test]
fn unweighted_subgraph_edge_results_to_dataframe_omit_weight_column() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let runtime = std::sync::Arc::new(explicit_runtime());
    let handle = ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).expect("handle");
    let handle = handle.enter("unweighted_subgraph_edge_results_to_dataframe_omit_weight_column").expect("entered");

    let src = DeviceArray::from_slice(&handle, support::UNDIRECTED_TRIANGLE_SRC_I32).expect("src");
    let dst = DeviceArray::from_slice(&handle, support::UNDIRECTED_TRIANGLE_DST_I32).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .properties(GraphProperties::undirected())
        .renumber()
        .build::<Forward>(&handle)
        .expect("graph");

    let k_core_params = KCoreParams {
        k: NonZeroUsize::new(2).expect("k"),
        degree_type: KCoreDegreeType::Out,
        do_expensive_check: false,
    };
    let k_core_df = graph.k_core(&handle, &k_core_params).expect("k_core").to_dataframe(&runtime).expect("k_core df");
    assert_eq!(k_core_df.num_columns(), 2);
    assert_eq!(k_core_df.column_names()[0], "source");
    assert_eq!(k_core_df.column_names()[1], "destination");

    let k_truss_df = graph
        .k_truss_subgraph(
            &handle,
            &KTrussSubgraphParams { k: NonZeroUsize::new(3).expect("k"), do_expensive_check: false },
        )
        .expect("k_truss")
        .to_dataframe(&runtime)
        .expect("k_truss df");
    assert_eq!(k_truss_df.num_columns(), 2);
    assert_eq!(k_truss_df.column_names()[0], "source");
    assert_eq!(k_truss_df.column_names()[1], "destination");
}

#[test]
fn extract_ego_result_to_dataframe() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let runtime = std::sync::Arc::new(explicit_runtime());
    let handle = ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).expect("handle");
    let handle = handle.enter("extract_ego_result_to_dataframe").expect("entered");

    let src = DeviceArray::from_slice(&handle, &[0i32, 1, 1, 2, 2, 2, 3, 3, 4]).expect("src");
    let dst = DeviceArray::from_slice(&handle, &[1i32, 3, 4, 0, 1, 3, 4, 5, 5]).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let mut graph = GraphBuilder::new(&src_view, &dst_view).renumber().build::<Forward>(&handle).expect("graph");
    let seeds = DeviceArray::from_slice(&handle, &[0i32, 1]).expect("seeds");
    let seeds_view = seeds.view().expect("seeds_view");
    let params = ExtractEgoParams { radius: NonZeroUsize::new(2).expect("radius"), do_expensive_check: false };

    let result = graph.extract_ego(&handle, &seeds_view, &params).expect("extract_ego");
    let expected_edges = result.sources().expect("sources").len();
    let df = result.to_dataframe(&runtime).expect("to_dataframe");

    assert_eq!(df.num_columns(), 2);
    assert_eq!(df.column_names()[0], "source");
    assert_eq!(df.column_names()[1], "destination");
    assert_eq!(df.num_rows(), expected_edges);
}

#[test]
fn pagerank_result_preserves_external_vertex_ids_for_renumbered_graphs() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let runtime = std::sync::Arc::new(explicit_runtime());
    let handle = ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).expect("handle");
    let handle = handle.enter("pagerank_result_preserves_external_vertex_ids_for_renumbered_graphs").expect("entered");

    let expected_vertices = vec![10_i32, 20, 40, 80];
    let src_data = vec![10_i32, 20, 40, 80, 20, 40, 80, 10];
    let dst_data = vec![20_i32, 40, 80, 10, 10, 20, 40, 80];

    let src = DeviceArray::from_slice(&handle, &src_data).expect("src");
    let dst = DeviceArray::from_slice(&handle, &dst_data).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .properties(GraphProperties::undirected())
        .renumber()
        .build::<Transposed>(&handle)
        .expect("graph");

    let result = graph.pagerank(&handle, &PageRankParams::default()).expect("pagerank");

    let mut raw_vertices = result.vertex_ids().expect("vertex_ids").to_vec_i32(&handle).expect("vertex copy");
    raw_vertices.sort_unstable();
    assert_eq!(raw_vertices, expected_vertices, "raw PageRank result leaked internal vertex ids");

    let df = result.to_dataframe(&runtime).expect("to_dataframe");
    let mut df_vertices: Vec<i32> = df.series("vertex").expect("vertex column").to_vec_strict().expect("vertex to_vec");
    df_vertices.sort_unstable();
    assert_eq!(df_vertices, expected_vertices, "PageRank DataFrame vertex column leaked internal vertex ids");
}

#[test]
fn louvain_result_preserves_external_vertex_ids_for_renumbered_graphs() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let runtime = std::sync::Arc::new(explicit_runtime());
    let handle = ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).expect("handle");
    let handle = handle.enter("louvain_result_preserves_external_vertex_ids_for_renumbered_graphs").expect("entered");

    let expected_vertices = vec![10_i32, 20, 40, 80];
    let src_data = vec![10_i32, 20, 40, 80, 20, 40, 80, 10];
    let dst_data = vec![20_i32, 40, 80, 10, 10, 20, 40, 80];
    let weights_data = vec![1.0_f32; src_data.len()];

    let src = DeviceArray::from_slice(&handle, &src_data).expect("src");
    let dst = DeviceArray::from_slice(&handle, &dst_data).expect("dst");
    let weights = DeviceArray::from_slice(&handle, &weights_data).expect("weights");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let weights_view = weights.view().expect("weights_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .weights(&weights_view)
        .properties(GraphProperties::undirected())
        .renumber()
        .build::<Forward>(&handle)
        .expect("graph");

    let result = graph.louvain(&handle, &LouvainParams::default()).expect("louvain");

    let mut raw_vertices = result.vertex_ids().expect("vertex_ids").to_vec_i32(&handle).expect("vertex copy");
    raw_vertices.sort_unstable();
    assert_eq!(raw_vertices, expected_vertices, "raw Louvain result leaked internal vertex ids");

    let df = result.to_dataframe(&runtime).expect("to_dataframe");
    let mut df_vertices: Vec<i32> = df.series("vertex").expect("vertex column").to_vec_strict().expect("vertex to_vec");
    df_vertices.sort_unstable();
    assert_eq!(df_vertices, expected_vertices, "Louvain DataFrame vertex column leaked internal vertex ids");
}

#[test]
fn layout_result_preserves_external_vertex_ids_for_renumbered_graphs() {
    use cugraph::{layout::ForceAtlas2Params, params::IterationCount, random::RngState};

    let _guard = common::cudf::cuda_test_serial_guard();
    let runtime = std::sync::Arc::new(explicit_runtime());
    let handle = ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).expect("handle");
    let handle = handle.enter("layout_result_preserves_external_vertex_ids_for_renumbered_graphs").expect("entered");
    let mut rng = RngState::new(&handle, TEST_LAYOUT_PRIMARY_SEED).expect("rng");

    let expected_vertices = vec![10_i32, 20, 40, 80];
    let src_data = vec![10_i32, 20, 40, 80, 20, 40, 80, 10];
    let dst_data = vec![20_i32, 40, 80, 10, 10, 20, 40, 80];

    let src = DeviceArray::from_slice(&handle, &src_data).expect("src");
    let dst = DeviceArray::from_slice(&handle, &dst_data).expect("dst");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .properties(GraphProperties::undirected())
        .renumber()
        .build::<Forward>(&handle)
        .expect("graph");

    let params = ForceAtlas2Params {
        max_iter: IterationCount::new(TEST_LAYOUT_SETTLE_ITERATIONS).unwrap(),
        ..Default::default()
    };
    let result = graph.force_atlas2(&handle, &mut rng, &params).expect("fa2");

    let mut raw_vertices = result.vertex_ids().expect("vertex_ids").to_vec_i32(&handle).expect("vertex copy");
    raw_vertices.sort_unstable();
    assert_eq!(raw_vertices, expected_vertices, "raw layout result leaked internal vertex ids");

    let df = result.to_dataframe(&runtime).expect("to_dataframe");
    let mut df_vertices: Vec<i32> = df.series("vertex").expect("vertex column").to_vec_strict().expect("vertex to_vec");
    df_vertices.sort_unstable();
    assert_eq!(df_vertices, expected_vertices, "layout DataFrame vertex column leaked internal vertex ids");
}

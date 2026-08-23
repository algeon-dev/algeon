#![allow(dead_code)]

use std::{collections::HashMap, path::Path};

use anyhow::Context;
use cudf::{DataFrame, Runtime, core::Table};
use cugraph::{
    array::DeviceArray,
    graph::{Forward, Graph},
    handle::EnteredContext,
    traversal::{BfsParams, BfsResult},
};

pub const BFS_SOURCE_VERTEX_I32: i32 = 0;
pub const BFS_GRAPH_SRC_I32: &[i32] = &[0, 1, 1, 2, 2, 2, 3, 4];
pub const BFS_GRAPH_DST_I32: &[i32] = &[1, 3, 4, 0, 1, 3, 5, 5];
pub const BFS_GRAPH_WEIGHT_F32: &[f32] = &[0.1, 2.1, 1.1, 5.1, 3.1, 4.1, 7.2, 3.2];

pub fn bfs_graph_table(runtime: &Runtime) -> Table {
    let src = runtime.column_from_slice(BFS_GRAPH_SRC_I32).expect("src");
    let dst = runtime.column_from_slice(BFS_GRAPH_DST_I32).expect("dst");
    let wgt = runtime.column_from_slice(BFS_GRAPH_WEIGHT_F32).expect("wgt");
    Table::from_columns(vec![src, dst, wgt]).expect("bfs graph table")
}

pub fn bfs_graph_dataframe(runtime: &Runtime) -> DataFrame {
    let src = runtime.series(BFS_GRAPH_SRC_I32).expect("src");
    let dst = runtime.series(BFS_GRAPH_DST_I32).expect("dst");
    let wgt = runtime.series(BFS_GRAPH_WEIGHT_F32).expect("weight");
    DataFrame::from_series([("src", src), ("dst", dst), ("weight", wgt)]).expect("bfs graph dataframe")
}

pub fn write_bfs_graph_parquet(path: &Path) -> anyhow::Result<()> {
    let runtime = Runtime::new().context("build cudf runtime for BFS fixture")?;
    let df = bfs_graph_dataframe(&runtime);
    df.write_parquet(path).context("write BFS fixture parquet")
}

pub const TEST_LAYOUT_PRIMARY_SEED: u64 = 42;
pub const TEST_LAYOUT_SECONDARY_SEED: u64 = 7;
pub const TEST_LAYOUT_SMOKE_ITERATIONS: u32 = 10;
pub const TEST_LAYOUT_SETTLE_ITERATIONS: u32 = 50;

pub const SIMPLE_CHAIN_SRC_I32: &[i32] = &[0, 1];
pub const SIMPLE_CHAIN_DST_I32: &[i32] = &[1, 2];
pub const SIMPLE_CHAIN_SRC_I64: &[i64] = &[0, 1];
pub const SIMPLE_CHAIN_DST_I64: &[i64] = &[1, 2];
pub const FOUR_VERTEX_CHAIN_SRC_I32: &[i32] = &[0, 1, 2];
pub const FOUR_VERTEX_CHAIN_DST_I32: &[i32] = &[1, 2, 3];
pub const FOUR_VERTEX_CHAIN_SRC_I64: &[i64] = &[0, 1, 2];
pub const FOUR_VERTEX_CHAIN_DST_I64: &[i64] = &[1, 2, 3];
pub const FIVE_VERTEX_CHAIN_SRC_I32: &[i32] = &[0, 1, 2, 3];
pub const FIVE_VERTEX_CHAIN_DST_I32: &[i32] = &[1, 2, 3, 4];
pub const TWO_CYCLE_SRC_I32: &[i32] = &[0, 1];
pub const TWO_CYCLE_DST_I32: &[i32] = &[1, 0];
pub const DISCONNECTED_TWO_CYCLES_SRC_I32: &[i32] = &[0, 1, 2, 3];
pub const DISCONNECTED_TWO_CYCLES_DST_I32: &[i32] = &[1, 0, 3, 2];
pub const DIRECTED_TRIANGLE_CYCLE_SRC_I32: &[i32] = &[0, 1, 2];
pub const DIRECTED_TRIANGLE_CYCLE_DST_I32: &[i32] = &[1, 2, 0];
pub const DIRECTED_TRIANGLE_CYCLE_SRC_I64: &[i64] = &[0, 1, 2];
pub const DIRECTED_TRIANGLE_CYCLE_DST_I64: &[i64] = &[1, 2, 0];
pub const TRIANGLE_VERTEX_IDS_I32: &[i32] = &[0, 1, 2];
pub const TRIANGLE_VERTEX_IDS_I64: &[i64] = &[0, 1, 2];
pub const TRIANGLE_TAIL_SRC_I32: &[i32] = &[0, 1, 2, 2];
pub const TRIANGLE_TAIL_DST_I32: &[i32] = &[1, 2, 0, 3];
pub const TRIANGLE_TAIL_SRC_I64: &[i64] = &[0, 1, 2, 2];
pub const TRIANGLE_TAIL_DST_I64: &[i64] = &[1, 2, 0, 3];
pub const TRIANGLE_TAIL_WEIGHT_F32: &[f32] = &[1.0, 2.0, 3.0, 4.0];
pub const SSSP_GRAPH_SRC_I32: &[i32] = &[0, 1, 1, 2, 2, 2, 3, 4];
pub const SSSP_GRAPH_DST_I32: &[i32] = &[1, 3, 4, 0, 1, 3, 5, 5];
pub const SSSP_GRAPH_WEIGHT_F32: &[f32] = &[0.1, 2.1, 1.1, 5.1, 3.1, 4.1, 7.2, 3.2];
pub const SSSP_GRAPH_WEIGHT_F64: &[f64] = &[0.1, 2.1, 1.1, 5.1, 3.1, 4.1, 7.2, 3.2];
pub const LOUVAIN_GRAPH_SRC_I32: &[i32] = &[0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5];
pub const LOUVAIN_GRAPH_DST_I32: &[i32] = &[1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4];
pub const LOUVAIN_GRAPH_WEIGHT_F32: &[f32] =
    &[0.1, 2.1, 1.1, 5.1, 3.1, 4.1, 7.2, 3.2, 0.1, 2.1, 1.1, 5.1, 3.1, 4.1, 7.2, 3.2];
pub const SIMILARITY_PAIR_FIRST_I32: &[i32] = &[0, 0, 0, 1, 1, 1, 2, 2, 2, 3];
pub const SIMILARITY_PAIR_SECOND_I32: &[i32] = &[1, 3, 4, 2, 3, 5, 3, 4, 5, 4];
pub const SIMILARITY_JACCARD_F32: &[f32] = &[0.2, 0.666667, 0.333333, 0.4, 0.166667, 0.5, 0.2, 0.25, 0.25, 0.666667];
pub const SIMILARITY_SORENSEN_F32: &[f32] =
    &[0.333333, 0.8, 0.5, 0.571429, 0.285714, 0.666667, 0.333333, 0.4, 0.4, 0.8];
pub const SIMILARITY_OVERLAP_F32: &[f32] = &[0.5, 1.0, 0.5, 0.666667, 0.333333, 1.0, 0.333333, 0.5, 0.5, 1.0];
pub const SIMILARITY_COSINE_F32: &[f32] = &[1.0; 10];
pub const ALL_PAIRS_JACCARD_FIRST_I32: &[i32] = &[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5];
pub const ALL_PAIRS_JACCARD_SECOND_I32: &[i32] = &[1, 2, 3, 4, 0, 2, 3, 5, 0, 1, 3, 4, 5, 0, 1, 2, 4, 0, 2, 3, 1, 2];
pub const ALL_PAIRS_JACCARD_F32: &[f32] = &[
    0.2, 0.25, 0.666667, 0.333333, 0.2, 0.4, 0.166667, 0.5, 0.25, 0.4, 0.2, 0.25, 0.25, 0.666667, 0.166667, 0.2,
    0.666667, 0.333333, 0.25, 0.666667, 0.5, 0.25,
];

pub const UNDIRECTED_TRIANGLE_SRC_I32: &[i32] = &[0, 1, 2, 1, 2, 0];
pub const UNDIRECTED_TRIANGLE_DST_I32: &[i32] = &[1, 2, 0, 0, 1, 2];
pub const UNDIRECTED_TRIANGLE_WEIGHT_F32: &[f32] = &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
pub const UNDIRECTED_TRIANGLE_SRC_I64: &[i64] = &[0, 1, 2, 1, 2, 0];
pub const UNDIRECTED_TRIANGLE_DST_I64: &[i64] = &[1, 2, 0, 0, 1, 2];
pub const UNDIRECTED_TRIANGLE_UNIT_WEIGHT_F64: &[f64] = &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

pub const DIRECTED_STAR_SRC_I32: &[i32] = &[1, 2, 3, 0, 0, 0];
pub const DIRECTED_STAR_DST_I32: &[i32] = &[0, 0, 0, 1, 2, 3];
pub const DIRECTED_STAR_SRC_I64: &[i64] = &[1, 2, 3, 0, 0, 0];
pub const DIRECTED_STAR_DST_I64: &[i64] = &[0, 0, 0, 1, 2, 3];
pub const DIRECTED_STAR_UNIT_WEIGHT_F64: &[f64] = &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

pub const DEGREE_QUERY_I32: &[i32] = &[0, 2];
pub const DEGREE_QUERY_I64: &[i64] = &[0, 2];
pub const DEGREE_SUBSET_SRC_I32: &[i32] = &[0, 1, 1, 2, 2, 0];
pub const DEGREE_SUBSET_DST_I32: &[i32] = &[1, 2, 3, 3, 0, 2];
pub const DEGREE_SUBSET_SRC_I64: &[i64] = &[0, 1, 2, 2];
pub const DEGREE_SUBSET_DST_I64: &[i64] = &[1, 2, 0, 3];
pub const SELF_LOOP_DROP_SRC_I32: &[i32] = &[0, 0, 1];
pub const SELF_LOOP_DROP_DST_I32: &[i32] = &[0, 1, 2];
pub const MULTI_EDGE_DROP_SRC_I32: &[i32] = &[0, 0, 0, 1];
pub const MULTI_EDGE_DROP_DST_I32: &[i32] = &[1, 1, 1, 2];

pub fn triangle_graph_dataframe(runtime: &cudf::Runtime) -> cudf::DataFrame {
    let src = runtime.series(UNDIRECTED_TRIANGLE_SRC_I32).expect("src");
    let dst = runtime.series(UNDIRECTED_TRIANGLE_DST_I32).expect("dst");
    let wgt = runtime.series(UNDIRECTED_TRIANGLE_WEIGHT_F32).expect("weight");
    cudf::DataFrame::from_series([("src", src), ("dst", dst), ("weight", wgt)]).expect("triangle graph dataframe")
}

pub fn bfs_distance_map_i32_result(handle: &EnteredContext<'_>, result: &BfsResult) -> HashMap<i32, i32> {
    let vertices = result.vertex_ids().expect("vertex_ids").to_vec_i32(handle).expect("vertices");
    let distances = result.distances().expect("distances").to_vec_i32(handle).expect("distances");
    vertices.into_iter().zip(distances).collect()
}

pub fn bfs_distance_map_i64_result(handle: &EnteredContext<'_>, result: &BfsResult) -> HashMap<i64, i64> {
    let vertices = result.vertex_ids().expect("vertex_ids").to_vec_i64(handle).expect("vertices");
    let distances = result.distances().expect("distances").to_vec_i64(handle).expect("distances");
    vertices.into_iter().zip(distances).collect()
}

pub fn bfs_distance_map_i32_graph(
    handle: &EnteredContext<'_>,
    graph: &mut Graph<Forward>,
    source: i32,
) -> HashMap<i32, i32> {
    let sources = DeviceArray::from_slice(handle, &[source]).expect("sources");
    let sources_view = sources.view().expect("sources_view");
    let result = graph.bfs(handle, &sources_view, &BfsParams::default()).expect("bfs");
    bfs_distance_map_i32_result(handle, &result)
}

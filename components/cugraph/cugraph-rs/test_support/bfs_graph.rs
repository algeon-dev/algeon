#![allow(dead_code)]

use std::path::Path;

use anyhow::Context;
use cudf::{DataFrame, Runtime, core::Table};

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
    DataFrame::from_columns([("src", src), ("dst", dst), ("weight", wgt)]).expect("bfs graph dataframe")
}

pub fn write_bfs_graph_parquet(path: &Path) -> anyhow::Result<()> {
    let runtime = Runtime::new().context("build cudf runtime for BFS fixture")?;
    let df = bfs_graph_dataframe(&runtime);
    df.write_parquet(path).context("write BFS fixture parquet")
}

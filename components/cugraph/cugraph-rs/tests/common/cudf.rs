use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use cudf::{core::Table, io::parquet};
use cugraph::{
    centrality::PageRankOptions,
    cudf_interop::{DataFrameGraphOptions, TableGraphOptions, graph_from_dataframe, graph_from_prepared_table_export},
    graph::{Forward, Graph, Transposed},
    handle::ResourceHandle,
    types::GraphProperties,
};

const EDGE_COLUMN_NAMES: &[&str] = &["src", "dst", "weight"];

use crate::common::bfs_graph::write_bfs_graph_parquet;
pub use crate::common::fs::unique_temp_dir;

pub fn explicit_runtime() -> cudf::Runtime {
    cudf::Runtime::builder().build().expect("runtime")
}

pub fn i32_series_device_ptr(series: &cudf::Series) -> usize {
    series.as_column_view().expect("column view").device_slice::<i32>().expect("i32 device slice").as_ptr() as usize
}

pub fn f32_series_device_ptr(series: &cudf::Series) -> usize {
    series.as_column_view().expect("column view").device_slice::<f32>().expect("f32 device slice").as_ptr() as usize
}

pub fn assert_named_edge_columns(df: &cudf::DataFrame) {
    assert_eq!(df.column_names().as_slice(), EDGE_COLUMN_NAMES);
}

pub fn build_directed_named_dataframe_graph(
    handle: &ResourceHandle,
    df: &cudf::DataFrame,
) -> cugraph::error::Result<Graph<Forward>> {
    let opts = DataFrameGraphOptions::new("src", "dst")
        .weight_col("weight")
        .properties(GraphProperties::directed())
        .renumber();
    graph_from_dataframe::<Forward>(handle, df, &opts)
}

pub fn build_undirected_named_dataframe_graph(
    handle: &ResourceHandle,
    df: &cudf::DataFrame,
) -> cugraph::error::Result<Graph<Transposed>> {
    let opts = DataFrameGraphOptions::new("src", "dst")
        .weight_col("weight")
        .properties(GraphProperties::undirected())
        .renumber();
    graph_from_dataframe::<Transposed>(handle, df, &opts)
}

pub fn build_undirected_prepared_graph(
    handle: &ResourceHandle,
    export: &cudf::interop::PreparedTableExport<'_>,
) -> cugraph::error::Result<Graph<Transposed>> {
    let opts = TableGraphOptions::new(0, 1).weight_col(2).properties(GraphProperties::undirected()).renumber();
    graph_from_prepared_table_export::<Transposed>(handle, export, &opts)
}

pub fn pagerank_score_map(handle: &ResourceHandle, graph: &mut Graph<Transposed>) -> HashMap<i32, f32> {
    let result = graph.pagerank(handle, &PageRankOptions::default()).expect("pagerank");
    let vertices: Vec<i32> = result.vertex_ids().expect("vertex_ids").to_vec_i32(handle).expect("vertices");
    let scores: Vec<f32> = result.scores().expect("scores").to_vec_f32(handle).expect("scores");
    vertices.into_iter().zip(scores).collect()
}

pub fn write_parquet_table(
    view: &cudf::core::TableView<'_>,
    path: &Path,
    column_names: &[&str],
    runtime: &cudf::Runtime,
) {
    let write_opts = parquet::ParquetTableWriteOptions::builder()
        .filepath(path)
        .column_names(column_names)
        .with_runtime(runtime)
        .build();
    parquet::write_parquet(view, &write_opts).expect("write parquet");
}

pub fn read_parquet_table(path: &Path, runtime: &cudf::Runtime) -> Table {
    let read_opts = parquet::ParquetTableReadOptions::builder().filepath(path).with_runtime(runtime).build();
    parquet::read_parquet_table(&read_opts).expect("read parquet").table
}

pub struct TempParquetFixture {
    dir: PathBuf,
    path: PathBuf,
}

impl TempParquetFixture {
    pub fn bfs_graph() -> Self {
        let dir = unique_temp_dir("bfs_graph");
        std::fs::create_dir_all(&dir).expect("create fixture dir");

        let path = dir.join("bfs_graph.parquet");
        write_bfs_graph_parquet(&path).expect("write BFS fixture parquet");

        Self { dir, path }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn path_str(&self) -> &str {
        self.path.to_str().expect("utf8 path")
    }
}

impl Drop for TempParquetFixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

use std::{
    collections::{BTreeSet, HashMap},
    path::{Path, PathBuf},
    sync::{Arc, Mutex, MutexGuard, OnceLock},
};

use cugraph::{
    array::DeviceArray,
    centrality::PageRankParams,
    cudf_interop::{TableGraphOptions, graph_from_prepared_table_export},
    graph::{Forward, Graph, GraphBuilder, Transposed},
    handle::{EnteredContext, ResourceHandle},
    types::GraphProperties,
};
use rapids_interop::PreparedTableExport;

const EDGE_COLUMN_NAMES: &[&str] = &["src", "dst", "weight"];

pub use crate::common::fs::unique_temp_dir;
use crate::common::support::write_bfs_graph_parquet;

pub fn cuda_test_serial_guard() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(())).lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

pub fn explicit_runtime() -> cudf::Runtime {
    cudf::Runtime::builder().build().expect("runtime")
}

pub fn explicit_runtime_owner() -> Arc<cudf::Runtime> {
    Arc::new(explicit_runtime())
}

pub fn runtime_bound_handle(runtime: &Arc<cudf::Runtime>) -> ResourceHandle {
    ResourceHandle::for_runtime(Arc::clone(runtime)).expect("runtime-bound handle")
}

pub fn i32_series_device_ptr(series: &cudf::Series) -> usize {
    series.as_column().device_slice::<i32>().expect("i32 device slice").as_ptr() as usize
}

pub fn f32_series_device_ptr(series: &cudf::Series) -> usize {
    series.as_column().device_slice::<f32>().expect("f32 device slice").as_ptr() as usize
}

pub fn assert_named_edge_columns(df: &cudf::DataFrame) {
    assert_eq!(df.column_names().as_slice(), EDGE_COLUMN_NAMES);
}

pub fn build_directed_named_dataframe_graph(
    handle: &EnteredContext<'_>,
    df: &cudf::DataFrame,
) -> cugraph::error::Result<Graph<Forward>> {
    let export = df.try_prepare_device_table_export().map_err(|error| {
        cugraph::Error::new(cugraph::ErrorKind::InvalidState, "build_directed_named_dataframe_graph")
            .with_message("failed to prepare cudf edge table")
            .set_source(error)
    })?;
    let opts = TableGraphOptions::new(0, 1).weight_col(2).properties(GraphProperties::directed()).renumber();
    graph_from_prepared_table_export::<Forward>(handle, &export, &opts)
}

pub fn build_undirected_named_dataframe_graph(
    handle: &EnteredContext<'_>,
    df: &cudf::DataFrame,
) -> cugraph::error::Result<Graph<Transposed>> {
    let export = df.try_prepare_device_table_export().map_err(|error| {
        cugraph::Error::new(cugraph::ErrorKind::InvalidState, "build_undirected_named_dataframe_graph")
            .with_message("failed to prepare cudf edge table")
            .set_source(error)
    })?;
    let opts = TableGraphOptions::new(0, 1).weight_col(2).properties(GraphProperties::undirected()).renumber();
    graph_from_prepared_table_export::<Transposed>(handle, &export, &opts)
}

pub fn build_undirected_prepared_graph(
    handle: &EnteredContext<'_>,
    export: &PreparedTableExport<'_>,
) -> cugraph::error::Result<Graph<Transposed>> {
    let opts = TableGraphOptions::new(0, 1).weight_col(2).properties(GraphProperties::undirected()).renumber();
    graph_from_prepared_table_export::<Transposed>(handle, export, &opts)
}

pub fn pagerank_score_map(handle: &EnteredContext<'_>, graph: &mut Graph<Transposed>) -> HashMap<i32, f32> {
    let result = graph.pagerank(handle, &PageRankParams::default()).expect("pagerank");
    let vertices: Vec<i32> = result.vertex_ids().expect("vertex_ids").to_vec_i32(handle).expect("vertices");
    let scores: Vec<f32> = result.scores().expect("scores").to_vec_f32(handle).expect("scores");
    vertices.into_iter().zip(scores).collect()
}

pub fn assert_score_maps_close_f32(
    actual: &HashMap<i32, f32>,
    expected: &HashMap<i32, f32>,
    tolerance: f32,
    label: &str,
) {
    let actual_vertices = actual.keys().copied().collect::<BTreeSet<_>>();
    let expected_vertices = expected.keys().copied().collect::<BTreeSet<_>>();
    assert_eq!(actual_vertices, expected_vertices, "{label} vertex ids mismatch");

    for (vertex, expected_score) in expected {
        let actual_score = actual.get(vertex).expect("score entry");
        assert!(
            (actual_score - expected_score).abs() <= tolerance,
            "{label} score mismatch for vertex {vertex}: expected {expected_score}, got {actual_score}"
        );
    }
}

pub fn python_weighted_reciprocal_dataframe(runtime: &cudf::Runtime) -> cudf::DataFrame {
    let src = runtime.series(&[0i32, 1, 1, 2]).expect("src");
    let dst = runtime.series(&[1i32, 0, 2, 1]).expect("dst");
    let wgt = runtime.series(&[5.0f32, 2.0, 1.0, 1.0]).expect("weight");
    cudf::DataFrame::from_series([("src", src), ("dst", dst), ("weight", wgt)])
        .expect("python-weighted reciprocal dataframe")
}

pub fn python_undirected_multigraph_dataframe(runtime: &cudf::Runtime) -> cudf::DataFrame {
    let src = runtime.series(&[0i32, 0, 1]).expect("src");
    let dst = runtime.series(&[1i32, 1, 2]).expect("dst");
    cudf::DataFrame::from_series([("src", src), ("dst", dst)]).expect("python undirected multigraph dataframe")
}

pub fn python_weighted_reciprocal_expected_pagerank_map(handle: &EnteredContext<'_>) -> HashMap<i32, f32> {
    let src = DeviceArray::from_slice(handle, &[0i32, 1, 1, 2]).expect("src");
    let dst = DeviceArray::from_slice(handle, &[1i32, 0, 2, 1]).expect("dst");
    let wgt = DeviceArray::from_slice(handle, &[2.0f32, 2.0, 1.0, 1.0]).expect("weight");
    let src_view = src.view().expect("src_view");
    let dst_view = dst.view().expect("dst_view");
    let wgt_view = wgt.view().expect("weight_view");

    let mut graph = GraphBuilder::new(&src_view, &dst_view)
        .weights(&wgt_view)
        .properties(GraphProperties::undirected())
        .renumber()
        .build::<Transposed>(handle)
        .expect("expected python-weighted graph");

    pagerank_score_map(handle, &mut graph)
}

pub fn write_parquet_dataframe(df: &cudf::DataFrame, path: &Path) {
    df.write_parquet(path).expect("write parquet");
}

pub fn read_parquet_dataframe(path: &Path, runtime: &cudf::Runtime) -> cudf::DataFrame {
    cudf::DataFrame::read_parquet(runtime, path).expect("read parquet")
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

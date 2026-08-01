use std::{fs, path::PathBuf};

use cugraph::{
    DeviceArray, EnteredContext, Forward, Graph, GraphBuilder, ResourceHandle, Transposed,
    centrality::{PageRankParams, PageRankResult},
    community::{LouvainParams, LouvainResult},
    cudf_interop::{
        DataFrameGraphOptions, LoadedGraph, ParquetGraphOptions, SeriesGraphOptions, graph_from_dataframe,
        graph_from_parquet, graph_from_series,
    },
    error::Result,
    layout::{ForceAtlas2Params, LayoutResult},
    random::RngState,
};
use static_assertions::assert_not_impl_any;

type DeviceArrayNewI32 = fn(&EnteredContext<'_>, usize) -> Result<DeviceArray<i32>>;
type DeviceArrayFromSliceI32 = fn(&EnteredContext<'_>, &[i32]) -> Result<DeviceArray<i32>>;
type BuildForwardGraph = fn(&GraphBuilder<'_>, &EnteredContext<'_>) -> Result<Graph<Forward>>;
type PageRankEntryPoint = fn(&mut Graph<Transposed>, &EnteredContext<'_>, &PageRankParams) -> Result<PageRankResult>;
type LouvainEntryPoint = fn(&mut Graph<Forward>, &EnteredContext<'_>, &LouvainParams) -> Result<LouvainResult>;
type ForceAtlas2EntryPoint =
    fn(&mut Graph<Forward>, &EnteredContext<'_>, &mut RngState, &ForceAtlas2Params) -> Result<LayoutResult>;
type RngStateNew = fn(&EnteredContext<'_>, u64) -> Result<RngState>;
type GraphFromDataFrameForward =
    fn(&EnteredContext<'_>, &cudf::DataFrame, &DataFrameGraphOptions<'_>) -> Result<Graph<Forward>>;
type GraphFromSeriesForward = fn(
    &EnteredContext<'_>,
    &cudf::Series,
    &cudf::Series,
    Option<&cudf::Series>,
    &SeriesGraphOptions,
) -> Result<Graph<Forward>>;
type GraphFromParquetForward = fn(&EnteredContext<'_>, &ParquetGraphOptions<'_>) -> Result<LoadedGraph<Forward>>;

#[test]
fn resource_handle_and_entered_context_are_not_send_or_sync() {
    assert_not_impl_any!(ResourceHandle: Send, Sync);
    assert_not_impl_any!(EnteredContext<'static>: Send, Sync);
}

fn build_forward(builder: &GraphBuilder<'_>, handle: &EnteredContext<'_>) -> Result<Graph<Forward>> {
    builder.build::<Forward>(handle)
}

#[test]
fn gpu_entry_points_require_entered_context() {
    let _: DeviceArrayNewI32 = DeviceArray::<i32>::new;
    let _: DeviceArrayFromSliceI32 = DeviceArray::<i32>::from_slice;
    let _: BuildForwardGraph = build_forward;
    let _: PageRankEntryPoint = Graph::<Transposed>::pagerank;
    let _: LouvainEntryPoint = Graph::<Forward>::louvain;
    let _: ForceAtlas2EntryPoint = Graph::<Forward>::force_atlas2;
    let _: RngStateNew = RngState::new;
}

#[test]
fn cudf_graph_ingress_entry_points_require_entered_context() {
    let _: GraphFromDataFrameForward = graph_from_dataframe::<Forward>;
    let _: GraphFromSeriesForward = graph_from_series::<Forward>;
    let _: GraphFromParquetForward = graph_from_parquet::<Forward>;
}

#[test]
fn runtime_bound_resource_handle_uses_no_secondary_stream_pool() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let shim = manifest_dir.join("shim").join("runtime_resource_handle.cpp");
    let source = fs::read_to_string(&shim).unwrap_or_else(|err| panic!("read {}: {err}", shim.display()));
    let normalized_source = source.split_whitespace().collect::<Vec<_>>().join(" ");

    assert!(
        normalized_source.contains(
            "std::make_unique<raft::handle_t>( stream, std::shared_ptr<rmm::cuda_stream_pool>{}, raft::mr::device_resource{rmm::mr::any_device_resource_from_raw(mr)})"
        ),
        "runtime-bound cuGraph handle should pass an empty RAFT stream pool so graph work remains on the runtime stream"
    );
    assert_eq!(
        source.matches("cuda_stream_pool").count(),
        1,
        "runtime-bound cuGraph handle must not introduce a secondary RAFT stream pool"
    );
}

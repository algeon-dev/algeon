use std::{collections::BTreeSet, fs, path::PathBuf};

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

fn rust_sources(root: &std::path::Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(root).unwrap_or_else(|error| panic!("read {}: {error}", root.display())) {
        let path = entry.expect("source directory entry").path();
        if path.is_dir() {
            rust_sources(&path, out);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            out.push(path);
        }
    }
}

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
    assert!(
        !source.contains("set_current_device_resource"),
        "runtime-bound graph work must not swap RMM's process-wide current resource"
    );
    assert!(
        !source.contains("runtime_graph_creation_memory_resource"),
        "the legacy global-RMM graph-construction bridge must stay removed"
    );

    let handle_source = fs::read_to_string(manifest_dir.join("src/handle.rs")).expect("runtime handle source");
    assert!(
        handle_source.contains("route_current_resource(runtime.stream(), runtime.memory_resource())"),
        "runtime-bound handles must register their explicit resource through cuDF's stable current-resource route"
    );
}

#[test]
fn every_device_bound_owner_uses_the_typed_destroy_bridge() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let shim = fs::read_to_string(manifest_dir.join("shim/runtime_resource_handle.cpp")).expect("native destroy shim");
    assert!(shim.contains("cugraph_rs_destroy_device_resource"));
    assert!(shim.contains("CUGRAPH_RS_DESTROY_FATAL_CUDA"));

    let direct_void_destructors = [
        "cugraph_similarity_result_free(",
        "cugraph_vertex_pairs_free(",
        "cugraph_rng_state_free(",
        "cugraph_core_result_free(",
        "cugraph_k_core_result_free(",
        "cugraph_degrees_result_free(",
        "cugraph_induced_subgraph_result_free(",
        "cugraph_layout_result_free(",
        "cugraph_hierarchical_clustering_result_free(",
        "cugraph_triangle_count_result_free(",
        "cugraph_random_walk_result_free(",
        "cugraph_sample_result_free(",
        "cugraph_clustering_result_free(",
        "cugraph_edgelist_free(",
        "cugraph_graph_free(",
        "cugraph_bfs_predicate_result_free(",
        "cugraph_paths_result_free(",
        "cugraph_extract_paths_result_free(",
        "cugraph_free_resource_handle(",
        "cugraph_centrality_result_free(",
        "cugraph_edge_centrality_result_free(",
        "cugraph_hits_result_free(",
        "cugraph_labeling_result_free(",
        "cugraph_type_erased_device_array_free(",
    ];
    let mut sources = Vec::new();
    rust_sources(&manifest_dir.join("src"), &mut sources);
    for source_path in sources {
        let source = fs::read_to_string(&source_path).expect("Rust source");
        for destructor in direct_void_destructors {
            assert!(
                !source.contains(destructor),
                "{} bypasses typed native teardown through {destructor}",
                source_path.display()
            );
        }
    }
}

#[test]
fn direct_void_destructors_are_limited_to_proven_host_metadata() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let expected = BTreeSet::from([
        "cugraph_error_free".to_owned(),
        "cugraph_sampling_options_free".to_owned(),
        "cugraph_type_erased_device_array_view_free".to_owned(),
        "cugraph_type_erased_host_array_view_free".to_owned(),
    ]);

    let mut sources = Vec::new();
    rust_sources(&manifest_dir.join("src"), &mut sources);
    let mut found = BTreeSet::new();
    for source_path in sources {
        let source = fs::read_to_string(&source_path).expect("Rust source");
        for (start, _) in source.match_indices("cugraph_") {
            let candidate = &source[start..];
            let name_len = candidate
                .find(|character: char| !(character.is_ascii_alphanumeric() || character == '_'))
                .unwrap_or(candidate.len());
            let name = &candidate[..name_len];
            if name.ends_with("_free") && candidate[name_len..].starts_with('(') {
                found.insert(name.to_owned());
            }
        }
    }
    assert_eq!(
        found, expected,
        "a direct upstream void destructor was added or a proven host-metadata exception changed"
    );

    // Pin the upstream implementation evidence behind the allowlist. If any
    // of these objects starts owning a CUDA resource, this contract must fail
    // and the Rust owner must move behind the typed device teardown bridge.
    let upstream = manifest_dir.parent().expect("workspace root").join("cugraph/cpp/src/c_api");
    let array_cpp = fs::read_to_string(upstream.join("array.cpp")).expect("upstream array.cpp");
    let array_hpp = fs::read_to_string(upstream.join("array.hpp")).expect("upstream array.hpp");
    let error_cpp = fs::read_to_string(upstream.join("error.cpp")).expect("upstream error.cpp");
    let error_hpp = fs::read_to_string(upstream.join("error.hpp")).expect("upstream error.hpp");
    let sampling_cpp = fs::read_to_string(upstream.join("sampling_result.cpp")).expect("upstream sampling_result.cpp");
    let sampling_hpp = fs::read_to_string(upstream.join("sampling_common.hpp")).expect("upstream sampling_common.hpp");

    assert!(array_cpp.contains("cugraph_type_erased_device_array_view_free"));
    assert!(array_cpp.contains("cugraph_type_erased_host_array_view_free"));
    assert!(array_hpp.contains("struct cugraph_type_erased_device_array_view_t {\n  void* data_;"));
    assert!(array_hpp.contains("struct cugraph_type_erased_host_array_view_t {\n  std::byte* data_;"));
    assert!(error_cpp.contains("cugraph_error_free"));
    assert!(error_hpp.contains("struct cugraph_error_t {\n  std::string error_message_{};"));
    assert!(sampling_cpp.contains("cugraph_sampling_options_free"));
    let sampling_options = sampling_hpp
        .split_once("struct cugraph_sampling_options_t {")
        .and_then(|(_, tail)| tail.split_once("struct sampling_flags_t"))
        .map(|(body, _)| body)
        .expect("sampling-options definition");
    assert!(!sampling_options.contains("device_"));
    assert!(!sampling_options.contains("rmm::"));
}

#[test]
fn rust_and_native_destroy_kinds_have_identical_discriminants() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let rust = fs::read_to_string(manifest_dir.join("src/cuda.rs")).expect("Rust destroy kinds");
    let native = fs::read_to_string(manifest_dir.join("shim/runtime_resource_handle.h")).expect("native destroy kinds");
    let kinds = [
        ("SimilarityResult", "SIMILARITY_RESULT", 1),
        ("VertexPairs", "VERTEX_PAIRS", 2),
        ("RngState", "RNG_STATE", 3),
        ("CoreResult", "CORE_RESULT", 4),
        ("KCoreResult", "K_CORE_RESULT", 5),
        ("DegreesResult", "DEGREES_RESULT", 6),
        ("InducedSubgraphResult", "INDUCED_SUBGRAPH_RESULT", 7),
        ("LayoutResult", "LAYOUT_RESULT", 8),
        ("HierarchicalClusteringResult", "HIERARCHICAL_CLUSTERING_RESULT", 9),
        ("TriangleCountResult", "TRIANGLE_COUNT_RESULT", 10),
        ("RandomWalkResult", "RANDOM_WALK_RESULT", 11),
        ("SampleResult", "SAMPLE_RESULT", 12),
        ("ClusteringResult", "CLUSTERING_RESULT", 13),
        ("Edgelist", "EDGELIST", 14),
        ("Graph", "GRAPH", 15),
        ("BfsPredicateResult", "BFS_PREDICATE_RESULT", 16),
        ("PathsResult", "PATHS_RESULT", 17),
        ("ExtractPathsResult", "EXTRACT_PATHS_RESULT", 18),
        ("RuntimeResourceHandle", "RUNTIME_RESOURCE_HANDLE", 19),
        ("ResourceHandle", "RESOURCE_HANDLE", 20),
        ("CentralityResult", "CENTRALITY_RESULT", 21),
        ("EdgeCentralityResult", "EDGE_CENTRALITY_RESULT", 22),
        ("HitsResult", "HITS_RESULT", 23),
        ("LabelingResult", "LABELING_RESULT", 24),
        ("DeviceArray", "DEVICE_ARRAY", 26),
    ];
    for (rust_name, native_name, value) in kinds {
        assert!(rust.contains(&format!("{rust_name} = {value},")), "missing Rust destroy kind {rust_name}");
        assert!(
            native.contains(&format!("CUGRAPH_RS_DEVICE_RESOURCE_{native_name} = {value}")),
            "missing native destroy kind {native_name}"
        );
    }
}

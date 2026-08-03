//! Interop between cudf and cugraph types.
//!
//! # Memory ownership model
//!
//! Functions in this module differ in whether they copy data or create
//! zero-copy views. The distinction matters for lifetime management:
//!
//! `GraphConstructionPolicy::PythonCugraph` adds one extra wrinkle: some
//! ingress paths normalize edge data before raw graph construction. For
//! borrowed and prepared compatibility inputs, that normalization may import
//! the selected edge columns into owned cudf objects on the consumer device
//! before building the graph.
//!
//! ## Primary shared-contract paths
//!
//! - [`graph_from_prepared_columns()`] — consumes single-GPU prepared-device
//!   exports via explicit read-lease activation.
//! - [`graph_from_prepared_table_export()`] — same as above for a prepared
//!   table export plus explicit column selection.
//! - [`vertex_pairs_from_prepared_columns()`] — creates explicit pairwise
//!   similarity input directly from two prepared device columns.
//! - [`vertex_pairs_from_prepared_table_export()`] — same as above for a
//!   prepared table export plus explicit pair-column selection.
//!
//! ## Compatibility zero-copy view paths
//!
//! - [`device_array_view_from_slice()`] — wraps a cudf column's GPU memory as a
//!   typed cugraph array view without copying, with a runtime type check.
//! - [`graph_from_borrowed_columns()`] — consumes the shared RAPIDS borrowed
//!   ABI directly.
//! - [`graph_from_borrowed_table_view()`] — borrowed-compatibility
//!   construction from an explicit RAPIDS borrowed table descriptor.
//! - [`graph_from_series()`] and [`graph_from_dataframe()`] — convenience
//!   producers that prepare device exports at the API boundary, then delegate
//!   into the prepared-export graph construction path.
//!
//! ## Copy-based paths
//!
//! - [`IntoSeries`] trait — converts cugraph result arrays to cudf Series.
//!   Performs a device-to-device copy (unavoidable: cugraph owns result memory).
//! - [`graph_from_parquet()`] — reads Parquet from disk into GPU memory
//!   (I/O + copy). Returns a [`LoadedGraph`] that owns both the [`Graph`](crate::Graph)
//!   and the backing table.
//!
//! ## Summary
//!
//! | Function | Copy? | Lifetime constraint |
//! |----------|-------|---------------------|
//! | [`device_array_view_from_slice`] | No | View borrows slice |
//! | [`graph_from_prepared_columns`] | Depends on policy | Producer export stays alive until explicit lease completion |
//! | [`graph_from_prepared_table_export`] | Depends on policy | Producer export stays alive until explicit lease completion |
//! | [`vertex_pairs_from_prepared_columns`] | No | Pair object retains producer owners until explicit lease completion |
//! | [`vertex_pairs_from_prepared_table_export`] | No | Pair object retains producer owner until explicit lease completion |
//! | [`graph_from_borrowed_columns`] | Depends on policy | Borrowed descriptors must outlive build |
//! | [`graph_from_series`] | No (input) | Series must outlive build |
//! | [`graph_from_dataframe`] | No (input) | DataFrame must outlive build |
//! | [`graph_from_borrowed_table_view`] | Depends on policy | Borrowed descriptors must outlive build |
//! | [`IntoSeries::to_series`] | Yes (D2D) | Series is independent |
//! | [`graph_from_parquet`] | Yes | [`LoadedGraph`] owns both |
//!
//! # Single-GPU contract position
//!
//! For new single-GPU consumer integrations, prefer the prepared-device export
//! path because it keeps read-start, completion, and producer liveness explicit
//! at the shared contract layer.
//!
//! The borrowed graph-construction entry points remain supported as
//! compatibility zero-copy paths, especially for callers that already operate
//! on explicit borrowed descriptors. Runtime-bound borrowed paths use
//! device-side ordering; non-runtime-bound compatibility paths may still rely
//! on host fencing.
//!
//! # Borrowed compatibility policy
//!
//! Borrowed graph construction is explicit-only and same-device-only:
//!
//! - cudf producers must expose explicit execution provenance
//! - ambient/default-only cudf objects are rejected
//! - all borrowed inputs must live on the same CUDA device
//! - that device must match the [`ResourceHandle`](crate::ResourceHandle) device
//! - on runtime-bound handles, same-device different-stream producers insert
//!   device-side ordering into the runtime stream before cuGraph reads
//!   borrowed memory
//! - on non-runtime-bound handles, compatibility borrowed imports may still
//!   fence producer streams on the host before cuGraph reads borrowed memory
//!
//! [`DeviceSlice`]: cudf::core::DeviceSlice

mod borrowed;
mod bridge;
mod egress;
mod errors;
mod ingress;
mod options;
mod parquet;
mod preflight;
mod prepared;
mod types;

pub use borrowed::{BorrowedInputOrderingPlan, testing};
pub use bridge::device_array_view_from_slice;
pub use egress::IntoSeries;
pub(crate) use egress::{
    betweenness_result_to_dataframe, bfs_result_to_dataframe, core_number_result_to_dataframe,
    cosine_similarity_result_to_dataframe, degrees_result_to_dataframe, ecg_result_to_dataframe,
    edge_betweenness_centrality_result_to_dataframe, eigenvector_centrality_result_to_dataframe,
    extract_ego_result_to_dataframe, hits_result_to_dataframe, in_degrees_result_to_dataframe,
    jaccard_similarity_result_to_dataframe, k_core_result_to_dataframe, k_truss_subgraph_result_to_dataframe,
    katz_centrality_result_to_dataframe, layout_result_to_dataframe, leiden_result_to_dataframe,
    louvain_result_to_dataframe, minimum_spanning_tree_result_to_dataframe, out_degrees_result_to_dataframe,
    overlap_similarity_result_to_dataframe, page_rank_result_to_dataframe, sorensen_similarity_result_to_dataframe,
    spectral_clustering_result_to_dataframe, sssp_result_to_dataframe,
    strongly_connected_components_result_to_dataframe, triangle_count_result_to_dataframe, vertex_pairs_to_dataframe,
    weakly_connected_components_result_to_dataframe,
};
pub use ingress::{
    graph_from_borrowed_columns, graph_from_borrowed_table_view, graph_from_dataframe, graph_from_prepared_columns,
    graph_from_prepared_table_export, graph_from_series, graph_from_table_view, vertex_pairs_from_prepared_columns,
    vertex_pairs_from_prepared_table_export,
};
pub use options::{
    DataFrameGraphOptions, GraphConstructionPolicy, ParquetGraphOptions, SeriesGraphOptions, TableGraphOptions,
    TableVertexPairOptions,
};
pub use parquet::{LoadedGraph, graph_from_parquet};
pub use preflight::{
    BfsSideInputPreflight, BfsSideInputPreflightInput, EdgelistSortPreflightInput,
    PersonalizedPageRankSideInputPreflight, PersonalizedPageRankSideInputPreflightInput,
    bfs_side_input_workspace_preflight, edgelist_sort_workspace_preflight,
    personalized_pagerank_side_input_workspace_preflight,
};
pub use types::{cudf_type_to_cugraph, cugraph_type_to_cudf};

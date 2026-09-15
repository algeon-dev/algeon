//! Canonical cuDF interoperability for the Nexus cuGraph backend.
//!
//! Graph construction consumes prepared device exports with explicit producer
//! readiness, read-lease activation, same-device completion ordering, and
//! retained owner lifetimes. Result egress copies cuGraph-owned arrays into
//! runtime-bound cuDF objects. Raw device-slice views remain available only
//! for algorithm side inputs that are not graph-construction tables.

mod bridge;
mod egress;
mod errors;
mod ingress;
mod options;
mod prepared;
mod prepared_support;
mod types;

pub use bridge::device_array_view_from_slice;
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
    graph_from_prepared_columns, graph_from_prepared_table_export, vertex_pairs_from_prepared_columns,
    vertex_pairs_from_prepared_table_export,
};
pub use options::{GraphConstructionPolicy, PreparedGraphOptions, TableGraphOptions, TableVertexPairOptions};
pub use types::{cudf_type_to_cugraph, cugraph_type_to_cudf};

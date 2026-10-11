mod bridge_owner;
#[path = "../support/common/mod.rs"]
mod common;
mod egress_owner;

use cudf::testing::run_named_cases;

#[test]
fn egress_cases_1() {
    run_named_cases(&[
        ("spectral_clustering_result_to_dataframe", egress_owner::spectral_clustering_result_to_dataframe),
        (
            "strongly_connected_components_result_to_dataframe",
            egress_owner::strongly_connected_components_result_to_dataframe,
        ),
        (
            "centrality_to_dataframe_write_parquet_roundtrip",
            egress_owner::centrality_to_dataframe_write_parquet_roundtrip,
        ),
        (
            "louvain_result_preserves_external_vertex_ids_for_renumbered_graphs",
            egress_owner::louvain_result_preserves_external_vertex_ids_for_renumbered_graphs,
        ),
        ("ecg_result_to_dataframe", egress_owner::ecg_result_to_dataframe),
        ("edge_betweenness_result_to_dataframe", egress_owner::edge_betweenness_result_to_dataframe),
        ("similarity_result_to_dataframe", egress_owner::similarity_result_to_dataframe),
        ("triangle_count_result_to_dataframe", egress_owner::triangle_count_result_to_dataframe),
        ("layout_result_to_dataframe", egress_owner::layout_result_to_dataframe),
        ("subgraph_edge_results_to_dataframe", egress_owner::subgraph_edge_results_to_dataframe),
        ("betweenness_result_to_dataframe", egress_owner::betweenness_result_to_dataframe),
        ("extract_ego_result_to_dataframe", egress_owner::extract_ego_result_to_dataframe),
        (
            "layout_result_preserves_external_vertex_ids_for_renumbered_graphs",
            egress_owner::layout_result_preserves_external_vertex_ids_for_renumbered_graphs,
        ),
        (
            "pagerank_result_preserves_external_vertex_ids_for_renumbered_graphs",
            egress_owner::pagerank_result_preserves_external_vertex_ids_for_renumbered_graphs,
        ),
        ("leiden_result_to_dataframe", egress_owner::leiden_result_to_dataframe),
        (
            "weakly_connected_components_result_to_dataframe",
            egress_owner::weakly_connected_components_result_to_dataframe,
        ),
        ("sssp_result_to_dataframe_with_predecessors", egress_owner::sssp_result_to_dataframe_with_predecessors),
        (
            "unweighted_subgraph_edge_results_to_dataframe_omit_weight_column",
            egress_owner::unweighted_subgraph_edge_results_to_dataframe_omit_weight_column,
        ),
        ("core_number_result_to_dataframe", egress_owner::core_number_result_to_dataframe),
        ("bfs_result_to_dataframe_with_predecessors", egress_owner::bfs_result_to_dataframe_with_predecessors),
        ("louvain_result_to_dataframe", egress_owner::louvain_result_to_dataframe),
        ("hits_result_to_dataframe", egress_owner::hits_result_to_dataframe),
        ("graph_bfs_host_slice_matches_typed_view_path", bridge_owner::graph_bfs_host_slice_matches_typed_view_path),
        ("device_array_view_from_slice_rejects_mismatch", bridge_owner::device_array_view_from_slice_rejects_mismatch),
        ("device_array_view_from_slice_i64", bridge_owner::device_array_view_from_slice_i64),
    ]);
}

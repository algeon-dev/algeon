use std::{fs, path::PathBuf};

fn bindings_rs_path() -> PathBuf {
    if let Some(path) = option_env!("CUGRAPH_SYS_BINDINGS_RS") {
        PathBuf::from(path)
    } else {
        panic!("CUGRAPH_SYS_BINDINGS_RS is not set; build.rs should export it for coverage tests");
    }
}

fn generated_bindings() -> String {
    let path = bindings_rs_path();
    fs::read_to_string(&path).unwrap_or_else(|_| panic!("failed to read generated bindings at {}", path.display()))
}

#[test]
fn bindings_include_critical_core_symbols() {
    let bindings = generated_bindings();

    let required_symbols = [
        // Resource handle lifecycle
        "cugraph_create_resource_handle",
        "cugraph_free_resource_handle",
        "cugraph_resource_handle_get_rank",
        "cugraph_resource_handle_get_comm_size",
        // Type-erased arrays
        "cugraph_type_erased_device_array_create",
        "cugraph_type_erased_device_array_free",
        "cugraph_type_erased_device_array_view",
        "cugraph_type_erased_device_array_view_copy_from_host",
        "cugraph_type_erased_device_array_view_copy_to_host",
        // Graph lifecycle
        "cugraph_graph_create_with_times_sg",
        "cugraph_graph_free",
        "cugraph_decompress_to_edgelist",
        "cugraph_edgelist_get_sources",
        "cugraph_edgelist_get_destinations",
        "cugraph_edgelist_free",
        // Centrality
        "cugraph_pagerank",
        "cugraph_pagerank_allow_nonconvergence",
        "cugraph_personalized_pagerank",
        "cugraph_personalized_pagerank_allow_nonconvergence",
        "cugraph_eigenvector_centrality",
        "cugraph_katz_centrality",
        "cugraph_centrality_result_get_vertices",
        "cugraph_centrality_result_get_values",
        "cugraph_centrality_result_get_num_iterations",
        "cugraph_centrality_result_converged",
        "cugraph_centrality_result_free",
        "cugraph_hits",
        "cugraph_hits_result_get_vertices",
        "cugraph_hits_result_get_hubs",
        "cugraph_hits_result_get_authorities",
        "cugraph_hits_result_get_hub_score_differences",
        "cugraph_hits_result_get_number_of_iterations",
        "cugraph_hits_result_free",
        "cugraph_edge_betweenness_centrality",
        "cugraph_edge_centrality_result_get_src_vertices",
        "cugraph_edge_centrality_result_get_dst_vertices",
        "cugraph_edge_centrality_result_get_edge_ids",
        "cugraph_edge_centrality_result_get_values",
        "cugraph_edge_centrality_result_free",
        // Traversal
        "cugraph_bfs",
        "cugraph_sssp",
        "cugraph_paths_result_get_vertices",
        "cugraph_paths_result_get_distances",
        "cugraph_paths_result_get_predecessors",
        "cugraph_paths_result_free",
        "cugraph_extract_paths",
        "cugraph_extract_paths_result_get_max_path_length",
        "cugraph_extract_paths_result_get_paths",
        "cugraph_extract_paths_result_free",
        // Community
        "cugraph_louvain",
        "cugraph_leiden",
        "cugraph_ecg",
        "cugraph_hierarchical_clustering_result_get_vertices",
        "cugraph_hierarchical_clustering_result_get_clusters",
        "cugraph_hierarchical_clustering_result_get_modularity",
        "cugraph_hierarchical_clustering_result_free",
        "cugraph_triangle_count",
        "cugraph_triangle_count_result_get_vertices",
        "cugraph_triangle_count_result_get_counts",
        "cugraph_triangle_count_result_free",
        "cugraph_balanced_cut_clustering",
        "cugraph_spectral_modularity_maximization",
        "cugraph_analyze_clustering_modularity",
        "cugraph_analyze_clustering_edge_cut",
        "cugraph_analyze_clustering_ratio_cut",
        "cugraph_clustering_result_get_vertices",
        "cugraph_clustering_result_get_clusters",
        "cugraph_clustering_result_free",
        // Sampling
        "cugraph_uniform_random_walks",
        "cugraph_biased_random_walks",
        "cugraph_node2vec_random_walks",
        "cugraph_random_walk_result_get_max_path_length",
        "cugraph_random_walk_result_get_paths",
        "cugraph_random_walk_result_get_weights",
        "cugraph_random_walk_result_get_path_sizes",
        "cugraph_random_walk_result_free",
        "cugraph_select_random_vertices",
        // Labeling
        "cugraph_weakly_connected_components",
        "cugraph_strongly_connected_components",
        "cugraph_labeling_result_get_vertices",
        "cugraph_labeling_result_get_labels",
        "cugraph_labeling_result_free",
        // Core
        "cugraph_core_number",
        "cugraph_core_result_get_vertices",
        "cugraph_core_result_get_core_numbers",
        "cugraph_core_result_free",
        "cugraph_k_core",
        "cugraph_k_core_result_get_src_vertices",
        "cugraph_k_core_result_get_dst_vertices",
        "cugraph_k_core_result_get_weights",
        "cugraph_k_core_result_free",
        // Subgraph / tree
        "cugraph_minimum_spanning_tree",
        "cugraph_k_truss_subgraph",
        "cugraph_extract_ego",
        "cugraph_induced_subgraph_get_sources",
        "cugraph_induced_subgraph_get_destinations",
        "cugraph_induced_subgraph_get_edge_weights",
        "cugraph_induced_subgraph_get_edge_ids",
        "cugraph_induced_subgraph_get_edge_type_ids",
        "cugraph_induced_subgraph_get_subgraph_offsets",
        "cugraph_induced_subgraph_result_free",
        // Similarity / vertex pairs
        "cugraph_create_vertex_pairs",
        "cugraph_vertex_pairs_get_first",
        "cugraph_vertex_pairs_get_second",
        "cugraph_vertex_pairs_free",
        "cugraph_similarity_result_get_vertex_pairs",
        "cugraph_similarity_result_get_similarity",
        "cugraph_similarity_result_free",
        "cugraph_jaccard_coefficients",
        "cugraph_sorensen_coefficients",
        "cugraph_overlap_coefficients",
        "cugraph_cosine_similarity_coefficients",
        "cugraph_all_pairs_jaccard_coefficients",
        "cugraph_all_pairs_sorensen_coefficients",
        "cugraph_all_pairs_overlap_coefficients",
        "cugraph_all_pairs_cosine_similarity_coefficients",
    ];

    for symbol in required_symbols {
        let needle = format!("pub fn {symbol}(");
        assert!(bindings.contains(&needle), "generated bindings missing critical symbol: {symbol}");
    }
}

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
        // Type-erased arrays
        "cugraph_type_erased_device_array_create",
        "cugraph_type_erased_device_array_free",
        "cugraph_type_erased_device_array_view",
        "cugraph_type_erased_device_array_view_copy_from_host",
        "cugraph_type_erased_device_array_view_copy_to_host",
        // Graph lifecycle
        "cugraph_graph_create_with_times_sg",
        "cugraph_graph_free",
        // Centrality
        "cugraph_pagerank",
        "cugraph_centrality_result_get_vertices",
        "cugraph_centrality_result_get_values",
        "cugraph_centrality_result_free",
        // Traversal
        "cugraph_bfs",
        "cugraph_paths_result_get_vertices",
        "cugraph_paths_result_get_distances",
        "cugraph_paths_result_free",
    ];

    for symbol in required_symbols {
        let needle = format!("pub fn {symbol}(");
        assert!(bindings.contains(&needle), "generated bindings missing critical symbol: {symbol}");
    }
}

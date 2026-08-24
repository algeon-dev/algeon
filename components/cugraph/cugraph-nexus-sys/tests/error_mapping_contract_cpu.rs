use std::{fs, path::PathBuf};

#[test]
fn c_api_run_algorithm_maps_allocator_failures_to_alloc_error() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let utils = manifest_dir.join("..").join("cugraph").join("cpp").join("src").join("c_api").join("utils.hpp");
    let source = fs::read_to_string(&utils).unwrap_or_else(|err| panic!("read {}: {err}", utils.display()));

    assert_allocator_failures_map_to_alloc_error(&source, "run_algorithm");
}

#[test]
fn c_api_graph_construction_maps_allocator_failures_to_alloc_error() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for relative_path in [
        ["..", "cugraph", "cpp", "src", "c_api", "graph_sg.cpp"],
        ["..", "cugraph", "cpp", "src", "c_api", "graph_mg.cpp"],
    ] {
        let path = relative_path.iter().fold(manifest_dir.clone(), |path, segment| path.join(segment));
        let source = fs::read_to_string(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
        let label = path.to_string_lossy().to_string();
        assert_no_local_allocator_catch(&source, &label);
        assert!(
            source.contains("run_with_error_boundary"),
            "{label}: graph construction entry points must route allocator failures through the shared boundary"
        );
    }
}

#[test]
fn c_api_mg_graph_type_resolution_collectives_map_allocator_failures_to_alloc_error() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let graph_mg = manifest_dir.join("..").join("cugraph").join("cpp").join("src").join("c_api").join("graph_mg.cpp");
    let source = fs::read_to_string(&graph_mg).unwrap_or_else(|err| panic!("read {}: {err}", graph_mg.display()));

    for call in [
        "num_edges = cugraph::host_scalar_allreduce",
        "vertex_types = cugraph::host_scalar_allgather",
        "weight_types = cugraph::host_scalar_allgather",
        "time_types = cugraph::host_scalar_allgather",
        "edge_type_id_types = cugraph::host_scalar_allgather",
    ] {
        assert_allocator_guarded_call(&source, call);
    }
}

/// The shared boundary in `utils.hpp` owns the catch ordering, so no C entry
/// point may classify allocator failures itself: a local catch would shadow the
/// boundary and drop the attributed payload it exists to preserve.
fn assert_no_local_allocator_catch(source: &str, label: &str) {
    for catch in ["catch (rmm::out_of_memory const&", "catch (rmm::bad_alloc const&", "catch (std::bad_alloc const&"] {
        assert!(
            !source.contains(catch),
            "{label}: `{catch}` must live only in the shared run_with_error_boundary, not at a C entry point"
        );
    }
}

fn assert_allocator_failures_map_to_alloc_error(source: &str, label: &str) {
    let generic_catches =
        source.match_indices("catch (std::exception const&").map(|(index, _)| index).collect::<Vec<_>>();
    assert!(!generic_catches.is_empty(), "{label} should keep a generic std::exception fallback");

    for generic_exception in generic_catches {
        let try_start = source[..generic_exception]
            .rfind("try {")
            .unwrap_or_else(|| panic!("{label}: generic exception fallback should be inside a try block"));
        let catch_region = &source[try_start..generic_exception];
        let rmm_out_of_memory = catch_region
            .find("catch (rmm::out_of_memory const&")
            .unwrap_or_else(|| panic!("{label}: RMM out-of-memory failures should be classified explicitly"));
        let rmm_bad_alloc = catch_region
            .find("catch (rmm::bad_alloc const&")
            .unwrap_or_else(|| panic!("{label}: RMM allocator failures should be classified explicitly"));
        let std_bad_alloc = catch_region
            .find("catch (std::bad_alloc const&")
            .unwrap_or_else(|| panic!("{label}: standard allocator failures should be classified explicitly"));

        assert!(rmm_out_of_memory < rmm_bad_alloc, "{label}: rmm::out_of_memory must be caught before rmm::bad_alloc");
        assert!(rmm_bad_alloc < std_bad_alloc, "{label}: rmm::bad_alloc must be caught before std::bad_alloc");
        assert!(
            catch_region[rmm_out_of_memory..rmm_bad_alloc].contains("return CUGRAPH_ALLOC_ERROR;"),
            "{label}: RMM out-of-memory failures should map to CUGRAPH_ALLOC_ERROR"
        );
        assert!(
            catch_region[rmm_bad_alloc..std_bad_alloc].contains("return CUGRAPH_ALLOC_ERROR;"),
            "{label}: RMM allocator failures should map to CUGRAPH_ALLOC_ERROR"
        );
        assert!(
            catch_region[std_bad_alloc..].contains("return CUGRAPH_ALLOC_ERROR;"),
            "{label}: std::bad_alloc should map to CUGRAPH_ALLOC_ERROR"
        );
    }
}

/// These collectives allocate before graph dispatch, so each one must sit inside
/// a `run_with_error_boundary` lambda rather than ahead of it: a failure raised
/// before the boundary opens escapes the C API as an unclassified exception.
fn assert_allocator_guarded_call(source: &str, call: &str) {
    let matches = source.match_indices(call).map(|(index, _)| index).collect::<Vec<_>>();
    assert!(!matches.is_empty(), "{call} should appear in graph_mg.cpp");

    for call_index in matches {
        let boundary_start = source[..call_index]
            .rfind("run_with_error_boundary")
            .unwrap_or_else(|| panic!("{call} should be inside a run_with_error_boundary lambda"));
        assert!(
            source[boundary_start..call_index].contains("[&]"),
            "{call} should be inside the boundary's lambda body, not among its arguments"
        );
    }
}

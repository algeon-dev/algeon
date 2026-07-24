use std::mem::size_of;

#[path = "../support/public_surface.rs"]
mod public_surface;

use public_surface::{assert_modules_are_private, assert_no_crate_root_reexports, assert_no_removed_public_surface};

#[test]
fn canonical_adapter_paths_expose_stable_contract_symbols() {
    let _rule_name: &str = datafusion_nexus::session::NEXUS_NATIVE_PHYSICAL_OPTIMIZER_RULE_NAME;
    let _metric_names: &[&str] = datafusion_nexus::native_exec::NATIVE_NEXUS_DATAFUSION_METRIC_NAMES;
    let _headers: [&str; 4] = [
        datafusion_nexus::native_report::NATIVE_DATAFUSION_EXECUTION_TSV_HEADER,
        datafusion_nexus::native_report::NATIVE_DATAFUSION_PLANNING_TSV_HEADER,
        datafusion_nexus::native_report::NATIVE_FALLBACK_PRESSURE_TSV_HEADER,
        datafusion_nexus::native_report::NATIVE_TPCDS_PARITY_TSV_HEADER,
    ];
    let _wrappers = (
        size_of::<datafusion_nexus::native_exec::NexusNativeExec>(),
        size_of::<datafusion_nexus::native_exec::NexusNativeComposedExec>(),
        size_of::<datafusion_nexus::native_exec::NexusNativeGpuProducerExec>(),
    );
    let _planning_report = datafusion_nexus::report::PlanningReport::default();
    let _workload_contracts = (
        datafusion_nexus_workloads::NEXUS_PARQUET_DIR_ENV,
        datafusion_nexus_workloads::tpcds::NEXUS_TPCDS_SCALE_FACTOR_ENV,
        datafusion_nexus_workloads::NEXUS_PARQUET_DIR_ENV,
        size_of::<datafusion_nexus_workloads::tpch::TpchQuery>(),
        size_of::<datafusion_nexus_workloads::tpcds::TpcdsQuery>(),
        size_of::<datafusion_nexus_workloads::clickbench::ClickbenchQuery>(),
    );
}

#[cfg(feature = "iceberg")]
#[test]
fn canonical_iceberg_path_exposes_catalog_contracts() {
    let _s3_access_key_env: &str = datafusion_nexus::table_format::iceberg::NEXUS_ICEBERG_S3_ACCESS_KEY_ID_ENV;
}

#[cfg(feature = "cugraph")]
#[test]
fn canonical_graph_path_exposes_sql_contracts() {
    let _bfs_function: &str = datafusion_nexus::graph_sql::CUGRAPH_BFS_FUNCTION;
}

#[cfg(feature = "server")]
#[test]
fn canonical_server_path_exposes_session_contracts() {
    let _auth_contracts = (
        size_of::<datafusion_nexus::server::AuthConfig>(),
        size_of::<datafusion_nexus::server::AuthInterceptor>(),
        size_of::<datafusion_nexus::server::AuthMode>(),
        size_of::<datafusion_nexus::server::PasswordHashError>(),
        size_of::<datafusion_nexus::server::Principal>(),
        size_of::<datafusion_nexus::server::StaticUserConfig>(),
        size_of::<datafusion_nexus::server::TokenStore>(),
        datafusion_nexus::server::handle_handshake,
        datafusion_nexus::server::hash_password,
        datafusion_nexus::server::verify_password,
    );
    let _diagnostic_contracts = (
        datafusion_nexus::server::FlightSqlNativeNotSupportedClass::Other,
        datafusion_nexus::server::log_native_planning_report,
        datafusion_nexus::server::log_native_planning_report_for_tables,
    );
    let _error_contracts = (
        datafusion_nexus::server::ERROR_CODE_METADATA_KEY,
        datafusion_nexus::server::ERROR_KIND_METADATA_KEY,
        datafusion_nexus::server::ERROR_STATUS_METADATA_KEY,
        datafusion_nexus::server::ERROR_OPERATION_METADATA_KEY,
        datafusion_nexus::server::UNSUPPORTED_COLUMN_METADATA_KEY,
        datafusion_nexus::server::UNSUPPORTED_COLUMN_TYPE_METADATA_KEY,
        datafusion_nexus::server::UNSUPPORTED_COLUMN_ROLE_METADATA_KEY,
        datafusion_nexus::server::datafusion_error_to_status,
        datafusion_nexus::server::datafusion_error_to_status_for_operation,
        size_of::<datafusion_nexus::server::ServerError>(),
    );
    let _result: datafusion_nexus::server::Result<()> = Ok(());
}

#[cfg(all(feature = "server", feature = "iceberg"))]
#[test]
fn canonical_server_path_exposes_iceberg_config_contracts() {
    let _contracts = (
        size_of::<datafusion_nexus::server::GlueServerBackendConfig>(),
        size_of::<datafusion_nexus::server::IcebergCatalogBackendConfig>(),
        size_of::<datafusion_nexus::server::IcebergServerConfig>(),
        size_of::<datafusion_nexus::server::IcebergTableRegistrationConfig>(),
    );
}

#[cfg(all(feature = "server", feature = "cugraph"))]
#[test]
fn canonical_server_path_exposes_cugraph_config_contract() {
    let _contract = size_of::<datafusion_nexus::server::CugraphServerConfig>();
}

#[test]
fn removed_adapter_facades_and_glob_reexports_do_not_reappear() {
    let root = workspace_root();
    assert!(!root.join("src/iceberg.rs").exists(), "the compatibility iceberg facade must stay deleted");
    assert_no_crate_root_reexports(&root.join("src/lib.rs"));
    assert_no_removed_public_surface(&root.join("src"));
    assert_modules_are_private(&root.join("src/server/mod.rs"), &["auth", "config", "diagnostics", "error"]);
}

/// The nexus-owned Cudf -> Nexus API rename is a deliberate pre-alpha break.
/// Keep retired project spellings out of the live public surface without
/// dual method names or compatibility aliases.
#[test]
fn retired_cudf_product_api_spellings_do_not_reappear() {
    let root = workspace_root();
    let retired = [
        "CudfOptimizerConfig",
        "CudfNativePhysicalOptimizerRule",
        "CudfSessionStateBuilderExt",
        "NativeCudfExec",
        "NativeCudfComposedExec",
        "NativeCudfGpuProducerExec",
        "NativeCudfExecutionConfig",
        "NativeCudfMetrics",
        "CudfTableFormatScanExec",
        "CudfTableFormatGpuOnlyLeafGuardRule",
        "CudfIcebergTableProvider",
        "CudfIcebergCatalogProvider",
        "CudfIcebergSchemaProvider",
        "QueryGpuCudfNativeCoverage",
        "cudf_native_coverage",
        "DatafusionCudfFlightSqlService",
        "CudfNvmlTelemetry",
        "CudfInventory",
        "NATIVE_CUDF_DATAFUSION_METRIC_NAMES",
        "with_cudf_native",
        "with_cudf_native_rule",
        "cudf_native_physical_optimizer_rules",
        "install_cudf_native_rule_with_session_config",
        "__cudf_",
    ];

    // Public facade modules where product APIs are declared/re-exported.
    let surface_roots = [root.join("src"), root.join("crates/nexus-query-engine/src")];
    for surface_root in surface_roots {
        assert_dir_omits_retired_product_spellings(&surface_root, &retired);
    }

    // Canonical new identities remain reachable on stable module paths.
    let _ = (
        size_of::<datafusion_nexus::planner::NexusNativeOptimizerConfig>(),
        size_of::<datafusion_nexus::native_exec::NexusNativeExecutionConfig>(),
        size_of::<datafusion_nexus::native_exec::NexusNativeExec>(),
        datafusion_nexus::native_exec::NATIVE_NEXUS_DATAFUSION_METRIC_NAMES,
        datafusion_nexus::session::NEXUS_NATIVE_PHYSICAL_OPTIMIZER_RULE_NAME,
    );
}

fn assert_dir_omits_retired_product_spellings(dir: &std::path::Path, retired: &[&str]) {
    for entry in std::fs::read_dir(dir).unwrap_or_else(|error| panic!("failed to read {}: {error}", dir.display())) {
        let path = entry.unwrap_or_else(|error| panic!("failed to read directory entry: {error}")).path();
        if path.is_dir() {
            // Retain the cudf_contract module; it speaks about cuDF I/O contracts.
            if path.file_name().is_some_and(|name| name == "cudf_contract") {
                continue;
            }
            assert_dir_omits_retired_product_spellings(&path, retired);
            continue;
        }
        if path.extension().is_none_or(|extension| extension != "rs") {
            continue;
        }
        let source =
            std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
        for spelling in retired {
            assert!(
                !source.contains(spelling),
                "{} must not retain retired product spelling `{spelling}`",
                path.display()
            );
        }
    }
}

#[test]
fn bench_workload_facade_does_not_reappear() {
    let root = workspace_root();
    let bench_src = root.join("crates/datafusion-nexus-bench/src");
    assert!(!bench_src.join("workload.rs").exists(), "bench must not own a workload facade");
    assert!(!bench_src.join("workload").exists(), "bench must not retain workload implementation modules");

    let bench_lib = std::fs::read_to_string(bench_src.join("lib.rs")).unwrap();
    assert!(!bench_lib.contains("pub mod workload"), "bench must not expose the removed workload module");
    assert!(!bench_lib.contains("pub use workload"), "bench must not re-export the removed workload module");
}

fn workspace_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

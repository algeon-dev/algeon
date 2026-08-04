#[test]
fn native_execution_config_forbids_runtime_injection() {
    // trybuild uses one generated workspace per test binary. Keep these facade
    // fixtures in one test so nextest cannot race concurrent Cargo.toml rewrites.
    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/ui/public_api/forbidden_native_execution_config_with_runtime.rs");
    cases.compile_fail("tests/ui/public_api/forbidden_native_pipeline_execution_mode.rs");
    cases.compile_fail("tests/ui/public_api/forbidden_grant_band_memory_share.rs");
    cases.compile_fail("tests/ui/public_api/forbidden_query_service_tenant_config.rs");
    cases.compile_fail("tests/ui/public_api/forbidden_backend_tenant_builder.rs");
    cases.compile_fail("tests/ui/public_api/forbidden_native_execution_config_requested_grants.rs");
    cases.compile_fail("tests/ui/public_api/forbidden_legacy_admission_policy.rs");
    cases.compile_fail("tests/ui/public_api/forbidden_legacy_admission_builder.rs");
    cases.compile_fail("tests/ui/public_api/forbidden_legacy_memory_policy.rs");
    cases.compile_fail("tests/ui/public_api/forbidden_legacy_memory_policy_builder.rs");
    cases.compile_fail("tests/ui/public_api/forbidden_legacy_memory_governor_reports.rs");
    cases.compile_fail("tests/ui/public_api/forbidden_native_adaptive_pipeline_candidate_diagnostics.rs");
    cases.compile_fail("tests/ui/public_api/forbidden_graph_memory_estimate_admission_evidence.rs");
    cases.compile_fail("tests/ui/public_api/forbidden_graph_execution_spi.rs");
    cases.compile_fail("tests/ui/public_api/forbidden_native_execution_config_with_shared_runtime.rs");
    cases.compile_fail("tests/ui/public_api/forbidden_iceberg_facade.rs");
    cases.compile_fail("tests/ui/public_api/forbidden_retired_cudf_core_api.rs");
    #[cfg(feature = "iceberg")]
    cases.compile_fail("tests/ui/public_api/forbidden_retired_cudf_iceberg_api.rs");
    #[cfg(feature = "server")]
    {
        cases.compile_fail("tests/ui/public_api/forbidden_retired_cudf_server_api.rs");
        cases.compile_fail("tests/ui/public_api/forbidden_server_private_modules.rs");
    }
}

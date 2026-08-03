#[test]
fn installed_backend_facade_is_the_only_public_gpu_resource_construction_path() {
    let cases = trybuild::TestCases::new();
    cases.pass("tests/ui/public_api/supported_admitted_context.rs");
    cases.pass("tests/ui/public_api/supported_core_session.rs");
    cases.compile_fail("tests/ui/public_api/forbidden_*.rs");
    cases.compile_fail("tests/ui/public_api/test_hooks_only_planner_byte_hint_admission_evidence.rs");
    #[cfg(feature = "cugraph")]
    cases.compile_fail("tests/ui/public_api/with_cugraph_graph_adaptive_retry_hooks.rs");
    #[cfg(not(feature = "cugraph"))]
    cases.compile_fail("tests/ui/public_api/without_cugraph_graph_adaptive_retry_hooks.rs");
}

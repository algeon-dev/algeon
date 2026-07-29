#[test]
fn installed_backend_facade_is_the_only_public_gpu_resource_construction_path() {
    let cases = trybuild::TestCases::new();
    cases.pass("tests/ui/public_api/supported_admitted_context.rs");
    cases.pass("tests/ui/public_api/supported_core_session.rs");
    cases.compile_fail("tests/ui/public_api/forbidden_*.rs");
}

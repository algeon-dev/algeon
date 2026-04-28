#[test]
fn native_execution_config_forbids_owned_runtime_injection() {
    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/ui/public_api/forbidden_native_execution_config_with_runtime.rs");
}

#[test]
fn native_execution_config_forbids_shared_runtime_injection() {
    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/ui/public_api/forbidden_native_execution_config_with_shared_runtime.rs");
}

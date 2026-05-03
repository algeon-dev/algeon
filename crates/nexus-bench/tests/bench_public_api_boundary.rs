#[test]
fn bench_uses_workload_contracts_without_a_public_workload_facade() {
    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/ui/public_api/forbidden_workload_facade.rs");
}

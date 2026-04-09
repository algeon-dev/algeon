#[test]
fn column_view_string_materialization_contract() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/column_view_to_strings_removed_fail.rs");
}

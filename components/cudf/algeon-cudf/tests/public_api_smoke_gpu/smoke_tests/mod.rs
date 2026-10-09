use cudf::testing::run_named_cases;

mod core_debug_tests;
mod dataframe_display_tests;
mod high_level_reexport_tests;
mod runtime_debug_tests;
mod series_display_tests;

#[test]
fn display_and_debug_cases() {
    let cases: Vec<(&str, fn())> = [
        core_debug_tests::CASES,
        dataframe_display_tests::CASES,
        runtime_debug_tests::CASES,
        series_display_tests::CASES,
    ]
    .into_iter()
    .flat_map(|cases| cases.iter().copied())
    .collect();
    run_named_cases(&cases);
}

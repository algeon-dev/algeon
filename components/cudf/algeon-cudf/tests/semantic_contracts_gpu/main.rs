use cudf::testing::run_named_cases;

#[path = "../support/common/mod.rs"]
mod common;

type NamedCase = (&'static str, fn());

mod core;
mod frame;
mod internal;
mod ops;

#[test]
fn core_cases() {
    let tables = [core::TABLES].concat();
    let cases = tables.concat();
    run_named_cases(&cases);
}

#[test]
fn frame_contract_and_internal_cases() {
    let tables = [frame::contracts::TABLES, internal::TABLES].concat();
    let cases = tables.concat();
    run_named_cases(&cases);
}

#[test]
fn dataframe_cases() {
    let tables = [frame::dataframe::TABLES].concat();
    let cases = tables.concat();
    run_named_cases(&cases);
}

#[test]
fn series_cases() {
    let tables = [frame::series::TABLES].concat();
    let cases = tables.concat();
    run_named_cases(&cases);
}

#[test]
fn groupby_cases() {
    let tables = [ops::groupby::TABLES].concat();
    let cases = tables.concat();
    run_named_cases(&cases);
}

#[test]
fn join_cases() {
    let tables = [ops::join::TABLES].concat();
    let cases = tables.concat();
    run_named_cases(&cases);
}

#[test]
fn filter_misc_sorting_cases() {
    let tables = [ops::filter::TABLES, ops::misc::TABLES, ops::sorting::TABLES].concat();
    let cases = tables.concat();
    run_named_cases(&cases);
}

#[test]
fn scalar_and_column_operation_cases() {
    let tables = [
        ops::binary::TABLES,
        ops::copying::TABLES,
        ops::datetime::TABLES,
        ops::reduction::TABLES,
        ops::strings::TABLES,
        ops::unary::TABLES,
    ]
    .concat();
    let cases = tables.concat();
    run_named_cases(&cases);
}

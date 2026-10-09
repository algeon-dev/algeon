use crate::NamedCase;

mod convenience_tests;
mod edge_case_tests;
mod ffill_bfill_tests;
mod host_materialization_tests;
mod null_arithmetic_tests;
mod null_reduction_tests;
mod row_ops_tests;
mod scalar_compare_unified_tests;
mod series_concat_tests;
mod series_constructor_tests;
mod series_ops_tests;
mod series_reduction_tests;
mod series_reduction_value_tests;
mod series_tests;

pub(crate) const TABLES: &[&[NamedCase]] = &[
    series_reduction_value_tests::CASES,
    scalar_compare_unified_tests::CASES,
    series_ops_tests::CASES,
    convenience_tests::CASES,
    null_reduction_tests::CASES,
    row_ops_tests::CASES,
    edge_case_tests::CASES,
    host_materialization_tests::CASES,
    series_tests::CASES,
    series_concat_tests::CASES,
    null_arithmetic_tests::CASES,
    series_constructor_tests::CASES,
    ffill_bfill_tests::CASES,
    series_reduction_tests::CASES,
];

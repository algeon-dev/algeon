use crate::NamedCase;

mod dataframe_group_cumcount_tests;
mod dataframe_group_scan_tests;
mod dataframe_groupby_tests;
mod groupby_null_tests;
mod groupby_tests;
mod high_level_groupby_tests;
mod shared_group_key_validation_tests;
mod shift_tests;

pub(crate) const TABLES: &[&[NamedCase]] = &[
    shift_tests::CASES,
    shared_group_key_validation_tests::CASES,
    groupby_null_tests::CASES,
    groupby_tests::CASES,
    dataframe_groupby_tests::CASES,
    dataframe_group_scan_tests::CASES,
    dataframe_group_cumcount_tests::CASES,
    high_level_groupby_tests::CASES,
];

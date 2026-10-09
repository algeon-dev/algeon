use crate::NamedCase;

mod conditional_join_tests;
mod dataframe_merge_split_tests;
mod dataframe_merge_tests;
mod join_tests;
mod mixed_join_tests;

pub(crate) const TABLES: &[&[NamedCase]] = &[
    dataframe_merge_tests::CASES,
    dataframe_merge_split_tests::CASES,
    conditional_join_tests::CASES,
    mixed_join_tests::CASES,
];

use crate::NamedCase;

mod dataframe_filter_expr_tests;
mod dataframe_filter_sort_tests;
mod dataframe_high_level_filter_tests;

pub(crate) const TABLES: &[&[NamedCase]] =
    &[dataframe_filter_sort_tests::CASES, dataframe_filter_expr_tests::CASES, dataframe_high_level_filter_tests::CASES];

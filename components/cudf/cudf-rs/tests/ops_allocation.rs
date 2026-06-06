#[path = "common/mod.rs"]
mod common;

// These tests temporarily replace the process-global current device resource
// via cudf-sys test hooks. Keep them in their own integration-test binary so
// they do not run in the same libtest process as the broader ops suite.

#[path = "ops/groupby/dataframe_groupby_allocation_tests.rs"]
mod dataframe_groupby_allocation_tests;
#[path = "ops/join/dataframe_merge_allocation_tests.rs"]
mod dataframe_merge_allocation_tests;
#[path = "ops/sorting/dataframe_sort_values_allocation_tests.rs"]
mod dataframe_sort_values_allocation_tests;

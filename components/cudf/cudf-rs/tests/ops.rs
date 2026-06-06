#[path = "common/mod.rs"]
mod common;

#[path = "ops/copying/concatenate_tests.rs"]
mod concatenate_tests;
#[path = "ops/copying/copying_tests.rs"]
mod copying_tests;

#[path = "ops/filter/dataframe_filter_sort_tests.rs"]
mod dataframe_filter_sort_tests;
#[path = "ops/filter/dataframe_high_level_filter_tests.rs"]
mod dataframe_high_level_filter_tests;
#[path = "ops/filter/filter_tests.rs"]
mod filter_tests;

#[path = "ops/groupby/dataframe_groupby_tests.rs"]
mod dataframe_groupby_tests;
#[path = "ops/groupby/groupby_tests.rs"]
mod groupby_tests;
#[path = "ops/groupby/high_level_groupby_tests.rs"]
mod high_level_groupby_tests;

#[path = "ops/join/dataframe_merge_split_tests.rs"]
mod dataframe_merge_split_tests;
#[path = "ops/join/dataframe_merge_tests.rs"]
mod dataframe_merge_tests;
#[path = "ops/join/join_tests.rs"]
mod join_tests;

#[path = "ops/misc/dictionary_tests.rs"]
mod dictionary_tests;
#[path = "ops/misc/expr_tests.rs"]
mod expr_tests;
#[path = "ops/misc/factorize_tests.rs"]
mod factorize_tests;
#[path = "ops/misc/hashing_tests.rs"]
mod hashing_tests;
#[path = "ops/misc/replace_tests.rs"]
mod replace_tests;
#[path = "ops/misc/search_tests.rs"]
mod search_tests;
#[path = "ops/misc/stream_compaction_tests.rs"]
mod stream_compaction_tests;

#[path = "ops/reduction/reduction_tests.rs"]
mod reduction_tests;

#[path = "ops/sorting/dataframe_sort_values_tests.rs"]
mod dataframe_sort_values_tests;
#[path = "ops/sorting/sorting_tests.rs"]
mod sorting_tests;

#[path = "ops/unary/binary_tests.rs"]
mod binary_tests;
#[path = "ops/unary/unary_tests.rs"]
mod unary_tests;

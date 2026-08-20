#[path = "../frame/contracts/cast_consistency_tests.rs"]
mod cast_consistency_tests;
#[path = "../frame/contracts/high_level_semantic_baseline_tests.rs"]
mod high_level_semantic_baseline_tests;
#[path = "../frame/contracts/shared_row_selection_contract_tests.rs"]
mod shared_row_selection_contract_tests;

#[path = "../frame/dataframe/arc_sharing_tests.rs"]
mod dataframe_arc_sharing_tests;
#[path = "../frame/dataframe/arrow_tests.rs"]
mod dataframe_arrow_tests;
#[path = "../frame/dataframe/basics_tests.rs"]
mod dataframe_basics_tests;
#[path = "../frame/dataframe/column_ops_tests.rs"]
mod dataframe_column_ops_tests;
#[path = "../frame/dataframe/concat_tests.rs"]
mod dataframe_concat_tests;
#[path = "../frame/dataframe/construction_contract_tests.rs"]
mod dataframe_construction_contract_tests;
#[path = "../frame/dataframe/gather_tests.rs"]
mod dataframe_gather_tests;
#[path = "../frame/dataframe/high_level_d3_tests.rs"]
mod dataframe_high_level_d3_tests;
#[path = "../frame/dataframe/high_level_entrypoint_contract_tests.rs"]
mod dataframe_high_level_entrypoint_contract_tests;
#[path = "../frame/dataframe/info_tests.rs"]
mod dataframe_info_tests;
#[path = "../frame/dataframe/metadata_ops_tests.rs"]
mod dataframe_metadata_ops_tests;
#[path = "../frame/dataframe/series_bridge_tests.rs"]
mod dataframe_series_bridge_tests;
#[path = "../frame/dataframe/slice_tests.rs"]
mod dataframe_slice_tests;

#[path = "../frame/series/edge_case_tests.rs"]
mod edge_case_tests;
#[path = "../frame/series/ffill_bfill_tests.rs"]
mod ffill_bfill_tests;
#[path = "../frame/series/null_arithmetic_tests.rs"]
mod null_arithmetic_tests;
#[path = "../frame/series/null_reduction_tests.rs"]
mod null_reduction_tests;
#[path = "../frame/series/series_concat_tests.rs"]
mod series_concat_tests;
#[path = "../frame/series/series_constructor_tests.rs"]
mod series_constructor_tests;
#[path = "../frame/series/context_delegation_tests.rs"]
mod series_context_delegation_tests;
#[path = "../frame/series/convenience_tests.rs"]
mod series_convenience_tests;
#[path = "../frame/series/host_materialization_tests.rs"]
mod series_host_materialization_tests;
#[path = "../frame/series/series_ops_tests.rs"]
mod series_ops_tests;
#[path = "../frame/series/series_reduction_tests.rs"]
mod series_reduction_tests;
#[path = "../frame/series/series_reduction_value_tests.rs"]
mod series_reduction_value_tests;
#[path = "../frame/series/row_ops_tests.rs"]
mod series_row_ops_tests;
#[path = "../frame/series/scalar_compare_unified_tests.rs"]
mod series_scalar_compare_unified_tests;
#[path = "../frame/series/series_tests.rs"]
mod series_tests;

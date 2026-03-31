#[path = "common/mod.rs"]
mod common;

#[path = "frame/contracts/accessor_naming_tests.rs"]
mod accessor_naming_tests;
#[path = "frame/contracts/cast_consistency_tests.rs"]
mod cast_consistency_tests;
#[path = "frame/contracts/high_level_reexport_tests.rs"]
mod high_level_reexport_tests;
#[path = "frame/contracts/high_level_semantic_baseline_tests.rs"]
mod high_level_semantic_baseline_tests;

#[path = "frame/dataframe/arc_sharing_tests.rs"]
mod dataframe_arc_sharing_tests;
#[path = "frame/dataframe/arrow_tests.rs"]
mod dataframe_arrow_tests;
#[path = "frame/dataframe/basics_tests.rs"]
mod dataframe_basics_tests;
#[path = "frame/dataframe/column_ops_tests.rs"]
mod dataframe_column_ops_tests;
#[path = "frame/dataframe/concat_tests.rs"]
mod dataframe_concat_tests;
#[path = "frame/dataframe/copy_naming_tests.rs"]
mod dataframe_copy_naming_tests;
#[path = "frame/dataframe/display_tests.rs"]
mod dataframe_display_tests;
#[path = "frame/dataframe/gather_tests.rs"]
mod dataframe_gather_tests;
#[path = "frame/dataframe/high_level_d3_tests.rs"]
mod dataframe_high_level_d3_tests;
#[path = "frame/dataframe/high_level_signature_tests.rs"]
mod dataframe_high_level_signature_tests;
#[path = "frame/dataframe/info_tests.rs"]
mod dataframe_info_tests;
#[path = "frame/dataframe/io_tests.rs"]
mod dataframe_io_tests;
#[path = "frame/dataframe/metadata_ops_tests.rs"]
mod dataframe_metadata_ops_tests;
#[path = "frame/dataframe/series_bridge_tests.rs"]
mod dataframe_series_bridge_tests;
#[path = "frame/dataframe/slice_tests.rs"]
mod dataframe_slice_tests;
#[path = "frame/dataframe/view_tests.rs"]
mod dataframe_view_tests;

#[path = "frame/series/convenience_tests.rs"]
mod series_convenience_tests;
#[path = "frame/series/series_tests.rs"]
mod series_tests;
#[path = "frame/series/view_tests.rs"]
mod series_view_tests;

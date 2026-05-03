#[path = "../core/debug_tests.rs"]
mod core_debug_tests;
#[path = "../frame/dataframe/debug_tests.rs"]
mod dataframe_debug_tests;
#[path = "../frame/dataframe/display_tests.rs"]
mod dataframe_display_tests;
#[path = "../frame/contracts/high_level_reexport_tests.rs"]
mod high_level_reexport_tests;
#[path = "../runtime/debug_tests.rs"]
mod runtime_debug_tests;
#[path = "../frame/series/display_tests.rs"]
mod series_display_tests;

#[test]
fn join_indices_and_hash_join_slice6_surface_is_publicly_importable() {
    use cudf::ops::join::{
        HashJoin, HashJoinOptions, JoinIndices, NullEquality, full_join_indices, inner_join_indices, left_join_indices,
    };

    let _options = HashJoinOptions::new().null_equality(NullEquality::Equal);
    let _: Option<HashJoin> = None;
    let _: Option<JoinIndices> = None;
    let _ = (inner_join_indices, left_join_indices, full_join_indices);
}

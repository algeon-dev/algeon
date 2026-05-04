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

#[test]
fn slice7_join_and_copying_surface_is_publicly_importable() {
    use cudf::ops::{
        copying::scatter,
        join::{SingleSideJoinIndices, cross_join, left_anti_join_indices, left_semi_join_indices},
    };

    let _: Option<SingleSideJoinIndices> = None;
    let _ = (left_semi_join_indices, left_anti_join_indices, cross_join, scatter);
}

#[test]
fn slice8_ast_conditional_and_mixed_join_surface_is_publicly_importable() {
    use cudf::ops::{
        ast::{AstExpression, AstOperator},
        join::{
            conditional_full_join_indices, conditional_inner_join_indices, conditional_left_anti_join_indices,
            conditional_left_join_indices, conditional_left_semi_join_indices, mixed_full_join_indices,
            mixed_inner_join_indices, mixed_left_anti_join_indices, mixed_left_join_indices,
            mixed_left_semi_join_indices,
        },
    };

    let _: Option<AstExpression> = None;
    let _operator = AstOperator::Less;
    let _ = (
        conditional_inner_join_indices,
        conditional_left_join_indices,
        conditional_full_join_indices,
        conditional_left_semi_join_indices,
        conditional_left_anti_join_indices,
        mixed_inner_join_indices,
        mixed_left_join_indices,
        mixed_full_join_indices,
        mixed_left_semi_join_indices,
        mixed_left_anti_join_indices,
    );
}

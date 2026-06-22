//! Semantic tests for mixed equality and conditional join index operations.

use cudf::{
    DataFrame, DataFrameCrossJoinOptions, DataFrameJoinOptions, DataFrameMixedJoinOptions,
    DataFramePredicateJoinOptions, ErrorCode, Runtime,
    core::Table,
    ops::{
        ast::AstExpression,
        join::{self, JoinIndices, JoinType, NullEquality, SingleSideJoinIndices},
    },
    testing::prelude::*,
};

use crate::common::{column_from_slice, column_from_slice_nullable};

const JOIN_NO_MATCH: i32 = i32::MIN;

fn i32_table(values: &[i32]) -> Table {
    Table::from_columns(vec![column_from_slice(values).unwrap()]).unwrap()
}

fn nullable_i32_table(values: &[i32], validity: &[bool]) -> Table {
    Table::from_columns(vec![column_from_slice_nullable(values, validity).unwrap()]).unwrap()
}

fn left_less_than_right() -> AstExpression {
    AstExpression::less(AstExpression::left_column(0).unwrap(), AstExpression::right_column(0).unwrap()).unwrap()
}

fn sorted_join_index_pairs(indices: &JoinIndices) -> Vec<(i32, i32)> {
    let left = indices.left.testing_view().unwrap().to_vec::<i32>().unwrap();
    let right = indices.right.testing_view().unwrap().to_vec::<i32>().unwrap();

    let mut pairs: Vec<_> = left.into_iter().zip(right).collect();
    pairs.sort();
    pairs
}

fn sorted_single_side_indices(indices: &SingleSideJoinIndices) -> Vec<i32> {
    let mut rows = indices.indices.testing_view().unwrap().to_vec::<i32>().unwrap();
    rows.sort();
    rows
}

fn mixed_inputs() -> (Table, Table, Table, Table) {
    let left_equality = i32_table(&[1, 1, 2, 2, 4]);
    let right_equality = i32_table(&[1, 1, 2, 2, 3]);
    let left_conditional = i32_table(&[5, 7, 3, 9, 1]);
    let right_conditional = i32_table(&[6, 8, 4, 7, 10]);

    (left_equality, right_equality, left_conditional, right_conditional)
}

#[test]
fn filter_join_indices_rejects_non_int32_index_columns() {
    let left_conditional = i32_table(&[1]);
    let right_conditional = i32_table(&[2]);
    let left_indices = column_from_slice(&[0i64]).unwrap();
    let right_indices = column_from_slice(&[0i32]).unwrap();
    let predicate = left_less_than_right();

    let err = join::filter_join_indices(
        &left_conditional.testing_view().unwrap(),
        &right_conditional.testing_view().unwrap(),
        &left_indices.testing_view().unwrap(),
        &right_indices.testing_view().unwrap(),
        &predicate,
        JoinType::Inner,
    )
    .unwrap_err();

    assert_eq!(err.code(), ErrorCode::TypeMismatch);
    assert_eq!(err.operation(), "join::filter_join_indices");
}

#[test]
fn filter_join_indices_rejects_mismatched_index_lengths() {
    let left_conditional = i32_table(&[1, 3]);
    let right_conditional = i32_table(&[2]);
    let left_indices = column_from_slice(&[0i32, 1]).unwrap();
    let right_indices = column_from_slice(&[0i32]).unwrap();
    let predicate = left_less_than_right();

    let err = join::filter_join_indices(
        &left_conditional.testing_view().unwrap(),
        &right_conditional.testing_view().unwrap(),
        &left_indices.testing_view().unwrap(),
        &right_indices.testing_view().unwrap(),
        &predicate,
        JoinType::Inner,
    )
    .unwrap_err();

    assert_eq!(err.code(), ErrorCode::ColumnLengthMismatch);
    assert_eq!(err.operation(), "join::filter_join_indices");
}

#[test]
fn filter_join_indices_rejects_nullable_index_columns() {
    let left_conditional = i32_table(&[1]);
    let right_conditional = i32_table(&[2]);
    let left_indices = column_from_slice_nullable(&[0i32], &[false]).unwrap();
    let right_indices = column_from_slice(&[0i32]).unwrap();
    let predicate = left_less_than_right();

    let err = join::filter_join_indices(
        &left_conditional.testing_view().unwrap(),
        &right_conditional.testing_view().unwrap(),
        &left_indices.testing_view().unwrap(),
        &right_indices.testing_view().unwrap(),
        &predicate,
        JoinType::Inner,
    )
    .unwrap_err();

    assert_eq!(err.code(), ErrorCode::NullUnexpected);
    assert_eq!(err.operation(), "join::filter_join_indices");
    assert_eq!(err.detail_value("reason"), Some("nullable_join_indices"));
}

#[test]
fn filter_join_indices_inner_matches_mixed_inner_join_indices() {
    let (left_equality, right_equality, left_conditional, right_conditional) = mixed_inputs();
    let predicate = left_less_than_right();

    let equality_indices = join::inner_join_indices(
        &left_equality.testing_view().unwrap(),
        &right_equality.testing_view().unwrap(),
        NullEquality::Equal,
    )
    .unwrap();
    let filtered = join::filter_join_indices(
        &left_conditional.testing_view().unwrap(),
        &right_conditional.testing_view().unwrap(),
        &equality_indices.left.testing_view().unwrap(),
        &equality_indices.right.testing_view().unwrap(),
        &predicate,
        JoinType::Inner,
    )
    .unwrap();
    let mixed = join::mixed_inner_join_indices(
        &left_equality.testing_view().unwrap(),
        &right_equality.testing_view().unwrap(),
        &left_conditional.testing_view().unwrap(),
        &right_conditional.testing_view().unwrap(),
        &predicate,
        NullEquality::Equal,
    )
    .unwrap();

    assert_eq!(sorted_join_index_pairs(&filtered), sorted_join_index_pairs(&mixed));
}

#[test]
fn filter_join_indices_left_matches_mixed_left_join_indices() {
    let (left_equality, right_equality, left_conditional, right_conditional) = mixed_inputs();
    let predicate = left_less_than_right();

    let equality_indices = join::left_join_indices(
        &left_equality.testing_view().unwrap(),
        &right_equality.testing_view().unwrap(),
        NullEquality::Equal,
    )
    .unwrap();
    let filtered = join::filter_join_indices(
        &left_conditional.testing_view().unwrap(),
        &right_conditional.testing_view().unwrap(),
        &equality_indices.left.testing_view().unwrap(),
        &equality_indices.right.testing_view().unwrap(),
        &predicate,
        JoinType::Left,
    )
    .unwrap();
    let mixed = join::mixed_left_join_indices(
        &left_equality.testing_view().unwrap(),
        &right_equality.testing_view().unwrap(),
        &left_conditional.testing_view().unwrap(),
        &right_conditional.testing_view().unwrap(),
        &predicate,
        NullEquality::Equal,
    )
    .unwrap();

    assert_eq!(sorted_join_index_pairs(&filtered), sorted_join_index_pairs(&mixed));
}

#[test]
fn filter_join_indices_full_matches_mixed_full_join_indices() {
    let left_equality = i32_table(&[1, 2, 3]);
    let right_equality = i32_table(&[1, 2, 4]);
    let left_conditional = i32_table(&[10, 20, 30]);
    let right_conditional = i32_table(&[15, 15, 25]);
    let predicate = left_less_than_right();

    let equality_indices = join::full_join_indices(
        &left_equality.testing_view().unwrap(),
        &right_equality.testing_view().unwrap(),
        NullEquality::Equal,
    )
    .unwrap();
    let filtered = join::filter_join_indices(
        &left_conditional.testing_view().unwrap(),
        &right_conditional.testing_view().unwrap(),
        &equality_indices.left.testing_view().unwrap(),
        &equality_indices.right.testing_view().unwrap(),
        &predicate,
        JoinType::Full,
    )
    .unwrap();
    let mixed = join::mixed_full_join_indices(
        &left_equality.testing_view().unwrap(),
        &right_equality.testing_view().unwrap(),
        &left_conditional.testing_view().unwrap(),
        &right_conditional.testing_view().unwrap(),
        &predicate,
        NullEquality::Equal,
    )
    .unwrap();

    assert_eq!(sorted_join_index_pairs(&filtered), sorted_join_index_pairs(&mixed));
}

#[test]
fn filter_join_indices_full_uses_raw_pair_semantics_for_duplicate_matches() {
    let (left_equality, right_equality, left_conditional, right_conditional) = mixed_inputs();
    let predicate = left_less_than_right();

    let equality_indices = join::full_join_indices(
        &left_equality.testing_view().unwrap(),
        &right_equality.testing_view().unwrap(),
        NullEquality::Equal,
    )
    .unwrap();
    let filtered = join::filter_join_indices(
        &left_conditional.testing_view().unwrap(),
        &right_conditional.testing_view().unwrap(),
        &equality_indices.left.testing_view().unwrap(),
        &equality_indices.right.testing_view().unwrap(),
        &predicate,
        JoinType::Full,
    )
    .unwrap();

    assert_eq!(
        sorted_join_index_pairs(&filtered),
        vec![
            (JOIN_NO_MATCH, 0),
            (JOIN_NO_MATCH, 2),
            (JOIN_NO_MATCH, 3),
            (JOIN_NO_MATCH, 4),
            (0, 0),
            (0, 1),
            (1, JOIN_NO_MATCH),
            (1, 1),
            (2, 2),
            (2, 3),
            (3, JOIN_NO_MATCH),
            (3, JOIN_NO_MATCH),
            (4, JOIN_NO_MATCH),
        ]
    );
}

#[test]
fn mixed_inner_join_indices_requires_equality_and_less_than_predicate() {
    let (left_equality, right_equality, left_conditional, right_conditional) = mixed_inputs();
    let predicate = left_less_than_right();

    let indices = join::mixed_inner_join_indices(
        &left_equality.testing_view().unwrap(),
        &right_equality.testing_view().unwrap(),
        &left_conditional.testing_view().unwrap(),
        &right_conditional.testing_view().unwrap(),
        &predicate,
        NullEquality::Equal,
    )
    .unwrap();

    assert_eq!(sorted_join_index_pairs(&indices), vec![(0, 0), (0, 1), (1, 1), (2, 2), (2, 3)]);
}

#[test]
fn mixed_left_join_indices_preserves_unmatched_left_rows() {
    let (left_equality, right_equality, left_conditional, right_conditional) = mixed_inputs();
    let predicate = left_less_than_right();

    let indices = join::mixed_left_join_indices(
        &left_equality.testing_view().unwrap(),
        &right_equality.testing_view().unwrap(),
        &left_conditional.testing_view().unwrap(),
        &right_conditional.testing_view().unwrap(),
        &predicate,
        NullEquality::Equal,
    )
    .unwrap();

    assert_eq!(
        sorted_join_index_pairs(&indices),
        vec![(0, 0), (0, 1), (1, 1), (2, 2), (2, 3), (3, JOIN_NO_MATCH), (4, JOIN_NO_MATCH)]
    );
}

#[test]
fn mixed_full_join_indices_preserves_unmatched_rows_from_both_sides() {
    let (left_equality, right_equality, left_conditional, right_conditional) = mixed_inputs();
    let predicate = left_less_than_right();

    let indices = join::mixed_full_join_indices(
        &left_equality.testing_view().unwrap(),
        &right_equality.testing_view().unwrap(),
        &left_conditional.testing_view().unwrap(),
        &right_conditional.testing_view().unwrap(),
        &predicate,
        NullEquality::Equal,
    )
    .unwrap();

    assert_eq!(
        sorted_join_index_pairs(&indices),
        vec![(JOIN_NO_MATCH, 4), (0, 0), (0, 1), (1, 1), (2, 2), (2, 3), (3, JOIN_NO_MATCH), (4, JOIN_NO_MATCH),]
    );
}

#[test]
fn mixed_left_semi_and_anti_join_indices_return_left_existence_shapes() {
    let (left_equality, right_equality, left_conditional, right_conditional) = mixed_inputs();
    let predicate = left_less_than_right();

    let semi = join::mixed_left_semi_join_indices(
        &left_equality.testing_view().unwrap(),
        &right_equality.testing_view().unwrap(),
        &left_conditional.testing_view().unwrap(),
        &right_conditional.testing_view().unwrap(),
        &predicate,
        NullEquality::Equal,
    )
    .unwrap();
    let anti = join::mixed_left_anti_join_indices(
        &left_equality.testing_view().unwrap(),
        &right_equality.testing_view().unwrap(),
        &left_conditional.testing_view().unwrap(),
        &right_conditional.testing_view().unwrap(),
        &predicate,
        NullEquality::Equal,
    )
    .unwrap();

    assert_eq!(sorted_single_side_indices(&semi), vec![0, 1, 2]);
    assert_eq!(sorted_single_side_indices(&anti), vec![3, 4]);
}

#[test]
fn mixed_inner_join_indices_honors_null_equality_for_equi_keys() {
    let left_equality = nullable_i32_table(&[0, 1], &[false, true]);
    let right_equality = nullable_i32_table(&[0, 1], &[false, true]);
    let left_conditional = i32_table(&[1, 1]);
    let right_conditional = i32_table(&[2, 2]);
    let predicate = left_less_than_right();

    let equal = join::mixed_inner_join_indices(
        &left_equality.testing_view().unwrap(),
        &right_equality.testing_view().unwrap(),
        &left_conditional.testing_view().unwrap(),
        &right_conditional.testing_view().unwrap(),
        &predicate,
        NullEquality::Equal,
    )
    .unwrap();
    let unequal = join::mixed_inner_join_indices(
        &left_equality.testing_view().unwrap(),
        &right_equality.testing_view().unwrap(),
        &left_conditional.testing_view().unwrap(),
        &right_conditional.testing_view().unwrap(),
        &predicate,
        NullEquality::Unequal,
    )
    .unwrap();

    assert_eq!(sorted_join_index_pairs(&equal), vec![(0, 0), (1, 1)]);
    assert_eq!(sorted_join_index_pairs(&unequal), vec![(1, 1)]);
}

#[test]
fn dataframe_projected_cross_join_supports_output_aliases() {
    let rt = Runtime::new().unwrap();
    let left = DataFrame::from_series([
        ("id", rt.series(&[1i32, 2]).unwrap()),
        ("left_val", rt.series(&[10i32, 20]).unwrap()),
    ])
    .unwrap();
    let right = DataFrame::from_series([
        ("id", rt.series(&[100i32, 200, 300]).unwrap()),
        ("right_val", rt.series(&[7i32, 8, 9]).unwrap()),
    ])
    .unwrap();

    let result = left
        .cross_join_projected(
            &right,
            DataFrameCrossJoinOptions::new()
                .left_projection([("id", "left_id"), ("left_val", "left_val")])
                .right_projection([("id", "right_id"), ("right_val", "right_val")]),
        )
        .unwrap();

    assert_eq!(result.num_rows(), 6);
    assert_eq!(result.column_names(), vec!["left_id", "left_val", "right_id", "right_val"]);
    assert_eq!(result.series("left_id").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 1, 1, 2, 2, 2]);
    assert_eq!(result.series("left_val").unwrap().to_vec_strict::<i32>().unwrap(), vec![10, 10, 10, 20, 20, 20]);
    assert_eq!(result.series("right_id").unwrap().to_vec_strict::<i32>().unwrap(), vec![100, 200, 300, 100, 200, 300]);
    assert_eq!(result.series("right_val").unwrap().to_vec_strict::<i32>().unwrap(), vec![7, 8, 9, 7, 8, 9]);
}

#[test]
fn dataframe_mixed_join_indices_use_named_equality_and_predicate_columns() {
    let rt = Runtime::new().unwrap();
    let left = DataFrame::from_series([
        ("id", rt.series(&[1i32, 1, 2, 2]).unwrap()),
        ("left_val", rt.series(&[5i32, 7, 3, 9]).unwrap()),
    ])
    .unwrap();
    let right = DataFrame::from_series([
        ("id", rt.series(&[1i32, 1, 2, 2]).unwrap()),
        ("right_val", rt.series(&[6i32, 8, 4, 7]).unwrap()),
    ])
    .unwrap();
    let predicate = left_less_than_right();

    let indices = left
        .mixed_inner_join_indices(
            &right,
            &predicate,
            DataFrameMixedJoinOptions::new()
                .on(["id"])
                .left_predicate_columns(["left_val"])
                .right_predicate_columns(["right_val"]),
        )
        .unwrap();

    let mut pairs = indices
        .left()
        .indices()
        .to_vec_strict::<i32>()
        .unwrap()
        .into_iter()
        .zip(indices.right().indices().to_vec_strict::<i32>().unwrap())
        .collect::<Vec<_>>();
    pairs.sort();
    assert_eq!(pairs, vec![(0, 0), (0, 1), (1, 1), (2, 2), (2, 3)]);
}

#[test]
fn dataframe_filter_join_indices_filter_existing_hash_probe_pairs() {
    let rt = Runtime::new().unwrap();
    let left = DataFrame::from_series([
        ("id", rt.series(&[1i32, 1, 2, 2]).unwrap()),
        ("left_val", rt.series(&[5i32, 7, 3, 9]).unwrap()),
    ])
    .unwrap();
    let right = DataFrame::from_series([
        ("id", rt.series(&[1i32, 1, 2, 2]).unwrap()),
        ("right_val", rt.series(&[6i32, 8, 4, 7]).unwrap()),
    ])
    .unwrap();
    let equality = left.inner_join_indices(&right, DataFrameJoinOptions::new().on(["id"])).unwrap();
    let predicate = left_less_than_right();

    let filtered = left
        .filter_join_indices(
            &right,
            equality.left(),
            equality.right(),
            &predicate,
            DataFramePredicateJoinOptions::new().left_columns(["left_val"]).right_columns(["right_val"]),
            cudf::JoinType::Inner,
        )
        .unwrap();

    let mut pairs = filtered
        .left()
        .indices()
        .to_vec_strict::<i32>()
        .unwrap()
        .into_iter()
        .zip(filtered.right().indices().to_vec_strict::<i32>().unwrap())
        .collect::<Vec<_>>();
    pairs.sort();
    assert_eq!(pairs, vec![(0, 0), (0, 1), (1, 1), (2, 2), (2, 3)]);
}

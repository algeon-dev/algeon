//! Semantic tests for conditional join index operations.

use cudf::{
    core::Table,
    ops::{
        ast::AstExpression,
        join::{self, JoinIndices, SingleSideJoinIndices},
    },
    testing::prelude::*,
};

use crate::common::{column_from_slice, column_from_slice_nullable, scalar};

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

#[test]
fn conditional_inner_join_indices_matches_less_than_predicate() {
    let left = i32_table(&[1, 3, 5]);
    let right = i32_table(&[2, 4]);
    let predicate = left_less_than_right();

    let indices =
        join::conditional_inner_join_indices(&left.testing_view().unwrap(), &right.testing_view().unwrap(), &predicate)
            .unwrap();

    assert_eq!(sorted_join_index_pairs(&indices), vec![(0, 0), (0, 1), (1, 1)]);
}

#[test]
fn conditional_left_join_indices_preserves_unmatched_left_rows() {
    let left = i32_table(&[1, 3, 5]);
    let right = i32_table(&[2, 4]);
    let predicate = left_less_than_right();

    let indices =
        join::conditional_left_join_indices(&left.testing_view().unwrap(), &right.testing_view().unwrap(), &predicate)
            .unwrap();

    assert_eq!(sorted_join_index_pairs(&indices), vec![(0, 0), (0, 1), (1, 1), (2, JOIN_NO_MATCH)]);
}

#[test]
fn conditional_full_join_indices_preserves_unmatched_rows_from_both_sides() {
    let left = i32_table(&[2, 5, 9]);
    let right = i32_table(&[1, 3, 6]);
    let predicate = left_less_than_right();

    let indices =
        join::conditional_full_join_indices(&left.testing_view().unwrap(), &right.testing_view().unwrap(), &predicate)
            .unwrap();

    assert_eq!(sorted_join_index_pairs(&indices), vec![(JOIN_NO_MATCH, 0), (0, 1), (0, 2), (1, 2), (2, JOIN_NO_MATCH)]);
}

#[test]
fn conditional_left_semi_and_anti_join_indices_return_left_existence_shapes() {
    let left = i32_table(&[1, 3, 5]);
    let right = i32_table(&[2, 4]);
    let predicate = left_less_than_right();

    let semi = join::conditional_left_semi_join_indices(
        &left.testing_view().unwrap(),
        &right.testing_view().unwrap(),
        &predicate,
    )
    .unwrap();
    let anti = join::conditional_left_anti_join_indices(
        &left.testing_view().unwrap(),
        &right.testing_view().unwrap(),
        &predicate,
    )
    .unwrap();

    assert_eq!(sorted_single_side_indices(&semi), vec![0, 1]);
    assert_eq!(sorted_single_side_indices(&anti), vec![2]);
}

#[test]
fn conditional_inner_join_indices_treats_null_predicate_as_not_matching() {
    let left = nullable_i32_table(&[1, 0, 3], &[true, false, true]);
    let right = i32_table(&[2]);
    let predicate = left_less_than_right();

    let indices =
        join::conditional_inner_join_indices(&left.testing_view().unwrap(), &right.testing_view().unwrap(), &predicate)
            .unwrap();

    assert_eq!(sorted_join_index_pairs(&indices), vec![(0, 0)]);
}

#[test]
fn conditional_inner_join_indices_keeps_literal_scalar_alive_inside_ast() {
    let left = i32_table(&[1, 3, 5]);
    let right = i32_table(&[0, 0]);
    let literal = AstExpression::literal(scalar(4i32).unwrap()).unwrap();
    let predicate = AstExpression::less(AstExpression::left_column(0).unwrap(), literal).unwrap();

    let indices =
        join::conditional_inner_join_indices(&left.testing_view().unwrap(), &right.testing_view().unwrap(), &predicate)
            .unwrap();

    assert_eq!(sorted_join_index_pairs(&indices), vec![(0, 0), (0, 1), (1, 0), (1, 1)]);
}

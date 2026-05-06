//! Semantic tests for conditional join index operations.

use std::sync::Arc;

use arrow::{
    array::Decimal128Array,
    datatypes::{DataType as ArrowDataType, Field, Schema},
    record_batch::RecordBatch,
};
use cudf::{
    DataFrame, Runtime,
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

fn decimal128_table_from_arrow(values: Vec<Option<i128>>, precision: u8, scale: i8) -> Table {
    let amount = Decimal128Array::from(values).with_precision_and_scale(precision, scale).unwrap();
    let schema = Schema::new(vec![Field::new("amount", ArrowDataType::Decimal128(precision, scale), true)]);
    let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(amount)]).unwrap();
    let runtime = Runtime::builder().build().unwrap();
    DataFrame::from_record_batch(&runtime, &batch).unwrap().to_table_copy().unwrap()
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
fn conditional_join_size_and_output_size_hint_match_indices() {
    let left = i32_table(&[1, 3, 5]);
    let right = i32_table(&[2, 4]);
    let left_view = left.testing_view().unwrap();
    let right_view = right.testing_view().unwrap();
    let predicate = left_less_than_right();

    let inner_size = join::conditional_inner_join_size(&left_view, &right_view, &predicate).unwrap();
    assert_eq!(inner_size, 3);
    let inner =
        join::conditional_inner_join_indices_with_output_size(&left_view, &right_view, &predicate, inner_size).unwrap();
    assert_eq!(sorted_join_index_pairs(&inner), vec![(0, 0), (0, 1), (1, 1)]);

    let left_size = join::conditional_left_join_size(&left_view, &right_view, &predicate).unwrap();
    assert_eq!(left_size, 4);
    let left_indices =
        join::conditional_left_join_indices_with_output_size(&left_view, &right_view, &predicate, left_size).unwrap();
    assert_eq!(sorted_join_index_pairs(&left_indices), vec![(0, 0), (0, 1), (1, 1), (2, JOIN_NO_MATCH)]);

    let semi_size = join::conditional_left_semi_join_size(&left_view, &right_view, &predicate).unwrap();
    assert_eq!(semi_size, 2);
    let semi =
        join::conditional_left_semi_join_indices_with_output_size(&left_view, &right_view, &predicate, semi_size)
            .unwrap();
    assert_eq!(sorted_single_side_indices(&semi), vec![0, 1]);

    let anti_size = join::conditional_left_anti_join_size(&left_view, &right_view, &predicate).unwrap();
    assert_eq!(anti_size, 1);
    let anti =
        join::conditional_left_anti_join_indices_with_output_size(&left_view, &right_view, &predicate, anti_size)
            .unwrap();
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

#[test]
fn conditional_inner_join_indices_supports_ast_in_list_predicate() {
    let left = i32_table(&[1, 3, 5]);
    let right = i32_table(&[2, 4]);
    let left_in = AstExpression::is_in_list(
        AstExpression::left_column(0).unwrap(),
        vec![
            AstExpression::literal(scalar(1i32).unwrap()).unwrap(),
            AstExpression::literal(scalar(5i32).unwrap()).unwrap(),
        ],
    )
    .unwrap();
    let right_in = AstExpression::is_in_list(
        AstExpression::right_column(0).unwrap(),
        vec![AstExpression::literal(scalar(2i32).unwrap()).unwrap()],
    )
    .unwrap();
    let predicate = AstExpression::logical_and(left_in, right_in).unwrap();

    let indices =
        join::conditional_inner_join_indices(&left.testing_view().unwrap(), &right.testing_view().unwrap(), &predicate)
            .unwrap();

    assert_eq!(sorted_join_index_pairs(&indices), vec![(0, 0), (2, 0)]);
}

#[test]
fn conditional_inner_join_indices_supports_ast_cast_predicate() {
    let left = i32_table(&[1, 3, 5]);
    let right = i32_table(&[0, 0]);
    let predicate = AstExpression::greater(
        AstExpression::cast_to_float64(AstExpression::left_column(0).unwrap()).unwrap(),
        AstExpression::literal(scalar(2.5f64).unwrap()).unwrap(),
    )
    .unwrap();

    let indices =
        join::conditional_inner_join_indices(&left.testing_view().unwrap(), &right.testing_view().unwrap(), &predicate)
            .unwrap();

    assert_eq!(sorted_join_index_pairs(&indices), vec![(1, 0), (1, 1), (2, 0), (2, 1)]);
}

#[test]
fn conditional_inner_join_indices_supports_decimal_ast_cast_to_float64() {
    let left = decimal128_table_from_arrow(vec![Some(12345), Some(20_050), None], 20, 2);
    let right = i32_table(&[0, 1]);
    let predicate = AstExpression::greater(
        AstExpression::cast_to_float64(AstExpression::left_column(0).unwrap()).unwrap(),
        AstExpression::literal(scalar(200.0f64).unwrap()).unwrap(),
    )
    .unwrap();

    let indices =
        join::conditional_inner_join_indices(&left.testing_view().unwrap(), &right.testing_view().unwrap(), &predicate)
            .unwrap();

    assert_eq!(sorted_join_index_pairs(&indices), vec![(1, 0), (1, 1)]);
}

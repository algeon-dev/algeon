//! Semantic tests for conditional join index operations.

use std::sync::Arc;

use arrow::{
    array::Decimal128Array,
    datatypes::{DataType as ArrowDataType, Field, Schema},
    record_batch::RecordBatch,
};
use cudf::{
    DataFrame, DataFrameJoinIndices, DataFrameJoinPairIndices, DataFramePredicateJoinOptions, Exec, Runtime,
    ops::ast::{AstExpression, AstOperator},
};

use crate::common::{run_named_cases, scalar};

const JOIN_NO_MATCH: i32 = i32::MIN;

fn decimal128_frame_from_arrow(rt: &Runtime, values: Vec<Option<i128>>, precision: u8, scale: i8) -> DataFrame {
    let amount = Decimal128Array::from(values).with_precision_and_scale(precision, scale).unwrap();
    let schema = Schema::new(vec![Field::new("amount", ArrowDataType::Decimal128(precision, scale), true)]);
    let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(amount)]).unwrap();
    DataFrame::from_record_batch(rt, &batch).unwrap()
}

fn left_less_than_right() -> AstExpression {
    AstExpression::less(AstExpression::left_column(0).unwrap(), AstExpression::right_column(0).unwrap()).unwrap()
}

fn left_add_right_non_boolean_predicate() -> AstExpression {
    AstExpression::binary(
        AstOperator::Add,
        AstExpression::left_column(0).unwrap(),
        AstExpression::right_column(0).unwrap(),
    )
    .unwrap()
}

fn dataframe_conditional_inputs(left_values: &[i32], right_values: &[i32]) -> (DataFrame, DataFrame) {
    let rt = Runtime::new().unwrap();
    let left = DataFrame::from_series([("left_val", rt.series(left_values).unwrap())]).unwrap();
    let right = DataFrame::from_series([("right_val", rt.series(right_values).unwrap())]).unwrap();
    (left, right)
}

fn dataframe_conditional_options() -> DataFramePredicateJoinOptions {
    DataFramePredicateJoinOptions::new().left_columns(["left_val"]).right_columns(["right_val"])
}

fn sorted_dataframe_join_index_pairs(indices: &DataFrameJoinPairIndices) -> Vec<(i32, i32)> {
    let mut pairs = indices
        .left()
        .indices()
        .to_vec_strict::<i32>()
        .unwrap()
        .into_iter()
        .zip(indices.right().indices().to_vec_strict::<i32>().unwrap())
        .collect::<Vec<_>>();
    pairs.sort();
    pairs
}

fn sorted_dataframe_join_indices(indices: &DataFrameJoinIndices) -> Vec<i32> {
    let mut rows = indices.indices().to_vec_strict::<i32>().unwrap();
    rows.sort();
    rows
}

fn conditional_inner_join_indices_matches_less_than_predicate() {
    let (left, right) = dataframe_conditional_inputs(&[1, 3, 5], &[2, 4]);

    let indices = left
        .on(Exec::RETAINED)
        .conditional_inner_join_indices(&right, &left_less_than_right(), dataframe_conditional_options())
        .unwrap();

    assert_eq!(sorted_dataframe_join_index_pairs(&indices), vec![(0, 0), (0, 1), (1, 1)]);
}

fn conditional_left_join_indices_preserves_unmatched_left_rows() {
    let (left, right) = dataframe_conditional_inputs(&[1, 3, 5], &[2, 4]);

    let indices = left
        .on(Exec::RETAINED)
        .conditional_left_join_indices(&right, &left_less_than_right(), dataframe_conditional_options())
        .unwrap();

    assert_eq!(sorted_dataframe_join_index_pairs(&indices), vec![(0, 0), (0, 1), (1, 1), (2, JOIN_NO_MATCH)]);
}

fn conditional_full_join_indices_preserves_unmatched_rows_from_both_sides() {
    let (left, right) = dataframe_conditional_inputs(&[2, 5, 9], &[1, 3, 6]);

    let indices = left
        .on(Exec::RETAINED)
        .conditional_full_join_indices(&right, &left_less_than_right(), dataframe_conditional_options())
        .unwrap();

    assert_eq!(
        sorted_dataframe_join_index_pairs(&indices),
        vec![(JOIN_NO_MATCH, 0), (0, 1), (0, 2), (1, 2), (2, JOIN_NO_MATCH)]
    );
}

fn dataframe_conditional_full_join_size_data_matches_materialized_pair_indices_for_unmatched_rows() {
    let (left, right) = dataframe_conditional_inputs(&[2, 5, 9], &[1, 3, 6]);
    let predicate = left_less_than_right();
    let options = dataframe_conditional_options();
    let expected = left.on(Exec::RETAINED).conditional_full_join_indices(&right, &predicate, options.clone()).unwrap();

    let size_data = left.on(Exec::RETAINED).conditional_full_join_size_data(&right, &predicate, options).unwrap();
    let output_size = size_data.output_size();
    let gather_reservation = size_data.gather_map_reservation_size_bytes().unwrap();

    assert_eq!(output_size, expected.len());
    assert_eq!(gather_reservation, 2 * output_size * std::mem::size_of::<i32>());
    assert_eq!(gather_reservation, DataFrameJoinPairIndices::reservation_size_bytes_for_len(output_size).unwrap());
    assert!(size_data.device_allocated_size_bytes().unwrap() > 0);

    let materialized = size_data.materialize_indices(Exec::RETAINED).unwrap();

    assert_eq!(materialized.len(), output_size);
    assert_eq!(materialized.device_allocated_size_bytes().unwrap(), gather_reservation);
    assert_eq!(sorted_dataframe_join_index_pairs(&materialized), sorted_dataframe_join_index_pairs(&expected));
}

fn dataframe_conditional_full_join_size_data_materialize_indices_in_uses_execution_override() {
    // libcudf binds conditional full join retained size data to the stream that built it
    // (cpp/src/join/conditional_join.cu), so size_data and materialize_indices_in must
    // share the same execution context. The override has to flow through both calls.
    let (left, right) = dataframe_conditional_inputs(&[2, 5, 9], &[1, 3, 6]);
    let predicate = left_less_than_right();
    let options = dataframe_conditional_options();
    let expected = left.on(Exec::RETAINED).conditional_full_join_indices(&right, &predicate, options.clone()).unwrap();
    let output_rt = Runtime::new().unwrap();
    let ctx = output_rt.exec();

    let size_data = left.on(ctx).conditional_full_join_size_data(&right, &predicate, options).unwrap();
    let output_size = size_data.output_size();
    let gather_reservation = size_data.gather_map_reservation_size_bytes().unwrap();

    let materialized = size_data.materialize_indices(ctx).unwrap();

    assert_eq!(materialized.len(), output_size);
    assert_eq!(materialized.device_allocated_size_bytes().unwrap(), gather_reservation);
    assert_eq!(materialized.device(), output_rt.device());
    assert_eq!(sorted_dataframe_join_index_pairs(&materialized), sorted_dataframe_join_index_pairs(&expected));
}

fn dataframe_conditional_full_join_size_data_materialize_indices_in_rejects_foreign_stream() {
    // Cross-stream materialization of conditional full join retained size data is a
    // hard upstream error. The binding must surface it cleanly instead of masking the
    // stream-identity contract.
    let (left, right) = dataframe_conditional_inputs(&[2, 5, 9], &[1, 3, 6]);
    let predicate = left_less_than_right();
    let options = dataframe_conditional_options();
    let foreign_rt = Runtime::new().unwrap();
    let foreign_ctx = foreign_rt.exec();

    let size_data = left.on(Exec::RETAINED).conditional_full_join_size_data(&right, &predicate, options).unwrap();
    let err = size_data
        .materialize_indices(foreign_ctx)
        .expect_err("full-join retained size data must be materialized on its creator stream");
    assert_eq!(err.code(), cudf::ErrorCode::ForeignInvalidArgument);
}

fn dataframe_conditional_full_join_size_data_no_match_reports_left_and_right_unmatched_rows() {
    let (left, right) = dataframe_conditional_inputs(&[9, 10], &[1, 2]);
    let predicate = left_less_than_right();
    let size_data = left
        .on(Exec::RETAINED)
        .conditional_full_join_size_data(&right, &predicate, dataframe_conditional_options())
        .unwrap();

    assert_eq!(size_data.output_size(), 4);
    assert_eq!(size_data.gather_map_reservation_size_bytes().unwrap(), 4 * 2 * std::mem::size_of::<i32>());
    assert!(size_data.device_allocated_size_bytes().unwrap() > 0);

    let materialized = size_data.materialize_indices(Exec::RETAINED).unwrap();
    assert_eq!(
        sorted_dataframe_join_index_pairs(&materialized),
        vec![(JOIN_NO_MATCH, 0), (JOIN_NO_MATCH, 1), (0, JOIN_NO_MATCH), (1, JOIN_NO_MATCH)]
    );
}

fn dataframe_conditional_full_join_size_data_empty_left_reports_right_unmatched_rows() {
    let (left, right) = dataframe_conditional_inputs(&[], &[1, 2]);
    let predicate = left_less_than_right();
    let size_data = left
        .on(Exec::RETAINED)
        .conditional_full_join_size_data(&right, &predicate, dataframe_conditional_options())
        .unwrap();

    assert_eq!(size_data.output_size(), 2);
    assert_eq!(size_data.gather_map_reservation_size_bytes().unwrap(), 2 * 2 * std::mem::size_of::<i32>());

    let materialized = size_data.materialize_indices(Exec::RETAINED).unwrap();
    assert_eq!(sorted_dataframe_join_index_pairs(&materialized), vec![(JOIN_NO_MATCH, 0), (JOIN_NO_MATCH, 1)]);
}

fn dataframe_conditional_full_join_size_data_empty_right_reports_left_unmatched_rows() {
    let (left, right) = dataframe_conditional_inputs(&[1, 2], &[]);
    let predicate = left_less_than_right();
    let size_data = left
        .on(Exec::RETAINED)
        .conditional_full_join_size_data(&right, &predicate, dataframe_conditional_options())
        .unwrap();

    assert_eq!(size_data.output_size(), 2);
    assert_eq!(size_data.gather_map_reservation_size_bytes().unwrap(), 2 * 2 * std::mem::size_of::<i32>());

    let materialized = size_data.materialize_indices(Exec::RETAINED).unwrap();
    assert_eq!(sorted_dataframe_join_index_pairs(&materialized), vec![(0, JOIN_NO_MATCH), (1, JOIN_NO_MATCH)]);
}

fn dataframe_conditional_full_join_size_data_empty_input_rejects_non_boolean_predicate() {
    let (left, right) = dataframe_conditional_inputs(&[], &[1, 2]);
    let predicate = left_add_right_non_boolean_predicate();

    left.on(Exec::RETAINED)
        .conditional_full_join_size_data(&right, &predicate, dataframe_conditional_options())
        .expect_err("empty-input retained conditional full join must validate predicate output type");
}

fn dataframe_join_gather_map_reservation_rejects_rows_above_cudf_size_type() {
    let output_rows = i32::MAX as usize + 1;

    DataFrameJoinIndices::reservation_size_bytes_for_len(output_rows)
        .expect_err("single gather-map reservation must reject rows above cudf::size_type");
    DataFrameJoinPairIndices::reservation_size_bytes_for_len(output_rows)
        .expect_err("pair gather-map reservation must reject rows above cudf::size_type");
}

fn conditional_left_semi_and_anti_join_indices_return_left_existence_shapes() {
    let (left, right) = dataframe_conditional_inputs(&[1, 3, 5], &[2, 4]);
    let predicate = left_less_than_right();

    let semi = left
        .on(Exec::RETAINED)
        .conditional_left_semi_join_indices(&right, &predicate, dataframe_conditional_options())
        .unwrap();
    let anti = left
        .on(Exec::RETAINED)
        .conditional_left_anti_join_indices(&right, &predicate, dataframe_conditional_options())
        .unwrap();

    assert_eq!(sorted_dataframe_join_indices(&semi), vec![0, 1]);
    assert_eq!(sorted_dataframe_join_indices(&anti), vec![2]);
}

fn conditional_join_size_and_output_size_hint_match_indices() {
    let (left, right) = dataframe_conditional_inputs(&[1, 3, 5], &[2, 4]);
    let left_ops = left.on(Exec::RETAINED);
    let predicate = left_less_than_right();
    let options = dataframe_conditional_options();

    let inner_size = left_ops.conditional_inner_join_size(&right, &predicate, options.clone()).unwrap();
    assert_eq!(inner_size, 3);
    let inner = left_ops
        .conditional_inner_join_indices_with_output_size(&right, &predicate, options.clone(), inner_size)
        .unwrap();
    assert_eq!(sorted_dataframe_join_index_pairs(&inner), vec![(0, 0), (0, 1), (1, 1)]);

    let left_size = left_ops.conditional_left_join_size(&right, &predicate, options.clone()).unwrap();
    assert_eq!(left_size, 4);
    let left_indices = left_ops
        .conditional_left_join_indices_with_output_size(&right, &predicate, options.clone(), left_size)
        .unwrap();
    assert_eq!(sorted_dataframe_join_index_pairs(&left_indices), vec![(0, 0), (0, 1), (1, 1), (2, JOIN_NO_MATCH)]);

    let semi_size = left_ops.conditional_left_semi_join_size(&right, &predicate, options.clone()).unwrap();
    assert_eq!(semi_size, 2);
    let semi = left_ops
        .conditional_left_semi_join_indices_with_output_size(&right, &predicate, options.clone(), semi_size)
        .unwrap();
    assert_eq!(sorted_dataframe_join_indices(&semi), vec![0, 1]);

    let anti_size = left_ops.conditional_left_anti_join_size(&right, &predicate, options.clone()).unwrap();
    assert_eq!(anti_size, 1);
    let anti =
        left_ops.conditional_left_anti_join_indices_with_output_size(&right, &predicate, options, anti_size).unwrap();
    assert_eq!(sorted_dataframe_join_indices(&anti), vec![2]);
}

fn conditional_inner_join_indices_treats_null_predicate_as_not_matching() {
    let rt = Runtime::new().unwrap();
    let left = DataFrame::from_series([("left_val", rt.nullable_series(&[1i32, 0, 3], &[true, false, true]).unwrap())])
        .unwrap();
    let right = DataFrame::from_series([("right_val", rt.series(&[2i32]).unwrap())]).unwrap();

    let indices = left
        .on(Exec::RETAINED)
        .conditional_inner_join_indices(&right, &left_less_than_right(), dataframe_conditional_options())
        .unwrap();

    assert_eq!(sorted_dataframe_join_index_pairs(&indices), vec![(0, 0)]);
}

fn conditional_inner_join_indices_keeps_literal_scalar_alive_inside_ast() {
    let (left, right) = dataframe_conditional_inputs(&[1, 3, 5], &[0, 0]);
    let literal = AstExpression::literal(scalar(4i32).unwrap()).unwrap();
    let predicate = AstExpression::less(AstExpression::left_column(0).unwrap(), literal).unwrap();

    let indices = left
        .on(Exec::RETAINED)
        .conditional_inner_join_indices(&right, &predicate, dataframe_conditional_options())
        .unwrap();

    assert_eq!(sorted_dataframe_join_index_pairs(&indices), vec![(0, 0), (0, 1), (1, 0), (1, 1)]);
}

fn conditional_inner_join_indices_supports_ast_in_list_predicate() {
    let (left, right) = dataframe_conditional_inputs(&[1, 3, 5], &[2, 4]);
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

    let indices = left
        .on(Exec::RETAINED)
        .conditional_inner_join_indices(&right, &predicate, dataframe_conditional_options())
        .unwrap();

    assert_eq!(sorted_dataframe_join_index_pairs(&indices), vec![(0, 0), (2, 0)]);
}

fn conditional_inner_join_indices_supports_ast_cast_predicate() {
    let (left, right) = dataframe_conditional_inputs(&[1, 3, 5], &[0, 0]);
    let predicate = AstExpression::greater(
        AstExpression::cast_to_float64(AstExpression::left_column(0).unwrap()).unwrap(),
        AstExpression::literal(scalar(2.5f64).unwrap()).unwrap(),
    )
    .unwrap();

    let indices = left
        .on(Exec::RETAINED)
        .conditional_inner_join_indices(&right, &predicate, dataframe_conditional_options())
        .unwrap();

    assert_eq!(sorted_dataframe_join_index_pairs(&indices), vec![(1, 0), (1, 1), (2, 0), (2, 1)]);
}

fn conditional_inner_join_indices_supports_decimal_ast_cast_to_float64() {
    let rt = Runtime::new().unwrap();
    let left = decimal128_frame_from_arrow(&rt, vec![Some(12345), Some(20_050), None], 20, 2);
    let right = DataFrame::from_series([("right_val", rt.series(&[0i32, 1]).unwrap())]).unwrap();
    let predicate = AstExpression::greater(
        AstExpression::cast_to_float64(AstExpression::left_column(0).unwrap()).unwrap(),
        AstExpression::literal(scalar(200.0f64).unwrap()).unwrap(),
    )
    .unwrap();

    let indices = left
        .on(Exec::RETAINED)
        .conditional_inner_join_indices(
            &right,
            &predicate,
            DataFramePredicateJoinOptions::new().left_columns(["amount"]).right_columns(["right_val"]),
        )
        .unwrap();

    assert_eq!(sorted_dataframe_join_index_pairs(&indices), vec![(1, 0), (1, 1)]);
}

#[test]
fn conditional_join_tests_cases() {
    run_named_cases(&[
        (
            "conditional_inner_join_indices_matches_less_than_predicate",
            conditional_inner_join_indices_matches_less_than_predicate,
        ),
        (
            "conditional_left_join_indices_preserves_unmatched_left_rows",
            conditional_left_join_indices_preserves_unmatched_left_rows,
        ),
        (
            "conditional_full_join_indices_preserves_unmatched_rows_from_both_sides",
            conditional_full_join_indices_preserves_unmatched_rows_from_both_sides,
        ),
        (
            "dataframe_conditional_full_join_size_data_matches_materialized_pair_indices_for_unmatched_rows",
            dataframe_conditional_full_join_size_data_matches_materialized_pair_indices_for_unmatched_rows,
        ),
        (
            "dataframe_conditional_full_join_size_data_materialize_indices_in_uses_execution_override",
            dataframe_conditional_full_join_size_data_materialize_indices_in_uses_execution_override,
        ),
        (
            "dataframe_conditional_full_join_size_data_materialize_indices_in_rejects_foreign_stream",
            dataframe_conditional_full_join_size_data_materialize_indices_in_rejects_foreign_stream,
        ),
        (
            "dataframe_conditional_full_join_size_data_no_match_reports_left_and_right_unmatched_rows",
            dataframe_conditional_full_join_size_data_no_match_reports_left_and_right_unmatched_rows,
        ),
        (
            "dataframe_conditional_full_join_size_data_empty_left_reports_right_unmatched_rows",
            dataframe_conditional_full_join_size_data_empty_left_reports_right_unmatched_rows,
        ),
        (
            "dataframe_conditional_full_join_size_data_empty_right_reports_left_unmatched_rows",
            dataframe_conditional_full_join_size_data_empty_right_reports_left_unmatched_rows,
        ),
        (
            "dataframe_conditional_full_join_size_data_empty_input_rejects_non_boolean_predicate",
            dataframe_conditional_full_join_size_data_empty_input_rejects_non_boolean_predicate,
        ),
        (
            "dataframe_join_gather_map_reservation_rejects_rows_above_cudf_size_type",
            dataframe_join_gather_map_reservation_rejects_rows_above_cudf_size_type,
        ),
        (
            "conditional_left_semi_and_anti_join_indices_return_left_existence_shapes",
            conditional_left_semi_and_anti_join_indices_return_left_existence_shapes,
        ),
        (
            "conditional_join_size_and_output_size_hint_match_indices",
            conditional_join_size_and_output_size_hint_match_indices,
        ),
        (
            "conditional_inner_join_indices_treats_null_predicate_as_not_matching",
            conditional_inner_join_indices_treats_null_predicate_as_not_matching,
        ),
        (
            "conditional_inner_join_indices_keeps_literal_scalar_alive_inside_ast",
            conditional_inner_join_indices_keeps_literal_scalar_alive_inside_ast,
        ),
        (
            "conditional_inner_join_indices_supports_ast_in_list_predicate",
            conditional_inner_join_indices_supports_ast_in_list_predicate,
        ),
        (
            "conditional_inner_join_indices_supports_ast_cast_predicate",
            conditional_inner_join_indices_supports_ast_cast_predicate,
        ),
        (
            "conditional_inner_join_indices_supports_decimal_ast_cast_to_float64",
            conditional_inner_join_indices_supports_decimal_ast_cast_to_float64,
        ),
    ]);
}

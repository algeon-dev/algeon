use cudf::{
    DataFrame, DataFrameCrossJoinOptions, DataFrameJoinIndices, DataFrameJoinOptions, DataFrameJoinPairIndices,
    DataFrameMixedJoinOptions, DataFramePredicateJoinOptions, ErrorCode, Exec, JoinType, NullEquality, Runtime, Series,
    ops::ast::AstExpression, runtime::CudaStream, testing::prelude::*,
};

use crate::common::{last_exec_mr_value, last_exec_stream_value, lock_event_observation, reset_exec_observation};

const JOIN_NO_MATCH: i32 = i32::MIN;

fn mixed_left_state_preserves_duplicate_rows_and_late_residual_matches() {
    let rt = Runtime::new().unwrap();
    let probe_rt = Runtime::builder().with_stream(CudaStream::new().unwrap()).build().unwrap();
    let left = DataFrame::from_series([
        ("key", rt.series(&[1i32, 1, 1, 2, 3]).unwrap()),
        ("value", rt.series(&[5i32, 5, 9, 2, 7]).unwrap()),
    ])
    .unwrap();
    let options = DataFrameMixedJoinOptions::new()
        .on(["key"])
        .left_predicate_columns(["value"])
        .right_predicate_columns(["value"]);
    let ctx = Exec::RETAINED;
    for anti in [false, true] {
        let mut state =
            cudf::DataFrameMixedLeftJoinState::new(&left, left_less_than_right(), options.clone(), ctx).unwrap();
        let retained = state.device_allocated_size_bytes().unwrap();
        assert!(retained >= left.num_rows() * 8);
        for (keys, values) in [(&[1i32, 2][..], &[4i32, 3][..]), (&[][..], &[][..]), (&[1, 1][..], &[6, 6][..])] {
            let batch = DataFrame::from_series([
                ("key", probe_rt.series(keys).unwrap()),
                ("value", probe_rt.series(values).unwrap()),
            ])
            .unwrap();
            state.consume(&batch).unwrap();
            drop(batch);
            assert_eq!(state.device_allocated_size_bytes().unwrap(), retained);
        }
        let finalized = state.seal(anti).unwrap();
        let expected = if anti { vec![2, 4] } else { vec![0, 1, 3] };
        assert_eq!(finalized.output_size(), expected.len());
        assert_eq!(finalized.gather_map_reservation_size_bytes().unwrap(), expected.len() * 4);
        let indices = finalized.materialize_indices().unwrap();
        drop(finalized);
        assert_eq!(sorted_dataframe_join_indices(&indices), expected);
    }
}

fn mixed_left_state_nulls_empty_inputs_and_zero_batches_obey_semi_anti_semantics() {
    let rt = Runtime::new().unwrap();
    let ctx = Exec::RETAINED;
    let left = DataFrame::from_series([
        ("key", rt.series_opt(&[None, Some(1i32), Some(2), Some(3)]).unwrap()),
        ("value", rt.series_opt(&[Some(1i32), None, Some(1), Some(5)]).unwrap()),
    ])
    .unwrap();
    let right = DataFrame::from_series([
        ("key", rt.series_opt(&[None, Some(1i32), Some(2), Some(3)]).unwrap()),
        ("value", rt.series_opt(&[Some(2i32), Some(8), None, Some(6)]).unwrap()),
    ])
    .unwrap();
    for nulls in [NullEquality::Equal, NullEquality::Unequal] {
        for anti in [false, true] {
            for consume in [false, true] {
                let options = DataFrameMixedJoinOptions::new()
                    .on(["key"])
                    .left_predicate_columns(["value"])
                    .right_predicate_columns(["value"])
                    .null_equality(nulls);
                let mut state =
                    cudf::DataFrameMixedLeftJoinState::new(&left, left_less_than_right(), options, ctx).unwrap();
                if consume {
                    state.consume(&right).unwrap();
                }
                let expected: Vec<i32> = (0..4)
                    .filter(|row| {
                        let matched = consume && (*row == 3 || (*row == 0 && nulls == NullEquality::Equal));
                        matched != anti
                    })
                    .collect();
                let finalized = state.seal(anti).unwrap();
                assert_eq!(sorted_dataframe_join_indices(&finalized.materialize_indices().unwrap()), expected);
            }
        }
    }
    let empty = DataFrame::from_series([
        ("key", rt.series(&[] as &[i32]).unwrap()),
        ("value", rt.series(&[] as &[i32]).unwrap()),
    ])
    .unwrap();
    for anti in [false, true] {
        let options = DataFrameMixedJoinOptions::new()
            .on(["key"])
            .left_predicate_columns(["value"])
            .right_predicate_columns(["value"]);
        let mut state = cudf::DataFrameMixedLeftJoinState::new(&empty, left_less_than_right(), options, ctx).unwrap();
        state.consume(&right).unwrap();
        assert!(state.seal(anti).unwrap().materialize_indices().unwrap().is_empty());
    }
}

fn left_less_than_right() -> AstExpression {
    AstExpression::less(AstExpression::left_column(0).unwrap(), AstExpression::right_column(0).unwrap()).unwrap()
}

fn sorted_dataframe_join_index_pairs(indices: &DataFrameJoinPairIndices) -> Vec<(i32, i32)> {
    let left = indices.left().indices().to_vec_strict::<i32>().unwrap();
    let right = indices.right().indices().to_vec_strict::<i32>().unwrap();

    let mut pairs: Vec<_> = left.into_iter().zip(right).collect();
    pairs.sort();
    pairs
}

fn sorted_dataframe_join_indices(indices: &DataFrameJoinIndices) -> Vec<i32> {
    let mut rows = indices.indices().to_vec_strict::<i32>().unwrap();
    rows.sort();
    rows
}

fn assert_pair_indices_execution_domain(
    indices: &DataFrameJoinPairIndices,
    expected_stream: usize,
    expected_mr: usize,
) {
    assert_eq!(indices.left().indices().as_column().testing_execution_domain_stream_value(), expected_stream);
    assert_eq!(indices.right().indices().as_column().testing_execution_domain_stream_value(), expected_stream);
    assert_eq!(indices.left().indices().as_column().testing_execution_domain_mr_value(), expected_mr);
    assert_eq!(indices.right().indices().as_column().testing_execution_domain_mr_value(), expected_mr);
}

fn assert_single_indices_execution_domain(indices: &DataFrameJoinIndices, expected_stream: usize, expected_mr: usize) {
    assert_eq!(indices.indices().as_column().testing_execution_domain_stream_value(), expected_stream);
    assert_eq!(indices.indices().as_column().testing_execution_domain_mr_value(), expected_mr);
}

fn dataframe_mixed_inputs() -> (DataFrame, DataFrame) {
    let rt = Runtime::new().unwrap();
    let left = DataFrame::from_series([
        ("id", rt.series(&[1i32, 1, 2, 2, 4]).unwrap()),
        ("left_val", rt.series(&[5i32, 7, 3, 9, 1]).unwrap()),
    ])
    .unwrap();
    let right = DataFrame::from_series([
        ("id", rt.series(&[1i32, 1, 2, 2, 3]).unwrap()),
        ("right_val", rt.series(&[6i32, 8, 4, 7, 10]).unwrap()),
    ])
    .unwrap();

    (left, right)
}

fn dataframe_mixed_join_options() -> DataFrameMixedJoinOptions {
    DataFrameMixedJoinOptions::new()
        .on(["id"])
        .left_predicate_columns(["left_val"])
        .right_predicate_columns(["right_val"])
}

fn dataframe_filter_join_indices(
    left: &DataFrame,
    right: &DataFrame,
    equality: &DataFrameJoinPairIndices,
    join_type: JoinType,
) -> DataFrameJoinPairIndices {
    left.on(Exec::RETAINED)
        .filter_join_indices(
            right,
            equality.left(),
            equality.right(),
            &left_less_than_right(),
            DataFramePredicateJoinOptions::new().left_columns(["left_val"]).right_columns(["right_val"]),
            join_type,
        )
        .unwrap()
}

fn filter_join_indices_error(rt: &Runtime, left_indices: Series, right_indices: Series) -> cudf::Error {
    let left = DataFrame::from_series([("v", rt.series(&[1i32, 3]).unwrap())]).unwrap();
    let right = DataFrame::from_series([("v", rt.series(&[2i32]).unwrap())]).unwrap();

    left.on(Exec::RETAINED)
        .filter_join_indices(
            &right,
            &DataFrameJoinIndices::from_series(left_indices),
            &DataFrameJoinIndices::from_series(right_indices),
            &left_less_than_right(),
            DataFramePredicateJoinOptions::new().left_columns(["v"]).right_columns(["v"]),
            JoinType::Inner,
        )
        .unwrap_err()
}

fn filter_join_indices_rejects_non_int32_index_columns() {
    let rt = Runtime::new().unwrap();

    let err = filter_join_indices_error(&rt, rt.series(&[0i64]).unwrap(), rt.series(&[0i32]).unwrap());

    assert_eq!(err.code(), ErrorCode::TypeMismatch);
}

fn filter_join_indices_rejects_mismatched_index_lengths() {
    let rt = Runtime::new().unwrap();

    let err = filter_join_indices_error(&rt, rt.series(&[0i32, 1]).unwrap(), rt.series(&[0i32]).unwrap());

    assert_eq!(err.code(), ErrorCode::ColumnLengthMismatch);
}

fn filter_join_indices_rejects_nullable_index_columns() {
    let rt = Runtime::new().unwrap();

    let err =
        filter_join_indices_error(&rt, rt.nullable_series(&[0i32], &[false]).unwrap(), rt.series(&[0i32]).unwrap());

    assert_eq!(err.code(), ErrorCode::NullUnexpected);
    assert_eq!(err.detail_value("reason"), Some("nullable_join_indices"));
}

fn filter_join_indices_inner_matches_mixed_inner_join_indices() {
    let (left, right) = dataframe_mixed_inputs();
    let equality = left.on(Exec::RETAINED).inner_join_indices(&right, DataFrameJoinOptions::new().on(["id"])).unwrap();

    let filtered = dataframe_filter_join_indices(&left, &right, &equality, JoinType::Inner);
    let mixed = left
        .on(Exec::RETAINED)
        .mixed_inner_join_indices(&right, &left_less_than_right(), dataframe_mixed_join_options())
        .unwrap();

    assert_eq!(sorted_dataframe_join_index_pairs(&filtered), sorted_dataframe_join_index_pairs(&mixed));
}

fn filter_join_indices_left_matches_mixed_left_join_indices() {
    let (left, right) = dataframe_mixed_inputs();
    let equality = left.on(Exec::RETAINED).left_join_indices(&right, DataFrameJoinOptions::new().on(["id"])).unwrap();

    let filtered = dataframe_filter_join_indices(&left, &right, &equality, JoinType::Left);
    let mixed = left
        .on(Exec::RETAINED)
        .mixed_left_join_indices(&right, &left_less_than_right(), dataframe_mixed_join_options())
        .unwrap();

    assert_eq!(sorted_dataframe_join_index_pairs(&filtered), sorted_dataframe_join_index_pairs(&mixed));
}

fn filter_join_indices_full_matches_mixed_full_join_indices() {
    let rt = Runtime::new().unwrap();
    let left = DataFrame::from_series([
        ("id", rt.series(&[1i32, 2, 3]).unwrap()),
        ("left_val", rt.series(&[10i32, 20, 30]).unwrap()),
    ])
    .unwrap();
    let right = DataFrame::from_series([
        ("id", rt.series(&[1i32, 2, 4]).unwrap()),
        ("right_val", rt.series(&[15i32, 15, 25]).unwrap()),
    ])
    .unwrap();
    let equality = left.on(Exec::RETAINED).full_join_indices(&right, DataFrameJoinOptions::new().on(["id"])).unwrap();

    let filtered = dataframe_filter_join_indices(&left, &right, &equality, JoinType::Full);
    let mixed = left
        .on(Exec::RETAINED)
        .mixed_full_join_indices(&right, &left_less_than_right(), dataframe_mixed_join_options())
        .unwrap();

    assert_eq!(sorted_dataframe_join_index_pairs(&filtered), sorted_dataframe_join_index_pairs(&mixed));
}

fn filter_join_indices_full_uses_raw_pair_semantics_for_duplicate_matches() {
    let (left, right) = dataframe_mixed_inputs();
    let equality = left.on(Exec::RETAINED).full_join_indices(&right, DataFrameJoinOptions::new().on(["id"])).unwrap();

    let filtered = dataframe_filter_join_indices(&left, &right, &equality, JoinType::Full);

    assert_eq!(
        sorted_dataframe_join_index_pairs(&filtered),
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

fn mixed_left_join_indices_preserves_unmatched_left_rows() {
    let (left, right) = dataframe_mixed_inputs();

    let indices = left
        .on(Exec::RETAINED)
        .mixed_left_join_indices(&right, &left_less_than_right(), dataframe_mixed_join_options())
        .unwrap();

    assert_eq!(
        sorted_dataframe_join_index_pairs(&indices),
        vec![(0, 0), (0, 1), (1, 1), (2, 2), (2, 3), (3, JOIN_NO_MATCH), (4, JOIN_NO_MATCH)]
    );
}

fn mixed_full_join_indices_preserves_unmatched_rows_from_both_sides() {
    let (left, right) = dataframe_mixed_inputs();

    let indices = left
        .on(Exec::RETAINED)
        .mixed_full_join_indices(&right, &left_less_than_right(), dataframe_mixed_join_options())
        .unwrap();

    assert_eq!(
        sorted_dataframe_join_index_pairs(&indices),
        vec![(JOIN_NO_MATCH, 4), (0, 0), (0, 1), (1, 1), (2, 2), (2, 3), (3, JOIN_NO_MATCH), (4, JOIN_NO_MATCH),]
    );
}

fn mixed_left_semi_and_anti_join_indices_return_left_existence_shapes() {
    let (left, right) = dataframe_mixed_inputs();
    let predicate = left_less_than_right();

    let semi = left
        .on(Exec::RETAINED)
        .mixed_left_semi_join_indices(&right, &predicate, dataframe_mixed_join_options())
        .unwrap();
    let anti = left
        .on(Exec::RETAINED)
        .mixed_left_anti_join_indices(&right, &predicate, dataframe_mixed_join_options())
        .unwrap();

    assert_eq!(sorted_dataframe_join_indices(&semi), vec![0, 1, 2]);
    assert_eq!(sorted_dataframe_join_indices(&anti), vec![3, 4]);
}

fn mixed_inner_join_indices_honors_null_equality_for_equi_keys() {
    let rt = Runtime::new().unwrap();
    let left = DataFrame::from_series([
        ("id", rt.nullable_series(&[0i32, 1], &[false, true]).unwrap()),
        ("left_val", rt.series(&[1i32, 1]).unwrap()),
    ])
    .unwrap();
    let right = DataFrame::from_series([
        ("id", rt.nullable_series(&[0i32, 1], &[false, true]).unwrap()),
        ("right_val", rt.series(&[2i32, 2]).unwrap()),
    ])
    .unwrap();
    let predicate = left_less_than_right();

    let equal = left
        .on(Exec::RETAINED)
        .mixed_inner_join_indices(&right, &predicate, dataframe_mixed_join_options().null_equality(NullEquality::Equal))
        .unwrap();
    let unequal = left
        .on(Exec::RETAINED)
        .mixed_inner_join_indices(
            &right,
            &predicate,
            dataframe_mixed_join_options().null_equality(NullEquality::Unequal),
        )
        .unwrap();

    assert_eq!(sorted_dataframe_join_index_pairs(&equal), vec![(0, 0), (1, 1)]);
    assert_eq!(sorted_dataframe_join_index_pairs(&unequal), vec![(1, 1)]);
}

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
        .on(Exec::RETAINED)
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
        .on(Exec::RETAINED)
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

fn dataframe_mixed_inner_join_size_data_materializes_existing_indices_contract() {
    let (left, right) = dataframe_mixed_inputs();
    let predicate = left_less_than_right();
    let options = dataframe_mixed_join_options();
    let expected = left.on(Exec::RETAINED).mixed_inner_join_indices(&right, &predicate, options.clone()).unwrap();

    let size_data = left.on(Exec::RETAINED).mixed_inner_join_size_data(&right, &predicate, options).unwrap();
    let output_size = size_data.output_size();
    let gather_reservation = size_data.gather_map_reservation_size_bytes().unwrap();

    assert_eq!(output_size, expected.len());
    assert_eq!(gather_reservation, DataFrameJoinPairIndices::reservation_size_bytes_for_len(output_size).unwrap());
    assert!(size_data.device_allocated_size_bytes().unwrap() > 0);

    let materialized = size_data.materialize_indices(Exec::RETAINED).unwrap();

    assert_eq!(materialized.len(), output_size);
    assert_eq!(materialized.device_allocated_size_bytes().unwrap(), gather_reservation);
    assert_eq!(sorted_dataframe_join_index_pairs(&materialized), sorted_dataframe_join_index_pairs(&expected));
}

fn dataframe_mixed_left_join_size_data_materialize_indices_in_matches_existing_indices_contract() {
    let (left, right) = dataframe_mixed_inputs();
    let predicate = left_less_than_right();
    let options = dataframe_mixed_join_options();
    let expected = left.on(Exec::RETAINED).mixed_left_join_indices(&right, &predicate, options.clone()).unwrap();
    let output_rt = Runtime::new().unwrap();
    let ctx = output_rt.exec();

    let size_data = left.on(Exec::RETAINED).mixed_left_join_size_data(&right, &predicate, options).unwrap();
    let output_size = size_data.output_size();
    let gather_reservation = size_data.gather_map_reservation_size_bytes().unwrap();

    assert_eq!(output_size, expected.len());
    assert_eq!(gather_reservation, DataFrameJoinPairIndices::reservation_size_bytes_for_len(output_size).unwrap());
    assert!(size_data.device_allocated_size_bytes().unwrap() > 0);

    let materialized = size_data.materialize_indices(ctx).unwrap();

    assert_eq!(materialized.len(), output_size);
    assert_eq!(materialized.device_allocated_size_bytes().unwrap(), gather_reservation);
    assert_eq!(sorted_dataframe_join_index_pairs(&materialized), sorted_dataframe_join_index_pairs(&expected));
}

fn dataframe_mixed_full_join_size_data_materializes_existing_indices_contract() {
    let (left, right) = dataframe_mixed_inputs();
    let predicate = left_less_than_right();
    let options = dataframe_mixed_join_options();
    let expected = left.on(Exec::RETAINED).mixed_full_join_indices(&right, &predicate, options.clone()).unwrap();

    let size_data = left.on(Exec::RETAINED).mixed_full_join_size_data(&right, &predicate, options).unwrap();
    let output_size = size_data.output_size();
    let gather_reservation = size_data.gather_map_reservation_size_bytes().unwrap();

    assert_eq!(output_size, expected.len());
    assert_eq!(gather_reservation, DataFrameJoinPairIndices::reservation_size_bytes_for_len(output_size).unwrap());
    assert!(size_data.device_allocated_size_bytes().unwrap() > 0);

    let materialized = size_data.materialize_indices(Exec::RETAINED).unwrap();

    assert_eq!(materialized.len(), output_size);
    assert_eq!(materialized.device_allocated_size_bytes().unwrap(), gather_reservation);
    assert_eq!(sorted_dataframe_join_index_pairs(&materialized), sorted_dataframe_join_index_pairs(&expected));
}

fn dataframe_mixed_full_join_size_data_begin_and_materialize_in_use_borrowed_handles() {
    // libcudf binds mixed full join retained size data to the stream that built it
    // (cpp/src/join/mixed_join.cu), so begin and materialize must share the same
    // execution context. The override still has to flow through both calls and
    // surface on the materialized indices.
    let _event_guard = lock_event_observation();
    let (left, right) = dataframe_mixed_inputs();
    let predicate = left_less_than_right();
    let options = dataframe_mixed_join_options();
    let expected = left.on(Exec::RETAINED).mixed_full_join_indices(&right, &predicate, options.clone()).unwrap();
    let borrowed_rt = Runtime::new().unwrap();
    let borrowed_ctx = borrowed_rt.exec();
    let borrowed_stream = borrowed_rt.stream().testing_raw_handle_value();
    let borrowed_mr = borrowed_rt.memory_resource().testing_raw_handle_value();

    reset_exec_observation();
    let size_data = left.on(borrowed_ctx).mixed_full_join_size_data(&right, &predicate, options).unwrap();

    assert_eq!(last_exec_stream_value(), borrowed_stream);
    assert_eq!(last_exec_mr_value(), borrowed_mr);
    let output_size = size_data.output_size();
    let gather_reservation = size_data.gather_map_reservation_size_bytes().unwrap();
    assert_eq!(output_size, expected.len());
    assert_eq!(gather_reservation, DataFrameJoinPairIndices::reservation_size_bytes_for_len(output_size).unwrap());
    assert!(size_data.device_allocated_size_bytes().unwrap() > 0);

    reset_exec_observation();
    let materialized = size_data.materialize_indices(borrowed_ctx).unwrap();

    assert_eq!(last_exec_stream_value(), borrowed_stream);
    assert_eq!(last_exec_mr_value(), borrowed_mr);
    assert_pair_indices_execution_domain(&materialized, borrowed_stream, borrowed_mr);
    assert_eq!(materialized.device(), borrowed_rt.device());
    assert_eq!(materialized.len(), output_size);
    assert_eq!(materialized.device_allocated_size_bytes().unwrap(), gather_reservation);
    assert_eq!(sorted_dataframe_join_index_pairs(&materialized), sorted_dataframe_join_index_pairs(&expected));
}

fn dataframe_mixed_full_join_size_data_materialize_indices_in_rejects_foreign_stream() {
    // Cross-stream materialization of mixed full join retained size data is a hard
    // upstream error. The binding surfaces it with the ForeignInvalidArgument code,
    // and the stream-identity check stays visible to callers.
    let (left, right) = dataframe_mixed_inputs();
    let predicate = left_less_than_right();
    let options = dataframe_mixed_join_options();
    let begin_rt = Runtime::new().unwrap();
    let materialize_rt = Runtime::new().unwrap();
    let begin_ctx = begin_rt.exec();
    let materialize_ctx = materialize_rt.exec();

    let size_data = left.on(begin_ctx).mixed_full_join_size_data(&right, &predicate, options).unwrap();
    let err = size_data
        .materialize_indices(materialize_ctx)
        .expect_err("mixed full join retained size data must be materialized on its creator stream");
    assert_eq!(err.code(), ErrorCode::ForeignInvalidArgument);
}

fn dataframe_mixed_left_semi_join_size_data_materializes_single_indices_contract() {
    let (left, right) = dataframe_mixed_inputs();
    let predicate = left_less_than_right();
    let options = dataframe_mixed_join_options();
    let expected = left.on(Exec::RETAINED).mixed_left_semi_join_indices(&right, &predicate, options.clone()).unwrap();

    let size_data = left.on(Exec::RETAINED).mixed_left_semi_join_size_data(&right, &predicate, options).unwrap();
    let output_size = size_data.output_size();
    let gather_reservation = size_data.gather_map_reservation_size_bytes().unwrap();

    assert_eq!(output_size, expected.len());
    assert_eq!(gather_reservation, DataFrameJoinIndices::reservation_size_bytes_for_len(output_size).unwrap());
    assert_eq!(DataFrameJoinPairIndices::reservation_size_bytes_for_len(output_size).unwrap(), gather_reservation * 2);
    assert!(size_data.device_allocated_size_bytes().unwrap() > 0);

    let materialized = size_data.materialize_indices(Exec::RETAINED).unwrap();

    assert_eq!(materialized.len(), output_size);
    assert_eq!(materialized.device_allocated_size_bytes().unwrap(), gather_reservation);
    assert_eq!(sorted_dataframe_join_indices(&materialized), sorted_dataframe_join_indices(&expected));
}

fn dataframe_mixed_left_semi_join_size_data_begin_and_materialize_in_use_borrowed_handles() {
    let _event_guard = lock_event_observation();
    let (left, right) = dataframe_mixed_inputs();
    let predicate = left_less_than_right();
    let options = dataframe_mixed_join_options();
    let expected = left.on(Exec::RETAINED).mixed_left_semi_join_indices(&right, &predicate, options.clone()).unwrap();
    let begin_rt = Runtime::new().unwrap();
    let materialize_rt = Runtime::new().unwrap();
    let begin_ctx = begin_rt.exec();
    let materialize_ctx = materialize_rt.exec();
    let begin_stream = begin_rt.stream().testing_raw_handle_value();
    let begin_mr = begin_rt.memory_resource().testing_raw_handle_value();
    let materialize_stream = materialize_rt.stream().testing_raw_handle_value();
    let materialize_mr = materialize_rt.memory_resource().testing_raw_handle_value();

    reset_exec_observation();
    let size_data = left.on(begin_ctx).mixed_left_semi_join_size_data(&right, &predicate, options).unwrap();

    assert_eq!(last_exec_stream_value(), begin_stream);
    assert_eq!(last_exec_mr_value(), begin_mr);
    let output_size = size_data.output_size();
    let gather_reservation = size_data.gather_map_reservation_size_bytes().unwrap();
    assert_eq!(output_size, expected.len());
    assert_eq!(gather_reservation, DataFrameJoinIndices::reservation_size_bytes_for_len(output_size).unwrap());
    assert_eq!(DataFrameJoinPairIndices::reservation_size_bytes_for_len(output_size).unwrap(), gather_reservation * 2);
    assert!(size_data.device_allocated_size_bytes().unwrap() > 0);

    reset_exec_observation();
    let materialized = size_data.materialize_indices(materialize_ctx).unwrap();

    assert_eq!(last_exec_stream_value(), materialize_stream);
    assert_eq!(last_exec_mr_value(), materialize_mr);
    assert_single_indices_execution_domain(&materialized, materialize_stream, materialize_mr);
    assert_eq!(materialized.device(), materialize_rt.device());
    assert_eq!(materialized.len(), output_size);
    assert_eq!(materialized.device_allocated_size_bytes().unwrap(), gather_reservation);
    assert_eq!(sorted_dataframe_join_indices(&materialized), sorted_dataframe_join_indices(&expected));
}

fn dataframe_mixed_left_anti_join_size_data_materialize_indices_in_matches_existing_indices_contract() {
    let (left, right) = dataframe_mixed_inputs();
    let predicate = left_less_than_right();
    let options = dataframe_mixed_join_options();
    let expected = left.on(Exec::RETAINED).mixed_left_anti_join_indices(&right, &predicate, options.clone()).unwrap();
    let output_rt = Runtime::new().unwrap();
    let ctx = output_rt.exec();

    let size_data = left.on(Exec::RETAINED).mixed_left_anti_join_size_data(&right, &predicate, options).unwrap();
    let output_size = size_data.output_size();
    let gather_reservation = size_data.gather_map_reservation_size_bytes().unwrap();

    assert_eq!(output_size, expected.len());
    assert_eq!(gather_reservation, DataFrameJoinIndices::reservation_size_bytes_for_len(output_size).unwrap());
    assert_eq!(DataFrameJoinPairIndices::reservation_size_bytes_for_len(output_size).unwrap(), gather_reservation * 2);
    assert!(size_data.device_allocated_size_bytes().unwrap() > 0);

    let materialized = size_data.materialize_indices(ctx).unwrap();

    assert_eq!(materialized.len(), output_size);
    assert_eq!(materialized.device_allocated_size_bytes().unwrap(), gather_reservation);
    assert_eq!(sorted_dataframe_join_indices(&materialized), sorted_dataframe_join_indices(&expected));
}

fn dataframe_mixed_left_anti_join_size_data_begin_and_materialize_in_use_borrowed_handles() {
    let _event_guard = lock_event_observation();
    let (left, right) = dataframe_mixed_inputs();
    let predicate = left_less_than_right();
    let options = dataframe_mixed_join_options();
    let expected = left.on(Exec::RETAINED).mixed_left_anti_join_indices(&right, &predicate, options.clone()).unwrap();
    let begin_rt = Runtime::new().unwrap();
    let materialize_rt = Runtime::new().unwrap();
    let begin_ctx = begin_rt.exec();
    let materialize_ctx = materialize_rt.exec();
    let begin_stream = begin_rt.stream().testing_raw_handle_value();
    let begin_mr = begin_rt.memory_resource().testing_raw_handle_value();
    let materialize_stream = materialize_rt.stream().testing_raw_handle_value();
    let materialize_mr = materialize_rt.memory_resource().testing_raw_handle_value();

    reset_exec_observation();
    let size_data = left.on(begin_ctx).mixed_left_anti_join_size_data(&right, &predicate, options).unwrap();

    assert_eq!(last_exec_stream_value(), begin_stream);
    assert_eq!(last_exec_mr_value(), begin_mr);
    let output_size = size_data.output_size();
    let gather_reservation = size_data.gather_map_reservation_size_bytes().unwrap();
    assert_eq!(output_size, expected.len());
    assert_eq!(gather_reservation, DataFrameJoinIndices::reservation_size_bytes_for_len(output_size).unwrap());
    assert_eq!(DataFrameJoinPairIndices::reservation_size_bytes_for_len(output_size).unwrap(), gather_reservation * 2);
    assert!(size_data.device_allocated_size_bytes().unwrap() > 0);

    reset_exec_observation();
    let materialized = size_data.materialize_indices(materialize_ctx).unwrap();

    assert_eq!(last_exec_stream_value(), materialize_stream);
    assert_eq!(last_exec_mr_value(), materialize_mr);
    assert_single_indices_execution_domain(&materialized, materialize_stream, materialize_mr);
    assert_eq!(materialized.device(), materialize_rt.device());
    assert_eq!(materialized.len(), output_size);
    assert_eq!(materialized.device_allocated_size_bytes().unwrap(), gather_reservation);
    assert_eq!(sorted_dataframe_join_indices(&materialized), sorted_dataframe_join_indices(&expected));
}

fn dataframe_mixed_inner_join_size_data_empty_result_has_zero_gather_reservation() {
    let rt = Runtime::new().unwrap();
    let left = DataFrame::from_series([
        ("id", rt.series(&[1i32, 2]).unwrap()),
        ("left_val", rt.series(&[10i32, 20]).unwrap()),
    ])
    .unwrap();
    let right = DataFrame::from_series([
        ("id", rt.series(&[3i32, 4]).unwrap()),
        ("right_val", rt.series(&[30i32, 40]).unwrap()),
    ])
    .unwrap();
    let predicate = left_less_than_right();

    let size_data =
        left.on(Exec::RETAINED).mixed_inner_join_size_data(&right, &predicate, dataframe_mixed_join_options()).unwrap();

    assert_eq!(size_data.output_size(), 0);
    assert_eq!(size_data.gather_map_reservation_size_bytes().unwrap(), 0);

    let materialized = size_data.materialize_indices(Exec::RETAINED).unwrap();

    assert!(materialized.is_empty());
    assert_eq!(materialized.device_allocated_size_bytes().unwrap(), 0);
}

fn dataframe_mixed_left_semi_join_size_data_empty_result_has_zero_gather_reservation() {
    let rt = Runtime::new().unwrap();
    let left = DataFrame::from_series([
        ("id", rt.series(&[1i32, 2]).unwrap()),
        ("left_val", rt.series(&[10i32, 20]).unwrap()),
    ])
    .unwrap();
    let right = DataFrame::from_series([
        ("id", rt.series(&[3i32, 4]).unwrap()),
        ("right_val", rt.series(&[30i32, 40]).unwrap()),
    ])
    .unwrap();
    let predicate = left_less_than_right();

    let size_data = left
        .on(Exec::RETAINED)
        .mixed_left_semi_join_size_data(&right, &predicate, dataframe_mixed_join_options())
        .unwrap();

    assert_eq!(size_data.output_size(), 0);
    assert_eq!(size_data.gather_map_reservation_size_bytes().unwrap(), 0);

    let materialized = size_data.materialize_indices(Exec::RETAINED).unwrap();

    assert!(materialized.is_empty());
    assert_eq!(materialized.device_allocated_size_bytes().unwrap(), 0);
}

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
    let equality = left.on(Exec::RETAINED).inner_join_indices(&right, DataFrameJoinOptions::new().on(["id"])).unwrap();
    let predicate = left_less_than_right();

    let filtered = left
        .on(Exec::RETAINED)
        .filter_join_indices(
            &right,
            equality.left(),
            equality.right(),
            &predicate,
            DataFramePredicateJoinOptions::new().left_columns(["left_val"]).right_columns(["right_val"]),
            JoinType::Inner,
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

pub(super) const CASES: &[(&str, fn())] = &[
    (
        "mixed_join_tests::mixed_left_state_preserves_duplicate_rows_and_late_residual_matches",
        mixed_left_state_preserves_duplicate_rows_and_late_residual_matches,
    ),
    (
        "mixed_join_tests::mixed_left_state_nulls_empty_inputs_and_zero_batches_obey_semi_anti_semantics",
        mixed_left_state_nulls_empty_inputs_and_zero_batches_obey_semi_anti_semantics,
    ),
    (
        "mixed_join_tests::filter_join_indices_rejects_non_int32_index_columns",
        filter_join_indices_rejects_non_int32_index_columns,
    ),
    (
        "mixed_join_tests::filter_join_indices_rejects_mismatched_index_lengths",
        filter_join_indices_rejects_mismatched_index_lengths,
    ),
    (
        "mixed_join_tests::filter_join_indices_rejects_nullable_index_columns",
        filter_join_indices_rejects_nullable_index_columns,
    ),
    (
        "mixed_join_tests::filter_join_indices_inner_matches_mixed_inner_join_indices",
        filter_join_indices_inner_matches_mixed_inner_join_indices,
    ),
    (
        "mixed_join_tests::filter_join_indices_left_matches_mixed_left_join_indices",
        filter_join_indices_left_matches_mixed_left_join_indices,
    ),
    (
        "mixed_join_tests::filter_join_indices_full_matches_mixed_full_join_indices",
        filter_join_indices_full_matches_mixed_full_join_indices,
    ),
    (
        "mixed_join_tests::filter_join_indices_full_uses_raw_pair_semantics_for_duplicate_matches",
        filter_join_indices_full_uses_raw_pair_semantics_for_duplicate_matches,
    ),
    (
        "mixed_join_tests::mixed_left_join_indices_preserves_unmatched_left_rows",
        mixed_left_join_indices_preserves_unmatched_left_rows,
    ),
    (
        "mixed_join_tests::mixed_full_join_indices_preserves_unmatched_rows_from_both_sides",
        mixed_full_join_indices_preserves_unmatched_rows_from_both_sides,
    ),
    (
        "mixed_join_tests::mixed_left_semi_and_anti_join_indices_return_left_existence_shapes",
        mixed_left_semi_and_anti_join_indices_return_left_existence_shapes,
    ),
    (
        "mixed_join_tests::mixed_inner_join_indices_honors_null_equality_for_equi_keys",
        mixed_inner_join_indices_honors_null_equality_for_equi_keys,
    ),
    (
        "mixed_join_tests::dataframe_projected_cross_join_supports_output_aliases",
        dataframe_projected_cross_join_supports_output_aliases,
    ),
    (
        "mixed_join_tests::dataframe_mixed_join_indices_use_named_equality_and_predicate_columns",
        dataframe_mixed_join_indices_use_named_equality_and_predicate_columns,
    ),
    (
        "mixed_join_tests::dataframe_mixed_inner_join_size_data_materializes_existing_indices_contract",
        dataframe_mixed_inner_join_size_data_materializes_existing_indices_contract,
    ),
    (
        "mixed_join_tests::dataframe_mixed_left_join_size_data_materialize_indices_in_matches_existing_indices_contract",
        dataframe_mixed_left_join_size_data_materialize_indices_in_matches_existing_indices_contract,
    ),
    (
        "mixed_join_tests::dataframe_mixed_full_join_size_data_materializes_existing_indices_contract",
        dataframe_mixed_full_join_size_data_materializes_existing_indices_contract,
    ),
    (
        "mixed_join_tests::dataframe_mixed_full_join_size_data_begin_and_materialize_in_use_borrowed_handles",
        dataframe_mixed_full_join_size_data_begin_and_materialize_in_use_borrowed_handles,
    ),
    (
        "mixed_join_tests::dataframe_mixed_full_join_size_data_materialize_indices_in_rejects_foreign_stream",
        dataframe_mixed_full_join_size_data_materialize_indices_in_rejects_foreign_stream,
    ),
    (
        "mixed_join_tests::dataframe_mixed_left_semi_join_size_data_materializes_single_indices_contract",
        dataframe_mixed_left_semi_join_size_data_materializes_single_indices_contract,
    ),
    (
        "mixed_join_tests::dataframe_mixed_left_semi_join_size_data_begin_and_materialize_in_use_borrowed_handles",
        dataframe_mixed_left_semi_join_size_data_begin_and_materialize_in_use_borrowed_handles,
    ),
    (
        "mixed_join_tests::dataframe_mixed_left_anti_join_size_data_materialize_indices_in_matches_existing_indices_contract",
        dataframe_mixed_left_anti_join_size_data_materialize_indices_in_matches_existing_indices_contract,
    ),
    (
        "mixed_join_tests::dataframe_mixed_left_anti_join_size_data_begin_and_materialize_in_use_borrowed_handles",
        dataframe_mixed_left_anti_join_size_data_begin_and_materialize_in_use_borrowed_handles,
    ),
    (
        "mixed_join_tests::dataframe_mixed_inner_join_size_data_empty_result_has_zero_gather_reservation",
        dataframe_mixed_inner_join_size_data_empty_result_has_zero_gather_reservation,
    ),
    (
        "mixed_join_tests::dataframe_mixed_left_semi_join_size_data_empty_result_has_zero_gather_reservation",
        dataframe_mixed_left_semi_join_size_data_empty_result_has_zero_gather_reservation,
    ),
    (
        "mixed_join_tests::dataframe_filter_join_indices_filter_existing_hash_probe_pairs",
        dataframe_filter_join_indices_filter_existing_hash_probe_pairs,
    ),
];

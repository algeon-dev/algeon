#[path = "../ops/datetime/datetime_tests.rs"]
mod datetime_tests;
mod join_hash_join_execution_tests {
    use cudf::{
        DataFrame, DataFrameCrossJoinOptions, DataFrameHashJoinOptions, DataFrameHashJoinPairIndices,
        DataFrameJoinIndices, DataFrameJoinOptions, DataFrameJoinPairIndices, DataFrameMixedJoinOptions,
        DataFramePredicateJoinOptions, ErrorCode, ErrorKind, Exec, JoinType, Runtime, Series,
        ops::{ast::AstExpression, join::NullEquality},
        testing::prelude::*,
    };

    use crate::common::{
        column_from_strs_nullable, dependency_wait_call_count_for_stream, explicit_i32_series, int64_array,
        int64_list_array, last_exec_mr_value, last_exec_stream_value, lock_event_observation, nested_dataframe,
        reset_dependency_observation, reset_exec_observation, run_named_cases, sample_runtime, struct_array,
    };

    const JOIN_NO_MATCH: i32 = i32::MIN;

    fn sorted_hash_join_index_pairs(indices: &DataFrameHashJoinPairIndices) -> Vec<(i32, i32)> {
        let probe = indices.probe().indices().to_vec_strict::<i32>().unwrap();
        let build = indices.build().indices().to_vec_strict::<i32>().unwrap();

        let mut pairs: Vec<_> = probe.into_iter().zip(build).collect();
        pairs.sort();
        pairs
    }

    fn id_frame(ids: Series) -> DataFrame {
        DataFrame::from_series([("id", ids)]).unwrap()
    }

    fn hash_join_options() -> DataFrameHashJoinOptions {
        DataFrameHashJoinOptions::new().null_equality(NullEquality::Equal)
    }

    fn assert_hash_join_indices_use_handles(indices: &DataFrameHashJoinPairIndices, stream: usize, mr: usize) {
        for side in [indices.probe().indices(), indices.build().indices()] {
            assert_eq!(side.as_column().testing_execution_domain_stream_value(), stream);
            assert_eq!(side.as_column().testing_execution_domain_mr_value(), mr);
        }
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

    fn left_less_than_right() -> AstExpression {
        AstExpression::less(AstExpression::left_column(0).unwrap(), AstExpression::right_column(0).unwrap()).unwrap()
    }

    fn column_frame(name: &str, values: &[i32]) -> DataFrame {
        DataFrame::from_series([(name, sample_runtime().series(values).unwrap())]).unwrap()
    }

    fn keyed_frame(value_name: &str, ids: &[i32], values: &[i32]) -> DataFrame {
        let rt = sample_runtime();
        DataFrame::from_series([("id", rt.series(ids).unwrap()), (value_name, rt.series(values).unwrap())]).unwrap()
    }

    fn join_options() -> DataFrameJoinOptions {
        DataFrameJoinOptions::new().on(["id"])
    }

    fn conditional_options() -> DataFramePredicateJoinOptions {
        DataFramePredicateJoinOptions::new().left_columns(["left_val"]).right_columns(["right_val"])
    }

    fn mixed_options() -> DataFrameMixedJoinOptions {
        DataFrameMixedJoinOptions::new()
            .on(["id"])
            .left_predicate_columns(["left_val"])
            .right_predicate_columns(["right_val"])
    }

    fn assert_pair_indices_use_domain(indices: &DataFrameJoinPairIndices, domain_id: u64, mr: usize) {
        for side in [indices.left(), indices.right()] {
            let column = side.indices().as_column();
            assert_eq!(column.testing_execution_domain_id(), domain_id);
            assert_eq!(column.testing_execution_domain_mr_value(), mr);
        }
    }

    fn assert_single_indices_use_domain(indices: &DataFrameJoinIndices, domain_id: u64, mr: usize) {
        let column = indices.indices().as_column();
        assert_eq!(column.testing_execution_domain_id(), domain_id);
        assert_eq!(column.testing_execution_domain_mr_value(), mr);
    }

    fn assert_pair_indices_use_handles(indices: &DataFrameJoinPairIndices, stream: usize, mr: usize) {
        for side in [indices.left(), indices.right()] {
            let column = side.indices().as_column();
            assert_eq!(column.testing_execution_domain_stream_value(), stream);
            assert_eq!(column.testing_execution_domain_mr_value(), mr);
        }
    }

    fn assert_single_indices_use_handles(indices: &DataFrameJoinIndices, stream: usize, mr: usize) {
        let column = indices.indices().as_column();
        assert_eq!(column.testing_execution_domain_stream_value(), stream);
        assert_eq!(column.testing_execution_domain_mr_value(), mr);
    }

    fn assert_hash_join_pre_build_invalid_argument(build: &DataFrame, key: &str, options: DataFrameHashJoinOptions) {
        let err = build
            .on(Exec::RETAINED)
            .hash_join_pre_build_reservation_size_bytes([key], options)
            .expect_err("unsupported hash-join build keys should be rejected before reservation succeeds");

        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert_eq!(err.code(), ErrorCode::ForeignInvalidArgument);
        assert_eq!(err.operation(), "dataframe::hash_join_pre_build_reservation_size_bytes");
        assert_eq!(err.detail_value("reservation_contract"), Some("hash_join_pre_build_reservation_size_bytes"));
    }

    fn assert_hash_join_pre_build_matches_built_size(
        build: &DataFrame,
        keys: &[&str],
        options: DataFrameHashJoinOptions,
    ) {
        let reservation_size =
            build.on(Exec::RETAINED).hash_join_pre_build_reservation_size_bytes(keys.iter().copied(), options).unwrap();
        let hash_join = build.on(Exec::RETAINED).build_hash_join(keys.iter().copied(), options).unwrap();

        assert_eq!(reservation_size, hash_join.device_allocated_size_bytes().unwrap());
    }

    fn one_shot_join_indices_support_mixed_same_device_domains_and_use_left_provenance() {
        let left = column_frame("id", &[1, 2, 2, 3]);
        let right = column_frame("id", &[2, 2, 4]);
        let expected_stream = left.testing_execution_domain_stream_value();
        let expected_mr = left.testing_execution_domain_mr_value();

        reset_exec_observation();
        let indices = left.on(Exec::RETAINED).inner_join_indices(&right, join_options()).unwrap();

        assert_pair_indices_use_domain(&indices, left.testing_execution_domain_id(), expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(sorted_dataframe_join_index_pairs(&indices), vec![(1, 0), (1, 1), (2, 0), (2, 1)]);
    }

    fn join_indices_in_uses_borrowed_stream_and_memory_resource() {
        let left = column_frame("id", &[1, 2, 3]);
        let right = column_frame("id", &[2, 3, 4]);
        let override_runtime = sample_runtime();
        let ctx = override_runtime.exec();
        let expected_stream = override_runtime.stream().testing_raw_handle_value();
        let expected_mr = override_runtime.memory_resource().testing_raw_handle_value();

        reset_exec_observation();
        let inner = left.on(ctx).inner_join_indices(&right, join_options()).unwrap();
        assert_pair_indices_use_handles(&inner, expected_stream, expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(sorted_dataframe_join_index_pairs(&inner), vec![(1, 0), (2, 1)]);

        reset_exec_observation();
        let left_join = left.on(ctx).left_join_indices(&right, join_options()).unwrap();
        assert_pair_indices_use_handles(&left_join, expected_stream, expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(sorted_dataframe_join_index_pairs(&left_join), vec![(0, JOIN_NO_MATCH), (1, 0), (2, 1)]);

        reset_exec_observation();
        let full = left.on(ctx).full_join_indices(&right, join_options()).unwrap();
        assert_pair_indices_use_handles(&full, expected_stream, expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(
            sorted_dataframe_join_index_pairs(&full),
            vec![(JOIN_NO_MATCH, 2), (0, JOIN_NO_MATCH), (1, 0), (2, 1)]
        );
    }

    fn one_shot_left_semi_join_indices_support_mixed_same_device_domains_and_use_left_provenance() {
        let left = column_frame("id", &[1, 2, 2, 3]);
        let right = column_frame("id", &[2, 2, 4]);
        let expected_stream = left.testing_execution_domain_stream_value();
        let expected_mr = left.testing_execution_domain_mr_value();

        reset_exec_observation();
        let indices = left.on(Exec::RETAINED).left_semi_join_indices(&right, join_options()).unwrap();

        assert_single_indices_use_domain(&indices, left.testing_execution_domain_id(), expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(sorted_dataframe_join_indices(&indices), vec![1, 2]);
    }

    fn one_shot_left_anti_join_indices_support_mixed_same_device_domains_and_use_left_provenance() {
        let left = column_frame("id", &[1, 2, 3, 4]);
        let right = column_frame("id", &[2, 4]);
        let expected_stream = left.testing_execution_domain_stream_value();
        let expected_mr = left.testing_execution_domain_mr_value();

        reset_exec_observation();
        let indices = left.on(Exec::RETAINED).left_anti_join_indices(&right, join_options()).unwrap();

        assert_single_indices_use_domain(&indices, left.testing_execution_domain_id(), expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(sorted_dataframe_join_indices(&indices), vec![0, 2]);
    }

    fn single_side_join_indices_in_uses_borrowed_stream_and_memory_resource() {
        let left = column_frame("id", &[1, 2, 3, 4]);
        let right = column_frame("id", &[2, 4]);
        let override_runtime = sample_runtime();
        let ctx = override_runtime.exec();
        let expected_stream = override_runtime.stream().testing_raw_handle_value();
        let expected_mr = override_runtime.memory_resource().testing_raw_handle_value();

        reset_exec_observation();
        let semi = left.on(ctx).left_semi_join_indices(&right, join_options()).unwrap();
        assert_single_indices_use_handles(&semi, expected_stream, expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(sorted_dataframe_join_indices(&semi), vec![1, 3]);

        reset_exec_observation();
        let anti = left.on(ctx).left_anti_join_indices(&right, join_options()).unwrap();
        assert_single_indices_use_handles(&anti, expected_stream, expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(sorted_dataframe_join_indices(&anti), vec![0, 2]);
    }

    fn cross_join_supports_mixed_same_device_domains_and_uses_left_provenance() {
        let left = column_frame("left_val", &[1, 2]);
        let right = column_frame("right_val", &[10, 20]);
        let expected_stream = left.testing_execution_domain_stream_value();
        let expected_mr = left.testing_execution_domain_mr_value();

        reset_exec_observation();
        let result = left
            .on(Exec::RETAINED)
            .cross_join_projected(
                &right,
                DataFrameCrossJoinOptions::new()
                    .left_projection([("left_val", "left_val")])
                    .right_projection([("right_val", "right_val")]),
            )
            .unwrap();

        assert_eq!(result.testing_execution_domain_id(), left.testing_execution_domain_id());
        assert_eq!(result.testing_execution_domain_mr_value(), expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(result.series("left_val").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 1, 2, 2]);
        assert_eq!(result.series("right_val").unwrap().to_vec_strict::<i32>().unwrap(), vec![10, 20, 10, 20]);
    }

    fn hash_join_indices_support_mixed_same_device_domains_and_use_probe_provenance() {
        let build = id_frame(explicit_i32_series("id", &[2, 2, 4], 960));
        let probe = id_frame(explicit_i32_series("id", &[1, 2, 4], 961));
        let expected_stream = probe.testing_execution_domain_stream_value();
        let expected_mr = probe.testing_execution_domain_mr_value();

        let hash_join = build.on(Exec::RETAINED).build_hash_join(["id"], hash_join_options()).unwrap();

        reset_exec_observation();
        let indices = hash_join.left_probe(&probe, ["id"], Exec::RETAINED).unwrap();

        for side in [indices.probe().indices(), indices.build().indices()] {
            assert_eq!(side.as_column().testing_execution_domain_id(), probe.testing_execution_domain_id());
        }
        assert_hash_join_indices_use_handles(&indices, expected_stream, expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(sorted_hash_join_index_pairs(&indices), vec![(0, JOIN_NO_MATCH), (1, 0), (1, 1), (2, 2)]);
    }

    fn hash_join_build_uses_borrowed_stream_and_memory_resource() {
        let build = id_frame(explicit_i32_series("id", &[2, 2, 4], 1019));
        let retained_build_stream = build.testing_execution_domain_stream_value();
        let override_runtime = Runtime::new().unwrap();
        let ctx = override_runtime.exec();
        let expected_stream = override_runtime.stream().testing_raw_handle_value();
        let expected_mr = override_runtime.memory_resource().testing_raw_handle_value();

        reset_exec_observation();
        let hash_join = build.on(ctx).build_hash_join(["id"], hash_join_options()).unwrap();

        assert_eq!(hash_join.build().testing_execution_domain_stream_value(), retained_build_stream);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
    }

    fn hash_join_pre_build_reservation_contract_matches_retained_build_artifact_bytes() {
        let runtime = Runtime::new().unwrap();
        let build = DataFrame::from_series([
            ("k0", runtime.series(&[2i32, 2, 4, 4, 8, 8, 16, 32]).unwrap()),
            ("k1", runtime.series(&[10i32, 11, 10, 12, 10, 13, 14, 15]).unwrap()),
        ])
        .unwrap();

        for load_factor in [0.5, 1.0] {
            assert_hash_join_pre_build_matches_built_size(
                &build,
                &["k0", "k1"],
                hash_join_options().load_factor(load_factor),
            );
        }
    }

    fn hash_join_pre_build_reservation_accepts_empty_rows_with_non_empty_schema() {
        let runtime = Runtime::new().unwrap();
        let build = DataFrame::from_series([
            ("k0", runtime.series::<i32>(&[]).unwrap()),
            ("k1", runtime.series::<i32>(&[]).unwrap()),
        ])
        .unwrap();

        assert_hash_join_pre_build_matches_built_size(&build, &["k0", "k1"], hash_join_options());
    }

    fn hash_join_pre_build_reservation_matches_larger_build_table() {
        let runtime = Runtime::new().unwrap();
        let key0 = (0..4096).map(|row| row % 257).collect::<Vec<i32>>();
        let key1 = (0..4096).map(|row| row % 31).collect::<Vec<i32>>();
        let build =
            DataFrame::from_series([("k0", runtime.series(&key0).unwrap()), ("k1", runtime.series(&key1).unwrap())])
                .unwrap();

        assert_hash_join_pre_build_matches_built_size(&build, &["k0", "k1"], hash_join_options());
    }

    fn hash_join_pre_build_reservation_matches_non_nullable_string_build_artifact() {
        let runtime = Runtime::new().unwrap();
        let build = id_frame(runtime.series(&["alpha", "beta", "gamma"]).unwrap());

        assert_hash_join_pre_build_matches_built_size(&build, &["id"], hash_join_options());
    }

    fn hash_join_pre_build_reservation_rejects_invalid_load_factors() {
        let build = id_frame(explicit_i32_series("id", &[2, 2, 4, 8], 12023));

        for load_factor in [-0.1, 0.0, 1.1] {
            let expected_load_factor = load_factor.to_string();
            let err = build
                .on(Exec::RETAINED)
                .hash_join_pre_build_reservation_size_bytes(["id"], hash_join_options().load_factor(load_factor))
                .expect_err("invalid load factor should be rejected");

            assert_eq!(err.kind(), ErrorKind::InvalidInput);
            assert_eq!(err.code(), ErrorCode::ForeignInvalidArgument);
            assert_eq!(err.operation(), "dataframe::hash_join_pre_build_reservation_size_bytes");
            assert_eq!(err.detail_value("reservation_contract"), Some("hash_join_pre_build_reservation_size_bytes"));
            assert_eq!(err.detail_value("load_factor"), Some(expected_load_factor.as_str()));
            assert_eq!(err.detail_value("unsupported_input"), None);
        }
    }

    fn hash_join_pre_build_reservation_rejects_nested_and_struct_build_keys() {
        let runtime = Runtime::builder().build().unwrap();
        let frame = nested_dataframe(
            &runtime,
            vec![
                ("items", int64_list_array(&[vec![1, 2], vec![3]])),
                ("point", struct_array(vec![("x", int64_array(&[1, 3])), ("y", int64_array(&[2, 4]))])),
                ("id", int64_array(&[1, 2])),
            ],
        );

        for key in ["items", "point"] {
            assert_hash_join_pre_build_invalid_argument(&frame, key, hash_join_options());
        }
    }

    fn hash_join_pre_build_reservation_matches_nullable_string_build_key_with_null_payload() {
        let build = id_frame(
            Series::from_column(
                column_from_strs_nullable(&["alpha", "null-payload", "omega"], &[true, false, true]).unwrap(),
            )
            .unwrap(),
        );

        assert_hash_join_pre_build_matches_built_size(&build, &["id"], hash_join_options());
    }

    fn hash_join_build_accepts_nullable_string_build_key_with_null_payload() {
        let build = id_frame(
            Series::from_column(
                column_from_strs_nullable(&["alpha", "null-payload", "omega"], &[true, false, true]).unwrap(),
            )
            .unwrap(),
        );

        build.on(Exec::RETAINED).build_hash_join(["id"], hash_join_options()).unwrap();
    }

    fn hash_join_probe_uses_borrowed_stream_and_memory_resource() {
        let build = id_frame(explicit_i32_series("id", &[2, 2, 4], 1020));
        let probe = id_frame(explicit_i32_series("id", &[1, 2, 4], 1021));
        let hash_join = build.on(Exec::RETAINED).build_hash_join(["id"], hash_join_options()).unwrap();
        let override_runtime = Runtime::new().unwrap();
        let ctx = override_runtime.exec();
        let expected_stream = override_runtime.stream().testing_raw_handle_value();
        let expected_mr = override_runtime.memory_resource().testing_raw_handle_value();

        reset_exec_observation();
        let inner = hash_join.inner_probe(&probe, ["id"], ctx).unwrap();
        assert_hash_join_indices_use_handles(&inner, expected_stream, expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(sorted_hash_join_index_pairs(&inner), vec![(1, 0), (1, 1), (2, 2)]);

        reset_exec_observation();
        let left = hash_join.left_probe(&probe, ["id"], ctx).unwrap();
        assert_hash_join_indices_use_handles(&left, expected_stream, expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(sorted_hash_join_index_pairs(&left), vec![(0, JOIN_NO_MATCH), (1, 0), (1, 1), (2, 2)]);

        reset_exec_observation();
        let full = hash_join.full_probe(&probe, ["id"], ctx).unwrap();
        assert_hash_join_indices_use_handles(&full, expected_stream, expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(sorted_hash_join_index_pairs(&full), vec![(0, JOIN_NO_MATCH), (1, 0), (1, 1), (2, 2)]);
    }

    fn conditional_inner_join_indices_support_mixed_same_device_domains_and_use_left_provenance() {
        let left = column_frame("left_val", &[1, 3, 5]);
        let right = column_frame("right_val", &[2, 4]);
        let expected_stream = left.testing_execution_domain_stream_value();
        let expected_mr = left.testing_execution_domain_mr_value();

        reset_exec_observation();
        let indices = left
            .on(Exec::RETAINED)
            .conditional_inner_join_indices(&right, &left_less_than_right(), conditional_options())
            .unwrap();

        assert_pair_indices_use_domain(&indices, left.testing_execution_domain_id(), expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(sorted_dataframe_join_index_pairs(&indices), vec![(0, 0), (0, 1), (1, 1)]);
    }

    fn conditional_left_join_indices_support_mixed_same_device_domains_and_use_left_provenance() {
        let left = column_frame("left_val", &[1, 3, 5]);
        let right = column_frame("right_val", &[2, 4]);
        let expected_stream = left.testing_execution_domain_stream_value();
        let expected_mr = left.testing_execution_domain_mr_value();

        reset_exec_observation();
        let indices = left
            .on(Exec::RETAINED)
            .conditional_left_join_indices(&right, &left_less_than_right(), conditional_options())
            .unwrap();

        assert_pair_indices_use_domain(&indices, left.testing_execution_domain_id(), expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(sorted_dataframe_join_index_pairs(&indices), vec![(0, 0), (0, 1), (1, 1), (2, JOIN_NO_MATCH)]);
    }

    fn conditional_full_join_indices_support_mixed_same_device_domains_and_use_left_provenance() {
        let left = column_frame("left_val", &[1, 3, 5]);
        let right = column_frame("right_val", &[2, 4, 6]);
        let expected_stream = left.testing_execution_domain_stream_value();
        let expected_mr = left.testing_execution_domain_mr_value();

        reset_exec_observation();
        let indices = left
            .on(Exec::RETAINED)
            .conditional_full_join_indices(&right, &left_less_than_right(), conditional_options())
            .unwrap();

        assert_pair_indices_use_domain(&indices, left.testing_execution_domain_id(), expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(sorted_dataframe_join_index_pairs(&indices), vec![(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]);
    }

    fn conditional_left_semi_join_indices_support_mixed_same_device_domains_and_use_left_provenance() {
        let left = column_frame("left_val", &[1, 3, 5]);
        let right = column_frame("right_val", &[2, 4]);
        let expected_stream = left.testing_execution_domain_stream_value();
        let expected_mr = left.testing_execution_domain_mr_value();

        reset_exec_observation();
        let indices = left
            .on(Exec::RETAINED)
            .conditional_left_semi_join_indices(&right, &left_less_than_right(), conditional_options())
            .unwrap();

        assert_single_indices_use_domain(&indices, left.testing_execution_domain_id(), expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(sorted_dataframe_join_indices(&indices), vec![0, 1]);
    }

    fn conditional_left_anti_join_indices_support_mixed_same_device_domains_and_use_left_provenance() {
        let left = column_frame("left_val", &[1, 3, 5]);
        let right = column_frame("right_val", &[2, 4]);
        let expected_stream = left.testing_execution_domain_stream_value();
        let expected_mr = left.testing_execution_domain_mr_value();

        reset_exec_observation();
        let indices = left
            .on(Exec::RETAINED)
            .conditional_left_anti_join_indices(&right, &left_less_than_right(), conditional_options())
            .unwrap();

        assert_single_indices_use_domain(&indices, left.testing_execution_domain_id(), expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(sorted_dataframe_join_indices(&indices), vec![2]);
    }

    fn conditional_join_indices_in_uses_borrowed_stream_and_memory_resource() {
        let left = column_frame("left_val", &[1, 3, 5]);
        let right = column_frame("right_val", &[2, 4]);
        let override_runtime = sample_runtime();
        let ctx = override_runtime.exec();
        let expected_stream = override_runtime.stream().testing_raw_handle_value();
        let expected_mr = override_runtime.memory_resource().testing_raw_handle_value();

        reset_exec_observation();
        let indices = left
            .on(ctx)
            .conditional_left_anti_join_indices(&right, &left_less_than_right(), conditional_options())
            .unwrap();

        assert_single_indices_use_handles(&indices, expected_stream, expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(sorted_dataframe_join_indices(&indices), vec![2]);
    }

    fn conditional_join_indices_orders_literal_scalar_domain_before_left_exec() {
        let _event_guard = lock_event_observation();
        let left = column_frame("left_val", &[1, 5]);
        let right = column_frame("right_val", &[10, 20]);
        let literal_runtime = sample_runtime();
        let literal = AstExpression::literal(literal_runtime.scalar(4i32).unwrap()).unwrap();
        let predicate = AstExpression::less(AstExpression::left_column(0).unwrap(), literal).unwrap();
        let expected_stream = left.testing_execution_domain_stream_value();
        let expected_mr = left.testing_execution_domain_mr_value();

        reset_exec_observation();
        reset_dependency_observation();
        let indices =
            left.on(Exec::RETAINED).conditional_inner_join_indices(&right, &predicate, conditional_options()).unwrap();

        assert_pair_indices_use_domain(&indices, left.testing_execution_domain_id(), expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(dependency_wait_call_count_for_stream(expected_stream), 2);
        assert_eq!(sorted_dataframe_join_index_pairs(&indices), vec![(0, 0), (0, 1)]);
    }

    fn mixed_inner_join_indices_support_mixed_same_device_domains_and_use_primary_left_provenance() {
        let left = keyed_frame("left_val", &[1, 1, 2], &[5, 7, 3]);
        let right = keyed_frame("right_val", &[1, 2], &[6, 4]);
        let expected_stream = left.testing_execution_domain_stream_value();
        let expected_mr = left.testing_execution_domain_mr_value();

        reset_exec_observation();
        let indices =
            left.on(Exec::RETAINED).mixed_inner_join_indices(&right, &left_less_than_right(), mixed_options()).unwrap();

        assert_pair_indices_use_domain(&indices, left.testing_execution_domain_id(), expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(sorted_dataframe_join_index_pairs(&indices), vec![(0, 0), (2, 1)]);
    }

    fn mixed_inner_join_indices_in_uses_borrowed_stream_and_memory_resource() {
        let left = keyed_frame("left_val", &[1, 1, 2], &[5, 7, 3]);
        let right = keyed_frame("right_val", &[1, 2], &[6, 4]);
        let override_runtime = sample_runtime();
        let ctx = override_runtime.exec();
        let expected_stream = override_runtime.stream().testing_raw_handle_value();
        let expected_mr = override_runtime.memory_resource().testing_raw_handle_value();

        reset_exec_observation();
        let indices = left.on(ctx).mixed_inner_join_indices(&right, &left_less_than_right(), mixed_options()).unwrap();

        assert_pair_indices_use_handles(&indices, expected_stream, expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(sorted_dataframe_join_index_pairs(&indices), vec![(0, 0), (2, 1)]);
    }

    fn mixed_full_semi_anti_join_indices_in_use_borrowed_stream_and_memory_resource() {
        let left = keyed_frame("left_val", &[1, 1, 2, 3], &[5, 7, 3, 1]);
        let right = keyed_frame("right_val", &[1, 2, 4], &[6, 4, 9]);
        let predicate = left_less_than_right();
        let override_runtime = sample_runtime();
        let ctx = override_runtime.exec();
        let expected_stream = override_runtime.stream().testing_raw_handle_value();
        let expected_mr = override_runtime.memory_resource().testing_raw_handle_value();

        reset_exec_observation();
        let full = left.on(ctx).mixed_full_join_indices(&right, &predicate, mixed_options()).unwrap();
        assert_pair_indices_use_handles(&full, expected_stream, expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(
            sorted_dataframe_join_index_pairs(&full),
            vec![(JOIN_NO_MATCH, 2), (0, 0), (1, JOIN_NO_MATCH), (2, 1), (3, JOIN_NO_MATCH),]
        );

        reset_exec_observation();
        let semi = left.on(ctx).mixed_left_semi_join_indices(&right, &predicate, mixed_options()).unwrap();
        assert_single_indices_use_handles(&semi, expected_stream, expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(sorted_dataframe_join_indices(&semi), vec![0, 2]);

        reset_exec_observation();
        let anti = left.on(ctx).mixed_left_anti_join_indices(&right, &predicate, mixed_options()).unwrap();
        assert_single_indices_use_handles(&anti, expected_stream, expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(sorted_dataframe_join_indices(&anti), vec![1, 3]);
    }

    fn filter_join_indices_support_mixed_same_device_domains_and_use_left_index_provenance() {
        let left = column_frame("left_val", &[1, 3, 5]);
        let right = column_frame("right_val", &[2, 4]);
        let left_indices = DataFrameJoinIndices::from_series(explicit_i32_series("left_indices", &[0, 1, 2], 1011));
        let right_indices = DataFrameJoinIndices::from_series(explicit_i32_series("right_indices", &[0, 0, 1], 1012));
        let index_column = left_indices.indices().as_column();
        let expected_domain_id = index_column.testing_execution_domain_id();
        let expected_stream = index_column.testing_execution_domain_stream_value();
        let expected_mr = index_column.testing_execution_domain_mr_value();

        reset_exec_observation();
        let indices = left
            .on(Exec::RETAINED)
            .filter_join_indices(
                &right,
                &left_indices,
                &right_indices,
                &left_less_than_right(),
                conditional_options(),
                JoinType::Inner,
            )
            .unwrap();

        assert_pair_indices_use_domain(&indices, expected_domain_id, expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(sorted_dataframe_join_index_pairs(&indices), vec![(0, 0)]);
    }

    fn filter_join_indices_in_uses_borrowed_stream_and_memory_resource() {
        let left = column_frame("left_val", &[1, 3, 5]);
        let right = column_frame("right_val", &[2, 4]);
        let left_indices = DataFrameJoinIndices::from_series(explicit_i32_series("left_indices", &[0, 1, 2], 1015));
        let right_indices = DataFrameJoinIndices::from_series(explicit_i32_series("right_indices", &[0, 0, 1], 1016));
        let override_runtime = sample_runtime();
        let ctx = override_runtime.exec();
        let expected_stream = override_runtime.stream().testing_raw_handle_value();
        let expected_mr = override_runtime.memory_resource().testing_raw_handle_value();

        reset_exec_observation();
        let indices = left
            .on(ctx)
            .filter_join_indices(
                &right,
                &left_indices,
                &right_indices,
                &left_less_than_right(),
                conditional_options(),
                JoinType::Inner,
            )
            .unwrap();

        assert_pair_indices_use_handles(&indices, expected_stream, expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(sorted_dataframe_join_index_pairs(&indices), vec![(0, 0)]);
    }

    fn mixed_left_join_indices_support_mixed_same_device_domains_and_use_primary_left_provenance() {
        let left = keyed_frame("left_val", &[1, 1, 2, 3], &[5, 7, 3, 1]);
        let right = keyed_frame("right_val", &[1, 2], &[6, 4]);
        let expected_stream = left.testing_execution_domain_stream_value();
        let expected_mr = left.testing_execution_domain_mr_value();

        reset_exec_observation();
        let indices =
            left.on(Exec::RETAINED).mixed_left_join_indices(&right, &left_less_than_right(), mixed_options()).unwrap();

        assert_pair_indices_use_domain(&indices, left.testing_execution_domain_id(), expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(
            sorted_dataframe_join_index_pairs(&indices),
            vec![(0, 0), (1, JOIN_NO_MATCH), (2, 1), (3, JOIN_NO_MATCH)]
        );
    }

    fn mixed_full_join_indices_support_mixed_same_device_domains_and_use_primary_left_provenance() {
        let left = keyed_frame("left_val", &[1, 1, 2, 3], &[5, 7, 3, 1]);
        let right = keyed_frame("right_val", &[1, 2, 4], &[6, 4, 9]);
        let expected_stream = left.testing_execution_domain_stream_value();
        let expected_mr = left.testing_execution_domain_mr_value();

        reset_exec_observation();
        let indices =
            left.on(Exec::RETAINED).mixed_full_join_indices(&right, &left_less_than_right(), mixed_options()).unwrap();

        assert_pair_indices_use_domain(&indices, left.testing_execution_domain_id(), expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(
            sorted_dataframe_join_index_pairs(&indices),
            vec![(JOIN_NO_MATCH, 2), (0, 0), (1, JOIN_NO_MATCH), (2, 1), (3, JOIN_NO_MATCH),]
        );
    }

    fn mixed_left_semi_join_indices_support_mixed_same_device_domains_and_use_primary_left_provenance() {
        let left = keyed_frame("left_val", &[1, 1, 2], &[5, 7, 3]);
        let right = keyed_frame("right_val", &[1, 2], &[6, 4]);
        let expected_stream = left.testing_execution_domain_stream_value();
        let expected_mr = left.testing_execution_domain_mr_value();

        reset_exec_observation();
        let indices = left
            .on(Exec::RETAINED)
            .mixed_left_semi_join_indices(&right, &left_less_than_right(), mixed_options())
            .unwrap();

        assert_single_indices_use_domain(&indices, left.testing_execution_domain_id(), expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(sorted_dataframe_join_indices(&indices), vec![0, 2]);
    }

    fn mixed_left_anti_join_indices_support_mixed_same_device_domains_and_use_primary_left_provenance() {
        let left = keyed_frame("left_val", &[1, 1, 2, 3], &[5, 7, 3, 1]);
        let right = keyed_frame("right_val", &[1, 2], &[6, 4]);
        let expected_stream = left.testing_execution_domain_stream_value();
        let expected_mr = left.testing_execution_domain_mr_value();

        reset_exec_observation();
        let indices = left
            .on(Exec::RETAINED)
            .mixed_left_anti_join_indices(&right, &left_less_than_right(), mixed_options())
            .unwrap();

        assert_single_indices_use_domain(&indices, left.testing_execution_domain_id(), expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(sorted_dataframe_join_indices(&indices), vec![1, 3]);
    }

    #[test]
    fn join_hash_join_execution_tests_cases() {
        run_named_cases(&[
            (
                "one_shot_join_indices_support_mixed_same_device_domains_and_use_left_provenance",
                one_shot_join_indices_support_mixed_same_device_domains_and_use_left_provenance,
            ),
            (
                "join_indices_in_uses_borrowed_stream_and_memory_resource",
                join_indices_in_uses_borrowed_stream_and_memory_resource,
            ),
            (
                "one_shot_left_semi_join_indices_support_mixed_same_device_domains_and_use_left_provenance",
                one_shot_left_semi_join_indices_support_mixed_same_device_domains_and_use_left_provenance,
            ),
            (
                "one_shot_left_anti_join_indices_support_mixed_same_device_domains_and_use_left_provenance",
                one_shot_left_anti_join_indices_support_mixed_same_device_domains_and_use_left_provenance,
            ),
            (
                "single_side_join_indices_in_uses_borrowed_stream_and_memory_resource",
                single_side_join_indices_in_uses_borrowed_stream_and_memory_resource,
            ),
            (
                "cross_join_supports_mixed_same_device_domains_and_uses_left_provenance",
                cross_join_supports_mixed_same_device_domains_and_uses_left_provenance,
            ),
            (
                "hash_join_indices_support_mixed_same_device_domains_and_use_probe_provenance",
                hash_join_indices_support_mixed_same_device_domains_and_use_probe_provenance,
            ),
            (
                "hash_join_build_uses_borrowed_stream_and_memory_resource",
                hash_join_build_uses_borrowed_stream_and_memory_resource,
            ),
            (
                "hash_join_pre_build_reservation_contract_matches_retained_build_artifact_bytes",
                hash_join_pre_build_reservation_contract_matches_retained_build_artifact_bytes,
            ),
            (
                "hash_join_pre_build_reservation_accepts_empty_rows_with_non_empty_schema",
                hash_join_pre_build_reservation_accepts_empty_rows_with_non_empty_schema,
            ),
            (
                "hash_join_pre_build_reservation_matches_larger_build_table",
                hash_join_pre_build_reservation_matches_larger_build_table,
            ),
            (
                "hash_join_pre_build_reservation_matches_non_nullable_string_build_artifact",
                hash_join_pre_build_reservation_matches_non_nullable_string_build_artifact,
            ),
            (
                "hash_join_pre_build_reservation_rejects_invalid_load_factors",
                hash_join_pre_build_reservation_rejects_invalid_load_factors,
            ),
            (
                "hash_join_pre_build_reservation_rejects_nested_and_struct_build_keys",
                hash_join_pre_build_reservation_rejects_nested_and_struct_build_keys,
            ),
            (
                "hash_join_pre_build_reservation_matches_nullable_string_build_key_with_null_payload",
                hash_join_pre_build_reservation_matches_nullable_string_build_key_with_null_payload,
            ),
            (
                "hash_join_build_accepts_nullable_string_build_key_with_null_payload",
                hash_join_build_accepts_nullable_string_build_key_with_null_payload,
            ),
            (
                "hash_join_probe_uses_borrowed_stream_and_memory_resource",
                hash_join_probe_uses_borrowed_stream_and_memory_resource,
            ),
            (
                "conditional_inner_join_indices_support_mixed_same_device_domains_and_use_left_provenance",
                conditional_inner_join_indices_support_mixed_same_device_domains_and_use_left_provenance,
            ),
            (
                "conditional_left_join_indices_support_mixed_same_device_domains_and_use_left_provenance",
                conditional_left_join_indices_support_mixed_same_device_domains_and_use_left_provenance,
            ),
            (
                "conditional_full_join_indices_support_mixed_same_device_domains_and_use_left_provenance",
                conditional_full_join_indices_support_mixed_same_device_domains_and_use_left_provenance,
            ),
            (
                "conditional_left_semi_join_indices_support_mixed_same_device_domains_and_use_left_provenance",
                conditional_left_semi_join_indices_support_mixed_same_device_domains_and_use_left_provenance,
            ),
            (
                "conditional_left_anti_join_indices_support_mixed_same_device_domains_and_use_left_provenance",
                conditional_left_anti_join_indices_support_mixed_same_device_domains_and_use_left_provenance,
            ),
            (
                "conditional_join_indices_in_uses_borrowed_stream_and_memory_resource",
                conditional_join_indices_in_uses_borrowed_stream_and_memory_resource,
            ),
            (
                "conditional_join_indices_orders_literal_scalar_domain_before_left_exec",
                conditional_join_indices_orders_literal_scalar_domain_before_left_exec,
            ),
            (
                "mixed_inner_join_indices_support_mixed_same_device_domains_and_use_primary_left_provenance",
                mixed_inner_join_indices_support_mixed_same_device_domains_and_use_primary_left_provenance,
            ),
            (
                "mixed_inner_join_indices_in_uses_borrowed_stream_and_memory_resource",
                mixed_inner_join_indices_in_uses_borrowed_stream_and_memory_resource,
            ),
            (
                "mixed_full_semi_anti_join_indices_in_use_borrowed_stream_and_memory_resource",
                mixed_full_semi_anti_join_indices_in_use_borrowed_stream_and_memory_resource,
            ),
            (
                "filter_join_indices_support_mixed_same_device_domains_and_use_left_index_provenance",
                filter_join_indices_support_mixed_same_device_domains_and_use_left_index_provenance,
            ),
            (
                "filter_join_indices_in_uses_borrowed_stream_and_memory_resource",
                filter_join_indices_in_uses_borrowed_stream_and_memory_resource,
            ),
            (
                "mixed_left_join_indices_support_mixed_same_device_domains_and_use_primary_left_provenance",
                mixed_left_join_indices_support_mixed_same_device_domains_and_use_primary_left_provenance,
            ),
            (
                "mixed_full_join_indices_support_mixed_same_device_domains_and_use_primary_left_provenance",
                mixed_full_join_indices_support_mixed_same_device_domains_and_use_primary_left_provenance,
            ),
            (
                "mixed_left_semi_join_indices_support_mixed_same_device_domains_and_use_primary_left_provenance",
                mixed_left_semi_join_indices_support_mixed_same_device_domains_and_use_primary_left_provenance,
            ),
            (
                "mixed_left_anti_join_indices_support_mixed_same_device_domains_and_use_primary_left_provenance",
                mixed_left_anti_join_indices_support_mixed_same_device_domains_and_use_primary_left_provenance,
            ),
        ]);
    }
}
#[path = "../ops/strings/strings_tests.rs"]
mod strings_tests;

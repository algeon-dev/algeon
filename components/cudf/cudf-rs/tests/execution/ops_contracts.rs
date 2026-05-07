#[path = "../ops/datetime/datetime_tests.rs"]
mod datetime_tests;
mod join_hash_join_execution_tests {
    use cudf::{
        DataFrame, ErrorCode, ErrorKind, ExecutionOverrides, Runtime,
        core::Table,
        frame::dataframe::JsonReadOptions,
        ops::{
            ast::AstExpression,
            join::{self, HashJoin, HashJoinOptions, JoinType, NullEquality},
        },
        testing::prelude::*,
    };
    use tempfile::TempDir;

    use crate::common::{
        column_from_strs_nullable, dependency_wait_call_count_for_stream, explicit_i32_column, last_exec_mr_value,
        last_exec_stream_value, lock_event_observation, reset_dependency_observation, reset_exec_observation,
    };

    const JOIN_NO_MATCH: i32 = i32::MIN;

    fn sorted_join_index_pairs(indices: &join::JoinIndices) -> Vec<(i32, i32)> {
        let left = indices.left.testing_view().unwrap().to_vec::<i32>().unwrap();
        let right = indices.right.testing_view().unwrap().to_vec::<i32>().unwrap();

        let mut pairs: Vec<_> = left.into_iter().zip(right).collect();
        pairs.sort();
        pairs
    }

    fn sorted_single_side_indices(indices: &join::SingleSideJoinIndices) -> Vec<i32> {
        let mut rows = indices.indices.testing_view().unwrap().to_vec::<i32>().unwrap();
        rows.sort();
        rows
    }

    fn left_less_than_right() -> AstExpression {
        AstExpression::less(AstExpression::left_column(0).unwrap(), AstExpression::right_column(0).unwrap()).unwrap()
    }

    fn read_json_dataframe(filename: &str, rows: &str) -> DataFrame {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join(filename);
        std::fs::write(&path, rows).unwrap();
        let runtime = Runtime::builder().build().unwrap();
        DataFrame::read_json_with_options(JsonReadOptions::builder().filepath(&path).with_runtime(&runtime).build())
            .unwrap()
    }

    fn assert_hash_join_pre_build_invalid_argument(build: &Table, options: HashJoinOptions) {
        let err = HashJoin::pre_build_reservation_size_bytes(build, options)
            .expect_err("unsupported hash-join build table should be rejected before reservation succeeds");

        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert_eq!(err.code(), ErrorCode::ForeignInvalidArgument);
        assert_eq!(err.operation(), "hash_join::pre_build_reservation_size_bytes_in");
        assert_eq!(err.detail_value("reservation_contract"), Some("hash_join_pre_build_reservation_size_bytes"));
    }

    fn assert_hash_join_pre_build_matches_built_size(build: &Table, options: HashJoinOptions) {
        let reservation_size = HashJoin::pre_build_reservation_size_bytes(build, options).unwrap();
        let hash_join = HashJoin::new(build.deep_copy().unwrap(), options).unwrap();

        assert_eq!(reservation_size, hash_join.device_allocated_size_bytes().unwrap());
    }

    #[test]
    fn one_shot_join_indices_support_mixed_same_device_domains_and_use_left_provenance() {
        let left_keys = Table::from_columns(vec![explicit_i32_column(&[1, 2, 2, 3], 958)]).unwrap();
        let right_keys = Table::from_columns(vec![explicit_i32_column(&[2, 2, 4], 959)]).unwrap();
        let expected_stream = left_keys.testing_execution_domain_stream_value();
        let expected_mr = left_keys.testing_execution_domain_mr_value();

        reset_exec_observation();
        let indices = join::inner_join_indices(
            &left_keys.testing_view().unwrap(),
            &right_keys.testing_view().unwrap(),
            NullEquality::Equal,
        )
        .unwrap();

        assert_eq!(indices.left.testing_execution_domain_id(), left_keys.testing_execution_domain_id());
        assert_eq!(indices.right.testing_execution_domain_id(), left_keys.testing_execution_domain_id());
        assert_eq!(indices.left.testing_execution_domain_mr_value(), expected_mr);
        assert_eq!(indices.right.testing_execution_domain_mr_value(), expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(sorted_join_index_pairs(&indices), vec![(1, 0), (1, 1), (2, 0), (2, 1)]);
    }

    #[test]
    fn join_indices_in_uses_borrowed_stream_and_memory_resource() {
        let left_keys = Table::from_columns(vec![explicit_i32_column(&[1, 2, 3], 1017)]).unwrap();
        let right_keys = Table::from_columns(vec![explicit_i32_column(&[2, 3, 4], 1018)]).unwrap();
        let override_runtime = Runtime::new().unwrap();
        let ctx = ExecutionOverrides::new()
            .with_stream(override_runtime.stream())
            .with_memory_resource(override_runtime.memory_resource());
        let expected_stream = override_runtime.stream().testing_raw_handle_value();
        let expected_mr = override_runtime.memory_resource().testing_raw_handle_value();

        reset_exec_observation();
        let inner = join::inner_join_indices_in(
            &left_keys.testing_view().unwrap(),
            &right_keys.testing_view().unwrap(),
            NullEquality::Equal,
            &ctx,
        )
        .unwrap();
        assert_eq!(inner.left.testing_execution_domain_stream_value(), expected_stream);
        assert_eq!(inner.right.testing_execution_domain_stream_value(), expected_stream);
        assert_eq!(inner.left.testing_execution_domain_mr_value(), expected_mr);
        assert_eq!(inner.right.testing_execution_domain_mr_value(), expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(sorted_join_index_pairs(&inner), vec![(1, 0), (2, 1)]);

        reset_exec_observation();
        let left = join::left_join_indices_in(
            &left_keys.testing_view().unwrap(),
            &right_keys.testing_view().unwrap(),
            NullEquality::Equal,
            &ctx,
        )
        .unwrap();
        assert_eq!(left.left.testing_execution_domain_stream_value(), expected_stream);
        assert_eq!(left.right.testing_execution_domain_stream_value(), expected_stream);
        assert_eq!(left.left.testing_execution_domain_mr_value(), expected_mr);
        assert_eq!(left.right.testing_execution_domain_mr_value(), expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(sorted_join_index_pairs(&left), vec![(0, JOIN_NO_MATCH), (1, 0), (2, 1)]);

        reset_exec_observation();
        let full = join::full_join_indices_in(
            &left_keys.testing_view().unwrap(),
            &right_keys.testing_view().unwrap(),
            NullEquality::Equal,
            &ctx,
        )
        .unwrap();
        assert_eq!(full.left.testing_execution_domain_stream_value(), expected_stream);
        assert_eq!(full.right.testing_execution_domain_stream_value(), expected_stream);
        assert_eq!(full.left.testing_execution_domain_mr_value(), expected_mr);
        assert_eq!(full.right.testing_execution_domain_mr_value(), expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(sorted_join_index_pairs(&full), vec![(JOIN_NO_MATCH, 2), (0, JOIN_NO_MATCH), (1, 0), (2, 1)]);
    }

    #[test]
    fn one_shot_left_semi_join_indices_support_mixed_same_device_domains_and_use_left_provenance() {
        let left_keys = Table::from_columns(vec![explicit_i32_column(&[1, 2, 2, 3], 962)]).unwrap();
        let right_keys = Table::from_columns(vec![explicit_i32_column(&[2, 2, 4], 963)]).unwrap();
        let expected_stream = left_keys.testing_execution_domain_stream_value();
        let expected_mr = left_keys.testing_execution_domain_mr_value();

        reset_exec_observation();
        let indices = join::left_semi_join_indices(
            &left_keys.testing_view().unwrap(),
            &right_keys.testing_view().unwrap(),
            NullEquality::Equal,
        )
        .unwrap();

        assert_eq!(indices.indices.testing_execution_domain_id(), left_keys.testing_execution_domain_id());
        assert_eq!(indices.indices.testing_execution_domain_mr_value(), expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(sorted_single_side_indices(&indices), vec![1, 2]);
    }

    #[test]
    fn one_shot_left_anti_join_indices_support_mixed_same_device_domains_and_use_left_provenance() {
        let left_keys = Table::from_columns(vec![explicit_i32_column(&[1, 2, 3, 4], 964)]).unwrap();
        let right_keys = Table::from_columns(vec![explicit_i32_column(&[2, 4], 965)]).unwrap();
        let expected_stream = left_keys.testing_execution_domain_stream_value();
        let expected_mr = left_keys.testing_execution_domain_mr_value();

        reset_exec_observation();
        let indices = join::left_anti_join_indices(
            &left_keys.testing_view().unwrap(),
            &right_keys.testing_view().unwrap(),
            NullEquality::Equal,
        )
        .unwrap();

        assert_eq!(indices.indices.testing_execution_domain_id(), left_keys.testing_execution_domain_id());
        assert_eq!(indices.indices.testing_execution_domain_mr_value(), expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(sorted_single_side_indices(&indices), vec![0, 2]);
    }

    #[test]
    fn single_side_join_indices_in_uses_borrowed_stream_and_memory_resource() {
        let left_keys = Table::from_columns(vec![explicit_i32_column(&[1, 2, 3, 4], 1022)]).unwrap();
        let right_keys = Table::from_columns(vec![explicit_i32_column(&[2, 4], 1023)]).unwrap();
        let override_runtime = Runtime::new().unwrap();
        let ctx = ExecutionOverrides::new()
            .with_stream(override_runtime.stream())
            .with_memory_resource(override_runtime.memory_resource());
        let expected_stream = override_runtime.stream().testing_raw_handle_value();
        let expected_mr = override_runtime.memory_resource().testing_raw_handle_value();

        reset_exec_observation();
        let semi = join::left_semi_join_indices_in(
            &left_keys.testing_view().unwrap(),
            &right_keys.testing_view().unwrap(),
            NullEquality::Equal,
            &ctx,
        )
        .unwrap();
        assert_eq!(semi.indices.testing_execution_domain_stream_value(), expected_stream);
        assert_eq!(semi.indices.testing_execution_domain_mr_value(), expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(sorted_single_side_indices(&semi), vec![1, 3]);

        reset_exec_observation();
        let anti = join::left_anti_join_indices_in(
            &left_keys.testing_view().unwrap(),
            &right_keys.testing_view().unwrap(),
            NullEquality::Equal,
            &ctx,
        )
        .unwrap();
        assert_eq!(anti.indices.testing_execution_domain_stream_value(), expected_stream);
        assert_eq!(anti.indices.testing_execution_domain_mr_value(), expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(sorted_single_side_indices(&anti), vec![0, 2]);
    }

    #[test]
    fn cross_join_supports_mixed_same_device_domains_and_uses_left_provenance() {
        let left = Table::from_columns(vec![explicit_i32_column(&[1, 2], 966)]).unwrap();
        let right = Table::from_columns(vec![explicit_i32_column(&[10, 20], 967)]).unwrap();
        let expected_stream = left.testing_execution_domain_stream_value();
        let expected_mr = left.testing_execution_domain_mr_value();

        reset_exec_observation();
        let result = join::cross_join(&left.testing_view().unwrap(), &right.testing_view().unwrap()).unwrap();

        assert_eq!(result.testing_execution_domain_id(), left.testing_execution_domain_id());
        assert_eq!(result.testing_execution_domain_mr_value(), expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(result.testing_view().unwrap().column(0).unwrap().to_vec::<i32>().unwrap(), vec![1, 1, 2, 2]);
        assert_eq!(result.testing_view().unwrap().column(1).unwrap().to_vec::<i32>().unwrap(), vec![10, 20, 10, 20]);
    }

    #[test]
    fn cross_join_in_uses_borrowed_stream_and_memory_resource() {
        let left = Table::from_columns(vec![explicit_i32_column(&[1, 2], 1024)]).unwrap();
        let right = Table::from_columns(vec![explicit_i32_column(&[10, 20], 1025)]).unwrap();
        let override_runtime = Runtime::new().unwrap();
        let ctx = ExecutionOverrides::new()
            .with_stream(override_runtime.stream())
            .with_memory_resource(override_runtime.memory_resource());
        let expected_stream = override_runtime.stream().testing_raw_handle_value();
        let expected_mr = override_runtime.memory_resource().testing_raw_handle_value();

        reset_exec_observation();
        let result = join::cross_join_in(&left.testing_view().unwrap(), &right.testing_view().unwrap(), &ctx).unwrap();

        assert_eq!(result.testing_execution_domain_stream_value(), expected_stream);
        assert_eq!(result.testing_execution_domain_mr_value(), expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(result.testing_view().unwrap().column(0).unwrap().to_vec::<i32>().unwrap(), vec![1, 1, 2, 2]);
        assert_eq!(result.testing_view().unwrap().column(1).unwrap().to_vec::<i32>().unwrap(), vec![10, 20, 10, 20]);
    }

    #[test]
    fn hash_join_indices_support_mixed_same_device_domains_and_use_probe_provenance() {
        let build_keys = Table::from_columns(vec![explicit_i32_column(&[2, 2, 4], 960)]).unwrap();
        let probe_keys = Table::from_columns(vec![explicit_i32_column(&[1, 2, 4], 961)]).unwrap();
        let expected_stream = probe_keys.testing_execution_domain_stream_value();
        let expected_mr = probe_keys.testing_execution_domain_mr_value();

        let hash_join = HashJoin::new(build_keys, HashJoinOptions::new().null_equality(NullEquality::Equal)).unwrap();

        reset_exec_observation();
        let indices = hash_join.left_join_indices(&probe_keys.testing_view().unwrap()).unwrap();

        assert_eq!(indices.left.testing_execution_domain_id(), probe_keys.testing_execution_domain_id());
        assert_eq!(indices.right.testing_execution_domain_id(), probe_keys.testing_execution_domain_id());
        assert_eq!(indices.left.testing_execution_domain_mr_value(), expected_mr);
        assert_eq!(indices.right.testing_execution_domain_mr_value(), expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(sorted_join_index_pairs(&indices), vec![(0, JOIN_NO_MATCH), (1, 0), (1, 1), (2, 2)]);
    }

    #[test]
    fn hash_join_new_in_uses_borrowed_stream_and_memory_resource() {
        let build_keys = Table::from_columns(vec![explicit_i32_column(&[2, 2, 4], 1019)]).unwrap();
        let retained_build_stream = build_keys.testing_execution_domain_stream_value();
        let override_runtime = Runtime::new().unwrap();
        let ctx = ExecutionOverrides::new()
            .with_stream(override_runtime.stream())
            .with_memory_resource(override_runtime.memory_resource());
        let expected_stream = override_runtime.stream().testing_raw_handle_value();
        let expected_mr = override_runtime.memory_resource().testing_raw_handle_value();

        reset_exec_observation();
        let hash_join =
            HashJoin::new_in(build_keys, HashJoinOptions::new().null_equality(NullEquality::Equal), &ctx).unwrap();

        assert_eq!(hash_join.build_keys().testing_execution_domain_stream_value(), retained_build_stream);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
    }

    #[test]
    fn hash_join_pre_build_reservation_contract_matches_retained_build_artifact_bytes() {
        let runtime = Runtime::new().unwrap();
        let build_keys = Table::from_columns(vec![
            runtime.column_from_slice(&[2i32, 2, 4, 4, 8, 8, 16, 32]).unwrap(),
            runtime.column_from_slice(&[10i32, 11, 10, 12, 10, 13, 14, 15]).unwrap(),
        ])
        .unwrap();

        for load_factor in [0.5, 1.0] {
            assert_hash_join_pre_build_matches_built_size(
                &build_keys,
                HashJoinOptions::new().null_equality(NullEquality::Equal).load_factor(load_factor),
            );
        }
    }

    #[test]
    fn hash_join_pre_build_reservation_accepts_empty_rows_with_non_empty_schema() {
        let runtime = Runtime::new().unwrap();
        let build_keys = Table::from_columns(vec![
            runtime.column_from_slice::<i32>(&[]).unwrap(),
            runtime.column_from_slice::<i32>(&[]).unwrap(),
        ])
        .unwrap();

        assert_hash_join_pre_build_matches_built_size(
            &build_keys,
            HashJoinOptions::new().null_equality(NullEquality::Equal),
        );
    }

    #[test]
    fn hash_join_pre_build_reservation_matches_larger_build_table() {
        let runtime = Runtime::new().unwrap();
        let key0 = (0..4096).map(|row| row % 257).collect::<Vec<i32>>();
        let key1 = (0..4096).map(|row| row % 31).collect::<Vec<i32>>();
        let build_keys = Table::from_columns(vec![
            runtime.column_from_slice(&key0).unwrap(),
            runtime.column_from_slice(&key1).unwrap(),
        ])
        .unwrap();

        assert_hash_join_pre_build_matches_built_size(
            &build_keys,
            HashJoinOptions::new().null_equality(NullEquality::Equal),
        );
    }

    #[test]
    fn hash_join_pre_build_reservation_rejects_empty_build_table() {
        let runtime = Runtime::new().unwrap();
        let build_keys = runtime.empty_table().unwrap();

        assert_hash_join_pre_build_invalid_argument(
            &build_keys,
            HashJoinOptions::new().null_equality(NullEquality::Equal),
        );
    }

    #[test]
    fn hash_join_pre_build_reservation_rejects_invalid_load_factors() {
        let build_keys = Table::from_columns(vec![explicit_i32_column(&[2, 2, 4, 8], 12023)]).unwrap();

        for load_factor in [-0.1, 0.0, 1.1] {
            let expected_load_factor = load_factor.to_string();
            let err = HashJoin::pre_build_reservation_size_bytes(
                &build_keys,
                HashJoinOptions::new().null_equality(NullEquality::Equal).load_factor(load_factor),
            )
            .expect_err("invalid load factor should be rejected");

            assert_eq!(err.kind(), ErrorKind::InvalidInput);
            assert_eq!(err.code(), ErrorCode::ForeignInvalidArgument);
            assert_eq!(err.operation(), "hash_join::pre_build_reservation_size_bytes_in");
            assert_eq!(err.detail_value("reservation_contract"), Some("hash_join_pre_build_reservation_size_bytes"));
            assert_eq!(err.detail_value("load_factor"), Some(expected_load_factor.as_str()));
            assert_eq!(err.detail_value("unsupported_input"), None);
        }
    }

    #[test]
    fn hash_join_pre_build_reservation_rejects_nested_and_struct_build_keys() {
        let frame = read_json_dataframe(
            "hash_join_pre_build_nested.jsonl",
            r#"{"items":[1,2],"point":{"x":1,"y":2},"id":1}
{"items":[3],"point":{"x":3,"y":4},"id":2}
"#,
        );

        for key in ["items", "point"] {
            let (build_keys, _) = frame.select([key]).unwrap().try_into_table().unwrap();
            assert_hash_join_pre_build_invalid_argument(
                &build_keys,
                HashJoinOptions::new().null_equality(NullEquality::Equal),
            );
        }
    }

    #[test]
    fn hash_join_pre_build_reservation_rejects_nullable_string_build_key() {
        let build_keys = Table::from_columns(vec![
            column_from_strs_nullable(&["alpha", "null-payload", "omega"], &[true, false, true]).unwrap(),
        ])
        .unwrap();

        assert_hash_join_pre_build_invalid_argument(
            &build_keys,
            HashJoinOptions::new().null_equality(NullEquality::Equal),
        );
    }

    #[test]
    fn hash_join_probe_indices_in_uses_borrowed_stream_and_memory_resource() {
        let build_keys = Table::from_columns(vec![explicit_i32_column(&[2, 2, 4], 1020)]).unwrap();
        let probe_keys = Table::from_columns(vec![explicit_i32_column(&[1, 2, 4], 1021)]).unwrap();
        let hash_join = HashJoin::new(build_keys, HashJoinOptions::new().null_equality(NullEquality::Equal)).unwrap();
        let override_runtime = Runtime::new().unwrap();
        let ctx = ExecutionOverrides::new()
            .with_stream(override_runtime.stream())
            .with_memory_resource(override_runtime.memory_resource());
        let expected_stream = override_runtime.stream().testing_raw_handle_value();
        let expected_mr = override_runtime.memory_resource().testing_raw_handle_value();

        reset_exec_observation();
        let inner = hash_join.inner_join_indices_in(&probe_keys.testing_view().unwrap(), &ctx).unwrap();
        assert_eq!(inner.left.testing_execution_domain_stream_value(), expected_stream);
        assert_eq!(inner.right.testing_execution_domain_stream_value(), expected_stream);
        assert_eq!(inner.left.testing_execution_domain_mr_value(), expected_mr);
        assert_eq!(inner.right.testing_execution_domain_mr_value(), expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(sorted_join_index_pairs(&inner), vec![(1, 0), (1, 1), (2, 2)]);

        reset_exec_observation();
        let left = hash_join.left_join_indices_in(&probe_keys.testing_view().unwrap(), &ctx).unwrap();
        assert_eq!(left.left.testing_execution_domain_stream_value(), expected_stream);
        assert_eq!(left.right.testing_execution_domain_stream_value(), expected_stream);
        assert_eq!(left.left.testing_execution_domain_mr_value(), expected_mr);
        assert_eq!(left.right.testing_execution_domain_mr_value(), expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(sorted_join_index_pairs(&left), vec![(0, JOIN_NO_MATCH), (1, 0), (1, 1), (2, 2)]);

        reset_exec_observation();
        let full = hash_join.full_join_indices_in(&probe_keys.testing_view().unwrap(), &ctx).unwrap();
        assert_eq!(full.left.testing_execution_domain_stream_value(), expected_stream);
        assert_eq!(full.right.testing_execution_domain_stream_value(), expected_stream);
        assert_eq!(full.left.testing_execution_domain_mr_value(), expected_mr);
        assert_eq!(full.right.testing_execution_domain_mr_value(), expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(sorted_join_index_pairs(&full), vec![(0, JOIN_NO_MATCH), (1, 0), (1, 1), (2, 2)]);
    }

    #[test]
    fn conditional_inner_join_indices_support_mixed_same_device_domains_and_use_left_provenance() {
        let left = Table::from_columns(vec![explicit_i32_column(&[1, 3, 5], 971)]).unwrap();
        let right = Table::from_columns(vec![explicit_i32_column(&[2, 4], 972)]).unwrap();
        let predicate = left_less_than_right();
        let expected_stream = left.testing_execution_domain_stream_value();
        let expected_mr = left.testing_execution_domain_mr_value();

        reset_exec_observation();
        let indices = join::conditional_inner_join_indices(
            &left.testing_view().unwrap(),
            &right.testing_view().unwrap(),
            &predicate,
        )
        .unwrap();

        assert_eq!(indices.left.testing_execution_domain_id(), left.testing_execution_domain_id());
        assert_eq!(indices.right.testing_execution_domain_id(), left.testing_execution_domain_id());
        assert_eq!(indices.left.testing_execution_domain_mr_value(), expected_mr);
        assert_eq!(indices.right.testing_execution_domain_mr_value(), expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(sorted_join_index_pairs(&indices), vec![(0, 0), (0, 1), (1, 1)]);
    }

    #[test]
    fn conditional_left_join_indices_support_mixed_same_device_domains_and_use_left_provenance() {
        let left = Table::from_columns(vec![explicit_i32_column(&[1, 3, 5], 983)]).unwrap();
        let right = Table::from_columns(vec![explicit_i32_column(&[2, 4], 984)]).unwrap();
        let predicate = left_less_than_right();
        let expected_stream = left.testing_execution_domain_stream_value();
        let expected_mr = left.testing_execution_domain_mr_value();

        reset_exec_observation();
        let indices = join::conditional_left_join_indices(
            &left.testing_view().unwrap(),
            &right.testing_view().unwrap(),
            &predicate,
        )
        .unwrap();

        assert_eq!(indices.left.testing_execution_domain_id(), left.testing_execution_domain_id());
        assert_eq!(indices.right.testing_execution_domain_id(), left.testing_execution_domain_id());
        assert_eq!(indices.left.testing_execution_domain_mr_value(), expected_mr);
        assert_eq!(indices.right.testing_execution_domain_mr_value(), expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(sorted_join_index_pairs(&indices), vec![(0, 0), (0, 1), (1, 1), (2, JOIN_NO_MATCH)]);
    }

    #[test]
    fn conditional_full_join_indices_support_mixed_same_device_domains_and_use_left_provenance() {
        let left = Table::from_columns(vec![explicit_i32_column(&[1, 3, 5], 985)]).unwrap();
        let right = Table::from_columns(vec![explicit_i32_column(&[2, 4, 6], 986)]).unwrap();
        let predicate = left_less_than_right();
        let expected_stream = left.testing_execution_domain_stream_value();
        let expected_mr = left.testing_execution_domain_mr_value();

        reset_exec_observation();
        let indices = join::conditional_full_join_indices(
            &left.testing_view().unwrap(),
            &right.testing_view().unwrap(),
            &predicate,
        )
        .unwrap();

        assert_eq!(indices.left.testing_execution_domain_id(), left.testing_execution_domain_id());
        assert_eq!(indices.right.testing_execution_domain_id(), left.testing_execution_domain_id());
        assert_eq!(indices.left.testing_execution_domain_mr_value(), expected_mr);
        assert_eq!(indices.right.testing_execution_domain_mr_value(), expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(sorted_join_index_pairs(&indices), vec![(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]);
    }

    #[test]
    fn conditional_left_semi_join_indices_support_mixed_same_device_domains_and_use_left_provenance() {
        let left = Table::from_columns(vec![explicit_i32_column(&[1, 3, 5], 973)]).unwrap();
        let right = Table::from_columns(vec![explicit_i32_column(&[2, 4], 974)]).unwrap();
        let predicate = left_less_than_right();
        let expected_stream = left.testing_execution_domain_stream_value();
        let expected_mr = left.testing_execution_domain_mr_value();

        reset_exec_observation();
        let indices = join::conditional_left_semi_join_indices(
            &left.testing_view().unwrap(),
            &right.testing_view().unwrap(),
            &predicate,
        )
        .unwrap();

        assert_eq!(indices.indices.testing_execution_domain_id(), left.testing_execution_domain_id());
        assert_eq!(indices.indices.testing_execution_domain_mr_value(), expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(sorted_single_side_indices(&indices), vec![0, 1]);
    }

    #[test]
    fn conditional_left_anti_join_indices_support_mixed_same_device_domains_and_use_left_provenance() {
        let left = Table::from_columns(vec![explicit_i32_column(&[1, 3, 5], 987)]).unwrap();
        let right = Table::from_columns(vec![explicit_i32_column(&[2, 4], 988)]).unwrap();
        let predicate = left_less_than_right();
        let expected_stream = left.testing_execution_domain_stream_value();
        let expected_mr = left.testing_execution_domain_mr_value();

        reset_exec_observation();
        let indices = join::conditional_left_anti_join_indices(
            &left.testing_view().unwrap(),
            &right.testing_view().unwrap(),
            &predicate,
        )
        .unwrap();

        assert_eq!(indices.indices.testing_execution_domain_id(), left.testing_execution_domain_id());
        assert_eq!(indices.indices.testing_execution_domain_mr_value(), expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(sorted_single_side_indices(&indices), vec![2]);
    }

    #[test]
    fn conditional_join_indices_in_uses_borrowed_stream_and_memory_resource() {
        let left = Table::from_columns(vec![explicit_i32_column(&[1, 3, 5], 989)]).unwrap();
        let right = Table::from_columns(vec![explicit_i32_column(&[2, 4], 990)]).unwrap();
        let predicate = left_less_than_right();
        let override_runtime = Runtime::new().unwrap();
        let ctx = ExecutionOverrides::new()
            .with_stream(override_runtime.stream())
            .with_memory_resource(override_runtime.memory_resource());
        let expected_stream = override_runtime.stream().testing_raw_handle_value();
        let expected_mr = override_runtime.memory_resource().testing_raw_handle_value();

        reset_exec_observation();
        let indices = join::conditional_left_anti_join_indices_in(
            &left.testing_view().unwrap(),
            &right.testing_view().unwrap(),
            &predicate,
            &ctx,
        )
        .unwrap();

        assert_eq!(indices.indices.testing_execution_domain_stream_value(), expected_stream);
        assert_eq!(indices.indices.testing_execution_domain_mr_value(), expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(sorted_single_side_indices(&indices), vec![2]);
    }

    #[test]
    fn conditional_join_indices_orders_literal_scalar_domain_before_left_exec() {
        let _event_guard = lock_event_observation();
        let left = Table::from_columns(vec![explicit_i32_column(&[1, 5], 1003)]).unwrap();
        let right = Table::from_columns(vec![explicit_i32_column(&[10, 20], 1004)]).unwrap();
        let literal_runtime = Runtime::new().unwrap();
        let literal = AstExpression::literal(literal_runtime.scalar(4i32).unwrap()).unwrap();
        let predicate = AstExpression::less(AstExpression::left_column(0).unwrap(), literal).unwrap();
        let expected_stream = left.testing_execution_domain_stream_value();
        let expected_mr = left.testing_execution_domain_mr_value();

        reset_exec_observation();
        reset_dependency_observation();
        let indices = join::conditional_inner_join_indices(
            &left.testing_view().unwrap(),
            &right.testing_view().unwrap(),
            &predicate,
        )
        .unwrap();

        assert_eq!(indices.left.testing_execution_domain_id(), left.testing_execution_domain_id());
        assert_eq!(indices.right.testing_execution_domain_id(), left.testing_execution_domain_id());
        assert_eq!(indices.left.testing_execution_domain_mr_value(), expected_mr);
        assert_eq!(indices.right.testing_execution_domain_mr_value(), expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(dependency_wait_call_count_for_stream(expected_stream), 2);
        assert_eq!(sorted_join_index_pairs(&indices), vec![(0, 0), (0, 1)]);
    }

    #[test]
    fn mixed_inner_join_indices_support_mixed_same_device_domains_and_use_primary_left_provenance() {
        let left_equality = Table::from_columns(vec![explicit_i32_column(&[1, 1, 2], 975)]).unwrap();
        let right_equality = Table::from_columns(vec![explicit_i32_column(&[1, 2], 976)]).unwrap();
        let left_conditional = Table::from_columns(vec![explicit_i32_column(&[5, 7, 3], 977)]).unwrap();
        let right_conditional = Table::from_columns(vec![explicit_i32_column(&[6, 4], 978)]).unwrap();
        let predicate = left_less_than_right();
        let expected_stream = left_equality.testing_execution_domain_stream_value();
        let expected_mr = left_equality.testing_execution_domain_mr_value();

        reset_exec_observation();
        let indices = join::mixed_inner_join_indices(
            &left_equality.testing_view().unwrap(),
            &right_equality.testing_view().unwrap(),
            &left_conditional.testing_view().unwrap(),
            &right_conditional.testing_view().unwrap(),
            &predicate,
            NullEquality::Equal,
        )
        .unwrap();

        assert_eq!(indices.left.testing_execution_domain_id(), left_equality.testing_execution_domain_id());
        assert_eq!(indices.right.testing_execution_domain_id(), left_equality.testing_execution_domain_id());
        assert_eq!(indices.left.testing_execution_domain_mr_value(), expected_mr);
        assert_eq!(indices.right.testing_execution_domain_mr_value(), expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(sorted_join_index_pairs(&indices), vec![(0, 0), (2, 1)]);
    }

    #[test]
    fn mixed_inner_join_indices_in_uses_borrowed_stream_and_memory_resource() {
        let left_equality = Table::from_columns(vec![explicit_i32_column(&[1, 1, 2], 1005)]).unwrap();
        let right_equality = Table::from_columns(vec![explicit_i32_column(&[1, 2], 1006)]).unwrap();
        let left_conditional = Table::from_columns(vec![explicit_i32_column(&[5, 7, 3], 1007)]).unwrap();
        let right_conditional = Table::from_columns(vec![explicit_i32_column(&[6, 4], 1008)]).unwrap();
        let predicate = left_less_than_right();
        let override_runtime = Runtime::new().unwrap();
        let ctx = ExecutionOverrides::new()
            .with_stream(override_runtime.stream())
            .with_memory_resource(override_runtime.memory_resource());
        let expected_stream = override_runtime.stream().testing_raw_handle_value();
        let expected_mr = override_runtime.memory_resource().testing_raw_handle_value();

        reset_exec_observation();
        let indices = join::mixed_inner_join_indices_in(
            &left_equality.testing_view().unwrap(),
            &right_equality.testing_view().unwrap(),
            &left_conditional.testing_view().unwrap(),
            &right_conditional.testing_view().unwrap(),
            &predicate,
            NullEquality::Equal,
            &ctx,
        )
        .unwrap();

        assert_eq!(indices.left.testing_execution_domain_stream_value(), expected_stream);
        assert_eq!(indices.right.testing_execution_domain_stream_value(), expected_stream);
        assert_eq!(indices.left.testing_execution_domain_mr_value(), expected_mr);
        assert_eq!(indices.right.testing_execution_domain_mr_value(), expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(sorted_join_index_pairs(&indices), vec![(0, 0), (2, 1)]);
    }

    #[test]
    fn mixed_full_semi_anti_join_indices_in_use_borrowed_stream_and_memory_resource() {
        let left_equality = Table::from_columns(vec![explicit_i32_column(&[1, 1, 2, 3], 1026)]).unwrap();
        let right_equality = Table::from_columns(vec![explicit_i32_column(&[1, 2, 4], 1027)]).unwrap();
        let left_conditional = Table::from_columns(vec![explicit_i32_column(&[5, 7, 3, 1], 1028)]).unwrap();
        let right_conditional = Table::from_columns(vec![explicit_i32_column(&[6, 4, 9], 1029)]).unwrap();
        let predicate = left_less_than_right();
        let override_runtime = Runtime::new().unwrap();
        let ctx = ExecutionOverrides::new()
            .with_stream(override_runtime.stream())
            .with_memory_resource(override_runtime.memory_resource());
        let expected_stream = override_runtime.stream().testing_raw_handle_value();
        let expected_mr = override_runtime.memory_resource().testing_raw_handle_value();

        reset_exec_observation();
        let full = join::mixed_full_join_indices_in(
            &left_equality.testing_view().unwrap(),
            &right_equality.testing_view().unwrap(),
            &left_conditional.testing_view().unwrap(),
            &right_conditional.testing_view().unwrap(),
            &predicate,
            NullEquality::Equal,
            &ctx,
        )
        .unwrap();
        assert_eq!(full.left.testing_execution_domain_stream_value(), expected_stream);
        assert_eq!(full.right.testing_execution_domain_stream_value(), expected_stream);
        assert_eq!(full.left.testing_execution_domain_mr_value(), expected_mr);
        assert_eq!(full.right.testing_execution_domain_mr_value(), expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(
            sorted_join_index_pairs(&full),
            vec![(JOIN_NO_MATCH, 2), (0, 0), (1, JOIN_NO_MATCH), (2, 1), (3, JOIN_NO_MATCH),]
        );

        reset_exec_observation();
        let semi = join::mixed_left_semi_join_indices_in(
            &left_equality.testing_view().unwrap(),
            &right_equality.testing_view().unwrap(),
            &left_conditional.testing_view().unwrap(),
            &right_conditional.testing_view().unwrap(),
            &predicate,
            NullEquality::Equal,
            &ctx,
        )
        .unwrap();
        assert_eq!(semi.indices.testing_execution_domain_stream_value(), expected_stream);
        assert_eq!(semi.indices.testing_execution_domain_mr_value(), expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(sorted_single_side_indices(&semi), vec![0, 2]);

        reset_exec_observation();
        let anti = join::mixed_left_anti_join_indices_in(
            &left_equality.testing_view().unwrap(),
            &right_equality.testing_view().unwrap(),
            &left_conditional.testing_view().unwrap(),
            &right_conditional.testing_view().unwrap(),
            &predicate,
            NullEquality::Equal,
            &ctx,
        )
        .unwrap();
        assert_eq!(anti.indices.testing_execution_domain_stream_value(), expected_stream);
        assert_eq!(anti.indices.testing_execution_domain_mr_value(), expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(sorted_single_side_indices(&anti), vec![1, 3]);
    }

    #[test]
    fn filter_join_indices_support_mixed_same_device_domains_and_use_left_index_provenance() {
        let left = Table::from_columns(vec![explicit_i32_column(&[1, 3, 5], 1009)]).unwrap();
        let right = Table::from_columns(vec![explicit_i32_column(&[2, 4], 1010)]).unwrap();
        let left_indices = explicit_i32_column(&[0, 1, 2], 1011);
        let right_indices = explicit_i32_column(&[0, 0, 1], 1012);
        let predicate = left_less_than_right();
        let expected_stream = left_indices.testing_execution_domain_stream_value();
        let expected_mr = left_indices.testing_execution_domain_mr_value();

        reset_exec_observation();
        let indices = join::filter_join_indices(
            &left.testing_view().unwrap(),
            &right.testing_view().unwrap(),
            &left_indices.testing_view().unwrap(),
            &right_indices.testing_view().unwrap(),
            &predicate,
            JoinType::Inner,
        )
        .unwrap();

        assert_eq!(indices.left.testing_execution_domain_id(), left_indices.testing_execution_domain_id());
        assert_eq!(indices.right.testing_execution_domain_id(), left_indices.testing_execution_domain_id());
        assert_eq!(indices.left.testing_execution_domain_mr_value(), expected_mr);
        assert_eq!(indices.right.testing_execution_domain_mr_value(), expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(sorted_join_index_pairs(&indices), vec![(0, 0)]);
    }

    #[test]
    fn filter_join_indices_in_uses_borrowed_stream_and_memory_resource() {
        let left = Table::from_columns(vec![explicit_i32_column(&[1, 3, 5], 1013)]).unwrap();
        let right = Table::from_columns(vec![explicit_i32_column(&[2, 4], 1014)]).unwrap();
        let left_indices = explicit_i32_column(&[0, 1, 2], 1015);
        let right_indices = explicit_i32_column(&[0, 0, 1], 1016);
        let predicate = left_less_than_right();
        let override_runtime = Runtime::new().unwrap();
        let ctx = ExecutionOverrides::new()
            .with_stream(override_runtime.stream())
            .with_memory_resource(override_runtime.memory_resource());
        let expected_stream = override_runtime.stream().testing_raw_handle_value();
        let expected_mr = override_runtime.memory_resource().testing_raw_handle_value();

        reset_exec_observation();
        let indices = join::filter_join_indices_in(
            &left.testing_view().unwrap(),
            &right.testing_view().unwrap(),
            &left_indices.testing_view().unwrap(),
            &right_indices.testing_view().unwrap(),
            &predicate,
            JoinType::Inner,
            &ctx,
        )
        .unwrap();

        assert_eq!(indices.left.testing_execution_domain_stream_value(), expected_stream);
        assert_eq!(indices.right.testing_execution_domain_stream_value(), expected_stream);
        assert_eq!(indices.left.testing_execution_domain_mr_value(), expected_mr);
        assert_eq!(indices.right.testing_execution_domain_mr_value(), expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(sorted_join_index_pairs(&indices), vec![(0, 0)]);
    }

    #[test]
    fn mixed_left_join_indices_support_mixed_same_device_domains_and_use_primary_left_provenance() {
        let left_equality = Table::from_columns(vec![explicit_i32_column(&[1, 1, 2, 3], 991)]).unwrap();
        let right_equality = Table::from_columns(vec![explicit_i32_column(&[1, 2], 992)]).unwrap();
        let left_conditional = Table::from_columns(vec![explicit_i32_column(&[5, 7, 3, 1], 993)]).unwrap();
        let right_conditional = Table::from_columns(vec![explicit_i32_column(&[6, 4], 994)]).unwrap();
        let predicate = left_less_than_right();
        let expected_stream = left_equality.testing_execution_domain_stream_value();
        let expected_mr = left_equality.testing_execution_domain_mr_value();

        reset_exec_observation();
        let indices = join::mixed_left_join_indices(
            &left_equality.testing_view().unwrap(),
            &right_equality.testing_view().unwrap(),
            &left_conditional.testing_view().unwrap(),
            &right_conditional.testing_view().unwrap(),
            &predicate,
            NullEquality::Equal,
        )
        .unwrap();

        assert_eq!(indices.left.testing_execution_domain_id(), left_equality.testing_execution_domain_id());
        assert_eq!(indices.right.testing_execution_domain_id(), left_equality.testing_execution_domain_id());
        assert_eq!(indices.left.testing_execution_domain_mr_value(), expected_mr);
        assert_eq!(indices.right.testing_execution_domain_mr_value(), expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(sorted_join_index_pairs(&indices), vec![(0, 0), (1, JOIN_NO_MATCH), (2, 1), (3, JOIN_NO_MATCH)]);
    }

    #[test]
    fn mixed_full_join_indices_support_mixed_same_device_domains_and_use_primary_left_provenance() {
        let left_equality = Table::from_columns(vec![explicit_i32_column(&[1, 1, 2, 3], 995)]).unwrap();
        let right_equality = Table::from_columns(vec![explicit_i32_column(&[1, 2, 4], 996)]).unwrap();
        let left_conditional = Table::from_columns(vec![explicit_i32_column(&[5, 7, 3, 1], 997)]).unwrap();
        let right_conditional = Table::from_columns(vec![explicit_i32_column(&[6, 4, 9], 998)]).unwrap();
        let predicate = left_less_than_right();
        let expected_stream = left_equality.testing_execution_domain_stream_value();
        let expected_mr = left_equality.testing_execution_domain_mr_value();

        reset_exec_observation();
        let indices = join::mixed_full_join_indices(
            &left_equality.testing_view().unwrap(),
            &right_equality.testing_view().unwrap(),
            &left_conditional.testing_view().unwrap(),
            &right_conditional.testing_view().unwrap(),
            &predicate,
            NullEquality::Equal,
        )
        .unwrap();

        assert_eq!(indices.left.testing_execution_domain_id(), left_equality.testing_execution_domain_id());
        assert_eq!(indices.right.testing_execution_domain_id(), left_equality.testing_execution_domain_id());
        assert_eq!(indices.left.testing_execution_domain_mr_value(), expected_mr);
        assert_eq!(indices.right.testing_execution_domain_mr_value(), expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(
            sorted_join_index_pairs(&indices),
            vec![(JOIN_NO_MATCH, 2), (0, 0), (1, JOIN_NO_MATCH), (2, 1), (3, JOIN_NO_MATCH),]
        );
    }

    #[test]
    fn mixed_left_semi_join_indices_support_mixed_same_device_domains_and_use_primary_left_provenance() {
        let left_equality = Table::from_columns(vec![explicit_i32_column(&[1, 1, 2], 979)]).unwrap();
        let right_equality = Table::from_columns(vec![explicit_i32_column(&[1, 2], 980)]).unwrap();
        let left_conditional = Table::from_columns(vec![explicit_i32_column(&[5, 7, 3], 981)]).unwrap();
        let right_conditional = Table::from_columns(vec![explicit_i32_column(&[6, 4], 982)]).unwrap();
        let predicate = left_less_than_right();
        let expected_stream = left_equality.testing_execution_domain_stream_value();
        let expected_mr = left_equality.testing_execution_domain_mr_value();

        reset_exec_observation();
        let indices = join::mixed_left_semi_join_indices(
            &left_equality.testing_view().unwrap(),
            &right_equality.testing_view().unwrap(),
            &left_conditional.testing_view().unwrap(),
            &right_conditional.testing_view().unwrap(),
            &predicate,
            NullEquality::Equal,
        )
        .unwrap();

        assert_eq!(indices.indices.testing_execution_domain_id(), left_equality.testing_execution_domain_id());
        assert_eq!(indices.indices.testing_execution_domain_mr_value(), expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(sorted_single_side_indices(&indices), vec![0, 2]);
    }

    #[test]
    fn mixed_left_anti_join_indices_support_mixed_same_device_domains_and_use_primary_left_provenance() {
        let left_equality = Table::from_columns(vec![explicit_i32_column(&[1, 1, 2, 3], 999)]).unwrap();
        let right_equality = Table::from_columns(vec![explicit_i32_column(&[1, 2], 1000)]).unwrap();
        let left_conditional = Table::from_columns(vec![explicit_i32_column(&[5, 7, 3, 1], 1001)]).unwrap();
        let right_conditional = Table::from_columns(vec![explicit_i32_column(&[6, 4], 1002)]).unwrap();
        let predicate = left_less_than_right();
        let expected_stream = left_equality.testing_execution_domain_stream_value();
        let expected_mr = left_equality.testing_execution_domain_mr_value();

        reset_exec_observation();
        let indices = join::mixed_left_anti_join_indices(
            &left_equality.testing_view().unwrap(),
            &right_equality.testing_view().unwrap(),
            &left_conditional.testing_view().unwrap(),
            &right_conditional.testing_view().unwrap(),
            &predicate,
            NullEquality::Equal,
        )
        .unwrap();

        assert_eq!(indices.indices.testing_execution_domain_id(), left_equality.testing_execution_domain_id());
        assert_eq!(indices.indices.testing_execution_domain_mr_value(), expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(sorted_single_side_indices(&indices), vec![1, 3]);
    }
}
mod copying_execution_tests {
    use cudf::{ExecutionOverrides, Runtime, core::Table, ops::copying, testing::prelude::*};

    use crate::common::{explicit_i32_column, last_exec_mr_value, last_exec_stream_value, reset_exec_observation};

    #[test]
    fn scatter_supports_mixed_same_device_domains_and_uses_target_provenance() {
        let source = Table::from_columns(vec![explicit_i32_column(&[100, 200], 968)]).unwrap();
        let scatter_map = explicit_i32_column(&[1, -1], 969);
        let target = Table::from_columns(vec![explicit_i32_column(&[10, 20, 30, 40], 970)]).unwrap();
        let expected_stream = target.testing_execution_domain_stream_value();
        let expected_mr = target.testing_execution_domain_mr_value();

        reset_exec_observation();
        let result = copying::scatter(
            &source.testing_view().unwrap(),
            &scatter_map.testing_view().unwrap(),
            &target.testing_view().unwrap(),
        )
        .unwrap();

        assert_eq!(result.testing_execution_domain_id(), target.testing_execution_domain_id());
        assert_eq!(result.testing_execution_domain_mr_value(), expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(result.testing_view().unwrap().column(0).unwrap().to_vec::<i32>().unwrap(), vec![10, 100, 30, 200]);
    }

    #[test]
    fn scatter_in_uses_borrowed_stream_and_memory_resource() {
        let source = Table::from_columns(vec![explicit_i32_column(&[100, 200], 1026)]).unwrap();
        let scatter_map = explicit_i32_column(&[1, -1], 1027);
        let target = Table::from_columns(vec![explicit_i32_column(&[10, 20, 30, 40], 1028)]).unwrap();
        let override_runtime = Runtime::new().unwrap();
        let ctx = ExecutionOverrides::new()
            .with_stream(override_runtime.stream())
            .with_memory_resource(override_runtime.memory_resource());
        let expected_stream = override_runtime.stream().testing_raw_handle_value();
        let expected_mr = override_runtime.memory_resource().testing_raw_handle_value();

        reset_exec_observation();
        let result = copying::scatter_in(
            &source.testing_view().unwrap(),
            &scatter_map.testing_view().unwrap(),
            &target.testing_view().unwrap(),
            &ctx,
        )
        .unwrap();

        assert_eq!(result.testing_execution_domain_stream_value(), expected_stream);
        assert_eq!(result.testing_execution_domain_mr_value(), expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(result.testing_view().unwrap().column(0).unwrap().to_vec::<i32>().unwrap(), vec![10, 100, 30, 200]);
    }

    #[test]
    fn scatter_unchecked_in_uses_borrowed_stream_and_memory_resource() {
        let source = Table::from_columns(vec![explicit_i32_column(&[100, 200], 1029)]).unwrap();
        let scatter_map = explicit_i32_column(&[0, 2], 1030);
        let target = Table::from_columns(vec![explicit_i32_column(&[10, 20, 30], 1031)]).unwrap();
        let override_runtime = Runtime::new().unwrap();
        let ctx = ExecutionOverrides::new()
            .with_stream(override_runtime.stream())
            .with_memory_resource(override_runtime.memory_resource());
        let expected_stream = override_runtime.stream().testing_raw_handle_value();
        let expected_mr = override_runtime.memory_resource().testing_raw_handle_value();

        reset_exec_observation();
        let result = unsafe {
            copying::scatter_unchecked_in(
                &source.testing_view().unwrap(),
                &scatter_map.testing_view().unwrap(),
                &target.testing_view().unwrap(),
                &ctx,
            )
        }
        .unwrap();

        assert_eq!(result.testing_execution_domain_stream_value(), expected_stream);
        assert_eq!(result.testing_execution_domain_mr_value(), expected_mr);
        assert_eq!(last_exec_stream_value(), expected_stream);
        assert_eq!(last_exec_mr_value(), expected_mr);
        assert_eq!(result.testing_view().unwrap().column(0).unwrap().to_vec::<i32>().unwrap(), vec![100, 20, 200]);
    }
}
#[path = "../ops/strings/strings_tests.rs"]
mod strings_tests;

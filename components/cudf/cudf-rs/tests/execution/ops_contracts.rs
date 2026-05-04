#[path = "../ops/datetime/datetime_tests.rs"]
mod datetime_tests;
mod join_hash_join_execution_tests {
    use cudf::{
        core::Table,
        ops::{
            ast::AstExpression,
            join::{self, HashJoin, HashJoinOptions, NullEquality},
        },
        testing::prelude::*,
    };

    use crate::common::{explicit_i32_column, last_exec_mr_value, last_exec_stream_value, reset_exec_observation};

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
}
mod copying_execution_tests {
    use cudf::{core::Table, ops::copying, testing::prelude::*};

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
}
#[path = "../ops/strings/strings_tests.rs"]
mod strings_tests;

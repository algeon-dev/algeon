#[path = "../ops/datetime/datetime_tests.rs"]
mod datetime_tests;
mod join_hash_join_execution_tests {
    use cudf::{
        core::Table,
        ops::join::{self, HashJoin, HashJoinOptions, NullEquality},
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
}
#[path = "../ops/strings/strings_tests.rs"]
mod strings_tests;

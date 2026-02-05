use super::*;

#[path = "mixed_residual/builders.rs"]
mod builders;
#[path = "mixed_residual/root.rs"]
mod root;

use builders::*;

pub(crate) fn admitted_mixed_residual_multi_join_plan() -> cudf_query_engine::QueryPlan {
    mixed_residual_multi_join_plan(
        JoinType::Inner,
        JoinSide::Right,
        vec![
            JoinOutputColumn::left(ColumnRef::input(0, "left_payload", "left_payload"), "left_payload"),
            JoinOutputColumn::right(ColumnRef::input(1, "right_payload", "right_payload"), "right_payload"),
        ],
    )
}

#[test]
fn bounded_multi_join_admits_mixed_keyed_residual_inner_left_semi_and_left_anti() {
    for (join_type, projection) in [
        (
            JoinType::Inner,
            vec![
                JoinOutputColumn::left(ColumnRef::input(0, "left_payload", "left_payload"), "left_payload"),
                JoinOutputColumn::right(ColumnRef::input(1, "right_payload", "right_payload"), "right_payload"),
            ],
        ),
        (
            JoinType::LeftSemi,
            vec![JoinOutputColumn::left(ColumnRef::input(0, "left_payload", "left_payload"), "left_payload")],
        ),
        (
            JoinType::LeftAnti,
            vec![JoinOutputColumn::left(ColumnRef::input(0, "left_payload", "left_payload"), "left_payload")],
        ),
    ] {
        let plan = mixed_residual_multi_join_plan(join_type, JoinSide::Right, projection);

        assert!(!plan.has_bounded_executable_root_join());
        assert!(!plan.has_bounded_executable_join_pipeline());
        assert!(plan.has_bounded_executable_multi_join_pipeline());
        assert!(plan.bounded_join_unsupported_shape_facts(ExecutionBoundaryPolicy::default()).is_none());
    }

    let plan = mixed_residual_multi_join_plan(
        JoinType::Left,
        JoinSide::Right,
        vec![
            JoinOutputColumn::left(ColumnRef::input(0, "left_payload", "left_payload"), "left_payload"),
            JoinOutputColumn::right(ColumnRef::input(1, "right_payload", "right_payload"), "right_payload"),
        ],
    );
    assert!(!plan.has_bounded_executable_root_join());
    assert!(!plan.has_bounded_executable_join_pipeline());
    assert!(!plan.has_bounded_executable_multi_join_pipeline());
    let facts = plan
        .bounded_join_unsupported_shape_facts(ExecutionBoundaryPolicy::default())
        .expect("unsupported mixed left multi-join shape should produce facts");
    assert_eq!(facts.get("bounded_join_lower_level_blockers").map(String::as_str), Some(""));
    assert_eq!(
        facts.get("bounded_join_specific_blocker").map(String::as_str),
        Some("bounded_multi_join_mixed_keyed_residual_left_topology_unsupported")
    );
    assert_eq!(facts.get("bounded_join_blocked_join_family"), None);
    assert_eq!(facts.get("bounded_join_blocked_join_type").map(String::as_str), Some("left"));
    assert_eq!(facts.get("bounded_join_required_shape_contract"), None);
    assert_eq!(facts.get("bounded_join_required_lower_level_contract").map(String::as_str), None);
    assert!(facts.iter().all(|(key, value)| !key.contains("estimate") && !value.contains("estimate")));

    let plan = mixed_residual_multi_join_plan(
        JoinType::Full,
        JoinSide::Right,
        vec![
            JoinOutputColumn::left(ColumnRef::input(0, "left_payload", "left_payload"), "left_payload"),
            JoinOutputColumn::right(ColumnRef::input(1, "right_payload", "right_payload"), "right_payload"),
        ],
    );
    assert!(!plan.has_bounded_executable_root_join());
    assert!(!plan.has_bounded_executable_join_pipeline());
    assert!(!plan.has_bounded_executable_multi_join_pipeline());
}

#[test]
fn bounded_multi_join_reports_left_full_blocker_after_valid_mixed_residual_semi_anti() {
    for (valid_join_type, invalid_join_type, expected_family, expected_blocker) in [
        (
            JoinType::LeftSemi,
            JoinType::Left,
            "mixed_keyed_residual_left",
            "bounded_multi_join_mixed_keyed_residual_left_topology_unsupported",
        ),
        (
            JoinType::LeftAnti,
            JoinType::Full,
            "mixed_keyed_residual_full",
            "bounded_multi_join_mixed_keyed_residual_full_topology_unsupported",
        ),
    ] {
        let plan =
            valid_mixed_residual_semi_anti_then_invalid_mixed_residual_join_plan(valid_join_type, invalid_join_type);

        assert!(!plan.has_bounded_executable_root_join());
        assert!(!plan.has_bounded_executable_join_pipeline());
        assert!(!plan.has_bounded_executable_multi_join_pipeline());
        let facts = plan
            .bounded_join_unsupported_shape_facts(ExecutionBoundaryPolicy::default())
            .expect("unsupported mixed left/full multi-join shape should produce facts");
        assert_eq!(facts.get("bounded_join_lower_level_blockers").map(String::as_str), Some(""));
        assert_eq!(facts.get("bounded_join_specific_blocker").map(String::as_str), Some(expected_blocker));
        let _ = expected_family;
        assert_eq!(facts.get("bounded_join_blocked_join_family"), None);
        let expected_join_type = invalid_join_type.to_string();
        assert_eq!(facts.get("bounded_join_blocked_join_type").map(String::as_str), Some(expected_join_type.as_str()));
        assert_eq!(facts.get("bounded_join_required_shape_contract"), None);
        assert_eq!(facts.get("bounded_join_required_lower_level_contract").map(String::as_str), None);
    }
}

#[test]
fn bounded_multi_join_rejects_invalid_mixed_residual_semi_anti_with_explicit_blocker_facts() {
    for (join_type, build_side, projection, expected_blocker) in [
        (
            JoinType::LeftSemi,
            JoinSide::Left,
            vec![JoinOutputColumn::left(ColumnRef::input(0, "left_payload", "left_payload"), "left_payload")],
            "multi_join_mixed_keyed_residual_semi_anti_requires_right_build_side",
        ),
        (
            JoinType::LeftAnti,
            JoinSide::Right,
            vec![JoinOutputColumn::right(ColumnRef::input(1, "right_payload", "right_payload"), "right_payload")],
            "multi_join_mixed_keyed_residual_semi_anti_requires_left_only_projection",
        ),
    ] {
        let plan = mixed_residual_multi_join_plan(join_type, build_side, projection);

        assert!(!plan.has_bounded_executable_root_join());
        assert!(!plan.has_bounded_executable_join_pipeline());
        assert!(!plan.has_bounded_executable_multi_join_pipeline());
        let facts = plan
            .bounded_join_unsupported_shape_facts(ExecutionBoundaryPolicy::default())
            .expect("unsupported mixed semi/anti multi-join shape should produce facts");
        assert_eq!(facts.get("bounded_join_lower_level_blockers").map(String::as_str), Some(""));
        assert_eq!(facts.get("bounded_join_specific_blocker").map(String::as_str), Some(expected_blocker));
        assert_eq!(facts.get("bounded_join_blocked_join_family"), None);
        let expected_join_type = join_type.to_string();
        assert_eq!(facts.get("bounded_join_blocked_join_type").map(String::as_str), Some(expected_join_type.as_str()));
        assert_eq!(facts.get("bounded_join_required_shape_contract"), None);
        assert_eq!(facts.get("bounded_join_required_lower_level_contract").map(String::as_str), None);
        assert!(facts.iter().all(|(key, value)| !key.contains("estimate") && !value.contains("estimate")));
    }
}

#[test]
fn bounded_multi_join_keeps_aggregate_inputs_out_of_join_subtrees() {
    let plan = mixed_residual_multi_join_with_aggregate_leaf_plan();

    assert!(!plan.has_bounded_executable_root_join());
    assert!(!plan.has_bounded_executable_join_pipeline());
    assert!(!plan.has_bounded_executable_multi_join_pipeline());

    for aggregate_leaf in
        [MultiJoinAggregateLeaf::FirstLeft, MultiJoinAggregateLeaf::FirstRight, MultiJoinAggregateLeaf::SecondRight]
    {
        let plan = hash_multi_join_with_aggregate_leaf_plan(aggregate_leaf);

        assert!(!plan.has_bounded_executable_root_join());
        assert!(!plan.has_bounded_executable_join_pipeline());
        assert!(
            !plan.has_bounded_executable_multi_join_pipeline(),
            "aggregate {aggregate_leaf:?} leaf must not be accepted below a bounded multi-join"
        );
        assert!(plan.bounded_join_unsupported_shape_facts(ExecutionBoundaryPolicy::default()).is_some());
    }
}

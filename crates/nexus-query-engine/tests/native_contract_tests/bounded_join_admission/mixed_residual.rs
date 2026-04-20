use super::*;

#[path = "mixed_residual/builders.rs"]
mod builders;
#[path = "mixed_residual/root.rs"]
mod root;

use builders::*;

pub(crate) fn admitted_mixed_residual_multi_join_plan() -> nexus_query_engine::plan::QueryPlan {
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

        assert!(!plan.capability_analysis().admits_root_join());
        assert!(!plan.capability_analysis().admits_single_join_post_root_pipeline());
        assert!(plan.capability_analysis().admits_closed_multi_join_subgraph());
        assert!(unsupported_facts(&plan, ExecutionBoundaryPolicy::default()).is_none());
    }

    let plan = mixed_residual_multi_join_plan(
        JoinType::Left,
        JoinSide::Right,
        vec![
            JoinOutputColumn::left(ColumnRef::input(0, "left_payload", "left_payload"), "left_payload"),
            JoinOutputColumn::right(ColumnRef::input(1, "right_payload", "right_payload"), "right_payload"),
        ],
    );
    assert!(!plan.capability_analysis().admits_root_join());
    assert!(!plan.capability_analysis().admits_single_join_post_root_pipeline());
    assert!(!plan.capability_analysis().admits_closed_multi_join_subgraph());
    let facts = unsupported_facts(&plan, ExecutionBoundaryPolicy::default())
        .expect("unsupported mixed left multi-join shape should produce facts");
    assert_rejection_facts_are_family_free(&facts, "mixed_left_multi_join");
    assert_eq!(facts.get("bounded_join_lower_level_blockers").map(String::as_str), Some(""));
    assert_eq!(facts.get("bounded_join_rejection_scope").map(String::as_str), Some("subgraph"));
    assert_eq!(facts.get("bounded_join_rejection_category").map(String::as_str), Some("missing_subgraph_invariant"));
    assert!(facts.contains_key("bounded_join_rejection_reason"));
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
    assert!(!plan.capability_analysis().admits_root_join());
    assert!(!plan.capability_analysis().admits_single_join_post_root_pipeline());
    assert!(!plan.capability_analysis().admits_closed_multi_join_subgraph());
}

#[test]
fn bounded_multi_join_reports_left_full_blocker_after_valid_mixed_residual_semi_anti() {
    for (valid_join_type, invalid_join_type) in
        [(JoinType::LeftSemi, JoinType::Left), (JoinType::LeftAnti, JoinType::Full)]
    {
        let plan =
            valid_mixed_residual_semi_anti_then_invalid_mixed_residual_join_plan(valid_join_type, invalid_join_type);

        assert!(!plan.capability_analysis().admits_root_join());
        assert!(!plan.capability_analysis().admits_single_join_post_root_pipeline());
        assert!(!plan.capability_analysis().admits_closed_multi_join_subgraph());
        let facts = unsupported_facts(&plan, ExecutionBoundaryPolicy::default())
            .expect("unsupported mixed left/full multi-join shape should produce facts");
        assert_rejection_facts_are_family_free(&facts, "mixed_left_full_multi_join");
        assert_eq!(facts.get("bounded_join_lower_level_blockers").map(String::as_str), Some(""));
        assert_eq!(facts.get("bounded_join_rejection_scope").map(String::as_str), Some("subgraph"));
        assert_eq!(
            facts.get("bounded_join_rejection_category").map(String::as_str),
            Some("missing_subgraph_invariant")
        );
        assert!(facts.contains_key("bounded_join_rejection_reason"));
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

        assert!(!plan.capability_analysis().admits_root_join());
        assert!(!plan.capability_analysis().admits_single_join_post_root_pipeline());
        assert!(!plan.capability_analysis().admits_closed_multi_join_subgraph());
        let facts = unsupported_facts(&plan, ExecutionBoundaryPolicy::default())
            .expect("unsupported mixed semi/anti multi-join shape should produce facts");
        assert_rejection_facts_are_family_free(&facts, "invalid_mixed_semi_anti_multi_join");
        assert_eq!(facts.get("bounded_join_lower_level_blockers").map(String::as_str), Some(""));
        assert!(
            matches!(facts.get("bounded_join_rejection_scope").map(String::as_str), Some("node" | "subgraph")),
            "invalid mixed semi/anti joins should report the first local node or subgraph blocker"
        );
        assert!(
            matches!(
                facts.get("bounded_join_rejection_category").map(String::as_str),
                Some("unsupported_node_operator" | "missing_subgraph_invariant")
            ),
            "invalid mixed semi/anti joins should not require a family-shaped blocker category"
        );
        assert!(
            facts
                .get("bounded_join_rejection_reason")
                .is_some_and(|reason| reason == expected_blocker || reason.contains("join"))
        );
        assert_eq!(facts.get("bounded_join_required_shape_contract"), None);
        assert_eq!(facts.get("bounded_join_required_lower_level_contract").map(String::as_str), None);
        assert!(facts.iter().all(|(key, value)| !key.contains("estimate") && !value.contains("estimate")));
    }
}

#[test]
fn bounded_multi_join_keeps_aggregate_inputs_out_of_join_subtrees() {
    let plan = mixed_residual_multi_join_with_aggregate_leaf_plan();

    assert!(!plan.capability_analysis().admits_root_join());
    assert!(!plan.capability_analysis().admits_single_join_post_root_pipeline());
    assert!(!plan.capability_analysis().admits_closed_multi_join_subgraph());

    for aggregate_leaf in
        [MultiJoinAggregateLeaf::FirstLeft, MultiJoinAggregateLeaf::FirstRight, MultiJoinAggregateLeaf::SecondRight]
    {
        let plan = hash_multi_join_with_aggregate_leaf_plan(aggregate_leaf);

        assert!(!plan.capability_analysis().admits_root_join());
        assert!(!plan.capability_analysis().admits_single_join_post_root_pipeline());
        assert!(
            !plan.capability_analysis().admits_closed_multi_join_subgraph(),
            "aggregate {aggregate_leaf:?} leaf must not be accepted below a bounded multi-join"
        );
        assert!(unsupported_facts(&plan, ExecutionBoundaryPolicy::default()).is_some());
    }
}

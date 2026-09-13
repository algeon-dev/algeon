use super::*;

#[path = "mixed_residual/builders.rs"]
mod builders;
#[path = "mixed_residual/root.rs"]
mod root;

use builders::*;

#[test]
fn bounded_multi_join_admits_mixed_keyed_residual_inner_left_left_semi_and_left_anti() {
    // A pure multi-join pipeline is admitted through the compiled task
    // contracts once every join group compiles; the retired closed-multi-join
    // envelope no longer vetoes join types the join kernels execute directly.
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
        for build_side in [JoinSide::Right, JoinSide::Left] {
            let plan = mixed_residual_multi_join_plan(join_type, build_side, projection.clone());

            assert!(!plan.capability_analysis(rapids_analysis_context()).admits_root_join());
            assert!(unsupported_facts(&plan, rapids_analysis_context()).is_none());
        }
    }

    // A mixed Left join with a right build is admitted the same way.
    let plan = mixed_residual_multi_join_plan(
        JoinType::Left,
        JoinSide::Right,
        vec![
            JoinOutputColumn::left(ColumnRef::input(0, "left_payload", "left_payload"), "left_payload"),
            JoinOutputColumn::right(ColumnRef::input(1, "right_payload", "right_payload"), "right_payload"),
        ],
    );
    assert!(!plan.capability_analysis(rapids_analysis_context()).admits_root_join());
    assert!(unsupported_facts(&plan, rapids_analysis_context()).is_none());

    // A mixed Left join building on the left is not a supported kernel shape:
    // the join node capability rejects it before any pipeline compiles.
    let plan = mixed_residual_multi_join_plan(
        JoinType::Left,
        JoinSide::Left,
        vec![
            JoinOutputColumn::left(ColumnRef::input(0, "left_payload", "left_payload"), "left_payload"),
            JoinOutputColumn::right(ColumnRef::input(1, "right_payload", "right_payload"), "right_payload"),
        ],
    );
    assert!(!plan.capability_analysis(rapids_analysis_context()).admits_root_join());
    let facts =
        unsupported_facts(&plan, rapids_analysis_context()).expect("left-build mixed left should produce facts");
    assert_rejection_facts_are_family_free(&facts, "left_build_mixed_left");
    assert_eq!(
        facts.get("bounded_join_rejection_reason").map(String::as_str),
        Some("unsupported_join_node_capability")
    );

    let plan = mixed_residual_multi_join_plan(
        JoinType::Full,
        JoinSide::Right,
        vec![
            JoinOutputColumn::left(ColumnRef::input(0, "left_payload", "left_payload"), "left_payload"),
            JoinOutputColumn::right(ColumnRef::input(1, "right_payload", "right_payload"), "right_payload"),
        ],
    );
    assert!(!plan.capability_analysis(rapids_analysis_context()).admits_root_join());
}

#[test]
fn bounded_multi_join_admits_full_tail_after_valid_mixed_residual_semi_anti() {
    // The retired closed-multi-join envelope no longer vetoes the Full tail:
    // `Full` is a supported join-kernel shape, so the composed pipeline admits
    // the whole multi-join once it compiles.
    let plan = valid_mixed_residual_semi_anti_then_invalid_mixed_residual_join_plan(JoinType::LeftAnti, JoinType::Full);

    assert!(!plan.capability_analysis(rapids_analysis_context()).admits_root_join());
    assert!(unsupported_facts(&plan, rapids_analysis_context()).is_none());
}

#[test]
fn bounded_multi_join_rejects_invalid_mixed_residual_semi_anti_with_explicit_blocker_facts() {
    for (join_type, build_side, projection, expected_blocker) in [
        (
            JoinType::LeftSemi,
            JoinSide::Left,
            vec![JoinOutputColumn::right(ColumnRef::input(1, "right_payload", "right_payload"), "right_payload")],
            "multi_join_mixed_keyed_residual_semi_anti_requires_left_only_projection",
        ),
        (
            JoinType::LeftAnti,
            JoinSide::Right,
            vec![JoinOutputColumn::right(ColumnRef::input(1, "right_payload", "right_payload"), "right_payload")],
            "multi_join_mixed_keyed_residual_semi_anti_requires_left_only_projection",
        ),
    ] {
        let plan = mixed_residual_multi_join_plan(join_type, build_side, projection);

        assert!(!plan.capability_analysis(rapids_analysis_context()).admits_root_join());
        let facts = unsupported_facts(&plan, rapids_analysis_context())
            .expect("unsupported mixed semi/anti multi-join shape should produce facts");
        assert_rejection_facts_are_family_free(&facts, "invalid_mixed_semi_anti_multi_join");
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
        assert!(facts.iter().all(|(key, value)| !key.contains("estimate") && !value.contains("estimate")));
    }
}

#[test]
fn bounded_multi_join_admits_aggregate_inputs_below_join_subtrees() {
    // S3 widened the migrated scope to aggregate stages below multi-joins: a
    // grouped aggregate feeding a join input compiles to ordinary frame tasks,
    // so the retired closed-multi-join veto no longer rejects these shapes.
    let plan = mixed_residual_multi_join_with_aggregate_leaf_plan();

    assert!(!plan.capability_analysis(rapids_analysis_context()).admits_root_join());
    assert!(unsupported_facts(&plan, rapids_analysis_context()).is_none());

    for aggregate_leaf in
        [MultiJoinAggregateLeaf::FirstLeft, MultiJoinAggregateLeaf::FirstRight, MultiJoinAggregateLeaf::SecondRight]
    {
        let plan = hash_multi_join_with_aggregate_leaf_plan(aggregate_leaf);

        assert!(!plan.capability_analysis(rapids_analysis_context()).admits_root_join());
        assert!(unsupported_facts(&plan, rapids_analysis_context()).is_none(), "{aggregate_leaf:?}");
    }
}

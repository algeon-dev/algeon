use super::*;

#[path = "mixed_residual/builders.rs"]
mod builders;
#[path = "mixed_residual/root.rs"]
mod root;

use builders::*;

#[test]
fn bounded_multi_join_admits_mixed_keyed_residual_inner_left_left_semi_and_left_anti() {
    // A pure multi-join pipeline is admitted through the compiled task
    // contracts once every join group compiles.
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
            JoinType::LEFT_ANTI,
            vec![JoinOutputColumn::left(ColumnRef::input(0, "left_payload", "left_payload"), "left_payload")],
        ),
    ] {
        for build_side in [JoinSide::Right, JoinSide::Left] {
            let plan = mixed_residual_multi_join_plan(join_type.clone(), build_side, projection.clone());

            assert!(
                unsupported_facts(
                    &plan,
                    StatisticalAggregateExecutionMode::default(),
                    rapids_analysis_context(PlanningPolicy::default())
                )
                .is_none()
            );
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
    assert!(
        unsupported_facts(
            &plan,
            StatisticalAggregateExecutionMode::default(),
            rapids_analysis_context(PlanningPolicy::default())
        )
        .is_none()
    );

    // A mixed Left join building on the left is not a supported kernel shape:
    // plan validation rejects it before capability analysis runs.
    let plan = mixed_residual_multi_join_plan(
        JoinType::Left,
        JoinSide::Left,
        vec![
            JoinOutputColumn::left(ColumnRef::input(0, "left_payload", "left_payload"), "left_payload"),
            JoinOutputColumn::right(ColumnRef::input(1, "right_payload", "right_payload"), "right_payload"),
        ],
    );
    let facts = unsupported_facts(
        &plan,
        StatisticalAggregateExecutionMode::default(),
        rapids_analysis_context(PlanningPolicy::default()),
    )
    .expect("left-build mixed left should produce facts");
    assert_rejection_facts_are_family_free(&facts, "left_build_mixed_left");
    assert_eq!(
        facts.get("bounded_join_rejection_reason").map(String::as_str),
        Some("native_plan_validation_unsupported")
    );
    let blocker_facts = facts.get("bounded_join_rejection_facts").expect("rejection facts");
    assert!(blocker_facts.contains("validation_code=unsupported_operator"), "{blocker_facts}");
    assert!(blocker_facts.contains("validation_operation=query_engine.join.validate"), "{blocker_facts}");
}

#[test]
fn bounded_multi_join_admits_full_tail_after_valid_mixed_residual_semi_anti() {
    // The retired closed-multi-join envelope no longer vetoes the Full tail:
    // `Full` is a supported join-kernel shape, so the composed pipeline admits
    // the whole multi-join once it compiles.
    let plan =
        valid_mixed_residual_semi_anti_then_invalid_mixed_residual_join_plan(JoinType::LEFT_ANTI, JoinType::Full);

    assert!(
        unsupported_facts(
            &plan,
            StatisticalAggregateExecutionMode::default(),
            rapids_analysis_context(PlanningPolicy::default())
        )
        .is_none()
    );
}

#[test]
fn bounded_multi_join_refuses_mixed_residual_semi_anti_projecting_right_columns_as_malformed() {
    // A semi or anti join outputs only left rows, so a right-side output
    // column has no value: the IR is malformed, not an unsupported shape.
    for (join_type, build_side) in [(JoinType::LeftSemi, JoinSide::Left), (JoinType::LEFT_ANTI, JoinSide::Right)] {
        let plan = mixed_residual_multi_join_plan(
            join_type.clone(),
            build_side,
            vec![JoinOutputColumn::right(ColumnRef::input(1, "right_payload", "right_payload"), "right_payload")],
        );

        let error = malformed_plan_error(&plan);
        assert_eq!(error.code(), QueryErrorCode::InvalidPlan, "{join_type:?}");
        assert_eq!(error.operation(), "query_engine.join.validate", "{join_type:?}");
    }
}

#[test]
fn bounded_multi_join_admits_aggregate_inputs_below_join_subtrees() {
    // S3 widened the migrated scope to aggregate stages below multi-joins: a
    // grouped aggregate feeding a join input compiles to ordinary frame tasks,
    // so the retired closed-multi-join veto no longer rejects these shapes.
    let plan = mixed_residual_multi_join_with_aggregate_leaf_plan();

    assert!(
        unsupported_facts(
            &plan,
            StatisticalAggregateExecutionMode::default(),
            rapids_analysis_context(PlanningPolicy::default())
        )
        .is_none()
    );

    for aggregate_leaf in
        [MultiJoinAggregateLeaf::FirstLeft, MultiJoinAggregateLeaf::FirstRight, MultiJoinAggregateLeaf::SecondRight]
    {
        let plan = hash_multi_join_with_aggregate_leaf_plan(aggregate_leaf);

        assert!(
            unsupported_facts(
                &plan,
                StatisticalAggregateExecutionMode::default(),
                rapids_analysis_context(PlanningPolicy::default())
            )
            .is_none(),
            "{aggregate_leaf:?}"
        );
    }
}

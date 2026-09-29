use std::sync::Arc;

use super::*;

#[test]
fn bounded_mixed_residual_semi_anti_root_projects_only_left_columns() {
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("left_payload", DataType::Int32, false),
    ]));
    let right_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("right_payload", DataType::Int32, false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![Field::new("left_payload", DataType::Int32, false)]));
    let plan = mixed_residual_join_plan(
        Arc::clone(&left_schema),
        Arc::clone(&right_schema),
        Arc::clone(&output_schema),
        JoinType::LeftSemi,
        JoinSide::Right,
        vec![JoinOutputColumn::left(ColumnRef::input(0, "left_payload", "left_payload"), "left_payload")],
    );

    assert!(
        unsupported_facts(
            &plan,
            StatisticalAggregateExecutionMode::default(),
            rapids_analysis_context(PlanningPolicy::default())
        )
        .is_none()
    );

    // Projecting a right column from a semi join is malformed IR.
    let right_projection_schema = Arc::new(Schema::new(vec![Field::new("right_payload", DataType::Int32, false)]));
    let plan = mixed_residual_join_plan(
        Arc::clone(&left_schema),
        Arc::clone(&right_schema),
        right_projection_schema,
        JoinType::LeftSemi,
        JoinSide::Right,
        vec![JoinOutputColumn::right(ColumnRef::input(1, "right_payload", "right_payload"), "right_payload")],
    );
    let error = malformed_plan_error(&plan);
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.operation(), "query_engine.join.validate");

    let plan = mixed_residual_join_plan(
        left_schema,
        right_schema,
        output_schema,
        JoinType::LEFT_ANTI,
        JoinSide::Left,
        vec![JoinOutputColumn::left(ColumnRef::input(0, "left_payload", "left_payload"), "left_payload")],
    );
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
fn bounded_keyless_conditional_left_semi_join_admits_left_reservation_orientation() {
    let left_schema = Arc::new(Schema::new(vec![Field::new("left_payload", DataType::Int32, false)]));
    let right_schema = Arc::new(Schema::new(vec![Field::new("right_payload", DataType::Int32, false)]));
    let output_schema = Arc::new(Schema::new(vec![Field::new("left_payload", DataType::Int32, false)]));

    let plan = keyless_conditional_left_semi_plan(left_schema, right_schema, output_schema, JoinSide::Left);
    assert!(
        unsupported_facts(
            &plan,
            StatisticalAggregateExecutionMode::default(),
            rapids_analysis_context(PlanningPolicy::default())
        )
        .is_none(),
        "keyless conditional left semi joins use the left side as the explicit reservation orientation"
    );
}

#[test]
fn bounded_mixed_residual_full_root_supports_pair_projection() {
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("left_payload", DataType::Int32, false),
    ]));
    let right_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("right_payload", DataType::Int32, false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("left_payload", DataType::Int32, true),
        Field::new("right_payload", DataType::Int32, true),
    ]));
    let plan = mixed_residual_join_plan(
        left_schema,
        right_schema,
        output_schema,
        JoinType::Full,
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
}

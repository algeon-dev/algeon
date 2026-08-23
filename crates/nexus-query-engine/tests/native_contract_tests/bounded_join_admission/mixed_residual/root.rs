use std::sync::Arc;

use super::*;

#[test]
fn bounded_mixed_residual_semi_anti_root_support_requires_left_only_right_build_orientation() {
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

    assert!(plan.capability_analysis().admits_root_join());
    assert!(unsupported_facts(&plan, ExecutionBoundaryPolicy::default()).is_none());

    let right_projection_schema = Arc::new(Schema::new(vec![Field::new("right_payload", DataType::Int32, false)]));
    let plan = mixed_residual_join_plan(
        Arc::clone(&left_schema),
        Arc::clone(&right_schema),
        right_projection_schema,
        JoinType::LeftSemi,
        JoinSide::Right,
        vec![JoinOutputColumn::right(ColumnRef::input(1, "right_payload", "right_payload"), "right_payload")],
    );
    assert!(!plan.capability_analysis().admits_root_join());
    let facts = unsupported_facts(&plan, ExecutionBoundaryPolicy::default())
        .expect("unsupported mixed semi root shape should produce facts");
    assert_rejection_facts_are_family_free(&facts, "mixed_semi_root_right_projection");
    assert_eq!(facts.get("bounded_join_lower_level_blockers").map(String::as_str), Some(""));
    assert_eq!(
        facts.get("bounded_join_rejection_reason").map(String::as_str),
        Some("unsupported_join_node_capability")
    );
    assert_eq!(facts.get("bounded_join_rejection_scope").map(String::as_str), Some("node"));
    assert_eq!(facts.get("bounded_join_rejection_category").map(String::as_str), Some("unsupported_node_operator"));
    assert_eq!(facts.get("bounded_join_required_shape_contract"), None);
    assert_eq!(facts.get("bounded_join_required_lower_level_contract").map(String::as_str), None);

    let plan = mixed_residual_join_plan(
        left_schema,
        right_schema,
        output_schema,
        JoinType::LeftAnti,
        JoinSide::Left,
        vec![JoinOutputColumn::left(ColumnRef::input(0, "left_payload", "left_payload"), "left_payload")],
    );
    assert!(!plan.capability_analysis().admits_root_join());
    let facts = unsupported_facts(&plan, ExecutionBoundaryPolicy::default())
        .expect("unsupported mixed anti root shape should produce facts");
    assert_rejection_facts_are_family_free(&facts, "mixed_anti_root_left_build");
    assert_eq!(facts.get("bounded_join_lower_level_blockers").map(String::as_str), Some(""));
    assert_eq!(
        facts.get("bounded_join_rejection_reason").map(String::as_str),
        Some("unsupported_join_node_capability")
    );
    assert_eq!(facts.get("bounded_join_rejection_scope").map(String::as_str), Some("node"));
    assert_eq!(facts.get("bounded_join_rejection_category").map(String::as_str), Some("unsupported_node_operator"));
    assert_eq!(facts.get("bounded_join_required_lower_level_contract").map(String::as_str), None);
}

#[test]
fn bounded_keyless_conditional_left_only_joins_require_left_reservation_orientation() {
    let left_schema = Arc::new(Schema::new(vec![Field::new("left_payload", DataType::Int32, false)]));
    let right_schema = Arc::new(Schema::new(vec![Field::new("right_payload", DataType::Int32, false)]));
    let output_schema = Arc::new(Schema::new(vec![Field::new("left_payload", DataType::Int32, false)]));

    let plan = keyless_conditional_left_semi_plan(
        Arc::clone(&left_schema),
        Arc::clone(&right_schema),
        Arc::clone(&output_schema),
        JoinSide::Right,
    );
    assert!(
        !plan.capability_analysis().admits_root_join(),
        "default right build-side orientation must not be accepted for keyless conditional left semi joins"
    );

    let plan = keyless_conditional_left_semi_plan(left_schema, right_schema, output_schema, JoinSide::Left);
    assert!(
        plan.capability_analysis().admits_root_join(),
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

    assert!(plan.capability_analysis().admits_root_join());
    assert!(unsupported_facts(&plan, ExecutionBoundaryPolicy::default()).is_none());
}

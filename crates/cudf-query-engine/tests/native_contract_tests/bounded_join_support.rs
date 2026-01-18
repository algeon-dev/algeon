use super::*;

#[path = "bounded_join_support/mixed_residual.rs"]
mod mixed_residual;
#[path = "bounded_join_support/plans.rs"]
mod plans;
#[path = "bounded_join_support/q11_aggregate_subplans.rs"]
mod q11_aggregate_subplans;
#[path = "bounded_join_support/q17_resident.rs"]
mod q17_resident;
#[path = "bounded_join_support/reservation_contract.rs"]
mod reservation_contract;
use plans::*;

#[test]
fn bounded_single_join_pipeline_admits_q13_shaped_serial_aggregate_spine() {
    let plan = q13_shaped_serial_aggregate_spine_plan();

    assert!(!plan.has_bounded_executable_root_join());
    assert!(plan.has_bounded_executable_join_pipeline());
    assert!(!plan.has_bounded_executable_multi_join_pipeline());
    assert!(plan.bounded_join_unsupported_shape_facts(ExecutionBoundaryPolicy::default()).is_none());
}

#[test]
fn bounded_multi_join_pipeline_admits_serial_aggregate_spine_with_projection_and_sort() {
    let plan = q16_shaped_serial_aggregate_multi_join_spine_plan();

    assert!(!plan.has_bounded_executable_root_join());
    assert!(!plan.has_bounded_executable_join_pipeline());
    assert!(plan.has_bounded_executable_multi_join_pipeline());
    assert!(plan.bounded_join_unsupported_shape_facts(ExecutionBoundaryPolicy::default()).is_none());
}

#[test]
fn bounded_single_join_pipeline_rejects_aggregate_join_input_leaves() {
    for aggregate_side in [JoinSide::Left, JoinSide::Right] {
        let plan = single_join_with_aggregate_input_plan(aggregate_side);

        assert!(!plan.has_bounded_executable_root_join());
        assert!(!plan.has_bounded_executable_join_pipeline());
        assert!(!plan.has_bounded_executable_multi_join_pipeline());
        assert!(
            plan.bounded_join_unsupported_shape_facts(ExecutionBoundaryPolicy::default()).is_some(),
            "aggregate {aggregate_side} join input must not be accepted as a bounded root join leaf"
        );
    }
}

#[test]
fn bounded_keyed_hash_join_resident_subplan_pipeline_reports_lower_level_blockers() {
    let plan =
        keyed_hash_join_resident_subplan_pipeline_plan(KeyedHashJoinResidentSubplanPipelineVariant::BlockedFamily);

    assert!(!plan.has_bounded_executable_root_join());
    assert!(!plan.has_bounded_executable_join_pipeline());
    assert!(!plan.has_bounded_executable_keyless_conditional_join_input_subplan_pipeline());
    assert!(!plan.has_bounded_executable_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline());
    assert!(!plan.has_bounded_executable_multi_join_pipeline());
    let facts = plan
        .bounded_join_unsupported_shape_facts(ExecutionBoundaryPolicy::default())
        .expect("blocked keyed hash join resident subplan pipeline should produce unsupported-shape facts");
    assert_eq!(facts.get("bounded_join_execution_status").map(String::as_str), Some("unsupported_plan_shape"));
    assert_eq!(
        facts.get("bounded_join_blocked_join_family").map(String::as_str),
        Some("keyed_hash_join_resident_subplan_pipeline")
    );
    assert_eq!(
        facts.get("bounded_join_shape_blocker").map(String::as_str),
        Some("bounded_keyed_hash_join_resident_subplan_lower_level_contracts")
    );
    assert_eq!(
        facts.get("bounded_join_lower_level_blockers").map(String::as_str),
        Some("source_contract,aggregate_contract,final_sort_contract")
    );
    assert_eq!(facts.get("blocked_reservations").map(String::as_str), Some("source_read,operator_output,final_sort"));
    assert!(facts.get("bounded_join_reservation_statuses").is_some_and(|value| {
        value.contains("source_read:blocked_by_lower_level")
            && value.contains("operator_output:blocked_by_lower_level")
            && value.contains("final_sort:blocked_by_lower_level")
    }));
    let reservation_requirements = facts
        .get("bounded_join_reservation_requirements")
        .expect("q15-like blocked family should report overridden reservation requirements");
    assert!(reservation_requirements.contains("source_read:source_contract:blocked_by_lower_level"));
    assert!(reservation_requirements.contains("operator_output:aggregate_contract:blocked_by_lower_level"));
    assert!(reservation_requirements.contains("final_sort:final_sort_contract:blocked_by_lower_level"));
    assert!(
        !reservation_requirements
            .contains("source_read:query_engine_source_read_reservation:modeled_before_source_handoff")
    );
    assert!(
        !reservation_requirements.contains(
            "operator_output:query_engine_unary_output_reservation:modeled_by_bounded_resident_frame_ownership"
        )
    );
    assert!(
        !reservation_requirements
            .contains("final_sort:query_engine_final_sort_reservation:modeled_by_bounded_resident_frame_ownership")
    );
    assert_eq!(
        facts.get("bounded_join_required_lower_level_contract").map(String::as_str),
        Some("source_contract,aggregate_contract,final_sort_contract")
    );
    assert!(facts.get("bounded_join_required_lower_level_contract").is_some_and(|value| {
        value.contains("source_contract")
            && value.contains("aggregate_contract")
            && value.contains("final_sort_contract")
    }));
}

#[test]
fn bounded_keyed_hash_join_resident_subplan_pipeline_recognizer_stays_narrow() {
    for variant in [
        KeyedHashJoinResidentSubplanPipelineVariant::ResidualRootJoin,
        KeyedHashJoinResidentSubplanPipelineVariant::SortBelowRoot,
        KeyedHashJoinResidentSubplanPipelineVariant::ScalarAggregateNotMax,
        KeyedHashJoinResidentSubplanPipelineVariant::GroupedAggregateNotSum,
    ] {
        let plan = keyed_hash_join_resident_subplan_pipeline_plan(variant);

        assert!(!plan.has_bounded_executable_root_join(), "{variant:?}");
        assert!(!plan.has_bounded_executable_join_pipeline(), "{variant:?}");
        assert!(!plan.has_bounded_executable_multi_join_pipeline(), "{variant:?}");
        let facts = plan
            .bounded_join_unsupported_shape_facts(ExecutionBoundaryPolicy::default())
            .expect("unsupported keyed hash join resident subplan variant should produce facts");
        assert_ne!(
            facts.get("bounded_join_blocked_join_family").map(String::as_str),
            Some("keyed_hash_join_resident_subplan_pipeline"),
            "{variant:?} must not be classified as the blocked q15-like family"
        );
        assert_eq!(facts.get("bounded_join_lower_level_blockers").map(String::as_str), Some(""));
    }
}

#[test]
fn bounded_remaining_tpch_topologies_report_structural_blockers_without_admission() {
    for (plan, expected_family, expected_blocker) in [
        (
            q02_shaped_aggregate_over_join_subplan_topk_plan(),
            "multi_join_with_aggregate_over_join_subplan_topk",
            "bounded_multi_join_aggregate_over_join_subplan_topk_executor_missing",
        ),
        (
            q18_shaped_grouped_aggregate_input_post_join_aggregate_topk_plan(),
            "hash_join_with_resident_grouped_aggregate_input_post_join_aggregate_topk",
            "bounded_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_executor_missing",
        ),
        (
            q20_shaped_grouped_aggregate_subquery_filter_plan(),
            "multi_join_with_resident_grouped_aggregate_subquery_filter",
            "bounded_multi_join_resident_grouped_aggregate_subquery_filter_executor_missing",
        ),
    ] {
        assert!(!plan.has_bounded_executable_root_join(), "{expected_family}");
        assert!(!plan.has_bounded_executable_join_pipeline(), "{expected_family}");
        assert!(!plan.has_bounded_executable_keyless_conditional_join_input_subplan_pipeline(), "{expected_family}");
        assert!(
            !plan.has_bounded_executable_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline(),
            "{expected_family}"
        );
        assert!(!plan.has_bounded_executable_multi_join_pipeline(), "{expected_family}");
        assert!(
            !plan.has_bounded_executable_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline(),
            "{expected_family}"
        );
        let facts = plan
            .bounded_join_unsupported_shape_facts(ExecutionBoundaryPolicy::default())
            .expect("remaining TPC-H topology should produce unsupported-shape facts");
        assert_eq!(facts.get("bounded_join_execution_status").map(String::as_str), Some("unsupported_plan_shape"));
        assert_eq!(facts.get("bounded_join_blocked_join_family").map(String::as_str), Some(expected_family));
        assert_eq!(facts.get("bounded_join_shape_blocker").map(String::as_str), Some(expected_blocker));
        assert_eq!(facts.get("bounded_join_specific_blocker").map(String::as_str), Some(expected_blocker));
        assert_eq!(facts.get("bounded_join_lower_level_blockers").map(String::as_str), Some(""));
        assert_eq!(facts.get("blocked_reservations").map(String::as_str), Some(""));
        assert_eq!(facts.get("missing_reservations").map(String::as_str), Some(""));
        assert_eq!(facts.get("bounded_join_required_lower_level_contract").map(String::as_str), None);
    }
}

#[test]
fn bounded_keyless_conditional_join_input_subplan_pipeline_admits_only_q22_bounded_shapes() {
    let plan = q22_keyless_conditional_join_input_subplan_plan(Q22InputSubplanVariant::Admitted);

    assert!(!plan.has_bounded_executable_root_join());
    assert!(!plan.has_bounded_executable_join_pipeline());
    assert!(plan.has_bounded_executable_keyless_conditional_join_input_subplan_pipeline());
    assert!(!plan.has_bounded_executable_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline());
    assert!(!plan.has_bounded_executable_multi_join_pipeline());
    assert!(plan.bounded_join_unsupported_shape_facts(ExecutionBoundaryPolicy::default()).is_none());

    for variant in [
        Q22InputSubplanVariant::GroupedAggregate,
        Q22InputSubplanVariant::SortInput,
        Q22InputSubplanVariant::LimitInput,
        Q22InputSubplanVariant::BareLeftAntiInput,
        Q22InputSubplanVariant::InnerJoinInput,
        Q22InputSubplanVariant::NestedJoinInput,
    ] {
        let plan = q22_keyless_conditional_join_input_subplan_plan(variant);

        assert!(!plan.has_bounded_executable_root_join(), "{variant:?}");
        assert!(!plan.has_bounded_executable_join_pipeline(), "{variant:?}");
        assert!(!plan.has_bounded_executable_keyless_conditional_join_input_subplan_pipeline(), "{variant:?}");
        assert!(
            !plan.has_bounded_executable_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline(),
            "{variant:?}"
        );
        assert!(!plan.has_bounded_executable_multi_join_pipeline(), "{variant:?}");
        let facts = plan
            .bounded_join_unsupported_shape_facts(ExecutionBoundaryPolicy::default())
            .expect("unsupported Q22-like subplan shape should produce facts");
        assert_eq!(
            facts.get("bounded_join_keyless_conditional_join_input_subplan_pipeline_capability").map(String::as_str),
            Some("keyless_conditional_join_input_subplan_pipeline")
        );
        assert_eq!(facts.get("bounded_join_lower_level_blockers").map(String::as_str), Some(""));
    }
}

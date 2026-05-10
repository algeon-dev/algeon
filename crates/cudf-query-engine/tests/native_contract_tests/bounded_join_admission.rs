use super::*;

#[path = "bounded_join_admission/aggregate_over_join_subplans.rs"]
mod aggregate_over_join_subplans;
#[path = "bounded_join_admission/mixed_residual.rs"]
mod mixed_residual;
#[path = "bounded_join_admission/plans.rs"]
mod plans;
#[path = "bounded_join_admission/reservation_contract.rs"]
mod reservation_contract;
#[path = "bounded_join_admission/residual_resident_grouped_aggregate.rs"]
mod residual_resident_grouped_aggregate;
use plans::*;

#[test]
fn bounded_single_join_pipeline_admits_serial_aggregate_spine() {
    let plan = serial_aggregate_single_join_spine_plan();

    assert!(!plan.has_bounded_executable_root_join());
    assert!(plan.has_bounded_executable_join_pipeline());
    assert!(!plan.has_bounded_executable_multi_join_pipeline());
    assert!(plan.bounded_join_unsupported_shape_facts(ExecutionBoundaryPolicy::default()).is_none());
}

#[test]
fn bounded_multi_join_pipeline_admits_serial_aggregate_spine_with_projection_and_sort() {
    let plan = serial_aggregate_multi_join_spine_plan();

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
fn bounded_keyed_hash_join_resident_subplan_pipeline_is_executable_with_modeled_contracts() {
    let plan =
        keyed_hash_join_resident_subplan_pipeline_plan(KeyedHashJoinResidentSubplanPipelineVariant::BlockedFamily);

    assert!(!plan.has_bounded_executable_root_join());
    assert!(!plan.has_bounded_executable_join_pipeline());
    assert!(!plan.has_bounded_executable_keyless_conditional_join_input_subplan_pipeline());
    assert!(!plan.has_bounded_executable_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline());
    assert!(!plan.has_bounded_executable_multi_join_pipeline());
    assert!(!plan.has_bounded_executable_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline());
    assert!(plan.has_bounded_executable_keyed_hash_join_resident_subplan_pipeline());
    assert!(plan.bounded_join_unsupported_shape_facts(ExecutionBoundaryPolicy::default()).is_none());
    let facts = plan
        .bounded_join_modeled_facts(ExecutionBoundaryPolicy::default())
        .expect("keyed hash join resident subplan pipeline should produce modeled bounded facts");
    assert_eq!(facts.get("bounded_join_execution_status").map(String::as_str), Some("enabled"));
    assert_eq!(
        facts.get("bounded_join_execution_family").map(String::as_str),
        Some("keyed_hash_join_resident_subplan_pipeline")
    );
    assert_eq!(facts.get("bounded_join_lower_level_blockers").map(String::as_str), Some(""));
    assert_eq!(facts.get("blocked_reservations").map(String::as_str), Some(""));
    assert_eq!(facts.get("missing_reservations").map(String::as_str), Some(""));
    assert!(facts.get("bounded_join_reservation_statuses").is_some_and(|value| {
        value.contains("source_read:modeled")
            && value.contains("operator_output:modeled")
            && value.contains("final_sort:modeled")
    }));
    let reservation_requirements = facts
        .get("bounded_join_reservation_requirements")
        .expect("resident-revenue blocked family should report shape-local reservation requirements");
    assert!(
        reservation_requirements
            .contains("source_read:source_contract:modeled_by_cudf_rs_hard_budgeted_report_with_unresolved_preflight")
    );
    assert!(reservation_requirements.contains(
        "operator_output:aggregate_contract:modeled_by_cudf_rs_hard_budgeted_report_with_unresolved_preflight"
    ));
    assert!(
        reservation_requirements.contains(
            "final_sort:final_sort_contract:modeled_by_cudf_rs_hard_budgeted_report_with_unresolved_preflight"
        )
    );
    assert!(
        facts.get("bounded_join_contract_modes").is_some_and(|value| value
            == "source_contract:hard_budgeted_report,aggregate_contract:hard_budgeted_report,final_sort_contract:hard_budgeted_report")
    );
    assert!(facts.get("bounded_join_unresolved_preflight_contracts").is_some_and(|value| {
        value.contains("source_contract:libcudf_parquet_decoded_output_preflight_unavailable")
            && value.contains("aggregate_contract:libcudf_groupby_aggregate_temporary_workspace_preflight_unavailable")
            && value.contains("final_sort_contract:libcudf_final_sort_order_preflight_unavailable")
    }));
    assert!(
        facts.get("bounded_join_contract_report_fields").is_some_and(|value| {
            value.contains("source_contract:allocator_high_water_bytes+retained_bytes+output_bytes+output_rows+context_facts")
                && value.contains("aggregate_contract:allocator_high_water_bytes+retained_bytes+output_bytes+output_rows")
                && value.contains(
                    "final_sort_contract:allocator_high_water_bytes+retained_sort_order_bytes+materialized_output_bytes+output_rows"
                )
        })
    );
    assert_eq!(facts.get("bounded_join_required_lower_level_contract").map(String::as_str), None);
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
        assert!(!plan.has_bounded_executable_keyed_hash_join_resident_subplan_pipeline(), "{variant:?}");
        let facts = plan
            .bounded_join_unsupported_shape_facts(ExecutionBoundaryPolicy::default())
            .expect("unsupported keyed hash join resident subplan variant should produce facts");
        assert_ne!(
            facts.get("bounded_join_blocked_join_family").map(String::as_str),
            Some("keyed_hash_join_resident_subplan_pipeline"),
            "{variant:?} must not be classified as the blocked resident-revenue family"
        );
        assert_eq!(facts.get("bounded_join_lower_level_blockers").map(String::as_str), Some(""));
    }
}

#[test]
fn bounded_remaining_tpch_topologies_report_structural_blockers_without_admission() {
    for (plan, expected_family, expected_blocker) in [
        (
            aggregate_over_join_subplan_topk_plan(),
            "multi_join_with_aggregate_over_join_subplan_topk",
            "bounded_multi_join_aggregate_over_join_subplan_topk_executor_missing",
        ),
        (
            resident_grouped_aggregate_input_post_join_aggregate_topk_plan(),
            "hash_join_with_resident_grouped_aggregate_input_post_join_aggregate_topk",
            "bounded_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_executor_missing",
        ),
        (
            resident_grouped_aggregate_subquery_filter_plan(),
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
fn bounded_keyless_conditional_join_input_subplan_pipeline_admits_only_keyless_input_subplan_bounded_shapes() {
    let plan = keyless_conditional_join_input_subplan_plan(KeylessInputSubplanVariant::Admitted);

    assert!(!plan.has_bounded_executable_root_join());
    assert!(!plan.has_bounded_executable_join_pipeline());
    assert!(plan.has_bounded_executable_keyless_conditional_join_input_subplan_pipeline());
    assert!(!plan.has_bounded_executable_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline());
    assert!(!plan.has_bounded_executable_multi_join_pipeline());
    assert!(plan.bounded_join_unsupported_shape_facts(ExecutionBoundaryPolicy::default()).is_none());

    for variant in [
        KeylessInputSubplanVariant::GroupedAggregate,
        KeylessInputSubplanVariant::SortInput,
        KeylessInputSubplanVariant::LimitInput,
        KeylessInputSubplanVariant::BareLeftAntiInput,
        KeylessInputSubplanVariant::InnerJoinInput,
        KeylessInputSubplanVariant::NestedJoinInput,
    ] {
        let plan = keyless_conditional_join_input_subplan_plan(variant);

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
            .expect("unsupported keyless input-subplan subplan shape should produce facts");
        assert_eq!(
            facts.get("bounded_join_keyless_conditional_join_input_subplan_pipeline_capability").map(String::as_str),
            Some("keyless_conditional_join_input_subplan_pipeline")
        );
        assert_eq!(facts.get("bounded_join_lower_level_blockers").map(String::as_str), Some(""));
    }
}

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

fn assert_modeled_family_reservation_facts_are_consistent(
    facts: &std::collections::BTreeMap<&'static str, String>,
    label: &str,
) {
    let kinds = csv_values(facts, "bounded_join_reservation_kinds", label);
    let statuses = csv_pairs(facts, "bounded_join_reservation_statuses", label);
    let requirements = csv_triples(facts, "bounded_join_reservation_requirements", label);
    let modeled = csv_values(facts, "modeled_reservations", label);
    let blocked = csv_values(facts, "blocked_reservations", label);
    let missing = csv_values(facts, "missing_reservations", label);

    let status_kinds = statuses.iter().map(|(kind, _)| *kind).collect::<Vec<_>>();
    let requirement_kinds = requirements.iter().map(|(kind, _, _)| *kind).collect::<Vec<_>>();
    assert_eq!(status_kinds, kinds, "{label}: reservation statuses must cover every reservation kind");
    assert_eq!(requirement_kinds, kinds, "{label}: reservation requirements must cover every reservation kind");

    let expected_modeled =
        statuses.iter().filter_map(|(kind, status)| (*status == "modeled").then_some(*kind)).collect::<Vec<_>>();
    let expected_blocked =
        statuses.iter().filter_map(|(kind, status)| (*status != "modeled").then_some(*kind)).collect::<Vec<_>>();
    assert_eq!(modeled, expected_modeled, "{label}: modeled_reservations must match modeled statuses");
    assert_eq!(blocked, expected_blocked, "{label}: blocked_reservations must match non-modeled statuses");
    assert_eq!(missing, expected_blocked, "{label}: missing_reservations must match non-modeled statuses");

    let contract_modes = csv_pairs(facts, "bounded_join_contract_modes", label);
    let unresolved_preflight = csv_pairs(facts, "bounded_join_unresolved_preflight_contracts", label);
    let report_fields = csv_pairs(facts, "bounded_join_contract_report_fields", label);
    for (contract, mode) in &contract_modes {
        if *mode == "hard_budgeted_report" {
            assert!(
                unresolved_preflight.iter().any(|(name, _)| name == contract),
                "{label}: hard-budgeted contract {contract} must name its unresolved preflight blocker"
            );
            assert!(
                report_fields.iter().any(|(name, _)| name == contract),
                "{label}: hard-budgeted contract {contract} must name report fields"
            );
        }
    }

    for (kind, contract, detail) in requirements {
        if detail.contains("hard_budgeted_report_with_unresolved_preflight") {
            assert!(
                contract_modes.iter().any(|(name, mode)| *name == contract && *mode == "hard_budgeted_report"),
                "{label}: {kind} requirement uses hard-budgeted detail but {contract} is absent from contract modes"
            );
            assert!(
                unresolved_preflight
                    .iter()
                    .any(|(name, blocker)| { *name == contract && blocker.ends_with("_preflight_unavailable") }),
                "{label}: {kind} requirement uses unresolved preflight but {contract} is absent from blockers"
            );
        }
    }
}

fn csv_values<'a>(facts: &'a std::collections::BTreeMap<&'static str, String>, key: &str, label: &str) -> Vec<&'a str> {
    facts
        .get(key)
        .unwrap_or_else(|| panic!("{label}: missing {key}"))
        .split(',')
        .filter(|value| !value.is_empty())
        .collect()
}

fn csv_pairs<'a>(
    facts: &'a std::collections::BTreeMap<&'static str, String>,
    key: &str,
    label: &str,
) -> Vec<(&'a str, &'a str)> {
    csv_values(facts, key, label)
        .into_iter()
        .map(|value| value.split_once(':').unwrap_or_else(|| panic!("{label}: {key} row {value:?} must be name:value")))
        .collect()
}

fn csv_triples<'a>(
    facts: &'a std::collections::BTreeMap<&'static str, String>,
    key: &str,
    label: &str,
) -> Vec<(&'a str, &'a str, &'a str)> {
    csv_values(facts, key, label)
        .into_iter()
        .map(|value| {
            let mut fields = value.splitn(3, ':');
            let first = fields.next().unwrap_or("");
            let second = fields.next().unwrap_or("");
            let third = fields.next().unwrap_or("");
            if first.is_empty() || second.is_empty() || third.is_empty() {
                panic!("{label}: {key} row {value:?} must be name:contract:detail");
            }
            (first, second, third)
        })
        .collect()
}

#[test]
fn bounded_capability_analysis_runs_as_side_channel_for_multi_join_pipeline() {
    let plan = serial_aggregate_multi_join_spine_plan();

    assert!(plan.has_bounded_executable_multi_join_pipeline());
    let analysis = plan.capability_analysis();

    assert_eq!(analysis.nodes.len(), plan.node_count());
    assert_eq!(
        analysis.compatibility_family_projection,
        Some(CompatibilityFamilyProjection::MultiJoinTreePostRootPipeline)
    );
    assert!(analysis.nodes.iter().any(|node| matches!(node.operator, NodeOperator::Join { .. })));
    assert!(analysis.edges.iter().any(|edge| matches!(edge.input_role, InputRole::JoinBuildInput { .. })));
    assert!(
        analysis
            .invariants
            .iter()
            .any(|invariant| matches!(invariant.kind, SubgraphInvariantKind::BuildBeforeProbe { .. }))
    );
    assert!(
        analysis
            .invariants
            .iter()
            .any(|invariant| { matches!(invariant.decision, CapabilityDecision::Conditional { .. }) })
    );
    assert!(analysis.edges.iter().any(|edge| {
        matches!(edge.consumer, EdgeConsumer::Sink(ResultSinkKind::HostArrow))
            && edge.reservation_requirements.iter().any(|requirement| {
                requirement.kind == ReservationKind::Materialization
                    && requirement.status == ContractStatus::Modeled
                    && requirement.quality == ContractQuality::QueryEngineOwnedBoundary
            })
    }));
    assert!(analysis.requirements.iter().any(|requirement| {
        requirement.kind == ReservationKind::SourceRead
            && requirement.status == ContractStatus::Modeled
            && requirement.quality == ContractQuality::HardBudgetedReportWithUnresolvedPreflight
    }));
}

#[test]
fn bounded_capability_analysis_preserves_status_quality_separation() {
    let plan =
        keyed_hash_join_resident_subplan_pipeline_plan(KeyedHashJoinResidentSubplanPipelineVariant::BlockedFamily);

    assert!(plan.has_bounded_executable_keyed_hash_join_resident_subplan_pipeline());
    let analysis = plan.capability_analysis();

    assert_eq!(
        analysis.compatibility_family_projection,
        Some(CompatibilityFamilyProjection::KeyedHashJoinResidentSubplanPipeline)
    );
    assert!(analysis.requirements.iter().any(|requirement| {
        requirement.kind == ReservationKind::FinalSort
            && requirement.status == ContractStatus::Modeled
            && requirement.quality == ContractQuality::HardBudgetedReportWithUnresolvedPreflight
    }));
    assert!(analysis.requirements.iter().any(|requirement| {
        requirement.kind == ReservationKind::OperatorOutput
            && requirement.status == ContractStatus::Modeled
            && requirement.quality == ContractQuality::HardBudgetedReportWithUnresolvedPreflight
    }));
    assert!(analysis.requirements.iter().any(|requirement| {
        requirement.kind == ReservationKind::HashTable
            && requirement.status == ContractStatus::Modeled
            && requirement.quality == ContractQuality::ExactPreflight
    }));
}

#[test]
fn bounded_capability_analysis_can_reject_node_capabilities_without_family_projection() {
    let plan = unsupported_cross_key_join_plan();
    let analysis = plan.capability_analysis();

    assert_eq!(analysis.compatibility_family_projection, None);
    assert!(analysis.has_rejections());
    let rejected_join = analysis
        .nodes
        .iter()
        .find(|node| matches!(node.operator, NodeOperator::Join { join_type: JoinType::Cross, .. }))
        .expect("unsupported plan should contain a cross join capability");
    let CapabilityDecision::Rejected { reason, facts } = &rejected_join.executable else {
        panic!("unsupported cross-key join should be rejected as a node capability");
    };
    assert_eq!(*reason, "unsupported_join_node_capability");
    assert_eq!(facts.get("join_type").map(String::as_str), Some("cross"));
    assert_eq!(facts.get("key_count").map(String::as_str), Some("1"));
}

fn unsupported_cross_key_join_plan() -> cudf_query_engine::QueryPlan {
    let left_schema = Arc::new(Schema::new(vec![Field::new("l_key", DataType::Int32, false)]));
    let right_schema = Arc::new(Schema::new(vec![Field::new("r_key", DataType::Int32, false)]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("l_key", DataType::Int32, false),
        Field::new("r_key", DataType::Int32, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/unsupported-cross-left.parquet",
        Arc::clone(&left_schema),
        vec!["l_key".into()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/unsupported-cross-right.parquet",
        Arc::clone(&right_schema),
        vec!["r_key".into()],
    )));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Cross,
            vec![JoinKey::new(ColumnRef::input(0, "l_key", "l_key"), ColumnRef::input(1, "r_key", "r_key"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "l_key", "l_key"), "l_key"),
            JoinOutputColumn::right(ColumnRef::input(1, "r_key", "r_key"), "r_key"),
        ]),
        output_schema,
    );
    builder.finish_with_host_sink(join)
}

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
    assert_modeled_family_reservation_facts_are_consistent(&facts, "keyed_hash_join_resident_subplan_pipeline");
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
fn bounded_multi_join_aggregate_over_join_subplan_topk_pipeline_is_executable() {
    let plan = aggregate_over_join_subplan_topk_plan();

    assert!(!plan.has_bounded_executable_root_join());
    assert!(!plan.has_bounded_executable_join_pipeline());
    assert!(!plan.has_bounded_executable_keyless_conditional_join_input_subplan_pipeline());
    assert!(!plan.has_bounded_executable_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline());
    assert!(!plan.has_bounded_executable_multi_join_pipeline());
    assert!(plan.has_bounded_executable_multi_join_aggregate_over_join_subplan_topk_pipeline());
    assert!(!plan.has_bounded_executable_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline());
    assert!(plan.bounded_join_unsupported_shape_facts(ExecutionBoundaryPolicy::default()).is_none());

    let facts = plan
        .bounded_join_modeled_facts(ExecutionBoundaryPolicy::default())
        .expect("aggregate-over-join subplan TopK pipeline should produce modeled bounded facts");
    assert_eq!(facts.get("bounded_join_execution_status").map(String::as_str), Some("enabled"));
    assert_eq!(
        facts.get("bounded_join_execution_family").map(String::as_str),
        Some("multi_join_with_aggregate_over_join_subplan_topk")
    );
    assert_eq!(facts.get("bounded_join_lower_level_blockers").map(String::as_str), Some(""));
    assert_eq!(facts.get("blocked_reservations").map(String::as_str), Some(""));
    assert_eq!(facts.get("missing_reservations").map(String::as_str), Some(""));
    assert!(facts.get("bounded_join_reservation_statuses").is_some_and(|value| {
        value.contains("source_read:modeled")
            && value.contains("operator_output:modeled")
            && value.contains("final_sort:modeled")
    }));
    assert!(
        facts
            .get("bounded_join_contract_modes")
            .is_some_and(|value| value.contains("final_sort_contract:hard_budgeted_report"))
    );
    assert_modeled_family_reservation_facts_are_consistent(&facts, "multi_join_with_aggregate_over_join_subplan_topk");
}

#[test]
fn bounded_multi_join_aggregate_over_join_subplan_topk_pipeline_recognizer_stays_narrow() {
    for variant in [
        AggregateOverJoinSubplanTopKVariant::MissingFetchLimit,
        AggregateOverJoinSubplanTopKVariant::ScalarAggregateSubplan,
        AggregateOverJoinSubplanTopKVariant::ShallowMultiJoinSide,
    ] {
        let plan = aggregate_over_join_subplan_topk_variant_plan(variant);

        assert!(
            !plan.has_bounded_executable_multi_join_aggregate_over_join_subplan_topk_pipeline(),
            "{variant:?} must not be admitted by the aggregate-over-join TopK family"
        );
        assert!(
            plan.bounded_join_unsupported_shape_facts(ExecutionBoundaryPolicy::default()).is_some(),
            "{variant:?} should remain rejected with bounded unsupported-shape facts"
        );
    }
}

#[test]
fn bounded_multi_join_resident_grouped_aggregate_subquery_filter_pipeline_is_executable() {
    let plan = resident_grouped_aggregate_subquery_filter_plan();

    assert!(!plan.has_bounded_executable_root_join());
    assert!(!plan.has_bounded_executable_join_pipeline());
    assert!(!plan.has_bounded_executable_keyless_conditional_join_input_subplan_pipeline());
    assert!(!plan.has_bounded_executable_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline());
    assert!(!plan.has_bounded_executable_multi_join_pipeline());
    assert!(!plan.has_bounded_executable_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline());
    assert!(
        !plan.has_bounded_executable_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_pipeline()
    );
    assert!(plan.has_bounded_executable_multi_join_resident_grouped_aggregate_subquery_filter_pipeline());
    assert!(plan.bounded_join_unsupported_shape_facts(ExecutionBoundaryPolicy::default()).is_none());

    let facts = plan
        .bounded_join_modeled_facts(ExecutionBoundaryPolicy::default())
        .expect("resident grouped aggregate subquery filter pipeline should produce modeled bounded facts");
    assert_eq!(facts.get("bounded_join_execution_status").map(String::as_str), Some("enabled"));
    assert_eq!(
        facts.get("bounded_join_execution_family").map(String::as_str),
        Some("multi_join_with_resident_grouped_aggregate_subquery_filter")
    );
    assert_eq!(facts.get("bounded_join_lower_level_blockers").map(String::as_str), Some(""));
    assert_eq!(facts.get("blocked_reservations").map(String::as_str), Some(""));
    assert_eq!(facts.get("missing_reservations").map(String::as_str), Some(""));
    assert!(
        facts
            .get("bounded_join_contract_modes")
            .is_some_and(|value| value.contains("final_sort_contract:hard_budgeted_report"))
    );
    assert_modeled_family_reservation_facts_are_consistent(
        &facts,
        "multi_join_with_resident_grouped_aggregate_subquery_filter",
    );
}

#[test]
fn bounded_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_pipeline_is_executable() {
    let plan = resident_grouped_aggregate_input_post_join_aggregate_topk_plan();

    assert!(!plan.has_bounded_executable_root_join());
    assert!(!plan.has_bounded_executable_join_pipeline());
    assert!(!plan.has_bounded_executable_keyless_conditional_join_input_subplan_pipeline());
    assert!(!plan.has_bounded_executable_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline());
    assert!(!plan.has_bounded_executable_multi_join_pipeline());
    assert!(!plan.has_bounded_executable_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline());
    assert!(plan.has_bounded_executable_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_pipeline());
    assert!(plan.bounded_join_unsupported_shape_facts(ExecutionBoundaryPolicy::default()).is_none());

    let facts = plan
        .bounded_join_modeled_facts(ExecutionBoundaryPolicy::default())
        .expect("resident grouped aggregate TopK pipeline should produce modeled bounded facts");
    assert_eq!(facts.get("bounded_join_execution_status").map(String::as_str), Some("enabled"));
    assert_eq!(
        facts.get("bounded_join_execution_family").map(String::as_str),
        Some("hash_join_with_resident_grouped_aggregate_input_post_join_aggregate_topk")
    );
    assert_eq!(facts.get("bounded_join_lower_level_blockers").map(String::as_str), Some(""));
    assert_eq!(facts.get("blocked_reservations").map(String::as_str), Some(""));
    assert_eq!(facts.get("missing_reservations").map(String::as_str), Some(""));
    assert!(facts.get("bounded_join_reservation_statuses").is_some_and(|value| {
        value.contains("source_read:modeled")
            && value.contains("operator_output:modeled")
            && value.contains("final_sort:modeled")
    }));
    assert!(
        facts
            .get("bounded_join_contract_modes")
            .is_some_and(|value| value.contains("final_sort_contract:hard_budgeted_report"))
    );
    assert_modeled_family_reservation_facts_are_consistent(
        &facts,
        "hash_join_with_resident_grouped_aggregate_input_post_join_aggregate_topk",
    );
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

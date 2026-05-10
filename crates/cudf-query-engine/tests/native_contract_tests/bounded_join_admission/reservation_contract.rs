use super::*;

#[test]
fn bounded_join_reservation_kinds_are_modeled_without_estimates() {
    let contract = BoundedJoinReservationContract;
    let requirements = contract.requirements();
    let expected = [
        (
            ReservationKind::SourceRead,
            "source_read",
            "join_source_read",
            BoundedJoinReservationStatus::Modeled,
            "modeled_before_source_handoff",
        ),
        (
            ReservationKind::OperatorOutput,
            "operator_output",
            "join_operator_output",
            BoundedJoinReservationStatus::Modeled,
            "modeled_by_bounded_resident_frame_ownership",
        ),
        (
            ReservationKind::BuildSide,
            "build_side",
            "join_build_side",
            BoundedJoinReservationStatus::Modeled,
            "modeled_before_execution",
        ),
        (
            ReservationKind::HashTable,
            "hash_table",
            "join_hash_table",
            BoundedJoinReservationStatus::Modeled,
            "modeled_by_cudf_rs_pre_build_sizing",
        ),
        (
            ReservationKind::ProbeSide,
            "probe_side",
            "join_probe_side",
            BoundedJoinReservationStatus::Modeled,
            "modeled_by_ordered_bounded_join_execution_path",
        ),
        (
            ReservationKind::FilteredProbeState,
            "filtered_probe_state",
            "join_filtered_probe_state",
            BoundedJoinReservationStatus::Modeled,
            "modeled_by_cudf_rs_filtered_probe_state_observability",
        ),
        (
            ReservationKind::MixedSizeData,
            "mixed_size_data",
            "join_mixed_size_data",
            BoundedJoinReservationStatus::Modeled,
            "modeled_by_cudf_rs_retained_join_size_data_observability",
        ),
        (
            ReservationKind::GatherMap,
            "gather_map",
            "join_gather_map",
            BoundedJoinReservationStatus::Modeled,
            "modeled_by_cudf_rs_join_gather_map_sizing",
        ),
        (
            ReservationKind::JoinOutput,
            "join_output",
            "join_output",
            BoundedJoinReservationStatus::Modeled,
            "modeled_by_cudf_rs_dataframe_gather_or_projected_cross_output_sizing",
        ),
        (
            ReservationKind::FinalSort,
            "final_sort",
            "join_pipeline_final_sort",
            BoundedJoinReservationStatus::Modeled,
            "modeled_by_bounded_resident_frame_ownership",
        ),
        (
            ReservationKind::Materialization,
            "materialization",
            "host_materialization",
            BoundedJoinReservationStatus::Modeled,
            "modeled_by_explicit_sink_boundary",
        ),
    ];

    assert_eq!(requirements.len(), expected.len());
    assert_eq!(contract.model_status(), "modeled");
    assert_eq!(contract.execution_status(), "enabled");
    assert_eq!(contract.execution_families().len(), 14);
    assert_eq!(
        contract.execution_families_csv(),
        "root_reusable_hash_join,root_filtered_semi_anti_join,keyless_conditional_join,keyless_conditional_join_input_subplan_pipeline,keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline,root_projected_cross_join,root_mixed_keyed_residual_join,multi_join_reusable_hash_pipeline,multi_join_mixed_keyed_residual_inner_left_semi_left_anti_pipeline,multi_join_with_aggregate_over_join_subplan_topk,mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline,hash_join_with_resident_grouped_aggregate_input_post_join_aggregate_topk,multi_join_with_resident_grouped_aggregate_subquery_filter,keyed_hash_join_resident_subplan_pipeline"
    );
    assert_eq!(
        contract.supported_shape_requirement(),
        "requires_one_of(root_reusable_hash_join,root_filtered_semi_anti_join,keyless_conditional_join,keyless_conditional_join_input_subplan_pipeline,keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline,root_projected_cross_join,root_mixed_keyed_residual_join,multi_join_reusable_hash_pipeline,multi_join_mixed_keyed_residual_inner_left_semi_left_anti_pipeline,multi_join_with_aggregate_over_join_subplan_topk,mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline,hash_join_with_resident_grouped_aggregate_input_post_join_aggregate_topk,multi_join_with_resident_grouped_aggregate_subquery_filter,keyed_hash_join_resident_subplan_pipeline)"
    );
    assert!(contract.enabled());

    for (requirement, (kind, expected_name, expected_boundary, expected_status, expected_detail)) in
        requirements.iter().zip(expected)
    {
        assert_eq!(kind.as_str(), expected_name);
        assert_eq!(requirement.reservation_kind(), kind);
        assert_eq!(requirement.memory_boundary(), expected_boundary);
        assert_eq!(requirement.status(), expected_status);
        assert_eq!(requirement.detail(), expected_detail);
        assert!(!requirement.required_contract().contains("estimate"));
        let request = MemoryReservationRequest::new(kind, 4096, 8192).with_stage("bounded_join");
        assert_eq!(request.reservation_kind, kind);
        assert_eq!(request.memory_boundary, expected_name);
        assert_eq!(request.requested_bytes, 4096);
        assert_eq!(request.budget_bytes, 8192);
        assert_eq!(request.stage.as_deref(), Some("bounded_join"));
    }

    assert_eq!(
        contract.reservation_kinds_csv(),
        "source_read,operator_output,build_side,hash_table,probe_side,filtered_probe_state,mixed_size_data,gather_map,join_output,final_sort,materialization"
    );
    assert_eq!(contract.lower_level_blockers_csv(), "");
    assert_eq!(contract.blocking_reservation_kinds_csv(), "");
    assert_eq!(
        contract.modeled_reservation_kinds_csv(),
        "source_read,operator_output,build_side,hash_table,probe_side,filtered_probe_state,mixed_size_data,gather_map,join_output,final_sort,materialization"
    );
    assert_eq!(contract.deferred_reservation_kinds_csv(), "");
}

#[test]
fn bounded_join_family_catalogue_is_explicitly_transitional_and_topology_scoped() {
    let contract = BoundedJoinReservationContract;
    let family_contracts = contract.execution_family_contracts();

    assert_eq!(contract.admission_model(), cudf_query_engine::BOUNDED_JOIN_TRANSITIONAL_ADMISSION_MODEL);
    assert_eq!(contract.admission_scope(), cudf_query_engine::BOUNDED_JOIN_TOPOLOGY_ONLY_SCOPE);
    assert_eq!(family_contracts.len(), contract.execution_families().len());
    assert_eq!(family_contracts, cudf_query_engine::BOUNDED_JOIN_EXECUTION_FAMILY_CONTRACTS);

    for (family, family_contract) in contract.execution_families().iter().zip(family_contracts) {
        assert_eq!(*family, family_contract.family());
        assert_eq!(family_contract.admission_model(), cudf_query_engine::BOUNDED_JOIN_TRANSITIONAL_ADMISSION_MODEL);
        assert!(
            !family_contract.topology_contract().is_empty(),
            "{} must document a Phase A topology contract",
            family.as_str()
        );
        assert_topology_contract_avoids_query_specific_terms(family.as_str(), family_contract.topology_contract());
    }

    let facts = contract.facts(ExecutionBoundaryPolicy::default());
    assert_eq!(
        facts.get("bounded_join_admission_model").map(String::as_str),
        Some(cudf_query_engine::BOUNDED_JOIN_TRANSITIONAL_ADMISSION_MODEL)
    );
    assert_eq!(
        facts.get("bounded_join_admission_scope").map(String::as_str),
        Some(cudf_query_engine::BOUNDED_JOIN_TOPOLOGY_ONLY_SCOPE)
    );
    assert_eq!(
        facts.get("bounded_join_execution_family_contracts").map(String::as_str),
        Some(contract.execution_family_contracts_csv().as_str())
    );
}

#[test]
fn bounded_join_modeled_reservation_facts_match_statuses() {
    let contract = BoundedJoinReservationContract;
    assert_modeled_reservations_match_statuses(&contract.facts(ExecutionBoundaryPolicy::default()));
}

#[test]
fn bounded_join_rejection_facts_are_shared_and_estimate_free() {
    let contract = BoundedJoinReservationContract;
    let facts = contract.rejection_facts(ExecutionBoundaryPolicy::default());
    let unsupported_shape_facts = contract.unsupported_shape_facts(ExecutionBoundaryPolicy::default());

    assert_eq!(
        facts.get("boundary_policy").map(String::as_str),
        Some(
            "source_read=direct_device_local_parquet,build_materialization=query_scoped_gpu_hash_join,probe_execution=reusable_gpu_hash_join,result_materialization=explicit_sink"
        )
    );
    assert_eq!(facts.get("source_read_boundary").map(String::as_str), Some("direct_device_local_parquet"));
    assert_eq!(facts.get("build_materialization_boundary").map(String::as_str), Some("query_scoped_gpu_hash_join"));
    assert_eq!(facts.get("probe_execution_boundary").map(String::as_str), Some("reusable_gpu_hash_join"));
    assert_eq!(facts.get("result_materialization_boundary").map(String::as_str), Some("explicit_sink"));
    assert_eq!(
        unsupported_shape_facts.get("bounded_join_execution_status").map(String::as_str),
        Some("unsupported_plan_shape")
    );
    assert_eq!(
        unsupported_shape_facts.get("bounded_join_unsupported_shape").map(String::as_str),
        Some(
            "requires_one_of(root_reusable_hash_join,root_filtered_semi_anti_join,keyless_conditional_join,keyless_conditional_join_input_subplan_pipeline,keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline,root_projected_cross_join,root_mixed_keyed_residual_join,multi_join_reusable_hash_pipeline,multi_join_mixed_keyed_residual_inner_left_semi_left_anti_pipeline,multi_join_with_aggregate_over_join_subplan_topk,mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline,hash_join_with_resident_grouped_aggregate_input_post_join_aggregate_topk,multi_join_with_resident_grouped_aggregate_subquery_filter,keyed_hash_join_resident_subplan_pipeline)"
        )
    );
    assert_eq!(
        unsupported_shape_facts.get("bounded_join_execution_families").map(String::as_str),
        facts.get("bounded_join_execution_families").map(String::as_str)
    );
    assert_eq!(facts.get("bounded_join_reservation_model").map(String::as_str), Some(contract.model_status()));
    assert_eq!(facts.get("bounded_join_execution_status").map(String::as_str), Some(contract.execution_status()));
    assert_eq!(
        facts.get("bounded_join_execution_families").map(String::as_str),
        Some(contract.execution_families_csv().as_str())
    );
    assert_eq!(facts.get("bounded_join_enabled").map(String::as_str), Some("true"));
    assert_eq!(
        facts.get("bounded_join_reservation_kinds").map(String::as_str),
        Some(contract.reservation_kinds_csv().as_str())
    );
    assert_eq!(
        facts.get("bounded_join_reservation_statuses").map(String::as_str),
        Some(contract.reservation_statuses_csv().as_str())
    );
    assert_eq!(
        facts.get("blocked_reservations").map(String::as_str),
        Some(contract.blocking_reservation_kinds_csv().as_str())
    );
    assert_eq!(
        facts.get("modeled_reservations").map(String::as_str),
        Some(contract.modeled_reservation_kinds_csv().as_str())
    );
    assert_eq!(
        facts.get("deferred_reservations").map(String::as_str),
        Some(contract.deferred_reservation_kinds_csv().as_str())
    );
    assert_eq!(
        facts.get("bounded_join_lower_level_blockers").map(String::as_str),
        Some(contract.lower_level_blockers_csv().as_str())
    );
    assert_eq!(
        facts.get("bounded_join_reservation_requirements").map(String::as_str),
        Some(contract.requirement_details_csv().as_str())
    );
    assert_eq!(facts.get("bounded_join_execution_family_alias_compatibility").map(String::as_str), None);
    assert_eq!(
        facts.get("cudf_rs_hash_join_pre_build_reservation_contract").map(String::as_str),
        Some("hash_join_pre_build_reservation_size_bytes")
    );
    assert_eq!(
        facts.get("cudf_rs_hash_join_pre_build_reservation_status").map(String::as_str),
        Some("available_exact_flat_hash_join_builds")
    );
    assert_eq!(
        facts.get("cudf_rs_filtered_join_hash_table_observability").map(String::as_str),
        Some("filtered_join_device_allocated_size_bytes")
    );
    assert_eq!(
        facts.get("cudf_rs_filtered_join_output_size_observability").map(String::as_str),
        Some("filtered_join_probe_state_output_size")
    );
    assert_eq!(
        facts.get("cudf_rs_filtered_join_probe_state_contract").map(String::as_str),
        Some("filtered_join_retained_probe_state")
    );
    assert_eq!(
        facts.get("cudf_rs_filtered_join_probe_state_observability").map(String::as_str),
        Some("filtered_join_probe_device_allocated_size_bytes")
    );
    assert_eq!(
        facts.get("cudf_rs_filtered_join_probe_materialization_contract").map(String::as_str),
        Some("filtered_join_probe_state_materialize_indices")
    );
    assert_eq!(
        facts.get("cudf_rs_dataframe_gather_output_reservation_contract").map(String::as_str),
        Some("dataframe_gather_output_reservation_size_bytes")
    );
    assert_eq!(
        facts.get("cudf_rs_conditional_join_output_size_observability").map(String::as_str),
        Some("conditional_join_exact_output_size")
    );
    assert_eq!(
        facts.get("cudf_rs_conditional_join_output_size_hint").map(String::as_str),
        Some("conditional_join_indices_with_output_size")
    );
    assert_eq!(
        facts.get("cudf_rs_mixed_join_output_size_observability").map(String::as_str),
        Some("mixed_inner_left_join_size_data_output_size")
    );
    assert_eq!(
        facts.get("cudf_rs_mixed_join_size_data_observability").map(String::as_str),
        Some("mixed_inner_left_join_size_data_device_allocated_size_bytes")
    );
    assert_eq!(
        facts.get("cudf_rs_mixed_join_gather_map_reservation_contract").map(String::as_str),
        Some("mixed_inner_left_join_size_data_gather_map_reservation_size_bytes")
    );
    assert_eq!(
        facts.get("cudf_rs_mixed_join_output_size_hint").map(String::as_str),
        Some("mixed_inner_left_join_size_data_materialize_indices")
    );
    assert_eq!(
        facts.get("cudf_rs_mixed_join_size_data_alias_compatibility").map(String::as_str),
        Some(
            "mixed_inner_left_join_size_data_values_are_legacy_aliases_for_mixed_keyed_residual_inner_left_full_left_semi_left_anti_size_data"
        )
    );
    assert_eq!(facts.get("cudf_rs_mixed_join_remaining_blockers").map(String::as_str), Some(""));
    assert_eq!(
        facts.get("bounded_join_mixed_keyed_residual_capability").map(String::as_str),
        Some("root_mixed_keyed_residual_join")
    );
    assert_eq!(
        facts.get("bounded_join_mixed_keyed_residual_capability_detail").map(String::as_str),
        Some("mixed_keyed_residual_root_join_types=inner,left,full,left_semi,left_anti")
    );
    assert_eq!(
        facts.get("bounded_join_mixed_keyed_residual_multi_join_capability").map(String::as_str),
        Some("multi_join_mixed_keyed_residual_inner_left_semi_left_anti_pipeline")
    );
    assert_eq!(
        facts.get("bounded_join_mixed_keyed_residual_multi_join_capability_detail").map(String::as_str),
        Some("mixed_keyed_residual_multi_join_types=inner,left_semi,left_anti")
    );
    assert_eq!(
        facts.get("bounded_join_keyless_conditional_join_input_subplan_pipeline_capability").map(String::as_str),
        Some("keyless_conditional_join_input_subplan_pipeline")
    );
    assert_eq!(
        facts.get("bounded_join_keyless_conditional_join_input_subplan_pipeline_capability_detail").map(String::as_str),
        Some(
            "root_join=keyless_conditional_inner;input_subplans=source_filter_projection_or_projection_wrapped_left_anti_hash_join_or_scalar_global_aggregate"
        )
    );
    assert_eq!(
        facts
            .get("bounded_join_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline_capability")
            .map(String::as_str),
        Some("keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline")
    );
    assert_eq!(
        facts
            .get("bounded_join_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline_capability_detail")
            .map(String::as_str),
        Some(
            "root_join=keyless_conditional_inner;left_input=projection_wrapped_grouped_aggregate_over_bounded_join_subplan;right_input=projection_wrapped_scalar_global_aggregate_over_bounded_join_subplan;root_output=left_only;post_root=projection_sort"
        )
    );
    assert_eq!(
        facts.get("cudf_rs_mixed_join_pair_materialization_contract").map(String::as_str),
        Some("mixed_join_retained_size_data_materialize_indices")
    );
    assert_eq!(
        facts.get("cudf_rs_mixed_join_single_side_materialization_contract").map(String::as_str),
        Some("mixed_join_retained_size_data_materialize_single_indices")
    );
    assert_eq!(
        facts.get("cudf_rs_projected_cross_join_reservation_contract").map(String::as_str),
        Some("projected_cross_join_reservation_size_bytes")
    );
    assert_eq!(
        facts.get("cudf_rs_conditional_full_join_output_size_observability").map(String::as_str),
        Some("conditional_full_join_retained_size_data_output_size")
    );
    assert_eq!(
        facts.get("cudf_rs_conditional_full_join_size_data_contract").map(String::as_str),
        Some("conditional_full_join_retained_size_data")
    );
    assert_eq!(
        facts.get("cudf_rs_conditional_full_join_gather_map_reservation_contract").map(String::as_str),
        Some("conditional_full_join_size_data_gather_map_reservation_size_bytes")
    );
    assert_eq!(
        facts.get("cudf_rs_conditional_full_join_materialization_contract").map(String::as_str),
        Some("conditional_full_join_retained_size_data_materialize_indices")
    );
    assert_eq!(
        facts.get("missing_reservations").map(String::as_str),
        Some(contract.blocking_reservation_kinds_csv().as_str())
    );
    assert!(facts.iter().all(|(key, value)| !key.contains("estimate") && !value.contains("estimate")));
}

fn assert_topology_contract_avoids_query_specific_terms(family: &str, topology_contract: &str) {
    let normalized = topology_contract.replace('-', "_").to_ascii_lowercase();
    for disallowed in [
        "tpch",
        "query_id",
        "q01",
        "q02",
        "q03",
        "q04",
        "q05",
        "q06",
        "q07",
        "q08",
        "q09",
        "q10",
        "q11",
        "q12",
        "q13",
        "q14",
        "q15",
        "q16",
        "q17",
        "q18",
        "q19",
        "q20",
        "q21",
        "q22",
        "lineitem",
        "orders",
        "customer",
        "supplier",
        "part",
        "partsupp",
        "nation",
        "region",
        "suppkey",
        "extendedprice",
        "totalrevenue",
        "maxrevenue",
        "literal",
        "parquet_path",
        "sql_text",
    ] {
        assert!(
            !normalized.contains(disallowed),
            "{family} topology contract must stay query-independent, found {disallowed} in {topology_contract}"
        );
    }
}

fn assert_modeled_reservations_match_statuses(facts: &std::collections::BTreeMap<&'static str, String>) {
    let statuses =
        facts.get("bounded_join_reservation_statuses").expect("bounded join facts must include reservation statuses");
    let modeled = facts.get("modeled_reservations").expect("bounded join facts must include modeled reservations");
    let blocked = facts.get("blocked_reservations").expect("bounded join facts must include blocked reservations");
    let missing = facts.get("missing_reservations").expect("bounded join facts must include missing reservations");
    let expected_modeled = statuses
        .split(',')
        .filter_map(|status| status.split_once(':'))
        .filter_map(|(kind, status)| (status == "modeled").then_some(kind))
        .collect::<Vec<_>>()
        .join(",");
    let expected_blocked = statuses
        .split(',')
        .filter_map(|status| status.split_once(':'))
        .filter_map(|(kind, status)| (status != "modeled").then_some(kind))
        .collect::<Vec<_>>()
        .join(",");

    assert_eq!(modeled, &expected_modeled);
    assert_eq!(blocked, &expected_blocked);
    assert_eq!(missing, &expected_blocked);
}

use super::*;

pub(super) fn assert_planning_rows(report: &NativeDataFusionPlanningReport, expected: &PlanningReportExpectations) {
    for row in &report.rows {
        if EXPECTED_BOUNDED_TPCH_ACCEPTED.contains(&row.query_id.as_str()) {
            assert_eq!(row.outcome, NativeDataFusionPlanningOutcome::AcceptedReplaced);
            assert_eq!(row.reason_code.as_deref(), None);
            assert_eq!(row.unsupported_category.as_deref(), None);
            if matches!(row.query_id.as_str(), "q01" | "q06") {
                assert!(row.bounded_join_reservation_facts.is_empty());
            } else {
                assert_eq!(
                    row.bounded_join_reservation_facts.get("bounded_join_execution_status").map(String::as_str),
                    Some("enabled"),
                    "accepted bounded join row should preserve modeled reservation facts for {}",
                    row.query_id
                );
                assert_eq!(
                    row.bounded_join_reservation_facts.get("blocked_reservations").map(String::as_str),
                    Some(""),
                    "accepted bounded join row should report no blocked modeled reservations for {}",
                    row.query_id
                );
                assert_eq!(
                    row.bounded_join_reservation_facts.get("bounded_join_execution_families").map(String::as_str),
                    Some(expected.execution_families.as_str()),
                    "accepted bounded join row should preserve execution family facts for {}",
                    row.query_id
                );
                if row.query_id == "q17" {
                    assert_eq!(
                        row.bounded_join_reservation_facts.get("bounded_join_execution_family").map(String::as_str),
                        Some("mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline"),
                        "q17 should report its admitted bounded family"
                    );
                    assert_eq!(
                        row.bounded_join_reservation_facts.get("modeled_reservations").map(String::as_str),
                        Some(
                            "source_read,operator_output,build_side,hash_table,probe_side,filtered_probe_state,mixed_size_data,gather_map,join_output,materialization"
                        ),
                        "q17 should not claim final_sort reservations"
                    );
                    assert_eq!(
                        row.bounded_join_reservation_facts.get("bounded_join_reservation_statuses").map(String::as_str),
                        Some(
                            "source_read:modeled,operator_output:modeled,build_side:modeled,hash_table:modeled,probe_side:modeled,filtered_probe_state:modeled,mixed_size_data:modeled,gather_map:modeled,join_output:modeled,materialization:modeled"
                        ),
                        "q17 should report shape-local reservation statuses"
                    );
                    assert_eq!(
                        row.bounded_join_reservation_facts.get("bounded_join_contract_modes").map(String::as_str),
                        Some("source_contract:hard_budgeted_report,aggregate_contract:hard_budgeted_report"),
                        "q17 should only claim hard-budgeted source/aggregate contracts"
                    );
                    assert_eq!(
                        row.bounded_join_reservation_facts
                            .get("bounded_join_unresolved_preflight_contracts")
                            .map(String::as_str),
                        Some(
                            "source_contract:libcudf_parquet_decoded_output_preflight_unavailable,aggregate_contract:libcudf_groupby_aggregate_temporary_workspace_preflight_unavailable"
                        ),
                        "q17 should keep source/aggregate preflight gaps as evidence"
                    );
                    assert!(
                        !row.bounded_join_reservation_facts
                            .get("bounded_join_contract_modes")
                            .is_some_and(|value| value.contains("final_sort_contract")),
                        "q17 should not claim final_sort contract modes"
                    );
                } else if row.query_id == "q15" {
                    assert_eq!(
                        row.bounded_join_reservation_facts.get("bounded_join_execution_family").map(String::as_str),
                        Some("keyed_hash_join_resident_subplan_pipeline"),
                        "q15 should report its admitted bounded family"
                    );
                    assert_eq!(
                        row.bounded_join_reservation_facts.get("modeled_reservations").map(String::as_str),
                        Some(expected.modeled_reservations.as_str()),
                        "q15 should model source, aggregate, hash join, final sort, and materialization reservations"
                    );
                    assert_eq!(
                        row.bounded_join_reservation_facts.get("bounded_join_contract_modes").map(String::as_str),
                        Some(
                            "source_contract:hard_budgeted_report,aggregate_contract:hard_budgeted_report,final_sort_contract:hard_budgeted_report"
                        ),
                        "q15 should claim the shared hard-budgeted source/aggregate/final-sort contracts"
                    );
                    assert_eq!(
                        row.bounded_join_reservation_facts
                            .get("bounded_join_unresolved_preflight_contracts")
                            .map(String::as_str),
                        Some(
                            "source_contract:libcudf_parquet_decoded_output_preflight_unavailable,aggregate_contract:libcudf_groupby_aggregate_temporary_workspace_preflight_unavailable,final_sort_contract:libcudf_final_sort_order_preflight_unavailable"
                        ),
                        "q15 should preserve unresolved preflight markers for hard-budgeted contracts"
                    );
                } else {
                    assert_eq!(
                        row.bounded_join_reservation_facts.get("modeled_reservations").map(String::as_str),
                        Some(expected.modeled_reservations.as_str()),
                        "accepted bounded join row should preserve modeled reservation facts for {}",
                        row.query_id
                    );
                    assert_eq!(
                        row.bounded_join_reservation_facts.get("bounded_join_reservation_statuses").map(String::as_str),
                        Some(expected.reservation_statuses.as_str()),
                        "accepted bounded join row should preserve reservation status facts for {}",
                        row.query_id
                    );
                    assert_eq!(
                        row.bounded_join_reservation_facts.get("bounded_join_contract_modes").map(String::as_str),
                        None,
                        "non-q17 accepted rows should not claim Q17 hard-budgeted report modes for {}",
                        row.query_id
                    );
                }
                assert_eq!(
                    row.bounded_join_reservation_facts.get("bounded_join_lower_level_blockers").map(String::as_str),
                    Some(""),
                    "accepted bounded join row should report no lower-level blockers for {}",
                    row.query_id
                );
                assert_eq!(
                    row.bounded_join_reservation_facts
                        .get("bounded_join_execution_family_alias_compatibility")
                        .map(String::as_str),
                    None,
                    "accepted bounded join row should not preserve mixed family alias compatibility for {}",
                    row.query_id
                );
                assert_eq!(
                    row.bounded_join_reservation_facts
                        .get("bounded_join_mixed_keyed_residual_capability_detail")
                        .map(String::as_str),
                    Some(expected.mixed_capability_detail),
                    "accepted bounded join row should document mixed full/semi/anti capability for {}",
                    row.query_id
                );
                assert_eq!(
                    row.bounded_join_reservation_facts
                        .get("bounded_join_mixed_keyed_residual_multi_join_capability")
                        .map(String::as_str),
                    Some(expected.mixed_multi_join_capability),
                    "accepted bounded join row should document mixed keyed-residual inner multi-join family for {}",
                    row.query_id
                );
                assert_eq!(
                    row.bounded_join_reservation_facts
                        .get("bounded_join_mixed_keyed_residual_multi_join_capability_detail")
                        .map(String::as_str),
                    Some(expected.mixed_multi_join_capability_detail),
                    "accepted bounded join row should document mixed keyed-residual inner-only multi-join scope for {}",
                    row.query_id
                );
                assert_eq!(
                    row.bounded_join_reservation_facts
                        .get("bounded_join_keyless_conditional_join_input_subplan_pipeline_capability")
                        .map(String::as_str),
                    Some(expected.keyless_input_subplan_capability),
                    "accepted bounded join row should document keyless conditional input-subplan capability for {}",
                    row.query_id
                );
                assert_eq!(
                    row.bounded_join_reservation_facts
                        .get("bounded_join_keyless_conditional_join_input_subplan_pipeline_capability_detail")
                        .map(String::as_str),
                    Some(expected.keyless_input_subplan_capability_detail),
                    "accepted bounded join row should document keyless conditional input-subplan scope for {}",
                    row.query_id
                );
                assert_eq!(
                    row.bounded_join_reservation_facts
                        .get("cudf_rs_mixed_join_size_data_alias_compatibility")
                        .map(String::as_str),
                    Some(expected.mixed_size_data_alias_compatibility),
                    "accepted bounded join row should document mixed size-data alias compatibility for {}",
                    row.query_id
                );
            }
            assert_eq!(row.post_native_plan_root, "NativeCudfExec");
            assert_eq!(row.post_native_execution_path, NativeDataFusionPostNativeExecutionPath::Native);
            assert!(!row.rejected_before_execution, "{} should be replaced by native execution", row.query_id);
        } else {
            let (expected_reason, expected_category) =
                expected_tpch_datafusion_planning_rejection(row.query_id.as_str());
            assert_eq!(row.outcome, NativeDataFusionPlanningOutcome::RejectedBeforeExecution);
            assert_eq!(row.reason_code.as_deref(), Some(expected_reason), "unexpected reason for {}", row.query_id);
            assert_eq!(
                row.unsupported_category.as_deref(),
                Some(expected_category),
                "unexpected unsupported category for {}",
                row.query_id
            );
            if expected_reason == "native_bounded_join_unsupported" {
                assert_eq!(
                    row.bounded_join_reservation_facts.get("blocked_reservations").map(String::as_str),
                    Some(expected_tpch_bounded_join_blocked_reservations(row.query_id.as_str())),
                    "missing bounded join blocker facts for {}",
                    row.query_id
                );
                assert_eq!(
                    row.bounded_join_reservation_facts.get("missing_reservations").map(String::as_str),
                    Some(expected_tpch_bounded_join_blocked_reservations(row.query_id.as_str())),
                    "missing bounded join missing-reservation facts for {}",
                    row.query_id
                );
                assert_eq!(
                    row.bounded_join_reservation_facts.get("bounded_join_lower_level_blockers").map(String::as_str),
                    Some(expected_tpch_bounded_join_lower_level_blockers(row.query_id.as_str())),
                    "missing lower-level blocker facts for {}",
                    row.query_id
                );
                assert_eq!(
                    row.bounded_join_reservation_facts.get("bounded_join_contract_modes").map(String::as_str),
                    None,
                    "rejected bounded join row should not claim admitted hard-budgeted contract modes for {}",
                    row.query_id
                );
                if let Some(expected_family) = expected_tpch_bounded_join_blocked_family(row.query_id.as_str()) {
                    assert_eq!(
                        row.bounded_join_reservation_facts.get("bounded_join_blocked_join_family").map(String::as_str),
                        Some(expected_family),
                        "missing bounded join blocked family fact for {}",
                        row.query_id
                    );
                    assert_eq!(
                        row.bounded_join_reservation_facts.get("bounded_join_shape_blocker").map(String::as_str),
                        expected_tpch_bounded_join_shape_blocker(row.query_id.as_str()),
                        "missing bounded join shape blocker fact for {}",
                        row.query_id
                    );
                    let expected_lower_level_blockers =
                        expected_tpch_bounded_join_lower_level_blockers(row.query_id.as_str());
                    if expected_lower_level_blockers.is_empty() {
                        assert_eq!(
                            row.bounded_join_reservation_facts
                                .get("bounded_join_required_lower_level_contract")
                                .map(String::as_str),
                            None,
                            "structural bounded topology rejections should not claim lower-level blockers for {}",
                            row.query_id
                        );
                    } else {
                        assert_eq!(
                            row.bounded_join_reservation_facts
                                .get("bounded_join_required_lower_level_contract")
                                .map(String::as_str),
                            Some(expected_lower_level_blockers),
                            "missing bounded join lower-level contract fact for {}",
                            row.query_id
                        );
                    }
                }
                assert_eq!(
                    row.bounded_join_reservation_facts.get("bounded_join_unsupported_shape").map(String::as_str),
                    Some(expected.supported_shape.as_str()),
                    "missing supported bounded join family fact for {}",
                    row.query_id
                );
                assert_eq!(
                    row.bounded_join_reservation_facts
                        .get("bounded_join_execution_family_alias_compatibility")
                        .map(String::as_str),
                    None,
                    "unsupported bounded join row should not preserve mixed family alias compatibility for {}",
                    row.query_id
                );
                assert_eq!(
                    row.bounded_join_reservation_facts
                        .get("bounded_join_mixed_keyed_residual_capability_detail")
                        .map(String::as_str),
                    Some(expected.mixed_capability_detail),
                    "missing mixed full/semi/anti capability detail fact for {}",
                    row.query_id
                );
                assert_eq!(
                    row.bounded_join_reservation_facts
                        .get("bounded_join_mixed_keyed_residual_multi_join_capability")
                        .map(String::as_str),
                    Some(expected.mixed_multi_join_capability),
                    "missing mixed keyed-residual inner multi-join family fact for {}",
                    row.query_id
                );
                assert_eq!(
                    row.bounded_join_reservation_facts
                        .get("bounded_join_mixed_keyed_residual_multi_join_capability_detail")
                        .map(String::as_str),
                    Some(expected.mixed_multi_join_capability_detail),
                    "missing mixed keyed-residual inner-only multi-join scope fact for {}",
                    row.query_id
                );
                assert_eq!(
                    row.bounded_join_reservation_facts
                        .get("bounded_join_keyless_conditional_join_input_subplan_pipeline_capability")
                        .map(String::as_str),
                    Some(expected.keyless_input_subplan_capability),
                    "missing keyless conditional input-subplan capability fact for {}",
                    row.query_id
                );
                assert_eq!(
                    row.bounded_join_reservation_facts
                        .get("bounded_join_keyless_conditional_join_input_subplan_pipeline_capability_detail")
                        .map(String::as_str),
                    Some(expected.keyless_input_subplan_capability_detail),
                    "missing keyless conditional input-subplan scope fact for {}",
                    row.query_id
                );
                assert_eq!(
                    row.bounded_join_reservation_facts
                        .get("cudf_rs_mixed_join_size_data_alias_compatibility")
                        .map(String::as_str),
                    Some(expected.mixed_size_data_alias_compatibility),
                    "missing mixed size-data alias compatibility fact for {}",
                    row.query_id
                );
            } else {
                assert!(row.bounded_join_reservation_facts.is_empty());
            }
            assert_ne!(row.post_native_execution_path, NativeDataFusionPostNativeExecutionPath::Native);
            assert!(row.rejected_before_execution, "{} should reject before native execution", row.query_id);
        }
        assert_eq!(row.physical_root, expected_tpch_physical_root(row.query_id.as_str()));
        assert_eq!(
            row.native_candidate_shape.as_deref().unwrap_or("none"),
            expected_tpch_native_candidate_shape(row.query_id.as_str()),
            "unexpected native candidate shape for {}",
            row.query_id
        );
        assert_eq!(
            row.native_candidate_shape_family.as_deref().unwrap_or("none"),
            expected_tpch_native_candidate_shape_family(row.query_id.as_str()),
            "unexpected native candidate shape family for {}",
            row.query_id
        );
        assert!(
            row.physical_shape.contains(expected_tpch_physical_shape_marker(row.query_id.as_str())),
            "unexpected physical shape for {}: {}",
            row.query_id,
            row.physical_shape
        );
        if EXPECTED_BOUNDED_TPCH_JOIN_ACCEPTED.contains(&row.query_id.as_str())
            && row.outcome == NativeDataFusionPlanningOutcome::AcceptedReplaced
        {
            assert!(
                row.join_diagnostics.is_empty(),
                "{} accepted replacement should not expose join diagnostics",
                row.query_id
            );
        } else {
            assert_tpch_join_diagnostics(row.query_id.as_str(), &row.join_diagnostics);
        }
    }
}

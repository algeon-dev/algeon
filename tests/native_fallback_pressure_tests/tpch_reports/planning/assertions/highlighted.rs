use super::*;

pub(super) fn assert_highlighted_planning_rows_and_tsv(
    report: &NativeDataFusionPlanningReport,
    metadata_fingerprint: &str,
    expected: &PlanningReportExpectations,
) {
    let q07 = planning_row(report, "q07");
    assert_eq!(q07.outcome, NativeDataFusionPlanningOutcome::AcceptedReplaced);
    assert_eq!(q07.post_native_plan_root, "NativeCudfExec");
    assert_eq!(q07.post_native_execution_path, NativeDataFusionPostNativeExecutionPath::Native);
    assert_eq!(
        q07.bounded_join_reservation_facts.get("bounded_join_execution_status").map(String::as_str),
        Some("enabled")
    );
    assert!(
        q07.bounded_join_reservation_facts.get("bounded_join_execution_families").is_some_and(|families| {
            families.contains("multi_join_mixed_keyed_residual_inner_left_semi_left_anti_pipeline")
        }),
        "q07 accepted facts must name the expanded mixed keyed-residual multi-join family"
    );
    assert_eq!(
        q07.bounded_join_reservation_facts
            .get("bounded_join_mixed_keyed_residual_multi_join_capability_detail")
            .map(String::as_str),
        Some("mixed_keyed_residual_multi_join_types=inner,left_semi,left_anti")
    );

    let q21 = planning_row(report, "q21");
    assert_eq!(q21.outcome, NativeDataFusionPlanningOutcome::AcceptedReplaced);
    assert_eq!(q21.reason_code, None);
    assert_eq!(q21.unsupported_category, None);
    assert_eq!(q21.post_native_plan_root, "NativeCudfExec");
    assert_eq!(q21.post_native_execution_path, NativeDataFusionPostNativeExecutionPath::Native);
    assert_eq!(
        q21.bounded_join_reservation_facts.get("bounded_join_execution_status").map(String::as_str),
        Some("enabled")
    );
    assert!(
        q21.bounded_join_reservation_facts.get("bounded_join_execution_families").is_some_and(|families| {
            families.contains("multi_join_mixed_keyed_residual_inner_left_semi_left_anti_pipeline")
        }),
        "q21 accepted facts must name the expanded mixed keyed-residual multi-join family"
    );
    assert_eq!(
        q21.bounded_join_reservation_facts
            .get("bounded_join_mixed_keyed_residual_multi_join_capability_detail")
            .map(String::as_str),
        Some("mixed_keyed_residual_multi_join_types=inner,left_semi,left_anti")
    );
    assert!(!q21.rejected_before_execution, "q21 should be replaced by native execution");

    let q11 = planning_row(report, "q11");
    assert_eq!(q11.outcome, NativeDataFusionPlanningOutcome::AcceptedReplaced);
    assert_eq!(q11.reason_code, None);
    assert_eq!(q11.unsupported_category, None);
    assert_eq!(q11.post_native_plan_root, "NativeCudfExec");
    assert_eq!(q11.post_native_execution_path, NativeDataFusionPostNativeExecutionPath::Native);
    assert_eq!(q11.native_candidate_shape_family.as_deref(), Some("sort_mixed_hash_nested_loop_join_input"));
    assert!(!q11.rejected_before_execution, "q11 should be replaced by native execution");
    assert_eq!(
        q11.bounded_join_reservation_facts.get("bounded_join_execution_status").map(String::as_str),
        Some("enabled")
    );
    assert_eq!(q11.bounded_join_reservation_facts.get("blocked_reservations").map(String::as_str), Some(""));
    assert_eq!(
        q11.bounded_join_reservation_facts.get("bounded_join_lower_level_blockers").map(String::as_str),
        Some("")
    );
    assert_eq!(
        q11.bounded_join_reservation_facts.get("bounded_join_execution_families").map(String::as_str),
        Some(expected.execution_families.as_str())
    );
    assert!(
        q11.bounded_join_reservation_facts.get("bounded_join_execution_families").is_some_and(|families| {
            families.contains("keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline")
        }),
        "q11 accepted facts must name the bounded keyless conditional aggregate-over-join subplan inputs family"
    );
    assert_eq!(
        q11.bounded_join_reservation_facts
            .get("bounded_join_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline_capability")
            .map(String::as_str),
        Some(expected.keyless_aggregate_over_join_subplan_inputs_capability)
    );
    assert_eq!(
        q11.bounded_join_reservation_facts
            .get("bounded_join_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline_capability_detail")
            .map(String::as_str),
        Some(expected.keyless_aggregate_over_join_subplan_inputs_capability_detail)
    );

    let q22 = planning_row(report, "q22");
    assert_eq!(q22.outcome, NativeDataFusionPlanningOutcome::AcceptedReplaced);
    assert_eq!(q22.reason_code, None);
    assert_eq!(q22.unsupported_category, None);
    assert_eq!(q22.post_native_plan_root, "NativeCudfExec");
    assert_eq!(q22.post_native_execution_path, NativeDataFusionPostNativeExecutionPath::Native);
    assert_eq!(
        q22.bounded_join_reservation_facts.get("bounded_join_execution_status").map(String::as_str),
        Some("enabled")
    );
    assert!(
        q22.bounded_join_reservation_facts
            .get("bounded_join_execution_families")
            .is_some_and(|families| { families.contains("keyless_conditional_join_input_subplan_pipeline") }),
        "q22 accepted facts must name the bounded keyless conditional input-subplan family"
    );
    assert_eq!(
        q22.bounded_join_reservation_facts
            .get("bounded_join_keyless_conditional_join_input_subplan_pipeline_capability")
            .map(String::as_str),
        Some("keyless_conditional_join_input_subplan_pipeline")
    );
    assert_eq!(
        q22.bounded_join_reservation_facts
            .get("bounded_join_keyless_conditional_join_input_subplan_pipeline_capability_detail")
            .map(String::as_str),
        Some(
            "root_join=keyless_conditional_inner;input_subplans=source_filter_projection_or_projection_wrapped_left_anti_hash_join_or_scalar_global_aggregate"
        )
    );
    assert!(!q22.rejected_before_execution, "q22 should be replaced by native execution");

    let q13 = planning_row(report, "q13");
    assert_eq!(q13.outcome, NativeDataFusionPlanningOutcome::AcceptedReplaced);
    assert_eq!(q13.reason_code, None);
    assert_eq!(q13.unsupported_category, None);
    assert_eq!(q13.post_native_plan_root, "NativeCudfExec");
    assert_eq!(q13.post_native_execution_path, NativeDataFusionPostNativeExecutionPath::Native);
    assert_eq!(
        q13.bounded_join_reservation_facts.get("bounded_join_execution_status").map(String::as_str),
        Some("enabled")
    );
    assert_eq!(
        q13.bounded_join_reservation_facts.get("bounded_join_lower_level_blockers").map(String::as_str),
        Some("")
    );
    assert!(!q13.rejected_before_execution, "q13 should be replaced by native execution");

    let q16 = planning_row(report, "q16");
    assert_eq!(q16.outcome, NativeDataFusionPlanningOutcome::AcceptedReplaced);
    assert_eq!(q16.reason_code, None);
    assert_eq!(q16.unsupported_category, None);
    assert_eq!(q16.post_native_plan_root, "NativeCudfExec");
    assert_eq!(q16.post_native_execution_path, NativeDataFusionPostNativeExecutionPath::Native);
    assert_eq!(
        q16.bounded_join_reservation_facts.get("bounded_join_execution_status").map(String::as_str),
        Some("enabled")
    );
    assert_eq!(
        q16.bounded_join_reservation_facts.get("bounded_join_lower_level_blockers").map(String::as_str),
        Some("")
    );
    assert_eq!(
        q16.bounded_join_reservation_facts
            .get("bounded_join_mixed_keyed_residual_multi_join_capability")
            .map(String::as_str),
        Some("multi_join_mixed_keyed_residual_inner_left_semi_left_anti_pipeline")
    );
    assert!(!q16.rejected_before_execution, "q16 should be replaced by native execution");

    let rejected_query_ids = report
        .rows
        .iter()
        .filter(|row| row.outcome == NativeDataFusionPlanningOutcome::RejectedBeforeExecution)
        .map(|row| row.query_id.as_str())
        .collect::<Vec<_>>();
    assert_eq!(rejected_query_ids, vec!["q02"]);

    let q02 = planning_row(report, "q02");
    assert_eq!(
        q02.bounded_join_reservation_facts.get("bounded_join_blocked_join_family").map(String::as_str),
        Some("multi_join_with_aggregate_over_join_subplan_topk")
    );
    assert_eq!(
        q02.bounded_join_reservation_facts.get("bounded_join_shape_blocker").map(String::as_str),
        Some("bounded_multi_join_aggregate_over_join_subplan_topk_executor_missing")
    );
    assert_eq!(
        q02.bounded_join_reservation_facts.get("bounded_join_lower_level_blockers").map(String::as_str),
        Some("")
    );
    assert_eq!(
        q02.bounded_join_reservation_facts.get("bounded_join_required_lower_level_contract").map(String::as_str),
        None
    );
    assert!(q02.rejected_before_execution, "q02 should remain rejected before native execution");

    let q18 = planning_row(report, "q18");
    assert_eq!(q18.outcome, NativeDataFusionPlanningOutcome::AcceptedReplaced);
    assert_eq!(q18.reason_code, None);
    assert_eq!(q18.unsupported_category, None);
    assert_eq!(q18.post_native_plan_root, "NativeCudfExec");
    assert_eq!(q18.post_native_execution_path, NativeDataFusionPostNativeExecutionPath::Native);
    assert_eq!(
        q18.bounded_join_reservation_facts.get("bounded_join_execution_family").map(String::as_str),
        Some("hash_join_with_resident_grouped_aggregate_input_post_join_aggregate_topk")
    );
    assert_eq!(
        q18.bounded_join_reservation_facts.get("bounded_join_lower_level_blockers").map(String::as_str),
        Some("")
    );
    assert_eq!(q18.bounded_join_reservation_facts.get("blocked_reservations").map(String::as_str), Some(""));
    assert_eq!(q18.bounded_join_reservation_facts.get("missing_reservations").map(String::as_str), Some(""));
    assert_eq!(
        q18.bounded_join_reservation_facts.get("bounded_join_contract_modes").map(String::as_str),
        Some(
            "source_contract:hard_budgeted_report,aggregate_contract:hard_budgeted_report,final_sort_contract:hard_budgeted_report"
        )
    );

    let q20 = planning_row(report, "q20");
    assert_eq!(q20.outcome, NativeDataFusionPlanningOutcome::AcceptedReplaced);
    assert_eq!(q20.reason_code, None);
    assert_eq!(q20.unsupported_category, None);
    assert_eq!(q20.post_native_plan_root, "NativeCudfExec");
    assert_eq!(q20.post_native_execution_path, NativeDataFusionPostNativeExecutionPath::Native);
    assert_eq!(
        q20.bounded_join_reservation_facts.get("bounded_join_execution_family").map(String::as_str),
        Some("multi_join_with_resident_grouped_aggregate_subquery_filter")
    );
    assert_eq!(
        q20.bounded_join_reservation_facts.get("bounded_join_lower_level_blockers").map(String::as_str),
        Some("")
    );
    assert_eq!(q20.bounded_join_reservation_facts.get("blocked_reservations").map(String::as_str), Some(""));
    assert_eq!(q20.bounded_join_reservation_facts.get("missing_reservations").map(String::as_str), Some(""));
    assert_eq!(
        q20.bounded_join_reservation_facts.get("bounded_join_contract_modes").map(String::as_str),
        Some(
            "source_contract:hard_budgeted_report,aggregate_contract:hard_budgeted_report,final_sort_contract:hard_budgeted_report"
        )
    );

    let q15 = planning_row(report, "q15");
    assert_eq!(q15.outcome, NativeDataFusionPlanningOutcome::AcceptedReplaced);
    assert_eq!(q15.reason_code, None);
    assert_eq!(q15.unsupported_category, None);
    assert_eq!(q15.post_native_plan_root, "NativeCudfExec");
    assert_eq!(q15.post_native_execution_path, NativeDataFusionPostNativeExecutionPath::Native);
    assert_eq!(
        q15.bounded_join_reservation_facts.get("bounded_join_execution_family").map(String::as_str),
        Some("keyed_hash_join_resident_subplan_pipeline")
    );
    assert_eq!(
        q15.bounded_join_reservation_facts.get("bounded_join_lower_level_blockers").map(String::as_str),
        Some("")
    );
    assert_eq!(q15.bounded_join_reservation_facts.get("blocked_reservations").map(String::as_str), Some(""));
    assert_eq!(q15.bounded_join_reservation_facts.get("missing_reservations").map(String::as_str), Some(""));
    assert_eq!(
        q15.bounded_join_reservation_facts.get("bounded_join_required_lower_level_contract").map(String::as_str),
        None
    );
    assert_eq!(
        q15.bounded_join_reservation_facts.get("modeled_reservations").map(String::as_str),
        Some(expected.modeled_reservations.as_str())
    );
    assert_eq!(
        q15.bounded_join_reservation_facts.get("bounded_join_contract_modes").map(String::as_str),
        Some(
            "source_contract:hard_budgeted_report,aggregate_contract:hard_budgeted_report,final_sort_contract:hard_budgeted_report"
        )
    );
    assert_eq!(
        q15.bounded_join_reservation_facts.get("bounded_join_unresolved_preflight_contracts").map(String::as_str),
        Some(
            "source_contract:libcudf_parquet_decoded_output_preflight_unavailable,aggregate_contract:libcudf_groupby_aggregate_temporary_workspace_preflight_unavailable,final_sort_contract:libcudf_final_sort_order_preflight_unavailable"
        )
    );
    assert!(!q15.rejected_before_execution, "q15 should be replaced by native execution");

    let q17 = planning_row(report, "q17");
    assert_eq!(q17.outcome, NativeDataFusionPlanningOutcome::AcceptedReplaced);
    assert_eq!(q17.reason_code, None);
    assert_eq!(q17.unsupported_category, None);
    assert_eq!(q17.post_native_plan_root, "NativeCudfExec");
    assert_eq!(q17.post_native_execution_path, NativeDataFusionPostNativeExecutionPath::Native);
    assert_eq!(
        q17.bounded_join_reservation_facts.get("bounded_join_execution_family").map(String::as_str),
        Some("mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline")
    );
    assert_eq!(
        q17.bounded_join_reservation_facts.get("bounded_join_lower_level_blockers").map(String::as_str),
        Some("")
    );
    assert_eq!(q17.bounded_join_reservation_facts.get("blocked_reservations").map(String::as_str), Some(""));
    assert_eq!(q17.bounded_join_reservation_facts.get("missing_reservations").map(String::as_str), Some(""));
    assert_eq!(
        q17.bounded_join_reservation_facts.get("bounded_join_contract_modes").map(String::as_str),
        Some("source_contract:hard_budgeted_report,aggregate_contract:hard_budgeted_report")
    );
    assert_eq!(
        q17.bounded_join_reservation_facts.get("bounded_join_unresolved_preflight_contracts").map(String::as_str),
        Some(
            "source_contract:libcudf_parquet_decoded_output_preflight_unavailable,aggregate_contract:libcudf_groupby_aggregate_temporary_workspace_preflight_unavailable"
        )
    );
    assert!(
        !q17.bounded_join_reservation_facts
            .get("bounded_join_contract_modes")
            .is_some_and(|value| value.contains("final_sort_contract"))
    );
    assert!(!q17.rejected_before_execution, "q17 should be replaced by native execution");

    let text = report.to_tsv();
    assert!(text.starts_with("# native_datafusion_planning_tsv_schema_version=6\n"));
    assert!(text.contains("# tpch_dataset_scale_factor=tiny\n"));
    assert!(text.contains(&format!("# tpch_dataset_metadata_fingerprint={}\n", metadata_fingerprint)));
    assert!(text.contains("# tpch_dataset_content_fingerprint=not_computed\n"));
    assert!(text.contains(&format!("{NATIVE_DATAFUSION_PLANNING_TSV_HEADER}\n")));
    assert!(text.contains("q01\t"));
    assert!(text.contains("q06\t"));
    assert!(text.contains("q22\t"));
    assert!(text.contains("\taccepted_replaced\t"));
}

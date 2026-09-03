use super::*;

pub(super) fn assert_highlighted_planning_rows_and_tsv(
    report: &NativeDataFusionPlanningReport,
    metadata_fingerprint: &str,
) {
    for query_id in ["q07", "q11", "q15", "q17", "q21", "q22"] {
        let row = planning_replacement_row(report, query_id);
        assert_eq!(row.outcome, Some(CandidateOutcomeKind::Selected));
        assert_eq!(row.reason_code, None);
        assert_eq!(row.unsupported_category, None);
        if row.final_disposition == Some(FinalPlanDisposition::Native) {
            assert_eq!(row.post_native_plan_root, "NexusNativeExec");
        } else {
            assert_eq!(row.final_disposition, Some(FinalPlanDisposition::Mixed));
        }
        assert!(!row.rejected_before_execution, "{query_id} should be replaced by native execution");
        assert!(
            row.capability_contract_rows.iter().map(|gate| gate.to_stable_kv_row()).all(|rendered| {
                admission_contract_row_is_family_path_free(&rendered)
                    && admission_contract_row_excludes_adaptive_terms(&rendered)
            }),
            "{query_id} should not expose retired family/path facts"
        );
    }

    let q21 = planning_replacement_row(report, "q21");
    assert_eq!(q21.physical_root, expected_tpch_physical_root("q21"));
    assert!(
        q21.capability_contract_rows.iter().any(|gate| gate.reservation_kind == ReservationKind::FinalSort
            && gate.contract_quality == ContractQuality::HardBudgetedReportWithUnresolvedPreflight
            && gate.unresolved_preflight_gap == Some("libcudf_final_sort_order_preflight_unavailable")),
        "q21 should report its TopK final-sort gate as hard-budgeted unresolved preflight"
    );

    let q15 = planning_replacement_row(report, "q15");
    if q15.final_disposition == Some(FinalPlanDisposition::Native) {
        assert!(
            q15.capability_contract_rows.iter().any(|gate| gate.reservation_kind == ReservationKind::FinalSort
                && gate.contract_quality == ContractQuality::HardBudgetedReportWithUnresolvedPreflight
                && gate.unresolved_preflight_gap == Some("libcudf_final_sort_order_preflight_unavailable")),
            "q15 should report final-sort as hard-budgeted with an unresolved lower-level preflight gap"
        );
        assert!(
            q15.capability_contract_rows.iter().any(|gate| gate.reservation_kind == ReservationKind::OperatorOutput
                && gate.contract_quality == ContractQuality::HardBudgetedReportWithUnresolvedPreflight
                && gate.unresolved_preflight_gap
                    == Some("libcudf_groupby_aggregate_temporary_workspace_preflight_unavailable")),
            "q15 resident/shared-subplan row should keep aggregate preflight gaps explicit"
        );
        assert!(
            q15.capability_compiled_graphs.iter().any(|graph| graph.resource_count > 0 && graph.gate_count > 0),
            "q15 resident/shared-subplan row should expose compiled graph resource and gate accounting"
        );
    }

    let q17 = planning_replacement_row(report, "q17");
    assert!(
        q17.capability_contract_rows.iter().any(|gate| gate.reservation_kind == ReservationKind::OperatorOutput
            && gate.contract_quality == ContractQuality::HardBudgetedReportWithUnresolvedPreflight
            && gate.unresolved_preflight_gap
                == Some("libcudf_groupby_aggregate_temporary_workspace_preflight_unavailable")),
        "q17 should keep aggregate preflight gaps as admission contract quality evidence"
    );
    assert!(
        !q17.capability_contract_rows.iter().any(|gate| gate.reservation_kind == ReservationKind::FinalSort),
        "q17 should not claim a final-sort gate"
    );

    let rejected_query_ids = tpch_query_ids()
        .filter(|query_id| {
            planning_replacement_row(report, query_id).outcome == Some(CandidateOutcomeKind::NotSupported)
        })
        .collect::<Vec<_>>();
    assert!(rejected_query_ids.is_empty(), "all TPC-H replacement rows should now be native-admitted");

    let text = report.to_tsv();
    assert!(text.starts_with("# workload=tpch\n"));
    assert!(text.contains("# workload=tpch\n"));
    assert!(text.contains("# dataset_scale_factor=tiny\n"));
    assert!(text.contains(&format!("# dataset_metadata_fingerprint={metadata_fingerprint}\n")));
    assert!(text.contains("# dataset_content_fingerprint=not_computed\n"));
    assert!(text.contains(&format!("{NATIVE_DATAFUSION_PLANNING_TSV_HEADER}\n")));
    assert!(
        text.contains(
            "capability_contract_rows\tcapability_compiled_graphs\tnative_optimizer\tedge_normalize_inserted"
        )
    );
    assert!(text.contains("execution_mode=bounded"));
    assert!(text.contains("contract_status=modeled"));
    assert!(text.contains("contract_quality="));
    assert!(!text.contains("bounded_join_reservation_facts"));
    assert!(!text.contains("bounded_join_execution_family="));
    assert!(!text.contains("memory_history"));
    assert!(!text.contains("spill"));
    assert!(!text.contains("downgrade"));
    assert!(text.contains("q01\t"));
    assert!(text.contains("q06\t"));
    assert!(text.contains("q22\t"));
    assert!(text.contains("\tselected\t"));
}

fn admission_contract_row_excludes_adaptive_terms(rendered: &str) -> bool {
    !rendered.contains("retry")
        && !rendered.contains("spill")
        && !rendered.contains("downgrade")
        && !rendered.contains("memory_history")
        && !rendered.contains("memory-history")
        && !rendered.contains("memory history")
}

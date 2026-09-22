use algeon_datafusion::native_report::NativeDataFusionPlanningRow;

use super::*;

pub(super) fn assert_planning_rows(report: &NativeDataFusionPlanningReport) {
    for query_id in tpch_query_ids() {
        let row = planning_replacement_row(report, query_id);
        assert_row_rejection_diagnostics_are_family_free(row);
        if EXPECTED_BOUNDED_TPCH_ACCEPTED.contains(&row.query_id.as_str()) {
            assert_eq!(row.outcome, Some(CandidateOutcomeKind::Selected));
            assert_eq!(row.reason_code.as_deref(), None);
            assert_eq!(row.unsupported_category.as_deref(), None);
            if EXPECTED_BOUNDED_TPCH_JOIN_ACCEPTED.contains(&row.query_id.as_str())
                && row.final_disposition == Some(FinalPlanDisposition::Native)
            {
                assert!(row.capability_decision_rows.is_empty());
                assert!(
                    !row.capability_compiled_graphs.is_empty(),
                    "{} accepted bounded join row should expose compiled graph diagnostics",
                    row.query_id
                );
            } else {
                if row.final_disposition == Some(FinalPlanDisposition::Native) {
                    assert!(
                        row.capability_decision_rows.is_empty(),
                        "{} accepted rows must not emit non-admitted capability decisions",
                        row.query_id
                    );
                    assert!(row.capability_compiled_graphs.is_empty());
                }
            }
            if row.final_disposition == Some(FinalPlanDisposition::Native) {
                assert_eq!(row.post_native_plan_root, "AlgeonNativeExec");
            } else {
                assert_eq!(row.final_disposition, Some(FinalPlanDisposition::Mixed));
            }
            assert!(!row.rejected_before_execution, "{} should be replaced by native execution", row.query_id);
        } else {
            let (expected_reason, expected_category) =
                expected_tpch_datafusion_planning_rejection(row.query_id.as_str());
            assert_eq!(row.outcome, Some(CandidateOutcomeKind::NotSupported));
            assert_eq!(row.reason_code.as_deref(), Some(expected_reason), "unexpected reason for {}", row.query_id);
            assert_eq!(
                row.unsupported_category.as_deref(),
                Some(expected_category),
                "unexpected unsupported category for {}",
                row.query_id
            );
            assert!(
                row.capability_decision_rows
                    .iter()
                    .map(|gate| gate.to_stable_kv_row())
                    .all(|rendered| admission_contract_row_is_family_path_free(&rendered)),
                "{} rejected admission contract rows should not preserve family/path vocabulary",
                row.query_id
            );
            assert_ne!(row.final_disposition, Some(FinalPlanDisposition::Native));
            assert!(row.rejected_before_execution, "{} should reject before native execution", row.query_id);
        }
        assert_eq!(row.physical_root, expected_tpch_physical_root(row.query_id.as_str()));
        if row.final_disposition == Some(FinalPlanDisposition::Native)
            || row.outcome == Some(CandidateOutcomeKind::NotSupported)
        {
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
        } else {
            assert!(
                row.native_candidate_shape.is_some() && row.native_candidate_shape_family.is_some(),
                "{} mixed-plan accepted rows should keep candidate diagnostics",
                row.query_id
            );
        }
        assert!(
            row.physical_shape.contains(expected_tpch_physical_shape_marker(row.query_id.as_str())),
            "unexpected physical shape for {}: {}",
            row.query_id,
            row.physical_shape
        );
        if EXPECTED_BOUNDED_TPCH_JOIN_ACCEPTED.contains(&row.query_id.as_str())
            && row.outcome == Some(CandidateOutcomeKind::Selected)
            && row.final_disposition == Some(FinalPlanDisposition::Native)
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

fn assert_row_rejection_diagnostics_are_family_free(row: &NativeDataFusionPlanningRow) {
    for gate in &row.capability_decision_rows {
        assert_no_legacy_rejection_terms(row.query_id.as_str(), &gate.to_stable_kv_row());
    }
    for graph in &row.capability_compiled_graphs {
        assert_no_legacy_rejection_terms(row.query_id.as_str(), &format!("{graph:?}"));
    }
    assert_no_legacy_rejection_terms(row.query_id.as_str(), &format!("{:?}", row.join_diagnostics));
    assert_no_legacy_rejection_terms(row.query_id.as_str(), &format!("{:?}", row.source_diagnostics));
}

fn assert_no_legacy_rejection_terms(label: &str, rendered: &str) {
    for retired in [
        "bounded_join_capability_rejection_",
        "bounded_join_blocked_join_family",
        "bounded_join_specific_blocker",
        "bounded_join_shape_blocker",
        "topology_unsupported",
        "BoundedJoinExecutionFamily",
        "BoundedJoinExecutionPath",
    ] {
        assert!(!rendered.contains(retired), "{label} must not expose retired rejection term {retired}");
    }
}

pub(super) fn admission_contract_row_is_family_path_free(rendered: &str) -> bool {
    !rendered.contains("bounded_join_execution_family")
        && !rendered.contains("bounded_join_required_shape_contract")
        && !rendered.contains("bounded_join_contract_modes")
        && !rendered.contains("bounded_join_unresolved_preflight_contracts")
        && !rendered.contains("BoundedJoinExecutionFamily")
        && !rendered.contains("BoundedJoinExecutionPath")
}

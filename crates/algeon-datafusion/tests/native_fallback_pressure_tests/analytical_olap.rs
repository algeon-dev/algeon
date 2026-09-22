use algeon_datafusion::native_report::NativeDataFusionPlanningRow;

use super::*;

#[tokio::test]
async fn native_pipeline_analytical_olap_fallback_report_pins_lowering_diagnostics() {
    let dir = unique_temp_dir("algeon-datafusion-analytical-olap-fallback-report");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();

    let items = analytical_olap_fallback_pressure_workload_items(&dir, tpch_report_session_config()).await.unwrap();
    let report = NativeFallbackPressureReport::collect(items);

    assert_eq!(report.rows.len(), ANALYTICAL_OLAP_QUERY_COUNT);
    assert_eq!(report.selected_count(), ANALYTICAL_OLAP_QUERY_COUNT);
    assert_eq!(report.not_selected_count(), 0);

    for query in ANALYTICAL_OLAP_QUERIES {
        let row = report_row(&report, query.id);
        assert_eq!(row.outcome, Some(CandidateOutcomeKind::Selected), "unexpected outcome for {}", query.id);
        assert_eq!(row.reason_code, None, "unexpected reason for {}", query.id);
        assert_eq!(row.unsupported_category, None, "unexpected unsupported category for {}", query.id);
        assert!(row.native_candidate_shape.as_deref().is_some_and(|shape| shape.contains("JoinExec")));
        assert!(row.native_candidate_shape_family.is_some(), "{} should have a report family label", query.id);
        assert!(!row.join_diagnostics.is_empty(), "{} should emit join diagnostics", query.id);
        assert!(!row.source_diagnostics.is_empty(), "{} should emit source diagnostics", query.id);
        assert!(!row.rejected_before_execution, "{} should lower before bounded admission checks", query.id);
        assert_no_legacy_rejection_terms(query.id, &format!("{:?}", row.join_diagnostics));
        assert_no_legacy_rejection_terms(query.id, &format!("{:?}", row.source_diagnostics));
    }

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_pipeline_analytical_olap_capability_first_planning_report_pins_capability_first_admission_surface() {
    let dir = unique_temp_dir("algeon-datafusion-analytical-olap-capability-first-planning");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();

    let items = analytical_olap_datafusion_planning_workload_items(&dir, tpch_report_session_config()).await.unwrap();
    let report = NativeDataFusionPlanningReport::collect(items);

    assert_eq!(planning_query_count(&report), ANALYTICAL_OLAP_QUERY_COUNT);
    assert!(
        ANALYTICAL_OLAP_QUERIES.iter().any(
            |query| query.capability_first_admission == AnalyticalOlapCapabilityFirstAdmission::ExpectedSupersetWin
        ),
        "Milestone F.2 supplement must include a planned capability-first superset win"
    );

    for query in ANALYTICAL_OLAP_QUERIES {
        let row = planning_replacement_row(&report, query.id);
        assert!(row.native_candidate_shape.as_deref().is_some_and(|shape| shape.contains("JoinExec")));
        assert_planning_row_has_family_free_rejection_diagnostics(query.id, row);

        assert_eq!(row.outcome, Some(CandidateOutcomeKind::Selected), "{}", query.id);
        assert_eq!(row.reason_code, None, "{}", query.id);
        assert!(!row.rejected_before_execution, "{}", query.id);
        assert!(row.capability_decision_rows.is_empty(), "{} should not emit accepted decision rows", query.id);
        assert!(!row.capability_compiled_graphs.is_empty(), "{} should emit compiled graph rows", query.id);
        if query.capability_first_admission == AnalyticalOlapCapabilityFirstAdmission::ExpectedSupersetWin {
            assert_eq!(
                row.outcome,
                Some(CandidateOutcomeKind::Selected),
                "{} expected superset win should be admitted by capability-first capability-first planning",
                query.id
            );
            assert!(
                row.capability_decision_rows.is_empty() && !row.capability_compiled_graphs.is_empty(),
                "{} expected superset win should emit compiled graph diagnostics",
                query.id
            );
        }
    }
    assert_no_legacy_rejection_terms("analytical_olap_capability_first_planning_tsv", &report.to_tsv());

    fs::remove_dir_all(dir).unwrap();
}

fn assert_planning_row_has_family_free_rejection_diagnostics(query_id: &str, row: &NativeDataFusionPlanningRow) {
    for gate in &row.capability_decision_rows {
        assert_no_legacy_rejection_terms(query_id, &gate.to_stable_kv_row());
    }
    for graph in &row.capability_compiled_graphs {
        assert_no_legacy_rejection_terms(query_id, &format!("{graph:?}"));
    }
    assert_no_legacy_rejection_terms(query_id, &format!("{:?}", row.join_diagnostics));
    assert_no_legacy_rejection_terms(query_id, &format!("{:?}", row.source_diagnostics));
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

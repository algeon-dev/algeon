use super::*;

#[tokio::test]
async fn native_pipeline_ad_hoc_olap_fallback_report_pins_lowering_diagnostics() {
    let dir = unique_temp_dir("datafusion-cudf-ad-hoc-olap-fallback-report");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();

    let items = ad_hoc_olap_fallback_pressure_workload_items(&dir, tpch_report_session_config()).await.unwrap();
    let report = NativeFallbackPressureReport::collect(items);

    assert_eq!(report.rows.len(), AD_HOC_OLAP_QUERY_COUNT);
    assert_eq!(report.rewritten_count(), AD_HOC_OLAP_QUERY_COUNT);
    assert_eq!(report.rejected_count(), 0);

    for query in AD_HOC_OLAP_QUERIES {
        let row = report_row(&report, query.id);
        assert_eq!(row.outcome, NativeFallbackPressureOutcome::Rewritten, "unexpected outcome for {}", query.id);
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
async fn native_pipeline_ad_hoc_olap_strict_planning_report_pins_capability_first_admission_surface() {
    let dir = unique_temp_dir("datafusion-cudf-ad-hoc-olap-strict-planning");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();

    let items = ad_hoc_olap_datafusion_planning_workload_items(&dir, tpch_report_session_config()).await.unwrap();
    let report = NativeDataFusionPlanningReport::collect(items);

    assert_eq!(report.rows.len(), AD_HOC_OLAP_QUERY_COUNT);
    assert!(
        AD_HOC_OLAP_QUERIES
            .iter()
            .any(|query| query.capability_first_admission == AdHocOlapCapabilityFirstAdmission::ExpectedSupersetWin),
        "Phase F.2 supplement must include a planned capability-first superset win"
    );

    for query in AD_HOC_OLAP_QUERIES {
        let row = planning_row(&report, query.id);
        assert!(
            row.adaptive_pipeline_rows.is_empty(),
            "{} strict planning row must not carry adaptive evidence",
            query.id
        );
        assert!(row.native_candidate_shape.as_deref().is_some_and(|shape| shape.contains("JoinExec")));
        assert_planning_row_has_family_free_rejection_diagnostics(query.id, row);

        match query.current_strict_admission {
            AdHocOlapCurrentStrictAdmission::Accepted => {
                assert_eq!(row.outcome, NativeDataFusionPlanningOutcome::AcceptedReplaced, "{}", query.id);
                assert_eq!(row.reason_code, None, "{}", query.id);
                assert!(!row.rejected_before_execution, "{}", query.id);
                assert!(!row.strict_bounded_contract_rows.is_empty(), "{} should emit strict gate rows", query.id);
                assert!(!row.strict_bounded_compiled_graphs.is_empty(), "{} should emit compiled graph rows", query.id);
            },
            AdHocOlapCurrentStrictAdmission::RejectedBeforeExecution => {
                assert_eq!(row.outcome, NativeDataFusionPlanningOutcome::RejectedBeforeExecution, "{}", query.id);
                assert_eq!(
                    row.reason_code.as_deref(),
                    Some(FallbackReason::NativeBoundedJoinUnsupported.code()),
                    "{}",
                    query.id
                );
                assert_eq!(row.unsupported_category.as_deref(), Some("operator_hashjoin"), "{}", query.id);
                assert!(row.rejected_before_execution, "{}", query.id);
                assert!(
                    row.strict_bounded_compiled_graphs.is_empty(),
                    "{} should not emit compiled graph rows",
                    query.id
                );
                assert_eq!(query.capability_first_admission, AdHocOlapCapabilityFirstAdmission::ExpectedSupersetWin);
            },
        }
        if query.capability_first_admission == AdHocOlapCapabilityFirstAdmission::ExpectedSupersetWin {
            assert_eq!(
                row.outcome,
                NativeDataFusionPlanningOutcome::AcceptedReplaced,
                "{} expected superset win should be admitted by capability-first strict planning",
                query.id
            );
            assert!(
                !row.strict_bounded_contract_rows.is_empty() && !row.strict_bounded_compiled_graphs.is_empty(),
                "{} expected superset win should emit strict gate rows and compiled graph diagnostics",
                query.id
            );
        }
    }
    assert_no_legacy_rejection_terms("ad_hoc_olap_strict_planning_tsv", &report.to_tsv());

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_pipeline_ad_hoc_olap_adaptive_planning_report_uses_adaptive_rows() {
    let dir = unique_temp_dir("datafusion-cudf-ad-hoc-olap-adaptive-planning");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();

    let optimizer_config =
        tpch_native_optimizer_config().with_native_pipeline_execution_mode(NativePipelineExecutionMode::Adaptive);
    let items = ad_hoc_olap_datafusion_planning_workload_items_with_optimizer_config(
        &dir,
        tpch_report_session_config(),
        optimizer_config,
    )
    .await
    .unwrap();
    let report = NativeDataFusionPlanningReport::collect(items);

    assert_eq!(report.rows.len(), AD_HOC_OLAP_QUERY_COUNT);
    for query in AD_HOC_OLAP_QUERIES {
        let row = planning_row(&report, query.id);
        assert_planning_row_has_family_free_rejection_diagnostics(query.id, row);
        assert!(
            row.strict_bounded_contract_rows.is_empty(),
            "{} adaptive row must not carry strict gate rows",
            query.id
        );
        assert!(
            row.strict_bounded_compiled_graphs.is_empty(),
            "{} adaptive row must not carry strict compiled graph rows",
            query.id
        );
        assert!(!row.adaptive_pipeline_rows.is_empty(), "{} adaptive planning should emit adaptive rows", query.id);
        assert!(
            row.adaptive_pipeline_rows
                .iter()
                .all(|adaptive_row| adaptive_row.execution_mode.as_str().contains("adaptive")),
            "{} adaptive rows should use adaptive execution-mode labels",
            query.id
        );
    }
    assert_no_legacy_rejection_terms("ad_hoc_olap_adaptive_planning_tsv", &report.to_tsv());

    fs::remove_dir_all(dir).unwrap();
}

fn assert_planning_row_has_family_free_rejection_diagnostics(
    query_id: &str,
    row: &datafusion_cudf::NativeDataFusionPlanningRow,
) {
    for gate in &row.strict_bounded_contract_rows {
        assert_no_legacy_rejection_terms(query_id, &gate.to_stable_kv_row());
    }
    for graph in &row.strict_bounded_compiled_graphs {
        assert_no_legacy_rejection_terms(query_id, &format!("{graph:?}"));
    }
    for adaptive in &row.adaptive_pipeline_rows {
        assert_no_legacy_rejection_terms(query_id, &adaptive.to_stable_kv_row());
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

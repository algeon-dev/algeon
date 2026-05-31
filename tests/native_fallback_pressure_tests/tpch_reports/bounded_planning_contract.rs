use super::super::*;

#[tokio::test]
async fn tpch_progression_contract_pairs_fallback_pressure_and_bounded_planning() {
    let dir = unique_temp_dir("datafusion-nexus-tpch-bounded-planning-contract");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();

    let fallback = NativeFallbackPressureReport::collect(
        tpch_fallback_pressure_workload_items(&dir, tpch_report_session_config()).await.unwrap(),
    );
    let planning = NativeDataFusionPlanningReport::collect(
        tpch_datafusion_planning_workload_items(&dir, tpch_report_session_config()).await.unwrap(),
    );

    assert_eq!(fallback.rows.len(), TPCH_QUERY_COUNT);
    assert_eq!(planning.rows.len(), TPCH_QUERY_COUNT);
    for query_id in tpch_query_ids() {
        let fallback_row = report_row(&fallback, query_id);
        let planning_row = planning_row(&planning, query_id);
        assert_eq!(
            fallback_row.native_candidate_shape_family, planning_row.native_candidate_shape_family,
            "{query_id} should preserve the same native candidate family across fallback-pressure and bounded planning"
        );

        match fallback_row.outcome {
            NativeFallbackPressureOutcome::Rewritten if matches!(query_id, "q01" | "q06") => {
                assert_eq!(planning_row.outcome, NativeDataFusionPlanningOutcome::AcceptedReplaced);
                assert_eq!(planning_row.reason_code, None);
                assert_eq!(planning_row.post_native_execution_path, NativeDataFusionPostNativeExecutionPath::Native);
            },
            NativeFallbackPressureOutcome::Rewritten if EXPECTED_BOUNDED_TPCH_JOIN_ACCEPTED.contains(&query_id) => {
                assert_eq!(
                    planning_row.outcome,
                    NativeDataFusionPlanningOutcome::AcceptedReplaced,
                    "{query_id} should be bounded-admitted: reason={:?} category={:?}",
                    planning_row.reason_code,
                    planning_row.unsupported_category
                );
                assert_eq!(planning_row.reason_code, None);
                assert_eq!(planning_row.post_native_execution_path, NativeDataFusionPostNativeExecutionPath::Native);
            },
            NativeFallbackPressureOutcome::Rewritten => {
                assert_eq!(planning_row.outcome, NativeDataFusionPlanningOutcome::RejectedBeforeExecution);
                assert_eq!(planning_row.reason_code.as_deref(), Some("native_bounded_join_unsupported"));
                assert_ne!(planning_row.post_native_execution_path, NativeDataFusionPostNativeExecutionPath::Native);
            },
            NativeFallbackPressureOutcome::Rejected => {
                assert_eq!(planning_row.outcome, NativeDataFusionPlanningOutcome::RejectedBeforeExecution);
                assert_eq!(
                    planning_row.reason_code, fallback_row.reason_code,
                    "{query_id} should keep the same precise pre-execution rejection reason"
                );
                assert_eq!(
                    planning_row.unsupported_category, fallback_row.unsupported_category,
                    "{query_id} should keep the same unsupported category"
                );
            },
        }
    }

    fs::remove_dir_all(dir).unwrap();
}

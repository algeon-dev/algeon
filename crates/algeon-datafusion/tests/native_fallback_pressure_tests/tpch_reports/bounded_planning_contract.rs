use crate::fallback_pressure::*;

#[tokio::test]
async fn tpch_progression_contract_pairs_fallback_pressure_and_bounded_planning() {
    let dir = unique_temp_dir("algeon-datafusion-tpch-bounded-planning-contract");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();

    let fallback = NativeFallbackPressureReport::collect(
        tpch_fallback_pressure_workload_items(&dir, tpch_report_session_config()).await.unwrap(),
    );
    let planning = NativeDataFusionPlanningReport::collect(
        tpch_datafusion_planning_workload_items(&dir, tpch_report_session_config()).await.unwrap(),
    );

    assert_eq!(fallback.rows.len(), TPCH_QUERY_COUNT);
    assert_eq!(planning_query_count(&planning), TPCH_QUERY_COUNT);
    for query_id in tpch_query_ids() {
        let fallback_row = report_row(&fallback, query_id);
        let whole_plan_row = whole_plan_planning_row(&planning, query_id);
        let replacement_row = planning_replacement_row(&planning, query_id);
        if replacement_row.final_disposition == Some(FinalPlanDisposition::Native) {
            assert_eq!(
                fallback_row.native_candidate_shape_family, replacement_row.native_candidate_shape_family,
                "{query_id} should preserve the same native candidate family across whole-plan bounded planning"
            );
        } else {
            assert!(
                replacement_row.native_candidate_shape_family.is_some(),
                "{query_id} mixed-plan planning should still expose the selected candidate family"
            );
        }

        match fallback_row.outcome {
            Some(CandidateOutcomeKind::Selected) if matches!(query_id, "q01" | "q06") => {
                assert_eq!(whole_plan_row.outcome, Some(CandidateOutcomeKind::Selected));
                assert_eq!(whole_plan_row.reason_code, None);
                assert_eq!(whole_plan_row.final_disposition, Some(FinalPlanDisposition::Native));
            },
            Some(CandidateOutcomeKind::Selected) if EXPECTED_BOUNDED_TPCH_JOIN_ACCEPTED.contains(&query_id) => {
                assert_eq!(
                    replacement_row.outcome,
                    Some(CandidateOutcomeKind::Selected),
                    "{query_id} should be bounded-admitted: reason={:?} category={:?}",
                    replacement_row.reason_code,
                    replacement_row.unsupported_category
                );
                assert_eq!(replacement_row.reason_code, None);
                assert!(
                    matches!(
                        replacement_row.final_disposition,
                        Some(FinalPlanDisposition::Native | FinalPlanDisposition::Mixed)
                    ),
                    "{query_id} should be admitted by whole-plan native or mixed-plan execution"
                );
            },
            Some(CandidateOutcomeKind::Selected) => {
                assert_eq!(whole_plan_row.outcome, Some(CandidateOutcomeKind::NotSupported));
                assert_eq!(whole_plan_row.reason_code.as_deref(), Some("native_bounded_join_unsupported"));
                assert_ne!(whole_plan_row.final_disposition, Some(FinalPlanDisposition::Native));
            },
            Some(CandidateOutcomeKind::NotSupported) => {
                assert_eq!(whole_plan_row.outcome, Some(CandidateOutcomeKind::NotSupported));
                assert_eq!(
                    whole_plan_row.reason_code, fallback_row.reason_code,
                    "{query_id} should keep the same precise pre-execution rejection reason"
                );
                assert_eq!(
                    whole_plan_row.unsupported_category, fallback_row.unsupported_category,
                    "{query_id} should keep the same unsupported category"
                );
            },
            Some(CandidateOutcomeKind::NotSelectedByCost) => {
                panic!("fallback-pressure reports do not apply cost selection")
            },
            Some(CandidateOutcomeKind::Deferred) => {
                panic!("fallback-pressure reports do not defer candidates")
            },
            None => panic!("terminal planning failure is outside this physical-plan workload"),
        }
    }

    fs::remove_dir_all(dir).unwrap();
}

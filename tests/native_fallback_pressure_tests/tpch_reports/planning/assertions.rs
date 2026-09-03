use super::*;

#[path = "assertions/highlighted.rs"]
mod highlighted;
#[path = "assertions/rows.rs"]
mod rows;

use highlighted::assert_highlighted_planning_rows_and_tsv;
use rows::{admission_contract_row_is_family_path_free, assert_planning_rows};

pub(super) fn assert_tpch_datafusion_planning_report_contract(
    report: &NativeDataFusionPlanningReport,
    metadata_fingerprint: &str,
) {
    assert_eq!(planning_query_count(report), TPCH_QUERY_COUNT);
    assert!(report.rows.len() >= TPCH_QUERY_COUNT);
    assert_duplicate_query_rows_are_mixed_plan_fragments(report);
    let accepted_primary_count = tpch_query_ids()
        .filter(|query_id| planning_replacement_row(report, query_id).outcome == Some(CandidateOutcomeKind::Selected))
        .count();
    assert_eq!(accepted_primary_count, EXPECTED_BOUNDED_TPCH_ACCEPTED.len());
    assert_eq!(TPCH_QUERY_COUNT - accepted_primary_count, TPCH_QUERY_COUNT - EXPECTED_BOUNDED_TPCH_ACCEPTED.len());

    let actual_ids = report
        .rows
        .iter()
        .map(|row| row.query_id.as_str())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let expected_ids = tpch_query_ids().collect::<Vec<_>>();
    assert_eq!(actual_ids, expected_ids);

    assert_planning_rows(report);
    assert_highlighted_planning_rows_and_tsv(report, metadata_fingerprint);
}

fn assert_duplicate_query_rows_are_mixed_plan_fragments(report: &NativeDataFusionPlanningReport) {
    for query_id in tpch_query_ids() {
        let rows = planning_rows(report, query_id);
        if rows.len() <= 1 {
            continue;
        }
        let whole_plan = whole_plan_planning_row(report, query_id);
        assert_eq!(
            whole_plan.outcome,
            Some(CandidateOutcomeKind::NotSupported),
            "{query_id} duplicate diagnostics should keep the whole-plan bounded rejection as the first row"
        );
        assert!(
            whole_plan.rejected_before_execution,
            "{query_id} whole-plan duplicate diagnostic must be marked rejected before execution"
        );
        assert!(
            whole_plan.reason_code.is_some() && whole_plan.unsupported_category.is_some(),
            "{query_id} whole-plan duplicate diagnostic must carry rejection details"
        );
        assert!(
            rows.iter().all(|row| row.final_disposition == Some(FinalPlanDisposition::Mixed)),
            "{query_id} duplicate rows must be scoped to mixed-plan planning diagnostics"
        );
        let accepted =
            rows.iter().filter(|row| row.outcome == Some(CandidateOutcomeKind::Selected)).collect::<Vec<_>>();
        assert!(
            !accepted.is_empty(),
            "{query_id} duplicate mixed-plan diagnostics must include accepted fragment rows"
        );
        for accepted in accepted {
            assert!(!accepted.rejected_before_execution, "{query_id} accepted fragment must not be marked rejected");
            assert_eq!(accepted.reason_code, None, "{query_id} accepted fragment must not carry a rejection reason");
            assert_eq!(
                accepted.unsupported_category, None,
                "{query_id} accepted fragment must not carry an unsupported category"
            );
            assert!(
                accepted.native_candidate_shape.is_some() && accepted.native_candidate_shape_family.is_some(),
                "{query_id} accepted fragment must carry candidate diagnostics"
            );
        }
        let rejected =
            rows.iter().filter(|row| row.outcome == Some(CandidateOutcomeKind::NotSupported)).collect::<Vec<_>>();
        assert!(
            !rejected.is_empty(),
            "{query_id} duplicate mixed-plan diagnostics must include rejected whole-plan candidate rows"
        );
        for rejected in rejected {
            assert!(
                rejected.rejected_before_execution,
                "{query_id} rejected duplicate diagnostic must be marked rejected before execution"
            );
            assert!(
                rejected.reason_code.is_some(),
                "{query_id} rejected duplicate diagnostic must carry a reason code"
            );
            assert!(
                rejected.unsupported_category.is_some(),
                "{query_id} rejected duplicate diagnostic must carry an unsupported category"
            );
        }
    }
}

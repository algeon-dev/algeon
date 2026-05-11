use super::*;

#[path = "assertions/highlighted.rs"]
mod highlighted;
#[path = "assertions/rows.rs"]
mod rows;

use highlighted::assert_highlighted_planning_rows_and_tsv;
use rows::{assert_planning_rows, strict_contract_row_is_family_path_free};

pub(super) fn assert_tpch_datafusion_planning_report_contract(
    report: &NativeDataFusionPlanningReport,
    metadata_fingerprint: &str,
) {
    assert_eq!(report.rows.len(), TPCH_QUERY_COUNT);
    assert_eq!(report.accepted_count(), EXPECTED_BOUNDED_TPCH_ACCEPTED.len());
    assert_eq!(report.rejected_count(), TPCH_QUERY_COUNT - EXPECTED_BOUNDED_TPCH_ACCEPTED.len());

    let actual_ids = report.rows.iter().map(|row| row.query_id.as_str()).collect::<Vec<_>>();
    let expected_ids = tpch_query_ids().collect::<Vec<_>>();
    assert_eq!(actual_ids, expected_ids);

    assert_planning_rows(report);
    assert_highlighted_planning_rows_and_tsv(report, metadata_fingerprint);
}

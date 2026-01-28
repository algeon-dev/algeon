use super::*;

#[path = "assertions/highlighted.rs"]
mod highlighted;
#[path = "assertions/rows.rs"]
mod rows;

use highlighted::assert_highlighted_planning_rows_and_tsv;
use rows::assert_planning_rows;

pub(super) struct PlanningReportExpectations {
    execution_families: String,
    modeled_reservations: String,
    supported_shape: String,
    mixed_capability_detail: &'static str,
    mixed_multi_join_capability: &'static str,
    mixed_multi_join_capability_detail: &'static str,
    keyless_input_subplan_capability: &'static str,
    keyless_input_subplan_capability_detail: &'static str,
    keyless_aggregate_over_join_subplan_inputs_capability: &'static str,
    keyless_aggregate_over_join_subplan_inputs_capability_detail: &'static str,
    mixed_size_data_alias_compatibility: &'static str,
}

impl PlanningReportExpectations {
    fn from_contract(contract: &BoundedJoinReservationContract) -> Self {
        Self {
            execution_families: contract.execution_families_csv(),
            modeled_reservations: contract.modeled_reservation_kinds_csv(),
            supported_shape: contract.supported_shape_requirement(),
            mixed_capability_detail: "mixed_keyed_residual_root_join_types=inner,left,full,left_semi,left_anti",
            mixed_multi_join_capability: "multi_join_mixed_keyed_residual_inner_left_semi_left_anti_pipeline",
            mixed_multi_join_capability_detail: "mixed_keyed_residual_multi_join_types=inner,left_semi,left_anti",
            keyless_input_subplan_capability: "keyless_conditional_join_input_subplan_pipeline",
            keyless_input_subplan_capability_detail: "root_join=keyless_conditional_inner;input_subplans=source_filter_projection_or_projection_wrapped_left_anti_hash_join_or_scalar_global_aggregate",
            keyless_aggregate_over_join_subplan_inputs_capability: "keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline",
            keyless_aggregate_over_join_subplan_inputs_capability_detail: "root_join=keyless_conditional_inner;left_input=projection_wrapped_grouped_aggregate_over_bounded_join_subplan;right_input=projection_wrapped_scalar_global_aggregate_over_bounded_join_subplan;root_output=left_only;post_root=projection_sort",
            mixed_size_data_alias_compatibility: "mixed_inner_left_join_size_data_values_are_legacy_aliases_for_mixed_keyed_residual_inner_left_full_left_semi_left_anti_size_data",
        }
    }
}

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

    let contract = BoundedJoinReservationContract;
    let expected = PlanningReportExpectations::from_contract(&contract);
    assert_planning_rows(report, &expected);
    assert_highlighted_planning_rows_and_tsv(report, metadata_fingerprint, &expected);
}

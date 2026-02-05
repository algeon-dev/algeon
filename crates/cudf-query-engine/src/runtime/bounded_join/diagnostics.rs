use std::collections::BTreeMap;

use super::*;

impl BoundedJoinReservationContract {
    /// Human/report-facing unsupported-shape requirement text.
    pub fn supported_shape_requirement(self) -> String {
        "requires_strict_bounded_capability_admission".to_string()
    }

    /// CSV of all reservation kinds in requirement order.
    pub fn reservation_kinds_csv(self) -> String {
        csv_join(self.requirements().iter().map(|requirement| requirement.reservation_kind().as_str()))
    }

    /// CSV of `reservation_kind:status` pairs in requirement order.
    pub fn reservation_statuses_csv(self) -> String {
        csv_join(self.requirements().iter().map(|requirement| {
            format!("{}:{}", requirement.reservation_kind().as_str(), requirement.status().as_str())
        }))
    }

    /// CSV of reservation kinds that still block bounded execution.
    pub fn blocking_reservation_kinds_csv(self) -> String {
        csv_join(self.requirements().iter().filter_map(|requirement| {
            (requirement.status() != BoundedJoinReservationStatus::Modeled)
                .then_some(requirement.reservation_kind().as_str())
        }))
    }

    /// CSV of reservation kinds currently modeled by bounded execution.
    pub fn modeled_reservation_kinds_csv(self) -> String {
        csv_join(self.requirements().iter().filter_map(|requirement| {
            (requirement.status() == BoundedJoinReservationStatus::Modeled)
                .then_some(requirement.reservation_kind().as_str())
        }))
    }

    /// CSV of reservation kinds deferred until another prerequisite lands.
    pub fn deferred_reservation_kinds_csv(self) -> String {
        csv_join(self.requirements().iter().filter_map(|requirement| {
            (requirement.status() == BoundedJoinReservationStatus::DeferredUntilPrerequisite)
                .then_some(requirement.reservation_kind().as_str())
        }))
    }

    /// CSV of lower-level contracts that currently block bounded execution.
    pub fn lower_level_blockers_csv(self) -> String {
        csv_join(self.requirements().iter().filter_map(|requirement| {
            (requirement.status() == BoundedJoinReservationStatus::BlockedByLowerLevel)
                .then_some(requirement.required_contract())
        }))
    }

    /// CSV of `reservation_kind:required_contract:detail` rows.
    pub fn requirement_details_csv(self) -> String {
        csv_join(self.requirements().iter().map(|requirement| {
            format!(
                "{}:{}:{}",
                requirement.reservation_kind().as_str(),
                requirement.required_contract(),
                requirement.detail()
            )
        }))
    }

    /// Builds the bounded join reservation fact map.
    ///
    /// Fact keys and many values here are consumed by reports and tests. This
    /// map documents cudf-rs observability/reservation contracts that bounded
    /// execution relies on, including exact hash-table sizing, gather-map
    /// sizing, retained size-data observability, and explicit sink ownership.
    pub fn facts(self, boundary_policy: ExecutionBoundaryPolicy) -> BTreeMap<&'static str, String> {
        BTreeMap::from([
            ("boundary_policy", boundary_policy.to_string()),
            ("source_read_boundary", boundary_policy.source_read().to_string()),
            ("build_materialization_boundary", boundary_policy.build_materialization().to_string()),
            ("probe_execution_boundary", boundary_policy.probe_execution().to_string()),
            ("result_materialization_boundary", boundary_policy.result_materialization().to_string()),
            ("bounded_join_reservation_model", self.model_status().to_string()),
            ("bounded_join_execution_status", self.execution_status().to_string()),
            ("bounded_join_enabled", self.enabled().to_string()),
            ("bounded_join_reservation_kinds", self.reservation_kinds_csv()),
            ("bounded_join_reservation_statuses", self.reservation_statuses_csv()),
            ("blocked_reservations", self.blocking_reservation_kinds_csv()),
            ("modeled_reservations", self.modeled_reservation_kinds_csv()),
            ("deferred_reservations", self.deferred_reservation_kinds_csv()),
            ("bounded_join_lower_level_blockers", self.lower_level_blockers_csv()),
            ("bounded_join_reservation_requirements", self.requirement_details_csv()),
            ("cudf_rs_hash_table_observability", "hash_table_device_allocated_size_bytes".to_string()),
            ("cudf_rs_gather_map_observability", "join_indices_device_allocated_size_bytes".to_string()),
            (
                "cudf_rs_hash_join_gather_map_reservation_contract",
                "hash_join_gather_map_reservation_size_bytes".to_string(),
            ),
            ("cudf_rs_hash_join_output_size_observability", "hash_join_exact_output_size".to_string()),
            ("cudf_rs_hash_join_probe_output_size_hint", "hash_join_probe_with_output_size".to_string()),
            ("cudf_rs_filtered_join_hash_table_observability", "filtered_join_device_allocated_size_bytes".to_string()),
            ("cudf_rs_filtered_join_probe_state_contract", "filtered_join_retained_probe_state".to_string()),
            (
                "cudf_rs_filtered_join_probe_state_observability",
                "filtered_join_probe_device_allocated_size_bytes".to_string(),
            ),
            ("cudf_rs_filtered_join_output_size_observability", "filtered_join_probe_state_output_size".to_string()),
            (
                "cudf_rs_filtered_join_probe_materialization_contract",
                "filtered_join_probe_state_materialize_indices".to_string(),
            ),
            ("cudf_rs_conditional_join_output_size_observability", "conditional_join_exact_output_size".to_string()),
            ("cudf_rs_conditional_join_output_size_hint", "conditional_join_indices_with_output_size".to_string()),
            ("cudf_rs_mixed_join_output_size_observability", "mixed_inner_left_join_size_data_output_size".to_string()),
            (
                "cudf_rs_mixed_join_size_data_observability",
                "mixed_inner_left_join_size_data_device_allocated_size_bytes".to_string(),
            ),
            (
                "cudf_rs_mixed_join_gather_map_reservation_contract",
                "mixed_inner_left_join_size_data_gather_map_reservation_size_bytes".to_string(),
            ),
            ("cudf_rs_mixed_join_output_size_hint", "mixed_inner_left_join_size_data_materialize_indices".to_string()),
            (
                "cudf_rs_mixed_join_size_data_alias_compatibility",
                "mixed_inner_left_join_size_data_values_are_legacy_aliases_for_mixed_keyed_residual_inner_left_full_left_semi_left_anti_size_data".to_string(),
            ),
            ("cudf_rs_mixed_join_remaining_blockers", "".to_string()),
            (
                "bounded_join_mixed_keyed_residual_capability",
                "root_mixed_keyed_residual_join".to_string(),
            ),
            (
                "bounded_join_mixed_keyed_residual_capability_detail",
                "mixed_keyed_residual_root_join_types=inner,left,full,left_semi,left_anti".to_string(),
            ),
            (
                "bounded_join_mixed_keyed_residual_multi_join_capability",
                "multi_join_mixed_keyed_residual_inner_left_semi_left_anti_pipeline".to_string(),
            ),
            (
                "bounded_join_mixed_keyed_residual_multi_join_capability_detail",
                "mixed_keyed_residual_multi_join_types=inner,left_semi,left_anti".to_string(),
            ),
            (
                "bounded_join_keyless_conditional_join_input_subplan_pipeline_capability",
                "keyless_conditional_join_input_subplan_pipeline".to_string(),
            ),
            (
                "bounded_join_keyless_conditional_join_input_subplan_pipeline_capability_detail",
                "root_join=keyless_conditional_inner;input_subplans=source_filter_projection_or_projection_wrapped_left_anti_hash_join_or_scalar_global_aggregate".to_string(),
            ),
            (
                "bounded_join_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline_capability",
                "keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline".to_string(),
            ),
            (
                "bounded_join_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline_capability_detail",
                "root_join=keyless_conditional_inner;left_input=projection_wrapped_grouped_aggregate_over_bounded_join_subplan;right_input=projection_wrapped_scalar_global_aggregate_over_bounded_join_subplan;root_output=left_only;post_root=projection_sort".to_string(),
            ),
            (
                "cudf_rs_mixed_join_pair_materialization_contract",
                "mixed_join_retained_size_data_materialize_indices".to_string(),
            ),
            (
                "cudf_rs_mixed_join_single_side_materialization_contract",
                "mixed_join_retained_size_data_materialize_single_indices".to_string(),
            ),
            (
                "cudf_rs_projected_cross_join_reservation_contract",
                "projected_cross_join_reservation_size_bytes".to_string(),
            ),
            (
                "cudf_rs_conditional_full_join_output_size_observability",
                "conditional_full_join_retained_size_data_output_size".to_string(),
            ),
            (
                "cudf_rs_conditional_full_join_size_data_contract",
                "conditional_full_join_retained_size_data".to_string(),
            ),
            (
                "cudf_rs_conditional_full_join_gather_map_reservation_contract",
                "conditional_full_join_size_data_gather_map_reservation_size_bytes".to_string(),
            ),
            (
                "cudf_rs_conditional_full_join_materialization_contract",
                "conditional_full_join_retained_size_data_materialize_indices".to_string(),
            ),
            (
                "cudf_rs_dataframe_gather_output_reservation_contract",
                "dataframe_gather_output_reservation_size_bytes".to_string(),
            ),
            (
                "cudf_rs_hash_join_pre_build_reservation_contract",
                "hash_join_pre_build_reservation_size_bytes".to_string(),
            ),
            ("cudf_rs_hash_join_pre_build_reservation_status", "available_exact_flat_hash_join_builds".to_string()),
            ("missing_reservations", self.blocking_reservation_kinds_csv()),
        ])
    }

    /// Fact map attached to bounded join rejections.
    pub fn rejection_facts(self, boundary_policy: ExecutionBoundaryPolicy) -> BTreeMap<&'static str, String> {
        self.facts(boundary_policy)
    }

    /// Fact map for plans outside the supported bounded join shape catalogue.
    pub fn unsupported_shape_facts(self, boundary_policy: ExecutionBoundaryPolicy) -> BTreeMap<&'static str, String> {
        let mut facts = self.rejection_facts(boundary_policy);
        facts.insert("bounded_join_execution_status", "unsupported_plan_shape".to_string());
        facts.insert("bounded_join_unsupported_shape", self.supported_shape_requirement());
        facts
    }
}

fn csv_join<I, T>(values: I) -> String
where
    I: IntoIterator<Item = T>,
    T: ToString,
{
    let mut output = String::new();
    for value in values {
        if !output.is_empty() {
            output.push(',');
        }
        output.push_str(&value.to_string());
    }
    output
}

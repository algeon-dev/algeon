use std::collections::BTreeMap;

use super::{q15::matches_keyed_hash_join_resident_subplan_pipeline, *};
use crate::runtime::{BoundedJoinReservationContract, ExecutionBoundaryPolicy};

const KEYED_HASH_JOIN_RESIDENT_SUBPLAN_PIPELINE_FAMILY: &str = "keyed_hash_join_resident_subplan_pipeline";
const KEYED_HASH_JOIN_RESIDENT_SUBPLAN_PIPELINE_BLOCKER: &str =
    "bounded_keyed_hash_join_resident_subplan_lower_level_contracts";
const KEYED_HASH_JOIN_RESIDENT_SUBPLAN_PIPELINE_SHAPE_CONTRACT: &str =
    "sort_root_keyed_inner_hash_join_with_projection_wrapped_keyed_inner_join_and_scalar_revenue_aggregate";
const KEYED_HASH_JOIN_RESIDENT_SUBPLAN_PIPELINE_LOWER_LEVEL_CONTRACT: &str =
    "source_contract,aggregate_contract,final_sort_contract";
const KEYED_HASH_JOIN_RESIDENT_SUBPLAN_PIPELINE_BLOCKED_RESERVATIONS: &str = "source_read,operator_output,final_sort";
const KEYED_HASH_JOIN_RESIDENT_SUBPLAN_PIPELINE_RESERVATION_STATUSES: &str = "source_read:blocked_by_lower_level,operator_output:blocked_by_lower_level,build_side:modeled,hash_table:modeled,probe_side:modeled,filtered_probe_state:modeled,mixed_size_data:modeled,gather_map:modeled,join_output:modeled,final_sort:blocked_by_lower_level,materialization:modeled";
const KEYED_HASH_JOIN_RESIDENT_SUBPLAN_PIPELINE_RESERVATION_REQUIREMENTS: &str = "source_read:source_contract:blocked_by_lower_level,operator_output:aggregate_contract:blocked_by_lower_level,build_side:query_engine_build_side_source_reservation:modeled_before_execution,hash_table:hash_join_pre_build_reservation_size_bytes:modeled_by_cudf_rs_pre_build_sizing,probe_side:query_engine_probe_side_source_reservation:modeled_by_ordered_bounded_join_execution_path,filtered_probe_state:filtered_join_retained_probe_state:modeled_by_cudf_rs_filtered_probe_state_observability,mixed_size_data:mixed_keyed_residual_or_conditional_full_retained_size_data:modeled_by_cudf_rs_retained_join_size_data_observability,gather_map:join_gather_map_reservation_size_bytes:modeled_by_cudf_rs_join_gather_map_sizing,join_output:dataframe_gather_output_reservation_size_bytes+projected_cross_join_reservation_size_bytes:modeled_by_cudf_rs_dataframe_gather_or_projected_cross_output_sizing,final_sort:final_sort_contract:blocked_by_lower_level,materialization:query_engine_explicit_sink_materialization_reservation:modeled_by_explicit_sink_boundary";
const KEYED_HASH_JOIN_RESIDENT_SUBPLAN_PIPELINE_MODELED_RESERVATIONS: &str =
    "build_side,hash_table,probe_side,filtered_probe_state,mixed_size_data,gather_map,join_output,materialization";
const MIXED_KEYED_RESIDUAL_JOIN_RESIDENT_GROUPED_AGGREGATE_INPUT_PIPELINE_FAMILY: &str =
    "mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline";
const MIXED_KEYED_RESIDUAL_JOIN_RESIDENT_GROUPED_AGGREGATE_INPUT_PIPELINE_BLOCKER: &str =
    "bounded_mixed_keyed_residual_join_resident_grouped_aggregate_input_lower_level_contracts";
const MIXED_KEYED_RESIDUAL_JOIN_RESIDENT_GROUPED_AGGREGATE_INPUT_PIPELINE_SHAPE_CONTRACT: &str = "projected_global_aggregate_over_keyed_residual_inner_hash_join_with_projection_wrapped_join_subtree_and_projection_wrapped_grouped_local_parquet_aggregate_input";
const MIXED_KEYED_RESIDUAL_JOIN_RESIDENT_GROUPED_AGGREGATE_INPUT_PIPELINE_LOWER_LEVEL_CONTRACT: &str =
    "source_contract,aggregate_contract";
const MIXED_KEYED_RESIDUAL_JOIN_RESIDENT_GROUPED_AGGREGATE_INPUT_PIPELINE_CONTRACT_MODES: &str =
    "source_contract:hard_budgeted_report,aggregate_contract:hard_budgeted_report";
const MIXED_KEYED_RESIDUAL_JOIN_RESIDENT_GROUPED_AGGREGATE_INPUT_PIPELINE_UNRESOLVED_PREFLIGHT_CONTRACTS: &str = "source_contract:libcudf_parquet_decoded_output_preflight_unavailable,aggregate_contract:libcudf_groupby_aggregate_temporary_workspace_preflight_unavailable";
const MIXED_KEYED_RESIDUAL_JOIN_RESIDENT_GROUPED_AGGREGATE_INPUT_PIPELINE_CONTRACT_REPORT_FIELDS: &str = "source_contract:allocator_high_water_bytes+retained_bytes+output_bytes+output_rows+context_facts,aggregate_contract:allocator_high_water_bytes+retained_bytes+output_bytes+output_rows";
const MIXED_KEYED_RESIDUAL_JOIN_RESIDENT_GROUPED_AGGREGATE_INPUT_PIPELINE_BLOCKED_RESERVATIONS: &str = "";
const MIXED_KEYED_RESIDUAL_JOIN_RESIDENT_GROUPED_AGGREGATE_INPUT_PIPELINE_RESERVATION_KINDS: &str = "source_read,operator_output,build_side,hash_table,probe_side,filtered_probe_state,mixed_size_data,gather_map,join_output,materialization";
const MIXED_KEYED_RESIDUAL_JOIN_RESIDENT_GROUPED_AGGREGATE_INPUT_PIPELINE_RESERVATION_STATUSES: &str = "source_read:modeled,operator_output:modeled,build_side:modeled,hash_table:modeled,probe_side:modeled,filtered_probe_state:modeled,mixed_size_data:modeled,gather_map:modeled,join_output:modeled,materialization:modeled";
const MIXED_KEYED_RESIDUAL_JOIN_RESIDENT_GROUPED_AGGREGATE_INPUT_PIPELINE_RESERVATION_REQUIREMENTS: &str = "source_read:source_contract:modeled_by_cudf_rs_hard_budgeted_report_with_unresolved_preflight,operator_output:aggregate_contract:modeled_by_cudf_rs_hard_budgeted_report_with_unresolved_preflight,build_side:query_engine_build_side_source_reservation:modeled_before_execution,hash_table:hash_join_pre_build_reservation_size_bytes:modeled_by_cudf_rs_pre_build_sizing,probe_side:query_engine_probe_side_source_reservation:modeled_by_ordered_bounded_join_execution_path,filtered_probe_state:filtered_join_retained_probe_state:modeled_by_cudf_rs_filtered_probe_state_observability,mixed_size_data:mixed_keyed_residual_or_conditional_full_retained_size_data:modeled_by_cudf_rs_retained_join_size_data_observability,gather_map:join_gather_map_reservation_size_bytes:modeled_by_cudf_rs_join_gather_map_sizing,join_output:dataframe_gather_output_reservation_size_bytes+projected_cross_join_reservation_size_bytes:modeled_by_cudf_rs_dataframe_gather_or_projected_cross_output_sizing,materialization:query_engine_explicit_sink_materialization_reservation:modeled_by_explicit_sink_boundary";
const MIXED_KEYED_RESIDUAL_JOIN_RESIDENT_GROUPED_AGGREGATE_INPUT_PIPELINE_MODELED_RESERVATIONS: &str = "source_read,operator_output,build_side,hash_table,probe_side,filtered_probe_state,mixed_size_data,gather_map,join_output,materialization";

pub(crate) fn bounded_join_unsupported_shape_facts(
    plan: &QueryPlan,
    boundary_policy: ExecutionBoundaryPolicy,
) -> BTreeMap<&'static str, String> {
    let mut facts = BoundedJoinReservationContract.unsupported_shape_facts(boundary_policy);
    if matches_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline(plan) {
        facts.insert(
            "bounded_join_blocked_join_family",
            MIXED_KEYED_RESIDUAL_JOIN_RESIDENT_GROUPED_AGGREGATE_INPUT_PIPELINE_FAMILY.to_string(),
        );
        facts.insert(
            "bounded_join_specific_blocker",
            MIXED_KEYED_RESIDUAL_JOIN_RESIDENT_GROUPED_AGGREGATE_INPUT_PIPELINE_BLOCKER.to_string(),
        );
        facts.insert(
            "bounded_join_shape_blocker",
            MIXED_KEYED_RESIDUAL_JOIN_RESIDENT_GROUPED_AGGREGATE_INPUT_PIPELINE_BLOCKER.to_string(),
        );
        facts.insert(
            "blocked_reservations",
            MIXED_KEYED_RESIDUAL_JOIN_RESIDENT_GROUPED_AGGREGATE_INPUT_PIPELINE_BLOCKED_RESERVATIONS.to_string(),
        );
        facts.insert(
            "missing_reservations",
            MIXED_KEYED_RESIDUAL_JOIN_RESIDENT_GROUPED_AGGREGATE_INPUT_PIPELINE_BLOCKED_RESERVATIONS.to_string(),
        );
        facts.insert(
            "modeled_reservations",
            MIXED_KEYED_RESIDUAL_JOIN_RESIDENT_GROUPED_AGGREGATE_INPUT_PIPELINE_MODELED_RESERVATIONS.to_string(),
        );
        facts.insert(
            "bounded_join_reservation_kinds",
            MIXED_KEYED_RESIDUAL_JOIN_RESIDENT_GROUPED_AGGREGATE_INPUT_PIPELINE_RESERVATION_KINDS.to_string(),
        );
        facts.insert(
            "bounded_join_reservation_statuses",
            MIXED_KEYED_RESIDUAL_JOIN_RESIDENT_GROUPED_AGGREGATE_INPUT_PIPELINE_RESERVATION_STATUSES.to_string(),
        );
        facts.insert(
            "bounded_join_reservation_requirements",
            MIXED_KEYED_RESIDUAL_JOIN_RESIDENT_GROUPED_AGGREGATE_INPUT_PIPELINE_RESERVATION_REQUIREMENTS.to_string(),
        );
        facts.insert(
            "bounded_join_required_shape_contract",
            MIXED_KEYED_RESIDUAL_JOIN_RESIDENT_GROUPED_AGGREGATE_INPUT_PIPELINE_SHAPE_CONTRACT.to_string(),
        );
        facts.insert(
            "bounded_join_required_lower_level_contract",
            MIXED_KEYED_RESIDUAL_JOIN_RESIDENT_GROUPED_AGGREGATE_INPUT_PIPELINE_LOWER_LEVEL_CONTRACT.to_string(),
        );
        facts.insert(
            "bounded_join_lower_level_blockers",
            MIXED_KEYED_RESIDUAL_JOIN_RESIDENT_GROUPED_AGGREGATE_INPUT_PIPELINE_LOWER_LEVEL_CONTRACT.to_string(),
        );
    } else if matches_keyed_hash_join_resident_subplan_pipeline(plan) {
        facts.insert("bounded_join_blocked_join_family", KEYED_HASH_JOIN_RESIDENT_SUBPLAN_PIPELINE_FAMILY.to_string());
        facts.insert("bounded_join_specific_blocker", KEYED_HASH_JOIN_RESIDENT_SUBPLAN_PIPELINE_BLOCKER.to_string());
        facts.insert("bounded_join_shape_blocker", KEYED_HASH_JOIN_RESIDENT_SUBPLAN_PIPELINE_BLOCKER.to_string());
        facts
            .insert("blocked_reservations", KEYED_HASH_JOIN_RESIDENT_SUBPLAN_PIPELINE_BLOCKED_RESERVATIONS.to_string());
        facts
            .insert("missing_reservations", KEYED_HASH_JOIN_RESIDENT_SUBPLAN_PIPELINE_BLOCKED_RESERVATIONS.to_string());
        facts
            .insert("modeled_reservations", KEYED_HASH_JOIN_RESIDENT_SUBPLAN_PIPELINE_MODELED_RESERVATIONS.to_string());
        facts.insert(
            "bounded_join_reservation_statuses",
            KEYED_HASH_JOIN_RESIDENT_SUBPLAN_PIPELINE_RESERVATION_STATUSES.to_string(),
        );
        facts.insert(
            "bounded_join_reservation_requirements",
            KEYED_HASH_JOIN_RESIDENT_SUBPLAN_PIPELINE_RESERVATION_REQUIREMENTS.to_string(),
        );
        facts.insert(
            "bounded_join_required_shape_contract",
            KEYED_HASH_JOIN_RESIDENT_SUBPLAN_PIPELINE_SHAPE_CONTRACT.to_string(),
        );
        facts.insert(
            "bounded_join_required_lower_level_contract",
            KEYED_HASH_JOIN_RESIDENT_SUBPLAN_PIPELINE_LOWER_LEVEL_CONTRACT.to_string(),
        );
        facts.insert(
            "bounded_join_lower_level_blockers",
            KEYED_HASH_JOIN_RESIDENT_SUBPLAN_PIPELINE_LOWER_LEVEL_CONTRACT.to_string(),
        );
    } else if let Some(join) = first_unsupported_keyed_residual_semi_anti_join(plan) {
        facts.insert("bounded_join_specific_blocker", mixed_keyed_residual_left_only_blocker(plan, join).to_string());
        facts.insert("bounded_join_blocked_join_family", "mixed_keyed_residual_left_semi_left_anti".to_string());
        facts.insert("bounded_join_blocked_join_type", join.spec.join_type.to_string());
        facts.insert("bounded_join_required_shape_contract", mixed_keyed_residual_left_only_shape_contract(plan, join));
        facts.insert(
            "bounded_join_shape_blocker",
            "bounded_mixed_keyed_residual_left_only_join_shape_or_orientation".to_string(),
        );
    } else if let Some(join) = first_keyed_residual_full_join(plan) {
        facts.insert("bounded_join_specific_blocker", mixed_keyed_residual_full_blocker(plan, join).to_string());
        facts.insert("bounded_join_blocked_join_family", "mixed_keyed_residual_full".to_string());
        facts.insert("bounded_join_blocked_join_type", join.spec.join_type.to_string());
        facts.insert("bounded_join_required_shape_contract", mixed_keyed_residual_full_shape_contract(plan, join));
        facts.insert("bounded_join_shape_blocker", "bounded_mixed_keyed_residual_full_join_shape".to_string());
    } else if let Some(join) = first_keyed_residual_left_join(plan) {
        facts.insert("bounded_join_specific_blocker", mixed_keyed_residual_left_blocker(plan, join).to_string());
        facts.insert("bounded_join_blocked_join_family", "mixed_keyed_residual_left".to_string());
        facts.insert("bounded_join_blocked_join_type", join.spec.join_type.to_string());
        facts.insert("bounded_join_required_shape_contract", mixed_keyed_residual_left_shape_contract(plan, join));
        facts.insert(
            "bounded_join_shape_blocker",
            "bounded_mixed_keyed_residual_left_join_shape_or_orientation".to_string(),
        );
    }
    facts
}

pub(crate) fn bounded_join_modeled_facts(
    plan: &QueryPlan,
    boundary_policy: ExecutionBoundaryPolicy,
) -> BTreeMap<&'static str, String> {
    let mut facts = BoundedJoinReservationContract.facts(boundary_policy);
    if analyze_bounded_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline(plan).is_ok() {
        facts.insert(
            "bounded_join_execution_family",
            MIXED_KEYED_RESIDUAL_JOIN_RESIDENT_GROUPED_AGGREGATE_INPUT_PIPELINE_FAMILY.to_string(),
        );
        facts.insert(
            "bounded_join_required_shape_contract",
            MIXED_KEYED_RESIDUAL_JOIN_RESIDENT_GROUPED_AGGREGATE_INPUT_PIPELINE_SHAPE_CONTRACT.to_string(),
        );
        facts.insert(
            "bounded_join_contract_modes",
            MIXED_KEYED_RESIDUAL_JOIN_RESIDENT_GROUPED_AGGREGATE_INPUT_PIPELINE_CONTRACT_MODES.to_string(),
        );
        facts.insert(
            "bounded_join_unresolved_preflight_contracts",
            MIXED_KEYED_RESIDUAL_JOIN_RESIDENT_GROUPED_AGGREGATE_INPUT_PIPELINE_UNRESOLVED_PREFLIGHT_CONTRACTS
                .to_string(),
        );
        facts.insert(
            "bounded_join_contract_report_fields",
            MIXED_KEYED_RESIDUAL_JOIN_RESIDENT_GROUPED_AGGREGATE_INPUT_PIPELINE_CONTRACT_REPORT_FIELDS.to_string(),
        );
        facts.insert(
            "bounded_join_reservation_kinds",
            MIXED_KEYED_RESIDUAL_JOIN_RESIDENT_GROUPED_AGGREGATE_INPUT_PIPELINE_RESERVATION_KINDS.to_string(),
        );
        facts.insert(
            "bounded_join_reservation_statuses",
            MIXED_KEYED_RESIDUAL_JOIN_RESIDENT_GROUPED_AGGREGATE_INPUT_PIPELINE_RESERVATION_STATUSES.to_string(),
        );
        facts.insert(
            "bounded_join_reservation_requirements",
            MIXED_KEYED_RESIDUAL_JOIN_RESIDENT_GROUPED_AGGREGATE_INPUT_PIPELINE_RESERVATION_REQUIREMENTS.to_string(),
        );
        facts.insert("blocked_reservations", String::new());
        facts.insert("missing_reservations", String::new());
        facts.insert(
            "modeled_reservations",
            MIXED_KEYED_RESIDUAL_JOIN_RESIDENT_GROUPED_AGGREGATE_INPUT_PIPELINE_MODELED_RESERVATIONS.to_string(),
        );
        facts.insert("bounded_join_lower_level_blockers", String::new());
    }
    facts
}

fn matches_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline(plan: &QueryPlan) -> bool {
    analyze_bounded_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline(plan).is_ok()
}

fn first_unsupported_keyed_residual_semi_anti_join(plan: &QueryPlan) -> Option<&JoinNode> {
    let multi_join = reachable_join_count(plan) > 1;
    plan.nodes().iter().find_map(|node| {
        let PlanNode::Join(join) = &node.kind else {
            return None;
        };
        if join.spec.keys.is_empty()
            || join.spec.residual_predicate.is_none()
            || !matches!(join.spec.join_type, JoinType::LeftSemi | JoinType::LeftAnti)
        {
            return None;
        }

        if multi_join {
            (!bounded_multi_join_mixed_keyed_residual_compatible(join)).then_some(join)
        } else {
            (!bounded_root_mixed_join_compatible(join) || root_join_leaf_blocker(plan, join).is_some()).then_some(join)
        }
    })
}

fn first_keyed_residual_full_join(plan: &QueryPlan) -> Option<&JoinNode> {
    plan.nodes().iter().find_map(|node| {
        let PlanNode::Join(join) = &node.kind else {
            return None;
        };
        (!join.spec.keys.is_empty() && join.spec.residual_predicate.is_some() && join.spec.join_type == JoinType::Full)
            .then_some(join)
    })
}

fn first_keyed_residual_left_join(plan: &QueryPlan) -> Option<&JoinNode> {
    plan.nodes().iter().find_map(|node| {
        let PlanNode::Join(join) = &node.kind else {
            return None;
        };
        (!join.spec.keys.is_empty() && join.spec.residual_predicate.is_some() && join.spec.join_type == JoinType::Left)
            .then_some(join)
    })
}

fn mixed_keyed_residual_left_only_blocker(plan: &QueryPlan, join: &JoinNode) -> &'static str {
    if reachable_join_count(plan) > 1 {
        if join.spec.build_side != JoinSide::Right {
            "multi_join_mixed_keyed_residual_semi_anti_requires_right_build_side"
        } else if join.spec.output_projection.is_empty() {
            "multi_join_mixed_keyed_residual_semi_anti_requires_left_projection"
        } else if join.spec.output_projection.iter().any(|column| column.side != JoinSide::Left) {
            "multi_join_mixed_keyed_residual_semi_anti_requires_left_only_projection"
        } else {
            "bounded_multi_join_mixed_keyed_residual_semi_anti_shape_unsupported"
        }
    } else if join.spec.build_side != JoinSide::Right {
        "root_mixed_keyed_residual_semi_anti_requires_right_build_side"
    } else if join.spec.output_projection.is_empty() {
        "root_mixed_keyed_residual_semi_anti_requires_left_projection"
    } else if join.spec.output_projection.iter().any(|column| column.side != JoinSide::Left) {
        "root_mixed_keyed_residual_semi_anti_requires_left_only_projection"
    } else if let Some(blocker) = root_join_leaf_blocker(plan, join) {
        blocker
    } else {
        "bounded_root_mixed_keyed_residual_semi_anti_shape_unsupported"
    }
}

fn mixed_keyed_residual_left_only_shape_contract(plan: &QueryPlan, join: &JoinNode) -> String {
    if reachable_join_count(plan) > 1 {
        if join.spec.build_side != JoinSide::Right
            || join.spec.output_projection.is_empty()
            || join.spec.output_projection.iter().any(|column| column.side != JoinSide::Left)
        {
            "multi_join_mixed_keyed_residual_left_only_join_requires_right_build_left_projection".to_string()
        } else {
            "multi_join_pipeline_requires_reusable_hash_filtered_semi_anti_or_mixed_keyed_residual_inner_left_semi_left_anti_joins".to_string()
        }
    } else if join.spec.build_side != JoinSide::Right
        || join.spec.output_projection.is_empty()
        || join.spec.output_projection.iter().any(|column| column.side != JoinSide::Left)
    {
        "root_mixed_keyed_residual_left_only_join_requires_right_build_left_projection".to_string()
    } else if root_join_leaf_blocker(plan, join).is_some() {
        "root_mixed_keyed_residual_join_requires_source_leaf_inputs".to_string()
    } else {
        BoundedJoinReservationContract.supported_shape_requirement()
    }
}

fn mixed_keyed_residual_full_blocker(plan: &QueryPlan, join: &JoinNode) -> &'static str {
    if reachable_join_count(plan) > 1 {
        "bounded_multi_join_mixed_keyed_residual_full_topology_unsupported"
    } else if join.spec.output_projection.is_empty() {
        "root_mixed_keyed_residual_full_requires_projected_output"
    } else if let Some(blocker) = root_join_leaf_blocker(plan, join) {
        blocker
    } else {
        "bounded_root_mixed_keyed_residual_full_shape_unsupported"
    }
}

fn mixed_keyed_residual_full_shape_contract(plan: &QueryPlan, join: &JoinNode) -> String {
    if reachable_join_count(plan) > 1 {
        "multi_join_pipeline_requires_reusable_hash_filtered_semi_anti_or_mixed_keyed_residual_inner_left_semi_left_anti_joins".to_string()
    } else if join.spec.output_projection.is_empty() {
        "root_mixed_keyed_residual_full_join_requires_projected_output".to_string()
    } else if root_join_leaf_blocker(plan, join).is_some() {
        "root_mixed_keyed_residual_join_requires_source_leaf_inputs".to_string()
    } else {
        BoundedJoinReservationContract.supported_shape_requirement()
    }
}

fn mixed_keyed_residual_left_blocker(plan: &QueryPlan, join: &JoinNode) -> &'static str {
    if reachable_join_count(plan) > 1 {
        "bounded_multi_join_mixed_keyed_residual_left_topology_unsupported"
    } else if join.spec.build_side != JoinSide::Right {
        "root_mixed_keyed_residual_left_requires_right_build_side"
    } else if join.spec.output_projection.is_empty() {
        "root_mixed_keyed_residual_left_requires_projected_output"
    } else if let Some(blocker) = root_join_leaf_blocker(plan, join) {
        blocker
    } else {
        "bounded_root_mixed_keyed_residual_left_shape_unsupported"
    }
}

fn mixed_keyed_residual_left_shape_contract(plan: &QueryPlan, join: &JoinNode) -> String {
    if reachable_join_count(plan) > 1 {
        "multi_join_pipeline_requires_reusable_hash_filtered_semi_anti_or_mixed_keyed_residual_inner_left_semi_left_anti_joins".to_string()
    } else if join.spec.build_side != JoinSide::Right {
        "root_mixed_keyed_residual_left_join_requires_right_build".to_string()
    } else if join.spec.output_projection.is_empty() {
        "root_mixed_keyed_residual_left_join_requires_projected_output".to_string()
    } else if root_join_leaf_blocker(plan, join).is_some() {
        "root_mixed_keyed_residual_join_requires_source_leaf_inputs".to_string()
    } else {
        BoundedJoinReservationContract.supported_shape_requirement()
    }
}

fn root_join_leaf_blocker(plan: &QueryPlan, join: &JoinNode) -> Option<&'static str> {
    join.inputs.iter().find_map(|input| root_join_input_leaf_blocker(plan, *input))
}

fn root_join_input_leaf_blocker(plan: &QueryPlan, node_id: NodeId) -> Option<&'static str> {
    let node = plan.node(node_id)?;
    match &node.kind {
        PlanNode::Source(_) => None,
        PlanNode::Filter(filter) => root_join_input_leaf_blocker(plan, filter.input),
        PlanNode::Projection(projection) => root_join_input_leaf_blocker(plan, projection.input),
        PlanNode::Join(_) => Some("bounded_root_join_nested_join_input_topology_unsupported"),
        PlanNode::Aggregate(_) => Some("bounded_root_join_aggregate_leaf_unsupported"),
        PlanNode::Sort(_) => Some("bounded_root_join_sort_leaf_unsupported"),
        PlanNode::Limit(_) => Some("bounded_root_join_limit_leaf_unsupported"),
    }
}

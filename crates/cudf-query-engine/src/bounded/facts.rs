use std::collections::BTreeMap;

use super::*;
use crate::runtime::{BoundedJoinReservationContract, ExecutionBoundaryPolicy};
const MIXED_KEYED_RESIDUAL_JOIN_RESIDENT_GROUPED_AGGREGATE_INPUT_PIPELINE_FAMILY: &str =
    "mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline";
const MIXED_KEYED_RESIDUAL_JOIN_RESIDENT_GROUPED_AGGREGATE_INPUT_PIPELINE_BLOCKER: &str =
    "bounded_mixed_keyed_residual_join_resident_grouped_aggregate_input_lower_level_contracts";
const MIXED_KEYED_RESIDUAL_JOIN_RESIDENT_GROUPED_AGGREGATE_INPUT_PIPELINE_SHAPE_CONTRACT: &str = "projected_global_aggregate_over_keyed_residual_inner_hash_join_with_projection_wrapped_join_subtree_and_projection_wrapped_grouped_local_parquet_aggregate_input";
const MIXED_KEYED_RESIDUAL_JOIN_RESIDENT_GROUPED_AGGREGATE_INPUT_PIPELINE_LOWER_LEVEL_CONTRACT: &str =
    "source_contract,aggregate_contract";
const MIXED_KEYED_RESIDUAL_JOIN_RESIDENT_GROUPED_AGGREGATE_INPUT_PIPELINE_BLOCKED_RESERVATIONS: &str = "";
const MIXED_KEYED_RESIDUAL_JOIN_RESIDENT_GROUPED_AGGREGATE_INPUT_PIPELINE_RESERVATION_KINDS: &str = "source_read,operator_output,build_side,hash_table,probe_side,filtered_probe_state,mixed_size_data,gather_map,join_output,materialization";
const MIXED_KEYED_RESIDUAL_JOIN_RESIDENT_GROUPED_AGGREGATE_INPUT_PIPELINE_RESERVATION_STATUSES: &str = "source_read:modeled,operator_output:modeled,build_side:modeled,hash_table:modeled,probe_side:modeled,filtered_probe_state:modeled,mixed_size_data:modeled,gather_map:modeled,join_output:modeled,materialization:modeled";
const MIXED_KEYED_RESIDUAL_JOIN_RESIDENT_GROUPED_AGGREGATE_INPUT_PIPELINE_RESERVATION_REQUIREMENTS: &str = "source_read:source_contract:modeled_by_cudf_rs_hard_budgeted_report_with_unresolved_preflight,operator_output:aggregate_contract:modeled_by_cudf_rs_hard_budgeted_report_with_unresolved_preflight,build_side:query_engine_build_side_source_reservation:modeled_before_execution,hash_table:hash_join_pre_build_reservation_size_bytes:modeled_by_cudf_rs_pre_build_sizing,probe_side:query_engine_probe_side_source_reservation:modeled_by_ordered_bounded_join_execution_path,filtered_probe_state:filtered_join_retained_probe_state:modeled_by_cudf_rs_filtered_probe_state_observability,mixed_size_data:mixed_keyed_residual_or_conditional_full_retained_size_data:modeled_by_cudf_rs_retained_join_size_data_observability,gather_map:join_gather_map_reservation_size_bytes:modeled_by_cudf_rs_join_gather_map_sizing,join_output:dataframe_gather_output_reservation_size_bytes+projected_cross_join_reservation_size_bytes:modeled_by_cudf_rs_dataframe_gather_or_projected_cross_output_sizing,materialization:query_engine_explicit_sink_materialization_reservation:modeled_by_explicit_sink_boundary";
const MIXED_KEYED_RESIDUAL_JOIN_RESIDENT_GROUPED_AGGREGATE_INPUT_PIPELINE_MODELED_RESERVATIONS: &str = "source_read,operator_output,build_side,hash_table,probe_side,filtered_probe_state,mixed_size_data,gather_map,join_output,materialization";
const MULTI_JOIN_AGGREGATE_OVER_JOIN_SUBPLAN_TOPK_FAMILY: &str = "multi_join_with_aggregate_over_join_subplan_topk";
const MULTI_JOIN_AGGREGATE_OVER_JOIN_SUBPLAN_TOPK_BLOCKER: &str =
    "bounded_multi_join_aggregate_over_join_subplan_topk_executor_missing";
const MULTI_JOIN_AGGREGATE_OVER_JOIN_SUBPLAN_TOPK_SHAPE_CONTRACT: &str = "final_topk_over_projection_over_root_hash_join_with_multi_join_input_and_projection_wrapped_aggregate_over_join_subplan_input";
const HASH_JOIN_RESIDENT_GROUPED_AGGREGATE_INPUT_POST_JOIN_AGGREGATE_TOPK_FAMILY: &str =
    "hash_join_with_resident_grouped_aggregate_input_post_join_aggregate_topk";
const HASH_JOIN_RESIDENT_GROUPED_AGGREGATE_INPUT_POST_JOIN_AGGREGATE_TOPK_BLOCKER: &str =
    "bounded_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_executor_missing";
const HASH_JOIN_RESIDENT_GROUPED_AGGREGATE_INPUT_POST_JOIN_AGGREGATE_TOPK_SHAPE_CONTRACT: &str = "final_topk_over_projection_over_grouped_aggregate_over_hash_join_with_grouped_aggregate_input_and_multi_join_input";
const MULTI_JOIN_RESIDENT_GROUPED_AGGREGATE_SUBQUERY_FILTER_FAMILY: &str =
    "multi_join_with_resident_grouped_aggregate_subquery_filter";
const MULTI_JOIN_RESIDENT_GROUPED_AGGREGATE_SUBQUERY_FILTER_BLOCKER: &str =
    "bounded_multi_join_resident_grouped_aggregate_subquery_filter_executor_missing";
const MULTI_JOIN_RESIDENT_GROUPED_AGGREGATE_SUBQUERY_FILTER_SHAPE_CONTRACT: &str =
    "final_sort_over_multi_join_tree_with_projection_wrapped_grouped_aggregate_subquery_input_and_residual_filter";

pub(crate) fn bounded_join_unsupported_shape_facts(
    plan: &QueryPlan,
    boundary_policy: ExecutionBoundaryPolicy,
) -> BTreeMap<&'static str, String> {
    let mut facts = BoundedJoinReservationContract.unsupported_shape_facts(boundary_policy);
    let analysis = plan.capability_analysis();
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
    } else if matches_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk(plan) {
        insert_structural_topology_blocker(
            &mut facts,
            HASH_JOIN_RESIDENT_GROUPED_AGGREGATE_INPUT_POST_JOIN_AGGREGATE_TOPK_FAMILY,
            HASH_JOIN_RESIDENT_GROUPED_AGGREGATE_INPUT_POST_JOIN_AGGREGATE_TOPK_BLOCKER,
            HASH_JOIN_RESIDENT_GROUPED_AGGREGATE_INPUT_POST_JOIN_AGGREGATE_TOPK_SHAPE_CONTRACT,
        );
    } else if matches_multi_join_aggregate_over_join_subplan_topk(plan) {
        insert_structural_topology_blocker(
            &mut facts,
            MULTI_JOIN_AGGREGATE_OVER_JOIN_SUBPLAN_TOPK_FAMILY,
            MULTI_JOIN_AGGREGATE_OVER_JOIN_SUBPLAN_TOPK_BLOCKER,
            MULTI_JOIN_AGGREGATE_OVER_JOIN_SUBPLAN_TOPK_SHAPE_CONTRACT,
        );
    } else if matches_multi_join_resident_grouped_aggregate_subquery_filter(plan) {
        insert_structural_topology_blocker(
            &mut facts,
            MULTI_JOIN_RESIDENT_GROUPED_AGGREGATE_SUBQUERY_FILTER_FAMILY,
            MULTI_JOIN_RESIDENT_GROUPED_AGGREGATE_SUBQUERY_FILTER_BLOCKER,
            MULTI_JOIN_RESIDENT_GROUPED_AGGREGATE_SUBQUERY_FILTER_SHAPE_CONTRACT,
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
    insert_capability_reservation_facts(&mut facts, &analysis);
    insert_capability_rejection_facts(&mut facts, &analysis);
    facts
}

pub(crate) fn bounded_join_modeled_facts(
    plan: &QueryPlan,
    boundary_policy: ExecutionBoundaryPolicy,
) -> BTreeMap<&'static str, String> {
    let mut facts = BoundedJoinReservationContract.facts(boundary_policy);
    let analysis = plan.capability_analysis();
    if let Some(projection) = crate::capability::CompatibilityReportProjection::from_analysis(&analysis) {
        projection.insert_modeled_facts(&mut facts);
    }
    facts
}

fn matches_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline(plan: &QueryPlan) -> bool {
    crate::capability::admits_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline(plan)
}

fn insert_structural_topology_blocker(
    facts: &mut BTreeMap<&'static str, String>,
    family: &'static str,
    blocker: &'static str,
    shape_contract: &'static str,
) {
    facts.insert("bounded_join_blocked_join_family", family.to_string());
    facts.insert("bounded_join_specific_blocker", blocker.to_string());
    facts.insert("bounded_join_shape_blocker", blocker.to_string());
    facts.insert("bounded_join_required_shape_contract", shape_contract.to_string());
}

fn insert_capability_rejection_facts(
    facts: &mut BTreeMap<&'static str, String>,
    analysis: &crate::capability::CapabilityAnalysis,
) {
    let Some(rejection) = analysis.compatibility_first_rejection_projection() else {
        return;
    };
    facts.insert("bounded_join_capability_rejection_scope", rejection.scope.to_string());
    facts.insert("bounded_join_capability_rejection_reason", rejection.reason.to_string());
    facts.insert("bounded_join_capability_rejection_facts", capability_rejection_facts_csv(&rejection.facts));
}

fn insert_capability_reservation_facts(
    facts: &mut BTreeMap<&'static str, String>,
    analysis: &crate::capability::CapabilityAnalysis,
) {
    let requirements = analysis.compatibility_reservation_requirements();
    if requirements.is_empty() {
        return;
    }
    facts.insert("bounded_join_reservation_kinds", analysis.compatibility_reservation_kinds_csv());
    facts.insert("bounded_join_reservation_statuses", analysis.compatibility_reservation_statuses_csv());
    facts.insert("bounded_join_reservation_requirements", analysis.compatibility_reservation_requirements_csv());
    facts.insert("modeled_reservations", analysis.compatibility_modeled_reservations_csv());
    facts.insert("blocked_reservations", capability_blocked_reservations_csv(&requirements));
    facts.insert("missing_reservations", capability_blocked_reservations_csv(&requirements));
    facts.insert("bounded_join_lower_level_blockers", capability_lower_level_blockers_csv(&requirements));
}

fn capability_blocked_reservations_csv(requirements: &[crate::capability::ReservationRequirement]) -> String {
    csv_join(requirements.iter().filter_map(|requirement| {
        (requirement.status != crate::capability::ContractStatus::Modeled).then_some(requirement.kind.as_str())
    }))
}

fn capability_lower_level_blockers_csv(requirements: &[crate::capability::ReservationRequirement]) -> String {
    csv_join(requirements.iter().filter_map(|requirement| {
        (requirement.status == crate::capability::ContractStatus::BlockedByLowerLevel)
            .then_some(requirement.required_contract)
    }))
}

fn capability_rejection_facts_csv(facts: &BTreeMap<&'static str, String>) -> String {
    let mut output = String::new();
    for (key, value) in facts {
        if !output.is_empty() {
            output.push(',');
        }
        output.push_str(key);
        output.push('=');
        output.push_str(value);
    }
    output
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

fn matches_multi_join_aggregate_over_join_subplan_topk(plan: &QueryPlan) -> bool {
    crate::capability::admits_multi_join_aggregate_over_join_subplan_topk_pipeline(plan)
}

fn matches_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk(plan: &QueryPlan) -> bool {
    crate::capability::admits_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_pipeline(plan)
}

fn matches_multi_join_resident_grouped_aggregate_subquery_filter(plan: &QueryPlan) -> bool {
    crate::capability::admits_multi_join_resident_grouped_aggregate_subquery_filter_pipeline(plan)
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

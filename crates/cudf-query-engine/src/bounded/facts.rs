use std::collections::BTreeMap;

use crate::{
    capability::{keyed_residual_mixed_join_capable, multi_join_keyed_residual_mixed_join_capable},
    plan::{JoinNode, JoinSide, JoinType, NodeId, PlanNode, QueryPlan},
    runtime::{BoundedJoinReservationContract, ExecutionBoundaryPolicy},
};
const MIXED_KEYED_RESIDUAL_JOIN_RESIDENT_GROUPED_AGGREGATE_INPUT_PIPELINE_BLOCKER: &str =
    "bounded_mixed_keyed_residual_join_resident_grouped_aggregate_input_lower_level_contracts";
const MIXED_KEYED_RESIDUAL_JOIN_RESIDENT_GROUPED_AGGREGATE_INPUT_PIPELINE_LOWER_LEVEL_CONTRACT: &str =
    "source_contract,aggregate_contract";
const MULTI_JOIN_AGGREGATE_OVER_JOIN_SUBPLAN_TOPK_BLOCKER: &str =
    "bounded_multi_join_aggregate_over_join_subplan_topk_executor_missing";
const HASH_JOIN_RESIDENT_GROUPED_AGGREGATE_INPUT_POST_JOIN_AGGREGATE_TOPK_BLOCKER: &str =
    "bounded_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_executor_missing";
const MULTI_JOIN_RESIDENT_GROUPED_AGGREGATE_SUBQUERY_FILTER_BLOCKER: &str =
    "bounded_multi_join_resident_grouped_aggregate_subquery_filter_executor_missing";

pub(crate) fn bounded_join_unsupported_shape_facts(
    plan: &QueryPlan,
    boundary_policy: ExecutionBoundaryPolicy,
) -> BTreeMap<&'static str, String> {
    let mut facts = BoundedJoinReservationContract.unsupported_shape_facts(boundary_policy);
    let analysis = plan.capability_analysis();
    if matches_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline(plan) {
        facts.insert(
            "bounded_join_specific_blocker",
            MIXED_KEYED_RESIDUAL_JOIN_RESIDENT_GROUPED_AGGREGATE_INPUT_PIPELINE_BLOCKER.to_string(),
        );
        facts.insert(
            "bounded_join_shape_blocker",
            MIXED_KEYED_RESIDUAL_JOIN_RESIDENT_GROUPED_AGGREGATE_INPUT_PIPELINE_BLOCKER.to_string(),
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
            HASH_JOIN_RESIDENT_GROUPED_AGGREGATE_INPUT_POST_JOIN_AGGREGATE_TOPK_BLOCKER,
        );
    } else if matches_multi_join_aggregate_over_join_subplan_topk(plan) {
        insert_structural_topology_blocker(&mut facts, MULTI_JOIN_AGGREGATE_OVER_JOIN_SUBPLAN_TOPK_BLOCKER);
    } else if matches_multi_join_resident_grouped_aggregate_subquery_filter(plan) {
        insert_structural_topology_blocker(&mut facts, MULTI_JOIN_RESIDENT_GROUPED_AGGREGATE_SUBQUERY_FILTER_BLOCKER);
    } else if let Some(join) = first_unsupported_keyed_residual_semi_anti_join(plan) {
        facts.insert("bounded_join_specific_blocker", mixed_keyed_residual_left_only_blocker(plan, join).to_string());
        facts.insert("bounded_join_blocked_join_type", join.spec.join_type.to_string());
        facts.insert(
            "bounded_join_shape_blocker",
            "bounded_mixed_keyed_residual_left_only_join_shape_or_orientation".to_string(),
        );
    } else if let Some(join) = first_keyed_residual_full_join(plan) {
        facts.insert("bounded_join_specific_blocker", mixed_keyed_residual_full_blocker(plan, join).to_string());
        facts.insert("bounded_join_blocked_join_type", join.spec.join_type.to_string());
        facts.insert("bounded_join_shape_blocker", "bounded_mixed_keyed_residual_full_join_shape".to_string());
    } else if let Some(join) = first_keyed_residual_left_join(plan) {
        facts.insert("bounded_join_specific_blocker", mixed_keyed_residual_left_blocker(plan, join).to_string());
        facts.insert("bounded_join_blocked_join_type", join.spec.join_type.to_string());
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
    insert_capability_reservation_facts(&mut facts, &analysis);
    facts
}

fn matches_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline(plan: &QueryPlan) -> bool {
    crate::capability::admits_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline(plan)
}

fn insert_structural_topology_blocker(facts: &mut BTreeMap<&'static str, String>, blocker: &'static str) {
    facts.insert("bounded_join_specific_blocker", blocker.to_string());
    facts.insert("bounded_join_shape_blocker", blocker.to_string());
}

fn insert_capability_rejection_facts(
    facts: &mut BTreeMap<&'static str, String>,
    analysis: &crate::capability::CapabilityAnalysis,
) {
    let Some(rejection) = analysis.first_rejection_projection() else {
        return;
    };
    facts.insert("bounded_join_capability_rejection_scope", rejection.contract_scope.as_str().to_string());
    facts.insert("bounded_join_capability_rejection_boundary", rejection.memory_boundary.to_string());
    facts.insert("bounded_join_capability_rejection_reason", rejection.reason.to_string());
    facts.insert("bounded_join_capability_rejection_facts", capability_rejection_facts_csv(&rejection.facts));
}

fn insert_capability_reservation_facts(
    facts: &mut BTreeMap<&'static str, String>,
    analysis: &crate::capability::CapabilityAnalysis,
) {
    let requirements = &analysis.requirements;
    if requirements.is_empty() {
        return;
    }
    facts.insert("bounded_join_reservation_kinds", capability_reservation_kinds_csv(requirements));
    facts.insert("bounded_join_reservation_statuses", capability_reservation_statuses_csv(requirements));
    facts.insert("bounded_join_reservation_requirements", capability_reservation_requirements_csv(requirements));
    facts.insert("modeled_reservations", capability_modeled_reservations_csv(requirements));
    facts.insert("blocked_reservations", capability_blocked_reservations_csv(requirements));
    facts.insert("missing_reservations", capability_blocked_reservations_csv(requirements));
    facts.insert("bounded_join_lower_level_blockers", capability_lower_level_blockers_csv(requirements));
}

fn capability_reservation_kinds_csv(requirements: &[crate::capability::ReservationRequirement]) -> String {
    csv_join(requirements.iter().map(|requirement| requirement.kind.as_str()))
}

fn capability_reservation_statuses_csv(requirements: &[crate::capability::ReservationRequirement]) -> String {
    csv_join(
        requirements.iter().map(|requirement| format!("{}:{}", requirement.kind.as_str(), requirement.status.as_str())),
    )
}

fn capability_reservation_requirements_csv(requirements: &[crate::capability::ReservationRequirement]) -> String {
    csv_join(requirements.iter().map(|requirement| {
        format!("{}:{}:{}", requirement.kind.as_str(), requirement.required_contract, requirement.quality.as_str())
    }))
}

fn capability_modeled_reservations_csv(requirements: &[crate::capability::ReservationRequirement]) -> String {
    csv_join(requirements.iter().filter_map(|requirement| {
        (requirement.status == crate::capability::ContractStatus::Modeled).then_some(requirement.kind.as_str())
    }))
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
            (!multi_join_keyed_residual_mixed_join_capable(join)).then_some(join)
        } else {
            (!keyed_residual_mixed_join_capable(join) || root_join_leaf_blocker(plan, join).is_some()).then_some(join)
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

fn reachable_join_count(plan: &QueryPlan) -> usize {
    let mut reachable = vec![false; plan.nodes().len()];
    let mut stack = vec![plan.sink().input()];
    let mut count = 0;
    while let Some(id) = stack.pop() {
        let Some(node) = plan.node(id) else {
            continue;
        };
        if reachable[id.index()] {
            continue;
        }
        reachable[id.index()] = true;
        if matches!(node.kind, PlanNode::Join(_)) {
            count += 1;
        }
        stack.extend(node.input_ids().iter().copied());
    }
    count
}

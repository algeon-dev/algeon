use std::collections::BTreeMap;

use crate::{
    capability::{
        CapabilityAnalysis, CapabilityRejectionOwner, CapabilityRejectionProjection, ContractStatus,
        ReservationRequirement,
    },
    plan::QueryPlan,
    runtime::{BoundedJoinReservationContract, ExecutionBoundaryPolicy},
};

pub(crate) fn bounded_join_unsupported_shape_facts(
    plan: &QueryPlan,
    boundary_policy: ExecutionBoundaryPolicy,
) -> BTreeMap<&'static str, String> {
    let mut facts = BoundedJoinReservationContract.unsupported_shape_facts(boundary_policy);
    let analysis = plan.capability_analysis();
    insert_capability_reservation_facts(&mut facts, &analysis);
    insert_local_rejection_facts(&mut facts, &analysis);
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

fn insert_local_rejection_facts(facts: &mut BTreeMap<&'static str, String>, analysis: &CapabilityAnalysis) {
    if let Some(rejection) = analysis.first_rejection_projection() {
        facts.insert("bounded_join_rejection_scope", rejection.contract_scope.as_str().to_string());
        facts.insert("bounded_join_rejection_owner", rejection_owner_fact(&rejection.owner));
        facts.insert("bounded_join_rejection_boundary", rejection.memory_boundary.to_string());
        facts.insert("bounded_join_rejection_category", rejection_category(&rejection).to_string());
        facts.insert("bounded_join_rejection_reason", rejection.reason.to_string());
        facts.insert("bounded_join_rejection_facts", capability_rejection_facts_csv(&rejection.facts));
        return;
    }

    facts.insert("bounded_join_rejection_scope", "subgraph".to_string());
    facts.insert("bounded_join_rejection_owner", "strict_bounded_admission".to_string());
    facts.insert("bounded_join_rejection_boundary", "strict_bounded_admission".to_string());
    facts.insert("bounded_join_rejection_category", "missing_subgraph_invariant".to_string());
    facts.insert("bounded_join_rejection_reason", "strict_bounded_capability_admission_not_satisfied".to_string());
    facts.insert("bounded_join_rejection_facts", "strict_bounded_admission=false".to_string());
}

fn rejection_owner_fact(owner: &CapabilityRejectionOwner) -> String {
    match owner {
        CapabilityRejectionOwner::Node(node_id) => format!("node:{node_id}"),
        CapabilityRejectionOwner::Edge { producer, consumer } => format!("edge:{producer}->{consumer:?}"),
        CapabilityRejectionOwner::Invariant(invariant_id) => format!("invariant:i{}", invariant_id.index()),
        CapabilityRejectionOwner::Admission(admission) => format!("admission:{admission}"),
    }
}

fn rejection_category(rejection: &CapabilityRejectionProjection) -> &'static str {
    if rejection.reason == "blocked_lower_level_contract" || rejection.memory_boundary == "lower_level_contract" {
        return "missing_lower_level_contract";
    }
    if rejection.facts.contains_key("expression_family") {
        return "unsupported_expression_family";
    }
    if rejection.facts.contains_key("data_type")
        || rejection.facts.contains_key("dtype")
        || rejection.reason.contains("data_type")
        || rejection.reason.contains("dtype")
    {
        return "unsupported_arrow_dtype";
    }
    match rejection.owner {
        CapabilityRejectionOwner::Node(_) => "unsupported_node_operator",
        CapabilityRejectionOwner::Edge { .. } => "missing_edge_contract",
        CapabilityRejectionOwner::Invariant(_) | CapabilityRejectionOwner::Admission(_) => "missing_subgraph_invariant",
    }
}

fn insert_capability_reservation_facts(facts: &mut BTreeMap<&'static str, String>, analysis: &CapabilityAnalysis) {
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

fn capability_reservation_kinds_csv(requirements: &[ReservationRequirement]) -> String {
    csv_join(requirements.iter().map(|requirement| requirement.kind.as_str()))
}

fn capability_reservation_statuses_csv(requirements: &[ReservationRequirement]) -> String {
    csv_join(
        requirements.iter().map(|requirement| format!("{}:{}", requirement.kind.as_str(), requirement.status.as_str())),
    )
}

fn capability_reservation_requirements_csv(requirements: &[ReservationRequirement]) -> String {
    csv_join(requirements.iter().map(|requirement| {
        format!("{}:{}:{}", requirement.kind.as_str(), requirement.required_contract, requirement.quality.as_str())
    }))
}

fn capability_modeled_reservations_csv(requirements: &[ReservationRequirement]) -> String {
    csv_join(
        requirements.iter().filter_map(|requirement| {
            (requirement.status == ContractStatus::Modeled).then_some(requirement.kind.as_str())
        }),
    )
}

fn capability_blocked_reservations_csv(requirements: &[ReservationRequirement]) -> String {
    csv_join(
        requirements.iter().filter_map(|requirement| {
            (requirement.status != ContractStatus::Modeled).then_some(requirement.kind.as_str())
        }),
    )
}

fn capability_lower_level_blockers_csv(requirements: &[ReservationRequirement]) -> String {
    csv_join(requirements.iter().filter_map(|requirement| {
        (requirement.status == ContractStatus::BlockedByLowerLevel).then_some(requirement.required_contract)
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

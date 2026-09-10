use std::collections::BTreeMap;

use crate::{
    capability::{CapabilityAnalysis, CapabilityRejectionOwner, CapabilityRejectionProjection},
    error::QueryError,
    plan::QueryPlan,
};

/// Returns diagnostic facts for join plans rejected by bounded admission.
///
/// These facts are explanatory planning/execution diagnostics. They do not
/// toggle executor behavior and should not be treated as optimizer hints.
pub fn bounded_join_unsupported_shape_facts(plan: &QueryPlan) -> Option<BTreeMap<&'static str, String>> {
    if !plan.contains_join() {
        return None;
    }

    match super::bounded_pipeline_rejection_projection(plan) {
        Ok(Some(projection)) => Some(bounded_join_unsupported_shape_facts_with_projection(plan, Some(&projection))),
        Ok(None) => None,
        Err(error) => {
            let mut facts = bounded_join_unsupported_shape_facts_with_projection(plan, None);
            insert_bounded_join_rejection_projection_error_facts(&mut facts, &error);
            Some(facts)
        },
    }
}

/// Returns modeled bounded-join facts for plans accepted by bounded admission.
pub fn bounded_join_modeled_facts(plan: &QueryPlan) -> Option<BTreeMap<&'static str, String>> {
    if !plan.contains_join() {
        return None;
    }
    matches!(super::bounded_pipeline_rejection_projection(plan), Ok(None))
        .then(|| bounded_join_modeled_facts_unchecked(plan))
}

#[doc(hidden)]
pub fn bounded_join_unsupported_shape_facts_with_projection(
    plan: &QueryPlan,
    rejection_projection: Option<&CapabilityRejectionProjection>,
) -> BTreeMap<&'static str, String> {
    let mut facts = BTreeMap::new();
    let analysis = plan.capability_analysis();
    insert_join_strategy_facts(&mut facts, plan);
    insert_local_rejection_facts(&mut facts, &analysis, rejection_projection);
    facts
}

fn bounded_join_modeled_facts_unchecked(plan: &QueryPlan) -> BTreeMap<&'static str, String> {
    let mut facts = BTreeMap::new();
    insert_join_strategy_facts(&mut facts, plan);
    facts
}

#[doc(hidden)]
pub fn insert_bounded_join_rejection_projection_error_facts(
    facts: &mut BTreeMap<&'static str, String>,
    error: &QueryError,
) {
    facts.insert("bounded_join_rejection_projection_error_code", error.code().to_string());
    facts.insert("bounded_join_rejection_projection_error_operation", error.operation().to_string());
    facts.insert("bounded_join_rejection_projection_error_message", error.message().to_string());
}

fn insert_join_strategy_facts(facts: &mut BTreeMap<&'static str, String>, plan: &QueryPlan) {
    let mut join_strategy = Vec::new();
    let mut reusable_rejections = Vec::new();
    let mut partitioned_rejections = Vec::new();
    let mut estimated_build_rows = Vec::new();
    let mut observed_build_rows = Vec::new();
    let mut build_side_bytes = Vec::new();
    let mut probe_side_bytes = Vec::new();
    let mut probe_reuse_count = Vec::new();
    let mut reusable_prebuild_supported = Vec::new();
    let mut partition_counts = Vec::new();
    let mut partition_target_bytes = Vec::new();

    for node in plan.nodes() {
        let crate::plan::PlanNode::Join(join) = &node.kind else {
            continue;
        };
        let Some(decision) = crate::capability::keyed_hash_join_strategy_decision_for_plan(plan, node.id, join) else {
            continue;
        };
        let node_label = node.id.to_string();
        let decision_facts = decision.facts();
        if let Some(value) = decision_facts.get("join_strategy") {
            join_strategy.push(format!("{node_label}:{value}"));
        }
        if let Some(value) = decision_facts.get("reusable_hash_join_rejected_reason") {
            reusable_rejections.push(format!("{node_label}:{value}"));
        }
        if let Some(value) = decision_facts.get("partitioned_hash_join_rejected_reason") {
            partitioned_rejections.push(format!("{node_label}:{value}"));
        }
        if let Some(value) = decision_facts.get("estimated_build_rows") {
            estimated_build_rows.push(format!("{node_label}:{value}"));
        }
        if let Some(value) = decision_facts.get("observed_build_rows") {
            observed_build_rows.push(format!("{node_label}:{value}"));
        }
        if let Some(value) = decision_facts.get("build_side_bytes") {
            build_side_bytes.push(format!("{node_label}:{value}"));
        }
        if let Some(value) = decision_facts.get("probe_side_bytes") {
            probe_side_bytes.push(format!("{node_label}:{value}"));
        }
        if let Some(value) = decision_facts.get("probe_reuse_count") {
            probe_reuse_count.push(format!("{node_label}:{value}"));
        }
        if let Some(value) = decision_facts.get("reusable_prebuild_supported") {
            reusable_prebuild_supported.push(format!("{node_label}:{value}"));
        }
        if let Some(value) = decision_facts.get("partition_count") {
            partition_counts.push(format!("{node_label}:{value}"));
        }
        if let Some(value) = decision_facts.get("partition_target_bytes") {
            partition_target_bytes.push(format!("{node_label}:{value}"));
        }
    }

    insert_csv_fact(facts, "join_strategy", join_strategy);
    insert_csv_fact(facts, "reusable_hash_join_rejected_reason", reusable_rejections);
    insert_csv_fact(facts, "partitioned_hash_join_rejected_reason", partitioned_rejections);
    insert_csv_fact(facts, "estimated_build_rows", estimated_build_rows);
    insert_csv_fact(facts, "observed_build_rows", observed_build_rows);
    insert_csv_fact(facts, "build_side_bytes", build_side_bytes);
    insert_csv_fact(facts, "probe_side_bytes", probe_side_bytes);
    insert_csv_fact(facts, "probe_reuse_count", probe_reuse_count);
    insert_csv_fact(facts, "reusable_prebuild_supported", reusable_prebuild_supported);
    insert_csv_fact(facts, "partition_count", partition_counts);
    insert_csv_fact(facts, "partition_target_bytes", partition_target_bytes);
}

fn insert_csv_fact(facts: &mut BTreeMap<&'static str, String>, key: &'static str, values: Vec<String>) {
    if !values.is_empty() {
        facts.insert(key, csv_join(values));
    }
}

fn insert_local_rejection_facts(
    facts: &mut BTreeMap<&'static str, String>,
    analysis: &CapabilityAnalysis,
    rejection_projection: Option<&CapabilityRejectionProjection>,
) {
    let analysis_projection = analysis.first_rejection_projection();
    if let Some(rejection) = rejection_projection.or(analysis_projection.as_ref()) {
        facts.insert("bounded_join_rejection_scope", rejection.contract_scope.as_str().to_string());
        facts.insert("bounded_join_rejection_owner", rejection_owner_fact(&rejection.owner));
        facts.insert("bounded_join_rejection_boundary", rejection.boundary.to_string());
        facts.insert("bounded_join_rejection_category", rejection.category.as_str().to_string());
        facts.insert("bounded_join_rejection_reason", rejection.reason.to_string());
        facts.insert("bounded_join_rejection_facts", capability_rejection_facts_csv(&rejection.facts));
        return;
    }

    facts.insert("bounded_join_rejection_scope", "subgraph".to_string());
    facts.insert("bounded_join_rejection_owner", "capability_envelope_admission".to_string());
    facts.insert("bounded_join_rejection_boundary", "capability_envelope_admission".to_string());
    facts.insert("bounded_join_rejection_category", "missing_subgraph_invariant".to_string());
    facts.insert("bounded_join_rejection_reason", "capability_envelope_not_satisfied".to_string());
    facts.insert("bounded_join_rejection_facts", "capability_envelope_admission=false".to_string());
}

fn rejection_owner_fact(owner: &CapabilityRejectionOwner) -> String {
    match owner {
        CapabilityRejectionOwner::Node(node_id) => format!("node:{node_id}"),
        CapabilityRejectionOwner::Edge { producer, consumer } => format!("edge:{producer}->{consumer:?}"),
        CapabilityRejectionOwner::Invariant(invariant_id) => format!("invariant:i{}", invariant_id.index()),
        CapabilityRejectionOwner::Admission(admission) => format!("admission:{admission}"),
    }
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

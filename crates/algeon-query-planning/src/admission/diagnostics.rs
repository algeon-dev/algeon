//! Report facts for bounded join admission.
//!
//! Fact keys and the `node:value` comma-separated list encoding are TSV report
//! contracts. A change here changes the report family in place together with
//! its consumers, goldens and fixtures.

use std::collections::BTreeMap;

use super::AdmissionDecision;
use crate::{
    backend::AnalysisContext,
    capability::{Rejection, choose_join_strategy, classify_join, keyed_hash_join_strategy_input_facts_for_plan},
    plan::{PlanNode, QueryPlan},
};

/// Returns diagnostic facts for a join plan `decision` rejected.
///
/// These facts are explanatory planning/execution diagnostics. They do not
/// toggle executor behavior and should not be treated as optimizer hints.
pub fn bounded_join_unsupported_shape_facts(
    plan: &QueryPlan,
    decision: &AdmissionDecision,
    ctx: AnalysisContext<'_>,
) -> Option<BTreeMap<&'static str, String>> {
    if !plan.contains_join() {
        return None;
    }
    decision.rejection().map(|rejection| bounded_join_unsupported_shape_facts_with_rejection(plan, rejection, ctx))
}

/// Returns modeled bounded-join facts for a join plan `decision` admitted.
pub fn bounded_join_modeled_facts(
    plan: &QueryPlan,
    decision: &AdmissionDecision,
    ctx: AnalysisContext<'_>,
) -> Option<BTreeMap<&'static str, String>> {
    (plan.contains_join() && decision.admission().is_some()).then(|| bounded_join_modeled_facts_unchecked(plan, ctx))
}

fn bounded_join_unsupported_shape_facts_with_rejection(
    plan: &QueryPlan,
    rejection: &Rejection,
    ctx: AnalysisContext<'_>,
) -> BTreeMap<&'static str, String> {
    let mut facts = BTreeMap::new();
    insert_join_strategy_facts(&mut facts, plan, ctx);
    insert_local_rejection_facts(&mut facts, rejection);
    facts
}

fn bounded_join_modeled_facts_unchecked(plan: &QueryPlan, ctx: AnalysisContext<'_>) -> BTreeMap<&'static str, String> {
    let mut facts = BTreeMap::new();
    insert_join_strategy_facts(&mut facts, plan, ctx);
    facts
}

fn insert_join_strategy_facts(facts: &mut BTreeMap<&'static str, String>, plan: &QueryPlan, ctx: AnalysisContext<'_>) {
    let mut join_strategy = Vec::new();
    let mut reusable_rejections = Vec::new();
    let mut partitioned_rejections = Vec::new();
    let mut estimated_build_rows = Vec::new();
    let mut observed_build_rows = Vec::new();
    let mut build_side_bytes = Vec::new();
    let mut probe_side_bytes = Vec::new();
    let mut probe_reuse_count = Vec::new();
    let mut reusable_prebuild_supported = Vec::new();

    for node in plan.nodes() {
        let PlanNode::Join(join) = &node.kind else {
            continue;
        };
        let Ok(form) = classify_join(node.id, join) else {
            continue;
        };
        let input = keyed_hash_join_strategy_input_facts_for_plan(plan, node.id, join);
        let choice = choose_join_strategy(form, join, input, &ctx.policy().keyed_hash_join);
        let Some(evidence) = choice.evidence() else {
            continue;
        };
        let node_label = node.id.to_string();
        let label = |value: String| format!("{node_label}:{value}");
        join_strategy.push(label(choice.strategy().as_str().to_string()));
        if let Some(reason) = evidence.reusable_rejected_reason() {
            reusable_rejections.push(label(reason.to_string()));
        }
        if let Some(reason) = evidence.partitioned_rejected_reason() {
            partitioned_rejections.push(label(reason.to_string()));
        }
        let input = evidence.input();
        if let Some(rows) = input.estimated_build_rows {
            estimated_build_rows.push(label(rows.to_string()));
        }
        if let Some(rows) = input.observed_build_rows {
            observed_build_rows.push(label(rows.to_string()));
        }
        if let Some(bytes) = input.build_side_bytes {
            build_side_bytes.push(label(bytes.to_string()));
        }
        if let Some(bytes) = input.probe_side_bytes {
            probe_side_bytes.push(label(bytes.to_string()));
        }
        probe_reuse_count.push(label(input.probe_reuse_count.to_string()));
        reusable_prebuild_supported.push(label(input.reusable_prebuild_supported.to_string()));
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
}

fn insert_csv_fact(facts: &mut BTreeMap<&'static str, String>, key: &'static str, values: Vec<String>) {
    if !values.is_empty() {
        facts.insert(key, csv_join(values));
    }
}

fn insert_local_rejection_facts(facts: &mut BTreeMap<&'static str, String>, rejection: &Rejection) {
    facts.insert("bounded_join_rejection_scope", rejection.contract_scope().as_str().to_string());
    facts.insert("bounded_join_rejection_owner", rejection.owner.to_string());
    facts.insert("bounded_join_rejection_boundary", rejection.boundary().to_string());
    facts.insert("bounded_join_rejection_category", rejection.category().as_str().to_string());
    facts.insert("bounded_join_rejection_reason", rejection.reason.to_string());
    facts.insert("bounded_join_rejection_facts", capability_rejection_facts_csv(&rejection.facts));
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

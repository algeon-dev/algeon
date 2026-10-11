//! Proof that filtering a shared source in place cannot change another
//! consumer's result.
//!
//! A local Parquet source with several parents may take a source-level dynamic
//! filter only when every path upward from it, following the key column through
//! filters, renaming projections, sorts, grouping-only aggregates and join output
//! projections, reaches an inner `NullsDoNotMatch` join that equates the column
//! with the same producer key set. Rows the filter removes then match nothing on
//! any path and contribute to no output. A limit or window ends a path
//! negatively because it selects rows by position, so removing rows below it
//! changes its output even when a join above would constrain the key.

use std::collections::BTreeSet;

use super::{
    AggregateGroupKeyLineage, AggregateNode, JoinNode, JoinNullEquality, JoinSide, JoinType, NodeId, PlanNode,
    QueryPlan, aggregate_group_key_output_name, direct_projection_column_name, subtree_contains_node, unique_lineage,
};

/// Whether every consumer of `source_column` at `source_node_id` is constrained
/// by the key set that `producer_keyset_node_id` produces from
/// `producer_source_column` of `producer_source_node_id`. A path that leaves the
/// column behind, or revisits a node, fails the proof.
pub(super) fn shared_source_filter_is_keyset_constrained(
    plan: &QueryPlan,
    parent_counts: &[usize],
    source_node_id: NodeId,
    source_column: &str,
    producer_keyset_node_id: NodeId,
    producer_source_node_id: NodeId,
    producer_source_column: &str,
) -> bool {
    let mut visited = BTreeSet::new();
    column_consumers_are_keyset_constrained(
        plan,
        parent_counts,
        source_node_id,
        source_column,
        producer_keyset_node_id,
        producer_source_node_id,
        producer_source_column,
        &mut visited,
    )
}

#[expect(clippy::too_many_arguments, reason = "planning inputs are distinct proof and constraint values")]
fn column_consumers_are_keyset_constrained(
    plan: &QueryPlan,
    parent_counts: &[usize],
    node_id: NodeId,
    column: &str,
    producer_keyset_node_id: NodeId,
    producer_source_node_id: NodeId,
    producer_source_column: &str,
    visited: &mut std::collections::BTreeSet<(NodeId, String)>,
) -> bool {
    if !visited.insert((node_id, column.to_string())) {
        return false;
    }
    let parents = parent_nodes(plan, node_id);
    if parents.is_empty() {
        return false;
    }
    parents.into_iter().all(|parent_id| {
        parent_constrains_or_carries_column(
            plan,
            parent_counts,
            node_id,
            column,
            parent_id,
            producer_keyset_node_id,
            producer_source_node_id,
            producer_source_column,
            visited,
        )
    })
}

#[expect(clippy::too_many_arguments, reason = "planning inputs are distinct proof and constraint values")]
fn parent_constrains_or_carries_column(
    plan: &QueryPlan,
    parent_counts: &[usize],
    child_id: NodeId,
    column: &str,
    parent_id: NodeId,
    producer_keyset_node_id: NodeId,
    producer_source_node_id: NodeId,
    producer_source_column: &str,
    visited: &mut std::collections::BTreeSet<(NodeId, String)>,
) -> bool {
    let Some(parent) = plan.node(parent_id) else {
        return false;
    };
    match &parent.kind {
        PlanNode::Filter(filter) if filter.input == child_id => column_consumers_are_keyset_constrained(
            plan,
            parent_counts,
            parent_id,
            column,
            producer_keyset_node_id,
            producer_source_node_id,
            producer_source_column,
            visited,
        ),
        PlanNode::Projection(projection) if projection.input == child_id => {
            projection.outputs.iter().filter(|output| direct_projection_column_name(&output.expr) == Some(column)).any(
                |output| {
                    column_consumers_are_keyset_constrained(
                        plan,
                        parent_counts,
                        parent_id,
                        output.field.name(),
                        producer_keyset_node_id,
                        producer_source_node_id,
                        producer_source_column,
                        visited,
                    )
                },
            )
        },
        PlanNode::Sort(sort) if sort.input == child_id => column_consumers_are_keyset_constrained(
            plan,
            parent_counts,
            parent_id,
            column,
            producer_keyset_node_id,
            producer_source_node_id,
            producer_source_column,
            visited,
        ),
        PlanNode::Join(join) => join_constrains_or_carries_column(
            plan,
            parent_counts,
            child_id,
            column,
            parent_id,
            join,
            producer_keyset_node_id,
            producer_source_node_id,
            producer_source_column,
            visited,
        ),
        PlanNode::Aggregate(aggregate) if aggregate.input == child_id => {
            aggregate_group_key_output_for_column(aggregate, column).is_some_and(|output_column| {
                column_consumers_are_keyset_constrained(
                    plan,
                    parent_counts,
                    parent_id,
                    output_column,
                    producer_keyset_node_id,
                    producer_source_node_id,
                    producer_source_column,
                    visited,
                )
            })
        },
        // `Limit`/`Window` are intentionally rejected (not recursed). Filtering the shared
        // source in place removes rows before these order/position-dependent operators run, so
        // a `Limit(N)` above would select a different row set than the unfiltered plan even when a
        // further-downstream inner join would constrain the key. This mirrors `lineages_for_column`,
        // which also refuses to cross `Limit` on the consumer side.
        PlanNode::Filter(_)
        | PlanNode::Projection(_)
        | PlanNode::Sort(_)
        | PlanNode::Limit(_)
        | PlanNode::Aggregate(_)
        | PlanNode::Source(_)
        | PlanNode::Union(_)
        | PlanNode::Window(_)
        | PlanNode::EdgeNormalize(_)
        | PlanNode::GraphAlgorithm(_)
        | PlanNode::VectorAlgorithm(_)
        | PlanNode::MlAlgorithm(_) => false,
    }
}

#[expect(clippy::too_many_arguments, reason = "planning inputs are distinct proof and constraint values")]
fn join_constrains_or_carries_column(
    plan: &QueryPlan,
    parent_counts: &[usize],
    child_id: NodeId,
    column: &str,
    join_node_id: NodeId,
    join: &JoinNode,
    producer_keyset_node_id: NodeId,
    producer_source_node_id: NodeId,
    producer_source_column: &str,
    visited: &mut std::collections::BTreeSet<(NodeId, String)>,
) -> bool {
    let Some(child_side) = join_input_side(join, child_id) else {
        return false;
    };
    if join_directly_constrains_column(
        plan,
        parent_counts,
        join,
        child_side,
        column,
        producer_keyset_node_id,
        producer_source_node_id,
        producer_source_column,
    ) {
        return true;
    }
    join.spec.output_projection.iter().filter(|output| output.side == child_side && output.column.name() == column).any(
        |output| {
            column_consumers_are_keyset_constrained(
                plan,
                parent_counts,
                join_node_id,
                output.output_name.as_str(),
                producer_keyset_node_id,
                producer_source_node_id,
                producer_source_column,
                visited,
            )
        },
    )
}

/// Whether `join` equates `column` on `child_side` with the producer key: the
/// other input must contain the producer key-set node and the other key's
/// unique lineage must be the producer's source column, so the two sides are
/// filtered by the same values.
#[expect(clippy::too_many_arguments, reason = "planning inputs are distinct proof and constraint values")]
fn join_directly_constrains_column(
    plan: &QueryPlan,
    parent_counts: &[usize],
    join: &JoinNode,
    child_side: JoinSide,
    column: &str,
    producer_keyset_node_id: NodeId,
    producer_source_node_id: NodeId,
    producer_source_column: &str,
) -> bool {
    if join.spec.join_type != JoinType::Inner || join.spec.null_equality != JoinNullEquality::NullsDoNotMatch {
        return false;
    }
    join.spec.keys.iter().any(|key| {
        let (child_key, other_key, other_input) = match child_side {
            JoinSide::Left => (&key.left, &key.right, join.inputs[1]),
            JoinSide::Right => (&key.right, &key.left, join.inputs[0]),
        };
        child_key.name() == column
            && subtree_contains_node(plan, other_input, producer_keyset_node_id)
            && unique_lineage(plan, parent_counts, other_input, other_key.name(), AggregateGroupKeyLineage::Producer)
                .is_some_and(|lineage| {
                    lineage.source_node_id == producer_source_node_id && lineage.source_column == producer_source_column
                })
    })
}

fn join_input_side(join: &JoinNode, input: NodeId) -> Option<JoinSide> {
    if join.inputs[0] == input {
        Some(JoinSide::Left)
    } else if join.inputs[1] == input {
        Some(JoinSide::Right)
    } else {
        None
    }
}

fn aggregate_group_key_output_for_column<'a>(aggregate: &'a AggregateNode, column: &str) -> Option<&'a str> {
    if aggregate.spec.grouping_sets.is_some() {
        return None;
    }
    aggregate.spec.group_exprs.iter().find_map(|expr| {
        (direct_projection_column_name(expr) == Some(column)).then(|| aggregate_group_key_output_name(expr)).flatten()
    })
}

fn parent_nodes(plan: &QueryPlan, node_id: NodeId) -> Vec<NodeId> {
    plan.nodes()
        .iter()
        .filter(|candidate| candidate.input_ids().contains(&node_id))
        .map(|candidate| candidate.id)
        .collect()
}

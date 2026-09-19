//! Provable row-count upper bounds derived purely from plan structure.
//!
//! Unlike descriptive planning estimates, the functions here are
//! mathematically hard bounds: no bytes, no memory model, no calibration.
//! Admission may use them for resident side inputs where a wrong estimate
//! would be unsafe; descriptive estimates remain unsuitable for that role.

use std::collections::HashMap;

use super::{JoinSide, JoinType, NodeId, PlanNode, QueryPlan};

/// Provenance for a row bound returned by this module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RowBoundProvenance {
    /// The bound is an exact row count.
    PlanExact,
    /// The bound is a hard upper bound, not necessarily exact.
    PlanHardUpperBound,
}

impl RowBoundProvenance {
    /// Stable diagnostic string for this provenance.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PlanExact => "plan_exact",
            Self::PlanHardUpperBound => "plan_hard_upper_bound",
        }
    }
}

/// Returns a mathematically hard upper bound on the root row count when the
/// native plan can prove one without executing any source.
pub fn query_plan_hard_row_upper_bound(plan: &QueryPlan) -> Option<u64> {
    query_plan_node_hard_row_upper_bound(plan, plan.root_id())
}

#[doc(hidden)]
pub fn query_plan_node_hard_row_upper_bound(plan: &QueryPlan, node_id: NodeId) -> Option<u64> {
    hard_row_upper_bound_for_plan_node(plan, node_id).map(|(rows, _)| rows)
}

#[doc(hidden)]
pub fn query_plan_node_hard_row_upper_bound_with_resident_rows(
    plan: &QueryPlan,
    node_id: NodeId,
    resident_rows: &HashMap<String, u64>,
) -> Option<u64> {
    hard_row_upper_bound_with_resident_rows(plan, node_id, resident_rows).map(|(rows, _)| rows)
}

/// Derive only mathematically hard cardinality evidence.  This deliberately
/// does not reuse descriptive output-row estimates, whose contract is
/// descriptive and unsuitable for bounded admission.
#[doc(hidden)]
pub fn hard_row_upper_bound_for_plan_node(plan: &QueryPlan, node_id: NodeId) -> Option<(u64, RowBoundProvenance)> {
    hard_row_upper_bound_with_resident_rows(plan, node_id, &HashMap::new())
}

#[doc(hidden)]
pub fn hard_row_upper_bound_with_resident_rows(
    plan: &QueryPlan,
    node_id: NodeId,
    resident_rows: &HashMap<String, u64>,
) -> Option<(u64, RowBoundProvenance)> {
    let node = plan.node(node_id)?;
    let input_bound = |input| hard_row_upper_bound_with_resident_rows(plan, input, resident_rows);
    match &node.kind {
        PlanNode::Source(source) => match source.as_ref() {
            crate::source::SourceNode::LiteralBatch(source) => {
                Some((source.row_count() as u64, RowBoundProvenance::PlanExact))
            },
            crate::source::SourceNode::LocalParquet(source) => {
                let rows = source.exact_selected_row_count()?;
                let rows = source.scan_limit.map_or(rows, |limit| rows.min(limit as u64));
                Some((
                    source.max_rows.map_or(rows, |max_rows| rows.min(max_rows as u64)),
                    RowBoundProvenance::PlanHardUpperBound,
                ))
            },
            crate::source::SourceNode::GpuResident(source) => resident_rows
                .get(&source.input_name)
                .copied()
                .or(source.row_count_upper_bound)
                .map(|rows| (rows, RowBoundProvenance::PlanHardUpperBound)),
        },
        PlanNode::Projection(projection) => input_bound(projection.input),
        PlanNode::Filter(filter) => {
            input_bound(filter.input).map(|(rows, _)| (rows, RowBoundProvenance::PlanHardUpperBound))
        },
        PlanNode::Window(window) => input_bound(window.input),
        PlanNode::Sort(sort) => input_bound(sort.input),
        PlanNode::Limit(limit) => match limit.spec.fetch {
            Some(fetch) => Some((fetch as u64, RowBoundProvenance::PlanHardUpperBound)),
            None => input_bound(limit.input)
                .map(|(rows, _)| (rows.saturating_sub(limit.spec.skip as u64), RowBoundProvenance::PlanHardUpperBound)),
        },
        PlanNode::Union(union) => union
            .inputs
            .iter()
            .try_fold(0_u64, |total, input| input_bound(*input)?.0.checked_add(total))
            .map(|rows| (rows, RowBoundProvenance::PlanHardUpperBound)),
        PlanNode::Aggregate(aggregate) if aggregate.spec.group_exprs.is_empty() => {
            Some((1, RowBoundProvenance::PlanExact))
        },
        PlanNode::Aggregate(aggregate) => {
            input_bound(aggregate.input).map(|(rows, _)| (rows, RowBoundProvenance::PlanHardUpperBound))
        },
        PlanNode::Join(join) => {
            let left_rows = input_bound(join.inputs[0])?.0;
            let rows = match join.spec.join_type {
                JoinType::LeftSemi | JoinType::LeftAnti | JoinType::LeftMark => Some(left_rows),
                JoinType::Inner
                    if !join.spec.keys.is_empty()
                        && join.spec.has_at_most_one_match_per_bounded_row(JoinSide::Left) =>
                {
                    Some(left_rows)
                },
                JoinType::Inner
                    if !join.spec.keys.is_empty()
                        && join.spec.has_at_most_one_match_per_bounded_row(JoinSide::Right) =>
                {
                    Some(input_bound(join.inputs[1])?.0)
                },
                JoinType::Inner | JoinType::Cross => left_rows.checked_mul(input_bound(join.inputs[1])?.0),
                JoinType::Left
                    if !join.spec.keys.is_empty()
                        && join.spec.has_at_most_one_match_per_bounded_row(JoinSide::Left) =>
                {
                    Some(left_rows)
                },
                JoinType::Left => left_rows.checked_mul(input_bound(join.inputs[1])?.0.max(1)),
                JoinType::Full => {
                    let right_rows = input_bound(join.inputs[1])?.0;
                    left_rows.checked_mul(right_rows)?.checked_add(left_rows)?.checked_add(right_rows)
                },
            }?;
            Some((rows, RowBoundProvenance::PlanHardUpperBound))
        },
        // A graph algorithm node's output cardinality depended on the deleted
        // memory model's edge/vertex accounting; it no longer contributes a
        // provable row bound.
        PlanNode::GraphAlgorithm(_) => None,
        PlanNode::EdgeNormalize(_) | PlanNode::VectorAlgorithm(_) => None,
    }
}

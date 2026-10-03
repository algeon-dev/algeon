//! Provable row-count upper bounds derived purely from plan structure.
//!
//! Unlike descriptive planning estimates, the functions here are
//! mathematically hard bounds: no bytes, no memory model, no calibration.
//! Admission may use them for resident side inputs where a wrong estimate
//! would be unsafe; descriptive estimates remain unsuitable for that role.

use super::{JoinSide, JoinType, NodeId, PlanNode, QueryPlan};
use crate::source::SourceNode;

/// Provenance for a row bound returned by this module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum RowBoundProvenance {
    /// The bound is an exact row count.
    PlanExact,
    /// The bound is a hard upper bound, not necessarily exact.
    PlanHardUpperBound,
}

/// Returns a mathematically hard upper bound on the root row count when the
/// native plan can prove one without executing any source.
pub fn query_plan_hard_row_upper_bound(plan: &QueryPlan) -> Option<u64> {
    query_plan_node_hard_row_upper_bound(plan, plan.root_id())
}

pub(crate) fn query_plan_node_hard_row_upper_bound(plan: &QueryPlan, node_id: NodeId) -> Option<u64> {
    hard_row_upper_bound_for_plan_node(plan, node_id).map(|(rows, _)| rows)
}

/// Derive only mathematically hard cardinality evidence.  This deliberately
/// does not reuse descriptive output-row estimates, whose contract is
/// descriptive and unsuitable for bounded admission.
pub(crate) fn hard_row_upper_bound_for_plan_node(
    plan: &QueryPlan,
    node_id: NodeId,
) -> Option<(u64, RowBoundProvenance)> {
    let node = plan.node(node_id)?;
    let input_bound = |input| hard_row_upper_bound_for_plan_node(plan, input);
    match &node.kind {
        PlanNode::Source(source) => match source.as_ref() {
            SourceNode::LiteralBatch(source) => Some((source.row_count() as u64, RowBoundProvenance::PlanExact)),
            SourceNode::LocalParquet(source) => {
                let rows = source.exact_selected_row_count()?;
                let rows = source.scan_limit.map_or(rows, |limit| rows.min(limit as u64));
                Some((
                    source.max_rows.map_or(rows, |max_rows| rows.min(max_rows as u64)),
                    RowBoundProvenance::PlanHardUpperBound,
                ))
            },
            SourceNode::GpuResident(source) => {
                source.row_count_upper_bound.map(|rows| (rows, RowBoundProvenance::PlanHardUpperBound))
            },
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
        PlanNode::Aggregate(aggregate) => match &aggregate.spec.grouping_sets {
            None if aggregate.spec.group_exprs.is_empty() => Some((1, RowBoundProvenance::PlanExact)),
            None => input_bound(aggregate.input).map(|(rows, _)| (rows, RowBoundProvenance::PlanHardUpperBound)),
            Some(sets) => {
                let (rows, _) = input_bound(aggregate.input)?;
                // A grouping set emits at most one row per input row, except
                // the empty set (every key nulled): its grand-total row exists
                // even for empty input.
                sets.groups
                    .iter()
                    .try_fold(0_u64, |total, mask| {
                        total.checked_add(if mask.iter().all(|nulled| *nulled) { 1 } else { rows })
                    })
                    .map(|rows| (rows, RowBoundProvenance::PlanHardUpperBound))
            },
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
        PlanNode::EdgeNormalize(_) | PlanNode::VectorAlgorithm(_) | PlanNode::MlAlgorithm(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::{
        array::Int32Array,
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    };

    use super::query_plan_hard_row_upper_bound;
    use crate::{
        expr::{AggregateExpr, AggregateFunction, AggregateSpec, Expr, GroupingSetsSpec},
        plan::QueryPlanBuilder,
        source::{LiteralBatchSource, SourceNode},
    };

    #[test]
    fn hard_row_bound_counts_every_grouping_set() {
        let input_schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(Arc::clone(&input_schema), vec![Arc::new(Int32Array::from(vec![0]))]).unwrap();
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(input_schema, vec![batch])));
        let aggregate = builder
            .add_aggregate(
                source,
                AggregateSpec::new(
                    vec![Expr::column("a")],
                    vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("a"), "count")],
                )
                .with_grouping_sets(GroupingSetsSpec::new(vec![vec![false], vec![true]], "grouping_id")),
                Arc::new(Schema::new(vec![
                    Field::new("a", DataType::Int32, true),
                    Field::new("count", DataType::UInt64, false),
                ])),
            )
            .unwrap();
        let plan = builder.finish_with_host_sink(aggregate).unwrap();

        assert_eq!(query_plan_hard_row_upper_bound(&plan), Some(2));
    }

    #[test]
    fn hard_row_bound_keeps_the_empty_grouping_set_total_row_for_empty_input() {
        let input_schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let empty =
            RecordBatch::try_new(Arc::clone(&input_schema), vec![Arc::new(Int32Array::from(Vec::<i32>::new()))])
                .unwrap();
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(input_schema, vec![empty])));
        let aggregate = builder
            .add_aggregate(
                source,
                AggregateSpec::new(
                    vec![Expr::column("a")],
                    vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("a"), "count")],
                )
                .with_grouping_sets(GroupingSetsSpec::new(vec![vec![false], vec![true]], "grouping_id")),
                Arc::new(Schema::new(vec![
                    Field::new("a", DataType::Int32, true),
                    Field::new("count", DataType::UInt64, false),
                ])),
            )
            .unwrap();
        let plan = builder.finish_with_host_sink(aggregate).unwrap();

        assert_eq!(query_plan_hard_row_upper_bound(&plan), Some(1), "ROLLUP emits its total row for empty input");
    }
}

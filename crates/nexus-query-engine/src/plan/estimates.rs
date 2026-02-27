//! Conservative row and byte estimates derived from plan nodes.
//!
//! These are descriptive admission / optimizer signals only — they must not
//! be treated as exact runtime cardinalities. Unknown estimates return
//! `None` rather than zero so callers can distinguish "missing evidence"
//! from "empty".

use super::*;

/// Returns a conservative row-count estimate for a plan node when available.
///
/// Estimates are descriptive optimizer/admission evidence only. They must not
/// be treated as exact execution results.
pub(crate) fn estimated_output_rows(plan: &QueryPlan, node_id: NodeId) -> Option<u64> {
    match &plan.node(node_id)?.kind {
        PlanNode::Source(source) => match source.as_ref() {
            SourceNode::LocalParquet(source) => {
                let rows = source
                    .files
                    .iter()
                    .map(|file| {
                        file.row_group_rows.as_ref()?.iter().try_fold(0_u64, |sum, rows| sum.checked_add(*rows))
                    })
                    .try_fold(0_u64, |sum, rows| sum.checked_add(rows?))?;
                Some(source.scan_limit.map_or(rows, |limit| rows.min(limit as u64)))
            },
            SourceNode::GpuResident(_) => None,
            SourceNode::LiteralBatch(source) => Some(source.row_count() as u64),
        },
        PlanNode::Projection(projection) => estimated_output_rows(plan, projection.input),
        PlanNode::Filter(filter) => estimated_output_rows(plan, filter.input),
        PlanNode::Window(window) => estimated_output_rows(plan, window.input),
        PlanNode::Sort(sort) => estimated_output_rows(plan, sort.input),
        PlanNode::Limit(limit) => estimated_output_rows(plan, limit.input).map(|rows| {
            limit.spec.fetch.map_or(rows.saturating_sub(limit.spec.skip as u64), |fetch| {
                rows.min(limit.spec.skip.saturating_add(fetch) as u64).saturating_sub(limit.spec.skip as u64)
            })
        }),
        PlanNode::Union(union) => {
            union.inputs.iter().try_fold(0_u64, |total, input| total.checked_add(estimated_output_rows(plan, *input)?))
        },
        PlanNode::Aggregate(aggregate) if aggregate.spec.group_exprs.is_empty() => Some(1),
        PlanNode::Aggregate(_) | PlanNode::Join(_) | PlanNode::EdgeNormalize(_) | PlanNode::GraphAlgorithm(_) => None,
    }
}

/// Returns a conservative byte estimate for a plan node when available.
///
/// Source estimates prefer projected uncompressed bytes because that is the
/// closest retained-output signal available during planning.
pub(crate) fn estimated_output_bytes(plan: &QueryPlan, node_id: NodeId) -> Option<u64> {
    match &plan.node(node_id)?.kind {
        PlanNode::Source(source) => match source.as_ref() {
            SourceNode::LocalParquet(source) => source
                .diagnostics
                .selected_row_group_stats
                .as_ref()
                .and_then(|stats| stats.selected_source_bytes())
                .or(source.diagnostics.known_bytes),
            SourceNode::GpuResident(_) => None,
            SourceNode::LiteralBatch(source) => source
                .batches
                .iter()
                .try_fold(0_u64, |sum, batch| sum.checked_add(batch.get_array_memory_size() as u64)),
        },
        PlanNode::Projection(projection) => estimated_output_bytes(plan, projection.input),
        PlanNode::Filter(filter) => estimated_output_bytes(plan, filter.input),
        PlanNode::Window(_) => None,
        PlanNode::Sort(sort) => estimated_output_bytes(plan, sort.input),
        PlanNode::Union(union) => {
            union.inputs.iter().try_fold(0_u64, |total, input| total.checked_add(estimated_output_bytes(plan, *input)?))
        },
        PlanNode::Limit(_)
        | PlanNode::Aggregate(_)
        | PlanNode::Join(_)
        | PlanNode::EdgeNormalize(_)
        | PlanNode::GraphAlgorithm(_) => None,
    }
}

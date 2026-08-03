//! Conservative row and byte estimates derived from plan nodes.
//!
//! These are descriptive admission / optimizer signals only — they must not
//! be treated as exact runtime cardinalities. Unknown estimates return
//! `None` rather than zero so callers can distinguish "missing evidence"
//! from "empty".

use std::collections::HashSet;

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
                let rows = source.scan_limit.map_or(rows, |limit| rows.min(limit as u64));
                Some(source.max_rows.map_or(rows, |max_rows| rows.min(max_rows as u64)))
            },
            // A generic GPU-resident handle has no plan-time cardinality, but
            // a frontend may attach a proved hard upper bound. Preserve that
            // evidence for all admission and optimizer consumers instead of
            // making graph-specific callers inspect source internals.
            SourceNode::GpuResident(source) => source.row_count_upper_bound,
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
        PlanNode::Aggregate(_)
        | PlanNode::Join(_)
        | PlanNode::EdgeNormalize(_)
        | PlanNode::GraphAlgorithm(_)
        | PlanNode::VectorAlgorithm(_) => None,
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
        PlanNode::Join(join)
            if matches!(join.spec.join_type, JoinType::LeftSemi | JoinType::LeftAnti)
                && !join.spec.output_projection.is_empty()
                && join.spec.output_projection.iter().all(|column| column.side == JoinSide::Left)
                && left_output_columns_are_unique(join) =>
        {
            // A left semi/anti join retains at most one copy of each left row.
            // With a unique all-left projection, the left source estimate
            // remains a descriptive upper bound for the result. Repeated
            // aliases can duplicate a physical left column, so their output
            // has no similarly trustworthy byte estimate.
            estimated_output_bytes(plan, join.inputs[0])
        },
        PlanNode::Limit(_)
        | PlanNode::Aggregate(_)
        | PlanNode::Join(_)
        | PlanNode::EdgeNormalize(_)
        | PlanNode::GraphAlgorithm(_)
        | PlanNode::VectorAlgorithm(_) => None,
    }
}

fn left_output_columns_are_unique(join: &JoinNode) -> bool {
    let mut physical_column_names = HashSet::with_capacity(join.spec.output_projection.len());
    join.spec.output_projection.iter().all(|column| physical_column_names.insert(column.column.name()))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;
    use crate::{
        expr::ColumnRef,
        plan::{JoinKey, JoinOutputColumn, JoinSide, JoinSpec, JoinType},
        source::{GpuResidentSource, LocalParquetFile, LocalParquetSource, SourceNode},
    };

    #[test]
    fn gpu_resident_source_exposes_frontend_proved_row_upper_bound() {
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
        let mut builder = QueryPlanBuilder::new();
        let input = builder.add_source(SourceNode::GpuResident(
            GpuResidentSource::new("input", schema).with_row_count_upper_bound(42),
        ));
        let plan = builder.finish_with_host_sink(input);

        assert_eq!(estimated_output_rows(&plan, input), Some(42));
    }

    #[test]
    fn gpu_resident_source_without_proved_bound_remains_unknown() {
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
        let mut builder = QueryPlanBuilder::new();
        let input = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", schema)));
        let plan = builder.finish_with_host_sink(input);

        assert_eq!(estimated_output_rows(&plan, input), None);
    }

    #[test]
    fn left_semi_output_uses_left_source_byte_estimate() {
        let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int64, false)]));
        let mut builder = QueryPlanBuilder::new();
        let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
            vec![
                LocalParquetFile::new("/tmp/left-semi-estimate-left.parquet")
                    .with_row_groups(vec![0])
                    .with_row_group_bytes(vec![64])
                    .with_projected_row_group_bytes(vec![64], vec![64])
                    .with_row_group_rows(vec![1]),
            ],
            Arc::clone(&schema),
            vec!["key".to_owned()],
        )));
        let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
            "/tmp/left-semi-estimate-right.parquet",
            Arc::clone(&schema),
            vec!["key".to_owned()],
        )));
        let join = builder.add_join(
            left,
            right,
            JoinSpec::new(
                JoinType::LeftSemi,
                vec![JoinKey::new(ColumnRef::input(0, "key", "key"), ColumnRef::input(1, "key", "key"))],
            )
            .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "key", "key"), "key")])
            .with_build_side(JoinSide::Right),
            schema,
        );
        let plan = builder.finish_with_host_sink(join);

        assert_eq!(estimated_output_bytes(&plan, join), Some(64));
    }

    #[test]
    fn left_semi_output_with_aliased_left_column_has_no_conservative_byte_estimate() {
        let input_schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int64, false)]));
        let output_schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Int64, false),
            Field::new("key_alias", DataType::Int64, false),
        ]));
        let mut builder = QueryPlanBuilder::new();
        let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
            vec![
                LocalParquetFile::new("/tmp/left-semi-estimate-aliased-left.parquet")
                    .with_row_groups(vec![0])
                    .with_row_group_bytes(vec![64])
                    .with_projected_row_group_bytes(vec![64], vec![64])
                    .with_row_group_rows(vec![1]),
            ],
            Arc::clone(&input_schema),
            vec!["key".to_owned()],
        )));
        let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
            "/tmp/left-semi-estimate-aliased-right.parquet",
            input_schema,
            vec!["key".to_owned()],
        )));
        let join = builder.add_join(
            left,
            right,
            JoinSpec::new(
                JoinType::LeftSemi,
                vec![JoinKey::new(ColumnRef::input(0, "key", "key"), ColumnRef::input(1, "key", "key"))],
            )
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "key", "key"), "key"),
                JoinOutputColumn::left(ColumnRef::input(0, "key", "key_alias"), "key_alias"),
            ])
            .with_build_side(JoinSide::Right),
            output_schema,
        );
        let plan = builder.finish_with_host_sink(join);

        assert_eq!(estimated_output_bytes(&plan, join), None);
    }
}

//! Conservative row and byte estimates derived from plan nodes.
//!
//! These are descriptive admission / optimizer signals only — they must not
//! be treated as exact runtime cardinalities. Unknown estimates return
//! `None` rather than zero so callers can distinguish "missing evidence"
//! from "empty".

use std::collections::HashSet;

use super::*;
use crate::source::LocalParquetSource;

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
            SourceNode::LocalParquet(source) => selected_local_parquet_source_bytes(source),
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

/// Returns the descriptive source-stage size for one exact local-Parquet
/// rowset. Projected uncompressed bytes are preferred; a separately known
/// source size is a modeled fallback, never a claim of exact device residency.
pub(crate) fn estimated_local_parquet_source_bytes(source: &LocalParquetSource) -> Option<u64> {
    selected_local_parquet_source_bytes(source)
}

/// Returns the descriptive retained-output bytes for one local-Parquet source.
/// Unlike a source-stage budget, a metadata-only count source legitimately
/// retains zero physical columns.
fn selected_local_parquet_source_bytes(source: &LocalParquetSource) -> Option<u64> {
    source
        .diagnostics
        .selected_row_group_stats
        .as_ref()
        .and_then(|stats| stats.selected_source_bytes())
        .or(source.diagnostics.known_bytes)
}

/// Returns the sum of selected local-Parquet source bytes reachable from the
/// plan sink when every reachable source has a known local-Parquet estimate.
///
/// This is a plan-level working-set signal, not an output-size estimate: an
/// aggregate can reduce its output to one row while still retaining a large
/// input during execution.  It deliberately refuses mixed or unknown source
/// plans rather than manufacturing a partial target from only the sources
/// that happened to retain byte metadata.
pub(crate) fn estimated_reachable_local_parquet_source_bytes(plan: &QueryPlan) -> Option<u64> {
    let unreachable = plan.unreachable_node_ids().into_iter().collect::<HashSet<_>>();
    let mut source_count = 0_usize;
    let mut total_bytes = 0_u64;

    for node in plan.nodes() {
        if unreachable.contains(&node.id) {
            continue;
        }
        let PlanNode::Source(source) = &node.kind else {
            continue;
        };
        if !matches!(source.as_ref(), SourceNode::LocalParquet(_)) {
            return None;
        }
        source_count = source_count.checked_add(1)?;
        total_bytes = total_bytes.checked_add(estimated_output_bytes(plan, node.id)?)?;
    }

    (source_count != 0).then_some(total_bytes)
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
        expr::{AggregateExpr, AggregateFunction, AggregateSpec, ColumnRef, Expr},
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

    #[test]
    fn reachable_local_parquet_source_bytes_sum_inputs_even_when_aggregate_output_is_unknown() {
        let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int64, false)]));
        let mut builder = QueryPlanBuilder::new();
        let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
            vec![
                LocalParquetFile::new("/tmp/reachable-estimate-left.parquet")
                    .with_row_groups(vec![0])
                    .with_row_group_bytes(vec![20])
                    .with_projected_row_group_bytes(vec![10], vec![30]),
            ],
            Arc::clone(&schema),
            vec!["key".to_owned()],
        )));
        let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
            vec![
                LocalParquetFile::new("/tmp/reachable-estimate-right.parquet")
                    .with_row_groups(vec![0])
                    .with_row_group_bytes(vec![40])
                    .with_projected_row_group_bytes(vec![20], vec![50]),
            ],
            Arc::clone(&schema),
            vec!["key".to_owned()],
        )));
        let union = builder.add_union(vec![left, right], Arc::clone(&schema));
        let aggregate_schema = Arc::new(Schema::new(vec![Field::new("count", DataType::UInt64, false)]));
        let aggregate = builder.add_aggregate(
            union,
            AggregateSpec::new(
                Vec::new(),
                vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("key"), "count")],
            ),
            aggregate_schema,
        );
        let plan = builder.finish_with_host_sink(aggregate);

        assert_eq!(estimated_output_bytes(&plan, aggregate), None);
        assert_eq!(estimated_reachable_local_parquet_source_bytes(&plan), Some(80));
    }

    #[test]
    fn zero_column_count_source_keeps_output_bytes_descriptive() {
        let source = LocalParquetSource::zero_column_count(vec![
            LocalParquetFile::new("/tmp/reachable-estimate-count.parquet")
                .with_row_groups(vec![0])
                .with_row_group_bytes(vec![4_096])
                .with_projected_row_group_bytes(vec![0], vec![0]),
        ]);
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(source));
        let plan = builder.finish_with_host_sink(source);

        assert_eq!(estimated_output_bytes(&plan, source), Some(0));
        assert_eq!(estimated_reachable_local_parquet_source_bytes(&plan), Some(0));
    }

    #[test]
    fn reachable_local_parquet_source_bytes_refuse_unknown_or_mixed_sources() {
        let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int64, false)]));
        let mut unknown_builder = QueryPlanBuilder::new();
        let unknown = unknown_builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
            "/tmp/reachable-estimate-unknown.parquet",
            Arc::clone(&schema),
            vec!["key".to_owned()],
        )));
        let unknown_plan = unknown_builder.finish_with_host_sink(unknown);
        assert_eq!(estimated_reachable_local_parquet_source_bytes(&unknown_plan), None);

        let mut mixed_builder = QueryPlanBuilder::new();
        let known = mixed_builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
            vec![
                LocalParquetFile::new("/tmp/reachable-estimate-known.parquet")
                    .with_row_groups(vec![0])
                    .with_projected_row_group_bytes(vec![10], vec![20]),
            ],
            Arc::clone(&schema),
            vec!["key".to_owned()],
        )));
        let resident = mixed_builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("resident", schema)));
        let union = mixed_builder
            .add_union(vec![known, resident], Arc::new(Schema::new(vec![Field::new("key", DataType::Int64, false)])));
        let mixed_plan = mixed_builder.finish_with_host_sink(union);
        assert_eq!(estimated_reachable_local_parquet_source_bytes(&mixed_plan), None);
    }
}

//! Conservative row and byte estimates derived from plan nodes.
//!
//! These are descriptive admission / optimizer signals only; they must not
//! be treated as exact runtime cardinalities. Unknown estimates return
//! `None` and never zero, so callers can distinguish "missing evidence"
//! from "empty".
//!
//! The `advisory_*` functions expose a frontend's own output estimates
//! (see [`super::AdvisoryOutputStatistics`]). They are kept apart from the
//! conservative estimates so a caller never mistakes a frontend guess for a
//! bound: the conservative functions stay unknown for joins, and the advisory
//! functions never read source facts.

use std::collections::HashSet;

use super::{JoinNode, JoinSide, JoinType, NodeId, PlanNode, QueryPlan, SourceNode};
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
                        let row_group_rows = file.row_group_rows.as_ref()?;
                        match file.row_groups.as_ref() {
                            Some(selected_row_groups) => selected_row_groups
                                .iter()
                                .try_fold(0_u64, |sum, row_group| sum.checked_add(*row_group_rows.get(*row_group)?)),
                            None => row_group_rows.iter().try_fold(0_u64, |sum, rows| sum.checked_add(*rows)),
                        }
                    })
                    .try_fold(0_u64, |sum, rows| sum.checked_add(rows?))?;
                let rows = source.scan_limit.map_or(rows, |limit| rows.min(limit as u64));
                Some(source.max_rows.map_or(rows, |max_rows| rows.min(max_rows as u64)))
            },
            // A generic GPU-resident handle has no plan-time cardinality, but
            // a frontend may attach a proved hard upper bound. Preserve that
            // evidence for all admission and optimizer consumers, so
            // graph-specific callers do not inspect source internals.
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
        // A grouped aggregate's row count is its key cardinality, which no plan
        // fact bounds below the input; a join's depends on key distribution. Both
        // stay unknown so no caller treats the input count as the output count.
        PlanNode::Aggregate(_)
        | PlanNode::Join(_)
        | PlanNode::EdgeNormalize(_)
        | PlanNode::GraphAlgorithm(_)
        | PlanNode::VectorAlgorithm(_)
        | PlanNode::MlAlgorithm(_) => None,
    }
}

/// Returns a conservative byte estimate for a plan node when available.
///
/// Source estimates prefer projected uncompressed bytes because that is the
/// closest retained-output signal available during planning.
pub fn estimated_output_bytes(plan: &QueryPlan, node_id: NodeId) -> Option<u64> {
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
            if matches!(join.spec.join_type, JoinType::LeftSemi | JoinType::LeftAnti { .. })
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
        | PlanNode::VectorAlgorithm(_)
        | PlanNode::MlAlgorithm(_) => None,
    }
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
/// Computes a plan-level working-set metric rather than output size. Plans with
/// mixed or unknown source representations return `None` rather than partial
/// totals.
#[doc(hidden)]
#[cfg(test)]
pub fn estimated_reachable_local_parquet_source_bytes(plan: &QueryPlan) -> Option<u64> {
    let mut source_count = 0_usize;
    let mut total_bytes = 0_u64;

    for node in plan.nodes() {
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

/// Returns the frontend's advisory row estimate for a node output.
///
/// Advisory statistics live on join nodes; order-preserving and row-preserving
/// operators above a join carry its estimate upward the same way the
/// conservative walkers do. Every other node has no advisory value.
pub(crate) fn advisory_output_rows(plan: &QueryPlan, node_id: NodeId) -> Option<u64> {
    match &plan.node(node_id)?.kind {
        PlanNode::Join(join) => join.spec.advisory_output_statistics?.rows.value(),
        PlanNode::Projection(projection) => advisory_output_rows(plan, projection.input),
        PlanNode::Filter(filter) => advisory_output_rows(plan, filter.input),
        PlanNode::Window(window) => advisory_output_rows(plan, window.input),
        PlanNode::Sort(sort) => advisory_output_rows(plan, sort.input),
        PlanNode::Limit(limit) => advisory_output_rows(plan, limit.input).map(|rows| {
            limit.spec.fetch.map_or(rows.saturating_sub(limit.spec.skip as u64), |fetch| {
                rows.min(limit.spec.skip.saturating_add(fetch) as u64).saturating_sub(limit.spec.skip as u64)
            })
        }),
        PlanNode::Source(_)
        | PlanNode::Union(_)
        | PlanNode::Aggregate(_)
        | PlanNode::EdgeNormalize(_)
        | PlanNode::GraphAlgorithm(_)
        | PlanNode::VectorAlgorithm(_)
        | PlanNode::MlAlgorithm(_) => None,
    }
}

/// Returns the frontend's advisory byte estimate for a node output.
///
/// Follows the same node walk as [`advisory_output_rows`] except that
/// limit and window outputs have no advisory byte estimate.
pub(crate) fn advisory_output_bytes(plan: &QueryPlan, node_id: NodeId) -> Option<u64> {
    match &plan.node(node_id)?.kind {
        PlanNode::Join(join) => join.spec.advisory_output_statistics?.bytes.value(),
        PlanNode::Projection(projection) => advisory_output_bytes(plan, projection.input),
        PlanNode::Filter(filter) => advisory_output_bytes(plan, filter.input),
        PlanNode::Sort(sort) => advisory_output_bytes(plan, sort.input),
        PlanNode::Source(_)
        | PlanNode::Window(_)
        | PlanNode::Limit(_)
        | PlanNode::Union(_)
        | PlanNode::Aggregate(_)
        | PlanNode::EdgeNormalize(_)
        | PlanNode::GraphAlgorithm(_)
        | PlanNode::VectorAlgorithm(_)
        | PlanNode::MlAlgorithm(_) => None,
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
        expr::{AggregateExpr, AggregateFunction, AggregateSpec, ColumnRef, Expr},
        plan::{JoinKey, JoinOutputColumn, JoinSide, JoinSpec, JoinType, QueryPlanBuilder},
        source::{GpuResidentSource, LocalParquetFile, LocalParquetSource, SourceNode},
    };

    #[test]
    fn gpu_resident_source_exposes_frontend_proved_row_upper_bound() {
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
        let mut builder = QueryPlanBuilder::new();
        let input = builder.add_source(SourceNode::GpuResident(
            GpuResidentSource::new("input", schema).with_row_count_upper_bound(42),
        ));
        let plan = builder.finish_with_host_sink(input).unwrap();

        assert_eq!(estimated_output_rows(&plan, input), Some(42));
    }

    #[test]
    fn gpu_resident_source_without_proved_bound_remains_unknown() {
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
        let mut builder = QueryPlanBuilder::new();
        let input = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", schema)));
        let plan = builder.finish_with_host_sink(input).unwrap();

        assert_eq!(estimated_output_rows(&plan, input), None);
    }

    #[test]
    fn local_parquet_output_rows_use_selected_physical_row_groups() {
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
        let selected_source = LocalParquetSource::from_files(
            vec![
                LocalParquetFile::new("/tmp/selected-row-estimate.parquet")
                    .with_row_groups(vec![1, 3])
                    .with_row_group_rows(vec![10, 20, 30, 40]),
            ],
            Arc::clone(&schema),
            vec!["value".to_owned()],
        );
        let mut selected_builder = QueryPlanBuilder::new();
        let selected = selected_builder.add_source(SourceNode::LocalParquet(selected_source));
        let selected_plan = selected_builder.finish_with_host_sink(selected).unwrap();
        assert_eq!(estimated_output_rows(&selected_plan, selected), Some(60));

        let invalid_source = LocalParquetSource::from_files(
            vec![
                LocalParquetFile::new("/tmp/invalid-row-estimate.parquet")
                    .with_row_groups(vec![4])
                    .with_row_group_rows(vec![10, 20, 30, 40]),
            ],
            schema,
            vec!["value".to_owned()],
        );
        let mut invalid_builder = QueryPlanBuilder::new();
        let invalid = invalid_builder.add_source(SourceNode::LocalParquet(invalid_source));
        let invalid_plan = invalid_builder.finish_with_host_sink(invalid).unwrap();
        assert_eq!(estimated_output_rows(&invalid_plan, invalid), None);
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
        let join = builder
            .add_join(
                left,
                right,
                JoinSpec::new(
                    JoinType::LeftSemi,
                    vec![JoinKey::new(ColumnRef::input(0, "key", "key"), ColumnRef::input(1, "key", "key"))],
                )
                .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "key", "key"), "key")])
                .with_build_side(JoinSide::Right),
                schema,
            )
            .unwrap();
        let plan = builder.finish_with_host_sink(join).unwrap();

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
        let join = builder
            .add_join(
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
            )
            .unwrap();
        let plan = builder.finish_with_host_sink(join).unwrap();

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
        let union = builder.add_union(vec![left, right], Arc::clone(&schema)).unwrap();
        let aggregate_schema = Arc::new(Schema::new(vec![Field::new("count", DataType::UInt64, false)]));
        let aggregate = builder
            .add_aggregate(
                union,
                AggregateSpec::new(
                    Vec::new(),
                    vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("key"), "count")],
                ),
                aggregate_schema,
            )
            .unwrap();
        let plan = builder.finish_with_host_sink(aggregate).unwrap();

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
        let plan = builder.finish_with_host_sink(source).unwrap();

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
        let unknown_plan = unknown_builder.finish_with_host_sink(unknown).unwrap();
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
            .add_union(vec![known, resident], Arc::new(Schema::new(vec![Field::new("key", DataType::Int64, false)])))
            .unwrap();
        let mixed_plan = mixed_builder.finish_with_host_sink(union).unwrap();
        assert_eq!(estimated_reachable_local_parquet_source_bytes(&mixed_plan), None);
    }
}

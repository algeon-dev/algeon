use std::{collections::BTreeSet, fs::File};

use arrow::{
    array::{
        Array, BooleanArray, Date32Array, Float32Array, Float64Array, Int32Array, Int64Array, UInt32Array, UInt64Array,
    },
    datatypes::DataType,
    record_batch::RecordBatch,
};
use parquet::{
    arrow::{ProjectionMask, arrow_reader::ParquetRecordBatchReaderBuilder},
    file::{
        metadata::RowGroupMetaData,
        reader::{FileReader, SerializedFileReader},
        statistics::Statistics,
    },
};

use super::rewrite;
use crate::{
    expr::{BinaryOp, ColumnRef, Expr, LiteralValue},
    plan::{JoinNode, JoinSide, JoinType, NodeId, PlanNode, QueryNode, QueryPlan},
    source::{LocalParquetFile, LocalParquetSource, SourceColumnRef, SourceNode},
};

/// Stable rule name used in optimizer diagnostics.
pub const DIMENSION_DERIVED_ROW_GROUP_PRUNING_RULE_NAME: &str = "dimension_derived_row_group_pruning";

const DEFAULT_MAX_DIMENSION_ROWS: u64 = 1_000_000;

/// Configuration for dimension-derived fact row-group pruning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DimensionDerivedRowGroupPruningOptions {
    /// Whether the pass may read small filtered dimension sources and prune joined fact row groups.
    pub enabled: bool,
    /// Maximum selected dimension rows the pass may scan while deriving a key range.
    pub max_dimension_rows: u64,
}

impl DimensionDerivedRowGroupPruningOptions {
    /// Returns options with the pass disabled and default limits preserved.
    pub fn disabled() -> Self {
        Self { enabled: false, max_dimension_rows: DEFAULT_MAX_DIMENSION_ROWS }
    }
}

impl Default for DimensionDerivedRowGroupPruningOptions {
    fn default() -> Self {
        Self { enabled: true, max_dimension_rows: DEFAULT_MAX_DIMENSION_ROWS }
    }
}

/// Diagnostics for dimension-derived row-group pruning candidates.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DimensionDerivedRowGroupPruningDiagnostics {
    /// Number of join-side candidates considered.
    pub considered_count: usize,
    /// Number of candidates that pruned at least one target row group.
    pub applied_count: usize,
    /// Per-candidate evidence in discovery order.
    pub candidates: Vec<DimensionDerivedRowGroupPruningCandidateDiagnostics>,
}

impl DimensionDerivedRowGroupPruningDiagnostics {
    /// Returns true when the pass produced no candidate evidence.
    pub fn is_empty(&self) -> bool {
        self.considered_count == 0 && self.applied_count == 0 && self.candidates.is_empty()
    }

    /// Merges another diagnostics value into this one.
    pub fn merge(&mut self, other: Self) {
        self.considered_count += other.considered_count;
        self.applied_count += other.applied_count;
        self.candidates.extend(other.candidates);
    }
}

/// Per-candidate dimension-derived row-group pruning evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DimensionDerivedRowGroupPruningCandidateDiagnostics {
    /// Stable rule name for this diagnostic row.
    pub rule_name: &'static str,
    /// Join node whose equality key supplied the dimension/fact mapping.
    pub candidate_node_id: NodeId,
    /// Source node scanned to derive the dimension key range.
    pub dimension_source_node_id: Option<NodeId>,
    /// Source node whose row-group selection was narrowed.
    pub target_source_node_id: Option<NodeId>,
    /// Dimension join-key column after lineage tracing.
    pub dimension_key_column: Option<String>,
    /// Target source join-key column after lineage tracing.
    pub target_key_column: Option<String>,
    /// Inclusive minimum key observed from filtered dimension rows.
    pub derived_min_key: Option<i64>,
    /// Inclusive maximum key observed from filtered dimension rows.
    pub derived_max_key: Option<i64>,
    /// Target row groups selected before pruning.
    pub selected_row_groups_before: Option<usize>,
    /// Target row groups selected after pruning.
    pub selected_row_groups_after: Option<usize>,
    /// True when the returned plan uses the narrowed row-group selection.
    pub applied: bool,
    /// Rejection reason when no pruning was applied.
    pub rejected_reason: Option<DimensionDerivedRowGroupPruningRejectedReason>,
}

/// Reasons a dimension-derived pruning candidate was not applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DimensionDerivedRowGroupPruningRejectedReason {
    /// Pass was disabled.
    PassDisabled,
    /// Join shape is not an inner single-key equi join without residual predicates.
    UnsupportedJoinShape,
    /// The dimension join key could not be traced to one local Parquet source column.
    DimensionKeyLineageUnavailable,
    /// The target join key could not be traced to one local Parquet source column.
    TargetKeyLineageUnavailable,
    /// Dimension and target keys resolved to the same source node.
    SameSource,
    /// The target lineage crosses a reused DAG node and cannot be safely mutated in place.
    TargetLineageReusedInPlan,
    /// The dimension source had no reader predicate to make the derived range selective evidence.
    DimensionPredicateAbsent,
    /// Dimension source metadata was unavailable or exceeded the configured row scan limit.
    DimensionScanNotAdmitted,
    /// The dimension predicate or key type is not supported by this pass.
    UnsupportedDimensionPredicate,
    /// No non-null dimension key survived the dimension reader predicate.
    EmptyDimensionKeyRange,
    /// The target source was not a local Parquet source with usable key row-group statistics.
    TargetStatisticsUnavailable,
    /// Target row-group statistics did not allow any represented pruning.
    NoRowGroupReduction,
}

pub(crate) struct DimensionDerivedRowGroupPruningResult {
    pub plan: QueryPlan,
    pub change: super::NativeOptimizerChange,
    pub diagnostics: DimensionDerivedRowGroupPruningDiagnostics,
}

pub(crate) fn optimize_dimension_derived_row_group_pruning(
    plan: QueryPlan,
    options: &DimensionDerivedRowGroupPruningOptions,
) -> DimensionDerivedRowGroupPruningResult {
    if !options.enabled {
        return DimensionDerivedRowGroupPruningResult {
            plan,
            change: super::NativeOptimizerChange::Unchanged,
            diagnostics: DimensionDerivedRowGroupPruningDiagnostics::default(),
        };
    }

    let original_plan = plan.clone();
    let (plan, seeded_reader_predicates) = seed_reader_predicates_from_group_key_filters(plan);
    let mut nodes = plan.nodes().to_vec();
    let parent_counts = rewrite::parent_counts(&plan);
    let mut diagnostics = DimensionDerivedRowGroupPruningDiagnostics::default();
    let mut changed = seeded_reader_predicates;

    for node in plan.nodes() {
        let PlanNode::Join(join) = &node.kind else {
            continue;
        };
        if !join_shape_supported(join) {
            continue;
        }

        for orientation in candidate_orientations(join) {
            diagnostics.considered_count += 1;
            let mut candidate = DimensionDerivedRowGroupPruningCandidateDiagnostics {
                rule_name: DIMENSION_DERIVED_ROW_GROUP_PRUNING_RULE_NAME,
                candidate_node_id: node.id,
                dimension_source_node_id: None,
                target_source_node_id: None,
                dimension_key_column: None,
                target_key_column: None,
                derived_min_key: None,
                derived_max_key: None,
                selected_row_groups_before: None,
                selected_row_groups_after: None,
                applied: false,
                rejected_reason: None,
            };

            let Some(dimension_lineage) =
                unique_lineage(&plan, &parent_counts, orientation.dimension_input, orientation.dimension_key.name())
            else {
                candidate.rejected_reason =
                    Some(DimensionDerivedRowGroupPruningRejectedReason::DimensionKeyLineageUnavailable);
                diagnostics.candidates.push(candidate);
                continue;
            };
            candidate.dimension_source_node_id = Some(dimension_lineage.source_node_id);
            candidate.dimension_key_column = Some(dimension_lineage.source_column.clone());

            let Some(target_lineage) =
                unique_lineage(&plan, &parent_counts, orientation.target_input, orientation.target_key.name())
            else {
                candidate.rejected_reason =
                    Some(DimensionDerivedRowGroupPruningRejectedReason::TargetKeyLineageUnavailable);
                diagnostics.candidates.push(candidate);
                continue;
            };
            candidate.target_source_node_id = Some(target_lineage.source_node_id);
            candidate.target_key_column = Some(target_lineage.source_column.clone());

            if dimension_lineage.source_node_id == target_lineage.source_node_id {
                candidate.rejected_reason = Some(DimensionDerivedRowGroupPruningRejectedReason::SameSource);
                diagnostics.candidates.push(candidate);
                continue;
            }
            if target_lineage.has_shared_path {
                candidate.rejected_reason =
                    Some(DimensionDerivedRowGroupPruningRejectedReason::TargetLineageReusedInPlan);
                diagnostics.candidates.push(candidate);
                continue;
            }

            let Some(dimension_source) = local_parquet_source(&plan, dimension_lineage.source_node_id) else {
                candidate.rejected_reason =
                    Some(DimensionDerivedRowGroupPruningRejectedReason::DimensionKeyLineageUnavailable);
                diagnostics.candidates.push(candidate);
                continue;
            };
            if dimension_source.reader_predicate.is_none() {
                candidate.rejected_reason =
                    Some(DimensionDerivedRowGroupPruningRejectedReason::DimensionPredicateAbsent);
                diagnostics.candidates.push(candidate);
                continue;
            }
            if !dimension_scan_admitted(dimension_source, options.max_dimension_rows) {
                candidate.rejected_reason =
                    Some(DimensionDerivedRowGroupPruningRejectedReason::DimensionScanNotAdmitted);
                diagnostics.candidates.push(candidate);
                continue;
            }

            let Some(key_filter) = derive_dimension_key_filter(dimension_source, &dimension_lineage.source_column)
            else {
                candidate.rejected_reason =
                    Some(DimensionDerivedRowGroupPruningRejectedReason::UnsupportedDimensionPredicate);
                diagnostics.candidates.push(candidate);
                continue;
            };
            let Some(key_filter) = key_filter else {
                candidate.rejected_reason = Some(DimensionDerivedRowGroupPruningRejectedReason::EmptyDimensionKeyRange);
                diagnostics.candidates.push(candidate);
                continue;
            };
            let key_bounds = key_filter.bounds();
            candidate.derived_min_key = Some(key_bounds.min);
            candidate.derived_max_key = Some(key_bounds.max);

            let Some(target_source) = mutable_local_parquet_source(&mut nodes, target_lineage.source_node_id) else {
                candidate.rejected_reason =
                    Some(DimensionDerivedRowGroupPruningRejectedReason::TargetKeyLineageUnavailable);
                diagnostics.candidates.push(candidate);
                continue;
            };
            let Some(prune_result) = prune_target_source(target_source, &target_lineage.source_column, &key_filter)
            else {
                candidate.rejected_reason =
                    Some(DimensionDerivedRowGroupPruningRejectedReason::TargetStatisticsUnavailable);
                diagnostics.candidates.push(candidate);
                continue;
            };
            candidate.selected_row_groups_before = Some(prune_result.before);
            candidate.selected_row_groups_after = Some(prune_result.after);

            if !prune_result.applied {
                candidate.rejected_reason = Some(DimensionDerivedRowGroupPruningRejectedReason::NoRowGroupReduction);
                diagnostics.candidates.push(candidate);
                continue;
            }

            candidate.applied = true;
            diagnostics.applied_count += 1;
            changed = true;
            diagnostics.candidates.push(candidate);
        }
    }

    if seeded_reader_predicates && diagnostics.applied_count == 0 {
        return DimensionDerivedRowGroupPruningResult {
            plan: original_plan,
            change: super::NativeOptimizerChange::Unchanged,
            diagnostics,
        };
    }

    let change = if changed { super::NativeOptimizerChange::Changed } else { super::NativeOptimizerChange::Unchanged };
    let root = plan.root_id();
    DimensionDerivedRowGroupPruningResult {
        plan: rewrite::finish_rewritten_plan(nodes, root, plan.sink()),
        change,
        diagnostics,
    }
}

fn seed_reader_predicates_from_group_key_filters(plan: QueryPlan) -> (QueryPlan, bool) {
    let mut nodes = plan.nodes().to_vec();
    let parent_counts = rewrite::parent_counts(&plan);
    let mut changed = false;

    for node in plan.nodes() {
        let PlanNode::Filter(filter) = &node.kind else {
            continue;
        };
        for conjunct in predicate_conjuncts(&filter.predicate) {
            changed |= push_filter_conjunct_to_reader_source(
                &plan,
                &parent_counts,
                &mut nodes,
                filter.input,
                conjunct.clone(),
                false,
                false,
            );
        }
    }

    if !changed {
        return (plan, false);
    }
    let root = plan.root_id();
    (rewrite::finish_rewritten_plan(nodes, root, plan.sink()), true)
}

/// Returns a copy of `plan` with eligible post-aggregate group-key filters
/// mirrored onto local Parquet dimension-reader predicates.
///
/// The original post-aggregate filter remains in the plan. This helper is used
/// by adapter-side object-store pruning, where row-group statistics are not
/// available inside `nexus-query-engine` but the seeded dimension predicate is
/// still needed to derive a key range.
pub fn seed_dimension_derived_reader_predicates(plan: QueryPlan) -> QueryPlan {
    seed_reader_predicates_from_group_key_filters(plan).0
}

fn predicate_conjuncts(expr: &Expr) -> Vec<&Expr> {
    match expr {
        Expr::Binary { left, op: BinaryOp::And, right, .. } => {
            let mut conjuncts = predicate_conjuncts(left);
            conjuncts.extend(predicate_conjuncts(right));
            conjuncts
        },
        _ => vec![expr],
    }
}

fn push_filter_conjunct_to_reader_source(
    plan: &QueryPlan,
    parent_counts: &[usize],
    nodes: &mut [QueryNode],
    node_id: NodeId,
    predicate: Expr,
    crossed_aggregate: bool,
    crossed_shared_path: bool,
) -> bool {
    let Some(node) = plan.node(node_id) else {
        return false;
    };
    let crossed_shared_path =
        crossed_shared_path || parent_counts.get(node_id.index()).copied().unwrap_or_default() > 1;
    match &node.kind {
        PlanNode::Source(source) => match source.as_ref() {
            SourceNode::LocalParquet(source)
                if crossed_aggregate && !crossed_shared_path && seed_predicate_is_supported(source, &predicate) =>
            {
                let Some(source) = mutable_local_parquet_source(nodes, node_id) else {
                    return false;
                };
                if source.reader_predicate.as_ref() == Some(&predicate) {
                    return false;
                }
                source.reader_predicate = Some(match source.reader_predicate.take() {
                    Some(existing) => Expr::binary(existing, BinaryOp::And, predicate),
                    None => predicate,
                });
                source.refresh_diagnostics();
                true
            },
            _ => false,
        },
        PlanNode::Filter(filter) => push_filter_conjunct_to_reader_source(
            plan,
            parent_counts,
            nodes,
            filter.input,
            predicate,
            crossed_aggregate,
            crossed_shared_path,
        ),
        PlanNode::Projection(projection) => {
            remap_projection_predicate(&predicate, projection).is_some_and(|predicate| {
                push_filter_conjunct_to_reader_source(
                    plan,
                    parent_counts,
                    nodes,
                    projection.input,
                    predicate,
                    crossed_aggregate,
                    crossed_shared_path,
                )
            })
        },
        PlanNode::Aggregate(aggregate) => {
            remap_aggregate_group_predicate(&predicate, aggregate).is_some_and(|predicate| {
                push_filter_conjunct_to_reader_source(
                    plan,
                    parent_counts,
                    nodes,
                    aggregate.input,
                    predicate,
                    true,
                    crossed_shared_path,
                )
            })
        },
        PlanNode::Join(join) => remap_join_output_predicate(&predicate, join).is_some_and(|(side, predicate)| {
            let input = match side {
                JoinSide::Left => join.inputs[0],
                JoinSide::Right => join.inputs[1],
            };
            push_filter_conjunct_to_reader_source(
                plan,
                parent_counts,
                nodes,
                input,
                predicate,
                crossed_aggregate,
                crossed_shared_path,
            )
        }),
        PlanNode::Sort(sort) => push_filter_conjunct_to_reader_source(
            plan,
            parent_counts,
            nodes,
            sort.input,
            predicate,
            crossed_aggregate,
            crossed_shared_path,
        ),
        PlanNode::Limit(_)
        | PlanNode::Window(_)
        | PlanNode::Union(_)
        | PlanNode::EdgeNormalize(_)
        | PlanNode::GraphAlgorithm(_) => false,
    }
}

fn seed_predicate_is_supported(source: &LocalParquetSource, predicate: &Expr) -> bool {
    if !predicate_is_reader_visible(source, predicate) {
        return false;
    }
    let combined = match source.reader_predicate.as_ref() {
        Some(existing) => Expr::binary(existing.clone(), BinaryOp::And, predicate.clone()),
        None => predicate.clone(),
    };
    source.reader_predicate_is_supported(&combined)
}

fn predicate_is_reader_visible(source: &LocalParquetSource, predicate: &Expr) -> bool {
    let columns = predicate_columns(predicate);
    !columns.is_empty()
        && columns.iter().all(|column| matches!(source.source_column_ref(column), Some(SourceColumnRef::File(_))))
}

fn remap_projection_predicate(predicate: &Expr, projection: &crate::plan::ProjectionNode) -> Option<Expr> {
    remap_unqualified_columns(predicate, &mut |column| {
        let output = projection.outputs.iter().find(|output| output.field.name() == column.name())?;
        simple_projection_expr(&output.expr)
    })
}

fn remap_aggregate_group_predicate(predicate: &Expr, aggregate: &crate::plan::AggregateNode) -> Option<Expr> {
    if aggregate.spec.grouping_sets.is_some() {
        return None;
    }
    remap_unqualified_columns(predicate, &mut |column| {
        aggregate
            .spec
            .group_exprs
            .iter()
            .find(|expr| aggregate_group_output_name(expr) == Some(column.name()))
            .and_then(simple_projection_expr)
    })
}

fn remap_join_output_predicate(predicate: &Expr, join: &JoinNode) -> Option<(JoinSide, Expr)> {
    if !matches!(join.spec.join_type, JoinType::Inner | JoinType::Left) || join.spec.output_projection.is_empty() {
        return None;
    }
    let mut side = None;
    let remapped = remap_unqualified_columns(predicate, &mut |column| {
        let output = join.spec.output_projection.iter().find(|output| output.output_name == column.name())?;
        if join.spec.join_type == JoinType::Left && output.side != JoinSide::Left {
            return None;
        }
        match side {
            Some(existing) if existing != output.side => return None,
            Some(_) => {},
            None => side = Some(output.side),
        }
        Some(Expr::column(output.column.name()))
    })?;
    side.map(|side| (side, remapped))
}

fn remap_unqualified_columns(expr: &Expr, map_column: &mut impl FnMut(&ColumnRef) -> Option<Expr>) -> Option<Expr> {
    match expr {
        Expr::Column(column) if column.qualifier().is_none() => map_column(column),
        Expr::Column(_) => None,
        Expr::Literal(value) => Some(Expr::Literal(value.clone())),
        Expr::Alias { expr, alias } => Some(remap_unqualified_columns(expr, map_column)?.alias(alias.clone())),
        Expr::Cast { expr, target_type } => {
            Some(Expr::cast(remap_unqualified_columns(expr, map_column)?, target_type.clone()))
        },
        Expr::TryCast { expr, target_type } => {
            Some(Expr::try_cast(remap_unqualified_columns(expr, map_column)?, target_type.clone()))
        },
        Expr::Binary { left, op, right, output_type } => Some(Expr::Binary {
            left: Box::new(remap_unqualified_columns(left, map_column)?),
            op: *op,
            right: Box::new(remap_unqualified_columns(right, map_column)?),
            output_type: output_type.clone(),
        }),
        Expr::Not(expr) => Some(Expr::logical_not(remap_unqualified_columns(expr, map_column)?)),
        Expr::IsNull(expr) => Some(Expr::is_null(remap_unqualified_columns(expr, map_column)?)),
        Expr::IsNotNull(expr) => Some(Expr::is_not_null(remap_unqualified_columns(expr, map_column)?)),
        Expr::InList { expr, list, negated } => Some(Expr::InList {
            expr: Box::new(remap_unqualified_columns(expr, map_column)?),
            list: list.iter().map(|expr| remap_unqualified_columns(expr, map_column)).collect::<Option<Vec<_>>>()?,
            negated: *negated,
        }),
        Expr::Like { .. }
        | Expr::Substring { .. }
        | Expr::Lower(_)
        | Expr::Upper(_)
        | Expr::StringLength { .. }
        | Expr::RegexpReplace { .. }
        | Expr::Concat(_)
        | Expr::Abs(_)
        | Expr::Sqrt { .. }
        | Expr::Round { .. }
        | Expr::IsDistinctFrom { .. }
        | Expr::IsNotDistinctFrom { .. }
        | Expr::Case { .. }
        | Expr::Extract { .. }
        | Expr::DateTrunc { .. } => None,
    }
}

fn simple_projection_expr(expr: &Expr) -> Option<Expr> {
    match expr {
        Expr::Column(column) => Some(Expr::column(column.name())),
        Expr::Alias { expr, .. } => simple_projection_expr(expr),
        _ => None,
    }
}

fn aggregate_group_output_name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Column(column) => Some(column.display_name()),
        Expr::Alias { alias, .. } => Some(alias.as_str()),
        _ => None,
    }
}

/// A dimension-derived fact row-group pruning opportunity discovered by plan
/// shape alone, without reading any Parquet data.
///
/// [`optimize_dimension_derived_row_group_pruning`] both discovers and applies
/// these for local-path Parquet fact sources. Table-format (Iceberg/Delta) fact
/// sources resolve to object-store locations that the in-engine pass cannot
/// read, so the DataFusion adapter discovers requests with
/// [`discover_dimension_derived_pruning_requests`] and applies them with its own
/// object-store footer reads — keeping `nexus-query-engine` free of object-store
/// and catalog dependencies.
///
/// A request only records that the plan SHAPE is eligible (the dimension source
/// carries a reader predicate); it does not guarantee the range can be derived
/// or that any row group will be pruned. The resolver must apply the same
/// conservative checks the local pass applies after discovery (scan-row limit,
/// supported predicate/key evaluation, non-empty derived range, target key
/// statistics present, and an actual row-group reduction) before narrowing the
/// fact source, and fall back to the unpruned scan otherwise.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DimensionDerivedPruningRequest {
    /// Source node whose reader predicate, once evaluated, yields the join-key range.
    pub dimension_source_node_id: NodeId,
    /// Dimension source column that carries the join key.
    pub dimension_key_column: String,
    /// Fact source node whose row groups can be pruned by the derived range.
    pub target_source_node_id: NodeId,
    /// Fact source column that carries the join key.
    pub target_key_column: String,
}

/// Discovers dimension-derived fact row-group pruning opportunities by plan
/// shape only, without reading any Parquet data.
///
/// This mirrors the candidate-discovery half of
/// [`optimize_dimension_derived_row_group_pruning`]: inner single-key joins
/// whose dimension and fact join keys each resolve to a unique local-Parquet
/// source lineage, where the two sources differ, the fact lineage is not reused
/// elsewhere in the plan, and the dimension source carries a reader predicate
/// that can bound the key. Admission limits and the actual key-range derivation
/// are intentionally left to the caller, which supplies the per-source key
/// range and footer statistics (for object-store sources, via its own IO).
pub fn discover_dimension_derived_pruning_requests(plan: &QueryPlan) -> Vec<DimensionDerivedPruningRequest> {
    let parent_counts = rewrite::parent_counts(plan);
    let mut requests = Vec::new();
    for node in plan.nodes() {
        let PlanNode::Join(join) = &node.kind else {
            continue;
        };
        if !join_shape_supported(join) {
            continue;
        }
        for orientation in candidate_orientations(join) {
            let Some(dimension) =
                unique_lineage(plan, &parent_counts, orientation.dimension_input, orientation.dimension_key.name())
            else {
                continue;
            };
            let Some(target) =
                unique_lineage(plan, &parent_counts, orientation.target_input, orientation.target_key.name())
            else {
                continue;
            };
            if dimension.source_node_id == target.source_node_id || target.has_shared_path {
                continue;
            }
            let Some(dimension_source) = local_parquet_source(plan, dimension.source_node_id) else {
                continue;
            };
            if dimension_source.reader_predicate.is_none() {
                continue;
            }
            requests.push(DimensionDerivedPruningRequest {
                dimension_source_node_id: dimension.source_node_id,
                dimension_key_column: dimension.source_column,
                target_source_node_id: target.source_node_id,
                target_key_column: target.source_column,
            });
        }
    }
    requests
}

fn join_shape_supported(join: &JoinNode) -> bool {
    join.spec.join_type == JoinType::Inner
        && join.spec.keys.len() == 1
        && join.spec.residual_predicate.is_none()
        && !join.spec.output_projection.is_empty()
        && join.spec.build_artifact_id.is_none()
}

fn candidate_orientations(join: &JoinNode) -> [PruningOrientation<'_>; 2] {
    let key = &join.spec.keys[0];
    [
        PruningOrientation {
            dimension_input: join.inputs[0],
            target_input: join.inputs[1],
            dimension_key: &key.left,
            target_key: &key.right,
        },
        PruningOrientation {
            dimension_input: join.inputs[1],
            target_input: join.inputs[0],
            dimension_key: &key.right,
            target_key: &key.left,
        },
    ]
}

struct PruningOrientation<'a> {
    dimension_input: NodeId,
    target_input: NodeId,
    dimension_key: &'a ColumnRef,
    target_key: &'a ColumnRef,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SourceColumnLineage {
    source_node_id: NodeId,
    source_column: String,
    has_shared_path: bool,
}

fn unique_lineage(
    plan: &QueryPlan,
    parent_counts: &[usize],
    node_id: NodeId,
    column_name: &str,
) -> Option<SourceColumnLineage> {
    let mut lineages = lineages_for_column(plan, parent_counts, node_id, column_name);
    lineages.sort_by(|left, right| {
        left.source_node_id.cmp(&right.source_node_id).then_with(|| {
            left.source_column.cmp(&right.source_column).then_with(|| left.has_shared_path.cmp(&right.has_shared_path))
        })
    });
    lineages.dedup();
    (lineages.len() == 1).then(|| lineages.remove(0))
}

fn lineages_for_column(
    plan: &QueryPlan,
    parent_counts: &[usize],
    node_id: NodeId,
    column_name: &str,
) -> Vec<SourceColumnLineage> {
    let Some(node) = plan.node(node_id) else {
        return Vec::new();
    };
    let has_shared_path = parent_counts.get(node_id.index()).copied().unwrap_or_default() > 1;
    let mark_shared = |mut lineage: SourceColumnLineage| {
        lineage.has_shared_path |= has_shared_path;
        lineage
    };
    match &node.kind {
        PlanNode::Source(source) => match source.as_ref() {
            SourceNode::LocalParquet(source) if source.source_column_ref(column_name).is_some() => {
                vec![SourceColumnLineage {
                    source_node_id: node_id,
                    source_column: column_name.to_string(),
                    has_shared_path,
                }]
            },
            _ => Vec::new(),
        },
        PlanNode::Filter(filter) => {
            lineages_for_column(plan, parent_counts, filter.input, column_name).into_iter().map(mark_shared).collect()
        },
        PlanNode::Projection(projection) => projection
            .outputs
            .iter()
            .filter(|output| output.field.name() == column_name)
            .flat_map(|output| projection_expr_lineages(plan, parent_counts, projection.input, &output.expr))
            .map(mark_shared)
            .collect(),
        PlanNode::Sort(sort) => {
            lineages_for_column(plan, parent_counts, sort.input, column_name).into_iter().map(mark_shared).collect()
        },
        PlanNode::Limit(limit) => {
            lineages_for_column(plan, parent_counts, limit.input, column_name).into_iter().map(mark_shared).collect()
        },
        PlanNode::Join(join) => join
            .spec
            .output_projection
            .iter()
            .filter(|output| output.output_name == column_name)
            .flat_map(|output| {
                let input = match output.side {
                    JoinSide::Left => join.inputs[0],
                    JoinSide::Right => join.inputs[1],
                };
                lineages_for_column(plan, parent_counts, input, output.column.name())
            })
            .map(mark_shared)
            .collect(),
        PlanNode::Aggregate(_)
        | PlanNode::Window(_)
        | PlanNode::Union(_)
        | PlanNode::EdgeNormalize(_)
        | PlanNode::GraphAlgorithm(_) => Vec::new(),
    }
}

fn projection_expr_lineages(
    plan: &QueryPlan,
    parent_counts: &[usize],
    input: NodeId,
    expr: &Expr,
) -> Vec<SourceColumnLineage> {
    match expr {
        Expr::Column(column) => lineages_for_column(plan, parent_counts, input, column.name()),
        Expr::Alias { expr, .. } => projection_expr_lineages(plan, parent_counts, input, expr),
        _ => Vec::new(),
    }
}

fn local_parquet_source(plan: &QueryPlan, node_id: NodeId) -> Option<&LocalParquetSource> {
    let node = plan.node(node_id)?;
    let PlanNode::Source(source) = &node.kind else {
        return None;
    };
    let SourceNode::LocalParquet(source) = source.as_ref() else {
        return None;
    };
    Some(source)
}

fn mutable_local_parquet_source(nodes: &mut [QueryNode], node_id: NodeId) -> Option<&mut LocalParquetSource> {
    let node = nodes.get_mut(node_id.index())?;
    if node.id != node_id {
        return None;
    }
    let PlanNode::Source(source) = &mut node.kind else {
        return None;
    };
    let SourceNode::LocalParquet(source) = source.as_mut() else {
        return None;
    };
    Some(source)
}

fn dimension_scan_admitted(source: &LocalParquetSource, max_dimension_rows: u64) -> bool {
    let mut total_rows = 0u64;
    for file in &source.files {
        let Some(path) = file.location.as_local_path() else {
            return false;
        };
        let Ok(reader) = SerializedFileReader::try_from(path) else {
            return false;
        };
        let metadata = reader.metadata();
        let row_groups = selected_or_all_row_groups(file, metadata.num_row_groups());
        for row_group in row_groups {
            let Some(row_group_metadata) = metadata.row_group(row_group).num_rows().try_into().ok() else {
                return false;
            };
            total_rows = total_rows.saturating_add(row_group_metadata);
            if total_rows > max_dimension_rows {
                return false;
            }
        }
    }
    true
}

fn selected_or_all_row_groups(file: &LocalParquetFile, row_group_count: usize) -> Vec<usize> {
    if let Some(row_groups) = &file.row_groups
        && !row_groups.is_empty()
    {
        return row_groups.clone();
    }
    (0..row_group_count).collect()
}

/// Inclusive integer key range derived from a filtered dimension, used to prune
/// joined fact row groups. Exposed so an out-of-engine resolver (the
/// DataFusion/object-store adapter) can apply the same pruning semantics to
/// object-store table-format fact sources that this in-engine pass can only
/// apply to local-path sources.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyRange {
    /// Inclusive minimum key value among rows matching the dimension predicate.
    pub min: i64,
    /// Inclusive maximum key value among rows matching the dimension predicate.
    pub max: i64,
}

/// Maximum coalesced runs a [`DimensionKeyFilter`] keeps before collapsing back
/// to a single bounding run. Bounds the filter's memory and per-row-group overlap
/// cost, and guarantees the derived filter is never coarser than the legacy
/// single-`[min,max]` behavior. Shared by the in-engine pass and the out-of-engine
/// resolver so both derive identical filters.
pub const DIMENSION_KEY_FILTER_MAX_RUNS: usize = 4096;

/// The set of dimension key values satisfying a dimension predicate, kept as a
/// sorted list of disjoint, non-adjacent inclusive runs. A fact row group is
/// retained only when at least one run overlaps its key statistics, so a
/// scattered predicate (e.g. `d_moy = 11` across years, whose `[min,max]` spans
/// the whole date-sorted fact) prunes the gaps between qualifying runs instead of
/// collapsing to a single table-spanning range.
///
/// A contiguous predicate yields a one-run filter, which is exactly the legacy
/// `[min,max]` behavior. Exposed so an out-of-engine resolver (the
/// DataFusion/object-store adapter) applies identical pruning semantics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DimensionKeyFilter {
    /// Sorted by `min`, disjoint, non-adjacent, non-empty.
    runs: Vec<KeyRange>,
}

impl DimensionKeyFilter {
    /// Builds a filter from ascending, unique keys, coalescing contiguous
    /// integers into runs. Collapses to a single bounding run when the run count
    /// would exceed [`DIMENSION_KEY_FILTER_MAX_RUNS`]. Returns `None` when the
    /// input is empty. The caller must guarantee the input is sorted ascending
    /// and deduplicated (a `BTreeSet` iteration order satisfies this).
    fn from_sorted_unique_keys(sorted_unique_keys: &[i64]) -> Option<Self> {
        let (&first, rest) = sorted_unique_keys.split_first()?;
        let mut runs = Vec::new();
        let mut run = KeyRange { min: first, max: first };
        for &value in rest {
            if value == run.max.saturating_add(1) {
                run.max = value;
            } else {
                runs.push(run);
                run = KeyRange { min: value, max: value };
            }
        }
        runs.push(run);
        if runs.len() > DIMENSION_KEY_FILTER_MAX_RUNS {
            return Some(Self::from_bounds(KeyRange {
                min: runs.first().expect("non-empty").min,
                max: runs.last().expect("non-empty").max,
            }));
        }
        Some(Self { runs })
    }

    /// Builds a filter from arbitrary integer keys, sorting and deduplicating
    /// before coalescing contiguous values into runs. Collapses to a single
    /// bounding run when the run count would exceed
    /// [`DIMENSION_KEY_FILTER_MAX_RUNS`]. Returns `None` when `keys` is empty.
    /// Used by callers that already hold a materialized keyset (e.g. a runtime
    /// dynamic-filter payload).
    pub fn from_keys(keys: &[i64]) -> Option<Self> {
        let mut sorted = keys.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        Self::from_sorted_unique_keys(&sorted)
    }

    /// A single-run filter spanning `[min,max]` — the legacy bounding-range
    /// behavior, retained for callers that only have a range.
    pub fn from_bounds(bounds: KeyRange) -> Self {
        Self { runs: vec![bounds] }
    }

    /// Inclusive `[min,max]` envelope across every run.
    pub fn bounds(&self) -> KeyRange {
        KeyRange {
            min: self.runs.first().expect("filter has at least one run").min,
            max: self.runs.last().expect("filter has at least one run").max,
        }
    }

    /// Number of disjoint runs (1 for a contiguous/bounding filter).
    pub fn run_count(&self) -> usize {
        self.runs.len()
    }

    /// Whether any run overlaps `candidate` (a fact row group's key statistics).
    pub fn overlaps(&self, candidate: KeyRange) -> bool {
        // Runs are sorted by `min` and disjoint, hence also sorted by `max`, so
        // `run.max < candidate.min` is monotonic over the slice. Find the first
        // run that could reach `candidate`; if it overlaps, the row group is kept,
        // and no later run can either (later runs start strictly higher).
        let index = self.runs.partition_point(|run| run.max < candidate.min);
        self.runs.get(index).is_some_and(|run| ranges_overlap(*run, candidate))
    }
}

/// Accumulates the qualifying dimension key values across decoded dimension
/// batches and coalesces them into a [`DimensionKeyFilter`]. Shared by the
/// in-engine local pass ([`derive_dimension_key_filter`]) and the out-of-engine
/// resolver, which folds object-store dimension batches through
/// [`Self::observe_batch`] to guarantee identical filter semantics on both sides.
#[derive(Debug, Clone, Default)]
pub struct DimensionKeyAccumulator {
    keys: BTreeSet<i64>,
}

impl DimensionKeyAccumulator {
    /// Folds one decoded dimension batch into the running key set, counting only
    /// rows that satisfy `predicate`. Returns `None` if the predicate or key
    /// column cannot be evaluated against the batch (the caller must then abandon
    /// pruning), matching the legacy per-batch primitive's contract.
    pub fn observe_batch(&mut self, batch: &RecordBatch, key_column: &str, predicate: &Expr) -> Option<()> {
        let key = batch.column(batch.schema().index_of(key_column).ok()?);
        for row in 0..batch.num_rows() {
            if !eval_bool(predicate, batch, row)? {
                continue;
            }
            let Some(value) = array_i64_value(key.as_ref(), row)? else {
                continue;
            };
            self.keys.insert(value);
        }
        Some(())
    }

    /// Coalesces the collected keys into a filter, or `None` when no key
    /// qualified (an empty dimension key set).
    pub fn finish(&self) -> Option<DimensionKeyFilter> {
        let sorted_unique_keys = self.keys.iter().copied().collect::<Vec<_>>();
        DimensionKeyFilter::from_sorted_unique_keys(&sorted_unique_keys)
    }
}

/// Folds one decoded dimension batch into a fresh [`DimensionKeyAccumulator`],
/// preserving the legacy `Option<Option<_>>` shape for callers that thread a
/// single accumulator across batches. Returns `None` on an unevaluable
/// predicate/key, `Some(None)` when the batch is empty of the accumulator's
/// prior state and no key qualified, or `Some(Some(filter))` otherwise.
///
/// Most callers should use [`DimensionKeyAccumulator`] directly; this exists for
/// single-batch derivations and tests.
pub fn dimension_key_filter_from_batch(
    batch: &RecordBatch,
    key_column: &str,
    predicate: &Expr,
) -> Option<Option<DimensionKeyFilter>> {
    let mut accumulator = DimensionKeyAccumulator::default();
    accumulator.observe_batch(batch, key_column, predicate)?;
    Some(accumulator.finish())
}

/// Selects the fact row groups whose key statistics overlap `filter`, applying
/// the same conservative rules as the in-engine pass: row groups with unknown
/// statistics are retained, and pruning is only reported as applied when it
/// strictly narrows a non-empty selection.
///
/// `row_group_key_ranges` carries one entry per candidate row group in selection
/// order; `None` means the key statistics were unavailable for that row group.
pub fn select_overlapping_row_groups(
    row_group_key_ranges: &[Option<KeyRange>],
    filter: &DimensionKeyFilter,
) -> RowGroupOverlapSelection {
    let before = row_group_key_ranges.len();
    let mut saw_statistics = false;
    let mut kept = Vec::with_capacity(before);
    for (index, range) in row_group_key_ranges.iter().enumerate() {
        match range {
            Some(range) => {
                saw_statistics = true;
                if filter.overlaps(*range) {
                    kept.push(index);
                }
            },
            None => kept.push(index),
        }
    }
    let applied = saw_statistics && !kept.is_empty() && kept.len() < before;
    RowGroupOverlapSelection { kept, before, saw_statistics, applied }
}

/// Result of [`select_overlapping_row_groups`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowGroupOverlapSelection {
    /// Indices (into the input order) of row groups to retain.
    pub kept: Vec<usize>,
    /// Total candidate row groups considered.
    pub before: usize,
    /// Whether any candidate row group exposed usable key statistics.
    pub saw_statistics: bool,
    /// Whether pruning strictly narrowed a non-empty selection.
    pub applied: bool,
}

fn derive_dimension_key_filter(source: &LocalParquetSource, key_column: &str) -> Option<Option<DimensionKeyFilter>> {
    let predicate = source.reader_predicate.as_ref()?;
    let projection_indices = reader_projection_indices(source, key_column, predicate)?;
    let mut accumulator = DimensionKeyAccumulator::default();
    for source_file in &source.files {
        let path = source_file.location.as_local_path()?;
        let file = File::open(path).ok()?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).ok()?;
        let row_group_count = builder.metadata().num_row_groups();
        let mask = ProjectionMask::roots(builder.parquet_schema(), projection_indices.iter().copied());
        let row_groups = selected_or_all_row_groups(source_file, row_group_count);
        let reader = builder.with_projection(mask).with_row_groups(row_groups).build().ok()?;
        for batch in reader {
            let batch = batch.ok()?;
            accumulator.observe_batch(&batch, key_column, predicate)?;
        }
    }
    Some(accumulator.finish())
}

fn reader_projection_indices(
    source: &LocalParquetSource,
    key_column: &str,
    predicate: &Expr,
) -> Option<BTreeSet<usize>> {
    let mut columns = BTreeSet::new();
    let SourceColumnRef::File(key_index) = source.source_column_ref(key_column)? else {
        return None;
    };
    columns.insert(key_index);
    for column in predicate_columns(predicate) {
        let SourceColumnRef::File(column_index) = source.source_column_ref(&column)? else {
            return None;
        };
        columns.insert(column_index);
    }
    Some(columns)
}

fn predicate_columns(expr: &Expr) -> Vec<String> {
    let mut columns = Vec::new();
    collect_expr_columns(expr, &mut columns);
    columns.sort();
    columns.dedup();
    columns
}

fn collect_expr_columns(expr: &Expr, columns: &mut Vec<String>) {
    match expr {
        Expr::Column(column) => columns.push(column.name().to_string()),
        Expr::Not(inner)
        | Expr::IsNull(inner)
        | Expr::IsNotNull(inner)
        | Expr::Alias { expr: inner, .. }
        | Expr::Cast { expr: inner, .. }
        | Expr::TryCast { expr: inner, .. }
        | Expr::Lower(inner)
        | Expr::Upper(inner)
        | Expr::Abs(inner) => collect_expr_columns(inner, columns),
        Expr::Binary { left, right, .. }
        | Expr::IsDistinctFrom { left, right }
        | Expr::IsNotDistinctFrom { left, right } => {
            collect_expr_columns(left, columns);
            collect_expr_columns(right, columns);
        },
        Expr::InList { expr, list, .. } => {
            collect_expr_columns(expr, columns);
            for item in list {
                collect_expr_columns(item, columns);
            }
        },
        Expr::Like { expr, .. }
        | Expr::Substring { expr, .. }
        | Expr::StringLength { expr, .. }
        | Expr::RegexpReplace { expr, .. }
        | Expr::Sqrt { expr, .. }
        | Expr::Round { expr, .. }
        | Expr::Extract { expr, .. }
        | Expr::DateTrunc { expr, .. } => collect_expr_columns(expr, columns),
        Expr::Concat(exprs) => {
            for expr in exprs {
                collect_expr_columns(expr, columns);
            }
        },
        Expr::Case { branches, else_expr } => {
            for branch in branches {
                collect_expr_columns(&branch.when, columns);
                collect_expr_columns(&branch.then, columns);
            }
            collect_expr_columns(else_expr, columns);
        },
        Expr::Literal(_) => {},
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Scalar {
    Boolean(bool),
    Int64(i64),
    UInt64(u64),
    Float64(f64),
    Null,
}

fn eval_bool(expr: &Expr, batch: &RecordBatch, row: usize) -> Option<bool> {
    match eval_scalar(expr, batch, row)? {
        Scalar::Boolean(value) => Some(value),
        Scalar::Null => Some(false),
        _ => None,
    }
}

fn eval_scalar(expr: &Expr, batch: &RecordBatch, row: usize) -> Option<Scalar> {
    match expr {
        Expr::Column(column) => array_scalar(batch.column(batch.schema().index_of(column.name()).ok()?).as_ref(), row),
        Expr::Literal(value) => literal_scalar(value),
        Expr::Alias { expr, .. } => eval_scalar(expr, batch, row),
        Expr::Cast { expr, target_type } => cast_scalar(eval_scalar(expr, batch, row)?, target_type),
        Expr::Binary { left, op, right, .. } => {
            eval_binary(eval_scalar(left, batch, row)?, *op, eval_scalar(right, batch, row)?)
        },
        Expr::Not(inner) => match eval_scalar(inner, batch, row)? {
            Scalar::Boolean(value) => Some(Scalar::Boolean(!value)),
            Scalar::Null => Some(Scalar::Null),
            _ => None,
        },
        Expr::IsNull(inner) => Some(Scalar::Boolean(matches!(eval_scalar(inner, batch, row)?, Scalar::Null))),
        Expr::IsNotNull(inner) => Some(Scalar::Boolean(!matches!(eval_scalar(inner, batch, row)?, Scalar::Null))),
        Expr::InList { expr, list, negated } => {
            let value = eval_scalar(expr, batch, row)?;
            if matches!(value, Scalar::Null) {
                return Some(Scalar::Null);
            }
            let mut matched = false;
            for item in list {
                if scalar_eq(&value, &eval_scalar(item, batch, row)?)? {
                    matched = true;
                    break;
                }
            }
            Some(Scalar::Boolean(if *negated { !matched } else { matched }))
        },
        Expr::TryCast { .. }
        | Expr::Like { .. }
        | Expr::Substring { .. }
        | Expr::Lower(_)
        | Expr::Upper(_)
        | Expr::StringLength { .. }
        | Expr::RegexpReplace { .. }
        | Expr::Concat(_)
        | Expr::Abs(_)
        | Expr::Sqrt { .. }
        | Expr::Round { .. }
        | Expr::IsDistinctFrom { .. }
        | Expr::IsNotDistinctFrom { .. }
        | Expr::Case { .. }
        | Expr::Extract { .. }
        | Expr::DateTrunc { .. } => None,
    }
}

fn array_scalar(array: &dyn Array, row: usize) -> Option<Scalar> {
    if array.is_null(row) {
        return Some(Scalar::Null);
    }
    match array.data_type() {
        DataType::Boolean => Some(Scalar::Boolean(array.as_any().downcast_ref::<BooleanArray>()?.value(row))),
        DataType::Int32 => Some(Scalar::Int64(i64::from(array.as_any().downcast_ref::<Int32Array>()?.value(row)))),
        DataType::Date32 => Some(Scalar::Int64(i64::from(array.as_any().downcast_ref::<Date32Array>()?.value(row)))),
        DataType::Int64 => Some(Scalar::Int64(array.as_any().downcast_ref::<Int64Array>()?.value(row))),
        DataType::UInt32 => Some(Scalar::UInt64(u64::from(array.as_any().downcast_ref::<UInt32Array>()?.value(row)))),
        DataType::UInt64 => Some(Scalar::UInt64(array.as_any().downcast_ref::<UInt64Array>()?.value(row))),
        DataType::Float32 => {
            Some(Scalar::Float64(f64::from(array.as_any().downcast_ref::<Float32Array>()?.value(row))))
        },
        DataType::Float64 => Some(Scalar::Float64(array.as_any().downcast_ref::<Float64Array>()?.value(row))),
        _ => None,
    }
}

fn array_i64_value(array: &dyn Array, row: usize) -> Option<Option<i64>> {
    if array.is_null(row) {
        return Some(None);
    }
    match array.data_type() {
        DataType::Int32 => Some(Some(i64::from(array.as_any().downcast_ref::<Int32Array>()?.value(row)))),
        DataType::Date32 => Some(Some(i64::from(array.as_any().downcast_ref::<Date32Array>()?.value(row)))),
        DataType::Int64 => Some(Some(array.as_any().downcast_ref::<Int64Array>()?.value(row))),
        DataType::UInt32 => Some(Some(i64::from(array.as_any().downcast_ref::<UInt32Array>()?.value(row)))),
        DataType::UInt64 => i64::try_from(array.as_any().downcast_ref::<UInt64Array>()?.value(row)).ok().map(Some),
        _ => None,
    }
}

fn literal_scalar(value: &LiteralValue) -> Option<Scalar> {
    match value {
        LiteralValue::Boolean(value) => Some(Scalar::Boolean(*value)),
        LiteralValue::Int8(value) => Some(Scalar::Int64(i64::from(*value))),
        LiteralValue::Int16(value) => Some(Scalar::Int64(i64::from(*value))),
        LiteralValue::Int32(value) | LiteralValue::Date32(value) => Some(Scalar::Int64(i64::from(*value))),
        LiteralValue::Int64(value) => Some(Scalar::Int64(*value)),
        LiteralValue::UInt8(value) => Some(Scalar::UInt64(u64::from(*value))),
        LiteralValue::UInt16(value) => Some(Scalar::UInt64(u64::from(*value))),
        LiteralValue::UInt32(value) => Some(Scalar::UInt64(u64::from(*value))),
        LiteralValue::UInt64(value) => Some(Scalar::UInt64(*value)),
        LiteralValue::Float32(value) => Some(Scalar::Float64(f64::from(*value))),
        LiteralValue::Float64(value) => Some(Scalar::Float64(*value)),
        LiteralValue::Null(_) => Some(Scalar::Null),
        LiteralValue::Utf8(_)
        | LiteralValue::LargeUtf8(_)
        | LiteralValue::Decimal128 { .. }
        | LiteralValue::Timestamp { .. }
        | LiteralValue::Duration { .. }
        | LiteralValue::IntervalMonthDayNano { .. } => None,
    }
}

fn cast_scalar(value: Scalar, target_type: &DataType) -> Option<Scalar> {
    if matches!(value, Scalar::Null) {
        return Some(Scalar::Null);
    }
    match target_type {
        DataType::Int32 | DataType::Int64 | DataType::Date32 => scalar_to_i64(value).map(Scalar::Int64),
        DataType::UInt32 | DataType::UInt64 => scalar_to_u64(value).map(Scalar::UInt64),
        DataType::Float32 | DataType::Float64 => scalar_to_f64(value).map(Scalar::Float64),
        DataType::Boolean => match value {
            Scalar::Boolean(value) => Some(Scalar::Boolean(value)),
            _ => None,
        },
        _ => None,
    }
}

fn eval_binary(left: Scalar, op: BinaryOp, right: Scalar) -> Option<Scalar> {
    if matches!(op, BinaryOp::And | BinaryOp::Or) {
        return eval_boolean_binary(left, op, right);
    }
    if matches!(left, Scalar::Null) || matches!(right, Scalar::Null) {
        return Some(Scalar::Null);
    }
    match op {
        BinaryOp::Eq => scalar_eq(&left, &right).map(Scalar::Boolean),
        BinaryOp::NotEq => scalar_eq(&left, &right).map(|value| Scalar::Boolean(!value)),
        BinaryOp::Lt => scalar_cmp(&left, &right).map(|ordering| Scalar::Boolean(ordering < 0)),
        BinaryOp::LtEq => scalar_cmp(&left, &right).map(|ordering| Scalar::Boolean(ordering <= 0)),
        BinaryOp::Gt => scalar_cmp(&left, &right).map(|ordering| Scalar::Boolean(ordering > 0)),
        BinaryOp::GtEq => scalar_cmp(&left, &right).map(|ordering| Scalar::Boolean(ordering >= 0)),
        BinaryOp::Add
        | BinaryOp::Subtract
        | BinaryOp::Multiply
        | BinaryOp::Divide
        | BinaryOp::Modulo
        | BinaryOp::BitwiseAnd
        | BinaryOp::BitwiseOr
        | BinaryOp::BitwiseXor
        | BinaryOp::ShiftLeft
        | BinaryOp::ShiftRight
        | BinaryOp::And
        | BinaryOp::Or => None,
    }
}

fn eval_boolean_binary(left: Scalar, op: BinaryOp, right: Scalar) -> Option<Scalar> {
    let left = match left {
        Scalar::Boolean(value) => value,
        Scalar::Null => false,
        _ => return None,
    };
    let right = match right {
        Scalar::Boolean(value) => value,
        Scalar::Null => false,
        _ => return None,
    };
    match op {
        BinaryOp::And => Some(Scalar::Boolean(left && right)),
        BinaryOp::Or => Some(Scalar::Boolean(left || right)),
        _ => None,
    }
}

fn scalar_eq(left: &Scalar, right: &Scalar) -> Option<bool> {
    if matches!((left, right), (Scalar::Null, _) | (_, Scalar::Null)) {
        return Some(false);
    }
    Some(scalar_cmp(left, right)? == 0)
}

fn scalar_cmp(left: &Scalar, right: &Scalar) -> Option<i8> {
    if let (Some(left), Some(right)) = (scalar_to_i64(left.clone()), scalar_to_i64(right.clone())) {
        return Some(left.cmp(&right) as i8);
    }
    let left = scalar_to_f64(left.clone())?;
    let right = scalar_to_f64(right.clone())?;
    left.partial_cmp(&right).map(|ordering| match ordering {
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Equal => 0,
        std::cmp::Ordering::Greater => 1,
    })
}

fn scalar_to_i64(value: Scalar) -> Option<i64> {
    match value {
        Scalar::Int64(value) => Some(value),
        Scalar::UInt64(value) => i64::try_from(value).ok(),
        Scalar::Float64(value) if value.fract() == 0.0 => Some(value as i64),
        Scalar::Boolean(_) | Scalar::Float64(_) | Scalar::Null => None,
    }
}

fn scalar_to_u64(value: Scalar) -> Option<u64> {
    match value {
        Scalar::Int64(value) => u64::try_from(value).ok(),
        Scalar::UInt64(value) => Some(value),
        Scalar::Float64(value) if value.fract() == 0.0 && value >= 0.0 => Some(value as u64),
        Scalar::Boolean(_) | Scalar::Float64(_) | Scalar::Null => None,
    }
}

fn scalar_to_f64(value: Scalar) -> Option<f64> {
    match value {
        Scalar::Int64(value) => Some(value as f64),
        Scalar::UInt64(value) => Some(value as f64),
        Scalar::Float64(value) => Some(value),
        Scalar::Boolean(_) | Scalar::Null => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TargetPruneResult {
    before: usize,
    after: usize,
    applied: bool,
}

fn prune_target_source(
    source: &mut LocalParquetSource,
    key_column: &str,
    key_filter: &DimensionKeyFilter,
) -> Option<TargetPruneResult> {
    let SourceColumnRef::File(key_file_index) = source.source_column_ref(key_column)? else {
        return None;
    };
    let mut before = 0usize;
    let mut after = 0usize;
    let mut saw_statistics = false;
    let mut new_row_groups_by_file = Vec::with_capacity(source.files.len());

    for file in &source.files {
        let path = file.location.as_local_path()?;
        let reader = SerializedFileReader::try_from(path).ok()?;
        let metadata = reader.metadata();
        let selected = selected_or_all_row_groups(file, metadata.num_row_groups());
        before += selected.len();
        let mut kept = Vec::new();
        for row_group_index in selected {
            let row_group = metadata.row_group(row_group_index);
            match row_group_key_range(row_group, key_file_index) {
                Some(row_group_range) => {
                    saw_statistics = true;
                    if key_filter.overlaps(row_group_range) {
                        kept.push(row_group_index);
                    }
                },
                None => kept.push(row_group_index),
            }
        }
        after += kept.len();
        new_row_groups_by_file.push(kept);
    }

    if !saw_statistics {
        return None;
    }
    if after == 0 || after >= before {
        return Some(TargetPruneResult { before, after, applied: false });
    }

    for (file, row_groups) in source.files.iter_mut().zip(new_row_groups_by_file) {
        file.row_groups = Some(row_groups);
    }
    source.refresh_diagnostics();
    Some(TargetPruneResult { before, after, applied: true })
}

/// Outcome of [`apply_dimension_derived_pruning_with_stats`].
#[derive(Debug, Clone)]
pub struct DimensionDerivedApplyResult {
    /// Plan with the fact source's row-group selection narrowed when pruning applied.
    pub plan: QueryPlan,
    /// Whether pruning strictly narrowed the fact source.
    pub applied: bool,
    /// Total candidate fact row groups before pruning.
    pub before: usize,
    /// Retained fact row groups after pruning.
    pub after: usize,
}

/// One per-file row-group selection derived outside the native engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalParquetRowGroupSelection {
    /// Physical Parquet row-group count in the file footer.
    pub total_row_groups: usize,
    /// Physical row-group ids retained by the external pruning pass.
    pub row_groups: Vec<usize>,
}

/// Outcome of [`apply_local_parquet_row_group_selections`].
#[derive(Debug, Clone)]
pub struct LocalParquetRowGroupSelectionApplyResult {
    /// Plan with the source row-group selection narrowed when pruning applied.
    pub plan: QueryPlan,
    /// Whether pruning strictly narrowed the source.
    pub applied: bool,
    /// Total candidate row groups before pruning.
    pub before: usize,
    /// Retained row groups after pruning.
    pub after: usize,
}

/// Applies externally-derived row-group selections to a `LocalParquetSource`.
///
/// The supplied selections are physical Parquet row-group ids, one entry per
/// source file. Existing file-level selections are narrowed, never widened. An
/// empty result is treated as non-applied for now because native empty-source
/// representation is not modeled by this patching API.
pub fn apply_local_parquet_row_group_selections(
    plan: QueryPlan,
    source_node_id: NodeId,
    selections: &[LocalParquetRowGroupSelection],
) -> LocalParquetRowGroupSelectionApplyResult {
    let mut nodes = plan.nodes().to_vec();
    let root = plan.root_id();
    let sink = plan.sink().clone();
    let dynamic_filter_routes = plan.dynamic_filter_routes().to_vec();
    let unchanged = |nodes: Vec<QueryNode>| LocalParquetRowGroupSelectionApplyResult {
        plan: rewrite::finish_rewritten_plan(nodes, root, &sink)
            .with_dynamic_filter_routes(dynamic_filter_routes.clone()),
        applied: false,
        before: 0,
        after: 0,
    };

    let Some(source) = mutable_local_parquet_source(&mut nodes, source_node_id) else {
        return unchanged(nodes);
    };
    if selections.len() != source.files.len() {
        return unchanged(nodes);
    }

    let mut before = 0usize;
    let mut after = 0usize;
    let mut new_row_groups_by_file = Vec::with_capacity(source.files.len());
    for (file, selection) in source.files.iter().zip(selections) {
        if selection.total_row_groups == 0 {
            return unchanged(nodes);
        }
        let candidates = selected_or_all_row_groups(file, selection.total_row_groups);
        if candidates.iter().any(|row_group| *row_group >= selection.total_row_groups) {
            return unchanged(nodes);
        }
        let candidate_set = candidates.iter().copied().collect::<BTreeSet<_>>();
        let mut seen = BTreeSet::new();
        let mut previous_candidate_position = None;
        for row_group in &selection.row_groups {
            if !seen.insert(*row_group) {
                return unchanged(nodes);
            }
            if !candidate_set.contains(row_group) {
                return unchanged(nodes);
            }
            let Some(candidate_position) = candidates.iter().position(|candidate| candidate == row_group) else {
                return unchanged(nodes);
            };
            if previous_candidate_position.is_some_and(|previous| candidate_position <= previous) {
                return unchanged(nodes);
            }
            previous_candidate_position = Some(candidate_position);
        }
        before += candidates.len();
        after += selection.row_groups.len();
        new_row_groups_by_file.push(selection.row_groups.clone());
    }

    if after == 0 || after >= before {
        return unchanged(nodes);
    }

    for (file, row_groups) in source.files.iter_mut().zip(new_row_groups_by_file) {
        file.row_groups = Some(row_groups);
    }
    source.refresh_diagnostics();
    LocalParquetRowGroupSelectionApplyResult {
        plan: rewrite::finish_rewritten_plan(nodes, root, &sink).with_dynamic_filter_routes(dynamic_filter_routes),
        applied: true,
        before,
        after,
    }
}

/// Applies an externally-derived dimension key filter to a fact `LocalParquetSource`
/// using caller-supplied per-row-group key statistics, returning a patched plan.
///
/// This is the object-store-free counterpart of the in-engine
/// [`prune_target_source`] local-footer path: an out-of-engine resolver (the
/// DataFusion/object-store adapter) reads the fact Parquet footers itself and
/// passes one `Vec<Option<KeyRange>>` per source file (aligned with
/// `source.files`). Within each file the stats are in the file's
/// CURRENTLY-SELECTED row-group order — i.e. aligned with `file.row_groups`, or
/// all physical row groups when no selection is set — and `None` marks a row
/// group whose key statistics are absent. Kept positions are mapped back to
/// physical row-group ids through that existing selection, so a pre-existing
/// `row_groups` is narrowed, never replaced by raw positions.
///
/// `filter` is the set-aware [`DimensionKeyFilter`] the resolver builds with the
/// same [`DimensionKeyAccumulator`] the in-engine pass uses, so a row group is
/// retained iff one of the filter's runs overlaps its key statistics.
///
/// The selection rule is identical to [`prune_target_source`] and conservative —
/// row groups with unknown stats are retained, pruning never empties a
/// selection, and the plan is only changed when pruning strictly narrows a
/// non-empty selection. On any mismatch (missing source, wrong file/stat arity,
/// no usable statistics, or no reduction) the plan is returned unchanged with
/// `applied = false`.
pub fn apply_dimension_derived_pruning_with_stats(
    plan: QueryPlan,
    target_source_node_id: NodeId,
    per_file_row_group_key_ranges: &[Vec<Option<KeyRange>>],
    filter: &DimensionKeyFilter,
) -> DimensionDerivedApplyResult {
    let mut nodes = plan.nodes().to_vec();
    let root = plan.root_id();
    let sink = plan.sink().clone();
    let unchanged = |nodes: Vec<QueryNode>| DimensionDerivedApplyResult {
        plan: rewrite::finish_rewritten_plan(nodes, root, &sink),
        applied: false,
        before: 0,
        after: 0,
    };

    let Some(source) = mutable_local_parquet_source(&mut nodes, target_source_node_id) else {
        return unchanged(nodes);
    };
    if per_file_row_group_key_ranges.len() != source.files.len() {
        return unchanged(nodes);
    }

    let mut before = 0usize;
    let mut after = 0usize;
    let mut saw_statistics = false;
    let mut new_row_groups_by_file = Vec::with_capacity(source.files.len());
    for (file, row_group_key_ranges) in source.files.iter().zip(per_file_row_group_key_ranges) {
        // The caller supplies one key range per CURRENTLY-SELECTED row group, in
        // selection order. Map the kept positions back to physical row-group ids
        // through the file's existing selection so a pre-existing `row_groups`
        // (e.g. a table-format file-level selection) is narrowed, never replaced
        // by raw positions. If the supplied stats do not align with the file's
        // selection, skip entirely rather than risk an unsound remap.
        let candidates = selected_or_all_row_groups(file, row_group_key_ranges.len());
        if candidates.len() != row_group_key_ranges.len() {
            return unchanged(nodes);
        }
        let selection = select_overlapping_row_groups(row_group_key_ranges, filter);
        before += selection.before;
        after += selection.kept.len();
        saw_statistics |= selection.saw_statistics;
        new_row_groups_by_file.push(selection.kept.iter().map(|position| candidates[*position]).collect::<Vec<_>>());
    }

    if !saw_statistics || after == 0 || after >= before {
        return unchanged(nodes);
    }
    for (file, row_groups) in source.files.iter_mut().zip(new_row_groups_by_file) {
        file.row_groups = Some(row_groups);
    }
    source.refresh_diagnostics();
    DimensionDerivedApplyResult {
        plan: rewrite::finish_rewritten_plan(nodes, root, &sink),
        applied: true,
        before,
        after,
    }
}

fn row_group_key_range(row_group: &RowGroupMetaData, column_index: usize) -> Option<KeyRange> {
    let statistics = row_group.columns().get(column_index)?.statistics()?;
    match statistics {
        Statistics::Int32(stats) => {
            Some(KeyRange { min: i64::from(*stats.min_opt()?), max: i64::from(*stats.max_opt()?) })
        },
        Statistics::Int64(stats) => Some(KeyRange { min: *stats.min_opt()?, max: *stats.max_opt()? }),
        _ => None,
    }
}

fn ranges_overlap(left: KeyRange, right: KeyRange) -> bool {
    left.max >= right.min && right.max >= left.min
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;
    use crate::plan::QueryPlanBuilder;

    #[test]
    fn apply_local_parquet_row_group_selections_narrows_existing_selection() {
        let (plan, source_node_id) = test_plan_with_source_row_groups(Some(vec![0, 1, 2]));

        let result = apply_local_parquet_row_group_selections(
            plan,
            source_node_id,
            &[LocalParquetRowGroupSelection { total_row_groups: 5, row_groups: vec![1, 2] }],
        );

        assert!(result.applied);
        assert_eq!(result.before, 3);
        assert_eq!(result.after, 2);
        let source = local_parquet_source(&result.plan, source_node_id).unwrap();
        assert_eq!(source.files[0].row_groups, Some(vec![1, 2]));
    }

    #[test]
    fn apply_local_parquet_row_group_selections_rejects_widening_existing_selection() {
        let (plan, source_node_id) = test_plan_with_source_row_groups(Some(vec![0, 1, 2]));

        let result = apply_local_parquet_row_group_selections(
            plan,
            source_node_id,
            &[LocalParquetRowGroupSelection { total_row_groups: 5, row_groups: vec![1, 4] }],
        );

        assert!(!result.applied);
        let source = local_parquet_source(&result.plan, source_node_id).unwrap();
        assert_eq!(source.files[0].row_groups, Some(vec![0, 1, 2]));
    }

    #[test]
    fn apply_local_parquet_row_group_selections_rejects_empty_selection() {
        let (plan, source_node_id) = test_plan_with_source_row_groups(None);

        let result = apply_local_parquet_row_group_selections(
            plan,
            source_node_id,
            &[LocalParquetRowGroupSelection { total_row_groups: 5, row_groups: Vec::new() }],
        );

        assert!(!result.applied);
        let source = local_parquet_source(&result.plan, source_node_id).unwrap();
        assert_eq!(source.files[0].row_groups, None);
    }

    #[test]
    fn apply_local_parquet_row_group_selections_rejects_duplicate_selection() {
        let (plan, source_node_id) = test_plan_with_source_row_groups(Some(vec![0, 1, 2]));

        let result = apply_local_parquet_row_group_selections(
            plan,
            source_node_id,
            &[LocalParquetRowGroupSelection { total_row_groups: 5, row_groups: vec![1, 1] }],
        );

        assert!(!result.applied);
        let source = local_parquet_source(&result.plan, source_node_id).unwrap();
        assert_eq!(source.files[0].row_groups, Some(vec![0, 1, 2]));
    }

    #[test]
    fn apply_local_parquet_row_group_selections_rejects_noncanonical_order() {
        let (plan, source_node_id) = test_plan_with_source_row_groups(Some(vec![0, 1, 2]));

        let result = apply_local_parquet_row_group_selections(
            plan,
            source_node_id,
            &[LocalParquetRowGroupSelection { total_row_groups: 5, row_groups: vec![2, 1] }],
        );

        assert!(!result.applied);
        let source = local_parquet_source(&result.plan, source_node_id).unwrap();
        assert_eq!(source.files[0].row_groups, Some(vec![0, 1, 2]));
    }

    fn test_plan_with_source_row_groups(row_groups: Option<Vec<usize>>) -> (QueryPlan, NodeId) {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)]));
        let mut file = LocalParquetFile::new("/tmp/test.parquet");
        file.row_groups = row_groups;
        let source = LocalParquetSource::from_files(vec![file], schema, vec!["x".to_string()]);
        let mut builder = QueryPlanBuilder::new();
        let source_node_id = builder.add_source(SourceNode::LocalParquet(source));
        (builder.finish_with_host_sink(source_node_id), source_node_id)
    }
}

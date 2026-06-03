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

    let mut nodes = plan.nodes().to_vec();
    let parent_counts = rewrite::parent_counts(&plan);
    let mut diagnostics = DimensionDerivedRowGroupPruningDiagnostics::default();
    let mut changed = false;

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

            let Some(key_range) = derive_dimension_key_range(dimension_source, &dimension_lineage.source_column) else {
                candidate.rejected_reason =
                    Some(DimensionDerivedRowGroupPruningRejectedReason::UnsupportedDimensionPredicate);
                diagnostics.candidates.push(candidate);
                continue;
            };
            let Some(key_range) = key_range else {
                candidate.rejected_reason = Some(DimensionDerivedRowGroupPruningRejectedReason::EmptyDimensionKeyRange);
                diagnostics.candidates.push(candidate);
                continue;
            };
            candidate.derived_min_key = Some(key_range.min);
            candidate.derived_max_key = Some(key_range.max);

            let Some(target_source) = mutable_local_parquet_source(&mut nodes, target_lineage.source_node_id) else {
                candidate.rejected_reason =
                    Some(DimensionDerivedRowGroupPruningRejectedReason::TargetKeyLineageUnavailable);
                diagnostics.candidates.push(candidate);
                continue;
            };
            let Some(prune_result) = prune_target_source(target_source, &target_lineage.source_column, key_range)
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

    let change = if changed { super::NativeOptimizerChange::Changed } else { super::NativeOptimizerChange::Unchanged };
    let sink = plan.sink().clone();
    let root = plan.root_id();
    DimensionDerivedRowGroupPruningResult {
        plan: QueryPlan::from_rewritten_parts(nodes, root, sink),
        change,
        diagnostics,
    }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct KeyRange {
    min: i64,
    max: i64,
}

fn derive_dimension_key_range(source: &LocalParquetSource, key_column: &str) -> Option<Option<KeyRange>> {
    let predicate = source.reader_predicate.as_ref()?;
    let projection_indices = reader_projection_indices(source, key_column, predicate)?;
    let mut range: Option<KeyRange> = None;
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
            range = update_key_range_from_batch(range, &batch, key_column, predicate)?;
        }
    }
    Some(range)
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

fn update_key_range_from_batch(
    mut range: Option<KeyRange>,
    batch: &RecordBatch,
    key_column: &str,
    predicate: &Expr,
) -> Option<Option<KeyRange>> {
    let key = batch.column(batch.schema().index_of(key_column).ok()?);
    for row in 0..batch.num_rows() {
        if !eval_bool(predicate, batch, row)? {
            continue;
        }
        let Some(value) = array_i64_value(key.as_ref(), row)? else {
            continue;
        };
        range = Some(match range {
            Some(existing) => KeyRange { min: existing.min.min(value), max: existing.max.max(value) },
            None => KeyRange { min: value, max: value },
        });
    }
    Some(range)
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
    key_range: KeyRange,
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
                    if ranges_overlap(row_group_range, key_range) {
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

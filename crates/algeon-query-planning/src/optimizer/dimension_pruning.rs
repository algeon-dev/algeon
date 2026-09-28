use std::collections::BTreeSet;

use arrow::{
    array::{
        Array, BooleanArray, Date32Array, Float32Array, Float64Array, Int32Array, Int64Array, LargeStringArray,
        StringArray, StringViewArray, UInt32Array, UInt64Array,
    },
    datatypes::DataType,
    record_batch::RecordBatch,
};

use super::{
    access::{
        local_parquet_source, mutable_local_parquet_source, predicate_columns, reader_projection_indices,
        reader_projection_indices_for_exprs,
    },
    lineage::{dedup_lineages, simple_lineages_for_column, unique_simple_lineage},
    rewrite,
};
use crate::{
    backend::AnalysisContext,
    error::{QueryError, QueryErrorCode, Result},
    expr::{BinaryOp, ColumnRef, Expr, LiteralValue},
    plan::{
        AggregateNode, JoinNode, JoinSide, JoinType, NodeId, PlanNode, ProjectionNode, QueryNode, QueryPlan,
        RewrittenNodes,
    },
    source::{LocalParquetSource, SourceColumnRef, SourceNode},
};

mod eval;
mod key_filter;
mod resolver;
mod seed;

pub use eval::I64ExprRangeAccumulator;
pub(crate) use eval::derive_i64_expr_range_from_filtered_subtree;
use eval::{array_i64_value, derive_dimension_key_filter, eval_bool};
pub(crate) use key_filter::select_overlapping_row_groups;
pub use key_filter::{
    DIMENSION_KEY_FILTER_MAX_RUNS, DimensionKeyAccumulator, DimensionKeyFilter, KeyRange,
    dimension_key_filter_from_batch,
};
pub use resolver::LocalParquetPruningResolver;
#[cfg(test)]
pub use resolver::NoDimensionPruningFacts;
pub use seed::seed_dimension_derived_reader_predicates;
use seed::{aggregate_group_output_name, remap_unqualified_columns, seed_reader_predicates_from_group_key_filters};

/// Stable rule name used in optimizer diagnostics.
pub const DIMENSION_DERIVED_ROW_GROUP_PRUNING_RULE_NAME: &str = "dimension_derived_row_group_pruning";

/// Configuration for dimension-derived fact row-group pruning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DimensionDerivedRowGroupPruningOptions {
    /// Whether the pass may read small filtered dimension sources and prune joined fact row groups.
    pub enabled: bool,
    /// Maximum selected dimension rows the pass may scan while deriving a key range.
    pub max_dimension_rows: u64,
}

impl DimensionDerivedRowGroupPruningOptions {
    /// Default selected dimension row cap for deriving pruning key ranges.
    pub const DEFAULT_MAX_DIMENSION_ROWS: u64 = 1_000_000;

    /// Returns options with the pass disabled and default limits preserved.
    pub fn disabled() -> Self {
        Self { enabled: false, max_dimension_rows: Self::DEFAULT_MAX_DIMENSION_ROWS }
    }

    /// Validates internal consistency for dimension-derived pruning limits.
    pub fn validate(&self) -> Result<()> {
        if self.max_dimension_rows == 0 {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "query_engine.dimension_derived_row_group_pruning_options.validate",
                "dimension-derived row-group pruning option must be greater than zero",
            )
            .with_fact("field", "max_dimension_rows")
            .with_fact("value", self.max_dimension_rows)
            .with_fact("reason", "zero"));
        }
        Ok(())
    }
}

impl Default for DimensionDerivedRowGroupPruningOptions {
    fn default() -> Self {
        Self { enabled: true, max_dimension_rows: Self::DEFAULT_MAX_DIMENSION_ROWS }
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
    pub(crate) fn is_empty(&self) -> bool {
        self.considered_count == 0 && self.applied_count == 0 && self.candidates.is_empty()
    }

    /// Merges another diagnostics value into this one.
    pub(crate) fn merge(&mut self, other: Self) {
        self.considered_count += other.considered_count;
        self.applied_count += other.applied_count;
        self.candidates.extend(other.candidates);
    }
}

/// Per-candidate dimension-derived row-group pruning evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DimensionDerivedRowGroupPruningCandidateDiagnostics {
    /// Stable rule name for this diagnostic row.
    pub(crate) rule_name: &'static str,
    /// Join node whose equality key supplied the dimension/fact mapping.
    pub(crate) candidate_node_id: NodeId,
    /// Source node scanned to derive the dimension key range.
    pub(crate) dimension_source_node_id: Option<NodeId>,
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
    /// Configured row limit for scanning a filtered dimension source.
    pub(crate) max_dimension_rows: u64,
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

#[doc(hidden)]
pub struct DimensionDerivedRowGroupPruningResult {
    pub plan: QueryPlan,
    pub change: super::NativeOptimizerChange,
    pub diagnostics: DimensionDerivedRowGroupPruningDiagnostics,
}

fn record_dimension_pruning_candidate(
    diagnostics: &mut DimensionDerivedRowGroupPruningDiagnostics,
    candidate: DimensionDerivedRowGroupPruningCandidateDiagnostics,
) {
    diagnostics.considered_count += 1;
    diagnostics.candidates.push(candidate);
}

#[doc(hidden)]
pub fn optimize_dimension_derived_row_group_pruning(
    plan: QueryPlan,
    options: &DimensionDerivedRowGroupPruningOptions,
    ctx: AnalysisContext<'_>,
    resolver: &dyn LocalParquetPruningResolver,
) -> Result<DimensionDerivedRowGroupPruningResult> {
    if !options.enabled {
        return Ok(DimensionDerivedRowGroupPruningResult {
            plan,
            change: super::NativeOptimizerChange::Unchanged,
            diagnostics: DimensionDerivedRowGroupPruningDiagnostics::default(),
        });
    }

    let original_plan = plan.clone();
    let (plan, seeded_reader_predicates) = seed_reader_predicates_from_group_key_filters(plan, ctx)?;
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
                max_dimension_rows: options.max_dimension_rows,
                applied: false,
                rejected_reason: None,
            };

            let Some(dimension_lineage) = unique_simple_lineage(
                &plan,
                &parent_counts,
                orientation.dimension_input,
                orientation.dimension_key.name(),
            ) else {
                candidate.rejected_reason =
                    Some(DimensionDerivedRowGroupPruningRejectedReason::DimensionKeyLineageUnavailable);
                record_dimension_pruning_candidate(&mut diagnostics, candidate);
                continue;
            };
            candidate.dimension_source_node_id = Some(dimension_lineage.source_node_id);
            candidate.dimension_key_column = Some(dimension_lineage.source_column.clone());

            let target_lineages = dedup_lineages(simple_lineages_for_column(
                &plan,
                &parent_counts,
                orientation.target_input,
                orientation.target_key.name(),
            ));
            if target_lineages.is_empty() {
                candidate.rejected_reason =
                    Some(DimensionDerivedRowGroupPruningRejectedReason::TargetKeyLineageUnavailable);
                record_dimension_pruning_candidate(&mut diagnostics, candidate);
                continue;
            }

            let mut target_candidates = Vec::new();
            for target_lineage in target_lineages {
                let mut target_candidate = candidate.clone();
                target_candidate.target_source_node_id = Some(target_lineage.source_node_id);
                target_candidate.target_key_column = Some(target_lineage.source_column.clone());

                if dimension_lineage.source_node_id == target_lineage.source_node_id {
                    target_candidate.rejected_reason = Some(DimensionDerivedRowGroupPruningRejectedReason::SameSource);
                    record_dimension_pruning_candidate(&mut diagnostics, target_candidate);
                    continue;
                }
                if target_lineage.has_shared_path {
                    target_candidate.rejected_reason =
                        Some(DimensionDerivedRowGroupPruningRejectedReason::TargetLineageReusedInPlan);
                    record_dimension_pruning_candidate(&mut diagnostics, target_candidate);
                    continue;
                }

                target_candidates.push((target_candidate, target_lineage));
            }
            if target_candidates.is_empty() {
                continue;
            }

            let Some(dimension_source) = local_parquet_source(&plan, dimension_lineage.source_node_id) else {
                for (mut candidate, _) in target_candidates {
                    candidate.rejected_reason =
                        Some(DimensionDerivedRowGroupPruningRejectedReason::DimensionKeyLineageUnavailable);
                    record_dimension_pruning_candidate(&mut diagnostics, candidate);
                }
                continue;
            };
            if dimension_source.reader_predicate.is_none() {
                for (mut candidate, _) in target_candidates {
                    candidate.rejected_reason =
                        Some(DimensionDerivedRowGroupPruningRejectedReason::DimensionPredicateAbsent);
                    record_dimension_pruning_candidate(&mut diagnostics, candidate);
                }
                continue;
            }
            if !resolver.dimension_scan_admitted(dimension_source, options.max_dimension_rows) {
                for (mut candidate, _) in target_candidates {
                    candidate.rejected_reason =
                        Some(DimensionDerivedRowGroupPruningRejectedReason::DimensionScanNotAdmitted);
                    record_dimension_pruning_candidate(&mut diagnostics, candidate);
                }
                continue;
            }

            let Some(key_filter) =
                derive_dimension_key_filter(dimension_source, &dimension_lineage.source_column, resolver)
            else {
                for (mut candidate, _) in target_candidates {
                    candidate.rejected_reason =
                        Some(DimensionDerivedRowGroupPruningRejectedReason::UnsupportedDimensionPredicate);
                    record_dimension_pruning_candidate(&mut diagnostics, candidate);
                }
                continue;
            };
            let Some(key_filter) = key_filter else {
                for (mut candidate, _) in target_candidates {
                    candidate.rejected_reason =
                        Some(DimensionDerivedRowGroupPruningRejectedReason::EmptyDimensionKeyRange);
                    record_dimension_pruning_candidate(&mut diagnostics, candidate);
                }
                continue;
            };
            let key_bounds = key_filter.bounds();

            for (mut candidate, target_lineage) in target_candidates {
                candidate.derived_min_key = Some(key_bounds.min);
                candidate.derived_max_key = Some(key_bounds.max);

                let Some(target_source) = mutable_local_parquet_source(&mut nodes, target_lineage.source_node_id)
                else {
                    candidate.rejected_reason =
                        Some(DimensionDerivedRowGroupPruningRejectedReason::TargetKeyLineageUnavailable);
                    record_dimension_pruning_candidate(&mut diagnostics, candidate);
                    continue;
                };
                let Some(prune_result) =
                    prune_target_source(target_source, &target_lineage.source_column, &key_filter, resolver)
                else {
                    candidate.rejected_reason =
                        Some(DimensionDerivedRowGroupPruningRejectedReason::TargetStatisticsUnavailable);
                    record_dimension_pruning_candidate(&mut diagnostics, candidate);
                    continue;
                };
                candidate.selected_row_groups_before = Some(prune_result.before);
                candidate.selected_row_groups_after = Some(prune_result.after);

                if !prune_result.applied {
                    candidate.rejected_reason =
                        Some(DimensionDerivedRowGroupPruningRejectedReason::NoRowGroupReduction);
                    record_dimension_pruning_candidate(&mut diagnostics, candidate);
                    continue;
                }

                candidate.applied = true;
                diagnostics.applied_count += 1;
                changed = true;
                record_dimension_pruning_candidate(&mut diagnostics, candidate);
            }
        }
    }

    if seeded_reader_predicates && diagnostics.applied_count == 0 {
        return Ok(DimensionDerivedRowGroupPruningResult {
            plan: original_plan,
            change: super::NativeOptimizerChange::Unchanged,
            diagnostics,
        });
    }

    let change = if changed { super::NativeOptimizerChange::Changed } else { super::NativeOptimizerChange::Unchanged };
    let root = plan.root_id();
    Ok(DimensionDerivedRowGroupPruningResult {
        plan: plan.rewrite(RewrittenNodes::refined_sources(nodes, root))?,
        change,
        diagnostics,
    })
}

/// Plan-shape candidate for dimension-derived fact row-group pruning.
///
/// Local-path sources are discovered and applied by
/// `optimize_dimension_derived_row_group_pruning`. Object-store sources are
/// discovered here and resolved by the DataFusion adapter, keeping this crate
/// free of object-store dependencies.
///
/// A request is only eligibility evidence. The resolver must still derive a
/// non-empty key range, find target stats, and prove a real row-group reduction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DimensionDerivedPruningRequest {
    /// Source node whose reader predicate, once evaluated, yields the join-key range.
    pub(crate) dimension_source_node_id: NodeId,
    /// Dimension source column that carries the join key.
    pub dimension_key_column: String,
    /// Fact source node whose row groups can be pruned by the derived range.
    pub(crate) target_source_node_id: NodeId,
    /// Fact source column that carries the join key.
    pub target_key_column: String,
}

/// Discovers dimension-derived pruning requests without reading Parquet data.
///
/// Mirrors the candidate-discovery half of
/// `optimize_dimension_derived_row_group_pruning`. The caller supplies
/// admission checks, key-range derivation, and footer statistics.
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
            let Some(dimension) = unique_simple_lineage(
                plan,
                &parent_counts,
                orientation.dimension_input,
                orientation.dimension_key.name(),
            ) else {
                continue;
            };
            let targets = dedup_lineages(simple_lineages_for_column(
                plan,
                &parent_counts,
                orientation.target_input,
                orientation.target_key.name(),
            ));
            if targets.is_empty() {
                continue;
            }
            let Some(dimension_source) = local_parquet_source(plan, dimension.source_node_id) else {
                continue;
            };
            if dimension_source.reader_predicate.is_none() {
                continue;
            }
            for target in targets {
                if dimension.source_node_id == target.source_node_id || target.has_shared_path {
                    continue;
                }
                requests.push(DimensionDerivedPruningRequest {
                    dimension_source_node_id: dimension.source_node_id,
                    dimension_key_column: dimension.source_column.clone(),
                    target_source_node_id: target.source_node_id,
                    target_key_column: target.source_column,
                });
            }
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
    resolver: &dyn LocalParquetPruningResolver,
) -> Option<TargetPruneResult> {
    let SourceColumnRef::File(key_file_index) = source.source_column_ref(key_column)? else {
        return None;
    };
    let per_file_row_group_key_ranges = resolver.target_row_group_key_ranges(source, key_file_index)?;
    if per_file_row_group_key_ranges.len() != source.files.len() {
        return None;
    }

    let mut before = 0usize;
    let mut after = 0usize;
    let mut saw_statistics = false;
    let mut new_row_groups_by_file = Vec::with_capacity(source.files.len());

    // Reuses the same pure per-row-group selection `apply_dimension_derived_pruning_with_stats`
    // applies to resolver-supplied stats, so the local and resolver-supplied
    // (e.g. object-store) pruning paths never diverge on selection logic.
    for (file, row_group_key_ranges) in source.files.iter().zip(&per_file_row_group_key_ranges) {
        let candidates = file.selected_or_all_row_groups(row_group_key_ranges.len());
        if candidates.len() != row_group_key_ranges.len() {
            return None;
        }
        let selection = select_overlapping_row_groups(row_group_key_ranges, key_filter);
        before += selection.before;
        after += selection.kept.len();
        saw_statistics |= selection.saw_statistics;
        new_row_groups_by_file.push(selection.kept.iter().map(|position| candidates[*position]).collect::<Vec<_>>());
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
    /// Exact row counts for every physical row group, in footer order.
    ///
    /// Keeping the full footer order lets the engine validate selected ids
    /// before carrying the source-owned cardinality into the rewritten plan.
    pub row_group_rows: Vec<u64>,
}

/// Outcome of [`apply_local_parquet_row_group_selections`].
#[derive(Debug, Clone)]
pub(crate) struct LocalParquetRowGroupSelectionApplyResult {
    /// Plan with the source row-group selection narrowed when pruning applied.
    pub(crate) plan: QueryPlan,
    /// Whether pruning strictly narrowed the source.
    pub(crate) applied: bool,
    /// Total candidate row groups before pruning.
    pub(crate) before: usize,
    /// Retained row groups after pruning.
    pub(crate) after: usize,
}

/// Applies externally-derived row-group selections to a `LocalParquetSource`.
///
/// The supplied selections are physical Parquet row-group ids and full footer
/// row counts, one entry per source file. Existing file-level selections are
/// narrowed, never widened. An empty result is treated as non-applied for now
/// because native empty-source representation is not modeled by this patching
/// API.
pub(crate) fn apply_local_parquet_row_group_selections(
    plan: QueryPlan,
    source_node_id: NodeId,
    selections: &[LocalParquetRowGroupSelection],
) -> Result<LocalParquetRowGroupSelectionApplyResult> {
    let mut nodes = plan.nodes().to_vec();
    let root = plan.root_id();
    let unchanged = |nodes: Vec<QueryNode>| -> Result<LocalParquetRowGroupSelectionApplyResult> {
        Ok(LocalParquetRowGroupSelectionApplyResult {
            plan: plan.rewrite(RewrittenNodes::refined_sources(nodes, root))?,
            applied: false,
            before: 0,
            after: 0,
        })
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
    let mut row_group_rows_by_file = Vec::with_capacity(source.files.len());
    for (file, selection) in source.files.iter().zip(selections) {
        if selection.total_row_groups == 0 || selection.row_group_rows.len() != selection.total_row_groups {
            return unchanged(nodes);
        }
        let candidates = file.selected_or_all_row_groups(selection.total_row_groups);
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
        row_group_rows_by_file.push(selection.row_group_rows.clone());
    }

    if after == 0 || after >= before {
        return unchanged(nodes);
    }

    for ((file, row_groups), row_group_rows) in
        source.files.iter_mut().zip(new_row_groups_by_file).zip(row_group_rows_by_file)
    {
        file.row_groups = Some(row_groups);
        file.row_group_rows = Some(row_group_rows);
    }
    source.refresh_diagnostics();
    Ok(LocalParquetRowGroupSelectionApplyResult {
        plan: plan.rewrite(RewrittenNodes::refined_sources(nodes, root))?,
        applied: true,
        before,
        after,
    })
}

/// Applies an externally-derived dimension key filter to a fact source.
///
/// The caller supplies one `Vec<Option<KeyRange>>` per source file, aligned with
/// the file's current row-group selection. Kept positions are mapped back
/// through that selection, so existing `row_groups` are narrowed, not replaced.
///
/// The rule matches `prune_target_source`: unknown stats are retained, pruning
/// never empties a selection, and mismatches return the original plan with
/// `applied = false`.
pub fn apply_dimension_derived_pruning_with_stats(
    plan: QueryPlan,
    target_source_node_id: NodeId,
    per_file_row_group_key_ranges: &[Vec<Option<KeyRange>>],
    filter: &DimensionKeyFilter,
) -> Result<DimensionDerivedApplyResult> {
    let mut nodes = plan.nodes().to_vec();
    let root = plan.root_id();
    let unchanged = |nodes: Vec<QueryNode>| -> Result<DimensionDerivedApplyResult> {
        Ok(DimensionDerivedApplyResult {
            plan: plan.rewrite(RewrittenNodes::refined_sources(nodes, root))?,
            applied: false,
            before: 0,
            after: 0,
        })
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
        // (e.g. a frontend file-level selection) is narrowed, never replaced
        // by raw positions. If the supplied stats do not align with the file's
        // selection, skip entirely rather than risk an unsound remap.
        let candidates = file.selected_or_all_row_groups(row_group_key_ranges.len());
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
    Ok(DimensionDerivedApplyResult {
        plan: plan.rewrite(RewrittenNodes::refined_sources(nodes, root))?,
        applied: true,
        before,
        after,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;
    use crate::{
        dynamic_filter::{
            DynamicFilterConsumer, DynamicFilterConsumerLowering, DynamicFilterId, DynamicFilterKind,
            DynamicFilterProducer, DynamicFilterRoute,
        },
        plan::QueryPlanBuilder,
        source::LocalParquetFile,
    };

    #[test]
    fn apply_local_parquet_row_group_selections_narrows_existing_selection() {
        let (plan, source_node_id) = test_plan_with_source_row_groups(Some(vec![0, 1, 2]));

        let result = apply_local_parquet_row_group_selections(
            plan,
            source_node_id,
            &[LocalParquetRowGroupSelection {
                total_row_groups: 5,
                row_groups: vec![1, 2],
                row_group_rows: vec![10, 20, 30, 40, 50],
            }],
        )
        .unwrap();

        assert!(result.applied);
        assert_eq!(result.before, 3);
        assert_eq!(result.after, 2);
        let source = local_parquet_source(&result.plan, source_node_id).unwrap();
        assert_eq!(source.files[0].row_groups, Some(vec![1, 2]));
        assert_eq!(source.files[0].row_group_rows, Some(vec![10, 20, 30, 40, 50]));
        assert_eq!(source.exact_selected_row_count(), Some(50));
        let stats = source
            .diagnostics
            .selected_row_group_stats
            .as_ref()
            .expect("row-count metadata should be reflected in diagnostics");
        assert_eq!(stats.selected_rows, Some(50));
        assert_eq!(stats.total_rows, Some(150));
    }

    #[test]
    fn apply_local_parquet_row_group_selections_rejects_widening_existing_selection() {
        let (plan, source_node_id) = test_plan_with_source_row_groups(Some(vec![0, 1, 2]));

        let result = apply_local_parquet_row_group_selections(
            plan,
            source_node_id,
            &[LocalParquetRowGroupSelection {
                total_row_groups: 5,
                row_groups: vec![1, 4],
                row_group_rows: vec![10, 20, 30, 40, 50],
            }],
        )
        .unwrap();

        assert!(!result.applied);
        let source = local_parquet_source(&result.plan, source_node_id).unwrap();
        assert_eq!(source.files[0].row_groups, Some(vec![0, 1, 2]));
    }

    #[test]
    fn apply_local_parquet_row_group_selections_keeps_explicitly_empty_source_empty() {
        // A dynamic filter marks a source proven empty as `Some(vec![])`.
        let (plan, source_node_id) = test_plan_with_source_row_groups(Some(Vec::new()));

        let result = apply_local_parquet_row_group_selections(
            plan,
            source_node_id,
            &[LocalParquetRowGroupSelection {
                total_row_groups: 5,
                row_groups: vec![1, 2],
                row_group_rows: vec![10, 20, 30, 40, 50],
            }],
        )
        .unwrap();

        assert!(!result.applied);
        let source = local_parquet_source(&result.plan, source_node_id).unwrap();
        assert_eq!(source.files[0].row_groups, Some(Vec::new()));
    }

    #[test]
    fn apply_local_parquet_row_group_selections_rejects_empty_selection() {
        let (plan, source_node_id) = test_plan_with_source_row_groups(None);

        let result = apply_local_parquet_row_group_selections(
            plan,
            source_node_id,
            &[LocalParquetRowGroupSelection {
                total_row_groups: 5,
                row_groups: Vec::new(),
                row_group_rows: vec![10, 20, 30, 40, 50],
            }],
        )
        .unwrap();

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
            &[LocalParquetRowGroupSelection {
                total_row_groups: 5,
                row_groups: vec![1, 1],
                row_group_rows: vec![10, 20, 30, 40, 50],
            }],
        )
        .unwrap();

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
            &[LocalParquetRowGroupSelection {
                total_row_groups: 5,
                row_groups: vec![2, 1],
                row_group_rows: vec![10, 20, 30, 40, 50],
            }],
        )
        .unwrap();

        assert!(!result.applied);
        let source = local_parquet_source(&result.plan, source_node_id).unwrap();
        assert_eq!(source.files[0].row_groups, Some(vec![0, 1, 2]));
    }

    #[test]
    fn apply_local_parquet_row_group_selections_rejects_misaligned_row_counts() {
        let (plan, source_node_id) = test_plan_with_source_row_groups(Some(vec![0, 1, 2]));

        let result = apply_local_parquet_row_group_selections(
            plan,
            source_node_id,
            &[LocalParquetRowGroupSelection {
                total_row_groups: 5,
                row_groups: vec![1, 2],
                row_group_rows: vec![10, 20],
            }],
        )
        .unwrap();

        assert!(!result.applied);
        let source = local_parquet_source(&result.plan, source_node_id).unwrap();
        assert_eq!(source.files[0].row_groups, Some(vec![0, 1, 2]));
        assert_eq!(source.files[0].row_group_rows, None);
    }

    #[test]
    fn apply_dimension_derived_pruning_preserves_dynamic_filter_routes_when_applied() {
        let (plan, source_node_id) = test_plan_with_source_row_groups(None);
        let route = test_dynamic_filter_route(source_node_id);
        let plan = plan.with_dynamic_filter_routes(vec![route.clone()]);

        let result = apply_dimension_derived_pruning_with_stats(
            plan,
            source_node_id,
            &[vec![
                Some(KeyRange { min: 0, max: 2 }),
                Some(KeyRange { min: 5, max: 5 }),
                Some(KeyRange { min: 10, max: 12 }),
            ]],
            &DimensionKeyFilter::from_bounds(KeyRange { min: 5, max: 5 }),
        )
        .unwrap();

        assert!(result.applied);
        assert_eq!(result.plan.dynamic_filter_routes(), &[route]);
        let source = local_parquet_source(&result.plan, source_node_id).unwrap();
        assert_eq!(source.files[0].row_groups, Some(vec![1]));
    }

    #[test]
    fn apply_dimension_derived_pruning_preserves_dynamic_filter_routes_when_unchanged() {
        let (plan, source_node_id) = test_plan_with_source_row_groups(Some(vec![0, 1, 2]));
        let route = test_dynamic_filter_route(source_node_id);
        let plan = plan.with_dynamic_filter_routes(vec![route.clone()]);

        let result = apply_dimension_derived_pruning_with_stats(
            plan,
            source_node_id,
            &[vec![Some(KeyRange { min: 0, max: 2 })]],
            &DimensionKeyFilter::from_bounds(KeyRange { min: 5, max: 5 }),
        )
        .unwrap();

        assert!(!result.applied);
        assert_eq!(result.plan.dynamic_filter_routes(), &[route]);
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
        (builder.finish_with_host_sink(source_node_id).unwrap(), source_node_id)
    }

    fn test_dynamic_filter_route(source_node_id: NodeId) -> DynamicFilterRoute {
        DynamicFilterRoute::new(
            DynamicFilterId::new(0),
            DynamicFilterProducer::AggregateGroupKeys { aggregate_node_id: source_node_id },
            DynamicFilterConsumer::LocalParquetSource { source_node_id },
            vec!["x".to_string()],
            vec!["x".to_string()],
            vec![DynamicFilterKind::Range],
            DynamicFilterConsumerLowering::ReaderPredicate,
        )
    }
}

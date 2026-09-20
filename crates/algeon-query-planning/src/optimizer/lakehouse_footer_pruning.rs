//! Engine-owned sequencing for adapter-resolved lakehouse footer pruning.
//!
//! The engine owns the native-plan candidate discovery, target limits, and
//! row-group selection application.  The adapter supplies only the
//! DataFusion/object-store operations needed to resolve Parquet metadata and
//! filtered dimension key ranges.

use super::{
    DimensionDerivedPruningRequest, DimensionKeyFilter, KeyRange, LocalParquetRowGroupSelection,
    apply_dimension_derived_pruning_with_stats, apply_local_parquet_row_group_selections,
    discover_dimension_derived_pruning_requests, local_parquet_source, reader_projection_indices,
    seed_dimension_derived_reader_predicates,
};
use crate::{
    backend::AnalysisContext,
    expr::Expr,
    plan::{NodeId, PlanNode, QueryPlan},
    source::{LocalParquetSource, ParquetSourceLocation, SourceColumnRef, SourceNode},
};

/// Target-file limit enforced before adapter footer reads begin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LakehouseFooterPruningOptions {
    /// Maximum remote target files considered by either footer-pruning phase.
    pub(crate) max_target_files: usize,
}

impl LakehouseFooterPruningOptions {
    /// Creates lakehouse footer-pruning options with one shared target-file cap.
    pub const fn new(max_target_files: usize) -> Self {
        Self { max_target_files }
    }
}

/// One observable result from engine-owned lakehouse footer-pruning sequencing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LakehouseFooterPruningEvent {
    /// Static reader-predicate footer pruning narrowed one or more row groups.
    StaticReaderPredicateApplied {
        /// Number of candidate row groups before pruning.
        before: usize,
        /// Number of retained row groups after pruning.
        after: usize,
    },
    /// A static or target-file-cap check skipped footer pruning.
    Skipped {
        /// Stable reason supplied by the engine or adapter resolver.
        reason: String,
    },
    /// Dimension-derived footer pruning narrowed a fact source.
    DimensionDerivedApplied {
        /// Fact source whose selection was narrowed.
        target_source_node_id: NodeId,
        /// Number of candidate row groups before pruning.
        before: usize,
        /// Number of retained row groups after pruning.
        after: usize,
    },
    /// One dimension-derived candidate could not be resolved safely.
    DimensionDerivedSkipped {
        /// Fact source considered for pruning.
        target_source_node_id: NodeId,
        /// Stable reason for retaining the original selection.
        reason: &'static str,
    },
    /// Adapter footer or dimension resolution failed for one candidate.
    DimensionDerivedRequestError {
        /// Adapter-provided diagnostic, to be redacted by the adapter logger.
        error: String,
    },
}

/// Completed lakehouse footer-pruning attempt.
#[derive(Debug, Clone)]
pub struct LakehouseFooterPruningResult {
    /// Original plan when no row-group selection narrowed; otherwise the patched plan.
    pub plan: QueryPlan,
    /// Ordered adapter-observable outcomes from the pruning attempt.
    pub events: Vec<LakehouseFooterPruningEvent>,
}

/// DataFusion/object-store operations required to resolve lakehouse pruning evidence.
///
/// Implementations may perform remote footer reads and filtered dimension scans,
/// but must not mutate the native plan.  The engine owns all plan inspection,
/// candidate ordering, capacity checks, and row-group selection application.
#[expect(async_fn_in_trait, reason = "internal trait consumed within engine runtime async contexts")]
pub trait LakehouseFooterPruningResolver {
    /// Resolves static reader-predicate selections for one remote source.
    ///
    /// `Ok(None)` means the predicate is not safely footer-prunable.
    async fn static_row_group_selections(
        &mut self,
        source_node_id: NodeId,
        source: &LocalParquetSource,
        predicate: &Expr,
    ) -> std::result::Result<Option<Vec<LocalParquetRowGroupSelection>>, String>;

    /// Resolves the filtered dimension key range for one candidate.
    ///
    /// `Ok(None)` means no safe, non-empty range could be derived.
    async fn dimension_key_filter(
        &mut self,
        source_node_id: NodeId,
        source: &LocalParquetSource,
        key_column: &str,
        predicate: &Expr,
        projection_indices: &[usize],
    ) -> std::result::Result<Option<DimensionKeyFilter>, String>;

    /// Resolves one key-range observation for each selected target row group.
    async fn target_row_group_key_ranges(
        &mut self,
        source_node_id: NodeId,
        source: &LocalParquetSource,
        key_column_index: usize,
    ) -> std::result::Result<Vec<Vec<Option<KeyRange>>>, String>;
}

/// Applies adapter-resolved lakehouse footer evidence through one engine entry point.
///
/// This function deliberately owns the order of static and dimension-derived
/// pruning.  The adapter remains responsible only for resolving remote data;
/// it cannot independently discover or patch native optimizer candidates.
pub async fn optimize_lakehouse_footer_pruning<R>(
    plan: QueryPlan,
    options: LakehouseFooterPruningOptions,
    resolver: &mut R,
    ctx: AnalysisContext<'_>,
) -> LakehouseFooterPruningResult
where
    R: LakehouseFooterPruningResolver + ?Sized,
{
    let original_plan = plan.clone();
    let mut plan = seed_dimension_derived_reader_predicates(plan, ctx);
    let mut any_applied = false;
    let mut events = Vec::new();

    let static_source_node_ids = static_reader_predicate_source_node_ids(&plan);
    let static_target_file_count: usize = static_source_node_ids
        .iter()
        .filter_map(|source_node_id| local_parquet_source(&plan, *source_node_id))
        .map(|source| source.files.len())
        .sum();
    if static_target_file_count > options.max_target_files {
        events.push(LakehouseFooterPruningEvent::Skipped {
            reason: format!(
                "too_many_static_predicate_files target_file_count={static_target_file_count} max_target_file_count={}",
                options.max_target_files
            ),
        });
        return LakehouseFooterPruningResult { plan: original_plan, events };
    }

    let mut static_plan = plan.clone();
    let mut static_before = 0usize;
    let mut static_after = 0usize;
    let mut static_applied = false;
    let mut static_error = None;
    for source_node_id in static_source_node_ids {
        let Some(source) = local_parquet_source(&static_plan, source_node_id).cloned() else {
            continue;
        };
        let Some(predicate) = source.reader_predicate.as_ref() else {
            continue;
        };
        match resolver.static_row_group_selections(source_node_id, &source, predicate).await {
            Ok(Some(selections)) => {
                let result = apply_local_parquet_row_group_selections(static_plan, source_node_id, &selections);
                if result.applied {
                    static_before = static_before.saturating_add(result.before);
                    static_after = static_after.saturating_add(result.after);
                    static_applied = true;
                }
                static_plan = result.plan;
            },
            Ok(None) => {},
            Err(reason) => {
                static_error = Some(reason);
                break;
            },
        }
    }
    if let Some(reason) = static_error {
        events.push(LakehouseFooterPruningEvent::Skipped { reason });
    } else if static_applied {
        events.push(LakehouseFooterPruningEvent::StaticReaderPredicateApplied {
            before: static_before,
            after: static_after,
        });
        plan = static_plan;
        any_applied = true;
    }

    let requests = discover_dimension_derived_pruning_requests(&plan);
    if requests.is_empty() && !any_applied {
        return LakehouseFooterPruningResult { plan: original_plan, events };
    }

    let target_file_count: usize = requests
        .iter()
        .filter_map(|request| local_parquet_source(&plan, request.target_source_node_id))
        .filter(|source| !source.files.iter().all(|file| file.location.as_local_path().is_some()))
        .map(|source| source.files.len())
        .sum();
    if target_file_count > options.max_target_files {
        events.push(LakehouseFooterPruningEvent::Skipped {
            reason: format!(
                "too_many_target_files target_file_count={target_file_count} max_target_file_count={}",
                options.max_target_files
            ),
        });
        return LakehouseFooterPruningResult { plan: if any_applied { plan } else { original_plan }, events };
    }

    for request in requests {
        let target_source_node_id = request.target_source_node_id;
        match resolve_dimension_derived_request(plan.clone(), resolver, &request).await {
            DimensionDerivedRequestOutcome::Applied { plan: patched, before, after } => {
                events.push(LakehouseFooterPruningEvent::DimensionDerivedApplied {
                    target_source_node_id,
                    before,
                    after,
                });
                plan = *patched;
                any_applied = true;
            },
            DimensionDerivedRequestOutcome::Skipped { reason } => {
                events.push(LakehouseFooterPruningEvent::DimensionDerivedSkipped { target_source_node_id, reason });
            },
            DimensionDerivedRequestOutcome::Error { error } => {
                events.push(LakehouseFooterPruningEvent::DimensionDerivedRequestError { error });
            },
        }
    }

    LakehouseFooterPruningResult { plan: if any_applied { plan } else { original_plan }, events }
}

fn static_reader_predicate_source_node_ids(plan: &QueryPlan) -> Vec<NodeId> {
    plan.nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Source(source) => match source.as_ref() {
                SourceNode::LocalParquet(source)
                    if source.reader_predicate.is_some()
                        && !source.files.is_empty()
                        && source
                            .files
                            .iter()
                            .all(|file| matches!(file.location, ParquetSourceLocation::ObjectStoreUri { .. })) =>
                {
                    Some(node.id)
                },
                SourceNode::LocalParquet(_) | SourceNode::GpuResident(_) | SourceNode::LiteralBatch(_) => None,
            },
            _ => None,
        })
        .collect()
}

enum DimensionDerivedRequestOutcome {
    Applied { plan: Box<QueryPlan>, before: usize, after: usize },
    Skipped { reason: &'static str },
    Error { error: String },
}

async fn resolve_dimension_derived_request<R>(
    plan: QueryPlan,
    resolver: &mut R,
    request: &DimensionDerivedPruningRequest,
) -> DimensionDerivedRequestOutcome
where
    R: LakehouseFooterPruningResolver + ?Sized,
{
    let target_source_node_id = request.target_source_node_id;
    let Some(target_source) = local_parquet_source(&plan, target_source_node_id).cloned() else {
        return DimensionDerivedRequestOutcome::Skipped { reason: "missing_target_source" };
    };
    if target_source.files.iter().all(|file| file.location.as_local_path().is_some()) {
        return DimensionDerivedRequestOutcome::Skipped { reason: "target_is_local_path" };
    }
    if !target_source.files.iter().all(|file| matches!(file.location, ParquetSourceLocation::ObjectStoreUri { .. })) {
        return DimensionDerivedRequestOutcome::Skipped { reason: "mixed_or_unsupported_target_locations" };
    }

    let Some(dimension_source) = local_parquet_source(&plan, request.dimension_source_node_id).cloned() else {
        return DimensionDerivedRequestOutcome::Skipped { reason: "missing_dimension_source" };
    };
    let Some(predicate) = dimension_source.reader_predicate.as_ref() else {
        return DimensionDerivedRequestOutcome::Skipped { reason: "missing_dimension_predicate" };
    };

    let Some(projection_indices) =
        reader_projection_indices(&dimension_source, request.dimension_key_column.as_str(), predicate)
    else {
        return DimensionDerivedRequestOutcome::Error { error: "unsupported_dimension_projection".to_string() };
    };

    let projection_indices = projection_indices.into_iter().collect::<Vec<_>>();
    let key_filter = match resolver
        .dimension_key_filter(
            request.dimension_source_node_id,
            &dimension_source,
            request.dimension_key_column.as_str(),
            predicate,
            &projection_indices,
        )
        .await
    {
        Ok(Some(key_filter)) => key_filter,
        Ok(None) => {
            return DimensionDerivedRequestOutcome::Skipped { reason: "empty_or_unsupported_dimension_range" };
        },
        Err(error) => return DimensionDerivedRequestOutcome::Error { error },
    };

    let target_key_column_index = match target_source.source_column_ref(request.target_key_column.as_str()) {
        Some(SourceColumnRef::File(index)) => index,
        Some(_) => return DimensionDerivedRequestOutcome::Skipped { reason: "target_key_not_file_column" },
        None => return DimensionDerivedRequestOutcome::Error { error: "missing_target_key_column".to_string() },
    };

    let per_file_row_group_key_ranges = match resolver
        .target_row_group_key_ranges(target_source_node_id, &target_source, target_key_column_index)
        .await
    {
        Ok(ranges) => ranges,
        Err(error) => return DimensionDerivedRequestOutcome::Error { error },
    };

    let result = apply_dimension_derived_pruning_with_stats(
        plan,
        target_source_node_id,
        &per_file_row_group_key_ranges,
        &key_filter,
    );
    if result.applied {
        DimensionDerivedRequestOutcome::Applied {
            plan: Box::new(result.plan),
            before: result.before,
            after: result.after,
        }
    } else {
        DimensionDerivedRequestOutcome::Skipped { reason: "no_fact_row_group_reduction" }
    }
}

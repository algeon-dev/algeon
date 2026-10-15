//! Native query-plan optimizer surface.
//!
//! Optimizer rules in this module operate on [`QueryPlan`] IR only. They do
//! not inspect SQL text, DataFusion plans, benchmark query ids, or adapter
//! report rows.
//!
//! Every rule receives a plan that passed [`QueryPlan`] construction and
//! returns one through `QueryPlan::rewrite`. Consequently, rules can safely assume
//! that node IDs match positions, inputs precede consumers, and output schemas
//! are fully derived. Rules cannot rely on the execution of prior rules because
//! optimization steps may be independently disabled. When a rule identifies no
//! rewrite opportunities, it returns the input unmodified alongside `Unchanged`.
//! Rule diagnostics merge into their respective entries within
//! [`NativeOptimizerDiagnostics`] on each invocation, accurately recording multiple
//! executions. The design contract is documented in
//! `website/docs/design/native-lowering.mdx`.

mod semijoin_reduction;

mod access;
mod aggregate_join_pushdown;
mod aggregate_join_pushdown_policy;
mod count_non_null_pruning;
mod diagnostics;
mod dimension_pruning;
mod dynamic_filter;
mod join_build_side;
mod join_filter_reassociation;
mod lakehouse_footer_pruning;
mod lineage;
mod local_parquet_source_reuse;
mod outer_join_null_rejection;
mod outer_join_null_rejection_policy;
mod projection_simplification;
mod rewrite;
mod shared_resident_extraction;
mod shared_subplan_materialization;
mod single_distinct_aggregate_split;
mod source_target_policy;
mod stream_fanout_replay;

use std::collections::BTreeMap;

pub use access::local_parquet_source;
pub(crate) use access::reader_projection_indices;
use access::{
    consumers_by_node, simple_projection_column, simple_projection_column_single_alias, unique_join_output_for_name,
};
pub use aggregate_join_pushdown::{
    AGGREGATE_JOIN_PUSHDOWN_RULE_NAME, AggregateJoinPushdownCandidateDiagnostics, AggregateJoinPushdownDiagnostics,
    AggregateJoinPushdownOptions, AggregateJoinPushdownRejectedReason,
};
pub use count_non_null_pruning::{
    COUNT_NON_NULL_COLUMN_PRUNING_RULE_NAME, CountNonNullColumnPruningCandidateDiagnostics,
    CountNonNullColumnPruningDiagnostics, CountNonNullColumnPruningOptions, CountNonNullColumnPruningRejectedReason,
};
pub(crate) use dimension_pruning::apply_local_parquet_row_group_selections;
pub use dimension_pruning::{
    DIMENSION_DERIVED_ROW_GROUP_PRUNING_RULE_NAME, DIMENSION_KEY_FILTER_MAX_RUNS, DimensionDerivedApplyResult,
    DimensionDerivedPruningRequest, DimensionDerivedRowGroupPruningCandidateDiagnostics,
    DimensionDerivedRowGroupPruningDiagnostics, DimensionDerivedRowGroupPruningOptions,
    DimensionDerivedRowGroupPruningRejectedReason, DimensionKeyAccumulator, DimensionKeyFilter,
    I64ExprRangeAccumulator, KeyRange, LocalParquetPruningResolver, LocalParquetRowGroupSelection,
    apply_dimension_derived_pruning_with_stats, dimension_key_filter_from_batch,
    discover_dimension_derived_pruning_requests, seed_dimension_derived_reader_predicates,
};
pub use dynamic_filter::{
    DYNAMIC_FILTER_ROUTE_DISCOVERY_RULE_NAME, DynamicFilterRouteCandidateDiagnostics,
    DynamicFilterRouteDiscoveryDiagnostics, DynamicFilterRouteDiscoveryOptions, DynamicFilterRouteRejectedReason,
};
pub use join_build_side::{
    JOIN_BUILD_SIDE_SELECTION_RULE_NAME, JoinBuildSideSelectionBasis, JoinBuildSideSelectionCandidateDiagnostics,
    JoinBuildSideSelectionDiagnostics, JoinBuildSideSelectionOptions, JoinBuildSideSelectionRejectedReason,
};
pub use join_filter_reassociation::{
    JOIN_FILTER_REASSOCIATION_RULE_NAME, JoinFilterReassociationCandidateDiagnostics,
    JoinFilterReassociationDiagnostics, JoinFilterReassociationOptions, JoinFilterReassociationRejectedReason,
};
pub use lakehouse_footer_pruning::{
    LakehouseFooterPruningEvent, LakehouseFooterPruningOptions, LakehouseFooterPruningResolver,
    LakehouseFooterPruningResult, optimize_lakehouse_footer_pruning,
};
pub use local_parquet_source_reuse::{
    LOCAL_PARQUET_SOURCE_REUSE_RULE_NAME, LocalParquetSourceReuseCandidateDiagnostics,
    LocalParquetSourceReuseDiagnostics, LocalParquetSourceReuseOptions, LocalParquetSourceReuseRejectedReason,
};
pub use outer_join_null_rejection::{
    OUTER_JOIN_NULL_REJECTION_RULE_NAME, OuterJoinNullRejectionCandidateDiagnostics, OuterJoinNullRejectionDiagnostics,
    OuterJoinNullRejectionOptions, OuterJoinNullRejectionRejectedReason,
};
pub use projection_simplification::{
    PROJECTION_SIMPLIFICATION_RULE_NAME, ProjectionSimplificationCandidateDiagnostics,
    ProjectionSimplificationDiagnostics, ProjectionSimplificationOptions, ProjectionSimplificationRejectedReason,
};
pub(crate) use rewrite::parent_counts;
pub use semijoin_reduction::{
    DEFAULT_SEMIJOIN_REDUCTION_COST_THRESHOLD_MULTIPLIER, SEMIJOIN_REDUCTION_RULE_NAME,
    SemijoinReductionCandidateDiagnostics, SemijoinReductionCostGateDiagnostics, SemijoinReductionCostSignal,
    SemijoinReductionDiagnostics, SemijoinReductionKeyMappingDiagnostics, SemijoinReductionOptions,
    SemijoinReductionRejectedReason,
};
#[cfg(any(test, feature = "test-hooks"))]
#[doc(hidden)]
pub use shared_resident_extraction::SharedResidentPlan;
#[doc(hidden)]
pub use shared_resident_extraction::SharedResidentProducerPlan;
pub use shared_resident_extraction::{
    SHARED_RESIDENT_EXTRACTION_RULE_NAME, SharedResidentExtractionCandidateDiagnostics,
    SharedResidentExtractionDecision, SharedResidentExtractionDiagnostics, SharedResidentExtractionReason,
};
#[doc(hidden)]
pub use shared_resident_extraction::{SharedResidentFragments, extract_shared_resident_fragments};
pub use shared_subplan_materialization::{
    SHARED_SUBPLAN_MATERIALIZATION_RULE_NAME, SharedSubplanMaterializationCandidateDiagnostics,
    SharedSubplanMaterializationDiagnostics, SharedSubplanMaterializationOptions,
    SharedSubplanMaterializationRejectedReason,
};
pub use single_distinct_aggregate_split::{
    SINGLE_DISTINCT_AGGREGATE_SPLIT_RULE_NAME, SingleDistinctAggregateSplitCandidateDiagnostics,
    SingleDistinctAggregateSplitDiagnostics, SingleDistinctAggregateSplitOptions,
    SingleDistinctAggregateSplitRejectedReason,
};
pub use stream_fanout_replay::{
    STREAM_FANOUT_REPLAY_RULE_NAME, StreamFanoutReplayCandidateDiagnostics, StreamFanoutReplayDiagnostics,
    StreamFanoutReplayOptions,
};

use crate::{backend::AnalysisContext, error::Result, plan::QueryPlan};

/// Native optimizer pass configuration.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NativeOptimizerOptions {
    /// Options for reducing rollup/grouping-set aggregates before dimension joins.
    pub aggregate_join_pushdown: AggregateJoinPushdownOptions,
    /// Options for rewriting footer-proven non-null `COUNT(column)` arguments.
    pub count_non_null_column_pruning: CountNonNullColumnPruningOptions,
    /// Options for splitting exact distinct counts that share one argument into
    /// two grouped aggregates.
    pub single_distinct_aggregate_split: SingleDistinctAggregateSplitOptions,
    /// Options for removing redundant identity projections.
    pub projection_simplification: ProjectionSimplificationOptions,
    /// Options for pushing left-semi key filters below eligible joins.
    pub semijoin_reduction: SemijoinReductionOptions,
    /// Options for rewriting null-rejected left outer joins to inner joins.
    pub outer_join_null_rejection: OuterJoinNullRejectionOptions,
    /// Options for moving join filters across preserved-side join structure.
    pub join_filter_reassociation: JoinFilterReassociationOptions,
    /// Options for pruning fact row groups from filtered dimension key ranges.
    pub dimension_derived_row_group_pruning: DimensionDerivedRowGroupPruningOptions,
    /// Options for choosing smaller hash-join build sides.
    pub join_build_side_selection: JoinBuildSideSelectionOptions,
    /// Options for replaying shared stream probe-input subtrees.
    pub stream_fanout_replay: StreamFanoutReplayOptions,
    /// Options for sharing compatible repeated local Parquet source variants.
    pub local_parquet_source_reuse: LocalParquetSourceReuseOptions,
    /// Options for discovering repeated native subplans that should be materialized once.
    pub shared_subplan_materialization: SharedSubplanMaterializationOptions,
    /// Options for discovering native dynamic-filter producer/consumer routes.
    pub dynamic_filter_route_discovery: DynamicFilterRouteDiscoveryOptions,
}

impl NativeOptimizerOptions {
    /// Returns options with all optimizer passes disabled.
    pub fn disabled() -> Self {
        Self {
            aggregate_join_pushdown: AggregateJoinPushdownOptions::disabled(),
            count_non_null_column_pruning: CountNonNullColumnPruningOptions::disabled(),
            single_distinct_aggregate_split: SingleDistinctAggregateSplitOptions::disabled(),
            projection_simplification: ProjectionSimplificationOptions::disabled(),
            semijoin_reduction: SemijoinReductionOptions::disabled(),
            outer_join_null_rejection: OuterJoinNullRejectionOptions::disabled(),
            join_filter_reassociation: JoinFilterReassociationOptions::disabled(),
            dimension_derived_row_group_pruning: DimensionDerivedRowGroupPruningOptions::disabled(),
            join_build_side_selection: JoinBuildSideSelectionOptions::disabled(),
            stream_fanout_replay: StreamFanoutReplayOptions::disabled(),
            local_parquet_source_reuse: LocalParquetSourceReuseOptions::disabled(),
            shared_subplan_materialization: SharedSubplanMaterializationOptions::disabled(),
            dynamic_filter_route_discovery: DynamicFilterRouteDiscoveryOptions::disabled(),
        }
    }

    /// Validates production-facing optimizer option consistency.
    ///
    /// `optimize_query_plan` stays non-fallible for expert callers. Production
    /// adapter/service boundaries should call this before an option set is used
    /// to capture policy into a native plan or report.
    pub fn validate(&self) -> Result<()> {
        self.semijoin_reduction.validate()?;
        self.dimension_derived_row_group_pruning.validate()?;
        self.join_build_side_selection.validate()?;
        self.local_parquet_source_reuse.validate()?;
        Ok(())
    }
}

/// Whether an optimizer invocation changed the native IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeOptimizerChange {
    Unchanged,
    Changed,
}

impl NativeOptimizerChange {
    pub(crate) fn changed(self) -> bool {
        matches!(self, Self::Changed)
    }
}

/// Result returned by the native optimizer entry point.
#[derive(Debug, Clone, PartialEq)]
pub struct NativeOptimizerResult {
    /// Optimized native query plan.
    pub plan: QueryPlan,
    /// Closed repeated fragments extracted behind resident sources.
    #[doc(hidden)]
    pub shared_resident_producers: Vec<SharedResidentProducerPlan>,
    /// Whether any pass changed the plan, or shared-resident extraction produced
    /// at least one producer.
    pub change: NativeOptimizerChange,
    /// Rule diagnostics collected while optimizing.
    pub diagnostics: NativeOptimizerDiagnostics,
}

impl NativeOptimizerResult {
    pub fn changed(&self) -> bool {
        self.change.changed()
    }
}

/// Diagnostics emitted by native optimizer rules.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NativeOptimizerDiagnostics {
    /// Diagnostics from the aggregate-join pushdown pass.
    pub aggregate_join_pushdown: AggregateJoinPushdownDiagnostics,
    /// Diagnostics from the footer-proven count argument pruning pass.
    pub count_non_null_column_pruning: CountNonNullColumnPruningDiagnostics,
    /// Diagnostics from the single-distinct aggregate split pass.
    pub single_distinct_aggregate_split: SingleDistinctAggregateSplitDiagnostics,
    /// Diagnostics from the projection-simplification pass.
    pub projection_simplification: ProjectionSimplificationDiagnostics,
    /// Diagnostics from the semijoin-reduction pass.
    pub semijoin_reduction: SemijoinReductionDiagnostics,
    /// Diagnostics from the outer-join null-rejection pass.
    pub outer_join_null_rejection: OuterJoinNullRejectionDiagnostics,
    /// Diagnostics from the join-filter reassociation pass.
    pub join_filter_reassociation: JoinFilterReassociationDiagnostics,
    /// Diagnostics from the dimension-derived row-group pruning pass.
    pub dimension_derived_row_group_pruning: DimensionDerivedRowGroupPruningDiagnostics,
    /// Diagnostics from the join build-side selection pass.
    pub join_build_side_selection: JoinBuildSideSelectionDiagnostics,
    /// Diagnostics from the stream fanout replay pass.
    pub stream_fanout_replay: StreamFanoutReplayDiagnostics,
    /// Diagnostics from the local Parquet source reuse pass.
    pub local_parquet_source_reuse: LocalParquetSourceReuseDiagnostics,
    /// Diagnostics from the shared-subplan materialization and scalar derivation passes.
    pub shared_subplan_materialization: SharedSubplanMaterializationDiagnostics,
    /// Diagnostics from the shared-resident extraction stage.
    pub shared_resident_extraction: SharedResidentExtractionDiagnostics,
    /// Diagnostics from the dynamic-filter route discovery pass.
    pub dynamic_filter_route_discovery: DynamicFilterRouteDiscoveryDiagnostics,
}

impl NativeOptimizerDiagnostics {
    /// Returns true when no optimizer pass considered or applied candidates.
    pub fn is_empty(&self) -> bool {
        self.aggregate_join_pushdown.is_empty()
            && self.count_non_null_column_pruning.is_empty()
            && self.single_distinct_aggregate_split.is_empty()
            && self.projection_simplification.is_empty()
            && self.semijoin_reduction.is_empty()
            && self.outer_join_null_rejection.is_empty()
            && self.join_filter_reassociation.is_empty()
            && self.dimension_derived_row_group_pruning.is_empty()
            && self.join_build_side_selection.is_empty()
            && self.stream_fanout_replay.is_empty()
            && self.local_parquet_source_reuse.is_empty()
            && self.shared_subplan_materialization.is_empty()
            && self.shared_resident_extraction.is_empty()
            && self.dynamic_filter_route_discovery.is_empty()
    }

    /// Merges diagnostics from another optimizer invocation into this one.
    pub fn merge(&mut self, other: Self) {
        self.aggregate_join_pushdown.merge(other.aggregate_join_pushdown);
        self.count_non_null_column_pruning.merge(other.count_non_null_column_pruning);
        self.single_distinct_aggregate_split.merge(other.single_distinct_aggregate_split);
        self.projection_simplification.merge(other.projection_simplification);
        self.semijoin_reduction.merge(other.semijoin_reduction);
        self.outer_join_null_rejection.merge(other.outer_join_null_rejection);
        self.join_filter_reassociation.merge(other.join_filter_reassociation);
        self.dimension_derived_row_group_pruning.merge(other.dimension_derived_row_group_pruning);
        self.join_build_side_selection.merge(other.join_build_side_selection);
        self.stream_fanout_replay.merge(other.stream_fanout_replay);
        self.local_parquet_source_reuse.merge(other.local_parquet_source_reuse);
        self.shared_subplan_materialization.merge(other.shared_subplan_materialization);
        self.shared_resident_extraction.merge(other.shared_resident_extraction);
        self.dynamic_filter_route_discovery.merge(other.dynamic_filter_route_discovery);
    }
}

/// One rule invocation in `STEPS`. The same pass may appear several times;
/// `run_pass` maps it to the rule function and the diagnostics slot it merges
/// into.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Pass {
    ProjectionSimplification,
    SemijoinReduction,
    JoinFilterReassociation,
    JoinBuildSideSelection,
    OuterJoinNullRejection,
    ScalarAggregateDerivation,
    AggregateJoinPushdown,
    DimensionDerivedRowGroupPruning,
    SingleDistinctAggregateSplit,
    CountNonNullColumnPruning,
    StreamFanoutReplay,
    SharedSubplanMaterialization,
    LocalParquetSourceReuse,
    DynamicFilterRouteDiscovery,
}

/// When a step of the pass sequence runs.
#[derive(Debug, Clone, Copy)]
enum When {
    Always,
    /// Only when the latest run of this pass changed the plan.
    AfterChangeBy(Pass),
}

/// The pass sequence. Aggregate pushdown and dimension pruning can expose new
/// build-side and null-rejection opportunities, so those passes run again
/// after either changes the plan. The single-distinct split runs before count
/// pruning, which can turn a global `COUNT(column)` into a literal count-star
/// argument that a grouped partial aggregate cannot read. Shared-resident
/// extraction runs after the table: it splits the plan into producers and a
/// consumer.
const STEPS: [(Pass, When); 18] = [
    (Pass::ProjectionSimplification, When::Always),
    (Pass::SemijoinReduction, When::Always),
    (Pass::JoinFilterReassociation, When::Always),
    (Pass::JoinBuildSideSelection, When::Always),
    (Pass::OuterJoinNullRejection, When::Always),
    (Pass::ScalarAggregateDerivation, When::Always),
    (Pass::AggregateJoinPushdown, When::Always),
    (Pass::JoinBuildSideSelection, When::AfterChangeBy(Pass::AggregateJoinPushdown)),
    (Pass::OuterJoinNullRejection, When::AfterChangeBy(Pass::AggregateJoinPushdown)),
    (Pass::DimensionDerivedRowGroupPruning, When::Always),
    (Pass::JoinBuildSideSelection, When::AfterChangeBy(Pass::DimensionDerivedRowGroupPruning)),
    (Pass::SingleDistinctAggregateSplit, When::Always),
    (Pass::CountNonNullColumnPruning, When::Always),
    (Pass::StreamFanoutReplay, When::Always),
    (Pass::SharedSubplanMaterialization, When::Always),
    (Pass::LocalParquetSourceReuse, When::Always),
    (Pass::ProjectionSimplification, When::Always),
    (Pass::DynamicFilterRouteDiscovery, When::Always),
];

/// Runs the native optimizer pass sequence over a query plan.
///
/// The plan is consumed; a caller that needs the unoptimized plan clones it
/// first. An `Err` means a rule built a node list that fails the plan
/// invariants or `resolver` failed, and no partially optimized plan is
/// returned. `options` are not validated here; see
/// [`NativeOptimizerOptions::validate`]. Shared-resident extraction runs after
/// the pass table and may split the plan into producers and a consumer, which
/// counts as a change even when no pass changed anything.
pub fn optimize_query_plan(
    plan: QueryPlan,
    options: &NativeOptimizerOptions,
    ctx: AnalysisContext<'_>,
    resolver: &dyn LocalParquetPruningResolver,
) -> Result<NativeOptimizerResult> {
    let mut plan = plan;
    let mut diagnostics = NativeOptimizerDiagnostics::default();
    let mut changed = false;
    let mut latest_change = BTreeMap::new();
    for (pass, when) in STEPS {
        if let When::AfterChangeBy(trigger) = when
            && latest_change.get(&trigger) != Some(&true)
        {
            continue;
        }
        let (next, change) = run_pass(pass, plan, options, ctx, resolver, &mut diagnostics)?;
        plan = next;
        changed |= change.changed();
        latest_change.insert(pass, change.changed());
    }
    let shared_resident = shared_resident_extraction::optimize_shared_resident_extraction(plan)?;
    changed |= !shared_resident.producers.is_empty();
    diagnostics.shared_resident_extraction = shared_resident.diagnostics;
    Ok(NativeOptimizerResult {
        change: if changed { NativeOptimizerChange::Changed } else { NativeOptimizerChange::Unchanged },
        plan: shared_resident.consumer,
        shared_resident_producers: shared_resident.producers,
        diagnostics,
    })
}

/// Runs one pass and merges its diagnostics into the pass's report.
fn run_pass(
    pass: Pass,
    plan: QueryPlan,
    options: &NativeOptimizerOptions,
    ctx: AnalysisContext<'_>,
    resolver: &dyn LocalParquetPruningResolver,
    diagnostics: &mut NativeOptimizerDiagnostics,
) -> Result<(QueryPlan, NativeOptimizerChange)> {
    Ok(match pass {
        Pass::ProjectionSimplification => {
            let result = projection_simplification::optimize_projection_simplification(
                plan,
                &options.projection_simplification,
            )?;
            diagnostics.projection_simplification.merge(result.diagnostics);
            (result.plan, result.change)
        },
        Pass::SemijoinReduction => {
            let result = semijoin_reduction::optimize_semijoin_reduction(plan, &options.semijoin_reduction)?;
            diagnostics.semijoin_reduction.merge(result.diagnostics);
            (result.plan, result.change)
        },
        Pass::JoinFilterReassociation => {
            let result =
                optimize_join_filter_reassociation_fixpoint(plan, &options.join_filter_reassociation, resolver)?;
            diagnostics.join_filter_reassociation.merge(result.diagnostics);
            (result.plan, result.change)
        },
        Pass::JoinBuildSideSelection => {
            let result = join_build_side::optimize_join_build_side_selection(plan, &options.join_build_side_selection)?;
            diagnostics.join_build_side_selection.merge(result.diagnostics);
            (result.plan, result.change)
        },
        Pass::OuterJoinNullRejection => {
            let result = outer_join_null_rejection::optimize_outer_join_null_rejection(
                plan,
                &options.outer_join_null_rejection,
            )?;
            diagnostics.outer_join_null_rejection.merge(result.diagnostics);
            (result.plan, result.change)
        },
        Pass::ScalarAggregateDerivation => {
            let result = shared_subplan_materialization::optimize_scalar_aggregate_from_grouped_aggregate_if_enabled(
                plan,
                &options.shared_subplan_materialization,
            )?;
            diagnostics.shared_subplan_materialization.merge(result.diagnostics);
            (result.plan, result.change)
        },
        Pass::AggregateJoinPushdown => {
            let result =
                aggregate_join_pushdown::optimize_aggregate_join_pushdown(plan, &options.aggregate_join_pushdown)?;
            diagnostics.aggregate_join_pushdown.merge(result.diagnostics);
            (result.plan, result.change)
        },
        Pass::DimensionDerivedRowGroupPruning => {
            let result = dimension_pruning::optimize_dimension_derived_row_group_pruning(
                plan,
                &options.dimension_derived_row_group_pruning,
                ctx,
                resolver,
            )?;
            diagnostics.dimension_derived_row_group_pruning.merge(result.diagnostics);
            (result.plan, result.change)
        },
        Pass::SingleDistinctAggregateSplit => {
            let result = single_distinct_aggregate_split::optimize_single_distinct_aggregate_split(
                plan,
                &options.single_distinct_aggregate_split,
            )?;
            diagnostics.single_distinct_aggregate_split.merge(result.diagnostics);
            (result.plan, result.change)
        },
        Pass::CountNonNullColumnPruning => {
            let result = count_non_null_pruning::optimize_count_non_null_column_pruning(
                plan,
                &options.count_non_null_column_pruning,
            )?;
            diagnostics.count_non_null_column_pruning.merge(result.diagnostics);
            (result.plan, result.change)
        },
        Pass::StreamFanoutReplay => {
            let result = stream_fanout_replay::optimize_stream_fanout_replay(plan, &options.stream_fanout_replay)?;
            diagnostics.stream_fanout_replay.merge(result.diagnostics);
            (result.plan, result.change)
        },
        Pass::SharedSubplanMaterialization => {
            let result =
                shared_subplan_materialization::optimize_shared_subplan_materialization_without_scalar_derivation(
                    plan,
                    &options.shared_subplan_materialization,
                )?;
            diagnostics.shared_subplan_materialization.merge(result.diagnostics);
            (result.plan, result.change)
        },
        Pass::LocalParquetSourceReuse => {
            let result = local_parquet_source_reuse::optimize_local_parquet_source_reuse(
                plan,
                &options.local_parquet_source_reuse,
                &options.dynamic_filter_route_discovery,
                ctx,
            )?;
            diagnostics.local_parquet_source_reuse.merge(result.diagnostics);
            (result.plan, result.change)
        },
        Pass::DynamicFilterRouteDiscovery => {
            let result = dynamic_filter::optimize_dynamic_filter_route_discovery(
                plan,
                &options.dynamic_filter_route_discovery,
                ctx,
            );
            diagnostics.dynamic_filter_route_discovery.merge(result.diagnostics);
            (result.plan, result.change)
        },
    })
}

/// Repeats join-filter reassociation until a run reports no change.
///
/// Each applied move can expose the next one, so one run is not enough. The
/// iteration bound is the node count, which exceeds the number of joins any
/// filter can cross; it ends only a loop that a rule defect would otherwise
/// continue. The returned diagnostics cover every run.
fn optimize_join_filter_reassociation_fixpoint(
    plan: QueryPlan,
    options: &JoinFilterReassociationOptions,
    resolver: &dyn LocalParquetPruningResolver,
) -> Result<join_filter_reassociation::JoinFilterReassociationResult> {
    let max_iterations = plan.node_count().max(1);
    let mut current = plan;
    let mut diagnostics = JoinFilterReassociationDiagnostics::default();
    let mut changed = NativeOptimizerChange::Unchanged;

    for _ in 0..max_iterations {
        let result = join_filter_reassociation::optimize_join_filter_reassociation(current, options, resolver)?;
        diagnostics.merge(result.diagnostics);
        if !result.change.changed() {
            return Ok(join_filter_reassociation::JoinFilterReassociationResult {
                plan: result.plan,
                change: changed,
                diagnostics,
            });
        }
        changed = NativeOptimizerChange::Changed;
        current = result.plan;
    }

    Ok(join_filter_reassociation::JoinFilterReassociationResult { plan: current, change: changed, diagnostics })
}

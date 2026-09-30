//! Native query-plan optimizer surface.
//!
//! Optimizer rules in this module operate on [`QueryPlan`] IR only. They do
//! not inspect SQL text, DataFusion plans, benchmark query ids, or adapter
//! report rows.

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
mod source_target_policy;
mod stream_fanout_replay;

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
    discover_dimension_derived_pruning_requests, optimize_dimension_derived_row_group_pruning,
    seed_dimension_derived_reader_predicates,
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
    SemijoinReductionAdmissionDecision, SemijoinReductionAdmissionMode, SemijoinReductionCandidateDiagnostics,
    SemijoinReductionCostGateDiagnostics, SemijoinReductionCostSignal, SemijoinReductionDiagnostics,
    SemijoinReductionKeyMappingDiagnostics, SemijoinReductionOptions, SemijoinReductionRejectedReason,
    SemijoinReductionResult, optimize_semijoin_reduction,
};
#[doc(hidden)]
pub use shared_resident_extraction::{
    SHARED_RESIDENT_EXTRACTION_RULE_NAME, SharedResidentExtractionCandidateDiagnostics,
    SharedResidentExtractionDecision, SharedResidentExtractionDiagnostics, SharedResidentExtractionReason,
    SharedResidentPlan, SharedResidentProducerPlan, optimize_shared_resident_extraction,
};
pub use shared_subplan_materialization::{
    SHARED_SUBPLAN_MATERIALIZATION_RULE_NAME, SharedSubplanMaterializationCandidateDiagnostics,
    SharedSubplanMaterializationDiagnostics, SharedSubplanMaterializationOptions,
    SharedSubplanMaterializationRejectedReason,
};
pub use stream_fanout_replay::{
    STREAM_FANOUT_REPLAY_RULE_NAME, StreamFanoutReplayCandidateDiagnostics, StreamFanoutReplayDiagnostics,
    StreamFanoutReplayOptions,
};

use crate::{backend::AnalysisContext, capability::KeyedHashJoinStrategyOptions, error::Result, plan::QueryPlan};

/// Native optimizer pass configuration.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NativeOptimizerOptions {
    /// Options for reducing rollup/grouping-set aggregates before dimension joins.
    pub aggregate_join_pushdown: AggregateJoinPushdownOptions,
    /// Options for rewriting footer-proven non-null `COUNT(column)` arguments.
    pub count_non_null_column_pruning: CountNonNullColumnPruningOptions,
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
    /// Keyed-hash join strategy policy captured into optimized plans.
    pub keyed_hash_join_strategy: KeyedHashJoinStrategyOptions,
}

impl NativeOptimizerOptions {
    /// Returns options with all optimizer passes disabled.
    pub fn disabled() -> Self {
        Self {
            aggregate_join_pushdown: AggregateJoinPushdownOptions::disabled(),
            count_non_null_column_pruning: CountNonNullColumnPruningOptions::disabled(),
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
            keyed_hash_join_strategy: KeyedHashJoinStrategyOptions::default(),
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
        self.dynamic_filter_route_discovery.execution.validate()?;
        self.keyed_hash_join_strategy.validate()?;
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
    /// Whether any pass changed the plan.
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
    /// Diagnostics from the shared-subplan materialization discovery pass.
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

/// Runs the native optimizer pass list over a query plan.
pub fn optimize_query_plan(
    plan: QueryPlan,
    options: &NativeOptimizerOptions,
    ctx: AnalysisContext<'_>,
    resolver: &dyn LocalParquetPruningResolver,
) -> NativeOptimizerResult {
    let plan = plan.with_keyed_hash_join_strategy_options(options.keyed_hash_join_strategy);
    let projection_simplification_result =
        projection_simplification::optimize_projection_simplification(plan, &options.projection_simplification);
    let semijoin_result =
        optimize_semijoin_reduction(projection_simplification_result.plan, &options.semijoin_reduction, ctx);
    let join_filter_reassociation_result =
        optimize_join_filter_reassociation_fixpoint(semijoin_result.plan, &options.join_filter_reassociation, resolver);
    let join_build_side_result = join_build_side::optimize_join_build_side_selection(
        join_filter_reassociation_result.plan,
        &options.join_build_side_selection,
    );
    let null_rejection_result = outer_join_null_rejection::optimize_outer_join_null_rejection(
        join_build_side_result.plan,
        &options.outer_join_null_rejection,
    );
    let pre_aggregate_scalar_derivation_result =
        shared_subplan_materialization::optimize_scalar_aggregate_from_grouped_aggregate_if_enabled(
            null_rejection_result.plan,
            &options.shared_subplan_materialization,
        );
    let aggregate_join_pushdown_result = aggregate_join_pushdown::optimize_aggregate_join_pushdown(
        pre_aggregate_scalar_derivation_result.plan,
        &options.aggregate_join_pushdown,
    );
    let aggregate_join_pushdown_change = aggregate_join_pushdown_result.change;
    let post_aggregate_join_build_side_result = rerun_optimizer_pass_if_changed(
        aggregate_join_pushdown_result.plan,
        aggregate_join_pushdown_change,
        |plan| join_build_side::optimize_join_build_side_selection(plan, &options.join_build_side_selection),
        |plan| join_build_side::JoinBuildSideSelectionResult {
            plan,
            change: NativeOptimizerChange::Unchanged,
            diagnostics: JoinBuildSideSelectionDiagnostics::default(),
        },
    );
    let post_aggregate_null_rejection_result = rerun_optimizer_pass_if_changed(
        post_aggregate_join_build_side_result.plan,
        aggregate_join_pushdown_change,
        |plan| outer_join_null_rejection::optimize_outer_join_null_rejection(plan, &options.outer_join_null_rejection),
        |plan| outer_join_null_rejection::OuterJoinNullRejectionResult {
            plan,
            change: NativeOptimizerChange::Unchanged,
            diagnostics: OuterJoinNullRejectionDiagnostics::default(),
        },
    );
    let dimension_pruning_result = dimension_pruning::optimize_dimension_derived_row_group_pruning(
        post_aggregate_null_rejection_result.plan,
        &options.dimension_derived_row_group_pruning,
        ctx,
        resolver,
    );
    let dimension_pruning_change = dimension_pruning_result.change;
    let post_dimension_join_build_side_result = rerun_optimizer_pass_if_changed(
        dimension_pruning_result.plan,
        dimension_pruning_change,
        |plan| join_build_side::optimize_join_build_side_selection(plan, &options.join_build_side_selection),
        |plan| join_build_side::JoinBuildSideSelectionResult {
            plan,
            change: NativeOptimizerChange::Unchanged,
            diagnostics: JoinBuildSideSelectionDiagnostics::default(),
        },
    );
    let count_non_null_column_pruning_result = count_non_null_pruning::optimize_count_non_null_column_pruning(
        post_dimension_join_build_side_result.plan,
        &options.count_non_null_column_pruning,
    );
    let stream_fanout_replay_result = stream_fanout_replay::optimize_stream_fanout_replay(
        count_non_null_column_pruning_result.plan,
        &options.stream_fanout_replay,
    );
    let shared_subplan_materialization_result =
        shared_subplan_materialization::optimize_shared_subplan_materialization_without_scalar_derivation(
            stream_fanout_replay_result.plan,
            &options.shared_subplan_materialization,
        );
    let local_parquet_source_reuse_result = local_parquet_source_reuse::optimize_local_parquet_source_reuse(
        shared_subplan_materialization_result.plan,
        &options.local_parquet_source_reuse,
        &options.dynamic_filter_route_discovery,
        ctx,
    );
    let final_projection_simplification_result = projection_simplification::optimize_projection_simplification(
        local_parquet_source_reuse_result.plan,
        &options.projection_simplification,
    );
    let dynamic_filter_route_result = dynamic_filter::optimize_dynamic_filter_route_discovery(
        final_projection_simplification_result.plan,
        &options.dynamic_filter_route_discovery,
        ctx,
    );
    let shared_resident_result = optimize_shared_resident_extraction(dynamic_filter_route_result.plan);
    let change = if projection_simplification_result.change.changed()
        || final_projection_simplification_result.change.changed()
        || semijoin_result.change.changed()
        || null_rejection_result.change.changed()
        || join_filter_reassociation_result.change.changed()
        || dimension_pruning_result.change.changed()
        || join_build_side_result.change.changed()
        || pre_aggregate_scalar_derivation_result.change.changed()
        || aggregate_join_pushdown_result.change.changed()
        || post_aggregate_join_build_side_result.change.changed()
        || post_aggregate_null_rejection_result.change.changed()
        || post_dimension_join_build_side_result.change.changed()
        || count_non_null_column_pruning_result.change.changed()
        || stream_fanout_replay_result.change.changed()
        || local_parquet_source_reuse_result.change.changed()
        || shared_subplan_materialization_result.change.changed()
        || dynamic_filter_route_result.change.changed()
        || !shared_resident_result.producers.is_empty()
    {
        NativeOptimizerChange::Changed
    } else {
        NativeOptimizerChange::Unchanged
    };
    NativeOptimizerResult {
        change,
        plan: shared_resident_result.consumer,
        shared_resident_producers: shared_resident_result.producers,
        diagnostics: NativeOptimizerDiagnostics {
            aggregate_join_pushdown: aggregate_join_pushdown_result.diagnostics,
            count_non_null_column_pruning: count_non_null_column_pruning_result.diagnostics,
            projection_simplification: {
                let mut diagnostics = projection_simplification_result.diagnostics;
                diagnostics.merge(final_projection_simplification_result.diagnostics);
                diagnostics
            },
            semijoin_reduction: semijoin_result.diagnostics,
            outer_join_null_rejection: {
                let mut diagnostics = null_rejection_result.diagnostics;
                diagnostics.merge(post_aggregate_null_rejection_result.diagnostics);
                diagnostics
            },
            join_filter_reassociation: join_filter_reassociation_result.diagnostics,
            dimension_derived_row_group_pruning: dimension_pruning_result.diagnostics,
            join_build_side_selection: {
                let mut diagnostics = join_build_side_result.diagnostics;
                diagnostics.merge(post_aggregate_join_build_side_result.diagnostics);
                diagnostics.merge(post_dimension_join_build_side_result.diagnostics);
                diagnostics
            },
            stream_fanout_replay: stream_fanout_replay_result.diagnostics,
            local_parquet_source_reuse: local_parquet_source_reuse_result.diagnostics,
            shared_subplan_materialization: {
                let mut diagnostics = pre_aggregate_scalar_derivation_result.diagnostics;
                let shared_diagnostics = shared_subplan_materialization_result.diagnostics;
                let suppress_empty_post_scalar_discovery = !diagnostics.is_empty()
                    && shared_diagnostics.candidate_count == 0
                    && shared_diagnostics.applied_count == 0
                    && shared_diagnostics.candidates.is_empty();
                if !suppress_empty_post_scalar_discovery {
                    diagnostics.merge(shared_diagnostics);
                }
                diagnostics
            },
            shared_resident_extraction: shared_resident_result.diagnostics,
            dynamic_filter_route_discovery: dynamic_filter_route_result.diagnostics,
        },
    }
}

fn rerun_optimizer_pass_if_changed<T>(
    plan: QueryPlan,
    change: NativeOptimizerChange,
    rerun: impl FnOnce(QueryPlan) -> T,
    unchanged: impl FnOnce(QueryPlan) -> T,
) -> T {
    if change.changed() { rerun(plan) } else { unchanged(plan) }
}

fn optimize_join_filter_reassociation_fixpoint(
    plan: QueryPlan,
    options: &JoinFilterReassociationOptions,
    resolver: &dyn LocalParquetPruningResolver,
) -> join_filter_reassociation::JoinFilterReassociationResult {
    let max_iterations = plan.node_count().max(1);
    let mut current = plan;
    let mut diagnostics = JoinFilterReassociationDiagnostics::default();
    let mut changed = NativeOptimizerChange::Unchanged;

    for _ in 0..max_iterations {
        let result = join_filter_reassociation::optimize_join_filter_reassociation(current, options, resolver);
        diagnostics.merge(result.diagnostics);
        if !result.change.changed() {
            return join_filter_reassociation::JoinFilterReassociationResult {
                plan: result.plan,
                change: changed,
                diagnostics,
            };
        }
        changed = NativeOptimizerChange::Changed;
        current = result.plan;
    }

    join_filter_reassociation::JoinFilterReassociationResult { plan: current, change: changed, diagnostics }
}

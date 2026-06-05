//! DataFusion-free native query-plan optimizer surface.
//!
//! Optimizer rules in this module operate on [`QueryPlan`] IR only. They do
//! not inspect SQL text, DataFusion plans, benchmark query ids, or adapter
//! report rows.

mod semijoin_reduction;

mod aggregate_join_pushdown;
mod diagnostics;
mod dimension_pruning;
mod dynamic_filter;
mod join_build_side;
mod join_filter_reassociation;
mod local_parquet_source_reuse;
mod outer_join_null_rejection;
mod projection_simplification;
mod rewrite;
mod shared_subplan_materialization;
mod stream_fanout_replay;

pub use aggregate_join_pushdown::{
    AGGREGATE_JOIN_PUSHDOWN_RULE_NAME, AggregateJoinPushdownCandidateDiagnostics, AggregateJoinPushdownDiagnostics,
    AggregateJoinPushdownOptions, AggregateJoinPushdownRejectedReason,
};
pub use dimension_pruning::{
    DIMENSION_DERIVED_ROW_GROUP_PRUNING_RULE_NAME, DIMENSION_KEY_FILTER_MAX_RUNS, DimensionDerivedApplyResult,
    DimensionDerivedPruningRequest, DimensionDerivedRowGroupPruningCandidateDiagnostics,
    DimensionDerivedRowGroupPruningDiagnostics, DimensionDerivedRowGroupPruningOptions,
    DimensionDerivedRowGroupPruningRejectedReason, DimensionKeyAccumulator, DimensionKeyFilter, KeyRange,
    RowGroupOverlapSelection, apply_dimension_derived_pruning_with_stats, dimension_key_filter_from_batch,
    discover_dimension_derived_pruning_requests, seed_dimension_derived_reader_predicates,
    select_overlapping_row_groups,
};
pub use dynamic_filter::{
    DYNAMIC_FILTER_ROUTE_DISCOVERY_RULE_NAME, DynamicFilterRouteCandidateDiagnostics,
    DynamicFilterRouteDiscoveryDiagnostics, DynamicFilterRouteDiscoveryOptions, DynamicFilterRouteRejectedReason,
};
pub use join_build_side::{
    JOIN_BUILD_SIDE_SELECTION_RULE_NAME, JoinBuildSideSelectionCandidateDiagnostics, JoinBuildSideSelectionDiagnostics,
    JoinBuildSideSelectionOptions, JoinBuildSideSelectionRejectedReason,
};
pub use join_filter_reassociation::{
    JOIN_FILTER_REASSOCIATION_RULE_NAME, JoinFilterReassociationCandidateDiagnostics,
    JoinFilterReassociationDiagnostics, JoinFilterReassociationOptions, JoinFilterReassociationRejectedReason,
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
pub use semijoin_reduction::{
    DEFAULT_SEMIJOIN_REDUCTION_COST_THRESHOLD_MULTIPLIER, SEMIJOIN_REDUCTION_RULE_NAME,
    SemijoinReductionAdmissionDecision, SemijoinReductionAdmissionMode, SemijoinReductionCandidateDiagnostics,
    SemijoinReductionCostGateDiagnostics, SemijoinReductionCostSignal, SemijoinReductionDiagnostics,
    SemijoinReductionKeyMappingDiagnostics, SemijoinReductionOptions, SemijoinReductionRejectedReason,
    SemijoinReductionResult, optimize_semijoin_reduction,
};
pub use shared_subplan_materialization::{
    SHARED_SUBPLAN_MATERIALIZATION_RULE_NAME, SharedSubplanMaterializationCandidateDiagnostics,
    SharedSubplanMaterializationDiagnostics, SharedSubplanMaterializationOptions,
    SharedSubplanMaterializationRejectedReason,
};
pub use stream_fanout_replay::{
    STREAM_FANOUT_REPLAY_RULE_NAME, StreamFanoutReplayCandidateDiagnostics, StreamFanoutReplayDiagnostics,
    StreamFanoutReplayOptions, StreamFanoutReplayRejectedReason,
};

use crate::plan::QueryPlan;

/// Native optimizer pass configuration.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NativeOptimizerOptions {
    /// Options for reducing rollup/grouping-set aggregates before dimension joins.
    pub aggregate_join_pushdown: AggregateJoinPushdownOptions,
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
}

/// Whether an optimizer invocation changed the native IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeOptimizerChange {
    Unchanged,
    Changed,
}

impl NativeOptimizerChange {
    pub fn changed(self) -> bool {
        matches!(self, Self::Changed)
    }
}

/// Result returned by the native optimizer entry point.
#[derive(Debug, Clone, PartialEq)]
pub struct NativeOptimizerResult {
    /// Optimized native query plan.
    pub plan: QueryPlan,
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
    /// Diagnostics from the dynamic-filter route discovery pass.
    pub dynamic_filter_route_discovery: DynamicFilterRouteDiscoveryDiagnostics,
}

impl NativeOptimizerDiagnostics {
    /// Returns true when no optimizer pass considered or applied candidates.
    pub fn is_empty(&self) -> bool {
        self.aggregate_join_pushdown.is_empty()
            && self.projection_simplification.is_empty()
            && self.semijoin_reduction.is_empty()
            && self.outer_join_null_rejection.is_empty()
            && self.join_filter_reassociation.is_empty()
            && self.dimension_derived_row_group_pruning.is_empty()
            && self.join_build_side_selection.is_empty()
            && self.stream_fanout_replay.is_empty()
            && self.local_parquet_source_reuse.is_empty()
            && self.shared_subplan_materialization.is_empty()
            && self.dynamic_filter_route_discovery.is_empty()
    }

    /// Merges diagnostics from another optimizer invocation into this one.
    pub fn merge(&mut self, other: Self) {
        self.aggregate_join_pushdown.merge(other.aggregate_join_pushdown);
        self.projection_simplification.merge(other.projection_simplification);
        self.semijoin_reduction.merge(other.semijoin_reduction);
        self.outer_join_null_rejection.merge(other.outer_join_null_rejection);
        self.join_filter_reassociation.merge(other.join_filter_reassociation);
        self.dimension_derived_row_group_pruning.merge(other.dimension_derived_row_group_pruning);
        self.join_build_side_selection.merge(other.join_build_side_selection);
        self.stream_fanout_replay.merge(other.stream_fanout_replay);
        self.local_parquet_source_reuse.merge(other.local_parquet_source_reuse);
        self.shared_subplan_materialization.merge(other.shared_subplan_materialization);
        self.dynamic_filter_route_discovery.merge(other.dynamic_filter_route_discovery);
    }
}

/// Runs the native optimizer pass list over a DataFusion-free query plan.
pub fn optimize_query_plan(plan: QueryPlan, options: &NativeOptimizerOptions) -> NativeOptimizerResult {
    let projection_simplification_result =
        projection_simplification::optimize_projection_simplification(plan, &options.projection_simplification);
    let semijoin_result =
        optimize_semijoin_reduction(projection_simplification_result.plan, &options.semijoin_reduction);
    let null_rejection_result = outer_join_null_rejection::optimize_outer_join_null_rejection(
        semijoin_result.plan,
        &options.outer_join_null_rejection,
    );
    let join_filter_reassociation_result =
        optimize_join_filter_reassociation_fixpoint(null_rejection_result.plan, &options.join_filter_reassociation);
    let dimension_pruning_result = dimension_pruning::optimize_dimension_derived_row_group_pruning(
        join_filter_reassociation_result.plan,
        &options.dimension_derived_row_group_pruning,
    );
    let join_build_side_result = join_build_side::optimize_join_build_side_selection(
        dimension_pruning_result.plan,
        &options.join_build_side_selection,
    );
    let aggregate_join_pushdown_result = aggregate_join_pushdown::optimize_aggregate_join_pushdown(
        join_build_side_result.plan,
        &options.aggregate_join_pushdown,
    );
    let stream_fanout_replay_result = stream_fanout_replay::optimize_stream_fanout_replay(
        aggregate_join_pushdown_result.plan,
        &options.stream_fanout_replay,
    );
    let shared_subplan_materialization_result = shared_subplan_materialization::optimize_shared_subplan_materialization(
        stream_fanout_replay_result.plan,
        &options.shared_subplan_materialization,
    );
    let local_parquet_source_reuse_result = local_parquet_source_reuse::optimize_local_parquet_source_reuse(
        shared_subplan_materialization_result.plan,
        &options.local_parquet_source_reuse,
    );
    let final_projection_simplification_result = projection_simplification::optimize_projection_simplification(
        local_parquet_source_reuse_result.plan,
        &options.projection_simplification,
    );
    let dynamic_filter_route_result = dynamic_filter::optimize_dynamic_filter_route_discovery(
        final_projection_simplification_result.plan,
        &options.dynamic_filter_route_discovery,
    );
    let change = if projection_simplification_result.change.changed()
        || final_projection_simplification_result.change.changed()
        || semijoin_result.change.changed()
        || null_rejection_result.change.changed()
        || join_filter_reassociation_result.change.changed()
        || dimension_pruning_result.change.changed()
        || join_build_side_result.change.changed()
        || aggregate_join_pushdown_result.change.changed()
        || stream_fanout_replay_result.change.changed()
        || local_parquet_source_reuse_result.change.changed()
        || shared_subplan_materialization_result.change.changed()
        || dynamic_filter_route_result.change.changed()
    {
        NativeOptimizerChange::Changed
    } else {
        NativeOptimizerChange::Unchanged
    };
    NativeOptimizerResult {
        change,
        plan: dynamic_filter_route_result.plan,
        diagnostics: NativeOptimizerDiagnostics {
            aggregate_join_pushdown: aggregate_join_pushdown_result.diagnostics,
            projection_simplification: {
                let mut diagnostics = projection_simplification_result.diagnostics;
                diagnostics.merge(final_projection_simplification_result.diagnostics);
                diagnostics
            },
            semijoin_reduction: semijoin_result.diagnostics,
            outer_join_null_rejection: null_rejection_result.diagnostics,
            join_filter_reassociation: join_filter_reassociation_result.diagnostics,
            dimension_derived_row_group_pruning: dimension_pruning_result.diagnostics,
            join_build_side_selection: join_build_side_result.diagnostics,
            stream_fanout_replay: stream_fanout_replay_result.diagnostics,
            local_parquet_source_reuse: local_parquet_source_reuse_result.diagnostics,
            shared_subplan_materialization: shared_subplan_materialization_result.diagnostics,
            dynamic_filter_route_discovery: dynamic_filter_route_result.diagnostics,
        },
    }
}

fn optimize_join_filter_reassociation_fixpoint(
    plan: QueryPlan,
    options: &JoinFilterReassociationOptions,
) -> join_filter_reassociation::JoinFilterReassociationResult {
    let max_iterations = plan.node_count().max(1);
    let mut current = plan;
    let mut diagnostics = JoinFilterReassociationDiagnostics::default();
    let mut changed = NativeOptimizerChange::Unchanged;

    for _ in 0..max_iterations {
        let result = join_filter_reassociation::optimize_join_filter_reassociation(current, options);
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

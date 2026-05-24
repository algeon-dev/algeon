//! DataFusion-free native query-plan optimizer surface.
//!
//! Optimizer rules in this module operate on [`QueryPlan`] IR only. They do
//! not inspect SQL text, DataFusion plans, benchmark query ids, or adapter
//! report rows.

mod semijoin_reduction;

mod join_build_side;
mod join_filter_reassociation;
mod outer_join_null_rejection;
mod stream_fanout_replay;

pub use join_build_side::{
    JOIN_BUILD_SIDE_SELECTION_RULE_NAME, JoinBuildSideSelectionCandidateDiagnostics, JoinBuildSideSelectionDiagnostics,
    JoinBuildSideSelectionOptions, JoinBuildSideSelectionRejectedReason,
};
pub use join_filter_reassociation::{
    JOIN_FILTER_REASSOCIATION_RULE_NAME, JoinFilterReassociationCandidateDiagnostics,
    JoinFilterReassociationDiagnostics, JoinFilterReassociationOptions, JoinFilterReassociationRejectedReason,
};
pub use outer_join_null_rejection::{
    OUTER_JOIN_NULL_REJECTION_RULE_NAME, OuterJoinNullRejectionCandidateDiagnostics, OuterJoinNullRejectionDiagnostics,
    OuterJoinNullRejectionOptions, OuterJoinNullRejectionRejectedReason,
};
pub use semijoin_reduction::{
    DEFAULT_SEMIJOIN_REDUCTION_COST_THRESHOLD_MULTIPLIER, SEMIJOIN_REDUCTION_RULE_NAME,
    SemijoinReductionAdmissionDecision, SemijoinReductionAdmissionMode, SemijoinReductionCandidateDiagnostics,
    SemijoinReductionCostGateDiagnostics, SemijoinReductionCostSignal, SemijoinReductionDiagnostics,
    SemijoinReductionKeyMappingDiagnostics, SemijoinReductionOptions, SemijoinReductionRejectedReason,
    SemijoinReductionResult, optimize_semijoin_reduction,
};
pub use stream_fanout_replay::{
    STREAM_FANOUT_REPLAY_RULE_NAME, StreamFanoutReplayCandidateDiagnostics, StreamFanoutReplayDiagnostics,
    StreamFanoutReplayOptions, StreamFanoutReplayRejectedReason,
};

use crate::plan::QueryPlan;

/// Native optimizer pass configuration.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NativeOptimizerOptions {
    /// Options for pushing left-semi key filters below eligible joins.
    pub semijoin_reduction: SemijoinReductionOptions,
    /// Options for rewriting null-rejected left outer joins to inner joins.
    pub outer_join_null_rejection: OuterJoinNullRejectionOptions,
    /// Options for moving join filters across preserved-side join structure.
    pub join_filter_reassociation: JoinFilterReassociationOptions,
    /// Options for choosing smaller hash-join build sides.
    pub join_build_side_selection: JoinBuildSideSelectionOptions,
    /// Options for replaying shared stream probe-input subtrees.
    pub stream_fanout_replay: StreamFanoutReplayOptions,
}

impl NativeOptimizerOptions {
    /// Returns options with all optimizer passes disabled.
    pub fn disabled() -> Self {
        Self {
            semijoin_reduction: SemijoinReductionOptions::disabled(),
            outer_join_null_rejection: OuterJoinNullRejectionOptions::disabled(),
            join_filter_reassociation: JoinFilterReassociationOptions::disabled(),
            join_build_side_selection: JoinBuildSideSelectionOptions::disabled(),
            stream_fanout_replay: StreamFanoutReplayOptions::disabled(),
        }
    }
}

/// Whether an optimizer invocation changed the native IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeOptimizerChange {
    /// No native IR nodes were changed.
    Unchanged,
    /// At least one optimizer rule changed the native IR.
    Changed,
}

impl NativeOptimizerChange {
    /// Returns true when an optimizer pass changed the native IR.
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
    /// Returns true when any optimizer pass changed the native IR.
    pub fn changed(&self) -> bool {
        self.change.changed()
    }
}

/// Diagnostics emitted by native optimizer rules.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NativeOptimizerDiagnostics {
    /// Diagnostics from the semijoin-reduction pass.
    pub semijoin_reduction: SemijoinReductionDiagnostics,
    /// Diagnostics from the outer-join null-rejection pass.
    pub outer_join_null_rejection: OuterJoinNullRejectionDiagnostics,
    /// Diagnostics from the join-filter reassociation pass.
    pub join_filter_reassociation: JoinFilterReassociationDiagnostics,
    /// Diagnostics from the join build-side selection pass.
    pub join_build_side_selection: JoinBuildSideSelectionDiagnostics,
    /// Diagnostics from the stream fanout replay pass.
    pub stream_fanout_replay: StreamFanoutReplayDiagnostics,
}

impl NativeOptimizerDiagnostics {
    /// Returns true when no optimizer pass considered or applied candidates.
    pub fn is_empty(&self) -> bool {
        self.semijoin_reduction.is_empty()
            && self.outer_join_null_rejection.is_empty()
            && self.join_filter_reassociation.is_empty()
            && self.join_build_side_selection.is_empty()
            && self.stream_fanout_replay.is_empty()
    }

    /// Merges diagnostics from another optimizer invocation into this one.
    pub fn merge(&mut self, other: Self) {
        self.semijoin_reduction.merge(other.semijoin_reduction);
        self.outer_join_null_rejection.merge(other.outer_join_null_rejection);
        self.join_filter_reassociation.merge(other.join_filter_reassociation);
        self.join_build_side_selection.merge(other.join_build_side_selection);
        self.stream_fanout_replay.merge(other.stream_fanout_replay);
    }
}

/// Runs the native optimizer pass list over a DataFusion-free query plan.
pub fn optimize_query_plan(plan: QueryPlan, options: &NativeOptimizerOptions) -> NativeOptimizerResult {
    let semijoin_result = optimize_semijoin_reduction(plan, &options.semijoin_reduction);
    let null_rejection_result = outer_join_null_rejection::optimize_outer_join_null_rejection(
        semijoin_result.plan,
        &options.outer_join_null_rejection,
    );
    let join_filter_reassociation_result = join_filter_reassociation::optimize_join_filter_reassociation(
        null_rejection_result.plan,
        &options.join_filter_reassociation,
    );
    let join_build_side_result = join_build_side::optimize_join_build_side_selection(
        join_filter_reassociation_result.plan,
        &options.join_build_side_selection,
    );
    let stream_fanout_replay_result =
        stream_fanout_replay::optimize_stream_fanout_replay(join_build_side_result.plan, &options.stream_fanout_replay);
    let change = if semijoin_result.change.changed()
        || null_rejection_result.change.changed()
        || join_filter_reassociation_result.change.changed()
        || join_build_side_result.change.changed()
        || stream_fanout_replay_result.change.changed()
    {
        NativeOptimizerChange::Changed
    } else {
        NativeOptimizerChange::Unchanged
    };
    NativeOptimizerResult {
        change,
        plan: stream_fanout_replay_result.plan,
        diagnostics: NativeOptimizerDiagnostics {
            semijoin_reduction: semijoin_result.diagnostics,
            outer_join_null_rejection: null_rejection_result.diagnostics,
            join_filter_reassociation: join_filter_reassociation_result.diagnostics,
            join_build_side_selection: join_build_side_result.diagnostics,
            stream_fanout_replay: stream_fanout_replay_result.diagnostics,
        },
    }
}

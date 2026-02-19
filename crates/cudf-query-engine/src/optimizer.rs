//! DataFusion-free native query-plan optimizer surface.
//!
//! Optimizer rules in this module operate on [`QueryPlan`] IR only. They do
//! not inspect SQL text, DataFusion plans, benchmark query ids, or adapter
//! report rows.

mod semijoin_reduction;

mod join_build_side;
mod join_filter_reassociation;
mod outer_join_null_rejection;

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

use crate::plan::QueryPlan;

/// Native optimizer pass configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeOptimizerOptions {
    pub semijoin_reduction: SemijoinReductionOptions,
    pub outer_join_null_rejection: OuterJoinNullRejectionOptions,
    pub join_filter_reassociation: JoinFilterReassociationOptions,
    pub join_build_side_selection: JoinBuildSideSelectionOptions,
}

impl Default for NativeOptimizerOptions {
    fn default() -> Self {
        Self {
            semijoin_reduction: SemijoinReductionOptions::default(),
            outer_join_null_rejection: OuterJoinNullRejectionOptions::default(),
            join_filter_reassociation: JoinFilterReassociationOptions::default(),
            join_build_side_selection: JoinBuildSideSelectionOptions::default(),
        }
    }
}

impl NativeOptimizerOptions {
    /// Returns options with all optimizer passes disabled.
    pub fn disabled() -> Self {
        Self {
            semijoin_reduction: SemijoinReductionOptions::disabled(),
            outer_join_null_rejection: OuterJoinNullRejectionOptions::disabled(),
            join_filter_reassociation: JoinFilterReassociationOptions::disabled(),
            join_build_side_selection: JoinBuildSideSelectionOptions::disabled(),
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
    pub plan: QueryPlan,
    pub change: NativeOptimizerChange,
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
    pub semijoin_reduction: SemijoinReductionDiagnostics,
    pub outer_join_null_rejection: OuterJoinNullRejectionDiagnostics,
    pub join_filter_reassociation: JoinFilterReassociationDiagnostics,
    pub join_build_side_selection: JoinBuildSideSelectionDiagnostics,
}

impl NativeOptimizerDiagnostics {
    pub fn is_empty(&self) -> bool {
        self.semijoin_reduction.is_empty()
            && self.outer_join_null_rejection.is_empty()
            && self.join_filter_reassociation.is_empty()
            && self.join_build_side_selection.is_empty()
    }

    pub fn merge(&mut self, other: Self) {
        self.semijoin_reduction.merge(other.semijoin_reduction);
        self.outer_join_null_rejection.merge(other.outer_join_null_rejection);
        self.join_filter_reassociation.merge(other.join_filter_reassociation);
        self.join_build_side_selection.merge(other.join_build_side_selection);
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
    let change = if semijoin_result.change.changed()
        || null_rejection_result.change.changed()
        || join_filter_reassociation_result.change.changed()
        || join_build_side_result.change.changed()
    {
        NativeOptimizerChange::Changed
    } else {
        NativeOptimizerChange::Unchanged
    };
    NativeOptimizerResult {
        change,
        plan: join_build_side_result.plan,
        diagnostics: NativeOptimizerDiagnostics {
            semijoin_reduction: semijoin_result.diagnostics,
            outer_join_null_rejection: null_rejection_result.diagnostics,
            join_filter_reassociation: join_filter_reassociation_result.diagnostics,
            join_build_side_selection: join_build_side_result.diagnostics,
        },
    }
}

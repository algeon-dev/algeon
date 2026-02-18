//! DataFusion-free native query-plan optimizer surface.
//!
//! Optimizer rules in this module operate on [`QueryPlan`] IR only. They do
//! not inspect SQL text, DataFusion plans, benchmark query ids, or adapter
//! report rows.

mod semijoin_reduction;

mod join_build_side;

pub use join_build_side::{
    JOIN_BUILD_SIDE_SELECTION_RULE_NAME, JoinBuildSideSelectionCandidateDiagnostics, JoinBuildSideSelectionDiagnostics,
    JoinBuildSideSelectionOptions, JoinBuildSideSelectionRejectedReason,
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
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NativeOptimizerOptions {
    pub semijoin_reduction: SemijoinReductionOptions,
    pub join_build_side_selection: JoinBuildSideSelectionOptions,
}

impl NativeOptimizerOptions {
    /// Returns options with all optimizer passes disabled.
    pub fn disabled() -> Self {
        Self {
            semijoin_reduction: SemijoinReductionOptions::disabled(),
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
    pub join_build_side_selection: JoinBuildSideSelectionDiagnostics,
}

impl NativeOptimizerDiagnostics {
    pub fn is_empty(&self) -> bool {
        self.semijoin_reduction.is_empty() && self.join_build_side_selection.is_empty()
    }

    pub fn merge(&mut self, other: Self) {
        self.semijoin_reduction.merge(other.semijoin_reduction);
        self.join_build_side_selection.merge(other.join_build_side_selection);
    }
}

/// Runs the native optimizer pass list over a DataFusion-free query plan.
pub fn optimize_query_plan(plan: QueryPlan, options: &NativeOptimizerOptions) -> NativeOptimizerResult {
    let semijoin_result = optimize_semijoin_reduction(plan, &options.semijoin_reduction);
    let join_build_side_result =
        join_build_side::optimize_join_build_side_selection(semijoin_result.plan, &options.join_build_side_selection);
    let change = if semijoin_result.change.changed() || join_build_side_result.change.changed() {
        NativeOptimizerChange::Changed
    } else {
        NativeOptimizerChange::Unchanged
    };
    NativeOptimizerResult {
        change,
        plan: join_build_side_result.plan,
        diagnostics: NativeOptimizerDiagnostics {
            semijoin_reduction: semijoin_result.diagnostics,
            join_build_side_selection: join_build_side_result.diagnostics,
        },
    }
}

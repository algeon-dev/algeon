//! DataFusion-free native query-plan optimizer surface.
//!
//! Optimizer rules in this module operate on [`QueryPlan`] IR only. They do
//! not inspect SQL text, DataFusion plans, benchmark query ids, or adapter
//! report rows.

mod semijoin_reduction;

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
}

impl NativeOptimizerOptions {
    /// Returns options with all optimizer passes disabled.
    pub fn disabled() -> Self {
        Self { semijoin_reduction: SemijoinReductionOptions::disabled() }
    }
}

impl Default for NativeOptimizerOptions {
    fn default() -> Self {
        Self { semijoin_reduction: SemijoinReductionOptions::default() }
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
}

impl NativeOptimizerDiagnostics {
    pub fn is_empty(&self) -> bool {
        self.semijoin_reduction.is_empty()
    }

    pub fn merge(&mut self, other: Self) {
        self.semijoin_reduction.merge(other.semijoin_reduction);
    }
}

/// Runs the native optimizer pass list over a DataFusion-free query plan.
pub fn optimize_query_plan(plan: QueryPlan, options: &NativeOptimizerOptions) -> NativeOptimizerResult {
    let semijoin_result = optimize_semijoin_reduction(plan, &options.semijoin_reduction);
    NativeOptimizerResult {
        change: semijoin_result.change,
        plan: semijoin_result.plan,
        diagnostics: NativeOptimizerDiagnostics { semijoin_reduction: semijoin_result.diagnostics },
    }
}

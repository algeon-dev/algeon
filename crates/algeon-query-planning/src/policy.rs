//! Execution policy shared by optimization, admission, and execution.

use crate::{capability::KeyedHashJoinStrategyOptions, dynamic_filter::DynamicFilterExecutionOptions, error::Result};

/// Thresholds one plan is optimized, admitted, and executed under.
///
/// The policy travels with the [`AnalysisContext`](crate::backend::AnalysisContext)
/// of each pass and is recorded by the admission artifact; the plan IR does
/// not carry it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PlanningPolicy {
    /// Keyed-hash join strategy thresholds.
    pub keyed_hash_join: KeyedHashJoinStrategyOptions,
    /// Dynamic-filter route and application thresholds.
    pub dynamic_filter: DynamicFilterExecutionOptions,
}

impl PlanningPolicy {
    /// Rejects thresholds the optimizer and the engine cannot honor.
    pub fn validate(&self) -> Result<()> {
        self.keyed_hash_join.validate()?;
        self.dynamic_filter.validate()
    }
}

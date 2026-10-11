//! Execution policy shared by optimization, admission, and execution.

use crate::{capability::KeyedHashJoinStrategyOptions, dynamic_filter::DynamicFilterExecutionOptions, error::Result};

/// Thresholds one plan is optimized, admitted, and executed under.
///
/// The policy travels with the [`AnalysisContext`](crate::backend::AnalysisContext)
/// of each pass and is recorded by the admission artifact; the plan IR does
/// not carry it. Keeping it out of the IR means two plans that differ only in
/// policy share one fingerprint, and the admitted artifact, which stores both,
/// is the only place the pairing is fixed. `Copy` because every pass takes it
/// by value and nothing in it is shared state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PlanningPolicy {
    /// Keyed-hash join strategy thresholds.
    pub keyed_hash_join: KeyedHashJoinStrategyOptions,
    /// Dynamic-filter route and application thresholds.
    pub dynamic_filter: DynamicFilterExecutionOptions,
}

impl PlanningPolicy {
    /// Rejects thresholds the optimizer and the engine cannot honor.
    ///
    /// Each component policy validates its own ordering constraints and
    /// reports the first failure with the field names in its facts. Zero is
    /// accepted everywhere as "branch disabled"; see the component docs.
    pub fn validate(&self) -> Result<()> {
        self.keyed_hash_join.validate()?;
        self.dynamic_filter.validate()
    }
}

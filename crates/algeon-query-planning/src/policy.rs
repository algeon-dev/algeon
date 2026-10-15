//! Execution policy shared by optimization, admission, and execution.

use crate::{capability::KeyedHashJoinStrategyOptions, dynamic_filter::DynamicFilterExecutionOptions, error::Result};

/// Thresholds applied during plan optimization, admission, and execution.
///
/// The policy accompanies the [`AnalysisContext`](crate::backend::AnalysisContext)
/// across planning passes and is captured in the resulting admission artifact
/// rather than embedded directly in plan IR. This separation allows plan IR
/// fingerprints to remain invariant to policy adjustments while recording the
/// active policy in the admitted artifact. The type implements `Copy` because
/// passes receive it by value without shared state.
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

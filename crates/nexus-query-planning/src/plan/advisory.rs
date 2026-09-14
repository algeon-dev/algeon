//! Advisory data-volume signals attached by a frontend planner.
//!
//! These signals describe a frontend's own estimate of a node's whole output
//! (all partitions). They are optimizer hints only: they never replace the
//! conservative estimates in [`super::estimates`], the proved bounds in
//! [`super::row_bounds`], or [`super::PlanFacts`], and they must not become
//! a capability, admission, or grant input.

/// One advisory quantity with the frontend's own precision claim.
///
/// `Absent` is a distinct state from zero so consumers can tell "no evidence"
/// from "estimated empty".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AdvisoryStatistic {
    /// The frontend claims the value is exact for its plan.
    Exact(u64),
    /// The frontend derived the value from estimation rules.
    Inexact(u64),
    /// The frontend had no estimate.
    Absent,
}

impl AdvisoryStatistic {
    /// Returns the estimated value regardless of precision.
    pub const fn value(self) -> Option<u64> {
        match self {
            Self::Exact(value) | Self::Inexact(value) => Some(value),
            Self::Absent => None,
        }
    }
}

/// Who produced an advisory statistic and how far it can be trusted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AdvisoryStatisticsProvenance {
    /// The embedding frontend's planner combined its source statistics with
    /// its own operator estimation rules. The value was never checked against
    /// native execution and can be wrong by orders of magnitude under key
    /// skew or correlated predicates.
    FrontendPlannerEstimate,
}

/// Advisory row and byte estimates for one node's whole output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AdvisoryOutputStatistics {
    /// Estimated output rows.
    pub rows: AdvisoryStatistic,
    /// Estimated output bytes in the frontend's own byte model, which is not
    /// device residency.
    pub bytes: AdvisoryStatistic,
    /// Origin of both estimates.
    pub provenance: AdvisoryStatisticsProvenance,
}

impl AdvisoryOutputStatistics {
    /// Returns true when neither rows nor bytes carry a value.
    pub const fn is_absent(self) -> bool {
        matches!((self.rows, self.bytes), (AdvisoryStatistic::Absent, AdvisoryStatistic::Absent))
    }
}

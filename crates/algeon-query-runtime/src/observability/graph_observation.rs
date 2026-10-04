//! Measured graph-execution observation.
//!
//! These rows record what a graph attempt actually did on the device. They
//! carry no predicted or budgeted bytes: the allocator enforces a query's
//! limit, so a planning-time claim would be an estimate masquerading as a
//! fact. Keeping only the measurement leaves future capacity decisions with
//! evidence instead of a model to calibrate.

use query_planning::{correlation::QueryId, plan::GraphAlgorithm};
use serde_json::{Value, json};

/// Direct source for one graph attempt's measured device-memory peak.
///
/// Consumers must not infer a source from a missing value or an error string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphMeasuredPeakProvenance {
    /// An exclusive resident-graph cache scope reported its own allocator
    /// high-water mark after the warm graph execution had retired. This is
    /// the only physical peak source for work that runs on the cache runtime
    /// rather than the admitted attempt runtime.
    CacheScopeHighWater,
    /// Neither runtime measurement completed. `measured_peak_bytes` is
    /// unavailable rather than a synthetic zero.
    Unavailable,
}

impl GraphMeasuredPeakProvenance {
    /// Stable label used by bounded public projections.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CacheScopeHighWater => "cache_scope_high_water",
            Self::Unavailable => "unavailable",
        }
    }
}

impl std::fmt::Display for GraphMeasuredPeakProvenance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Engine-owned lifecycle state at the graph executor boundary.
///
/// A row exists before the executor is entered, so a missing terminal state is
/// deliberately fail-closed and is not execution evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphClaimExecutionState {
    /// The row was recorded but no exact terminal graph boundary was observed.
    Pending,
    /// The native graph algorithm executor was entered.
    AlgorithmStarted,
}

impl GraphClaimExecutionState {
    /// Stable label used by bounded public projections.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::AlgorithmStarted => "algorithm_started",
        }
    }
}

impl std::fmt::Display for GraphClaimExecutionState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Fixed-shape measured-peak observation for one graph attempt.
///
/// The engine creates one of these before it enters the graph executor and
/// fills the runtime measurement when that attempt leaves it. It is
/// diagnostic only: the row cannot mint admission evidence or modify a device
/// grant. `node_index` is typed executor identity; report consumers should use
/// `node_id`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphMeasuredPeakRow {
    /// Query-wide diagnostic correlation id, when the execution context has
    /// one. The runtime metrics owner assigns this sequence.
    pub query_id: Option<QueryId>,
    /// Stable native graph node id for reports.
    pub node_id: String,
    /// Typed graph node identity used to correlate the post-execution sample
    /// without parsing a report string.
    pub node_index: usize,
    /// The graph algorithm that owned this measured peak.
    pub algorithm: GraphAlgorithm,
    /// Exact engine state at the graph executor boundary.
    pub execution_state: GraphClaimExecutionState,
    /// Measured graph-stage peak bytes when a runtime observation completed.
    pub measured_peak_bytes: Option<u64>,
    /// Source of [`Self::measured_peak_bytes`].
    pub measured_peak_provenance: GraphMeasuredPeakProvenance,
}

impl GraphMeasuredPeakRow {
    /// Opens a pending observation for one graph plan node.
    pub fn pending(
        query_id: Option<QueryId>,
        node_id: impl Into<String>,
        node_index: usize,
        algorithm: GraphAlgorithm,
    ) -> Self {
        Self {
            query_id,
            node_id: node_id.into(),
            node_index,
            algorithm,
            execution_state: GraphClaimExecutionState::Pending,
            measured_peak_bytes: None,
            measured_peak_provenance: GraphMeasuredPeakProvenance::Unavailable,
        }
    }

    /// Records one terminal graph-boundary state. A later observation cannot
    /// overwrite the first exact boundary.
    #[doc(hidden)]
    pub fn record_execution_state(&mut self, execution_state: GraphClaimExecutionState) {
        if self.execution_state == GraphClaimExecutionState::Pending {
            self.execution_state = execution_state;
        }
    }

    /// Records the one direct peak observation for this graph attempt.
    ///
    /// An unavailable source always clears the measured value. A zero
    /// high-water mark remains an observed zero.
    pub fn record_measured_peak(
        &mut self,
        measured_peak_bytes: Option<u64>,
        measured_peak_provenance: GraphMeasuredPeakProvenance,
    ) {
        match measured_peak_provenance {
            GraphMeasuredPeakProvenance::Unavailable => self.measured_peak_bytes = None,
            GraphMeasuredPeakProvenance::CacheScopeHighWater => self.measured_peak_bytes = measured_peak_bytes,
        }
        self.measured_peak_provenance = measured_peak_provenance;
    }

    /// Serializes this measured-peak row as named JSON.
    pub fn to_json(&self) -> Value {
        json!({
            "query_id": self.query_id.as_ref().map(QueryId::as_str),
            "node_id": self.node_id,
            "node_index": self.node_index,
            "algorithm": self.algorithm.as_str(),
            "execution_state": self.execution_state.as_str(),
            "measured_peak_bytes": self.measured_peak_bytes,
            "measured_peak_provenance": self.measured_peak_provenance.as_str(),
        })
    }
}

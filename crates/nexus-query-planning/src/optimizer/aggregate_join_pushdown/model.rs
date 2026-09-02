use super::{NodeId, QueryPlan};

/// Stable rule name used in optimizer diagnostics.
pub const AGGREGATE_JOIN_PUSHDOWN_RULE_NAME: &str = "aggregate_join_pushdown";

/// Configuration for reducing aggregates before joins.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AggregateJoinPushdownOptions {
    /// Whether the pass is allowed to rewrite eligible aggregate-over-join shapes.
    pub enabled: bool,
}

impl AggregateJoinPushdownOptions {
    /// Returns options with the pass disabled.
    pub fn disabled() -> Self {
        Self { enabled: false }
    }
}

impl Default for AggregateJoinPushdownOptions {
    fn default() -> Self {
        Self { enabled: true }
    }
}

/// Diagnostics for aggregate-join pushdown candidates.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AggregateJoinPushdownDiagnostics {
    /// Number of candidates considered.
    pub considered_count: usize,
    /// Number of candidates rewritten.
    pub applied_count: usize,
    /// Per-candidate evidence in discovery order.
    pub candidates: Vec<AggregateJoinPushdownCandidateDiagnostics>,
}

impl AggregateJoinPushdownDiagnostics {
    /// Returns true when the pass produced no candidate evidence.
    pub fn is_empty(&self) -> bool {
        self.considered_count == 0 && self.applied_count == 0 && self.candidates.is_empty()
    }

    /// Merges another diagnostics value into this one.
    pub fn merge(&mut self, other: Self) {
        self.considered_count += other.considered_count;
        self.applied_count += other.applied_count;
        self.candidates.extend(other.candidates);
    }
}

/// Per-candidate evidence for aggregate-join pushdown.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AggregateJoinPushdownCandidateDiagnostics {
    /// Stable rule name for this diagnostic row.
    pub rule_name: &'static str,
    /// Aggregate node considered for pushdown.
    pub aggregate_node_id: NodeId,
    /// Join node directly below the aggregate.
    pub join_node_id: Option<NodeId>,
    /// True when the returned plan contains the rewrite.
    pub applied: bool,
    /// Rejection reason when no rewrite was applied.
    pub rejected_reason: Option<AggregateJoinPushdownRejectedReason>,
}

/// Reasons an aggregate-join pushdown candidate was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregateJoinPushdownRejectedReason {
    /// The aggregate input is not a join.
    InputNotJoin,
    /// The join type is not currently supported by this pass.
    UnsupportedJoinType,
    /// The join has a residual predicate.
    JoinResidualUnsupported,
    /// Retired: the aggregate is not a grouping-set/rollup aggregate.
    GroupingSetsRequired,
    /// The aggregate contains an unsupported function, DISTINCT, filtered, or non-column expression.
    UnsupportedAggregateExpression,
    /// The aggregate grouping keys are not plain columns.
    UnsupportedGroupExpression,
    /// Aggregate input columns could not be mapped through the join output.
    MissingJoinOutputLineage,
    /// The eligible columns span sides in a pattern not covered by the rewrite.
    UnsupportedSidePattern,
    /// The pushed aggregate would require a missing input field.
    MissingInputField,
    /// Two required pre-aggregate columns would produce the same output name.
    ConflictingPreAggregateOutputName,
    /// A simple inner-join pre-aggregate matched the large fixed-width detail-risk guard.
    DetailKeyPreAggregateReductionNotEvidenced,
}

#[doc(hidden)]
pub struct AggregateJoinPushdownResult {
    pub plan: QueryPlan,
    pub change: crate::optimizer::NativeOptimizerChange,
    pub diagnostics: AggregateJoinPushdownDiagnostics,
}

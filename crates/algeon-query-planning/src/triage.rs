//! Shared triage vocabulary and compact diagnostic summaries.
//!
//! Planning, the runtime, and the adapter report all emit the same codes,
//! verdicts, and severities, so one JSON consumer can read a failure from any
//! of them. Every `as_str` label is a wire contract; the mapping from
//! [`QueryErrorCode`] to [`TriageCode`] is owned by the error module.

use serde_json::{Value, json};

use crate::{
    correlation::{CompiledTaskId, PlanNodeId, QueryId},
    error::{QueryError, QueryErrorCode, triage_code_for_query_error},
};

/// Shared symptom/cause code used by planning, runtime, and report JSON.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TriageCode {
    /// Execution ended after a cancellation request.
    Cancelled,
    /// A supported native candidate was left on DataFusion by cost policy.
    NotSelectedByCost,
    /// The physical plan shape cannot be lowered as a native candidate.
    UnsupportedPlanShape,
    /// An operation has no supported native implementation.
    UnsupportedOperator,
    /// An expression cannot be evaluated by the native backend.
    UnsupportedExpression,
    /// A dtype or physical representation is unsupported.
    UnsupportedType,
    /// The source cannot satisfy the native reader contract.
    UnsupportedSource,
    /// Caller-supplied data or configuration violates a required constraint.
    InvalidInput,
    /// The plan violates an execution or IR invariant.
    InvalidPlan,
    /// Capability validation prevents bounded native execution.
    CapabilityEnvelopeRejected,
    /// Admission could not reserve the requested memory capacity.
    MemoryReservationDenied,
    /// Execution exhausted an enforced or available resource.
    ResourceExhausted,
    /// A bounded service queue or execution capacity is saturated.
    ServiceOverloaded,
    /// Source access or reading failed.
    SourceFailure,
    /// A native or external dependency reported a failure.
    DependencyFailure,
    /// Execution violated an ownership, provenance or consistency invariant.
    InvariantViolation,
    /// An internal failure has no more specific triage classification.
    Internal,
}

impl TriageCode {
    /// Stable snake-case triage code.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cancelled => "cancelled",
            Self::NotSelectedByCost => "not_selected_by_cost",
            Self::UnsupportedPlanShape => "unsupported_plan_shape",
            Self::UnsupportedOperator => "unsupported_operator",
            Self::UnsupportedExpression => "unsupported_expression",
            Self::UnsupportedType => "unsupported_type",
            Self::UnsupportedSource => "unsupported_source",
            Self::InvalidInput => "invalid_input",
            Self::InvalidPlan => "invalid_plan",
            // Compatibility wire label retained for existing report consumers.
            Self::CapabilityEnvelopeRejected => "bounded_rejected",
            Self::MemoryReservationDenied => "memory_reservation_denied",
            Self::ResourceExhausted => "resource_exhausted",
            Self::ServiceOverloaded => "service_overloaded",
            Self::SourceFailure => "source_failure",
            Self::DependencyFailure => "dependency_failure",
            Self::InvariantViolation => "invariant_violation",
            Self::Internal => "internal",
        }
    }
}

impl std::fmt::Display for TriageCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<QueryErrorCode> for TriageCode {
    fn from(code: QueryErrorCode) -> Self {
        triage_code_for_query_error(code)
    }
}

/// Query-level triage verdict.
///
/// `Failed` is reserved for a query that returned an error. A query that
/// completed with warning signals is `Degraded`. A native candidate that ran on
/// the CPU by cost is `NotSelected`, and that choice does not count as a
/// failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TriageVerdict {
    /// The query completed without warning signals.
    Healthy,
    /// The query completed with warning signals.
    Degraded,
    /// The query returned an error.
    Failed,
    /// Cost policy retained CPU execution for a native candidate.
    NotSelected,
}

impl TriageVerdict {
    /// Stable snake-case verdict label.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Healthy => "healthy",
            Self::Degraded => "degraded",
            Self::Failed => "failed",
            Self::NotSelected => "not_selected",
        }
    }
}

impl std::fmt::Display for TriageVerdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Triage signal severity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TriageSeverity {
    /// Informational evidence that does not itself indicate a failure.
    Info,
    /// A condition requiring attention without a query error.
    Warning,
    /// A signal associated with a failed query.
    Error,
}

impl TriageSeverity {
    /// Stable snake-case severity label.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Info => "info",
            Self::Warning => "warning",
            Self::Error => "error",
        }
    }
}

impl std::fmt::Display for TriageSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// JSON evidence pointer into one diagnostic surface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TriageEvidenceRef {
    pub(crate) query_id: Option<QueryId>,
    pub(crate) plan_node_id: Option<PlanNodeId>,
    pub(crate) compiled_task_id: Option<CompiledTaskId>,
    pub(crate) surface: &'static str,
    pub(crate) row_index: Option<usize>,
}

impl TriageEvidenceRef {
    /// Builds an evidence reference for a named diagnostic surface.
    pub fn new(surface: &'static str) -> Self {
        Self { query_id: None, plan_node_id: None, compiled_task_id: None, surface, row_index: None }
    }

    /// Adds a query id.
    pub fn with_query_id(mut self, query_id: impl Into<QueryId>) -> Self {
        self.query_id = Some(query_id.into());
        self
    }

    /// Adds a plan node id.
    pub fn with_plan_node_id(mut self, plan_node_id: impl Into<PlanNodeId>) -> Self {
        self.plan_node_id = Some(plan_node_id.into());
        self
    }

    pub(crate) fn with_compiled_task_id(mut self, compiled_task_id: impl Into<CompiledTaskId>) -> Self {
        self.compiled_task_id = Some(compiled_task_id.into());
        self
    }

    /// Adds a row index inside the named surface.
    pub fn with_row_index(mut self, row_index: usize) -> Self {
        self.row_index = Some(row_index);
        self
    }

    /// Serializes the evidence reference as named JSON.
    pub(crate) fn to_json(&self) -> Value {
        json!({
            "query_id": self.query_id.as_ref().map(QueryId::as_str),
            "plan_node_id": self.plan_node_id.as_ref().map(PlanNodeId::as_str),
            "compiled_task_id": self.compiled_task_id.as_ref().map(CompiledTaskId::as_str),
            "surface": self.surface,
            "row_index": self.row_index,
        })
    }
}

/// One ranked triage signal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TriageSignal {
    pub(crate) severity: TriageSeverity,
    /// Machine-readable cause shared across planning and runtime diagnostics.
    pub code: TriageCode,
    pub(crate) message: String,
    pub(crate) evidence: Option<TriageEvidenceRef>,
}

impl TriageSignal {
    /// Builds a triage signal.
    pub fn new(severity: TriageSeverity, code: TriageCode, message: impl Into<String>) -> Self {
        Self { severity, code, message: message.into(), evidence: None }
    }

    /// Adds an evidence pointer.
    pub fn with_evidence(mut self, evidence: TriageEvidenceRef) -> Self {
        self.evidence = Some(evidence);
        self
    }

    /// Serializes this signal as named JSON.
    pub(crate) fn to_json(&self) -> Value {
        json!({
            "severity": self.severity.as_str(),
            "code": self.code.as_str(),
            "message": self.message,
            "evidence": self.evidence.as_ref().map(TriageEvidenceRef::to_json),
        })
    }
}

/// Compact top-level diagnostic summary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TriageSummary {
    /// Overall query outcome represented by this summary.
    pub verdict: TriageVerdict,
    /// Ordered diagnostic signals contributing to the summary.
    pub signals: Vec<TriageSignal>,
}

impl TriageSummary {
    /// Builds a summary from an explicit verdict and signal list.
    ///
    /// Signals are ordered by descending severity, and the sort is stable, so
    /// signals of equal severity keep the order the caller supplied.
    pub fn new(verdict: TriageVerdict, mut signals: Vec<TriageSignal>) -> Self {
        signals.sort_by_key(|signal| std::cmp::Reverse(signal.severity));
        Self { verdict, signals }
    }

    /// Builds a healthy empty summary.
    pub fn healthy() -> Self {
        Self::new(TriageVerdict::Healthy, Vec::new())
    }

    /// Serializes this summary as named JSON.
    pub fn to_json(&self) -> Value {
        json!({
            "verdict": self.verdict.as_str(),
            "signals": self.signals.iter().map(TriageSignal::to_json).collect::<Vec<_>>(),
        })
    }
}

impl QueryError {
    /// Builds a compact triage summary for this error.
    ///
    /// Structured context takes precedence over facts. The error's own query id
    /// and boundary node id are used first, and the `query_id` and `node_id`
    /// facts fill in only when the context lacks them. The `compiled_task_id`
    /// fact is used whenever it is present.
    pub fn triage_summary(&self) -> TriageSummary {
        let mut evidence = TriageEvidenceRef::new("query_error");
        if let Some(query_id) = self.query_id().cloned().or_else(|| self.fact_value("query_id").map(QueryId::new)) {
            evidence = evidence.with_query_id(query_id);
        }
        if let Some(node_id) = self.context().boundary.and_then(|boundary| boundary.node_id) {
            evidence = evidence.with_plan_node_id(node_id);
        } else if let Some(node_id) = self.fact_value("node_id") {
            evidence = evidence.with_plan_node_id(node_id);
        }
        if let Some(compiled_task_id) = self.fact_value("compiled_task_id") {
            evidence = evidence.with_compiled_task_id(compiled_task_id);
        }
        let signal = TriageSignal::new(TriageSeverity::Error, self.code().into(), self.message().to_string())
            .with_evidence(evidence);
        TriageSummary::new(TriageVerdict::Failed, vec![signal])
    }
}

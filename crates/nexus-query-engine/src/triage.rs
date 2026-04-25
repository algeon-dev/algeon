//! Shared triage vocabulary and compact diagnostic summaries.

use serde_json::{Value, json};

use crate::{
    adaptive_report::{AdaptiveCandidateStatus, AdaptivePipelineReport},
    capability::EdgeConsumer,
    contract_report::{StrictBoundedContractOwner, StrictBoundedContractReport},
    error::{QueryError, QueryErrorCode},
    metrics::{CompiledTaskId, PlanNodeId, QueryId, QueryMetrics},
};

/// Shared symptom/cause code used by planning, runtime, and report JSON.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TriageCode {
    Cancelled,
    ExplicitCpuPolicy,
    CpuPreferred,
    UnsupportedPlanShape,
    UnsupportedOperator,
    UnsupportedExpression,
    UnsupportedType,
    UnsupportedSource,
    InvalidInput,
    InvalidPlan,
    StrictBoundedRejected,
    AdaptivePipelineRejected,
    MemoryReservationDenied,
    ResourceExhausted,
    ServiceOverloaded,
    SourceFailure,
    DependencyFailure,
    InvariantViolation,
    AdaptiveRetry,
    AdaptiveSpill,
    AdaptiveDowngrade,
    Internal,
}

impl TriageCode {
    /// Stable snake-case triage code.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cancelled => "cancelled",
            Self::ExplicitCpuPolicy => "explicit_cpu_policy",
            Self::CpuPreferred => "cpu_preferred",
            Self::UnsupportedPlanShape => "unsupported_plan_shape",
            Self::UnsupportedOperator => "unsupported_operator",
            Self::UnsupportedExpression => "unsupported_expression",
            Self::UnsupportedType => "unsupported_type",
            Self::UnsupportedSource => "unsupported_source",
            Self::InvalidInput => "invalid_input",
            Self::InvalidPlan => "invalid_plan",
            Self::StrictBoundedRejected => "strict_bounded_rejected",
            Self::AdaptivePipelineRejected => "adaptive_pipeline_rejected",
            Self::MemoryReservationDenied => "memory_reservation_denied",
            Self::ResourceExhausted => "resource_exhausted",
            Self::ServiceOverloaded => "service_overloaded",
            Self::SourceFailure => "source_failure",
            Self::DependencyFailure => "dependency_failure",
            Self::InvariantViolation => "invariant_violation",
            Self::AdaptiveRetry => "adaptive_retry",
            Self::AdaptiveSpill => "adaptive_spill",
            Self::AdaptiveDowngrade => "adaptive_downgrade",
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
        match code {
            QueryErrorCode::Cancelled => Self::Cancelled,
            QueryErrorCode::UnsupportedPlanShape => Self::UnsupportedPlanShape,
            QueryErrorCode::UnsupportedOperator => Self::UnsupportedOperator,
            QueryErrorCode::UnsupportedExpression => Self::UnsupportedExpression,
            QueryErrorCode::UnsupportedType => Self::UnsupportedType,
            QueryErrorCode::CudfUnsupported => Self::UnsupportedPlanShape,
            QueryErrorCode::InvalidPlan => Self::InvalidPlan,
            QueryErrorCode::CudfInvalidInput => Self::InvalidInput,
            QueryErrorCode::MemoryReservationFailed => Self::MemoryReservationDenied,
            QueryErrorCode::CudfResourceExhausted
            | QueryErrorCode::SizeOverflow
            | QueryErrorCode::HostMaterializationLimitExceeded => Self::ResourceExhausted,
            QueryErrorCode::ServiceOverloaded => Self::ServiceOverloaded,
            QueryErrorCode::SourceAccessDenied
            | QueryErrorCode::SourceUnauthenticated
            | QueryErrorCode::SourceObjectNotFound
            | QueryErrorCode::SourceTemporaryFailure => Self::SourceFailure,
            QueryErrorCode::CudfTemporaryFailure
            | QueryErrorCode::CudfPersistentFailure
            | QueryErrorCode::CudfDependencyFailure => Self::DependencyFailure,
            QueryErrorCode::CudfInvariantViolation => Self::InvariantViolation,
            QueryErrorCode::Internal => Self::Internal,
        }
    }
}

/// Query-level triage verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TriageVerdict {
    Healthy,
    Degraded,
    Failed,
    FellBack,
}

impl TriageVerdict {
    /// Stable snake-case verdict label.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Healthy => "healthy",
            Self::Degraded => "degraded",
            Self::Failed => "failed",
            Self::FellBack => "fell_back",
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
    Info,
    Warning,
    Error,
}

impl TriageSeverity {
    /// Stable snake-case severity label.
    pub const fn as_str(self) -> &'static str {
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
    pub query_id: Option<QueryId>,
    pub plan_node_id: Option<PlanNodeId>,
    pub compiled_task_id: Option<CompiledTaskId>,
    pub surface: &'static str,
    pub row_index: Option<usize>,
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

    /// Adds a compiled task id.
    pub fn with_compiled_task_id(mut self, compiled_task_id: impl Into<CompiledTaskId>) -> Self {
        self.compiled_task_id = Some(compiled_task_id.into());
        self
    }

    /// Adds a row index inside the named surface.
    pub fn with_row_index(mut self, row_index: usize) -> Self {
        self.row_index = Some(row_index);
        self
    }

    /// Serializes the evidence reference as named JSON.
    pub fn to_json(&self) -> Value {
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
    pub severity: TriageSeverity,
    pub code: TriageCode,
    pub message: String,
    pub evidence: Option<TriageEvidenceRef>,
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
    pub fn to_json(&self) -> Value {
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
    pub verdict: TriageVerdict,
    pub signals: Vec<TriageSignal>,
}

impl TriageSummary {
    /// Builds a summary from an explicit verdict and signal list.
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

impl QueryMetrics {
    /// Builds a compact triage summary from runtime metrics.
    pub fn triage_summary(&self) -> TriageSummary {
        let mut signals = Vec::new();
        let query_id = self.query_id.is_assigned().then(|| self.query_id.clone());

        if self.reservation_denied_count > 0 {
            let evidence =
                self.reservation_events.iter().enumerate().find(|(_, event)| event.outcome.as_str() == "denied").map(
                    |(index, event)| {
                        let mut evidence =
                            TriageEvidenceRef::new("query_metrics.rows.memory_reservations").with_row_index(index);
                        if let Some(query_id) = event.query_id.clone().or_else(|| query_id.clone()) {
                            evidence = evidence.with_query_id(query_id);
                        }
                        if let Some(plan_node_id) = event.node_id.clone() {
                            evidence = evidence.with_plan_node_id(plan_node_id);
                        }
                        if let Some(compiled_task_id) = event.compiled_task_id.clone() {
                            evidence = evidence.with_compiled_task_id(compiled_task_id);
                        }
                        evidence
                    },
                );
            signals.push(maybe_with_evidence(
                TriageSignal::new(
                    TriageSeverity::Warning,
                    TriageCode::MemoryReservationDenied,
                    format!("{} memory reservation request(s) were denied", self.reservation_denied_count),
                ),
                evidence,
            ));
        }

        if self.retry_resource_exhausted_count > 0 {
            signals.push(TriageSignal::new(
                TriageSeverity::Warning,
                TriageCode::ResourceExhausted,
                format!("{} retry attempt(s) were caused by resource exhaustion", self.retry_resource_exhausted_count),
            ));
        }
        if self.retry_reservation_failure_count > 0 {
            signals.push(TriageSignal::new(
                TriageSeverity::Warning,
                TriageCode::MemoryReservationDenied,
                format!("{} retry attempt(s) were caused by reservation failure", self.retry_reservation_failure_count),
            ));
        }
        if self.adaptive_retry_count > 0 {
            signals.push(TriageSignal::new(
                TriageSeverity::Warning,
                TriageCode::AdaptiveRetry,
                format!("{} adaptive retry action(s) were recorded", self.adaptive_retry_count),
            ));
        }
        if self.adaptive_spill_count > 0 {
            signals.push(TriageSignal::new(
                TriageSeverity::Warning,
                TriageCode::AdaptiveSpill,
                format!("{} adaptive spill action(s) were recorded", self.adaptive_spill_count),
            ));
        }
        if self.adaptive_downgrade_count > 0 {
            signals.push(TriageSignal::new(
                TriageSeverity::Warning,
                TriageCode::AdaptiveDowngrade,
                format!("{} adaptive downgrade action(s) were recorded", self.adaptive_downgrade_count),
            ));
        }

        let verdict = if signals.is_empty() { TriageVerdict::Healthy } else { TriageVerdict::Degraded };
        TriageSummary::new(verdict, signals)
    }
}

impl QueryError {
    /// Builds a compact triage summary for this error.
    pub fn triage_summary(&self) -> TriageSummary {
        let mut evidence = TriageEvidenceRef::new("query_error");
        if let Some(query_id) = self.query_id().cloned().or_else(|| self.fact_value("query_id").map(QueryId::new)) {
            evidence = evidence.with_query_id(query_id);
        }
        if let Some(node_id) = self.fact_value("node_id") {
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

impl StrictBoundedContractReport {
    /// Builds a compact triage summary for strict bounded readiness.
    pub fn triage_summary(&self) -> TriageSummary {
        if self.is_strict_bounded_ready() {
            return TriageSummary::healthy();
        }
        let query_id = self.query_id().cloned();
        let signals = self
            .gate_rows()
            .iter()
            .enumerate()
            .filter(|(_, row)| row.contract_status.as_str() != "modeled")
            .map(|(index, row)| {
                let mut evidence =
                    TriageEvidenceRef::new("strict_bounded_contract_report.gate_rows").with_row_index(index);
                if let Some(query_id) = query_id.clone() {
                    evidence = evidence.with_query_id(query_id);
                }
                if let Some(plan_node_id) = strict_owner_plan_node_id(&row.owner) {
                    evidence = evidence.with_plan_node_id(plan_node_id);
                }
                TriageSignal::new(
                    TriageSeverity::Warning,
                    TriageCode::StrictBoundedRejected,
                    format!("strict bounded gate {} is {}", row.gate_id, row.contract_status.as_str()),
                )
                .with_evidence(evidence)
            })
            .collect::<Vec<_>>();
        TriageSummary::new(TriageVerdict::FellBack, signals)
    }
}

impl AdaptivePipelineReport {
    /// Builds a compact triage summary for adaptive pipeline admission/runtime actions.
    pub fn triage_summary(&self) -> TriageSummary {
        let mut signals = Vec::new();
        let row = self.row();
        let query_id = self.query_id().cloned();
        if row.candidate_status == AdaptiveCandidateStatus::Rejected {
            let mut evidence = TriageEvidenceRef::new("adaptive_pipeline_report.row");
            if let Some(query_id) = query_id.clone() {
                evidence = evidence.with_query_id(query_id);
            }
            if let Some(plan_node_id) =
                row.missing_strict_contracts.iter().find_map(|missing| adaptive_owner_plan_node_id(&missing.owner))
            {
                evidence = evidence.with_plan_node_id(plan_node_id);
            }
            signals.push(
                TriageSignal::new(
                    TriageSeverity::Warning,
                    TriageCode::AdaptivePipelineRejected,
                    row.rejection_reason.unwrap_or("adaptive pipeline candidate was rejected"),
                )
                .with_evidence(evidence),
            );
        }
        if row.runtime_action_counts.retry_count > 0 {
            signals.push(maybe_with_evidence(
                TriageSignal::new(
                    TriageSeverity::Warning,
                    TriageCode::AdaptiveRetry,
                    format!("{} adaptive retry action(s) were recorded", row.runtime_action_counts.retry_count),
                ),
                adaptive_row_evidence(query_id.clone()),
            ));
        }
        if row.runtime_action_counts.spill_count > 0 {
            signals.push(maybe_with_evidence(
                TriageSignal::new(
                    TriageSeverity::Warning,
                    TriageCode::AdaptiveSpill,
                    format!("{} adaptive spill action(s) were recorded", row.runtime_action_counts.spill_count),
                ),
                adaptive_row_evidence(query_id.clone()),
            ));
        }
        if row.runtime_action_counts.downgrade_count > 0 {
            signals.push(maybe_with_evidence(
                TriageSignal::new(
                    TriageSeverity::Warning,
                    TriageCode::AdaptiveDowngrade,
                    format!("{} adaptive downgrade action(s) were recorded", row.runtime_action_counts.downgrade_count),
                ),
                adaptive_row_evidence(query_id.clone()),
            ));
        }

        let verdict = if row.candidate_status == AdaptiveCandidateStatus::Rejected {
            TriageVerdict::FellBack
        } else if signals.is_empty() {
            TriageVerdict::Healthy
        } else {
            TriageVerdict::Degraded
        };
        TriageSummary::new(verdict, signals)
    }
}

fn maybe_with_evidence(signal: TriageSignal, evidence: Option<TriageEvidenceRef>) -> TriageSignal {
    match evidence {
        Some(evidence) => signal.with_evidence(evidence),
        None => signal,
    }
}

fn adaptive_row_evidence(query_id: Option<QueryId>) -> Option<TriageEvidenceRef> {
    let mut evidence = TriageEvidenceRef::new("adaptive_pipeline_report.row");
    if let Some(query_id) = query_id {
        evidence = evidence.with_query_id(query_id);
    }
    Some(evidence)
}

fn strict_owner_plan_node_id(owner: &StrictBoundedContractOwner) -> Option<PlanNodeId> {
    match owner {
        StrictBoundedContractOwner::CapabilityNode(node) => Some((*node).into()),
        StrictBoundedContractOwner::CapabilityEdge { producer, consumer } => match consumer {
            EdgeConsumer::Node(node) => Some((*node).into()),
            EdgeConsumer::Sink(_) => Some((*producer).into()),
        },
        StrictBoundedContractOwner::Compiled(_)
        | StrictBoundedContractOwner::CapabilitySubgraph(_)
        | StrictBoundedContractOwner::CapabilityAdmission(_) => None,
    }
}

fn adaptive_owner_plan_node_id(owner: &str) -> Option<PlanNodeId> {
    if owner.starts_with('n') {
        return Some(PlanNodeId::new(owner));
    }
    let edge = owner.strip_prefix("edge:")?;
    let producer = edge.split("->").next()?;
    producer.starts_with('n').then(|| PlanNodeId::new(producer))
}

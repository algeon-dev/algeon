//! Capability admission decisions exposed to diagnostics.

use std::{collections::BTreeMap, fmt};

use serde_json::{Value, json};

use crate::{
    capability::{
        CapabilityAnalysis, CapabilityEnvelopeAdmission, CapabilityRejectionCategory, CapabilityRejectionOwner,
        CapabilityRejectionProjection, CapabilityScope, EdgeConsumer, InvariantId, NodeOperator, ResultSinkKind,
    },
    correlation::{QueryId, TraceId},
    error::{QueryError, QueryErrorCode, Result},
    mode::CapabilityEnvelopeStatus,
    pipeline::CompiledPipeline,
    plan::NodeId,
};

/// Query-level capability admission report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityContractReport {
    query_id: Option<QueryId>,
    trace_id: Option<TraceId>,
    execution_mode: CapabilityEnvelopeStatus,
    decision_rows: Vec<CapabilityDecisionRow>,
}

impl CapabilityContractReport {
    /// Builds a report from explicit non-admitted capability decisions.
    pub fn from_decision_rows(
        execution_mode: CapabilityEnvelopeStatus,
        decision_rows: Vec<CapabilityDecisionRow>,
    ) -> Result<Self> {
        Self::new(execution_mode, decision_rows)
    }

    /// Builds a report after validating the compiled pipeline.
    pub fn from_compiled_pipeline(analysis: &CapabilityAnalysis, pipeline: &CompiledPipeline) -> Result<Self> {
        pipeline.validate()?;
        Self::from_capability_analysis(analysis)
    }

    /// Builds a report from capability analysis.
    pub fn from_capability_analysis(analysis: &CapabilityAnalysis) -> Result<Self> {
        let execution_mode = capability_envelope_status_for_analysis(analysis);
        let decision_rows = if execution_mode == CapabilityEnvelopeStatus::Rejected {
            analysis
                .first_rejection_projection()
                .map(|projection| CapabilityDecisionRow::from_rejection_projection(0, &projection))
                .transpose()?
                .into_iter()
                .collect()
        } else {
            Vec::new()
        };
        Self::new(execution_mode, decision_rows)
    }

    /// Returns the selected capability-envelope status.
    pub fn execution_mode(&self) -> CapabilityEnvelopeStatus {
        self.execution_mode
    }

    /// Returns non-admitted decisions in stable report order.
    pub fn decision_rows(&self) -> &[CapabilityDecisionRow] {
        &self.decision_rows
    }

    /// Returns whether the plan is ready for bounded execution.
    pub fn is_ready(&self) -> bool {
        self.execution_mode == CapabilityEnvelopeStatus::Admitted
    }

    /// Returns the query-wide diagnostic correlation id, when assigned.
    pub fn query_id(&self) -> Option<&QueryId> {
        self.query_id.as_ref()
    }

    /// Returns the optional parent trace id.
    pub fn trace_id(&self) -> Option<&TraceId> {
        self.trace_id.as_ref()
    }

    /// Returns this report with a query id assigned.
    #[must_use]
    pub fn with_query_id(mut self, query_id: impl Into<QueryId>) -> Self {
        self.query_id = Some(query_id.into());
        self
    }

    /// Returns this report with a parent trace id assigned.
    #[must_use]
    pub fn with_trace_id(mut self, trace_id: impl Into<TraceId>) -> Self {
        self.trace_id = Some(trace_id.into());
        self
    }

    /// Serializes each decision as stable comma-separated key/value pairs.
    pub fn stable_kv_rows(&self) -> Vec<String> {
        self.decision_rows.iter().map(CapabilityDecisionRow::to_stable_kv_row).collect()
    }

    /// Serializes the report.
    pub fn to_json(&self) -> Value {
        json!({
            "schema": "nexus_query_engine.capability_report.v2",
            "query_id": self.query_id.as_ref().map(QueryId::as_str),
            "trace_id": self.trace_id.as_ref().map(TraceId::as_str),
            "triage": self.triage_summary().to_json(),
            "execution_mode": self.execution_mode.as_str(),
            "capability_ready": self.is_ready(),
            "decision_rows": self.decision_rows.iter().map(CapabilityDecisionRow::to_json).collect::<Vec<_>>(),
        })
    }

    fn new(execution_mode: CapabilityEnvelopeStatus, decision_rows: Vec<CapabilityDecisionRow>) -> Result<Self> {
        for row in &decision_rows {
            validate_decision_row(row)?;
            if row.execution_mode != execution_mode {
                return Err(capability_report_error("capability decision has inconsistent envelope status")
                    .with_fact("row_execution_mode", row.execution_mode.as_str())
                    .with_fact("report_execution_mode", execution_mode.as_str())
                    .with_fact("decision_id", row.decision_id.to_string()));
            }
        }
        validate_report_status(execution_mode, &decision_rows)?;
        Ok(Self { query_id: None, trace_id: None, execution_mode, decision_rows })
    }
}

fn capability_envelope_status_for_analysis(analysis: &CapabilityAnalysis) -> CapabilityEnvelopeStatus {
    match analysis.capability_envelope_admission() {
        CapabilityEnvelopeAdmission::Admitted { .. } => CapabilityEnvelopeStatus::Admitted,
        CapabilityEnvelopeAdmission::Rejected { .. }
            if !analysis.nodes.iter().any(|node| matches!(node.operator, NodeOperator::Join { .. }))
                && !analysis.has_rejections() =>
        {
            CapabilityEnvelopeStatus::Admitted
        },
        CapabilityEnvelopeAdmission::Rejected { .. } => CapabilityEnvelopeStatus::Rejected,
    }
}

/// One deferred or rejected capability decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityDecisionRow {
    pub execution_mode: CapabilityEnvelopeStatus,
    pub scope: CapabilityScope,
    pub decision_id: CapabilityDecisionId,
    pub owner: CapabilityDecisionOwner,
    pub category: Option<CapabilityRejectionCategory>,
    pub boundary: &'static str,
    pub reason: &'static str,
    pub status: CapabilityDecisionStatus,
    pub facts: BTreeMap<&'static str, String>,
}

impl CapabilityDecisionRow {
    /// Builds a row from a rejected capability projection.
    pub fn from_rejection_projection(index: usize, projection: &CapabilityRejectionProjection) -> Result<Self> {
        let row = Self {
            execution_mode: CapabilityEnvelopeStatus::Rejected,
            scope: projection.contract_scope,
            decision_id: CapabilityDecisionId { scope: projection.contract_scope, index },
            owner: CapabilityDecisionOwner::from_rejection_owner(&projection.owner),
            category: Some(projection.category),
            boundary: projection.boundary,
            reason: projection.reason,
            status: CapabilityDecisionStatus::Rejected,
            facts: projection.facts.clone(),
        };
        validate_decision_row(&row)?;
        Ok(row)
    }

    /// Builds the deferred decision for a composed consumer waiting for its
    /// admitted producer occurrence.
    pub fn deferred_composed_producer_binding(index: usize) -> Result<Self> {
        let row = Self {
            execution_mode: CapabilityEnvelopeStatus::Deferred,
            scope: CapabilityScope::Subgraph,
            decision_id: CapabilityDecisionId { scope: CapabilityScope::Subgraph, index },
            owner: CapabilityDecisionOwner::Admission("composed_producer_binding"),
            category: None,
            boundary: "composed_consumer",
            reason: "producer_occurrence_output_evidence",
            status: CapabilityDecisionStatus::Deferred,
            facts: BTreeMap::new(),
        };
        validate_decision_row(&row)?;
        Ok(row)
    }

    pub fn to_stable_kv_row(&self) -> String {
        format!(
            "execution_mode={},scope={},decision_id={},owner={},category={},boundary={},reason={},status={}",
            self.execution_mode.as_str(),
            self.scope.as_str(),
            self.decision_id,
            self.owner,
            self.category.map_or("none", CapabilityRejectionCategory::as_str),
            self.boundary,
            self.reason,
            self.status.as_str(),
        )
    }

    pub fn to_json(&self) -> Value {
        json!({
            "execution_mode": self.execution_mode.as_str(),
            "scope": self.scope.as_str(),
            "decision_id": self.decision_id.to_string(),
            "owner": self.owner.to_string(),
            "category": self.category.map(CapabilityRejectionCategory::as_str),
            "boundary": self.boundary,
            "reason": self.reason,
            "status": self.status.as_str(),
            "facts": self.facts,
        })
    }
}

impl fmt::Display for CapabilityDecisionRow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_stable_kv_row())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CapabilityDecisionId {
    pub scope: CapabilityScope,
    pub index: usize,
}

impl fmt::Display for CapabilityDecisionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "capability:{}:{}", self.scope.as_str(), self.index)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CapabilityDecisionStatus {
    Deferred,
    Rejected,
}

impl CapabilityDecisionStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Deferred => "deferred",
            Self::Rejected => "rejected",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CapabilityDecisionOwner {
    Node(NodeId),
    Edge { producer: NodeId, consumer: EdgeConsumer },
    Subgraph(InvariantId),
    Admission(&'static str),
}

impl CapabilityDecisionOwner {
    fn from_rejection_owner(owner: &CapabilityRejectionOwner) -> Self {
        match owner {
            CapabilityRejectionOwner::Node(node) => Self::Node(*node),
            CapabilityRejectionOwner::Edge { producer, consumer } => {
                Self::Edge { producer: *producer, consumer: *consumer }
            },
            CapabilityRejectionOwner::Invariant(invariant) => Self::Subgraph(*invariant),
            CapabilityRejectionOwner::Admission(admission) => Self::Admission(admission),
        }
    }
}

impl fmt::Display for CapabilityDecisionOwner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Node(node) => write!(f, "node:{node}"),
            Self::Edge { producer, consumer } => write!(f, "edge:{producer}->{}", DisplayEdgeConsumer(*consumer)),
            Self::Subgraph(invariant) => write!(f, "subgraph:i{}", invariant.index()),
            Self::Admission(admission) => write!(f, "admission:{admission}"),
        }
    }
}

struct DisplayEdgeConsumer(EdgeConsumer);

impl fmt::Display for DisplayEdgeConsumer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            EdgeConsumer::Node(node) => write!(f, "node:{node}"),
            EdgeConsumer::Sink(ResultSinkKind::HostArrow) => f.write_str("sink:host_arrow"),
            EdgeConsumer::Sink(ResultSinkKind::GpuResident) => f.write_str("sink:gpu_resident"),
        }
    }
}

fn validate_decision_row(row: &CapabilityDecisionRow) -> Result<()> {
    if row.boundary.is_empty() || row.reason.is_empty() {
        return Err(capability_report_error("capability decision requires a boundary and reason")
            .with_fact("decision_id", row.decision_id.to_string()));
    }
    let owner_scope_matches = matches!(
        (&row.owner, row.scope),
        (CapabilityDecisionOwner::Node(_), CapabilityScope::Node)
            | (CapabilityDecisionOwner::Edge { .. }, CapabilityScope::Edge)
            | (CapabilityDecisionOwner::Subgraph(_), CapabilityScope::Subgraph)
            | (CapabilityDecisionOwner::Admission(_), CapabilityScope::Subgraph)
    );
    if !owner_scope_matches {
        return Err(capability_report_error("capability decision owner does not match its scope")
            .with_fact("decision_id", row.decision_id.to_string())
            .with_fact("owner", row.owner.to_string())
            .with_fact("scope", row.scope.as_str()));
    }
    match (row.status, row.category) {
        (CapabilityDecisionStatus::Deferred, None) | (CapabilityDecisionStatus::Rejected, Some(_)) => Ok(()),
        (CapabilityDecisionStatus::Deferred, Some(_)) | (CapabilityDecisionStatus::Rejected, None) => {
            Err(capability_report_error("capability decision category does not match its status")
                .with_fact("decision_id", row.decision_id.to_string()))
        },
    }
}

fn validate_report_status(
    execution_mode: CapabilityEnvelopeStatus,
    decision_rows: &[CapabilityDecisionRow],
) -> Result<()> {
    let valid = match execution_mode {
        CapabilityEnvelopeStatus::Admitted => decision_rows.is_empty(),
        CapabilityEnvelopeStatus::Deferred => {
            !decision_rows.is_empty()
                && decision_rows.iter().all(|row| row.status == CapabilityDecisionStatus::Deferred)
        },
        CapabilityEnvelopeStatus::Rejected => {
            !decision_rows.is_empty()
                && decision_rows.iter().all(|row| row.status == CapabilityDecisionStatus::Rejected)
        },
    };
    if valid {
        Ok(())
    } else {
        Err(capability_report_error("capability report rows do not match the report status")
            .with_fact("report_execution_mode", execution_mode.as_str())
            .with_fact("decision_row_count", decision_rows.len()))
    }
}

fn capability_report_error(message: &'static str) -> QueryError {
    QueryError::new(QueryErrorCode::InvalidPlan, "nexus_query_engine.capability_report", message)
}

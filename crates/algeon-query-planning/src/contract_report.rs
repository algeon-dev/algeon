//! Capability admission decisions exposed to diagnostics.

use std::{collections::BTreeMap, fmt};

use crate::{
    admission::AdmittedPlan,
    capability::{CapabilityRejectionCategory, CapabilityScope, Rejection, RejectionOwner},
    correlation::{QueryId, TraceId},
    error::{QueryError, QueryErrorCode, Result},
    mode::CapabilityEnvelopeStatus,
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

    /// Builds a report from an admitted artifact.
    ///
    /// The artifact is admitted by construction: the report projects that
    /// status directly instead of re-deriving an envelope verdict. A pipeline
    /// admitted through the composed contracts selects no named envelope, so
    /// re-deriving would misreport its admission as a rejection.
    pub fn from_admission(_admission: &AdmittedPlan) -> Result<Self> {
        Self::new(CapabilityEnvelopeStatus::Admitted, Vec::new())
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

/// One rejected capability decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityDecisionRow {
    pub execution_mode: CapabilityEnvelopeStatus,
    pub scope: CapabilityScope,
    pub decision_id: CapabilityDecisionId,
    pub owner: RejectionOwner,
    pub category: Option<CapabilityRejectionCategory>,
    pub boundary: &'static str,
    pub reason: &'static str,
    pub status: CapabilityDecisionStatus,
    pub facts: BTreeMap<&'static str, String>,
}

impl CapabilityDecisionRow {
    /// Builds a row from a rejection.
    pub fn from_rejection(index: usize, rejection: &Rejection) -> Result<Self> {
        let row = Self {
            execution_mode: CapabilityEnvelopeStatus::Rejected,
            scope: rejection.contract_scope(),
            decision_id: CapabilityDecisionId { scope: rejection.contract_scope(), index },
            owner: rejection.owner,
            category: Some(rejection.category()),
            boundary: rejection.boundary(),
            reason: rejection.reason.as_str(),
            status: CapabilityDecisionStatus::Rejected,
            facts: rejection.facts.clone(),
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
    Rejected,
}

impl CapabilityDecisionStatus {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Rejected => "rejected",
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
        (RejectionOwner::Node(_), CapabilityScope::Node)
            | (RejectionOwner::Edge { .. }, CapabilityScope::Edge)
            | (RejectionOwner::Invariant(_) | RejectionOwner::Plan, CapabilityScope::Subgraph)
    );
    if !owner_scope_matches {
        return Err(capability_report_error("capability decision owner does not match its scope")
            .with_fact("decision_id", row.decision_id.to_string())
            .with_fact("owner", row.owner.to_string())
            .with_fact("scope", row.scope.as_str()));
    }
    match (row.status, row.category) {
        (CapabilityDecisionStatus::Rejected, Some(_)) => Ok(()),
        (CapabilityDecisionStatus::Rejected, None) => {
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
    QueryError::new(QueryErrorCode::InvalidPlan, "query_engine.capability_report", message)
}

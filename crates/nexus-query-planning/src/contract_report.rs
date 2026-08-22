//! Capability-contract accounting rows.
//!
//! The report model is DataFusion-free and gate-first: it projects capability
//! requirements and compiled reservation gates without routing through bounded
//! join family/path compatibility vocabulary.

use std::fmt;

use serde_json::{Value, json};

pub use crate::mode::CapabilityEnvelopeStatus;
use crate::{
    capability::{
        CapabilityAnalysis, CapabilityEnvelopeAdmission, CapabilityRejectionOwner, CapabilityRejectionProjection,
        ContractQuality, ContractStatus, EdgeConsumer, InvariantId, NodeOperator, ReservationRequirement,
        ReservationRequirementScope, ResultSinkKind, lower_level_unresolved_preflight_gap, unresolved_preflight_gap,
    },
    correlation::{QueryId, TraceId},
    error::{QueryError, QueryErrorCode, Result},
    execution_contracts::ReservationKind,
    pipeline::{CompiledPipeline, ReservationGate, ReservationGateId, ReservationGateOwner},
    plan::NodeId,
};

/// Gate-first report for capability-envelope readiness.
///
/// Its JSON and TSV projections use stable wire labels, but this report is the
/// canonical single-controller evidence store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityContractReport {
    query_id: Option<QueryId>,
    trace_id: Option<TraceId>,
    execution_mode: CapabilityEnvelopeStatus,
    gate_rows: Vec<CapabilityContractGateRow>,
}

impl CapabilityContractReport {
    /// Builds a capability-contract report from already-materialized gate rows.
    pub fn from_gate_rows(
        execution_mode: CapabilityEnvelopeStatus,
        gate_rows: Vec<CapabilityContractGateRow>,
    ) -> Result<Self> {
        Self::new(execution_mode, gate_rows)
    }

    /// Builds a report from compiled reservation gates and validates capability backing.
    pub fn from_compiled_pipeline(analysis: &CapabilityAnalysis, pipeline: &CompiledPipeline) -> Result<Self> {
        let execution_mode = capability_envelope_status_for_analysis(analysis, || {
            pipeline.gates().iter().any(|gate| gate.status() != ContractStatus::Modeled)
        });

        let mut gate_rows = Vec::with_capacity(pipeline.gates().len());
        for gate in pipeline.gates() {
            let row = CapabilityContractGateRow::from_compiled_gate(execution_mode, gate)?;
            validate_requirement_is_backed_by_capability_analysis(analysis, gate.requirement())?;
            gate_rows.push(row);
        }
        append_rejection_projection_if_needed(execution_mode, analysis, &mut gate_rows)?;
        Self::new(execution_mode, gate_rows)
    }

    /// Builds a report directly from capability-analysis requirements.
    ///
    /// This is used before pipeline compilation or when admission failed before
    /// concrete reservation gates could be emitted.
    pub fn from_capability_analysis(analysis: &CapabilityAnalysis) -> Result<Self> {
        let execution_mode = capability_envelope_status_for_analysis(analysis, || {
            analysis.requirements.iter().any(|requirement| requirement.status != ContractStatus::Modeled)
        });
        let mut gate_rows = Vec::new();

        for node in &analysis.nodes {
            for requirement in &node.local_requirements {
                let gate_id = CapabilityContractGateId::CapabilityRequirement {
                    scope: ReservationRequirementScope::Node,
                    index: gate_rows.len(),
                };
                let owner = CapabilityContractOwner::CapabilityNode(node.node_id);
                gate_rows.push(CapabilityContractGateRow::from_requirement(
                    execution_mode,
                    gate_id,
                    owner,
                    requirement,
                )?);
            }
        }

        for edge in &analysis.edges {
            for requirement in &edge.reservation_requirements {
                let gate_id = CapabilityContractGateId::CapabilityRequirement {
                    scope: ReservationRequirementScope::Edge,
                    index: gate_rows.len(),
                };
                let owner =
                    CapabilityContractOwner::CapabilityEdge { producer: edge.producer, consumer: edge.consumer };
                gate_rows.push(CapabilityContractGateRow::from_requirement(
                    execution_mode,
                    gate_id,
                    owner,
                    requirement,
                )?);
            }
        }

        for invariant in &analysis.invariants {
            for requirement in &invariant.reservation_requirements {
                let gate_id = CapabilityContractGateId::CapabilityRequirement {
                    scope: ReservationRequirementScope::Subgraph,
                    index: gate_rows.len(),
                };
                let owner = CapabilityContractOwner::CapabilitySubgraph(invariant.invariant_id);
                gate_rows.push(CapabilityContractGateRow::from_requirement(
                    execution_mode,
                    gate_id,
                    owner,
                    requirement,
                )?);
            }
        }

        append_rejection_projection_if_needed(execution_mode, analysis, &mut gate_rows)?;

        Self::new(execution_mode, gate_rows)
    }

    /// Returns the capability-envelope status represented by every row.
    pub fn execution_mode(&self) -> CapabilityEnvelopeStatus {
        self.execution_mode
    }

    /// Returns gate rows in stable report order.
    pub fn gate_rows(&self) -> &[CapabilityContractGateRow] {
        &self.gate_rows
    }

    /// Returns true when every gate is modeled for the capability envelope.
    pub fn is_ready(&self) -> bool {
        self.execution_mode == CapabilityEnvelopeStatus::Admitted
            && self.gate_rows.iter().all(|row| row.contract_status == ContractStatus::Modeled)
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

    /// Serializes each row as a stable comma-separated key/value string.
    pub fn stable_kv_rows(&self) -> Vec<String> {
        self.gate_rows.iter().map(CapabilityContractGateRow::to_stable_kv_row).collect()
    }

    /// Serializes this report with its established compatibility schema.
    pub fn to_json(&self) -> Value {
        json!({
            "schema": "nexus_query_engine.bounded_contract_report.v1",
            "query_id": self.query_id.as_ref().map(QueryId::as_str),
            "trace_id": self.trace_id.as_ref().map(TraceId::as_str),
            "triage": self.triage_summary().to_json(),
            "execution_mode": self.execution_mode.as_str(),
            "bounded_ready": self.is_ready(),
            "gate_rows": self.gate_rows.iter().map(CapabilityContractGateRow::to_json).collect::<Vec<_>>(),
        })
    }

    fn new(execution_mode: CapabilityEnvelopeStatus, gate_rows: Vec<CapabilityContractGateRow>) -> Result<Self> {
        if gate_rows.is_empty() {
            return Err(contract_report_error("capability contract report has no gate rows"));
        }
        for row in &gate_rows {
            validate_report_row(row)?;
            if row.execution_mode != execution_mode {
                return Err(contract_report_error("capability contract report row has inconsistent envelope status")
                    .with_fact("row_execution_mode", row.execution_mode.as_str())
                    .with_fact("report_execution_mode", execution_mode.as_str())
                    .with_fact("gate_id", row.gate_id.to_string()));
            }
        }
        validate_report_execution_mode(execution_mode, &gate_rows)?;
        Ok(Self { query_id: None, trace_id: None, execution_mode, gate_rows })
    }
}

fn append_rejection_projection_if_needed(
    execution_mode: CapabilityEnvelopeStatus,
    analysis: &CapabilityAnalysis,
    gate_rows: &mut Vec<CapabilityContractGateRow>,
) -> Result<()> {
    if execution_mode == CapabilityEnvelopeStatus::Rejected
        && gate_rows.iter().all(|row| row.contract_status == ContractStatus::Modeled)
        && analysis_has_join_operator(analysis)
        && let Some(projection) = analysis.first_rejection_projection()
    {
        gate_rows.push(CapabilityContractGateRow::from_capability_rejection_projection(
            execution_mode,
            gate_rows.len(),
            &projection,
        )?);
    }
    Ok(())
}

fn analysis_has_join_operator(analysis: &CapabilityAnalysis) -> bool {
    analysis.nodes.iter().any(|node| matches!(node.operator, NodeOperator::Join { .. }))
}

fn capability_envelope_status_for_analysis(
    analysis: &CapabilityAnalysis,
    has_non_modeled_gates: impl FnOnce() -> bool,
) -> CapabilityEnvelopeStatus {
    let has_non_modeled_gates = has_non_modeled_gates();
    match analysis.capability_envelope_admission() {
        CapabilityEnvelopeAdmission::Admitted { .. } if !has_non_modeled_gates => CapabilityEnvelopeStatus::Admitted,
        CapabilityEnvelopeAdmission::Rejected { .. }
            if !analysis_has_join_operator(analysis) && !has_non_modeled_gates =>
        {
            CapabilityEnvelopeStatus::Admitted
        },
        CapabilityEnvelopeAdmission::Admitted { .. } | CapabilityEnvelopeAdmission::Rejected { .. } => {
            CapabilityEnvelopeStatus::Rejected
        },
    }
}

/// One stable row describing a capability-envelope reservation or gate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityContractGateRow {
    /// Capability-envelope status for this row.
    pub execution_mode: CapabilityEnvelopeStatus,
    /// Most local graph scope that owns the gate.
    pub contract_scope: ReservationRequirementScope,
    /// Stable row id from either a compiled gate or capability requirement.
    pub gate_id: CapabilityContractGateId,
    /// Graph object that owns the gate.
    pub owner: CapabilityContractOwner,
    /// Runtime reservation class.
    pub reservation_kind: ReservationKind,
    /// Stable report label for the resource or execution boundary.
    pub memory_boundary: &'static str,
    /// Stable report label for the required lower-level or query-engine contract.
    pub required_contract: &'static str,
    /// Whether the gate is modeled, blocked, deferred, or rejected.
    pub contract_status: ContractStatus,
    /// Quality of the evidence backing a modeled gate.
    pub contract_quality: ContractQuality,
    /// Named cudf/libcudf gap when quality is unresolved preflight.
    pub unresolved_preflight_gap: Option<&'static str>,
}

impl CapabilityContractGateRow {
    fn from_compiled_gate(execution_mode: CapabilityEnvelopeStatus, gate: &ReservationGate) -> Result<Self> {
        Self::from_requirement(
            execution_mode,
            CapabilityContractGateId::Compiled(gate.id()),
            CapabilityContractOwner::Compiled(gate.owner()),
            gate.requirement(),
        )
    }

    fn from_requirement(
        execution_mode: CapabilityEnvelopeStatus,
        gate_id: CapabilityContractGateId,
        owner: CapabilityContractOwner,
        requirement: &ReservationRequirement,
    ) -> Result<Self> {
        validate_report_requirement(&gate_id, requirement)?;
        Ok(Self {
            execution_mode,
            contract_scope: requirement.scope,
            gate_id,
            owner,
            reservation_kind: requirement.kind,
            memory_boundary: requirement.memory_boundary,
            required_contract: requirement.required_contract,
            contract_status: requirement.status,
            contract_quality: requirement.quality,
            unresolved_preflight_gap: lower_level_unresolved_preflight_gap(
                requirement.kind,
                requirement.required_contract,
            )
            .filter(|_| requirement.quality == ContractQuality::HardBudgetedReportWithUnresolvedPreflight),
        })
    }

    /// Builds a synthetic report row for the first capability rejection.
    pub fn from_capability_rejection_projection(
        execution_mode: CapabilityEnvelopeStatus,
        index: usize,
        projection: &CapabilityRejectionProjection,
    ) -> Result<Self> {
        let row = Self {
            execution_mode,
            contract_scope: projection.contract_scope,
            gate_id: CapabilityContractGateId::CapabilityRequirement { scope: projection.contract_scope, index },
            owner: CapabilityContractOwner::from_capability_rejection_owner(&projection.owner),
            reservation_kind: ReservationKind::CapabilityContract,
            memory_boundary: projection.memory_boundary,
            required_contract: projection.reason,
            contract_status: ContractStatus::RejectedByCapability,
            contract_quality: ContractQuality::CapabilityProof,
            unresolved_preflight_gap: None,
        };
        validate_report_row(&row)?;
        Ok(row)
    }

    /// Builds the gate row for a composed consumer waiting for resident
    /// producer output evidence during the prepared-admission stage.
    pub fn from_deferred_composed_producer_binding(index: usize) -> Result<Self> {
        let row = Self {
            execution_mode: CapabilityEnvelopeStatus::Deferred,
            contract_scope: ReservationRequirementScope::Subgraph,
            gate_id: CapabilityContractGateId::CapabilityRequirement {
                scope: ReservationRequirementScope::Subgraph,
                index,
            },
            owner: CapabilityContractOwner::CapabilityAdmission("composed_producer_binding"),
            reservation_kind: ReservationKind::CapabilityContract,
            memory_boundary: "composed_consumer",
            required_contract: "producer_occurrence_output_evidence",
            contract_status: ContractStatus::DeferredUntilPrerequisite,
            contract_quality: ContractQuality::CapabilityProof,
            unresolved_preflight_gap: None,
        };
        validate_report_row(&row)?;
        Ok(row)
    }

    /// Serializes this row as stable comma-separated key/value pairs.
    pub fn to_stable_kv_row(&self) -> String {
        format!(
            "execution_mode={},contract_scope={},gate_id={},owner={},reservation_kind={},memory_boundary={},required_contract={},contract_status={},contract_quality={},unresolved_preflight_gap={}",
            self.execution_mode.as_str(),
            self.contract_scope.as_str(),
            self.gate_id,
            self.owner,
            self.reservation_kind.as_str(),
            self.memory_boundary,
            self.required_contract,
            self.contract_status.as_str(),
            self.contract_quality.as_str(),
            self.unresolved_preflight_gap.unwrap_or("none")
        )
    }

    /// Serializes this row with the report's compatibility schema.
    pub fn to_json(&self) -> Value {
        json!({
            "execution_mode": self.execution_mode.as_str(),
            "contract_scope": self.contract_scope.as_str(),
            "gate_id": self.gate_id.to_string(),
            "owner": self.owner.to_string(),
            "reservation_kind": self.reservation_kind.as_str(),
            "memory_boundary": self.memory_boundary,
            "required_contract": self.required_contract,
            "contract_status": self.contract_status.as_str(),
            "contract_quality": self.contract_quality.as_str(),
            "unresolved_preflight_gap": self.unresolved_preflight_gap,
        })
    }
}

impl fmt::Display for CapabilityContractGateRow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_stable_kv_row())
    }
}

/// Stable identifier for a capability-contract report row.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CapabilityContractGateId {
    /// Row comes from a reservation gate in a compiled pipeline graph.
    Compiled(ReservationGateId),
    /// Row comes from a capability requirement before graph compilation.
    CapabilityRequirement {
        /// Capability scope that owns the requirement.
        scope: ReservationRequirementScope,
        /// Stable row index within the generated capability requirement rows.
        index: usize,
    },
}

impl fmt::Display for CapabilityContractGateId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Compiled(gate_id) => write!(f, "{gate_id}"),
            Self::CapabilityRequirement { scope, index } => {
                write!(f, "capability:{}:{index}", scope.as_str())
            },
        }
    }
}

/// Report-facing owner for a capability-contract gate.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CapabilityContractOwner {
    /// Gate owner from a compiled pipeline graph.
    Compiled(ReservationGateOwner),
    /// Node-local capability owner.
    CapabilityNode(NodeId),
    /// Producer-consumer capability owner.
    CapabilityEdge {
        /// Producer node for the capability edge.
        producer: NodeId,
        /// Consumer endpoint for the capability edge.
        consumer: EdgeConsumer,
    },
    /// Subgraph invariant capability owner.
    CapabilitySubgraph(InvariantId),
    /// Whole-analysis admission owner used only for fallback projection.
    CapabilityAdmission(&'static str),
}

impl CapabilityContractOwner {
    fn from_capability_rejection_owner(owner: &CapabilityRejectionOwner) -> Self {
        match owner {
            CapabilityRejectionOwner::Node(node) => Self::CapabilityNode(*node),
            CapabilityRejectionOwner::Edge { producer, consumer } => {
                Self::CapabilityEdge { producer: *producer, consumer: *consumer }
            },
            CapabilityRejectionOwner::Invariant(invariant) => Self::CapabilitySubgraph(*invariant),
            CapabilityRejectionOwner::Admission(admission) => Self::CapabilityAdmission(admission),
        }
    }
}

impl fmt::Display for CapabilityContractOwner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Compiled(owner) => write!(f, "{owner}"),
            Self::CapabilityNode(node) => write!(f, "node:{node}"),
            Self::CapabilityEdge { producer, consumer } => {
                write!(f, "edge:{producer}->{}", DisplayEdgeConsumer(*consumer))
            },
            Self::CapabilitySubgraph(invariant) => write!(f, "subgraph:i{}", invariant.index()),
            Self::CapabilityAdmission(admission) => write!(f, "admission:{admission}"),
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

fn validate_requirement_is_backed_by_capability_analysis(
    analysis: &CapabilityAnalysis,
    requirement: &ReservationRequirement,
) -> Result<()> {
    if analysis.requirements.iter().any(|candidate| candidate == requirement) {
        return Ok(());
    }
    Err(contract_report_error("compiled capability gate is not backed by capability analysis")
        .with_fact("reservation_kind", requirement.kind.as_str())
        .with_fact("memory_boundary", requirement.memory_boundary)
        .with_fact("required_contract", requirement.required_contract)
        .with_fact("contract_status", requirement.status.as_str())
        .with_fact("contract_quality", requirement.quality.as_str()))
}

fn validate_report_requirement(gate_id: &CapabilityContractGateId, requirement: &ReservationRequirement) -> Result<()> {
    if requirement.memory_boundary.is_empty() {
        return Err(contract_report_error("capability contract gate has empty memory boundary")
            .with_fact("gate_id", gate_id)
            .with_fact("reservation_kind", requirement.kind.as_str()));
    }
    if requirement.required_contract.is_empty() {
        return Err(contract_report_error("capability contract gate has empty required contract")
            .with_fact("gate_id", gate_id)
            .with_fact("reservation_kind", requirement.kind.as_str()));
    }
    if requirement.status.as_str().is_empty() {
        return Err(contract_report_error("capability contract gate lacks status")
            .with_fact("gate_id", gate_id)
            .with_fact("reservation_kind", requirement.kind.as_str()));
    }
    if requirement.quality.as_str().is_empty() {
        return Err(contract_report_error("capability contract gate lacks quality")
            .with_fact("gate_id", gate_id)
            .with_fact("reservation_kind", requirement.kind.as_str()));
    }

    let unresolved_gap = unresolved_preflight_gap(requirement.kind, requirement.required_contract);
    if requirement.quality == ContractQuality::HardBudgetedReportWithUnresolvedPreflight {
        if requirement.status != ContractStatus::Modeled {
            return Err(contract_report_error("hard-budgeted unresolved-preflight gate must remain modeled")
                .with_fact("gate_id", gate_id)
                .with_fact("reservation_kind", requirement.kind.as_str())
                .with_fact("contract_status", requirement.status.as_str())
                .with_fact("contract_quality", requirement.quality.as_str()));
        }
        if unresolved_gap.is_none() {
            return Err(contract_report_error("hard-budgeted gate is missing lower-level unresolved preflight gap")
                .with_fact("gate_id", gate_id)
                .with_fact("reservation_kind", requirement.kind.as_str())
                .with_fact("required_contract", requirement.required_contract));
        }
    }
    if unresolved_gap.is_some() && requirement.quality != ContractQuality::HardBudgetedReportWithUnresolvedPreflight {
        return Err(contract_report_error("unresolved-preflight gate must use unresolved-preflight quality")
            .with_fact("gate_id", gate_id)
            .with_fact("reservation_kind", requirement.kind.as_str())
            .with_fact("required_contract", requirement.required_contract)
            .with_fact("contract_status", requirement.status.as_str())
            .with_fact("contract_quality", requirement.quality.as_str()));
    }
    Ok(())
}

fn validate_report_row(row: &CapabilityContractGateRow) -> Result<()> {
    if row.memory_boundary.is_empty() {
        return Err(contract_report_error("capability contract gate has empty memory boundary")
            .with_fact("gate_id", row.gate_id.to_string())
            .with_fact("reservation_kind", row.reservation_kind.as_str()));
    }
    if row.required_contract.is_empty() {
        return Err(contract_report_error("capability contract gate has empty required contract")
            .with_fact("gate_id", row.gate_id.to_string())
            .with_fact("reservation_kind", row.reservation_kind.as_str()));
    }
    validate_report_row_owner_scope(row)?;
    validate_quality_gap_consistency(
        &row.gate_id,
        row.reservation_kind,
        row.required_contract,
        row.contract_status,
        row.contract_quality,
        row.unresolved_preflight_gap,
    )?;
    validate_quality_ownership(
        &row.gate_id,
        row.reservation_kind,
        row.required_contract,
        row.contract_status,
        row.contract_quality,
    )
}

fn validate_report_execution_mode(
    execution_mode: CapabilityEnvelopeStatus,
    gate_rows: &[CapabilityContractGateRow],
) -> Result<()> {
    let has_non_modeled_row = gate_rows.iter().any(|row| row.contract_status != ContractStatus::Modeled);
    match execution_mode {
        CapabilityEnvelopeStatus::Admitted if has_non_modeled_row => {
            Err(contract_report_error("admitted capability report cannot contain non-modeled contract rows")
                .with_fact("report_execution_mode", execution_mode.as_str()))
        },
        CapabilityEnvelopeStatus::Rejected if !has_non_modeled_row => {
            Err(contract_report_error("rejected capability report must contain a non-modeled contract row")
                .with_fact("report_execution_mode", execution_mode.as_str()))
        },
        CapabilityEnvelopeStatus::Deferred
            if gate_rows.iter().any(|row| row.contract_status != ContractStatus::DeferredUntilPrerequisite) =>
        {
            Err(contract_report_error("deferred capability report can contain only deferred contract rows")
                .with_fact("report_execution_mode", execution_mode.as_str()))
        },
        CapabilityEnvelopeStatus::Deferred if !has_non_modeled_row => {
            Err(contract_report_error("deferred capability report must contain a deferred contract row")
                .with_fact("report_execution_mode", execution_mode.as_str()))
        },
        CapabilityEnvelopeStatus::Admitted
        | CapabilityEnvelopeStatus::Deferred
        | CapabilityEnvelopeStatus::Rejected => Ok(()),
    }
}

fn validate_report_row_owner_scope(row: &CapabilityContractGateRow) -> Result<()> {
    let valid = match (&row.owner, row.contract_scope) {
        (CapabilityContractOwner::Compiled(ReservationGateOwner::Task(_)), ReservationRequirementScope::Node) => true,
        (CapabilityContractOwner::Compiled(ReservationGateOwner::Edge(_)), ReservationRequirementScope::Edge) => true,
        (CapabilityContractOwner::Compiled(ReservationGateOwner::SinkBoundary), ReservationRequirementScope::Edge) => {
            sink_boundary_edge_gate_kind_is_allowed(row.reservation_kind)
        },
        (
            CapabilityContractOwner::Compiled(ReservationGateOwner::Barrier(_)),
            ReservationRequirementScope::Subgraph,
        ) => true,
        (CapabilityContractOwner::CapabilityNode(_), ReservationRequirementScope::Node) => true,
        (CapabilityContractOwner::CapabilityEdge { .. }, ReservationRequirementScope::Edge) => true,
        (CapabilityContractOwner::CapabilitySubgraph(_), ReservationRequirementScope::Subgraph) => true,
        (CapabilityContractOwner::CapabilityAdmission(_), ReservationRequirementScope::Subgraph) => true,
        _ => false,
    };
    if valid {
        return Ok(());
    }
    Err(contract_report_error("capability contract owner does not match contract scope")
        .with_fact("gate_id", row.gate_id.to_string())
        .with_fact("owner", row.owner.to_string())
        .with_fact("contract_scope", row.contract_scope.as_str())
        .with_fact("reservation_kind", row.reservation_kind.as_str()))
}

fn sink_boundary_edge_gate_kind_is_allowed(kind: ReservationKind) -> bool {
    matches!(
        kind,
        ReservationKind::SourceRead
            | ReservationKind::GatherMap
            | ReservationKind::JoinOutput
            | ReservationKind::Materialization
    )
}

fn validate_quality_gap_consistency(
    gate_id: &CapabilityContractGateId,
    kind: ReservationKind,
    required_contract: &str,
    status: ContractStatus,
    quality: ContractQuality,
    reported_unresolved_preflight_gap: Option<&'static str>,
) -> Result<()> {
    let expected_gap = unresolved_preflight_gap(kind, required_contract).map(|gap| gap.unresolved_gap);
    if quality == ContractQuality::HardBudgetedReportWithUnresolvedPreflight {
        if status != ContractStatus::Modeled {
            return Err(contract_report_error("hard-budgeted unresolved-preflight gate must remain modeled")
                .with_fact("gate_id", gate_id.to_string())
                .with_fact("reservation_kind", kind.as_str())
                .with_fact("contract_status", status.as_str())
                .with_fact("contract_quality", quality.as_str()));
        }
        let Some(expected_gap) = expected_gap else {
            return Err(contract_report_error("hard-budgeted gate is missing lower-level unresolved preflight gap")
                .with_fact("gate_id", gate_id.to_string())
                .with_fact("reservation_kind", kind.as_str())
                .with_fact("required_contract", required_contract));
        };
        if reported_unresolved_preflight_gap != Some(expected_gap) {
            return Err(contract_report_error(
                "hard-budgeted gate has mismatched lower-level unresolved preflight gap",
            )
            .with_fact("gate_id", gate_id.to_string())
            .with_fact("reservation_kind", kind.as_str())
            .with_fact("required_contract", required_contract)
            .with_fact("expected_unresolved_preflight_gap", expected_gap)
            .with_fact("unresolved_preflight_gap", reported_unresolved_preflight_gap.unwrap_or("none")));
        }
    } else if reported_unresolved_preflight_gap.is_some() {
        return Err(contract_report_error("non-hard-budgeted gate must not carry unresolved preflight gap")
            .with_fact("gate_id", gate_id.to_string())
            .with_fact("reservation_kind", kind.as_str())
            .with_fact("required_contract", required_contract)
            .with_fact("contract_quality", quality.as_str())
            .with_fact("unresolved_preflight_gap", reported_unresolved_preflight_gap.unwrap_or("none")));
    }
    if expected_gap.is_some() && quality != ContractQuality::HardBudgetedReportWithUnresolvedPreflight {
        return Err(contract_report_error("unresolved-preflight gate must use unresolved-preflight quality")
            .with_fact("gate_id", gate_id.to_string())
            .with_fact("reservation_kind", kind.as_str())
            .with_fact("required_contract", required_contract)
            .with_fact("contract_status", status.as_str())
            .with_fact("contract_quality", quality.as_str()));
    }
    Ok(())
}

fn validate_quality_ownership(
    gate_id: &CapabilityContractGateId,
    kind: ReservationKind,
    required_contract: &str,
    status: ContractStatus,
    quality: ContractQuality,
) -> Result<()> {
    if status != ContractStatus::Modeled {
        return Ok(());
    }
    if matches!(kind, ReservationKind::FilteredProbeState | ReservationKind::MixedSizeData)
        && quality != ContractQuality::ObservedRetainedState
    {
        return Err(contract_report_error("retained join-state gate must report observed retained-state quality")
            .with_fact("gate_id", gate_id.to_string())
            .with_fact("reservation_kind", kind.as_str())
            .with_fact("required_contract", required_contract)
            .with_fact("contract_quality", quality.as_str()));
    }
    if quality == ContractQuality::ExactPreflight
        && !exact_preflight_contract_is_owned_by_lower_layer(kind, required_contract)
    {
        return Err(contract_report_error("exact-preflight gate is not backed by an owning lower-level contract")
            .with_fact("gate_id", gate_id.to_string())
            .with_fact("reservation_kind", kind.as_str())
            .with_fact("required_contract", required_contract)
            .with_fact("contract_quality", quality.as_str()));
    }
    Ok(())
}

fn exact_preflight_contract_is_owned_by_lower_layer(kind: ReservationKind, required_contract: &str) -> bool {
    match kind {
        ReservationKind::HashTable => required_contract == "hash_join_pre_build_reservation_size_bytes",
        ReservationKind::GatherMap => required_contract == "join_gather_map_reservation_size_bytes",
        ReservationKind::JoinOutput => {
            matches!(
                required_contract,
                "dataframe_gather_output_reservation_size_bytes"
                    | "projected_cross_join_reservation_size_bytes"
                    | "dataframe_gather_output_reservation_size_bytes+projected_cross_join_reservation_size_bytes"
                    | "keyed_equality_join_output_size_preflight"
            )
        },
        _ => true,
    }
}

fn contract_report_error(message: &'static str) -> QueryError {
    QueryError::new(QueryErrorCode::InvalidPlan, "nexus_query_engine.capability_contract_report", message)
}

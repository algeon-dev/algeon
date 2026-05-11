//! Strict bounded contract accounting rows.
//!
//! The report model is DataFusion-free and gate-first: it projects capability
//! requirements and compiled reservation gates without routing through bounded
//! join family/path compatibility vocabulary.

use std::fmt;

use crate::{
    capability::{
        CapabilityAnalysis, ContractQuality, ContractStatus, EdgeConsumer, InvariantId, ReservationRequirement,
        ReservationRequirementScope, ResultSinkKind, lower_level_unresolved_preflight_gap,
    },
    error::{QueryError, QueryErrorCode, Result},
    pipeline::{CompiledPipeline, ReservationGate, ReservationGateId, ReservationGateOwner},
    plan::NodeId,
    runtime::ReservationKind,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StrictBoundedExecutionMode {
    StrictBounded,
    UnsupportedStrictBounded,
}

impl StrictBoundedExecutionMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::StrictBounded => "strict_bounded",
            Self::UnsupportedStrictBounded => "unsupported_strict_bounded",
        }
    }
}

impl fmt::Display for StrictBoundedExecutionMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StrictBoundedContractReport {
    execution_mode: StrictBoundedExecutionMode,
    gate_rows: Vec<StrictBoundedContractGateRow>,
}

impl StrictBoundedContractReport {
    pub fn from_compiled_pipeline(analysis: &CapabilityAnalysis, pipeline: &CompiledPipeline) -> Result<Self> {
        let execution_mode = if analysis.has_rejections()
            || pipeline.gates().iter().any(|gate| gate.status() != ContractStatus::Modeled)
        {
            StrictBoundedExecutionMode::UnsupportedStrictBounded
        } else {
            StrictBoundedExecutionMode::StrictBounded
        };

        let mut gate_rows = Vec::with_capacity(pipeline.gates().len());
        for gate in pipeline.gates() {
            let row = StrictBoundedContractGateRow::from_compiled_gate(execution_mode, gate)?;
            validate_requirement_is_backed_by_capability_analysis(analysis, gate.requirement())?;
            gate_rows.push(row);
        }
        Self::new(execution_mode, gate_rows)
    }

    pub fn from_capability_analysis(analysis: &CapabilityAnalysis) -> Result<Self> {
        let execution_mode = if analysis.has_rejections()
            || analysis.requirements.iter().any(|requirement| requirement.status != ContractStatus::Modeled)
        {
            StrictBoundedExecutionMode::UnsupportedStrictBounded
        } else {
            StrictBoundedExecutionMode::StrictBounded
        };
        let mut gate_rows = Vec::new();

        for node in &analysis.nodes {
            for requirement in &node.local_requirements {
                let gate_id = StrictBoundedContractGateId::CapabilityRequirement {
                    scope: ReservationRequirementScope::Node,
                    index: gate_rows.len(),
                };
                let owner = StrictBoundedContractOwner::CapabilityNode(node.node_id);
                gate_rows.push(StrictBoundedContractGateRow::from_requirement(
                    execution_mode,
                    gate_id,
                    owner,
                    requirement,
                )?);
            }
        }

        for edge in &analysis.edges {
            for requirement in &edge.reservation_requirements {
                let gate_id = StrictBoundedContractGateId::CapabilityRequirement {
                    scope: ReservationRequirementScope::Edge,
                    index: gate_rows.len(),
                };
                let owner =
                    StrictBoundedContractOwner::CapabilityEdge { producer: edge.producer, consumer: edge.consumer };
                gate_rows.push(StrictBoundedContractGateRow::from_requirement(
                    execution_mode,
                    gate_id,
                    owner,
                    requirement,
                )?);
            }
        }

        for invariant in &analysis.invariants {
            for requirement in &invariant.reservation_requirements {
                let gate_id = StrictBoundedContractGateId::CapabilityRequirement {
                    scope: ReservationRequirementScope::Subgraph,
                    index: gate_rows.len(),
                };
                let owner = StrictBoundedContractOwner::CapabilitySubgraph(invariant.invariant_id);
                gate_rows.push(StrictBoundedContractGateRow::from_requirement(
                    execution_mode,
                    gate_id,
                    owner,
                    requirement,
                )?);
            }
        }

        Self::new(execution_mode, gate_rows)
    }

    pub fn execution_mode(&self) -> StrictBoundedExecutionMode {
        self.execution_mode
    }

    pub fn gate_rows(&self) -> &[StrictBoundedContractGateRow] {
        &self.gate_rows
    }

    pub fn is_strict_bounded_ready(&self) -> bool {
        self.execution_mode == StrictBoundedExecutionMode::StrictBounded
            && self.gate_rows.iter().all(|row| row.contract_status == ContractStatus::Modeled)
    }

    pub fn stable_kv_rows(&self) -> Vec<String> {
        self.gate_rows.iter().map(StrictBoundedContractGateRow::to_stable_kv_row).collect()
    }

    fn new(execution_mode: StrictBoundedExecutionMode, gate_rows: Vec<StrictBoundedContractGateRow>) -> Result<Self> {
        if gate_rows.is_empty() {
            return Err(contract_report_error("strict bounded contract report has no gate rows"));
        }
        for row in &gate_rows {
            if row.execution_mode != execution_mode {
                return Err(contract_report_error(
                    "strict bounded contract report row has inconsistent execution mode",
                )
                .with_fact("row_execution_mode", row.execution_mode.as_str())
                .with_fact("report_execution_mode", execution_mode.as_str())
                .with_fact("gate_id", row.gate_id.to_string()));
            }
        }
        Ok(Self { execution_mode, gate_rows })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StrictBoundedContractGateRow {
    pub execution_mode: StrictBoundedExecutionMode,
    pub contract_scope: ReservationRequirementScope,
    pub gate_id: StrictBoundedContractGateId,
    pub owner: StrictBoundedContractOwner,
    pub reservation_kind: ReservationKind,
    pub memory_boundary: &'static str,
    pub required_contract: &'static str,
    pub contract_status: ContractStatus,
    pub contract_quality: ContractQuality,
    pub unresolved_preflight_gap: Option<&'static str>,
}

impl StrictBoundedContractGateRow {
    fn from_compiled_gate(execution_mode: StrictBoundedExecutionMode, gate: &ReservationGate) -> Result<Self> {
        Self::from_requirement(
            execution_mode,
            StrictBoundedContractGateId::Compiled(gate.id()),
            StrictBoundedContractOwner::Compiled(gate.owner()),
            gate.requirement(),
        )
    }

    fn from_requirement(
        execution_mode: StrictBoundedExecutionMode,
        gate_id: StrictBoundedContractGateId,
        owner: StrictBoundedContractOwner,
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
}

impl fmt::Display for StrictBoundedContractGateRow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_stable_kv_row())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StrictBoundedContractGateId {
    Compiled(ReservationGateId),
    CapabilityRequirement { scope: ReservationRequirementScope, index: usize },
}

impl fmt::Display for StrictBoundedContractGateId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Compiled(gate_id) => write!(f, "{gate_id}"),
            Self::CapabilityRequirement { scope, index } => {
                write!(f, "capability:{}:{index}", scope.as_str())
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StrictBoundedContractOwner {
    Compiled(ReservationGateOwner),
    CapabilityNode(NodeId),
    CapabilityEdge { producer: NodeId, consumer: EdgeConsumer },
    CapabilitySubgraph(InvariantId),
}

impl fmt::Display for StrictBoundedContractOwner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Compiled(owner) => write!(f, "{owner}"),
            Self::CapabilityNode(node) => write!(f, "node:{node}"),
            Self::CapabilityEdge { producer, consumer } => {
                write!(f, "edge:{producer}->{}", DisplayEdgeConsumer(*consumer))
            },
            Self::CapabilitySubgraph(invariant) => write!(f, "subgraph:i{}", invariant.index()),
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
    Err(contract_report_error("compiled strict bounded gate is not backed by capability analysis")
        .with_fact("reservation_kind", requirement.kind.as_str())
        .with_fact("memory_boundary", requirement.memory_boundary)
        .with_fact("required_contract", requirement.required_contract)
        .with_fact("contract_status", requirement.status.as_str())
        .with_fact("contract_quality", requirement.quality.as_str()))
}

fn validate_report_requirement(
    gate_id: &StrictBoundedContractGateId,
    requirement: &ReservationRequirement,
) -> Result<()> {
    if requirement.memory_boundary.is_empty() {
        return Err(contract_report_error("strict bounded contract gate has empty memory boundary")
            .with_fact("gate_id", gate_id)
            .with_fact("reservation_kind", requirement.kind.as_str()));
    }
    if requirement.required_contract.is_empty() {
        return Err(contract_report_error("strict bounded contract gate has empty required contract")
            .with_fact("gate_id", gate_id)
            .with_fact("reservation_kind", requirement.kind.as_str()));
    }
    if requirement.status.as_str().is_empty() {
        return Err(contract_report_error("strict bounded contract gate lacks status")
            .with_fact("gate_id", gate_id)
            .with_fact("reservation_kind", requirement.kind.as_str()));
    }
    if requirement.quality.as_str().is_empty() {
        return Err(contract_report_error("strict bounded contract gate lacks quality")
            .with_fact("gate_id", gate_id)
            .with_fact("reservation_kind", requirement.kind.as_str()));
    }

    let unresolved_gap = lower_level_unresolved_preflight_gap(requirement.kind, requirement.required_contract);
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
    if unresolved_gap.is_some()
        && requirement.status == ContractStatus::Modeled
        && requirement.quality == ContractQuality::ExactPreflight
    {
        return Err(contract_report_error("unresolved-preflight gate cannot be reported as exact preflight")
            .with_fact("gate_id", gate_id)
            .with_fact("reservation_kind", requirement.kind.as_str())
            .with_fact("required_contract", requirement.required_contract)
            .with_fact("contract_quality", requirement.quality.as_str()));
    }
    Ok(())
}

fn contract_report_error(message: &'static str) -> QueryError {
    QueryError::new(QueryErrorCode::InvalidPlan, "cudf_query_engine.strict_bounded_contract_report", message)
}

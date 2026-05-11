//! Strict bounded contract accounting rows.
//!
//! The report model is DataFusion-free and gate-first: it projects capability
//! requirements and compiled reservation gates without routing through bounded
//! join family/path compatibility vocabulary.

use std::fmt;

use crate::{
    capability::{
        CapabilityAnalysis, CapabilityRejectionOwner, CapabilityRejectionProjection, ContractQuality, ContractStatus,
        EdgeConsumer, InvariantId, ReservationRequirement, ReservationRequirementScope, ResultSinkKind,
        lower_level_unresolved_preflight_gap,
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
    pub fn from_gate_rows(
        execution_mode: StrictBoundedExecutionMode,
        gate_rows: Vec<StrictBoundedContractGateRow>,
    ) -> Result<Self> {
        Self::new(execution_mode, gate_rows)
    }

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

        if execution_mode == StrictBoundedExecutionMode::UnsupportedStrictBounded
            && gate_rows.iter().all(|row| row.contract_status == ContractStatus::Modeled)
            && let Some(projection) = analysis.first_rejection_projection()
        {
            gate_rows.push(StrictBoundedContractGateRow::from_capability_rejection_projection(
                execution_mode,
                gate_rows.len(),
                &projection,
            )?);
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
            validate_report_row(row)?;
            if row.execution_mode != execution_mode {
                return Err(contract_report_error(
                    "strict bounded contract report row has inconsistent execution mode",
                )
                .with_fact("row_execution_mode", row.execution_mode.as_str())
                .with_fact("report_execution_mode", execution_mode.as_str())
                .with_fact("gate_id", row.gate_id.to_string()));
            }
        }
        validate_report_execution_mode(execution_mode, &gate_rows)?;
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

    pub fn from_capability_rejection_projection(
        execution_mode: StrictBoundedExecutionMode,
        index: usize,
        projection: &CapabilityRejectionProjection,
    ) -> Result<Self> {
        let row = Self {
            execution_mode,
            contract_scope: projection.contract_scope,
            gate_id: StrictBoundedContractGateId::CapabilityRequirement { scope: projection.contract_scope, index },
            owner: StrictBoundedContractOwner::from_capability_rejection_owner(&projection.owner),
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
    CapabilityAdmission(&'static str),
}

impl StrictBoundedContractOwner {
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

impl fmt::Display for StrictBoundedContractOwner {
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
        && requirement.quality != ContractQuality::HardBudgetedReportWithUnresolvedPreflight
    {
        return Err(contract_report_error("modeled unresolved-preflight gate must use unresolved-preflight quality")
            .with_fact("gate_id", gate_id)
            .with_fact("reservation_kind", requirement.kind.as_str())
            .with_fact("required_contract", requirement.required_contract)
            .with_fact("contract_quality", requirement.quality.as_str()));
    }
    Ok(())
}

fn validate_report_row(row: &StrictBoundedContractGateRow) -> Result<()> {
    if row.memory_boundary.is_empty() {
        return Err(contract_report_error("strict bounded contract gate has empty memory boundary")
            .with_fact("gate_id", row.gate_id.to_string())
            .with_fact("reservation_kind", row.reservation_kind.as_str()));
    }
    if row.required_contract.is_empty() {
        return Err(contract_report_error("strict bounded contract gate has empty required contract")
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
    execution_mode: StrictBoundedExecutionMode,
    gate_rows: &[StrictBoundedContractGateRow],
) -> Result<()> {
    let has_non_modeled_row = gate_rows.iter().any(|row| row.contract_status != ContractStatus::Modeled);
    match execution_mode {
        StrictBoundedExecutionMode::StrictBounded if has_non_modeled_row => {
            Err(contract_report_error("strict bounded report cannot contain non-modeled contract rows")
                .with_fact("report_execution_mode", execution_mode.as_str()))
        },
        StrictBoundedExecutionMode::UnsupportedStrictBounded if !has_non_modeled_row => {
            Err(contract_report_error("unsupported strict bounded report must contain a non-modeled contract row")
                .with_fact("report_execution_mode", execution_mode.as_str()))
        },
        StrictBoundedExecutionMode::StrictBounded | StrictBoundedExecutionMode::UnsupportedStrictBounded => Ok(()),
    }
}

fn validate_report_row_owner_scope(row: &StrictBoundedContractGateRow) -> Result<()> {
    let valid = match (&row.owner, row.contract_scope) {
        (StrictBoundedContractOwner::Compiled(ReservationGateOwner::Task(_)), ReservationRequirementScope::Node) => {
            true
        },
        (StrictBoundedContractOwner::Compiled(ReservationGateOwner::Edge(_)), ReservationRequirementScope::Edge) => {
            true
        },
        (
            StrictBoundedContractOwner::Compiled(ReservationGateOwner::SinkBoundary),
            ReservationRequirementScope::Edge,
        ) if row.reservation_kind == ReservationKind::Materialization => true,
        (
            StrictBoundedContractOwner::Compiled(ReservationGateOwner::Barrier(_)),
            ReservationRequirementScope::Subgraph,
        ) => true,
        (StrictBoundedContractOwner::CapabilityNode(_), ReservationRequirementScope::Node) => true,
        (StrictBoundedContractOwner::CapabilityEdge { .. }, ReservationRequirementScope::Edge) => true,
        (StrictBoundedContractOwner::CapabilitySubgraph(_), ReservationRequirementScope::Subgraph) => true,
        (StrictBoundedContractOwner::CapabilityAdmission(_), ReservationRequirementScope::Subgraph) => true,
        _ => false,
    };
    if valid {
        return Ok(());
    }
    Err(contract_report_error("strict bounded contract owner does not match contract scope")
        .with_fact("gate_id", row.gate_id.to_string())
        .with_fact("owner", row.owner.to_string())
        .with_fact("contract_scope", row.contract_scope.as_str())
        .with_fact("reservation_kind", row.reservation_kind.as_str()))
}

fn validate_quality_gap_consistency(
    gate_id: &StrictBoundedContractGateId,
    kind: ReservationKind,
    required_contract: &str,
    status: ContractStatus,
    quality: ContractQuality,
    unresolved_preflight_gap: Option<&'static str>,
) -> Result<()> {
    let expected_gap = lower_level_unresolved_preflight_gap(kind, required_contract);
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
        if unresolved_preflight_gap != Some(expected_gap) {
            return Err(contract_report_error(
                "hard-budgeted gate has mismatched lower-level unresolved preflight gap",
            )
            .with_fact("gate_id", gate_id.to_string())
            .with_fact("reservation_kind", kind.as_str())
            .with_fact("required_contract", required_contract)
            .with_fact("expected_unresolved_preflight_gap", expected_gap)
            .with_fact("unresolved_preflight_gap", unresolved_preflight_gap.unwrap_or("none")));
        }
    } else if unresolved_preflight_gap.is_some() {
        return Err(contract_report_error("non-hard-budgeted gate must not carry unresolved preflight gap")
            .with_fact("gate_id", gate_id.to_string())
            .with_fact("reservation_kind", kind.as_str())
            .with_fact("required_contract", required_contract)
            .with_fact("contract_quality", quality.as_str())
            .with_fact("unresolved_preflight_gap", unresolved_preflight_gap.unwrap_or("none")));
    }
    if expected_gap.is_some()
        && status == ContractStatus::Modeled
        && quality != ContractQuality::HardBudgetedReportWithUnresolvedPreflight
    {
        return Err(contract_report_error("modeled unresolved-preflight gate must use unresolved-preflight quality")
            .with_fact("gate_id", gate_id.to_string())
            .with_fact("reservation_kind", kind.as_str())
            .with_fact("required_contract", required_contract)
            .with_fact("contract_quality", quality.as_str()));
    }
    Ok(())
}

fn validate_quality_ownership(
    gate_id: &StrictBoundedContractGateId,
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
            )
        },
        _ => true,
    }
}

fn contract_report_error(message: &'static str) -> QueryError {
    QueryError::new(QueryErrorCode::InvalidPlan, "cudf_query_engine.strict_bounded_contract_report", message)
}

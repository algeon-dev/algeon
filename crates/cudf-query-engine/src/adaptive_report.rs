//! Adaptive bounded-pipeline admission and report rows.
//!
//! Adaptive reports are intentionally not strict contract rows. They may name
//! strict gaps as missing contracts, but they do not represent strict readiness
//! evidence and must never use `execution_mode=strict_bounded`.

use std::fmt;

use crate::{
    adaptive_runtime::AdaptiveMemoryHistorySummary,
    capability::{
        CapabilityAnalysis, CapabilityRejectionOwner, ContractQuality, ContractStatus, EdgeConsumer,
        ReservationRequirement, ReservationRequirementScope, ResultSinkKind, lower_level_unresolved_preflight_gap,
    },
    error::{QueryError, QueryErrorCode, Result},
    exec::QueryOutputMode,
    mode::AdaptiveExecutionMode,
    plan::{QueryPlan, ResultSink},
    runtime::ReservationKind,
};

pub const ADAPTIVE_DIAGNOSTIC_CONSTRUCTION_FAILED_REASON: &str = "adaptive_diagnostic_construction_failed";

/// Report for one adaptive bounded-pipeline admission decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdaptivePipelineReport {
    execution_mode: AdaptiveExecutionMode,
    row: AdaptivePipelineReportRow,
}

impl AdaptivePipelineReport {
    pub fn from_admission_parts(
        analysis: &CapabilityAnalysis,
        plan: &QueryPlan,
        candidate_status: AdaptiveCandidateStatus,
        adaptive_strategy: AdaptiveStrategy,
        runtime_evidence: AdaptiveAdmissionRuntimeEvidence,
        rejection_reason: Option<&'static str>,
    ) -> Result<Self> {
        let execution_mode = match candidate_status {
            AdaptiveCandidateStatus::Admitted => AdaptiveExecutionMode::Adaptive,
            AdaptiveCandidateStatus::Rejected => AdaptiveExecutionMode::UnsupportedAdaptive,
        };
        let row = AdaptivePipelineReportRow {
            execution_mode,
            candidate_status,
            missing_strict_contracts: missing_strict_contracts(
                analysis,
                candidate_status == AdaptiveCandidateStatus::Rejected,
            ),
            adaptive_strategy,
            runtime_actions_allowed: runtime_evidence.runtime_actions_allowed,
            memory_history_evidence: runtime_evidence.memory_history_evidence,
            runtime_action_counts: AdaptiveRuntimeActionCounts::default(),
            final_sink_boundary: AdaptiveFinalSinkBoundary::for_sink(plan.sink()),
            rejection_reason,
        };
        Self::new(execution_mode, row)
    }

    pub fn execution_mode(&self) -> AdaptiveExecutionMode {
        self.execution_mode
    }

    pub fn row(&self) -> &AdaptivePipelineReportRow {
        &self.row
    }

    pub fn stable_kv_row(&self) -> String {
        self.row.to_stable_kv_row()
    }

    pub fn diagnostic_failure(plan: &QueryPlan, rejection_reason: &'static str) -> Self {
        let execution_mode = AdaptiveExecutionMode::UnsupportedAdaptive;
        let row = AdaptivePipelineReportRow {
            execution_mode,
            candidate_status: AdaptiveCandidateStatus::Rejected,
            missing_strict_contracts: Vec::new(),
            adaptive_strategy: AdaptiveStrategy::Unsupported,
            runtime_actions_allowed: AdaptiveRuntimeActionsAllowed::default(),
            memory_history_evidence: AdaptiveMemoryHistoryEvidence::NotUsed,
            runtime_action_counts: AdaptiveRuntimeActionCounts::default(),
            final_sink_boundary: AdaptiveFinalSinkBoundary::for_sink(plan.sink()),
            rejection_reason: Some(rejection_reason),
        };
        Self { execution_mode, row }
    }

    fn new(execution_mode: AdaptiveExecutionMode, row: AdaptivePipelineReportRow) -> Result<Self> {
        if row.execution_mode != execution_mode {
            return Err(adaptive_report_error("adaptive report row has inconsistent execution mode")
                .with_fact("row_execution_mode", row.execution_mode.as_str())
                .with_fact("report_execution_mode", execution_mode.as_str()));
        }
        match (execution_mode, row.candidate_status) {
            (AdaptiveExecutionMode::Adaptive, AdaptiveCandidateStatus::Admitted)
            | (AdaptiveExecutionMode::UnsupportedAdaptive, AdaptiveCandidateStatus::Rejected) => {
                Ok(Self { execution_mode, row })
            },
            _ => Err(adaptive_report_error("adaptive report mode does not match candidate status")
                .with_fact("row_execution_mode", row.execution_mode.as_str())
                .with_fact("candidate_status", row.candidate_status.as_str())),
        }
    }
}

/// One report row for adaptive admission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdaptivePipelineReportRow {
    pub execution_mode: AdaptiveExecutionMode,
    pub candidate_status: AdaptiveCandidateStatus,
    pub missing_strict_contracts: Vec<AdaptiveMissingStrictContract>,
    pub adaptive_strategy: AdaptiveStrategy,
    pub runtime_actions_allowed: AdaptiveRuntimeActionsAllowed,
    pub memory_history_evidence: AdaptiveMemoryHistoryEvidence,
    pub runtime_action_counts: AdaptiveRuntimeActionCounts,
    pub final_sink_boundary: AdaptiveFinalSinkBoundary,
    pub rejection_reason: Option<&'static str>,
}

impl AdaptivePipelineReportRow {
    pub fn to_stable_kv_row(&self) -> String {
        format!(
            "execution_mode={},candidate_status={},missing_strict_contracts={},adaptive_strategy={},runtime_actions_allowed={},memory_history_evidence={},retry_count={},spill_count={},downgrade_count={},final_sink_boundary={},materialization_mode={},rejection_reason={}",
            self.execution_mode.as_str(),
            self.candidate_status.as_str(),
            format_missing_strict_contracts(&self.missing_strict_contracts),
            self.adaptive_strategy.as_str(),
            self.runtime_actions_allowed.to_stable_value(),
            self.memory_history_evidence.to_stable_value(),
            self.runtime_action_counts.retry_count,
            self.runtime_action_counts.spill_count,
            self.runtime_action_counts.downgrade_count,
            self.final_sink_boundary.sink_kind.as_str(),
            self.final_sink_boundary.materialization_mode.as_str(),
            self.rejection_reason.unwrap_or("none"),
        )
    }
}

/// Adaptive runtime evidence read during admission.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct AdaptiveAdmissionRuntimeEvidence {
    pub runtime_actions_allowed: AdaptiveRuntimeActionsAllowed,
    pub memory_history_evidence: AdaptiveMemoryHistoryEvidence,
}

/// Whether adaptive admission currently accepts the candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AdaptiveCandidateStatus {
    Admitted,
    Rejected,
}

impl AdaptiveCandidateStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Admitted => "admitted",
            Self::Rejected => "rejected",
        }
    }
}

/// Strategy selected by adaptive admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AdaptiveStrategy {
    ConservativeCompiledPipeline,
    Unsupported,
}

impl AdaptiveStrategy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ConservativeCompiledPipeline => "conservative_compiled_pipeline",
            Self::Unsupported => "unsupported",
        }
    }
}

/// Runtime actions that a future adaptive executor may take for this candidate.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct AdaptiveRuntimeActionsAllowed {
    pub memory_history_lookup: bool,
    pub oom_retry: bool,
    pub source_chunk_adjustment: bool,
    pub spill_to_host: bool,
    pub spill_to_disk: bool,
    pub downgrade_to_host_materialized: bool,
    pub partial_admission: bool,
}

impl AdaptiveRuntimeActionsAllowed {
    pub fn to_stable_value(self) -> String {
        let mut actions = Vec::new();
        if self.memory_history_lookup {
            actions.push("memory_history_lookup");
        }
        if self.oom_retry {
            actions.push("oom_retry");
        }
        if self.source_chunk_adjustment {
            actions.push("source_chunk_adjustment");
        }
        if self.spill_to_host {
            actions.push("spill_to_host");
        }
        if self.spill_to_disk {
            actions.push("spill_to_disk");
        }
        if self.downgrade_to_host_materialized {
            actions.push("downgrade_to_host_materialized");
        }
        if self.partial_admission {
            actions.push("partial_admission");
        }
        if actions.is_empty() { "none".to_string() } else { actions.join(",") }
    }
}

/// Adaptive memory-history evidence used by admission.
#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub enum AdaptiveMemoryHistoryEvidence {
    #[default]
    NotUsed,
    LookupMiss,
    Observed(AdaptiveMemoryHistorySummary),
}

impl AdaptiveMemoryHistoryEvidence {
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::NotUsed => "not_used",
            Self::LookupMiss => "lookup_miss",
            Self::Observed(_) => "observed",
        }
    }

    pub fn lookup_attempted(&self) -> bool {
        !matches!(self, Self::NotUsed)
    }

    pub fn lookup_hit(&self) -> bool {
        matches!(self, Self::Observed(_))
    }

    pub fn to_stable_value(&self) -> String {
        match self {
            Self::NotUsed => "not_used".to_string(),
            Self::LookupMiss => "lookup_miss".to_string(),
            Self::Observed(summary) => format!(
                "observed:observation_count={}|resident_dataframe_peak_bytes={}|reservation_granted_bytes={}|source_chunks_committed={}",
                summary.observation_count,
                summary
                    .max_resident_dataframe_peak_bytes
                    .map_or_else(|| "unknown".to_string(), |bytes| bytes.to_string()),
                summary.max_reservation_granted_bytes,
                summary.max_source_chunks_committed
            ),
        }
    }
}

/// Runtime action counters owned by adaptive mode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct AdaptiveRuntimeActionCounts {
    pub retry_count: u64,
    pub spill_count: u64,
    pub downgrade_count: u64,
}

/// Final sink and materialization labels for adaptive rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdaptiveFinalSinkBoundary {
    pub sink_kind: AdaptiveSinkKind,
    pub materialization_mode: QueryOutputMode,
}

impl AdaptiveFinalSinkBoundary {
    fn for_sink(sink: &ResultSink) -> Self {
        let sink_kind = match sink {
            ResultSink::HostArrow { .. } => AdaptiveSinkKind::HostArrow,
            ResultSink::GpuResident { .. } => AdaptiveSinkKind::GpuResident,
        };
        Self { sink_kind, materialization_mode: QueryOutputMode::for_sink(sink) }
    }
}

/// Sink kind used by adaptive rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AdaptiveSinkKind {
    HostArrow,
    GpuResident,
}

impl AdaptiveSinkKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::HostArrow => "host_arrow",
            Self::GpuResident => "gpu_resident",
        }
    }
}

/// A strict bounded gap referenced by an adaptive row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdaptiveMissingStrictContract {
    pub contract_scope: ReservationRequirementScope,
    pub owner: String,
    pub reservation_kind: ReservationKind,
    pub memory_boundary: &'static str,
    pub required_contract: &'static str,
    pub contract_status: ContractStatus,
    pub contract_quality: ContractQuality,
    pub unresolved_preflight_gap: Option<&'static str>,
}

impl AdaptiveMissingStrictContract {
    fn from_requirement(owner: String, requirement: &ReservationRequirement) -> Option<Self> {
        (requirement.status != ContractStatus::Modeled).then(|| Self {
            contract_scope: requirement.scope,
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

    fn from_rejection_projection(analysis: &CapabilityAnalysis) -> Option<Self> {
        let projection = analysis.first_rejection_projection()?;
        Some(Self {
            contract_scope: projection.contract_scope,
            owner: capability_rejection_owner_label(&projection.owner),
            reservation_kind: ReservationKind::CapabilityContract,
            memory_boundary: projection.memory_boundary,
            required_contract: projection.reason,
            contract_status: ContractStatus::RejectedByCapability,
            contract_quality: ContractQuality::CapabilityProof,
            unresolved_preflight_gap: None,
        })
    }

    fn to_stable_value(&self) -> String {
        format!(
            "scope={},owner={},reservation_kind={},memory_boundary={},required_contract={},contract_status={},contract_quality={},unresolved_preflight_gap={}",
            self.contract_scope.as_str(),
            stable_value(&self.owner),
            self.reservation_kind.as_str(),
            self.memory_boundary,
            self.required_contract,
            self.contract_status.as_str(),
            self.contract_quality.as_str(),
            self.unresolved_preflight_gap.unwrap_or("none"),
        )
    }
}

fn missing_strict_contracts(
    analysis: &CapabilityAnalysis,
    include_rejection_projection: bool,
) -> Vec<AdaptiveMissingStrictContract> {
    let mut missing = Vec::new();
    for node in &analysis.nodes {
        missing.extend(node.local_requirements.iter().filter_map(|requirement| {
            AdaptiveMissingStrictContract::from_requirement(node.node_id.to_string(), requirement)
        }));
    }
    for edge in &analysis.edges {
        let owner = format!("edge:{}->{}", edge.producer, edge_consumer_label(edge.consumer));
        missing.extend(
            edge.reservation_requirements
                .iter()
                .filter_map(|requirement| AdaptiveMissingStrictContract::from_requirement(owner.clone(), requirement)),
        );
    }
    for invariant in &analysis.invariants {
        let owner = format!("invariant:{}", invariant.invariant_id.index());
        missing.extend(
            invariant
                .reservation_requirements
                .iter()
                .filter_map(|requirement| AdaptiveMissingStrictContract::from_requirement(owner.clone(), requirement)),
        );
    }
    if missing.is_empty() {
        missing.extend(analysis.requirements.iter().filter_map(|requirement| {
            AdaptiveMissingStrictContract::from_requirement("analysis:flattened_requirement".to_string(), requirement)
        }));
    }
    if include_rejection_projection
        && missing.is_empty()
        && let Some(rejection) = AdaptiveMissingStrictContract::from_rejection_projection(analysis)
    {
        missing.push(rejection);
    }
    missing
}

fn format_missing_strict_contracts(missing: &[AdaptiveMissingStrictContract]) -> String {
    if missing.is_empty() {
        return "none".to_string();
    }
    missing.iter().map(AdaptiveMissingStrictContract::to_stable_value).collect::<Vec<_>>().join(";")
}

fn edge_consumer_label(consumer: EdgeConsumer) -> String {
    match consumer {
        EdgeConsumer::Node(node_id) => node_id.to_string(),
        EdgeConsumer::Sink(ResultSinkKind::HostArrow) => "sink:host_arrow".to_string(),
        EdgeConsumer::Sink(ResultSinkKind::GpuResident) => "sink:gpu_resident".to_string(),
    }
}

fn capability_rejection_owner_label(owner: &CapabilityRejectionOwner) -> String {
    match owner {
        CapabilityRejectionOwner::Node(node_id) => node_id.to_string(),
        CapabilityRejectionOwner::Edge { producer, consumer } => {
            format!("edge:{producer}->{}", edge_consumer_label(*consumer))
        },
        CapabilityRejectionOwner::Invariant(invariant_id) => format!("invariant:{}", invariant_id.index()),
        CapabilityRejectionOwner::Admission(admission) => format!("admission:{admission}"),
    }
}

fn stable_value(value: &str) -> String {
    value.replace(['\t', '\n', '\r', '|', ';'], " ")
}

fn adaptive_report_error(message: &'static str) -> QueryError {
    QueryError::new(QueryErrorCode::InvalidPlan, "cudf_query_engine.adaptive_pipeline_report", message)
}

impl fmt::Display for AdaptivePipelineReportRow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_stable_kv_row())
    }
}

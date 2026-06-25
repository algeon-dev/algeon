//! Adaptive bounded-pipeline admission and report rows.
//!
//! Adaptive reports are intentionally not strict contract rows. They may name
//! strict gaps as missing contracts, but they do not represent strict readiness
//! evidence and must never use `execution_mode=strict_bounded`.

use std::fmt;

use serde_json::{Value, json};

use crate::{
    adaptive_runtime::AdaptiveMemoryHistorySummary,
    capability::{
        CapabilityAnalysis, CapabilityRejectionOwner, CapabilityRejectionProjection, ContractQuality, ContractStatus,
        EdgeConsumer, ReservationRequirement, ReservationRequirementScope, ResultSinkKind,
        lower_level_unresolved_preflight_gap,
    },
    error::{QueryError, QueryErrorCode, Result},
    exec::QueryOutputMode,
    metrics::{QueryId, TraceId},
    mode::AdaptiveExecutionMode,
    plan::{QueryPlan, ResultSink},
    runtime::ReservationKind,
};

/// Stable rejection reason used when adaptive diagnostics cannot be constructed.
pub const ADAPTIVE_DIAGNOSTIC_CONSTRUCTION_FAILED_REASON: &str = "adaptive_diagnostic_construction_failed";

/// Report for one adaptive bounded-pipeline admission decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdaptivePipelineReport {
    query_id: Option<QueryId>,
    trace_id: Option<TraceId>,
    execution_mode: AdaptiveExecutionMode,
    row: AdaptivePipelineReportRow,
}

impl AdaptivePipelineReport {
    /// Build an adaptive report from capability analysis and runtime evidence.
    ///
    /// Admitted candidates use `execution_mode=adaptive`; rejected candidates
    /// use `execution_mode=unsupported_adaptive`. Strict contract gaps remain
    /// explanatory evidence only.
    pub fn from_admission_parts(
        analysis: &CapabilityAnalysis,
        plan: &QueryPlan,
        candidate_status: AdaptiveCandidateStatus,
        adaptive_strategy: AdaptiveStrategy,
        runtime_evidence: AdaptiveAdmissionRuntimeEvidence,
        rejection_reason: Option<&'static str>,
        rejection_projection: Option<&CapabilityRejectionProjection>,
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
                rejection_projection,
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

    /// Adaptive execution mode represented by this report.
    pub fn execution_mode(&self) -> AdaptiveExecutionMode {
        self.execution_mode
    }

    /// Borrow the stable row payload.
    pub fn row(&self) -> &AdaptivePipelineReportRow {
        &self.row
    }

    /// Serialize the row as stable comma-separated key/value diagnostics.
    pub fn stable_kv_row(&self) -> String {
        self.row.to_stable_kv_row()
    }

    /// Serializes this adaptive pipeline report as named JSON.
    pub fn to_json(&self) -> Value {
        json!({
            "schema": "nexus_query_engine.adaptive_pipeline_report.v1",
            "query_id": self.query_id.as_ref().map(QueryId::as_str),
            "trace_id": self.trace_id.as_ref().map(TraceId::as_str),
            "triage": self.triage_summary().to_json(),
            "execution_mode": self.execution_mode.as_str(),
            "row": self.row.to_json(),
        })
    }

    /// Returns the query-wide diagnostic correlation id, when assigned.
    pub fn query_id(&self) -> Option<&QueryId> {
        self.query_id.as_ref()
    }

    /// Returns the optional parent trace id.
    pub fn trace_id(&self) -> Option<&TraceId> {
        self.trace_id.as_ref()
    }

    /// Returns this adaptive report with a query id assigned.
    #[must_use]
    pub fn with_query_id(mut self, query_id: impl Into<QueryId>) -> Self {
        self.query_id = Some(query_id.into());
        self
    }

    /// Returns this adaptive report with a parent trace id assigned.
    #[must_use]
    pub fn with_trace_id(mut self, trace_id: impl Into<TraceId>) -> Self {
        self.trace_id = Some(trace_id.into());
        self
    }

    /// Build a rejected report when diagnostics themselves fail.
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
        Self { query_id: None, trace_id: None, execution_mode, row }
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
                Ok(Self { query_id: None, trace_id: None, execution_mode, row })
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
    /// Adaptive or unsupported-adaptive mode; never strict bounded.
    pub execution_mode: AdaptiveExecutionMode,
    /// Whether adaptive admission accepted the candidate.
    pub candidate_status: AdaptiveCandidateStatus,
    /// Strict bounded gaps referenced as explanatory evidence.
    pub missing_strict_contracts: Vec<AdaptiveMissingStrictContract>,
    /// Adaptive strategy selected for the candidate.
    pub adaptive_strategy: AdaptiveStrategy,
    /// Runtime actions this adaptive strategy may take.
    pub runtime_actions_allowed: AdaptiveRuntimeActionsAllowed,
    /// Memory-history evidence consulted during admission.
    pub memory_history_evidence: AdaptiveMemoryHistoryEvidence,
    /// Adaptive runtime actions observed or promised for this row.
    pub runtime_action_counts: AdaptiveRuntimeActionCounts,
    /// Final sink and materialization mode for the candidate.
    pub final_sink_boundary: AdaptiveFinalSinkBoundary,
    /// Stable rejection reason, or `None` for admitted rows.
    pub rejection_reason: Option<&'static str>,
}

impl AdaptivePipelineReportRow {
    /// Serialize this row using stable key names and sentinel values.
    ///
    /// Empty lists use `none`; absent rejection reasons use `none`.
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

    /// Serializes this adaptive pipeline row as named JSON.
    pub fn to_json(&self) -> Value {
        json!({
            "execution_mode": self.execution_mode.as_str(),
            "candidate_status": self.candidate_status.as_str(),
            "missing_strict_contracts": self.missing_strict_contracts.iter().map(AdaptiveMissingStrictContract::to_json).collect::<Vec<_>>(),
            "adaptive_strategy": self.adaptive_strategy.as_str(),
            "runtime_actions_allowed": self.runtime_actions_allowed.to_json(),
            "memory_history_evidence": self.memory_history_evidence.to_json(),
            "runtime_action_counts": self.runtime_action_counts.to_json(),
            "final_sink_boundary": self.final_sink_boundary.to_json(),
            "rejection_reason": self.rejection_reason,
        })
    }
}

/// Adaptive runtime evidence read during admission.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct AdaptiveAdmissionRuntimeEvidence {
    /// Runtime actions allowed by the selected adaptive strategy.
    pub runtime_actions_allowed: AdaptiveRuntimeActionsAllowed,
    /// Query-level memory-history evidence used by admission.
    pub memory_history_evidence: AdaptiveMemoryHistoryEvidence,
}

/// Whether adaptive admission currently accepts the candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AdaptiveCandidateStatus {
    /// Candidate can execute through the adaptive bounded pipeline.
    Admitted,
    /// Candidate lacks required adaptive support.
    Rejected,
}

impl AdaptiveCandidateStatus {
    /// Stable report string for this status.
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
    /// Run the compiled pipeline with conservative adaptive evidence.
    ConservativeCompiledPipeline,
    /// No adaptive execution strategy is available for the candidate.
    Unsupported,
}

impl AdaptiveStrategy {
    /// Stable report string for this strategy.
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
    /// Strategy may consult memory history during admission.
    pub memory_history_lookup: bool,
    /// Strategy may retry after out-of-memory/resource-exhausted failures.
    pub oom_retry: bool,
    /// Strategy may adjust source chunk sizing.
    pub source_chunk_adjustment: bool,
    /// Strategy may spill intermediate data to host memory.
    pub spill_to_host: bool,
    /// Strategy may spill intermediate data to disk.
    pub spill_to_disk: bool,
    /// Strategy may downgrade to host-materialized execution boundaries.
    pub downgrade_to_host_materialized: bool,
    /// Strategy may admit only part of the candidate.
    pub partial_admission: bool,
}

impl AdaptiveRuntimeActionsAllowed {
    /// Stable comma-separated action list, or `none`.
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

    /// Serializes allowed runtime actions as named booleans.
    pub fn to_json(self) -> Value {
        json!({
            "memory_history_lookup": self.memory_history_lookup,
            "oom_retry": self.oom_retry,
            "source_chunk_adjustment": self.source_chunk_adjustment,
            "spill_to_host": self.spill_to_host,
            "spill_to_disk": self.spill_to_disk,
            "downgrade_to_host_materialized": self.downgrade_to_host_materialized,
            "partial_admission": self.partial_admission,
        })
    }
}

/// Adaptive memory-history evidence used by admission.
#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub enum AdaptiveMemoryHistoryEvidence {
    #[default]
    /// Memory history was not consulted.
    NotUsed,
    /// Memory history was consulted but no matching observation existed.
    LookupMiss,
    /// Matching memory history was found.
    Observed(AdaptiveMemoryHistorySummary),
}

impl AdaptiveMemoryHistoryEvidence {
    /// Stable report string for the evidence class.
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::NotUsed => "not_used",
            Self::LookupMiss => "lookup_miss",
            Self::Observed(_) => "observed",
        }
    }

    /// Whether admission attempted a memory-history lookup.
    pub fn lookup_attempted(&self) -> bool {
        !matches!(self, Self::NotUsed)
    }

    /// Whether admission found matching memory-history evidence.
    pub fn lookup_hit(&self) -> bool {
        matches!(self, Self::Observed(_))
    }

    /// Stable report value including summary counters for observed history.
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

    /// Serializes memory-history evidence as named JSON.
    pub fn to_json(&self) -> Value {
        match self {
            Self::NotUsed => json!({
                "status": self.as_str(),
                "lookup_attempted": false,
                "lookup_hit": false,
                "summary": null,
            }),
            Self::LookupMiss => json!({
                "status": self.as_str(),
                "lookup_attempted": true,
                "lookup_hit": false,
                "summary": null,
            }),
            Self::Observed(summary) => json!({
                "status": self.as_str(),
                "lookup_attempted": true,
                "lookup_hit": true,
                "summary": {
                    "observation_count": summary.observation_count,
                    "max_resident_dataframe_peak_bytes": summary.max_resident_dataframe_peak_bytes,
                    "max_reservation_granted_bytes": summary.max_reservation_granted_bytes,
                    "max_source_chunks_committed": summary.max_source_chunks_committed,
                },
            }),
        }
    }
}

/// Runtime action counters owned by adaptive mode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct AdaptiveRuntimeActionCounts {
    /// Number of adaptive retry actions.
    pub retry_count: u64,
    /// Number of adaptive spill actions.
    pub spill_count: u64,
    /// Number of adaptive downgrade actions.
    pub downgrade_count: u64,
}

impl AdaptiveRuntimeActionCounts {
    /// Serializes runtime action counters as named JSON.
    pub fn to_json(self) -> Value {
        json!({
            "retry_count": self.retry_count,
            "spill_count": self.spill_count,
            "downgrade_count": self.downgrade_count,
        })
    }
}

/// Final sink and materialization labels for adaptive rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdaptiveFinalSinkBoundary {
    /// Sink kind requested by the native query plan.
    pub sink_kind: AdaptiveSinkKind,
    /// Output materialization mode derived from the sink.
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

    /// Serializes the final sink boundary as named JSON.
    pub fn to_json(&self) -> Value {
        json!({
            "sink_kind": self.sink_kind.as_str(),
            "materialization_mode": self.materialization_mode.as_str(),
        })
    }
}

/// Sink kind used by adaptive rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AdaptiveSinkKind {
    /// Host Arrow output sink.
    HostArrow,
    /// GPU-resident output sink.
    GpuResident,
}

impl AdaptiveSinkKind {
    /// Stable report string for this sink kind.
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
    /// Scope of the missing strict contract.
    pub contract_scope: ReservationRequirementScope,
    /// Stable owner label such as a node, edge, invariant, or admission name.
    pub owner: String,
    /// Reservation kind associated with the missing contract.
    pub reservation_kind: ReservationKind,
    /// Stable memory-boundary string.
    pub memory_boundary: &'static str,
    /// Required lower-level or query-engine contract string.
    pub required_contract: &'static str,
    /// Strict contract status explaining the gap.
    pub contract_status: ContractStatus,
    /// Quality of the strict contract evidence.
    pub contract_quality: ContractQuality,
    /// Lower-level preflight gap when the contract is blocked below this crate.
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

    fn from_rejection_projection(projection: &CapabilityRejectionProjection) -> Self {
        Self {
            contract_scope: projection.contract_scope,
            owner: capability_rejection_owner_label(&projection.owner),
            reservation_kind: ReservationKind::CapabilityContract,
            memory_boundary: projection.memory_boundary,
            required_contract: projection.reason,
            contract_status: ContractStatus::RejectedByCapability,
            contract_quality: ContractQuality::CapabilityProof,
            unresolved_preflight_gap: None,
        }
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

    /// Serializes this strict-contract gap as named JSON.
    pub fn to_json(&self) -> Value {
        json!({
            "contract_scope": self.contract_scope.as_str(),
            "owner": self.owner,
            "reservation_kind": self.reservation_kind.as_str(),
            "memory_boundary": self.memory_boundary,
            "required_contract": self.required_contract,
            "contract_status": self.contract_status.as_str(),
            "contract_quality": self.contract_quality.as_str(),
            "unresolved_preflight_gap": self.unresolved_preflight_gap,
        })
    }
}

fn missing_strict_contracts(
    analysis: &CapabilityAnalysis,
    include_rejection_projection: bool,
    rejection_projection: Option<&CapabilityRejectionProjection>,
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
    if include_rejection_projection && missing.is_empty() {
        let analysis_projection;
        let projection = match rejection_projection {
            Some(projection) => Some(projection),
            None => {
                analysis_projection = analysis.first_rejection_projection();
                analysis_projection.as_ref()
            },
        };
        if let Some(projection) = projection {
            missing.push(AdaptiveMissingStrictContract::from_rejection_projection(projection));
        }
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
    QueryError::new(QueryErrorCode::InvalidPlan, "nexus_query_engine.adaptive_pipeline_report", message)
}

impl fmt::Display for AdaptivePipelineReportRow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_stable_kv_row())
    }
}

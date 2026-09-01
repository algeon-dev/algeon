//! Public GPU coverage DTO and JSON surface.

use datafusion::physical_plan::ExecutionPlan;
use serde_json::{Value, json};

use super::preflight::QueryGpuRejection;
use crate::{
    final_plan_observation::FinalPlanObservation,
    physical_plan::physical_plan_shape,
    planner::{FinalPlanDisposition, NativePlanningDiagnostics},
    report::NativeCandidateCostEvidence,
};

/// Structured answer for "will this query's planned path avoid DataFusion CPU execution?"
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryGpuCoverage {
    pub overall: QueryGpuCoverageOverall,
    /// Final disposition of the exact physical plan, or `Rejected` for a
    /// terminal planning result with no executable plan.
    pub final_disposition: FinalPlanDisposition,
    pub domains: QueryGpuCoverageDomains,
    /// Native candidate selections retained independently from final-plan
    /// disposition.
    pub candidate_outcomes: Vec<QueryGpuCandidateOutcome>,
    pub cugraph_calls: Vec<QueryGpuCugraphCall>,
    pub runtime_caveats: Vec<QueryGpuRuntimeCaveat>,
    pub planning_did_io: bool,
    pub validated_under: QueryGpuValidatedUnder,
    pub confidence: QueryGpuCoverageConfidence,
    /// Root captured from the authoritative DataFusion physical plan before
    /// the native optimizer attempted replacement.
    pub input_physical_plan_root: Option<String>,
    /// Shape captured from the authoritative DataFusion physical plan before
    /// the native optimizer attempted replacement.
    pub input_physical_plan_shape: Option<String>,
    /// Root of the completed executable plan after native optimization.
    pub physical_plan_root: Option<String>,
    /// Shape of the completed executable plan after native optimization.
    pub physical_plan_shape: Option<String>,
}

impl QueryGpuCoverage {
    pub(super) fn planned(
        validated_under: QueryGpuValidatedUnder,
        plan: &dyn ExecutionPlan,
        observation: FinalPlanObservation,
        candidate_outcomes: Vec<QueryGpuCandidateOutcome>,
        cugraph_calls: Vec<QueryGpuCugraphCall>,
        planning_did_io: bool,
        diagnostics: Option<&NativePlanningDiagnostics>,
    ) -> Self {
        debug_assert_ne!(observation.disposition, FinalPlanDisposition::Rejected);
        let overall = match observation.disposition {
            FinalPlanDisposition::Native => QueryGpuCoverageOverall::NoCpuInPlannedPath,
            FinalPlanDisposition::Mixed => QueryGpuCoverageOverall::PartialGpuPlanned {
                gpu_fragments: observation.native_fragment_count + observation.explicit_gpu_island_count,
                cpu_boundaries: observation.datafusion_boundary_count,
            },
            FinalPlanDisposition::DataFusion => QueryGpuCoverageOverall::CpuOnly,
            FinalPlanDisposition::Rejected => unreachable!("an executable final plan cannot be rejected"),
        };
        let runtime_caveats = super::physical::runtime_caveats(observation, &cugraph_calls, &validated_under);
        let domains = QueryGpuCoverageDomains {
            native_fragment_count: observation.native_fragment_count,
            explicit_gpu_island_count: observation.explicit_gpu_island_count,
            datafusion_boundary_count: observation.datafusion_boundary_count,
            host_materialization_count: observation.host_materialization_count,
        };
        Self {
            overall,
            final_disposition: observation.disposition,
            domains,
            candidate_outcomes,
            cugraph_calls,
            runtime_caveats,
            planning_did_io,
            validated_under,
            confidence: QueryGpuCoverageConfidence::PlanTimeOnly,
            input_physical_plan_root: diagnostics
                .map(|diagnostics| diagnostics.input_plan_root.clone())
                .filter(|root| !root.is_empty()),
            input_physical_plan_shape: diagnostics
                .map(|diagnostics| diagnostics.input_plan_shape.clone())
                .filter(|shape| !shape.is_empty()),
            physical_plan_root: Some(plan.name().to_string()),
            physical_plan_shape: Some(physical_plan_shape(plan)),
        }
    }

    pub(super) fn rejected(validated_under: QueryGpuValidatedUnder, rejection: QueryGpuRejection) -> Self {
        Self::rejected_with_diagnostics(validated_under, rejection, None, false)
    }

    pub(super) fn rejected_with_diagnostics(
        validated_under: QueryGpuValidatedUnder,
        rejection: QueryGpuRejection,
        diagnostics: Option<&NativePlanningDiagnostics>,
        planning_did_io: bool,
    ) -> Self {
        let overall = QueryGpuCoverageOverall::Rejected {
            phase: rejection.phase,
            reason_code: rejection.reason_code,
            retry_advice: QueryGpuRetryAdviceAvailability::NoSealedAttemptReport,
            source_error_status: rejection.source_error_status.map(ToString::to_string),
            detail: rejection.detail,
        };
        Self {
            overall,
            final_disposition: FinalPlanDisposition::Rejected,
            domains: QueryGpuCoverageDomains::default(),
            candidate_outcomes: diagnostics
                .map(|diagnostics| super::physical::candidate_outcomes_from_report(&diagnostics.report))
                .unwrap_or_default(),
            cugraph_calls: Vec::new(),
            runtime_caveats: Vec::new(),
            planning_did_io,
            validated_under,
            confidence: QueryGpuCoverageConfidence::PlanTimeOnly,
            input_physical_plan_root: diagnostics
                .map(|diagnostics| diagnostics.input_plan_root.clone())
                .filter(|root| !root.is_empty()),
            input_physical_plan_shape: diagnostics
                .map(|diagnostics| diagnostics.input_plan_shape.clone())
                .filter(|shape| !shape.is_empty()),
            physical_plan_root: None,
            physical_plan_shape: None,
        }
    }

    /// Serialize the verdict to a structured JSON shape.
    pub fn to_json(&self) -> Value {
        json!({
            "overall": self.overall.to_json(),
            "final_disposition": self.final_disposition.as_str(),
            "domains": self.domains.to_json(),
            "candidate_outcomes": self.candidate_outcomes.iter().map(QueryGpuCandidateOutcome::to_json).collect::<Vec<_>>(),
            "cugraph_calls": self.cugraph_calls.iter().map(QueryGpuCugraphCall::to_json).collect::<Vec<_>>(),
            "runtime_caveats": self.runtime_caveats.iter().map(QueryGpuRuntimeCaveat::to_json).collect::<Vec<_>>(),
            "planning_did_io": self.planning_did_io,
            "validated_under": self.validated_under.to_json(),
            "confidence": self.confidence.as_str(),
            "input_physical_plan_root": self.input_physical_plan_root,
            "input_physical_plan_shape": self.input_physical_plan_shape,
            "physical_plan_root": self.physical_plan_root,
            "physical_plan_shape": self.physical_plan_shape,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryGpuCoverageOverall {
    NoCpuInPlannedPath,
    PartialGpuPlanned {
        gpu_fragments: usize,
        cpu_boundaries: usize,
    },
    CpuOnly,
    Rejected {
        phase: QueryGpuRejectionPhase,
        reason_code: String,
        /// A planning-only rejection has no execution attempt and therefore
        /// cannot expose a sealed retry decision.
        retry_advice: QueryGpuRetryAdviceAvailability,
        /// Coarse source status remains diagnostic context only.
        source_error_status: Option<String>,
        detail: String,
    },
}

/// Whether a public surface can truthfully expose sealed attempt retry advice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryGpuRetryAdviceAvailability {
    NoSealedAttemptReport,
}

impl QueryGpuRetryAdviceAvailability {
    pub const fn availability(self) -> &'static str {
        match self {
            Self::NoSealedAttemptReport => "unavailable",
        }
    }

    pub const fn reason(self) -> &'static str {
        match self {
            Self::NoSealedAttemptReport => "no_sealed_attempt_report",
        }
    }
}

impl QueryGpuCoverageOverall {
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::NoCpuInPlannedPath => "no_cpu_in_planned_path",
            Self::PartialGpuPlanned { .. } => "partial_gpu_planned",
            Self::CpuOnly => "cpu_only",
            Self::Rejected { .. } => "rejected",
        }
    }

    /// Stable, user-facing execution-path vocabulary shared by the UDTF and
    /// `EXPLAIN GPU` projection.
    pub const fn as_gpu_path(&self) -> &'static str {
        match self {
            Self::NoCpuInPlannedPath => "native",
            Self::PartialGpuPlanned { .. } => "partial_native",
            Self::CpuOnly => "cpu",
            Self::Rejected { .. } => "rejected",
        }
    }

    fn to_json(&self) -> Value {
        match self {
            Self::NoCpuInPlannedPath => json!({
                "verdict": self.as_str(),
                "message": "No DataFusion CPU execution is present in the planned physical path inspected.",
                "runtime_gpu_admission_unknown": true,
            }),
            Self::PartialGpuPlanned { gpu_fragments, cpu_boundaries } => json!({
                "verdict": self.as_str(),
                "gpu_fragments": gpu_fragments,
                "cpu_boundaries": cpu_boundaries,
            }),
            Self::CpuOnly => json!({ "verdict": self.as_str() }),
            Self::Rejected { phase, reason_code, retry_advice, source_error_status, detail } => json!({
                "verdict": self.as_str(),
                "phase": phase.as_str(),
                "reason_code": reason_code,
                "retry_advice": {
                    "availability": retry_advice.availability(),
                    "reason": retry_advice.reason(),
                },
                "source_error_status": source_error_status,
                "detail": detail,
            }),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryGpuRejectionPhase {
    Parsing,
    LogicalPlanning,
    PhysicalPlanning,
    PlanningIo,
    NativeLowering,
}

impl QueryGpuRejectionPhase {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Parsing => "parsing",
            Self::LogicalPlanning => "logical_planning",
            Self::PhysicalPlanning => "physical_planning",
            Self::PlanningIo => "planning_io",
            Self::NativeLowering => "native_lowering",
        }
    }
}

/// Final-plan execution domains derived from the shared observation walk.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct QueryGpuCoverageDomains {
    pub native_fragment_count: usize,
    pub explicit_gpu_island_count: usize,
    pub datafusion_boundary_count: usize,
    pub host_materialization_count: usize,
}

impl QueryGpuCoverageDomains {
    fn to_json(self) -> Value {
        json!({
            "native_fragment_count": self.native_fragment_count,
            "explicit_gpu_island_count": self.explicit_gpu_island_count,
            "datafusion_boundary_count": self.datafusion_boundary_count,
            "host_materialization_count": self.host_materialization_count,
        })
    }
}

/// Selection class for a candidate retained in coverage diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryGpuCandidateSelectionOutcome {
    Selected,
    Deferred,
    NotSupported,
    NotSelectedByCost,
}

impl QueryGpuCandidateSelectionOutcome {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Selected => "selected",
            Self::Deferred => "deferred",
            Self::NotSupported => "not_supported",
            Self::NotSelectedByCost => "not_selected_by_cost",
        }
    }
}

/// Reason category for a candidate selection result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryGpuCandidateReasonKind {
    Operator,
    Expression,
    ArrowType,
    Source,
    PlanShape,
    Cost,
}

impl QueryGpuCandidateReasonKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Operator => "operator",
            Self::Expression => "expression",
            Self::ArrowType => "arrow_type",
            Self::Source => "source",
            Self::PlanShape => "plan_shape",
            Self::Cost => "cost",
        }
    }
}

/// Candidate selection diagnostic retained independently from final coverage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryGpuCandidateOutcome {
    pub selection: QueryGpuCandidateSelectionOutcome,
    /// Stable family derived from the native candidate shape.
    pub candidate_family: String,
    /// Present only when a capability or cost decision has a stable reason.
    pub reason_code: Option<String>,
    /// Present only when a capability or cost decision has a reason category.
    pub reason_kind: Option<QueryGpuCandidateReasonKind>,
    pub detail: String,
    /// Triage category for an unselected candidate, when available.
    pub category: Option<String>,
    /// Actionable guidance for a stable capability or cost reason.
    pub remedy: Option<QueryGpuRemedy>,
    pub cost_evidence: Option<NativeCandidateCostEvidence>,
}

impl QueryGpuCandidateOutcome {
    fn to_json(&self) -> Value {
        json!({
            "outcome": self.selection.as_str(),
            "candidate_family": self.candidate_family,
            "reason_code": self.reason_code,
            "reason_kind": self.reason_kind.map(QueryGpuCandidateReasonKind::as_str),
            "detail": self.detail,
            "category": self.category,
            "remedy_code": self.remedy.map(|remedy| remedy.code),
            "remedy_kind": self.remedy.map(|remedy| remedy.kind.as_str()),
            "remedy": self.remedy.map(|remedy| remedy.guidance),
            "cost_evidence": self.cost_evidence.as_ref().map(|evidence| evidence.to_json()),
        })
    }
}

/// Stable category for a prescribed action after a candidate was not selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryGpuRemedyKind {
    UserAction,
    ConfigAction,
    EngineGap,
    CudfRsGap,
    CpuPreferred,
}

impl QueryGpuRemedyKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::UserAction => "user_action",
            Self::ConfigAction => "config_action",
            Self::EngineGap => "engine_gap",
            Self::CudfRsGap => "cudf_rs_gap",
            Self::CpuPreferred => "cpu_preferred",
        }
    }
}

/// Stable remedy code and human-readable guidance for a candidate reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueryGpuRemedy {
    pub code: &'static str,
    pub kind: QueryGpuRemedyKind,
    pub guidance: &'static str,
}

impl QueryGpuRemedy {
    pub const fn new(code: &'static str, kind: QueryGpuRemedyKind, guidance: &'static str) -> Self {
        Self { code, kind, guidance }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryGpuCugraphCall {
    pub algorithm: String,
    pub statically_valid: bool,
    pub notes: Vec<String>,
}

impl QueryGpuCugraphCall {
    fn to_json(&self) -> Value {
        json!({
            "algorithm": self.algorithm,
            "statically_valid": self.statically_valid,
            "notes": self.notes,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryGpuRuntimeCaveat {
    pub code: String,
    pub detail: String,
}

impl QueryGpuRuntimeCaveat {
    pub(super) fn new(code: impl Into<String>, detail: impl Into<String>) -> Self {
        Self { code: code.into(), detail: detail.into() }
    }

    fn to_json(&self) -> Value {
        json!({
            "code": self.code,
            "detail": self.detail,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryGpuValidatedUnder {
    pub native_optimizer_rule_installed: bool,
    pub optimizer_config_source: String,
    pub execution_mode: Option<String>,
    pub final_plan_requirements: Vec<String>,
    pub session_config_fingerprint: String,
    pub optimizer_fingerprint: String,
    pub catalog_snapshot: String,
}

impl QueryGpuValidatedUnder {
    fn to_json(&self) -> Value {
        json!({
            "native_optimizer_rule_installed": self.native_optimizer_rule_installed,
            "optimizer_config_source": self.optimizer_config_source,
            "execution_mode": self.execution_mode,
            "final_plan_requirements": self.final_plan_requirements,
            "session_config_fingerprint": self.session_config_fingerprint,
            "optimizer_fingerprint": self.optimizer_fingerprint,
            "catalog_snapshot": self.catalog_snapshot,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryGpuCoverageConfidence {
    PlanTimeOnly,
}

impl QueryGpuCoverageConfidence {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PlanTimeOnly => "plan_time_only",
        }
    }
}

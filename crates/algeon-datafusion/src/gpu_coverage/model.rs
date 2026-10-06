//! Public GPU coverage DTO and JSON surface.

use datafusion::physical_plan::ExecutionPlan;
use serde_json::{Value, json};

use super::preflight::QueryGpuRejection;
use crate::{
    physical_plan::physical_plan_shape,
    planner::{FinalPlanCoverage, FinalPlanDisposition, NativePlanningDiagnostics},
    report::{Blocking, CandidateOutcomeKind, NativeCandidateCostEvidence},
};

/// Single sum type representing whether GPU planning produced an executable plan or was rejected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryGpuPlanningOutcome {
    Planned(FinalPlanCoverage),
    Rejected(QueryGpuPlanningRejection),
}

impl QueryGpuPlanningOutcome {
    pub fn to_json(&self) -> Value {
        match self {
            Self::Planned(plan) => json!({
                "status": "planned",
                "final_plan": {
                    "disposition": plan.disposition().as_str(),
                    "native_fragment_count": plan.native_fragment_count(),
                    "explicit_gpu_island_count": plan.explicit_gpu_island_count(),
                    "datafusion_boundary_count": plan.datafusion_boundary_count(),
                    "host_materialization_count": plan.host_materialization_count(),
                    "has_gpu_runtime": plan.has_gpu_runtime(),
                },
                "rejection": null,
            }),
            Self::Rejected(rejection) => json!({
                "status": "rejected",
                "final_plan": null,
                "rejection": {
                    "phase": rejection.phase.as_str(),
                    "reason_code": &rejection.reason_code,
                    "source_error_status": &rejection.source_error_status,
                    "detail": &rejection.detail,
                },
            }),
        }
    }
}

/// Plan-time rejection details when planning fails before an executable physical plan is produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryGpuPlanningRejection {
    pub phase: QueryGpuRejectionPhase,
    pub reason_code: String,
    pub source_error_status: Option<String>,
    pub detail: String,
}

/// Structured answer for "will this query's planned path avoid DataFusion CPU execution?"
///
/// The answer is derived from planning alone, so every JSON projection reports
/// `"confidence": "plan_time_only"`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryGpuCoverage {
    pub planning_outcome: QueryGpuPlanningOutcome,
    pub candidate_outcomes: Vec<QueryGpuCandidateOutcome>,
    pub cugraph_calls: Vec<QueryGpuCugraphCall>,
    pub runtime_caveats: Vec<QueryGpuRuntimeCaveat>,
    pub planning_did_io: bool,
    pub validated_under: QueryGpuValidatedUnder,
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
        coverage: FinalPlanCoverage,
        candidate_outcomes: Vec<QueryGpuCandidateOutcome>,
        cugraph_calls: Vec<QueryGpuCugraphCall>,
        diagnostics: Option<&NativePlanningDiagnostics>,
    ) -> Self {
        let runtime_caveats = super::physical::runtime_caveats(coverage, &cugraph_calls, &validated_under);
        Self {
            planning_outcome: QueryGpuPlanningOutcome::Planned(coverage),
            candidate_outcomes,
            cugraph_calls,
            runtime_caveats,
            planning_did_io: false,
            validated_under,
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
        let planning_outcome = QueryGpuPlanningOutcome::Rejected(QueryGpuPlanningRejection {
            phase: rejection.phase,
            reason_code: rejection.reason_code,
            source_error_status: rejection.source_error_status.map(ToString::to_string),
            detail: rejection.detail,
        });
        Self {
            planning_outcome,
            candidate_outcomes: diagnostics
                .map(|diagnostics| super::physical::candidate_outcomes_from_report(&diagnostics.report))
                .unwrap_or_default(),
            cugraph_calls: Vec::new(),
            runtime_caveats: Vec::new(),
            planning_did_io,
            validated_under,
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

    pub fn planning_outcome(&self) -> &QueryGpuPlanningOutcome {
        &self.planning_outcome
    }

    pub fn final_plan_coverage(&self) -> Option<&FinalPlanCoverage> {
        match &self.planning_outcome {
            QueryGpuPlanningOutcome::Planned(coverage) => Some(coverage),
            QueryGpuPlanningOutcome::Rejected(_) => None,
        }
    }

    pub fn rejection(&self) -> Option<&QueryGpuPlanningRejection> {
        match &self.planning_outcome {
            QueryGpuPlanningOutcome::Planned(_) => None,
            QueryGpuPlanningOutcome::Rejected(rejection) => Some(rejection),
        }
    }

    pub fn gpu_path(&self) -> &'static str {
        match &self.planning_outcome {
            QueryGpuPlanningOutcome::Planned(coverage) => match coverage.disposition() {
                FinalPlanDisposition::Native => "native",
                FinalPlanDisposition::Mixed => "partial_native",
                FinalPlanDisposition::DataFusion => "cpu",
            },
            QueryGpuPlanningOutcome::Rejected(_) => "rejected",
        }
    }

    /// Serialize the verdict to a structured JSON shape.
    pub fn to_json(&self) -> Value {
        json!({
            "planning_outcome": self.planning_outcome.to_json(),
            "candidate_outcomes": self.candidate_outcomes.iter().map(QueryGpuCandidateOutcome::to_json).collect::<Vec<_>>(),
            "cugraph_calls": self.cugraph_calls.iter().map(QueryGpuCugraphCall::to_json).collect::<Vec<_>>(),
            "runtime_caveats": self.runtime_caveats.iter().map(QueryGpuRuntimeCaveat::to_json).collect::<Vec<_>>(),
            "planning_did_io": self.planning_did_io,
            "validated_under": self.validated_under.to_json(),
            "confidence": "plan_time_only",
            "input_physical_plan_root": self.input_physical_plan_root,
            "input_physical_plan_shape": self.input_physical_plan_shape,
            "physical_plan_root": self.physical_plan_root,
            "physical_plan_shape": self.physical_plan_shape,
        })
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
    pub selection: CandidateOutcomeKind,
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
    /// What blocked a not-supported candidate, when the producing site knew it.
    pub blocking: Option<Blocking>,
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
            "blocking": self.blocking.as_ref().map(Blocking::to_json),
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
    CpuPreferred,
}

impl QueryGpuRemedyKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::UserAction => "user_action",
            Self::ConfigAction => "config_action",
            Self::EngineGap => "engine_gap",
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
    /// Whether every input relation reaches the graph as a native GPU-resident
    /// producer. `false` means at least one input runs on DataFusion and is
    /// staged through host Arrow.
    pub inputs_native: bool,
    pub notes: Vec<String>,
}

impl QueryGpuCugraphCall {
    fn to_json(&self) -> Value {
        json!({
            "algorithm": self.algorithm,
            "statically_valid": self.statically_valid,
            "inputs_native": self.inputs_native,
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

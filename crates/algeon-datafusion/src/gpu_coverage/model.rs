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
    /// Planning produced an executable physical plan with the attached coverage.
    Planned(FinalPlanCoverage),
    /// Planning failed before producing an executable physical plan.
    Rejected(QueryGpuPlanningRejection),
}

impl QueryGpuPlanningOutcome {
    /// Serializes planning status and client-visible coverage or rejection details.
    ///
    /// Trusted source-chain diagnostics are excluded from this projection.
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
    /// Stage at which parsing, planning, or native validation failed.
    pub phase: QueryGpuRejectionPhase,
    /// The error code that executing the statement returns, or the stable
    /// code of a rejection only the validator makes.
    pub reason_code: String,
    /// Stable status of a typed source error, or `None` when no source status exists.
    pub source_error_status: Option<String>,
    /// The status message that executing the statement returns to its client,
    /// without the closing pointer to `EXPLAIN GPU` that a blocking plan
    /// element adds to it. A failure whose raw text the server withholds has
    /// the same static message here.
    pub detail: String,
    /// The source-chain text of the failure, for callers in the planning
    /// process, such as a developer tool that plans its own session.
    ///
    /// For a failure whose raw text the server withholds, this is that text:
    /// foreign error messages, local paths and caller identifiers. A
    /// rejection the validator makes itself repeats `detail`. No JSON, Arrow
    /// or Flight projection of the coverage carries this field, and it must
    /// not cross a trust boundary that `detail` may cross.
    pub trusted_diagnostic: String,
}

/// Structured answer for "will this query's planned path avoid DataFusion CPU execution?"
///
/// The answer is derived from planning alone, so every JSON projection reports
/// `"confidence": "plan_time_only"`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryGpuCoverage {
    /// Executable final-plan coverage or the rejection that prevented a plan.
    pub planning_outcome: QueryGpuPlanningOutcome,
    /// Native candidate decisions in optimizer discovery order.
    pub candidate_outcomes: Vec<QueryGpuCandidateOutcome>,
    /// Explicit cuGraph calls discovered during physical-plan validation.
    pub cugraph_calls: Vec<QueryGpuCugraphCall>,
    /// Execution requirements that planning alone does not verify.
    pub runtime_caveats: Vec<QueryGpuRuntimeCaveat>,
    /// Whether validation performed I/O while planning.
    pub planning_did_io: bool,
    /// Session and optimizer context captured for this validation pass.
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
            trusted_diagnostic: rejection.trusted_diagnostic,
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

    /// Borrows the planning success or rejection recorded by this validation pass.
    pub fn planning_outcome(&self) -> &QueryGpuPlanningOutcome {
        &self.planning_outcome
    }

    /// Returns final-plan coverage on planning success, or `None` after rejection.
    pub fn final_plan_coverage(&self) -> Option<&FinalPlanCoverage> {
        match &self.planning_outcome {
            QueryGpuPlanningOutcome::Planned(coverage) => Some(coverage),
            QueryGpuPlanningOutcome::Rejected(_) => None,
        }
    }

    /// Returns the rejection evidence when planning failed, or `None` on success.
    pub fn rejection(&self) -> Option<&QueryGpuPlanningRejection> {
        match &self.planning_outcome {
            QueryGpuPlanningOutcome::Planned(_) => None,
            QueryGpuPlanningOutcome::Rejected(rejection) => Some(rejection),
        }
    }

    /// Returns the report label `native`, `partial_native`, `cpu`, or `rejected`.
    ///
    /// The label describes plan-time coverage; it makes no runtime completion claim.
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
/// Planning stage at which validation stopped before obtaining an executable plan.
pub enum QueryGpuRejectionPhase {
    /// SQL text could not be parsed into a supported statement.
    Parsing,
    /// Logical plan construction or optimization failed.
    LogicalPlanning,
    /// DataFusion physical plan construction failed.
    PhysicalPlanning,
    /// Validation required planning I/O that its metadata-only contract disallows.
    PlanningIo,
    /// Native lowering or the requested final-plan coverage contract failed.
    NativeLowering,
}

impl QueryGpuRejectionPhase {
    /// Returns the stable snake-case label used in coverage JSON and report rows.
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
    /// A physical operator has no supported native representation.
    Operator,
    /// A physical expression has no supported native representation.
    Expression,
    /// An Arrow type or representation is unsupported.
    ArrowType,
    /// Source metadata or access semantics cannot be lowered.
    Source,
    /// The combined layout or semantics of plan nodes is unsupported.
    PlanShape,
    /// A native-capable candidate was declined by cost selection.
    Cost,
}

impl QueryGpuCandidateReasonKind {
    /// Returns the stable snake-case label used in coverage JSON and report rows.
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
    /// Whether the native optimizer selected, declined capability, or declined cost.
    pub selection: CandidateOutcomeKind,
    /// Stable family derived from the native candidate shape.
    pub candidate_family: String,
    /// Present only when a capability or cost decision has a stable reason.
    pub reason_code: Option<String>,
    /// Present only when a capability or cost decision has a reason category.
    pub reason_kind: Option<QueryGpuCandidateReasonKind>,
    /// Human-readable evidence explaining this candidate decision.
    pub detail: String,
    /// Triage category for an unselected candidate, when available.
    pub category: Option<String>,
    /// Actionable guidance for a stable capability or cost reason.
    pub remedy: Option<QueryGpuRemedy>,
    /// What blocked a not-supported candidate, when the producing site knew it.
    pub blocking: Option<Blocking>,
    /// Cost facts for a cost decline, or `None` for other candidate outcomes.
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
    /// Changing the query or input data can address the candidate reason.
    UserAction,
    /// Changing a supported session or backend setting can address the reason.
    ConfigAction,
    /// The native implementation lacks the required contract.
    EngineGap,
    /// Cost selection favors retaining DataFusion execution.
    CpuPreferred,
}

impl QueryGpuRemedyKind {
    /// Returns the stable snake-case label used in coverage JSON and report rows.
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
    /// Stable identifier of the prescribed action.
    pub code: &'static str,
    /// Category identifying who or what can address the candidate reason.
    pub kind: QueryGpuRemedyKind,
    /// Human-readable action guidance included in the coverage projection.
    pub guidance: &'static str,
}

impl QueryGpuRemedy {
    /// Constructs guidance with its stable code and action category.
    ///
    /// The supplied strings become report content without validation or rewriting.
    pub const fn new(code: &'static str, kind: QueryGpuRemedyKind, guidance: &'static str) -> Self {
        Self { code, kind, guidance }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Static validation and input-residency evidence for an explicit cuGraph call.
pub struct QueryGpuCugraphCall {
    /// Stable cuGraph algorithm code discovered in the plan.
    pub algorithm: String,
    /// Whether input schemas and options passed static validation.
    pub statically_valid: bool,
    /// Whether every input relation reaches the graph as a native GPU-resident
    /// producer. `false` means at least one input runs on DataFusion and is
    /// staged through host Arrow.
    pub inputs_native: bool,
    /// Planning observations that qualify this call's coverage.
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
/// Runtime condition that plan-time coverage cannot establish.
pub struct QueryGpuRuntimeCaveat {
    /// Stable identifier for the unverified runtime condition.
    pub code: String,
    /// Human-readable explanation of the condition.
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
/// Session, optimizer, and catalog context used to derive one coverage verdict.
///
/// Fingerprints identify this context for diagnostics; they do not reserve
/// resources or guarantee a later planning pass produces the same plan.
pub struct QueryGpuValidatedUnder {
    /// Whether the session has an installed Algeon native optimizer.
    pub native_optimizer_rule_installed: bool,
    /// Label identifying whether optimizer settings came from the session or validation options.
    pub optimizer_config_source: String,
    /// Configured execution-mode label, or `None` without an installed native optimizer.
    pub execution_mode: Option<String>,
    /// Requested final-plan coverage restrictions used for validation.
    pub final_plan_requirements: Vec<String>,
    /// Fingerprint of the relevant DataFusion session configuration.
    pub session_config_fingerprint: String,
    /// Fingerprint of the relevant native optimizer configuration.
    pub optimizer_fingerprint: String,
    /// Diagnostic description of the catalog snapshot used during planning.
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

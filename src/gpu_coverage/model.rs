//! Public GPU coverage DTO and JSON surface.

use datafusion::physical_plan::ExecutionPlan;
use serde_json::{Value, json};

use super::preflight::QueryGpuRejection;
use crate::{
    final_plan_observation::FinalPlanObservation,
    physical_plan::physical_plan_shape,
    planner::FinalPlanDisposition,
    report::{NativeCandidateCostEvidence, PlanningReport},
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
    pub physical_plan_root: Option<String>,
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
        report: Option<&PlanningReport>,
        planning_did_io: bool,
    ) -> Self {
        let overall = QueryGpuCoverageOverall::Rejected {
            phase: rejection.phase,
            reason_code: rejection.reason_code,
            retryable: rejection.retryable,
            detail: rejection.detail,
        };
        Self {
            overall,
            final_disposition: FinalPlanDisposition::Rejected,
            domains: QueryGpuCoverageDomains::default(),
            candidate_outcomes: report.map(super::physical::candidate_outcomes_from_report).unwrap_or_default(),
            cugraph_calls: Vec::new(),
            runtime_caveats: Vec::new(),
            planning_did_io,
            validated_under,
            confidence: QueryGpuCoverageConfidence::PlanTimeOnly,
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
            "physical_plan_root": self.physical_plan_root,
            "physical_plan_shape": self.physical_plan_shape,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryGpuCoverageOverall {
    NoCpuInPlannedPath,
    PartialGpuPlanned { gpu_fragments: usize, cpu_boundaries: usize },
    CpuOnly,
    Rejected { phase: QueryGpuRejectionPhase, reason_code: String, retryable: bool, detail: String },
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
            Self::Rejected { phase, reason_code, retryable, detail } => json!({
                "verdict": self.as_str(),
                "phase": phase.as_str(),
                "reason_code": reason_code,
                "retryable": retryable,
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
    NotSupported,
    NotSelectedByCost,
}

impl QueryGpuCandidateSelectionOutcome {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Selected => "selected",
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
    /// Present only when a capability or cost decision has a stable reason.
    pub reason_code: Option<String>,
    /// Present only when a capability or cost decision has a reason category.
    pub reason_kind: Option<QueryGpuCandidateReasonKind>,
    pub detail: String,
    /// Triage category for an unselected candidate, when available.
    pub category: Option<String>,
    pub cost_evidence: Option<NativeCandidateCostEvidence>,
}

impl QueryGpuCandidateOutcome {
    fn to_json(&self) -> Value {
        json!({
            "outcome": self.selection.as_str(),
            "reason_code": self.reason_code,
            "reason_kind": self.reason_kind.map(QueryGpuCandidateReasonKind::as_str),
            "detail": self.detail,
            "category": self.category,
            "cost_evidence": self.cost_evidence.as_ref().map(|evidence| evidence.to_json()),
        })
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

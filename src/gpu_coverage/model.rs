//! Public GPU coverage DTO and JSON surface.

use datafusion::physical_plan::ExecutionPlan;
use serde_json::{Value, json};

use super::{
    physical::{
        QueryGpuCoverageWalk, fallback_reasons_from_report, root_is_gpu_runtime_node, runtime_caveats,
        runtime_caveats_for_overall,
    },
    preflight::QueryGpuRejection,
};
use crate::{
    native_report::NativeDataFusionPostNativeExecutionPath, physical_plan::physical_plan_shape, report::PlanningReport,
};

/// Structured answer for "will this query's planned path avoid DataFusion CPU execution?"
///
/// `fallback_reasons` are per-candidate planning diagnostics retained from the
/// native optimizer. They can be non-empty alongside
/// [`QueryGpuCoverageOverall::NoCpuInPlannedPath`] when rejected candidates were
/// not on the final planned path.
///
/// `validated_under.catalog_snapshot` is a constant label describing when the
/// session state was cloned for validation. It is not a catalog fingerprint, so
/// verdicts are not comparable across catalog, view, or table-state changes by
/// that value alone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryGpuCoverage {
    pub overall: QueryGpuCoverageOverall,
    pub domains: QueryGpuCoverageDomains,
    pub fallback_reasons: Vec<QueryGpuFallbackReason>,
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
        cudf_native_path: NativeDataFusionPostNativeExecutionPath,
        walk: QueryGpuCoverageWalk,
        fallback_reasons: Vec<QueryGpuFallbackReason>,
        planning_did_io: bool,
    ) -> Self {
        let overall = if walk.has_gpu {
            if walk.all_gpu_covered && root_is_gpu_runtime_node(plan) {
                QueryGpuCoverageOverall::NoCpuInPlannedPath
            } else {
                QueryGpuCoverageOverall::PartialGpuPlanned {
                    gpu_fragments: walk.gpu_fragment_count,
                    cpu_boundaries: walk.cpu_boundary_count,
                }
            }
        } else {
            QueryGpuCoverageOverall::CpuOnly
        };
        let runtime_caveats = runtime_caveats(&walk, &overall, &validated_under);
        let domains = QueryGpuCoverageDomains {
            cudf_native_coverage: QueryGpuCudfNativeCoverage::from_post_native_path(cudf_native_path),
            cugraph_operator: if walk.cugraph_present {
                QueryGpuCugraphOperatorDomain::Present
            } else {
                QueryGpuCugraphOperatorDomain::Absent
            },
            datafusion_cpu_ingest_or_boundary: QueryGpuPresence::from_bool(walk.cpu_boundary_count > 0),
            host_output_materialization: QueryGpuPresence::from_bool(walk.host_output_materialization),
        };
        Self {
            overall,
            domains,
            fallback_reasons,
            cugraph_calls: walk.cugraph_calls,
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
        let runtime_caveats = runtime_caveats_for_overall(&overall, &validated_under, Vec::new());
        Self {
            overall,
            domains: QueryGpuCoverageDomains {
                cudf_native_coverage: QueryGpuCudfNativeCoverage::None,
                cugraph_operator: QueryGpuCugraphOperatorDomain::Absent,
                datafusion_cpu_ingest_or_boundary: QueryGpuPresence::Absent,
                host_output_materialization: QueryGpuPresence::Absent,
            },
            fallback_reasons: report.map(fallback_reasons_from_report).unwrap_or_default(),
            cugraph_calls: Vec::new(),
            runtime_caveats,
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
            "domains": self.domains.to_json(),
            "fallback_reasons": self.fallback_reasons.iter().map(QueryGpuFallbackReason::to_json).collect::<Vec<_>>(),
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryGpuCoverageDomains {
    pub cudf_native_coverage: QueryGpuCudfNativeCoverage,
    pub cugraph_operator: QueryGpuCugraphOperatorDomain,
    pub datafusion_cpu_ingest_or_boundary: QueryGpuPresence,
    pub host_output_materialization: QueryGpuPresence,
}

impl QueryGpuCoverageDomains {
    fn to_json(&self) -> Value {
        json!({
            "cudf_native_coverage": self.cudf_native_coverage.as_str(),
            "cugraph_operator": self.cugraph_operator.as_str(),
            "datafusion_cpu_ingest_or_boundary": self.datafusion_cpu_ingest_or_boundary.as_str(),
            "host_output_materialization": self.host_output_materialization.as_str(),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryGpuCudfNativeCoverage {
    Full,
    Partial,
    None,
}

impl QueryGpuCudfNativeCoverage {
    fn from_post_native_path(path: NativeDataFusionPostNativeExecutionPath) -> Self {
        match path {
            NativeDataFusionPostNativeExecutionPath::Native => Self::Full,
            NativeDataFusionPostNativeExecutionPath::PartialNative => Self::Partial,
            NativeDataFusionPostNativeExecutionPath::Cpu => Self::None,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Partial => "partial",
            Self::None => "none",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryGpuCugraphOperatorDomain {
    Present,
    Absent,
}

impl QueryGpuCugraphOperatorDomain {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Present => "present",
            Self::Absent => "absent",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryGpuPresence {
    Present,
    Absent,
}

impl QueryGpuPresence {
    fn from_bool(present: bool) -> Self {
        if present { Self::Present } else { Self::Absent }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Present => "present",
            Self::Absent => "absent",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryGpuFallbackReason {
    pub reason_code: String,
    pub kind: QueryGpuFallbackReasonKind,
    pub detail: String,
    pub category: String,
}

impl QueryGpuFallbackReason {
    fn to_json(&self) -> Value {
        json!({
            "reason_code": self.reason_code,
            "kind": self.kind.as_str(),
            "detail": self.detail,
            "category": self.category,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryGpuFallbackReasonKind {
    Operator,
    Expression,
    ArrowType,
    Source,
    PlanShape,
    Policy,
    Preference,
}

impl QueryGpuFallbackReasonKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Operator => "operator",
            Self::Expression => "expression",
            Self::ArrowType => "arrow_type",
            Self::Source => "source",
            Self::PlanShape => "plan_shape",
            Self::Policy => "policy",
            Self::Preference => "preference",
        }
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
    pub actual_policy: String,
    pub diagnostic_policy: String,
    pub optimizer_config_source: String,
    pub execution_mode: String,
    pub session_config_fingerprint: String,
    pub optimizer_fingerprint: String,
    pub catalog_snapshot: String,
}

impl QueryGpuValidatedUnder {
    fn to_json(&self) -> Value {
        json!({
            "actual_policy": self.actual_policy,
            "diagnostic_policy": self.diagnostic_policy,
            "optimizer_config_source": self.optimizer_config_source,
            "execution_mode": self.execution_mode,
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

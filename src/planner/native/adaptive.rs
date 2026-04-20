use datafusion::common::{DataFusionError, Result as DataFusionResult};
use nexus_query_engine::{
    adaptive_report::{AdaptiveCandidateStatus, AdaptivePipelineReport, AdaptivePipelineReportRow},
    plan::QueryPlan,
};

use crate::{error::Error, native_exec::NativeCudfExecutionConfig, report::GpuCandidate};

/// Adapter-side adaptive bounded diagnostics collected during planning.
///
/// This surface is separate from strict bounded contract diagnostics. It
/// carries adaptive rows only; strict gate rows stay in
/// `NativeStrictBoundedContractDiagnostics`.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct NativeAdaptivePipelineDiagnostics {
    /// Per-candidate adaptive admission reports captured during planning.
    pub candidates: Vec<NativeAdaptivePipelineCandidateDiagnostics>,
}

impl NativeAdaptivePipelineDiagnostics {
    /// Appends candidate reports from another optimizer stage.
    pub fn merge(&mut self, other: Self) {
        self.candidates.extend(other.candidates);
    }

    /// Returns true when no adaptive candidate reports were captured.
    pub fn is_empty(&self) -> bool {
        self.candidates.is_empty()
    }

    /// Returns adaptive report rows matching one planning candidate.
    pub fn rows_for_candidate(&self, candidate: &GpuCandidate) -> Vec<AdaptivePipelineReportRow> {
        self.candidates
            .iter()
            .filter(|entry| entry.candidate == *candidate)
            .map(|entry| entry.report.row().clone())
            .collect()
    }

    /// Builds diagnostics for a single adaptive candidate admission report.
    pub fn single(candidate: GpuCandidate, report: AdaptivePipelineReport) -> Self {
        Self { candidates: vec![NativeAdaptivePipelineCandidateDiagnostics { candidate, report }] }
    }
}

/// Adaptive admission report for one optimizer candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeAdaptivePipelineCandidateDiagnostics {
    /// Optimizer candidate associated with this adaptive report.
    pub candidate: GpuCandidate,
    /// Native-engine adaptive admission report.
    pub report: AdaptivePipelineReport,
}

pub(super) fn plan_adaptive_pipeline_diagnostics(
    plan: &QueryPlan,
    candidate: GpuCandidate,
    execution_config: &NativeCudfExecutionConfig,
) -> DataFusionResult<Option<NativeAdaptivePipelineDiagnostics>> {
    let memory_history = execution_config.adaptive_memory_history();
    let admission = nexus_query_engine::exec::adaptive_bounded_pipeline_admission_with_memory_history(
        plan,
        memory_history.as_deref(),
    )
    .map_err(|error| adaptive_pipeline_report_error(stamp_native_error_correlation(error, execution_config)))?;
    Ok(admission.map(|admission| {
        NativeAdaptivePipelineDiagnostics::single(
            candidate,
            stamp_adaptive_report_correlation(admission.report().clone(), execution_config),
        )
    }))
}

pub(super) fn adaptive_pipeline_was_rejected(diagnostics: &NativeAdaptivePipelineDiagnostics) -> bool {
    diagnostics
        .candidates
        .iter()
        .any(|candidate| candidate.report.row().candidate_status == AdaptiveCandidateStatus::Rejected)
}

fn adaptive_pipeline_report_error(error: nexus_query_engine::error::QueryError) -> DataFusionError {
    Error::from_native(
        "datafusion_nexus.native.adaptive_pipeline_diagnostics",
        "adaptive pipeline diagnostics could not be built",
        error,
    )
    .into()
}

fn stamp_adaptive_report_correlation(
    mut report: AdaptivePipelineReport,
    execution_config: &NativeCudfExecutionConfig,
) -> AdaptivePipelineReport {
    if let Some(query_id) = execution_config.query_id() {
        report = report.with_query_id(query_id.clone());
    }
    if let Some(trace_id) = execution_config.trace_id() {
        report = report.with_trace_id(trace_id.clone());
    }
    report
}

fn stamp_native_error_correlation(
    mut error: nexus_query_engine::error::QueryError,
    execution_config: &NativeCudfExecutionConfig,
) -> nexus_query_engine::error::QueryError {
    if let Some(query_id) = execution_config.query_id() {
        error = error.with_query_id(query_id.clone());
    }
    if let Some(trace_id) = execution_config.trace_id() {
        error = error.with_trace_id(trace_id.clone());
    }
    error
}

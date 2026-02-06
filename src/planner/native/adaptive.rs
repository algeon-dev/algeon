use cudf_query_engine::{AdaptivePipelineReport, AdaptivePipelineReportRow};

use crate::report::GpuCandidate;

/// Adapter-side adaptive bounded diagnostics collected during planning.
///
/// This surface is separate from strict bounded contract diagnostics. It
/// carries adaptive rows only; strict gate rows stay in
/// `NativeStrictBoundedContractDiagnostics`.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct NativeAdaptivePipelineDiagnostics {
    pub candidates: Vec<NativeAdaptivePipelineCandidateDiagnostics>,
}

impl NativeAdaptivePipelineDiagnostics {
    pub fn merge(&mut self, other: Self) {
        self.candidates.extend(other.candidates);
    }

    pub fn is_empty(&self) -> bool {
        self.candidates.is_empty()
    }

    pub fn rows_for_candidate(&self, candidate: &GpuCandidate) -> Vec<AdaptivePipelineReportRow> {
        self.candidates
            .iter()
            .filter(|entry| entry.candidate == *candidate)
            .map(|entry| entry.report.row().clone())
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeAdaptivePipelineCandidateDiagnostics {
    pub candidate: GpuCandidate,
    pub report: AdaptivePipelineReport,
}

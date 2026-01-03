use std::fmt::Write as _;

use super::{
    config::LargerIslandBenchmarkConfig,
    metrics::LargerIslandBenchmarkMetrics,
    timing::{TimedBenchmarkRun, duration_ms, speedup},
};
use crate::{CandidateOutcome, FallbackReason, PlanningReport};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LargerIslandBenchmarkPlanning {
    pub candidate_count: usize,
    pub rewritten_count: usize,
    pub rejected_count: usize,
    pub rewritten_shapes: Vec<String>,
    pub rejected_reasons: Vec<FallbackReason>,
}

impl LargerIslandBenchmarkPlanning {
    pub fn from_report(report: &PlanningReport) -> Self {
        let mut rewritten_shapes = Vec::new();
        let mut rejected_reasons = Vec::new();

        for outcome in report.outcomes() {
            match outcome {
                CandidateOutcome::Rewritten { candidate } => rewritten_shapes.push(candidate.shape.clone()),
                CandidateOutcome::Rejected { reason, .. } => rejected_reasons.push(*reason),
            }
        }

        Self {
            candidate_count: report.outcomes().len(),
            rewritten_count: report.rewritten_count(),
            rejected_count: report.rejected_count(),
            rewritten_shapes,
            rejected_reasons,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LargerIslandBenchmarkDecisionCriteria {
    minimum_gpu_speedup: f64,
    max_rejected_candidates: usize,
    max_materializations_per_rewritten_island: usize,
    require_peak_device_memory: bool,
}

impl Default for LargerIslandBenchmarkDecisionCriteria {
    fn default() -> Self {
        Self {
            minimum_gpu_speedup: 1.0,
            max_rejected_candidates: 0,
            max_materializations_per_rewritten_island: 1,
            require_peak_device_memory: false,
        }
    }
}

impl LargerIslandBenchmarkDecisionCriteria {
    pub fn minimum_gpu_speedup(&self) -> f64 {
        self.minimum_gpu_speedup
    }

    pub fn max_rejected_candidates(&self) -> usize {
        self.max_rejected_candidates
    }

    pub fn max_materializations_per_rewritten_island(&self) -> usize {
        self.max_materializations_per_rewritten_island
    }

    pub fn require_peak_device_memory(&self) -> bool {
        self.require_peak_device_memory
    }

    pub fn with_minimum_gpu_speedup(mut self, minimum_gpu_speedup: f64) -> Self {
        self.minimum_gpu_speedup = minimum_gpu_speedup;
        self
    }

    pub fn with_max_rejected_candidates(mut self, max_rejected_candidates: usize) -> Self {
        self.max_rejected_candidates = max_rejected_candidates;
        self
    }

    pub fn with_max_materializations_per_rewritten_island(
        mut self,
        max_materializations_per_rewritten_island: usize,
    ) -> Self {
        self.max_materializations_per_rewritten_island = max_materializations_per_rewritten_island;
        self
    }

    pub fn with_peak_device_memory_required(mut self, require_peak_device_memory: bool) -> Self {
        self.require_peak_device_memory = require_peak_device_memory;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchmarkGateStatus {
    ProceedToPhase4,
    ImproveCoverageBeforeNativeEngine,
    KeepExtensionAdapter,
    DoNotExpandNativeEngine,
    NeedMoreMeasurements,
}

impl BenchmarkGateStatus {
    pub const fn code(self) -> &'static str {
        match self {
            Self::ProceedToPhase4 => "proceed_to_phase_4",
            Self::ImproveCoverageBeforeNativeEngine => "improve_coverage_before_native_engine",
            Self::KeepExtensionAdapter => "keep_extension_adapter",
            Self::DoNotExpandNativeEngine => "do_not_expand_native_engine",
            Self::NeedMoreMeasurements => "need_more_measurements",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchmarkGateReason {
    MissingBenchmarkSamples,
    NoGpuIslandRewritten,
    GpuNotFasterThanCpu,
    RejectedFallbackCandidates,
    TooManyMaterializationBoundaries,
    PeakDeviceMemoryUnavailable,
}

impl BenchmarkGateReason {
    pub const fn code(self) -> &'static str {
        match self {
            Self::MissingBenchmarkSamples => "missing_benchmark_samples",
            Self::NoGpuIslandRewritten => "no_gpu_island_rewritten",
            Self::GpuNotFasterThanCpu => "gpu_not_faster_than_cpu",
            Self::RejectedFallbackCandidates => "rejected_fallback_candidates",
            Self::TooManyMaterializationBoundaries => "too_many_materialization_boundaries",
            Self::PeakDeviceMemoryUnavailable => "peak_device_memory_unavailable",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LargerIslandBenchmarkAssessment {
    pub status: BenchmarkGateStatus,
    pub reasons: Vec<BenchmarkGateReason>,
    pub gpu_speedup: f64,
    pub host_device_boundary_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LargerIslandBenchmarkReport {
    pub config: LargerIslandBenchmarkConfig,
    pub cpu_baseline: TimedBenchmarkRun,
    pub gpu_island: TimedBenchmarkRun,
    pub gpu_metrics: LargerIslandBenchmarkMetrics,
    pub planning: LargerIslandBenchmarkPlanning,
    pub output_batches: usize,
    pub output_rows: usize,
    pub peak_device_memory_bytes: Option<usize>,
}

impl LargerIslandBenchmarkReport {
    pub fn assess_phase4_readiness(
        &self,
        criteria: &LargerIslandBenchmarkDecisionCriteria,
    ) -> LargerIslandBenchmarkAssessment {
        let mut reasons = Vec::new();

        if self.cpu_baseline.iterations == 0 || self.gpu_island.iterations == 0 {
            reasons.push(BenchmarkGateReason::MissingBenchmarkSamples);
        }
        if self.planning.rewritten_count == 0 {
            reasons.push(BenchmarkGateReason::NoGpuIslandRewritten);
        }
        if self.planning.rejected_count > criteria.max_rejected_candidates {
            reasons.push(BenchmarkGateReason::RejectedFallbackCandidates);
        }

        let materialization_limit =
            criteria.max_materializations_per_rewritten_island.saturating_mul(self.planning.rewritten_count.max(1));
        if self.gpu_metrics.device_to_host_materializations > materialization_limit {
            reasons.push(BenchmarkGateReason::TooManyMaterializationBoundaries);
        }

        if criteria.require_peak_device_memory && self.peak_device_memory_bytes.is_none() {
            reasons.push(BenchmarkGateReason::PeakDeviceMemoryUnavailable);
        }

        let gpu_speedup = speedup(self.cpu_baseline.mean, self.gpu_island.mean);
        if gpu_speedup < criteria.minimum_gpu_speedup {
            reasons.push(BenchmarkGateReason::GpuNotFasterThanCpu);
        }

        let status = benchmark_gate_status(&reasons);
        LargerIslandBenchmarkAssessment {
            status,
            reasons,
            gpu_speedup,
            host_device_boundary_count: self
                .gpu_metrics
                .host_to_device_uploads
                .saturating_add(self.gpu_metrics.device_to_host_materializations),
        }
    }

    pub fn to_text(&self) -> String {
        let mut output = String::new();
        let assessment = self.assess_phase4_readiness(&LargerIslandBenchmarkDecisionCriteria::default());
        let _ = writeln!(output, "larger_island_benchmark");
        let _ = writeln!(output, "input_rows={}", self.config.input_rows);
        let _ = writeln!(output, "batch_size={}", self.config.batch_size);
        let _ = writeln!(output, "group_count={}", self.config.group_count);
        let _ = writeln!(output, "limit={}", self.config.limit);
        let _ = writeln!(output, "iterations={}", self.config.iterations);
        let _ = writeln!(output, "phase_c_memory_control_policy={}", self.gpu_metrics.phase_c_memory_control_policy);
        let _ = writeln!(output, "phase_c_runtime_gate={}", self.gpu_metrics.phase_c_runtime_gate);
        let _ = writeln!(output, "input_batches={}", self.gpu_metrics.input_batches);
        let _ = writeln!(output, "cpu_baseline_wall_ms={:.3}", duration_ms(self.cpu_baseline.mean));
        let _ = writeln!(output, "gpu_island_wall_ms={:.3}", duration_ms(self.gpu_island.mean));
        let _ =
            writeln!(output, "record_batch_upload_ms={:.3}", duration_ms(self.gpu_metrics.record_batch_upload_time));
        let _ = writeln!(output, "parquet_read_ms={:.3}", duration_ms(self.gpu_metrics.parquet_read_time));
        let _ = writeln!(output, "cudf_operator_ms={:.3}", duration_ms(self.gpu_metrics.cudf_operator_time));
        let _ = writeln!(
            output,
            "record_batch_materialization_ms={:.3}",
            duration_ms(self.gpu_metrics.record_batch_materialization_time)
        );
        let _ = writeln!(output, "host_to_device_uploads={}", self.gpu_metrics.host_to_device_uploads);
        let _ = writeln!(output, "host_to_device_uploaded_rows={}", self.gpu_metrics.host_to_device_uploaded_rows);
        let _ = writeln!(output, "host_to_device_uploaded_bytes={}", self.gpu_metrics.host_to_device_uploaded_bytes);
        let _ = writeln!(output, "parquet_files={}", self.gpu_metrics.parquet_files);
        let _ = writeln!(output, "parquet_bytes_known={}", self.gpu_metrics.parquet_bytes_known);
        let _ = writeln!(output, "row_groups_selected={}", self.gpu_metrics.row_groups_selected);
        let _ = writeln!(output, "read_columns={}", self.gpu_metrics.read_columns);
        let _ = writeln!(output, "reservation_requested_bytes={}", self.gpu_metrics.reservation_requested_bytes);
        let _ = writeln!(output, "reservation_granted_bytes={}", self.gpu_metrics.reservation_granted_bytes);
        let _ = writeln!(output, "reservation_denied_count={}", self.gpu_metrics.reservation_denied_count);
        let _ = writeln!(output, "source_chunks_planned={}", self.gpu_metrics.source_chunks_planned);
        let _ = writeln!(output, "source_chunks_attempted={}", self.gpu_metrics.source_chunks_attempted);
        let _ = writeln!(output, "source_chunks_committed={}", self.gpu_metrics.source_chunks_committed);
        let _ = writeln!(output, "source_chunks_retried={}", self.gpu_metrics.source_chunks_retried);
        let _ = writeln!(output, "row_groups_planned={}", self.gpu_metrics.row_groups_planned);
        let _ = writeln!(output, "row_groups_committed={}", self.gpu_metrics.row_groups_committed);
        let _ = writeln!(output, "partial_aggregate_count={}", self.gpu_metrics.partial_aggregate_count);
        let _ = writeln!(output, "partial_aggregate_rows={}", self.gpu_metrics.partial_aggregate_rows);
        let _ = writeln!(output, "retry_count={}", self.gpu_metrics.retry_count);
        let _ = writeln!(output, "retry_resource_exhausted_count={}", self.gpu_metrics.retry_resource_exhausted_count);
        let _ =
            writeln!(output, "retry_reservation_failure_count={}", self.gpu_metrics.retry_reservation_failure_count);
        let _ = writeln!(
            output,
            "resident_dataframe_peak_bytes={}",
            optional_usize_text(self.gpu_metrics.resident_dataframe_peak_bytes)
        );
        let _ = writeln!(
            output,
            "allocator_high_water_bytes={}",
            optional_usize_text(self.gpu_metrics.allocator_high_water_bytes)
        );
        let _ = writeln!(output, "final_merge_attempts={}", self.gpu_metrics.final_merge_attempts);
        let _ = writeln!(output, "final_merge_levels={}", self.gpu_metrics.final_merge_levels);
        let _ =
            writeln!(output, "device_to_host_materializations={}", self.gpu_metrics.device_to_host_materializations);
        let _ =
            writeln!(output, "device_to_host_materialized_rows={}", self.gpu_metrics.device_to_host_materialized_rows);
        let _ = writeln!(
            output,
            "device_to_host_materialized_bytes={}",
            self.gpu_metrics.device_to_host_materialized_bytes
        );
        let _ = writeln!(output, "fallback_candidate_count={}", self.planning.candidate_count);
        let _ = writeln!(output, "fallback_rewritten_count={}", self.planning.rewritten_count);
        let _ = writeln!(output, "fallback_rejected_count={}", self.planning.rejected_count);
        let rejected_reasons =
            self.planning.rejected_reasons.iter().map(|reason| reason.code()).collect::<Vec<_>>().join(",");
        let _ = writeln!(output, "fallback_rejected_reasons={rejected_reasons}");
        let gate_reasons = assessment.reasons.iter().map(|reason| reason.code()).collect::<Vec<_>>().join(",");
        let _ = writeln!(output, "benchmark_gate_status={}", assessment.status.code());
        let _ = writeln!(output, "benchmark_gate_reasons={gate_reasons}");
        let _ = writeln!(output, "gpu_speedup={:.3}", assessment.gpu_speedup);
        let _ = writeln!(output, "host_device_boundary_count={}", assessment.host_device_boundary_count);
        let _ = writeln!(output, "output_batches={}", self.output_batches);
        let _ = writeln!(output, "output_rows={}", self.output_rows);
        match self.peak_device_memory_bytes {
            Some(bytes) => {
                let _ = writeln!(output, "peak_device_memory_bytes={bytes}");
            },
            None => {
                let _ = writeln!(output, "peak_device_memory_bytes=unavailable");
            },
        }
        output
    }
}

fn optional_usize_text(value: Option<usize>) -> String {
    value.map_or_else(|| "unavailable".to_string(), |value| value.to_string())
}

fn benchmark_gate_status(reasons: &[BenchmarkGateReason]) -> BenchmarkGateStatus {
    if reasons.iter().any(|reason| {
        matches!(
            reason,
            BenchmarkGateReason::MissingBenchmarkSamples
                | BenchmarkGateReason::NoGpuIslandRewritten
                | BenchmarkGateReason::PeakDeviceMemoryUnavailable
        )
    }) {
        return BenchmarkGateStatus::NeedMoreMeasurements;
    }

    if reasons.contains(&BenchmarkGateReason::GpuNotFasterThanCpu) {
        return BenchmarkGateStatus::DoNotExpandNativeEngine;
    }

    if reasons.contains(&BenchmarkGateReason::RejectedFallbackCandidates) {
        return BenchmarkGateStatus::ImproveCoverageBeforeNativeEngine;
    }

    if reasons.contains(&BenchmarkGateReason::TooManyMaterializationBoundaries) {
        return BenchmarkGateStatus::KeepExtensionAdapter;
    }

    BenchmarkGateStatus::ProceedToPhase4
}

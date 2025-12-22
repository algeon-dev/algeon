use std::{
    fmt::Write as _,
    sync::Arc,
    time::{Duration, Instant},
};

use arrow::{
    array::{Int32Array, Int64Array},
    compute::SortOptions as ArrowSortOptions,
    datatypes::{DataType, Field, Schema, SchemaRef},
    record_batch::RecordBatch,
};
use datafusion::{
    common::{DataFusionError, Result as DataFusionResult, ScalarValue},
    datasource::memory::MemorySourceConfig,
    execution::context::TaskContext,
    functions_aggregate::sum::sum_udaf,
    logical_expr::Operator,
    physical_expr::{
        LexOrdering, PhysicalSortExpr,
        aggregate::AggregateExprBuilder,
        expressions::{binary, col, lit},
    },
    physical_plan::{
        ExecutionPlan,
        aggregates::{AggregateExec, AggregateMode, PhysicalGroupBy},
        collect,
        filter::FilterExec,
        limit::LocalLimitExec,
        metrics::MetricsSet,
        projection::{ProjectionExec, ProjectionExpr},
        sorts::sort::SortExec,
    },
};

use crate::{CandidateOutcome, CudfOptimizerConfig, FallbackReason, GpuFallbackPolicy, PlanningReport, optimize_plan};

const PHASE_C_MEMORY_CONTROL_UNBOUNDED: &str = "unbounded";
const PHASE_C_RUNTIME_GATE_NOT_EVALUATED: &str = "not_evaluated";
const PHASE_C_MEMORY_CONTROL_BOUNDED: &str = "bounded";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LargerIslandBenchmarkConfig {
    input_rows: usize,
    batch_size: usize,
    group_count: usize,
    limit: usize,
    warmup_iterations: usize,
    iterations: usize,
}

impl Default for LargerIslandBenchmarkConfig {
    fn default() -> Self {
        Self {
            input_rows: 1_000_000,
            batch_size: 65_536,
            group_count: 4_096,
            limit: 100,
            warmup_iterations: 1,
            iterations: 3,
        }
    }
}

impl LargerIslandBenchmarkConfig {
    pub fn input_rows(&self) -> usize {
        self.input_rows
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn group_count(&self) -> usize {
        self.group_count
    }

    pub fn limit(&self) -> usize {
        self.limit
    }

    pub fn warmup_iterations(&self) -> usize {
        self.warmup_iterations
    }

    pub fn iterations(&self) -> usize {
        self.iterations
    }

    pub fn with_input_rows(mut self, input_rows: usize) -> Self {
        self.input_rows = input_rows;
        self
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    pub fn with_group_count(mut self, group_count: usize) -> Self {
        self.group_count = group_count;
        self
    }

    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    pub fn with_warmup_iterations(mut self, warmup_iterations: usize) -> Self {
        self.warmup_iterations = warmup_iterations;
        self
    }

    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    pub fn validate(&self) -> DataFusionResult<()> {
        validate_nonzero("input_rows", self.input_rows)?;
        validate_nonzero("batch_size", self.batch_size)?;
        validate_nonzero("group_count", self.group_count)?;
        validate_nonzero("limit", self.limit)?;
        validate_nonzero("iterations", self.iterations)?;
        if self.group_count > i32::MAX as usize {
            return Err(DataFusionError::Plan(format!(
                "group_count must fit in Int32 benchmark keys, got {}",
                self.group_count
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimedBenchmarkRun {
    pub iterations: usize,
    pub total: Duration,
    pub min: Duration,
    pub max: Duration,
    pub mean: Duration,
}

impl TimedBenchmarkRun {
    pub fn from_samples(samples: impl IntoIterator<Item = Duration>) -> Self {
        let samples = samples.into_iter().collect::<Vec<_>>();
        if samples.is_empty() {
            return Self {
                iterations: 0,
                total: Duration::ZERO,
                min: Duration::ZERO,
                max: Duration::ZERO,
                mean: Duration::ZERO,
            };
        }

        let total = samples.iter().copied().sum();
        let min = samples.iter().copied().min().unwrap_or(Duration::ZERO);
        let max = samples.iter().copied().max().unwrap_or(Duration::ZERO);
        let mean = duration_div(total, samples.len());
        Self { iterations: samples.len(), total, min, max, mean }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LargerIslandBenchmarkMetrics {
    pub phase_c_memory_control_policy: &'static str,
    pub phase_c_runtime_gate: &'static str,
    pub input_batches: usize,
    pub input_rows: usize,
    pub resident_dataframe_peak_bytes: Option<usize>,
    pub allocator_high_water_bytes: Option<usize>,
    pub reservation_requested_bytes: usize,
    pub reservation_granted_bytes: usize,
    pub reservation_denied_count: usize,
    pub source_chunks_planned: usize,
    pub source_chunks_attempted: usize,
    pub source_chunks_committed: usize,
    pub source_chunks_retried: usize,
    pub row_groups_planned: usize,
    pub row_groups_committed: usize,
    pub partial_aggregate_count: usize,
    pub partial_aggregate_rows: usize,
    pub final_merge_attempts: usize,
    pub final_merge_levels: usize,
    pub retry_count: usize,
    pub retry_resource_exhausted_count: usize,
    pub retry_reservation_failure_count: usize,
    pub peak_device_memory_bytes: Option<usize>,
    pub host_to_device_uploads: usize,
    pub host_to_device_uploaded_rows: usize,
    pub host_to_device_uploaded_bytes: usize,
    pub device_to_host_materializations: usize,
    pub device_to_host_materialized_rows: usize,
    pub device_to_host_materialized_bytes: usize,
    pub parquet_files: usize,
    pub parquet_bytes_known: usize,
    pub row_groups_selected: usize,
    pub read_columns: usize,
    pub parquet_read_time: Duration,
    pub record_batch_upload_time: Duration,
    pub cudf_operator_time: Duration,
    pub record_batch_materialization_time: Duration,
    pub elapsed_compute: Option<Duration>,
    pub output_rows: Option<usize>,
}

impl LargerIslandBenchmarkMetrics {
    pub fn from_metrics(metrics: &MetricsSet) -> Self {
        Self {
            phase_c_memory_control_policy: PHASE_C_MEMORY_CONTROL_UNBOUNDED,
            phase_c_runtime_gate: PHASE_C_RUNTIME_GATE_NOT_EVALUATED,
            input_batches: metric_count(metrics, "input_batches"),
            input_rows: metric_count(metrics, "input_rows"),
            resident_dataframe_peak_bytes: None,
            allocator_high_water_bytes: None,
            reservation_requested_bytes: 0,
            reservation_granted_bytes: 0,
            reservation_denied_count: 0,
            source_chunks_planned: 0,
            source_chunks_attempted: 0,
            source_chunks_committed: 0,
            source_chunks_retried: 0,
            row_groups_planned: 0,
            row_groups_committed: 0,
            partial_aggregate_count: 0,
            partial_aggregate_rows: 0,
            final_merge_attempts: 0,
            final_merge_levels: 0,
            retry_count: 0,
            retry_resource_exhausted_count: 0,
            retry_reservation_failure_count: 0,
            peak_device_memory_bytes: None,
            host_to_device_uploads: metric_count(metrics, "host_to_device_uploads"),
            host_to_device_uploaded_rows: metric_count(metrics, "host_to_device_uploaded_rows"),
            host_to_device_uploaded_bytes: metric_count(metrics, "host_to_device_uploaded_bytes"),
            device_to_host_materializations: metric_count(metrics, "device_to_host_materializations"),
            device_to_host_materialized_rows: metric_count(metrics, "device_to_host_materialized_rows"),
            device_to_host_materialized_bytes: metric_count(metrics, "device_to_host_materialized_bytes"),
            parquet_files: metric_count(metrics, "parquet_files"),
            parquet_bytes_known: metric_count(metrics, "parquet_bytes_known"),
            row_groups_selected: metric_count(metrics, "row_groups_selected"),
            read_columns: metric_count(metrics, "read_columns"),
            parquet_read_time: metric_duration(metrics, "parquet_read_time"),
            record_batch_upload_time: metric_duration(metrics, "record_batch_upload_time"),
            cudf_operator_time: metric_duration(metrics, "cudf_operator_time"),
            record_batch_materialization_time: metric_duration(metrics, "record_batch_materialization_time"),
            elapsed_compute: metrics.elapsed_compute().map(duration_from_nanos),
            output_rows: metrics.output_rows(),
        }
    }

    pub fn assess_phase_c_runtime_gate(&self, output_equality_verified: bool) -> PhaseCRuntimeGateStatus {
        if self.phase_c_memory_control_policy == PHASE_C_MEMORY_CONTROL_UNBOUNDED {
            return PhaseCRuntimeGateStatus::NotEvaluated;
        }
        if self.phase_c_memory_control_policy != PHASE_C_MEMORY_CONTROL_BOUNDED {
            return PhaseCRuntimeGateStatus::Fail;
        }
        if !output_equality_verified {
            return PhaseCRuntimeGateStatus::Fail;
        }
        if self.host_to_device_uploads != 0
            || self.host_to_device_uploaded_rows != 0
            || self.host_to_device_uploaded_bytes != 0
        {
            return PhaseCRuntimeGateStatus::Fail;
        }
        if self.device_to_host_materializations != 1 {
            return PhaseCRuntimeGateStatus::Fail;
        }
        if self.source_chunks_planned == 0
            || self.source_chunks_committed == 0
            || self.source_chunks_committed != self.source_chunks_planned
            || self.row_groups_planned == 0
            || self.row_groups_committed != self.row_groups_planned
        {
            return PhaseCRuntimeGateStatus::Fail;
        }
        if self.resident_dataframe_peak_bytes.is_none() {
            return PhaseCRuntimeGateStatus::Fail;
        }

        PhaseCRuntimeGateStatus::Pass
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhaseCRuntimeGateStatus {
    Pass,
    Fail,
    NotEvaluated,
}

impl PhaseCRuntimeGateStatus {
    pub const fn code(self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::Fail => "fail",
            Self::NotEvaluated => PHASE_C_RUNTIME_GATE_NOT_EVALUATED,
        }
    }
}

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

pub fn build_larger_island_benchmark_plan(
    config: &LargerIslandBenchmarkConfig,
) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
    config.validate()?;

    let schema = benchmark_schema();
    let partitions = benchmark_partitions(config, Arc::clone(&schema))?;
    let source: Arc<dyn ExecutionPlan> = MemorySourceConfig::try_new_exec(&partitions, Arc::clone(&schema), None)?;

    let predicate = binary(col("payload", &schema)?, Operator::GtEq, lit(ScalarValue::Int32(Some(10))), &schema)?;
    let filter: Arc<dyn ExecutionPlan> = Arc::new(FilterExec::try_new(predicate, source)?);

    let filter_schema = filter.schema();
    let amount =
        binary(col("value", &filter_schema)?, Operator::Plus, lit(ScalarValue::Int64(Some(1))), &filter_schema)?;
    let projection: Arc<dyn ExecutionPlan> = Arc::new(ProjectionExec::try_new(
        [
            ProjectionExpr { expr: col("key", &filter_schema)?, alias: "key".to_string() },
            ProjectionExpr { expr: amount, alias: "amount".to_string() },
        ],
        filter,
    )?);

    let projection_schema = projection.schema();
    let group_by = PhysicalGroupBy::new_single(vec![(col("key", &projection_schema)?, "key".to_string())]);
    let total = AggregateExprBuilder::new(sum_udaf(), vec![col("amount", &projection_schema)?])
        .schema(Arc::clone(&projection_schema))
        .alias("total")
        .build()?;
    let aggregate: Arc<dyn ExecutionPlan> = Arc::new(AggregateExec::try_new(
        AggregateMode::Single,
        group_by,
        vec![Arc::new(total)],
        vec![None],
        projection,
        projection_schema,
    )?);

    let aggregate_schema = aggregate.schema();
    let sort_exprs =
        LexOrdering::new([PhysicalSortExpr::new(col("total", &aggregate_schema)?, ArrowSortOptions::new(true, false))])
            .ok_or_else(|| DataFusionError::Plan("benchmark sort ordering cannot be empty".to_string()))?;
    let sort: Arc<dyn ExecutionPlan> = Arc::new(SortExec::new(sort_exprs, aggregate));

    Ok(Arc::new(LocalLimitExec::new(sort, config.limit)))
}

pub fn run_larger_island_benchmark(
    config: LargerIslandBenchmarkConfig,
) -> DataFusionResult<LargerIslandBenchmarkReport> {
    let optimizer = CudfOptimizerConfig::new(GpuFallbackPolicy::GpuOnly).with_min_operators_per_island(2);
    run_larger_island_benchmark_with_optimizer(config, &optimizer)
}

pub fn run_larger_island_benchmark_with_optimizer(
    config: LargerIslandBenchmarkConfig,
    optimizer: &CudfOptimizerConfig,
) -> DataFusionResult<LargerIslandBenchmarkReport> {
    config.validate()?;

    for _ in 0..config.warmup_iterations {
        let _ = collect_cpu_once(&config)?;
        let _ = collect_gpu_once(&config, optimizer)?;
    }

    let mut cpu_samples = Vec::with_capacity(config.iterations);
    let mut final_cpu = None;
    for _ in 0..config.iterations {
        let measurement = collect_cpu_once(&config)?;
        cpu_samples.push(measurement.wall_time);
        final_cpu = Some(measurement);
    }

    let mut gpu_samples = Vec::with_capacity(config.iterations);
    let mut final_gpu = None;
    for _ in 0..config.iterations {
        let measurement = collect_gpu_once(&config, optimizer)?;
        gpu_samples.push(measurement.wall_time);
        final_gpu = Some(measurement);
    }

    let final_gpu = final_gpu.expect("validated benchmark config requires at least one measured iteration");
    let final_cpu = final_cpu.expect("validated benchmark config requires at least one measured iteration");
    if final_cpu.output_rows != final_gpu.output_rows {
        return Err(DataFusionError::Plan(format!(
            "CPU baseline output rows ({}) did not match GPU island output rows ({})",
            final_cpu.output_rows, final_gpu.output_rows
        )));
    }
    if final_cpu.output_batches != final_gpu.output_batches {
        return Err(DataFusionError::Plan(format!(
            "CPU baseline output batches ({}) did not match GPU island output batches ({})",
            final_cpu.output_batches, final_gpu.output_batches
        )));
    }

    let peak_device_memory_bytes = final_gpu.metrics.peak_device_memory_bytes;
    Ok(LargerIslandBenchmarkReport {
        config,
        cpu_baseline: TimedBenchmarkRun::from_samples(cpu_samples),
        gpu_island: TimedBenchmarkRun::from_samples(gpu_samples),
        gpu_metrics: final_gpu.metrics,
        planning: final_gpu.planning,
        output_batches: final_gpu.output_batches,
        output_rows: final_gpu.output_rows,
        peak_device_memory_bytes,
    })
}

#[derive(Debug)]
struct CollectMeasurement {
    wall_time: Duration,
    output_batches: usize,
    output_rows: usize,
}

#[derive(Debug)]
struct GpuCollectMeasurement {
    wall_time: Duration,
    output_batches: usize,
    output_rows: usize,
    metrics: LargerIslandBenchmarkMetrics,
    planning: LargerIslandBenchmarkPlanning,
}

fn collect_cpu_once(config: &LargerIslandBenchmarkConfig) -> DataFusionResult<CollectMeasurement> {
    let plan = build_larger_island_benchmark_plan(config)?;
    let started = Instant::now();
    let output = futures::executor::block_on(collect(plan, Arc::new(TaskContext::default())))?;
    Ok(CollectMeasurement {
        wall_time: started.elapsed(),
        output_batches: output.len(),
        output_rows: output.iter().map(RecordBatch::num_rows).sum(),
    })
}

fn collect_gpu_once(
    config: &LargerIslandBenchmarkConfig,
    optimizer: &CudfOptimizerConfig,
) -> DataFusionResult<GpuCollectMeasurement> {
    let plan = build_larger_island_benchmark_plan(config)?;
    let optimized = optimize_plan(plan, optimizer)?;
    let planning = LargerIslandBenchmarkPlanning::from_report(&optimized.report);
    if planning.rewritten_count == 0 {
        return Err(DataFusionError::Plan("larger-island benchmark optimizer produced no GPU island".to_string()));
    }
    let plan = optimized.plan;

    let started = Instant::now();
    let output = futures::executor::block_on(collect(Arc::clone(&plan), Arc::new(TaskContext::default())))?;
    let wall_time = started.elapsed();
    let metrics = plan
        .metrics()
        .map(|metrics| LargerIslandBenchmarkMetrics::from_metrics(&metrics))
        .unwrap_or_else(|| LargerIslandBenchmarkMetrics::from_metrics(&MetricsSet::new()));

    Ok(GpuCollectMeasurement {
        wall_time,
        output_batches: output.len(),
        output_rows: output.iter().map(RecordBatch::num_rows).sum(),
        metrics,
        planning,
    })
}

fn benchmark_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("value", DataType::Int64, false),
        Field::new("payload", DataType::Int32, false),
    ]))
}

fn benchmark_partitions(
    config: &LargerIslandBenchmarkConfig,
    schema: SchemaRef,
) -> DataFusionResult<Vec<Vec<RecordBatch>>> {
    let mut batches = Vec::with_capacity(config.input_rows.div_ceil(config.batch_size));
    let mut start = 0usize;

    while start < config.input_rows {
        let len = (config.input_rows - start).min(config.batch_size);
        let mut keys = Vec::with_capacity(len);
        let mut values = Vec::with_capacity(len);
        let mut payload = Vec::with_capacity(len);

        for offset in 0..len {
            let row = start + offset;
            keys.push((row % config.group_count) as i32);
            values.push((row as i64 % 10_000) + 1);
            payload.push(((row.wrapping_mul(31)) % 100) as i32);
        }

        batches.push(RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(keys)),
                Arc::new(Int64Array::from(values)),
                Arc::new(Int32Array::from(payload)),
            ],
        )?);
        start += len;
    }

    Ok(vec![batches])
}

fn validate_nonzero(name: &'static str, value: usize) -> DataFusionResult<()> {
    if value == 0 {
        return Err(DataFusionError::Plan(format!("{name} must be greater than zero")));
    }
    Ok(())
}

fn metric_count(metrics: &MetricsSet, name: &str) -> usize {
    metrics.sum_by_name(name).map(|value| value.as_usize()).unwrap_or(0)
}

fn metric_duration(metrics: &MetricsSet, name: &str) -> Duration {
    duration_from_nanos(metric_count(metrics, name))
}

fn duration_from_nanos(nanos: usize) -> Duration {
    Duration::from_nanos(nanos.min(u64::MAX as usize) as u64)
}

fn duration_div(duration: Duration, divisor: usize) -> Duration {
    if divisor == 0 {
        return Duration::ZERO;
    }
    let nanos = duration.as_nanos() / divisor as u128;
    Duration::from_nanos(nanos.min(u64::MAX as u128) as u64)
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
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

fn speedup(cpu: Duration, gpu: Duration) -> f64 {
    if cpu.is_zero() {
        return 0.0;
    }
    if gpu.is_zero() {
        return f64::INFINITY;
    }
    cpu.as_secs_f64() / gpu.as_secs_f64()
}

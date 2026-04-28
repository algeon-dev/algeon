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
    pub input_batches: usize,
    pub input_rows: usize,
    pub host_to_device_uploads: usize,
    pub host_to_device_uploaded_rows: usize,
    pub host_to_device_uploaded_bytes: usize,
    pub device_to_host_materializations: usize,
    pub device_to_host_materialized_rows: usize,
    pub device_to_host_materialized_bytes: usize,
    pub record_batch_upload_time: Duration,
    pub cudf_operator_time: Duration,
    pub record_batch_materialization_time: Duration,
    pub elapsed_compute: Option<Duration>,
    pub output_rows: Option<usize>,
}

impl LargerIslandBenchmarkMetrics {
    pub fn from_metrics(metrics: &MetricsSet) -> Self {
        Self {
            input_batches: metric_count(metrics, "input_batches"),
            input_rows: metric_count(metrics, "input_rows"),
            host_to_device_uploads: metric_count(metrics, "host_to_device_uploads"),
            host_to_device_uploaded_rows: metric_count(metrics, "host_to_device_uploaded_rows"),
            host_to_device_uploaded_bytes: metric_count(metrics, "host_to_device_uploaded_bytes"),
            device_to_host_materializations: metric_count(metrics, "device_to_host_materializations"),
            device_to_host_materialized_rows: metric_count(metrics, "device_to_host_materialized_rows"),
            device_to_host_materialized_bytes: metric_count(metrics, "device_to_host_materialized_bytes"),
            record_batch_upload_time: metric_duration(metrics, "record_batch_upload_time"),
            cudf_operator_time: metric_duration(metrics, "cudf_operator_time"),
            record_batch_materialization_time: metric_duration(metrics, "record_batch_materialization_time"),
            elapsed_compute: metrics.elapsed_compute().map(duration_from_nanos),
            output_rows: metrics.output_rows(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LargerIslandBenchmarkPlanning {
    pub candidate_count: usize,
    pub rewritten_count: usize,
    pub rejected_count: usize,
    pub rewritten_shapes: Vec<&'static str>,
    pub rejected_reasons: Vec<FallbackReason>,
}

impl LargerIslandBenchmarkPlanning {
    pub fn from_report(report: &PlanningReport) -> Self {
        let mut rewritten_shapes = Vec::new();
        let mut rejected_reasons = Vec::new();

        for outcome in report.outcomes() {
            match outcome {
                CandidateOutcome::Rewritten { candidate } => rewritten_shapes.push(candidate.shape),
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
    pub fn to_text(&self) -> String {
        let mut output = String::new();
        let _ = writeln!(output, "larger_island_benchmark");
        let _ = writeln!(output, "input_rows={}", self.config.input_rows);
        let _ = writeln!(output, "batch_size={}", self.config.batch_size);
        let _ = writeln!(output, "group_count={}", self.config.group_count);
        let _ = writeln!(output, "limit={}", self.config.limit);
        let _ = writeln!(output, "iterations={}", self.config.iterations);
        let _ = writeln!(output, "cpu_baseline_wall_ms={:.3}", duration_ms(self.cpu_baseline.mean));
        let _ = writeln!(output, "gpu_island_wall_ms={:.3}", duration_ms(self.gpu_island.mean));
        let _ =
            writeln!(output, "record_batch_upload_ms={:.3}", duration_ms(self.gpu_metrics.record_batch_upload_time));
        let _ = writeln!(output, "cudf_operator_ms={:.3}", duration_ms(self.gpu_metrics.cudf_operator_time));
        let _ = writeln!(
            output,
            "record_batch_materialization_ms={:.3}",
            duration_ms(self.gpu_metrics.record_batch_materialization_time)
        );
        let _ = writeln!(output, "host_to_device_uploads={}", self.gpu_metrics.host_to_device_uploads);
        let _ = writeln!(output, "host_to_device_uploaded_rows={}", self.gpu_metrics.host_to_device_uploaded_rows);
        let _ = writeln!(output, "host_to_device_uploaded_bytes={}", self.gpu_metrics.host_to_device_uploaded_bytes);
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

    Ok(LargerIslandBenchmarkReport {
        config,
        cpu_baseline: TimedBenchmarkRun::from_samples(cpu_samples),
        gpu_island: TimedBenchmarkRun::from_samples(gpu_samples),
        gpu_metrics: final_gpu.metrics,
        planning: final_gpu.planning,
        output_batches: final_gpu.output_batches,
        output_rows: final_gpu.output_rows,
        peak_device_memory_bytes: None,
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

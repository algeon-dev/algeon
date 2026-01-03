use std::{
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

use crate::{CudfOptimizerConfig, GpuFallbackPolicy, optimize_plan};

mod config;
mod metrics;
mod report;
mod timing;

pub use config::LargerIslandBenchmarkConfig;
pub use metrics::{LargerIslandBenchmarkMetrics, PhaseCRuntimeGateStatus};
pub use report::{
    BenchmarkGateReason, BenchmarkGateStatus, LargerIslandBenchmarkAssessment, LargerIslandBenchmarkDecisionCriteria,
    LargerIslandBenchmarkPlanning, LargerIslandBenchmarkReport,
};
pub use timing::TimedBenchmarkRun;

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

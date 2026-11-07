// Minimal benchmark runner for TPC-DS used to compare algeon-datafusion native
// GPU execution against tuned CPU DataFusion.
//
// Run one invocation per engine: native_gpu or cpu_datafusion. Each invocation
// registers the TPC-DS tables once, then runs every requested query through
// DataFusion `(warmups + iterations)` times, printing one TSV row per measured
// iteration.

use std::{
    error::Error,
    io::{self, Write},
    sync::Arc,
    time::{Duration, Instant},
};

use algeon_datafusion::{
    backend::AlgeonGpuBackend,
    native_exec::{AlgeonNativeExec, NATIVE_ALGEON_DATAFUSION_METRIC_NAMES},
    native_report::{NativeDataFusionExecutionMetrics, NativeTaskOperatorMetricRow, NativeTaskRuntimeMetricRow},
    planner::{NativePlanningDiagnostics, plan_with_native_diagnostics},
    report::CandidateOutcome,
};
use arrow::record_batch::RecordBatch;
use datafusion::{
    common::utils::memory::get_record_batch_memory_size,
    execution::{SessionStateBuilder, context::TaskContext},
    physical_plan::{
        ExecutionPlan, collect,
        metrics::{MetricValue, MetricsSet},
    },
    prelude::SessionContext,
};
use query_planning::mode::StatisticalAggregateExecutionMode;
use query_runtime::runtime::GpuDeviceMemoryLimit;
use workloads::tpcds::{TPCDS_QUERIES, TpcdsQuery};

use crate::report_workload::{
    RelationalExecutionSelection,
    tpcds::{
        ALGEON_TPCDS_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE_ENV, TpcdsNativeProfileSettings,
        register_tpcds_tables_with_native_metadata, tpcds_native_analytics_optimizer_config_with_profile_settings,
        tpcds_native_profile_settings_and_device_memory_limit_from_env, tpcds_native_profile_settings_with_env,
        tpcds_native_report_session_config,
    },
};

mod config;
mod report;

use config::build_runtime;
use report::{NativeTimingMetrics, TSV_HEADER, print_metadata, print_progress, print_row};
pub async fn run() -> Result<(), Box<dyn Error>> {
    let cfg = BenchConfig::from_env()?;
    eprintln!(
        "[tpcds_gpu_cpu_bench] engine={} root={} queries={} warmups={} iterations={} native_profile={} native_device_budget_bytes={} native_max_source_chunk_bytes={} native_max_row_groups_per_chunk={} native_source_chunk_planning_mode={} relational_execution={}",
        cfg.engine.as_str(),
        cfg.root.display(),
        cfg.queries.iter().map(|q| q.id).collect::<Vec<_>>().join(","),
        cfg.warmups,
        cfg.iterations,
        cfg.native_profile.profile_name,
        cfg.native_profile.device_budget_bytes,
        cfg.native_profile.max_source_chunk_bytes,
        cfg.native_profile.max_row_groups_per_chunk,
        cfg.native_profile.source_chunk_planning_mode.as_str(),
        cfg.relational_execution.as_str(),
    );

    let runtime = build_runtime(&cfg).await?;

    print_metadata(&cfg);
    println!("{TSV_HEADER}");
    for query in &cfg.queries {
        for warmup_index in 0..cfg.warmups {
            print_progress(&cfg, query, warmup_index, true, "start");
            let outcome = run_once(&runtime, query).await;
            print_row(query, warmup_index, true, &outcome, cfg.engine);
            print_progress(&cfg, query, warmup_index, true, outcome.outcome);
        }
        for iteration in 0..cfg.iterations {
            print_progress(&cfg, query, iteration, false, "start");
            let outcome = run_once(&runtime, query).await;
            print_row(query, iteration, false, &outcome, cfg.engine);
            print_progress(&cfg, query, iteration, false, outcome.outcome);
        }
    }
    runtime.close()?;
    Ok(())
}

#[derive(Clone, Copy, Debug)]
enum Engine {
    NativeGpu,
    CpuDataFusion,
}

impl Engine {
    fn as_str(self) -> &'static str {
        match self {
            Engine::NativeGpu => "native_gpu",
            Engine::CpuDataFusion => "cpu_datafusion",
        }
    }
}

struct BenchRuntime {
    execution_ctx: SessionContext,
    native_rule_installed: bool,
    backend: Option<AlgeonGpuBackend>,
}

impl BenchRuntime {
    fn close(self) -> Result<(), Box<dyn Error>> {
        let Self { execution_ctx, native_rule_installed, backend } = self;
        let backend = {
            let _contexts = (execution_ctx, native_rule_installed);
            backend
        };
        if let Some(backend) = backend {
            backend.close_and_wait(Duration::from_secs(30)).map_err(|error| Box::new(error) as Box<dyn Error>)?;
        }
        Ok(())
    }
}

struct BenchConfig {
    engine: Engine,
    root: std::path::PathBuf,
    queries: Vec<&'static TpcdsQuery>,
    warmups: usize,
    iterations: usize,
    native_profile: TpcdsNativeProfileSettings,
    native_device_memory_limit: GpuDeviceMemoryLimit,
    relational_execution: RelationalExecutionSelection,
}

struct IterOutcome {
    wall: Option<Duration>,
    outcome: &'static str,
    planning_outcome: String,
    output_batches: Option<usize>,
    output_rows: Option<usize>,
    output_bytes: Option<usize>,
    retry_count: Option<u64>,
    native_runtime_started: bool,
    native_metrics: Option<Box<NativeDataFusionExecutionMetrics>>,
    timing_metrics: NativeTimingMetrics,
    failure: Option<String>,
}

async fn run_once(runtime: &BenchRuntime, query: &TpcdsQuery) -> IterOutcome {
    let ctx = &runtime.execution_ctx;
    let native_rule_installed = runtime.native_rule_installed;
    let started_plan = Instant::now();
    let df = match ctx.sql(query.sql).await {
        Ok(df) => df,
        Err(err) => {
            return IterOutcome {
                wall: Some(started_plan.elapsed()),
                outcome: "sql_planning_failed",
                planning_outcome: String::from("sql_to_logical_plan_failed"),
                output_batches: None,
                output_rows: None,
                output_bytes: None,
                retry_count: None,
                native_runtime_started: false,
                native_metrics: None,
                timing_metrics: NativeTimingMetrics::default(),
                failure: Some(err.to_string()),
            };
        },
    };
    let planned = if native_rule_installed {
        let (state, logical) = df.into_parts();
        plan_with_native_diagnostics(&state, &logical).await.map(|(plan, diagnostics)| (plan, Some(diagnostics)))
    } else {
        df.create_physical_plan().await.map(|plan| (plan, None))
    };
    let (plan, diagnostics) = match planned {
        Ok(planned) => planned,
        Err(err) => {
            return IterOutcome {
                wall: Some(started_plan.elapsed()),
                outcome: "physical_planning_failed",
                planning_outcome: String::from("physical_planning_failed"),
                output_batches: None,
                output_rows: None,
                output_bytes: None,
                retry_count: None,
                native_runtime_started: false,
                native_metrics: None,
                timing_metrics: NativeTimingMetrics::default(),
                failure: Some(err.to_string()),
            };
        },
    };
    let planning_outcome = planning_summary(diagnostics.as_ref());
    let native_runtime_started = plan_contains_native_runtime(plan.as_ref());
    if native_rule_installed && !native_runtime_started {
        return IterOutcome {
            wall: None,
            outcome: "native_rejected_not_run",
            planning_outcome,
            output_batches: None,
            output_rows: None,
            output_bytes: None,
            retry_count: None,
            native_runtime_started: false,
            native_metrics: None,
            timing_metrics: NativeTimingMetrics::default(),
            failure: None,
        };
    }

    let task_ctx = Arc::new(TaskContext::default());
    let started = Instant::now();
    let collected = collect(Arc::clone(&plan), task_ctx).await;
    let wall = started.elapsed();

    match collected {
        Ok(batches) => {
            let output_batches = batches.len();
            let output_rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
            let output_bytes: usize = batches.iter().map(get_record_batch_memory_size).sum();
            let metrics = plan_metrics_recursive(plan.as_ref());
            let native_metrics = native_metrics_from_metrics(Some(&metrics));
            let timing_metrics = NativeTimingMetrics::from_metrics(Some(&metrics));
            IterOutcome {
                wall: Some(wall),
                outcome: "succeeded",
                planning_outcome,
                output_batches: Some(output_batches),
                output_rows: Some(output_rows),
                output_bytes: Some(output_bytes),
                retry_count: metric_sum(Some(&metrics), "retry_count"),
                native_runtime_started,
                native_metrics,
                timing_metrics,
                failure: None,
            }
        },
        Err(err) => IterOutcome {
            wall: Some(wall),
            outcome: "execution_failed",
            planning_outcome,
            output_batches: None,
            output_rows: None,
            output_bytes: None,
            retry_count: None,
            native_runtime_started,
            native_metrics: None,
            timing_metrics: NativeTimingMetrics::default(),
            failure: Some(err.to_string()),
        },
    }
}

fn plan_contains_native_runtime(plan: &dyn ExecutionPlan) -> bool {
    plan.downcast_ref::<AlgeonNativeExec>().is_some()
        || plan.children().into_iter().any(|child| plan_contains_native_runtime(child.as_ref()))
}

fn plan_metrics_recursive(plan: &dyn ExecutionPlan) -> MetricsSet {
    let mut metrics = MetricsSet::new();
    append_plan_metrics(&mut metrics, plan);
    metrics
}

fn append_plan_metrics(metrics: &mut MetricsSet, plan: &dyn ExecutionPlan) {
    if let Some(plan_metrics) = plan.metrics() {
        for metric in plan_metrics.iter() {
            metrics.push(Arc::clone(metric));
        }
    }
    for child in plan.children() {
        append_plan_metrics(metrics, child.as_ref());
    }
}

fn native_metrics_from_metrics(metrics: Option<&MetricsSet>) -> Option<Box<NativeDataFusionExecutionMetrics>> {
    let metrics = metrics?;
    metrics
        .iter()
        .any(|metric| NATIVE_ALGEON_DATAFUSION_METRIC_NAMES.contains(&metric.value().name()))
        .then(|| Box::new(NativeDataFusionExecutionMetrics::from_metrics(metrics)))
}

fn planning_summary(diagnostics: Option<&NativePlanningDiagnostics>) -> String {
    diagnostics
        .map(|diag| {
            let selected = diag.report.selected_count();
            let not_selected = diag.report.not_selected_count();
            let reasons = diag
                .report
                .outcomes()
                .iter()
                .filter_map(|outcome| match outcome {
                    CandidateOutcome::NotSupported { capability_reason, .. } => Some(capability_reason.code()),
                    CandidateOutcome::NotSelectedByCost { cost_reason, .. } => Some(cost_reason.code()),
                    CandidateOutcome::Selected { .. } => None,
                })
                .collect::<Vec<_>>();
            let reasons_text = reasons.join(",");
            if reasons_text.is_empty() {
                format!("selected={selected} not_selected={not_selected}")
            } else {
                format!("selected={selected} not_selected={not_selected} reasons={reasons_text}")
            }
        })
        .unwrap_or_else(|| String::from("cpu_only"))
}

fn metric_sum(metrics: Option<&MetricsSet>, name: &str) -> Option<u64> {
    let metrics = metrics?;
    let mut total: Option<u64> = None;
    for metric in metrics.iter() {
        if metric.value().name() == name {
            let value = match metric.value() {
                MetricValue::Count { count, .. } => count.value() as u64,
                MetricValue::Gauge { gauge, .. } => gauge.value() as u64,
                _ => continue,
            };
            total = Some(total.unwrap_or(0).saturating_add(value));
        }
    }
    total
}

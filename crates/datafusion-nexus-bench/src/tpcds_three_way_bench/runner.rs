// Minimal benchmark runner for TPC-DS used to compare datafusion-nexus native
// GPU execution against tuned CPU DataFusion.
//
// Run separate invocations per session: native_gpu, cpu_datafusion, and
// cost_declined_datafusion for native plans declined by cost. Each invocation registers
// the TPC-DS tables once, then runs every requested query through DataFusion
// `(warmups + iterations)` times, printing one TSV row per measured iteration.

use std::{
    collections::BTreeMap,
    error::Error,
    io::{self, Write},
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};

use arrow::record_batch::RecordBatch;
use datafusion::{
    common::utils::memory::get_record_batch_memory_size,
    execution::{SessionStateBuilder, context::TaskContext},
    physical_plan::{
        collect,
        metrics::{MetricValue, MetricsSet},
    },
    prelude::{SessionConfig, SessionContext},
};
use datafusion_nexus::{
    native_exec::{
        NATIVE_NEXUS_DATAFUSION_METRIC_NAMES, NexusNativeComposedExec, NexusNativeExec, NexusNativeGpuProducerExec,
    },
    native_report::{NativeDataFusionExecutionMetrics, NativeTaskOperatorMetricRow, NativeTaskRuntimeMetricRow},
    planner::NexusNativePhysicalOptimizerRule,
    report::CandidateOutcome,
    session::NexusSessionStateBuilderExt,
};
use datafusion_nexus_workloads::tpcds::{TPCDS_QUERIES, TpcdsQuery};
use nexus_query_engine::{
    harness::max_option as max_optional_usize,
    mode::{NativePipelineExecutionMode, StatisticalAggregateExecutionMode},
};

use crate::report_workload::{
    RelationalExecutionSelection,
    tpcds::{
        NEXUS_TPCDS_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE_ENV, NEXUS_TPCDS_NATIVE_STDDEV_SAMP_EXECUTION_MODE_ENV,
        TpcdsNativeProfileSettings, register_tpcds_tables_with_native_metadata,
        tpcds_native_analytics_optimizer_config_with_cost_selection_and_profile_settings,
        tpcds_native_analytics_optimizer_config_with_profile_settings, tpcds_native_profile_settings_from_env,
        tpcds_native_profile_settings_with_env, tpcds_native_report_session_config,
    },
};

mod config;
mod report;

use config::build_runtime;
use report::{NativeTimingMetrics, TSV_HEADER, print_metadata, print_progress, print_row};
pub async fn run() -> Result<(), Box<dyn Error>> {
    let cfg = BenchConfig::from_env()?;
    eprintln!(
        "[tpcds_three_way_bench] engine={} root={} queries={} warmups={} iterations={} native_profile={} native_device_budget_bytes={} native_max_source_chunk_bytes={} native_max_row_groups_per_chunk={} native_source_chunk_planning_mode={} relational_execution={} native_pipeline_execution_mode={}",
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
        cfg.native_pipeline_execution_mode.as_str(),
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
    Ok(())
}

#[derive(Clone, Copy, Debug)]
enum Engine {
    NativeGpu,
    CpuDataFusion,
    CostDeclinedDataFusion,
}

impl Engine {
    fn as_str(self) -> &'static str {
        match self {
            Engine::NativeGpu => "native_gpu",
            Engine::CpuDataFusion => "cpu_datafusion",
            Engine::CostDeclinedDataFusion => "cost_declined_datafusion",
        }
    }
}

struct BenchRuntime {
    execution_ctx: SessionContext,
    optimizer: Option<Arc<NexusNativePhysicalOptimizerRule>>,
    native_planning: Option<NativePlanningRuntime>,
    skip_cost_declined: bool,
}

struct NativePlanningRuntime {
    ctx: SessionContext,
    optimizer: Arc<NexusNativePhysicalOptimizerRule>,
}

struct BenchConfig {
    engine: Engine,
    root: std::path::PathBuf,
    queries: Vec<&'static TpcdsQuery>,
    warmups: usize,
    iterations: usize,
    native_profile: TpcdsNativeProfileSettings,
    relational_execution: RelationalExecutionSelection,
    native_pipeline_execution_mode: NativePipelineExecutionMode,
}

struct IterOutcome {
    wall: Option<Duration>,
    outcome: &'static str,
    planning_outcome: String,
    output_batches: Option<usize>,
    output_rows: Option<usize>,
    output_bytes: Option<usize>,
    reservation_live_peak_bytes: Option<u64>,
    reservation_denied_count: Option<u64>,
    retry_count: Option<u64>,
    native_runtime_started: bool,
    native_metrics: Option<NativeDataFusionExecutionMetrics>,
    timing_metrics: NativeTimingMetrics,
    failure: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PlanningSummary {
    planning_outcome: String,
    native_runtime_started: bool,
    cost_reason: Option<String>,
}

async fn run_once(runtime: &BenchRuntime, query: &TpcdsQuery) -> IterOutcome {
    if let Some(native_planning) = runtime.native_planning.as_ref() {
        return run_cost_declined_datafusion_once(&runtime.execution_ctx, native_planning, query).await;
    }

    run_query_collect(
        &runtime.execution_ctx,
        runtime.optimizer.as_deref(),
        query,
        None,
        None,
        runtime.skip_cost_declined,
    )
    .await
}

async fn run_cost_declined_datafusion_once(
    cpu_ctx: &SessionContext,
    native_planning: &NativePlanningRuntime,
    query: &TpcdsQuery,
) -> IterOutcome {
    let native_planning_summary = match plan_native_cost(&native_planning.ctx, &native_planning.optimizer, query).await
    {
        Ok(summary) => summary,
        Err(outcome) => return outcome,
    };

    let Some(reason) = native_planning_summary.cost_reason.as_ref() else {
        return IterOutcome {
            wall: None,
            outcome: "skipped_continue_native",
            planning_outcome: format!("continue_native native_planning={}", native_planning_summary.planning_outcome),
            output_batches: None,
            output_rows: None,
            output_bytes: None,
            reservation_live_peak_bytes: None,
            reservation_denied_count: None,
            retry_count: None,
            native_runtime_started: false,
            native_metrics: None,
            timing_metrics: NativeTimingMetrics::default(),
            failure: None,
        };
    };

    run_query_collect(
        cpu_ctx,
        None,
        query,
        Some(format!(
            "not_selected_by_cost cost_reason={reason} native_planning={}",
            native_planning_summary.planning_outcome
        )),
        Some(false),
        false,
    )
    .await
}

async fn plan_native_cost(
    ctx: &SessionContext,
    optimizer: &NexusNativePhysicalOptimizerRule,
    query: &TpcdsQuery,
) -> Result<PlanningSummary, IterOutcome> {
    let started_plan = Instant::now();
    let df = match ctx.sql(query.sql).await {
        Ok(df) => df,
        Err(err) => {
            return Err(IterOutcome {
                wall: Some(started_plan.elapsed()),
                outcome: "native_sql_planning_failed",
                planning_outcome: String::from("native_sql_to_logical_plan_failed"),
                output_batches: None,
                output_rows: None,
                output_bytes: None,
                reservation_live_peak_bytes: None,
                reservation_denied_count: None,
                retry_count: None,
                native_runtime_started: false,
                native_metrics: None,
                timing_metrics: NativeTimingMetrics::default(),
                failure: Some(err.to_string()),
            });
        },
    };
    if let Err(err) = df.create_physical_plan().await {
        return Err(IterOutcome {
            wall: Some(started_plan.elapsed()),
            outcome: "native_physical_planning_failed",
            planning_outcome: String::from("native_physical_planning_failed"),
            output_batches: None,
            output_rows: None,
            output_bytes: None,
            reservation_live_peak_bytes: None,
            reservation_denied_count: None,
            retry_count: None,
            native_runtime_started: false,
            native_metrics: None,
            timing_metrics: NativeTimingMetrics::default(),
            failure: Some(err.to_string()),
        });
    }

    Ok(planning_summary(Some(optimizer)))
}

async fn run_query_collect(
    ctx: &SessionContext,
    optimizer: Option<&NexusNativePhysicalOptimizerRule>,
    query: &TpcdsQuery,
    planning_outcome_override: Option<String>,
    native_runtime_started_override: Option<bool>,
    skip_cost_declined: bool,
) -> IterOutcome {
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
                reservation_live_peak_bytes: None,
                reservation_denied_count: None,
                retry_count: None,
                native_runtime_started: false,
                native_metrics: None,
                timing_metrics: NativeTimingMetrics::default(),
                failure: Some(err.to_string()),
            };
        },
    };
    let plan = match df.create_physical_plan().await {
        Ok(plan) => plan,
        Err(err) => {
            return IterOutcome {
                wall: Some(started_plan.elapsed()),
                outcome: "physical_planning_failed",
                planning_outcome: String::from("physical_planning_failed"),
                output_batches: None,
                output_rows: None,
                output_bytes: None,
                reservation_live_peak_bytes: None,
                reservation_denied_count: None,
                retry_count: None,
                native_runtime_started: false,
                native_metrics: None,
                timing_metrics: NativeTimingMetrics::default(),
                failure: Some(err.to_string()),
            };
        },
    };
    let planning_summary = planning_summary(optimizer);
    let planning_outcome = planning_outcome_override.unwrap_or(planning_summary.planning_outcome);
    let physical_plan_contains_native_runtime = plan_contains_native_runtime(plan.as_ref());
    let native_runtime_started = native_runtime_started_override.unwrap_or(physical_plan_contains_native_runtime);
    if skip_cost_declined && let Some(reason) = planning_summary.cost_reason {
        return IterOutcome {
            wall: None,
            outcome: "not_selected_by_cost",
            planning_outcome: format!("{planning_outcome} cost_reason={reason}"),
            output_batches: None,
            output_rows: None,
            output_bytes: None,
            reservation_live_peak_bytes: None,
            reservation_denied_count: None,
            retry_count: None,
            native_runtime_started: false,
            native_metrics: None,
            timing_metrics: NativeTimingMetrics::default(),
            failure: None,
        };
    }
    if skip_cost_declined && !physical_plan_contains_native_runtime {
        return IterOutcome {
            wall: None,
            outcome: "native_rejected_not_run",
            planning_outcome,
            output_batches: None,
            output_rows: None,
            output_bytes: None,
            reservation_live_peak_bytes: None,
            reservation_denied_count: None,
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
                reservation_live_peak_bytes: metric_sum(Some(&metrics), "reservation_live_peak_bytes"),
                reservation_denied_count: metric_sum(Some(&metrics), "reservation_denied_count"),
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
            reservation_live_peak_bytes: None,
            reservation_denied_count: None,
            retry_count: None,
            native_runtime_started,
            native_metrics: None,
            timing_metrics: NativeTimingMetrics::default(),
            failure: Some(err.to_string()),
        },
    }
}

fn plan_contains_native_runtime(plan: &dyn datafusion::physical_plan::ExecutionPlan) -> bool {
    plan.downcast_ref::<NexusNativeExec>().is_some()
        || plan.downcast_ref::<NexusNativeComposedExec>().is_some()
        || plan.downcast_ref::<NexusNativeGpuProducerExec>().is_some()
        || plan.children().into_iter().any(|child| plan_contains_native_runtime(child.as_ref()))
}

fn plan_metrics_recursive(plan: &dyn datafusion::physical_plan::ExecutionPlan) -> MetricsSet {
    let mut metrics = MetricsSet::new();
    append_plan_metrics(&mut metrics, plan);
    metrics
}

fn append_plan_metrics(metrics: &mut MetricsSet, plan: &dyn datafusion::physical_plan::ExecutionPlan) {
    if let Some(plan_metrics) = plan.metrics() {
        for metric in plan_metrics.iter() {
            metrics.push(Arc::clone(metric));
        }
    }
    for child in plan.children() {
        append_plan_metrics(metrics, child.as_ref());
    }
}

fn native_metrics_from_metrics(metrics: Option<&MetricsSet>) -> Option<NativeDataFusionExecutionMetrics> {
    let metrics = metrics?;
    metrics
        .iter()
        .any(|metric| NATIVE_NEXUS_DATAFUSION_METRIC_NAMES.contains(&metric.value().name()))
        .then(|| NativeDataFusionExecutionMetrics::from_metrics(metrics))
}

fn planning_summary(optimizer: Option<&NexusNativePhysicalOptimizerRule>) -> PlanningSummary {
    optimizer
        .and_then(|rule| rule.last_planning_diagnostics())
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
            let planning_outcome = if reasons_text.is_empty() {
                format!("selected={selected} not_selected={not_selected}")
            } else {
                format!("selected={selected} not_selected={not_selected} reasons={reasons_text}")
            };
            PlanningSummary {
                planning_outcome,
                native_runtime_started: selected > 0,
                cost_reason: reasons.iter().copied().find(|reason| cost_reason_code(reason)).map(str::to_string),
            }
        })
        .unwrap_or_else(|| PlanningSummary {
            planning_outcome: String::from("cpu_only"),
            native_runtime_started: false,
            cost_reason: None,
        })
}

fn cost_reason_code(reason: &str) -> bool {
    matches!(
        reason,
        "native_small_selective_topk_not_selected_by_cost"
            | "native_small_selective_residual_join_not_selected_by_cost"
            | "native_small_selective_full_sort_not_selected_by_cost"
    )
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

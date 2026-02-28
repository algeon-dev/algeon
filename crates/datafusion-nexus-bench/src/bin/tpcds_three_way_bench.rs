// Minimal benchmark runner for TPC-DS used to compare datafusion-nexus native
// GPU execution against tuned CPU DataFusion.
//
// Run separate invocations per session: native_gpu, cpu_datafusion, and
// hybrid_cpu_preferred for CPU-preferred native plans. Each invocation registers
// the TPC-DS tables once, then runs every requested query through DataFusion
// `(warmups + iterations)` times, printing one TSV row per measured iteration.

use std::{
    collections::BTreeMap,
    error::Error,
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
    CandidateOutcome, CudfNativePhysicalOptimizerRule, CudfSessionStateBuilderExt, NativeDataFusionExecutionMetrics,
    NativeTaskOperatorMetricRow, NativeTaskRuntimeMetricRow, StatisticalAggregateExecutionMode,
};
use datafusion_nexus_bench::{
    CUDF_TPCDS_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE_ENV, CUDF_TPCDS_NATIVE_STDDEV_SAMP_EXECUTION_MODE_ENV,
    TPCDS_QUERIES, TpcdsNativeProfileSettings, TpcdsQuery, register_tpcds_tables,
    tpcds_native_analytics_optimizer_config_with_profile_settings, tpcds_native_profile_settings_from_env,
    tpcds_native_profile_settings_with_env, tpcds_native_report_session_config,
};

const DEFAULT_QUERY_IDS: &[&str] =
    &["q03", "q07", "q22", "q26", "q32", "q37", "q42", "q52", "q55", "q62", "q82", "q85", "q92", "q93", "q97"];

const TSV_HEADER: &str = concat!(
    "engine\tquery_id\titeration\twarmup\twall_ms\toutcome\tplanning_outcome\toutput_batches\toutput_rows\toutput_bytes",
    "\treservation_live_peak_bytes\treservation_denied_count\tretry_count\tnative_runtime_started\tfailure",
    "\tparquet_read_time_ms\tsource_read_time_ms\tsource_read_queue_wait_time_ms\tsource_cache_lookup_time_ms",
    "\thost_fetch_budget_wait_time_ms\tsource_fetch_concurrency_wait_time_ms\tsource_fetch_time_ms\tsource_decode_time_ms",
    "\toperator_time_ms\tchunk_operator_time_ms\tfinal_merge_time_ms\tfinal_sort_time_ms\tmaterialization_time_ms",
    "\taggregate_libcudf_runtime_probe_count\taggregate_libcudf_runtime_probe_elapsed_ns",
    "\taggregate_libcudf_runtime_probe_input_rows\taggregate_exact_cardinality_probe_count",
    "\taggregate_exact_cardinality_probe_elapsed_ns\taggregate_exact_cardinality_probe_input_rows",
    "\taggregate_prepared_sort_groupby_elapsed_ns",
    "\taggregate_encoded_key_partial_elapsed_ns",
    "\truntime_operator_class\truntime_operator_class_ms\tpeak_memory_operator_class\tpeak_memory_operator_class_bytes",
    "\ttask_reservation_operator_class\ttask_reservation_operator_class_bytes",
    "\tsource_chunks_planned\tsource_chunks_attempted\tsource_chunks_committed\tsource_chunks_retried",
    "\tsource_read_prefetches_started\tsource_read_prefetches_completed\tsource_read_prefetches_skipped",
    "\tsource_read_prefetches_abandoned\tsource_cache_hits\tsource_cache_misses\tsource_cache_inserts",
    "\tsource_cache_insert_skipped\tsource_cache_retained_bytes\tsource_fetch_bytes\tsource_fetch_count",
    "\trow_groups_planned\trow_groups_committed",
    "\thash_join_build_count\tfiltered_join_build_count\tone_shot_join_count\tpartitioned_join_count",
    "\tjoin_partitioning_status\tdynamic_filter_produced_count\tdynamic_filter_skipped_count",
    "\tdynamic_filter_applied_count\tdynamic_filter_row_groups_pruned\tdynamic_filter_source_bytes_pruned",
    "\tdynamic_filter_reader_predicate_rows_before\tdynamic_filter_reader_predicate_rows_after",
    "\tdynamic_filter_reader_predicate_rows_filtered",
    "\tdynamic_filter_post_decode_rows_before\tdynamic_filter_post_decode_rows_after",
    "\tdynamic_filter_post_decode_rows_filtered\tdynamic_filter_source_post_decode_rows_filtered",
    "\tdynamic_filter_filter_post_decode_rows_filtered\tdynamic_filter_join_probe_post_decode_rows_filtered",
    "\tdynamic_filter_post_decode_reduction_stage",
    "\tdynamic_filter_measured_reduction_status\tdynamic_filter_work_avoidance_status\tdynamic_filter_runtime_rows",
    "\taggregate_strategy_rows\tjoin_strategy_rows\ttask_operator_rows\ttask_runtime_rows",
    "\tstatistical_aggregate_execution_mode\tstatistical_aggregate_tolerance_contract",
);

#[tokio::main(flavor = "multi_thread", worker_threads = 16)]
async fn main() -> Result<(), Box<dyn Error>> {
    let cfg = BenchConfig::from_env()?;
    eprintln!(
        "[tpcds_three_way_bench] engine={} root={} queries={} warmups={} iterations={} native_profile={} native_device_budget_bytes={} native_max_source_chunk_bytes={} native_max_row_groups_per_chunk={} native_source_chunk_planning_mode={}",
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
    );

    let runtime = build_runtime(&cfg).await?;

    print_metadata(&cfg);
    println!("{TSV_HEADER}");
    for query in &cfg.queries {
        for warmup_index in 0..cfg.warmups {
            let outcome = run_once(&runtime, query).await;
            print_row(query, warmup_index, true, &outcome, cfg.engine);
        }
        for iteration in 0..cfg.iterations {
            let outcome = run_once(&runtime, query).await;
            print_row(query, iteration, false, &outcome, cfg.engine);
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
enum Engine {
    NativeGpu,
    CpuDataFusion,
    HybridCpuPreferred,
}

impl Engine {
    fn as_str(self) -> &'static str {
        match self {
            Engine::NativeGpu => "native_gpu",
            Engine::CpuDataFusion => "cpu_datafusion",
            Engine::HybridCpuPreferred => "hybrid_cpu_preferred",
        }
    }
}

struct BenchRuntime {
    execution_ctx: SessionContext,
    optimizer: Option<Arc<CudfNativePhysicalOptimizerRule>>,
    native_planning: Option<NativePlanningRuntime>,
    skip_cpu_preferred: bool,
}

struct NativePlanningRuntime {
    ctx: SessionContext,
    optimizer: Arc<CudfNativePhysicalOptimizerRule>,
}

struct BenchConfig {
    engine: Engine,
    root: std::path::PathBuf,
    queries: Vec<&'static TpcdsQuery>,
    warmups: usize,
    iterations: usize,
    native_profile: TpcdsNativeProfileSettings,
}

impl BenchConfig {
    fn from_env() -> Result<Self, Box<dyn Error>> {
        let engine = match std::env::var("CUDF_TPCDS_BENCH_ENGINE").as_deref() {
            Ok("native_gpu") => Engine::NativeGpu,
            Ok("cpu_datafusion") => Engine::CpuDataFusion,
            Ok("hybrid_cpu_preferred") => Engine::HybridCpuPreferred,
            Ok(other) => return Err(format!("unsupported CUDF_TPCDS_BENCH_ENGINE={other}").into()),
            Err(_) => {
                return Err(
                    "CUDF_TPCDS_BENCH_ENGINE must be set to native_gpu, cpu_datafusion, or hybrid_cpu_preferred".into(),
                );
            },
        };
        let root = std::env::var("CUDF_TPCDS_PARQUET_DIR")
            .map_err(|_| "CUDF_TPCDS_PARQUET_DIR must be set to the SF100 parquet root")?;
        let root = std::path::PathBuf::from(root);
        let queries_csv = std::env::var("CUDF_TPCDS_BENCH_QUERIES").unwrap_or_else(|_| DEFAULT_QUERY_IDS.join(","));
        let queries = bench_queries_from_csv(&queries_csv)?;
        let warmups = env_usize("CUDF_TPCDS_BENCH_WARMUPS", 1)?;
        let iterations = env_usize("CUDF_TPCDS_BENCH_ITERATIONS", 2)?;
        if iterations == 0 {
            return Err("CUDF_TPCDS_BENCH_ITERATIONS must be >= 1".into());
        }
        let native_profile = match engine {
            Engine::CpuDataFusion => tpcds_native_profile_settings_with_env(|_| None),
            Engine::NativeGpu | Engine::HybridCpuPreferred => tpcds_native_profile_settings_from_env(),
        };
        Ok(Self { engine, root, queries, warmups, iterations, native_profile })
    }
}

fn print_metadata(cfg: &BenchConfig) {
    println!("# tpcds_three_way_bench_tsv_schema_version=1");
    println!("# tpcds_benchmark_engine={}", cfg.engine.as_str());
    println!("# tpcds_benchmark_root={}", cfg.root.display());
    println!("# tpcds_benchmark_query_ids={}", cfg.queries.iter().map(|q| q.id).collect::<Vec<_>>().join(","));
    println!("# tpcds_benchmark_warmups={}", cfg.warmups);
    println!("# tpcds_benchmark_iterations={}", cfg.iterations);
    println!("# tpcds_benchmark_wall_time_fields=per_measured_iteration");
    println!("# tpcds_benchmark_native_profile_name={}", cfg.native_profile.profile_name);
    println!("# tpcds_benchmark_native_profile_source={}", cfg.native_profile.profile_source);
    println!("# tpcds_benchmark_native_device_budget_bytes={}", cfg.native_profile.device_budget_bytes);
    println!("# tpcds_benchmark_native_max_source_chunk_bytes={}", cfg.native_profile.max_source_chunk_bytes);
    println!("# tpcds_benchmark_native_max_row_groups_per_chunk={}", cfg.native_profile.max_row_groups_per_chunk);
    println!("# tpcds_benchmark_native_max_retry_attempts={}", cfg.native_profile.max_retry_attempts);
    println!("# tpcds_benchmark_native_min_retry_chunk_row_groups={}", cfg.native_profile.min_retry_chunk_row_groups);
    println!(
        "# tpcds_benchmark_native_source_chunk_planning_mode={}",
        cfg.native_profile.source_chunk_planning_mode.as_str()
    );
    let statistical_aggregate_execution_mode = native_statistical_aggregate_execution_mode_from_env();
    println!(
        "# tpcds_benchmark_native_statistical_aggregate_execution_mode={}",
        statistical_aggregate_execution_mode.as_str()
    );
    println!(
        "# tpcds_benchmark_native_statistical_aggregate_tolerance_contract={}",
        statistical_aggregate_execution_mode.tolerance_contract().as_str()
    );
}

fn bench_queries_from_csv(queries_csv: &str) -> Result<Vec<&'static TpcdsQuery>, Box<dyn Error>> {
    if queries_csv.trim().eq_ignore_ascii_case("all") {
        return Ok(TPCDS_QUERIES.iter().collect());
    }
    let query_ids: Vec<String> =
        queries_csv.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
    let queries: Vec<&'static TpcdsQuery> = query_ids
        .iter()
        .map(|id| TPCDS_QUERIES.iter().find(|q| q.id == id).ok_or_else(|| format!("unknown TPC-DS query id: {id}")))
        .collect::<Result<_, _>>()?;
    Ok(queries)
}

fn env_usize(name: &str, default: usize) -> Result<usize, Box<dyn Error>> {
    match std::env::var(name) {
        Ok(value) => value.parse::<usize>().map_err(|e| format!("{name}={value}: {e}").into()),
        Err(_) => Ok(default),
    }
}

async fn build_runtime(cfg: &BenchConfig) -> Result<BenchRuntime, Box<dyn Error>> {
    match cfg.engine {
        Engine::CpuDataFusion => {
            let session_config = tuned_cpu_session_config();
            let ctx = SessionContext::new_with_config(session_config);
            register_tpcds_tables(&ctx, &cfg.root).await?;
            Ok(BenchRuntime { execution_ctx: ctx, optimizer: None, native_planning: None, skip_cpu_preferred: false })
        },
        Engine::NativeGpu => {
            let optimizer_config = tpcds_native_analytics_optimizer_config_with_profile_settings(&cfg.native_profile);
            let rule = Arc::new(CudfNativePhysicalOptimizerRule::new(optimizer_config));
            let state = SessionStateBuilder::new_with_default_features()
                .with_config(tpcds_native_report_session_config())
                .with_cudf_native_rule(Arc::clone(&rule))
                .build();
            let ctx = SessionContext::new_with_state(state);
            register_tpcds_tables(&ctx, &cfg.root).await?;
            Ok(BenchRuntime {
                execution_ctx: ctx,
                optimizer: Some(rule),
                native_planning: None,
                skip_cpu_preferred: true,
            })
        },
        Engine::HybridCpuPreferred => {
            let cpu_ctx = SessionContext::new_with_config(tuned_cpu_session_config());
            register_tpcds_tables(&cpu_ctx, &cfg.root).await?;
            let (native_ctx, optimizer) = build_native_planning_context(&cfg.root, &cfg.native_profile).await?;
            Ok(BenchRuntime {
                execution_ctx: cpu_ctx,
                optimizer: None,
                native_planning: Some(NativePlanningRuntime { ctx: native_ctx, optimizer }),
                skip_cpu_preferred: false,
            })
        },
    }
}

async fn build_native_planning_context(
    root: &Path,
    native_profile: &TpcdsNativeProfileSettings,
) -> Result<(SessionContext, Arc<CudfNativePhysicalOptimizerRule>), Box<dyn Error>> {
    let optimizer_config = tpcds_native_analytics_optimizer_config_with_profile_settings(native_profile);
    let rule = Arc::new(CudfNativePhysicalOptimizerRule::new(optimizer_config));
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(tpcds_native_report_session_config())
        .with_cudf_native_rule(Arc::clone(&rule))
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_tpcds_tables(&ctx, root).await?;
    Ok((ctx, rule))
}

fn tuned_cpu_session_config() -> SessionConfig {
    let target_partitions =
        std::env::var("CUDF_TPCDS_CPU_TARGET_PARTITIONS").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(16);
    let batch_size =
        std::env::var("CUDF_TPCDS_CPU_BATCH_SIZE").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(8192);
    let mut config = SessionConfig::new()
        .with_target_partitions(target_partitions)
        .with_batch_size(batch_size)
        .with_repartition_file_scans(true)
        .with_repartition_aggregations(true)
        .with_repartition_joins(true)
        .with_repartition_sorts(true);
    config.options_mut().optimizer.enable_topk_dynamic_filter_pushdown = true;
    config.options_mut().optimizer.enable_aggregate_dynamic_filter_pushdown = true;
    config.options_mut().optimizer.enable_join_dynamic_filter_pushdown = true;
    config.options_mut().optimizer.enable_dynamic_filter_pushdown = true;
    config.options_mut().execution.parquet.pushdown_filters = false;
    config
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
    cpu_preferred_reason: Option<String>,
}

async fn run_once(runtime: &BenchRuntime, query: &TpcdsQuery) -> IterOutcome {
    if let Some(native_planning) = runtime.native_planning.as_ref() {
        return run_hybrid_cpu_preferred_once(&runtime.execution_ctx, native_planning, query).await;
    }

    run_query_collect(
        &runtime.execution_ctx,
        runtime.optimizer.as_deref(),
        query,
        None,
        None,
        runtime.skip_cpu_preferred,
    )
    .await
}

async fn run_hybrid_cpu_preferred_once(
    cpu_ctx: &SessionContext,
    native_planning: &NativePlanningRuntime,
    query: &TpcdsQuery,
) -> IterOutcome {
    let native_planning_summary =
        match plan_native_preference(&native_planning.ctx, &native_planning.optimizer, query).await {
            Ok(summary) => summary,
            Err(outcome) => return outcome,
        };

    let Some(reason) = native_planning_summary.cpu_preferred_reason.as_ref() else {
        return IterOutcome {
            wall: None,
            outcome: "skipped_not_cpu_preferred",
            planning_outcome: format!("gpu_preferred native_planning={}", native_planning_summary.planning_outcome),
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
        Some(format!("cpu_preferred reason={reason} native_planning={}", native_planning_summary.planning_outcome)),
        Some(false),
        false,
    )
    .await
}

async fn plan_native_preference(
    ctx: &SessionContext,
    optimizer: &CudfNativePhysicalOptimizerRule,
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
    optimizer: Option<&CudfNativePhysicalOptimizerRule>,
    query: &TpcdsQuery,
    planning_outcome_override: Option<String>,
    native_runtime_started_override: Option<bool>,
    skip_cpu_preferred: bool,
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
    let native_runtime_started = native_runtime_started_override.unwrap_or(planning_summary.native_runtime_started);
    if skip_cpu_preferred && let Some(reason) = planning_summary.cpu_preferred_reason {
        return IterOutcome {
            wall: None,
            outcome: "cpu_preferred_not_run",
            planning_outcome: format!("{planning_outcome} cpu_preferred_reason={reason}"),
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
    if skip_cpu_preferred && !native_runtime_started {
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
            let metrics = plan.metrics();
            let native_metrics = metrics.as_ref().map(NativeDataFusionExecutionMetrics::from_metrics);
            let timing_metrics = NativeTimingMetrics::from_metrics(metrics.as_ref());
            IterOutcome {
                wall: Some(wall),
                outcome: "succeeded",
                planning_outcome,
                output_batches: Some(output_batches),
                output_rows: Some(output_rows),
                output_bytes: Some(output_bytes),
                reservation_live_peak_bytes: metric_sum(&metrics, "reservation_live_peak_bytes"),
                reservation_denied_count: metric_sum(&metrics, "reservation_denied_count"),
                retry_count: metric_sum(&metrics, "retry_count"),
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

fn planning_summary(optimizer: Option<&CudfNativePhysicalOptimizerRule>) -> PlanningSummary {
    optimizer
        .and_then(|rule| rule.last_planning_diagnostics())
        .map(|diag| {
            let rewritten = diag.report.rewritten_count();
            let rejected = diag.report.rejected_count();
            let reasons = diag
                .report
                .outcomes()
                .iter()
                .filter_map(|outcome| match outcome {
                    CandidateOutcome::Rejected { reason, .. } => Some(reason.code()),
                    CandidateOutcome::Rewritten { .. } => None,
                })
                .collect::<Vec<_>>();
            let reasons_text = reasons.join(",");
            let planning_outcome = if reasons_text.is_empty() {
                format!("rewritten={rewritten} rejected={rejected}")
            } else {
                format!("rewritten={rewritten} rejected={rejected} reasons={reasons_text}")
            };
            PlanningSummary {
                planning_outcome,
                native_runtime_started: rewritten > 0,
                cpu_preferred_reason: reasons
                    .iter()
                    .copied()
                    .find(|reason| cpu_preferred_reason_code(reason))
                    .map(str::to_string),
            }
        })
        .unwrap_or_else(|| PlanningSummary {
            planning_outcome: String::from("cpu_only"),
            native_runtime_started: false,
            cpu_preferred_reason: None,
        })
}

fn cpu_preferred_reason_code(reason: &str) -> bool {
    matches!(reason, "native_small_selective_topk_cpu_preferred" | "native_small_selective_residual_join_cpu_preferred")
}

fn metric_sum(metrics: &Option<MetricsSet>, name: &str) -> Option<u64> {
    let metrics = metrics.as_ref()?;
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

fn print_row(query: &TpcdsQuery, iteration: usize, warmup: bool, outcome: &IterOutcome, engine: Engine) {
    println!("{}", row_fields(query, iteration, warmup, outcome, engine).join("\t"));
}

fn row_fields(
    query: &TpcdsQuery,
    iteration: usize,
    warmup: bool,
    outcome: &IterOutcome,
    engine: Engine,
) -> Vec<String> {
    let p1 = P1BottleneckSummary::from_outcome(outcome);
    let metrics = outcome.native_metrics.as_ref();
    vec![
        engine.as_str().to_string(),
        query.id.to_string(),
        iteration.to_string(),
        if warmup { "true" } else { "false" }.to_string(),
        duration_ms_text(outcome.wall),
        outcome.outcome.to_string(),
        sanitize(&outcome.planning_outcome),
        format_opt(outcome.output_batches),
        format_opt(outcome.output_rows),
        format_opt(outcome.output_bytes),
        format_opt(outcome.reservation_live_peak_bytes),
        format_opt(outcome.reservation_denied_count),
        format_opt(outcome.retry_count),
        outcome.native_runtime_started.to_string(),
        sanitize(outcome.failure.as_deref().unwrap_or("")),
        optional_f64_text(outcome.timing_metrics.parquet_read_time_ms),
        optional_f64_text(outcome.timing_metrics.source_read_time_ms),
        optional_f64_text(outcome.timing_metrics.source_read_queue_wait_time_ms),
        optional_f64_text(outcome.timing_metrics.source_cache_lookup_time_ms),
        optional_f64_text(outcome.timing_metrics.host_fetch_budget_wait_time_ms),
        optional_f64_text(outcome.timing_metrics.source_fetch_concurrency_wait_time_ms),
        optional_f64_text(outcome.timing_metrics.source_fetch_time_ms),
        optional_f64_text(outcome.timing_metrics.source_decode_time_ms),
        optional_f64_text(outcome.timing_metrics.operator_time_ms),
        optional_f64_text(outcome.timing_metrics.chunk_operator_time_ms),
        optional_f64_text(outcome.timing_metrics.final_merge_time_ms),
        optional_f64_text(outcome.timing_metrics.final_sort_time_ms),
        optional_f64_text(outcome.timing_metrics.materialization_time_ms),
        metric_field(metrics, |metrics| metrics.aggregate_libcudf_runtime_probe_count()),
        metric_field(metrics, |metrics| metrics.aggregate_libcudf_runtime_probe_elapsed_ns()),
        metric_field(metrics, |metrics| metrics.aggregate_libcudf_runtime_probe_input_rows()),
        metric_field(metrics, |metrics| metrics.aggregate_exact_cardinality_probe_count()),
        metric_field(metrics, |metrics| metrics.aggregate_exact_cardinality_probe_elapsed_ns()),
        metric_field(metrics, |metrics| metrics.aggregate_exact_cardinality_probe_input_rows()),
        metric_field(metrics, |metrics| metrics.aggregate_prepared_sort_groupby_elapsed_ns()),
        metric_field(metrics, |metrics| metrics.aggregate_encoded_key_partial_elapsed_ns()),
        p1.runtime_operator_class,
        optional_f64_text(p1.runtime_operator_class_ms),
        p1.peak_memory_operator_class,
        p1.peak_memory_operator_class_bytes.to_string(),
        p1.task_reservation_operator_class,
        p1.task_reservation_operator_class_bytes.to_string(),
        metric_field(metrics, |metrics| metrics.source_chunks_planned),
        metric_field(metrics, |metrics| metrics.source_chunks_attempted),
        metric_field(metrics, |metrics| metrics.source_chunks_committed),
        metric_field(metrics, |metrics| metrics.source_chunks_retried),
        metric_field(metrics, |metrics| metrics.source_read_prefetches_started),
        metric_field(metrics, |metrics| metrics.source_read_prefetches_completed),
        metric_field(metrics, |metrics| metrics.source_read_prefetches_skipped),
        metric_field(metrics, |metrics| metrics.source_read_prefetches_abandoned),
        metric_field(metrics, |metrics| metrics.source_cache_hits),
        metric_field(metrics, |metrics| metrics.source_cache_misses),
        metric_field(metrics, |metrics| metrics.source_cache_inserts),
        metric_field(metrics, |metrics| metrics.source_cache_insert_skipped),
        metric_field(metrics, |metrics| metrics.source_cache_retained_bytes),
        metric_field(metrics, |metrics| metrics.source_fetch_bytes),
        metric_field(metrics, |metrics| metrics.source_fetch_count),
        metric_field(metrics, |metrics| metrics.row_groups_planned),
        metric_field(metrics, |metrics| metrics.row_groups_committed),
        metric_field(metrics, |metrics| metrics.hash_join_build_count),
        metric_field(metrics, |metrics| metrics.filtered_join_build_count),
        metric_field(metrics, |metrics| metrics.one_shot_join_count),
        metrics.map_or_else(|| "not_run".to_string(), partitioned_join_count_text),
        metrics.map_or_else(|| "not_run".to_string(), join_partitioning_status),
        metric_field(metrics, |metrics| metrics.dynamic_filter_produced_count),
        metric_field(metrics, |metrics| metrics.dynamic_filter_skipped_count),
        metric_field(metrics, |metrics| metrics.dynamic_filter_applied_count),
        metric_field(metrics, |metrics| metrics.dynamic_filter_row_groups_pruned),
        metric_field(metrics, |metrics| metrics.dynamic_filter_source_bytes_pruned),
        metric_field(metrics, |metrics| metrics.dynamic_filter_reader_predicate_rows_before),
        metric_field(metrics, |metrics| metrics.dynamic_filter_reader_predicate_rows_after),
        metric_field(metrics, |metrics| metrics.dynamic_filter_reader_predicate_rows_filtered),
        metric_field(metrics, |metrics| metrics.dynamic_filter_post_decode_rows_before),
        metric_field(metrics, |metrics| metrics.dynamic_filter_post_decode_rows_after),
        metric_field(metrics, |metrics| metrics.dynamic_filter_post_decode_rows_filtered),
        metric_field(metrics, |metrics| metrics.dynamic_filter_source_post_decode_rows_filtered),
        metric_field(metrics, |metrics| metrics.dynamic_filter_filter_post_decode_rows_filtered),
        metric_field(metrics, |metrics| metrics.dynamic_filter_join_probe_post_decode_rows_filtered),
        metrics.map_or_else(
            || "unknown".to_string(),
            |metrics| metrics.dynamic_filter_post_decode_reduction_stage().as_str().to_string(),
        ),
        metrics.map_or_else(
            || "unknown".to_string(),
            |metrics| metrics.dynamic_filter_measured_reduction_status().as_str().to_string(),
        ),
        metrics.map_or_else(
            || "unknown".to_string(),
            |metrics| metrics.dynamic_filter_work_avoidance_status().as_str().to_string(),
        ),
        metrics.map_or_else(|| "not_run".to_string(), format_dynamic_filter_runtime_rows),
        metrics.map_or_else(|| "not_run".to_string(), format_aggregate_strategy_rows),
        metrics.map_or_else(|| "not_run".to_string(), format_join_strategy_rows),
        metrics.map_or_else(|| "not_run".to_string(), format_task_operator_rows),
        metrics.map_or_else(|| "not_run".to_string(), format_task_runtime_rows),
        statistical_aggregate_execution_mode_for_row(engine),
        statistical_aggregate_tolerance_contract_for_row(engine),
    ]
}

fn statistical_aggregate_execution_mode_for_row(engine: Engine) -> String {
    match engine {
        Engine::NativeGpu | Engine::HybridCpuPreferred => {
            native_statistical_aggregate_execution_mode_from_env().as_str().to_string()
        },
        Engine::CpuDataFusion => "not_applicable".to_string(),
    }
}

fn statistical_aggregate_tolerance_contract_for_row(engine: Engine) -> String {
    match engine {
        Engine::NativeGpu | Engine::HybridCpuPreferred => {
            native_statistical_aggregate_execution_mode_from_env().tolerance_contract().as_str().to_string()
        },
        Engine::CpuDataFusion => "not_applicable".to_string(),
    }
}

fn native_statistical_aggregate_execution_mode_from_env() -> StatisticalAggregateExecutionMode {
    std::env::var(CUDF_TPCDS_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE_ENV)
        .or_else(|_| std::env::var(CUDF_TPCDS_NATIVE_STDDEV_SAMP_EXECUTION_MODE_ENV))
        .ok()
        .map(|value| {
            value.parse::<StatisticalAggregateExecutionMode>().unwrap_or_else(|error| {
                panic!(
                    "unsupported TPC-DS statistical aggregate execution mode={value}; expected datafusion_exact_host_welford or gpu_native_tolerant: {error}"
                )
            })
        })
        .unwrap_or(StatisticalAggregateExecutionMode::GpuNativeTolerant)
}

fn format_opt<T: std::fmt::Display>(value: Option<T>) -> String {
    match value {
        Some(v) => v.to_string(),
        None => String::from(""),
    }
}

fn duration_ms_text(value: Option<Duration>) -> String {
    value.map_or_else(String::new, |value| format!("{:.3}", value.as_secs_f64() * 1000.0))
}

fn sanitize(s: &str) -> String {
    s.replace(['\t', '\n'], " ")
}

#[derive(Debug, Clone, Default)]
struct NativeTimingMetrics {
    parquet_read_time_ms: Option<f64>,
    source_read_time_ms: Option<f64>,
    source_read_queue_wait_time_ms: Option<f64>,
    source_cache_lookup_time_ms: Option<f64>,
    host_fetch_budget_wait_time_ms: Option<f64>,
    source_fetch_concurrency_wait_time_ms: Option<f64>,
    source_fetch_time_ms: Option<f64>,
    source_decode_time_ms: Option<f64>,
    operator_time_ms: Option<f64>,
    chunk_operator_time_ms: Option<f64>,
    final_merge_time_ms: Option<f64>,
    final_sort_time_ms: Option<f64>,
    materialization_time_ms: Option<f64>,
}

impl NativeTimingMetrics {
    fn from_metrics(metrics: Option<&MetricsSet>) -> Self {
        Self {
            parquet_read_time_ms: metric_duration_ms(metrics, "parquet_read_time"),
            source_read_time_ms: metric_duration_ms(metrics, "source_read_time"),
            source_read_queue_wait_time_ms: metric_duration_ms(metrics, "source_read_queue_wait_time"),
            source_cache_lookup_time_ms: metric_duration_ms(metrics, "source_cache_lookup_time"),
            host_fetch_budget_wait_time_ms: metric_duration_ms(metrics, "host_fetch_budget_wait_time"),
            source_fetch_concurrency_wait_time_ms: metric_duration_ms(metrics, "source_fetch_concurrency_wait_time"),
            source_fetch_time_ms: metric_duration_ms(metrics, "source_fetch_time"),
            source_decode_time_ms: metric_duration_ms(metrics, "source_decode_time"),
            operator_time_ms: metric_duration_ms(metrics, "operator_time"),
            chunk_operator_time_ms: metric_duration_ms(metrics, "chunk_operator_time"),
            final_merge_time_ms: metric_duration_ms(metrics, "final_merge_time"),
            final_sort_time_ms: metric_duration_ms(metrics, "final_sort_time"),
            materialization_time_ms: metric_duration_ms(metrics, "materialization_time"),
        }
    }
}

fn metric_duration_ms(metrics: Option<&MetricsSet>, name: &str) -> Option<f64> {
    metrics?
        .sum_by_name(name)
        .map(|value| Duration::from_nanos(value.as_usize().min(u64::MAX as usize) as u64).as_secs_f64() * 1_000.0)
}

#[derive(Debug, Clone, PartialEq)]
struct P1BottleneckSummary {
    runtime_operator_class: String,
    runtime_operator_class_ms: Option<f64>,
    peak_memory_operator_class: String,
    peak_memory_operator_class_bytes: usize,
    task_reservation_operator_class: String,
    task_reservation_operator_class_bytes: usize,
}

impl P1BottleneckSummary {
    fn from_outcome(outcome: &IterOutcome) -> Self {
        let Some(metrics) = outcome.native_metrics.as_ref() else {
            return Self {
                runtime_operator_class: "not_run".to_string(),
                runtime_operator_class_ms: None,
                peak_memory_operator_class: "not_run".to_string(),
                peak_memory_operator_class_bytes: 0,
                task_reservation_operator_class: "not_run".to_string(),
                task_reservation_operator_class_bytes: 0,
            };
        };

        let summaries = task_high_water_summaries(metrics);
        let (peak_memory_operator_class, peak_memory_operator_class_bytes) =
            dominant_task_class_by(&summaries, |summary| summary.reservation_live_peak_bytes);
        let (task_reservation_operator_class, task_reservation_operator_class_bytes) =
            dominant_task_class_by(&summaries, |summary| summary.max_task_reservation_bytes);
        let (runtime_operator_class, runtime_operator_class_ms) =
            dominant_runtime_operator_class(metrics, &outcome.timing_metrics, &summaries);
        Self {
            runtime_operator_class: runtime_operator_class.to_string(),
            runtime_operator_class_ms,
            peak_memory_operator_class: peak_memory_operator_class.to_string(),
            peak_memory_operator_class_bytes,
            task_reservation_operator_class: task_reservation_operator_class.to_string(),
            task_reservation_operator_class_bytes,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct TaskHighWaterSummary {
    task_count: usize,
    elapsed_ns: usize,
    max_task_reservation_bytes: usize,
    reservation_live_peak_bytes: usize,
    max_observed_peak_bytes: Option<usize>,
    max_retained_bytes: Option<usize>,
}

impl TaskHighWaterSummary {
    fn merge_operator_row(&mut self, row: &NativeTaskOperatorMetricRow) {
        self.task_count = self.task_count.saturating_add(row.task_count);
        self.elapsed_ns = self.elapsed_ns.saturating_add(row.elapsed_ns);
        self.reservation_live_peak_bytes = self.reservation_live_peak_bytes.max(row.reservation_live_peak_bytes);
        self.max_observed_peak_bytes = max_optional_usize(self.max_observed_peak_bytes, row.max_observed_peak_bytes);
        self.max_retained_bytes = max_optional_usize(self.max_retained_bytes, row.max_retained_bytes);
    }

    fn merge_runtime_row(&mut self, row: &NativeTaskRuntimeMetricRow, query_reservation_live_peak_bytes: usize) {
        let task_reservation_bytes = if row.compiled_task_id.starts_with("legacy_bounded.") {
            row.reservation_live_peak_bytes.min(query_reservation_live_peak_bytes)
        } else {
            row.reservation_requested_bytes.max(row.reservation_granted_bytes)
        };
        self.max_task_reservation_bytes = self.max_task_reservation_bytes.max(task_reservation_bytes);
        let reservation_live_peak_bytes = if row.compiled_task_id.starts_with("legacy_bounded.") {
            row.reservation_live_peak_bytes.min(query_reservation_live_peak_bytes)
        } else {
            row.reservation_live_peak_bytes
        };
        self.reservation_live_peak_bytes = self.reservation_live_peak_bytes.max(reservation_live_peak_bytes);
        self.max_observed_peak_bytes = max_optional_usize(self.max_observed_peak_bytes, row.max_observed_peak_bytes);
        self.max_retained_bytes = max_optional_usize(self.max_retained_bytes, row.max_retained_bytes);
    }

    fn merge_synthetic_high_water(&mut self, task_count: usize, bytes: Option<usize>) {
        self.task_count = self.task_count.saturating_add(task_count);
        if let Some(bytes) = bytes {
            self.reservation_live_peak_bytes = self.reservation_live_peak_bytes.max(bytes);
            self.max_observed_peak_bytes = max_optional_usize(self.max_observed_peak_bytes, Some(bytes));
            self.max_retained_bytes = max_optional_usize(self.max_retained_bytes, Some(bytes));
        }
    }
}

const TASK_HIGH_WATER_CLASSES: &[&str] =
    &["source", "filter_projection", "partition", "join", "aggregate", "final_merge", "final_sort", "materialization"];

fn task_high_water_summaries(
    metrics: &NativeDataFusionExecutionMetrics,
) -> BTreeMap<&'static str, TaskHighWaterSummary> {
    let mut summaries = TASK_HIGH_WATER_CLASSES
        .iter()
        .copied()
        .map(|class| (class, TaskHighWaterSummary::default()))
        .collect::<BTreeMap<_, _>>();

    for row in &metrics.task_operator_rows {
        if let Some(class) = task_operator_class(&row.operator_kind) {
            summaries.entry(class).or_default().merge_operator_row(row);
        }
    }
    for row in &metrics.task_runtime_rows {
        if let Some(class) = task_operator_class(&row.operator_kind) {
            summaries.entry(class).or_default().merge_runtime_row(row, metrics.reservation_live_peak_bytes);
        }
    }

    let partition_task_count = metrics
        .join_strategy_rows
        .iter()
        .map(|row| {
            row.build_partition_batch_count.unwrap_or(0).saturating_add(row.probe_partition_batch_count.unwrap_or(0))
        })
        .sum::<usize>();
    let partition_bytes = metrics
        .join_strategy_rows
        .iter()
        .flat_map(|row| [row.build_partitioned_bytes, row.probe_partitioned_bytes])
        .flatten()
        .max();
    summaries.entry("partition").or_default().merge_synthetic_high_water(partition_task_count, partition_bytes);

    summaries.entry("final_merge").or_default().merge_synthetic_high_water(
        metrics.final_merge_attempts,
        (metrics.partial_aggregate_retained_bytes > 0).then_some(metrics.partial_aggregate_retained_bytes),
    );

    summaries
}

fn task_operator_class(operator_kind: &str) -> Option<&'static str> {
    match operator_kind {
        "source_read" | "gpu_resident_source" => Some("source"),
        "unary" => Some("filter_projection"),
        "partition" | "hash_partition" => Some("partition"),
        "join_build" | "join_probe" | "join_materialize" => Some("join"),
        "aggregate" => Some("aggregate"),
        "final_merge" => Some("final_merge"),
        "final_sort" | "limit" => Some("final_sort"),
        "sink_materialize" | "materialization" => Some("materialization"),
        _ => None,
    }
}

fn dominant_task_class_by(
    summaries: &BTreeMap<&'static str, TaskHighWaterSummary>,
    value: impl Fn(TaskHighWaterSummary) -> usize,
) -> (&'static str, usize) {
    TASK_HIGH_WATER_CLASSES
        .iter()
        .copied()
        .map(|class| {
            let summary = summaries.get(class).copied().unwrap_or_default();
            (class, value(summary))
        })
        .max_by(|left, right| left.1.cmp(&right.1).then_with(|| right.0.cmp(left.0)))
        .filter(|(_, bytes)| *bytes > 0)
        .unwrap_or(("none", 0))
}

fn dominant_runtime_operator_class(
    metrics: &NativeDataFusionExecutionMetrics,
    timing: &NativeTimingMetrics,
    summaries: &BTreeMap<&'static str, TaskHighWaterSummary>,
) -> (&'static str, Option<f64>) {
    let operator_class = dominant_operator_runtime_class(metrics, summaries);
    if let Some((class, summary)) = summaries
        .iter()
        .filter(|(_, summary)| summary.elapsed_ns > 0)
        .max_by(|left, right| left.1.elapsed_ns.cmp(&right.1.elapsed_ns).then_with(|| right.0.cmp(left.0)))
    {
        return (class, Some(summary.elapsed_ns as f64 / 1_000_000.0));
    }

    [
        ("source", timing.source_read_time_ms),
        (operator_class, timing.operator_time_ms),
        ("final_merge", timing.final_merge_time_ms),
        ("final_sort", timing.final_sort_time_ms),
        ("materialization", timing.materialization_time_ms),
    ]
    .into_iter()
    .filter_map(|(class, value)| value.map(|value| (class, value)))
    .max_by(|left, right| left.1.partial_cmp(&right.1).unwrap_or(std::cmp::Ordering::Equal))
    .map(|(class, value)| (class, Some(value)))
    .unwrap_or(("not_run", None))
}

fn dominant_operator_runtime_class(
    metrics: &NativeDataFusionExecutionMetrics,
    summaries: &BTreeMap<&'static str, TaskHighWaterSummary>,
) -> &'static str {
    let join_tasks = summaries.get("join").copied().unwrap_or_default().task_count;
    let aggregate_tasks = summaries.get("aggregate").copied().unwrap_or_default().task_count;
    match (join_tasks > 0 || !metrics.join_strategy_rows.is_empty(), aggregate_tasks > 0) {
        (true, true) => "join_aggregate",
        (true, false) => "join",
        (false, true) => "aggregate",
        (false, false) => "native_operator",
    }
}

fn partitioned_join_count_text(metrics: &NativeDataFusionExecutionMetrics) -> String {
    partitioned_join_count(metrics).to_string()
}

fn partitioned_join_count(metrics: &NativeDataFusionExecutionMetrics) -> usize {
    metrics
        .join_strategy_rows
        .iter()
        .filter(|row| row.join_strategy.starts_with("partitioned_") || row.partitioned_hash_join_count > 0)
        .count()
}

fn join_partitioning_status(metrics: &NativeDataFusionExecutionMetrics) -> String {
    let join_count = metrics.join_strategy_rows.len();
    let partitioned_count = partitioned_join_count(metrics);
    match (join_count, partitioned_count) {
        (0, _) => "no_joins",
        (_, 0) => "not_partitioned",
        (joins, partitioned) if joins == partitioned => "partitioned",
        _ => "mixed",
    }
    .to_string()
}

fn format_join_strategy_rows(metrics: &NativeDataFusionExecutionMetrics) -> String {
    if metrics.join_strategy_rows.is_empty() {
        return "none".to_string();
    }
    let mut rows = metrics.join_strategy_rows.iter().collect::<Vec<_>>();
    rows.sort_by_key(|row| row.to_stable_kv_row());
    rows.iter().map(|row| sanitize_nested(&row.to_stable_kv_row())).collect::<Vec<_>>().join("|")
}

fn format_aggregate_strategy_rows(metrics: &NativeDataFusionExecutionMetrics) -> String {
    if metrics.aggregate_strategy_rows.is_empty() {
        return "none".to_string();
    }
    let mut rows = metrics.aggregate_strategy_rows.iter().collect::<Vec<_>>();
    rows.sort_by_key(|row| row.to_stable_kv_row());
    rows.iter().map(|row| sanitize_nested(&row.to_stable_kv_row())).collect::<Vec<_>>().join("|")
}

fn format_dynamic_filter_runtime_rows(metrics: &NativeDataFusionExecutionMetrics) -> String {
    if metrics.dynamic_filter_runtime_rows.is_empty() {
        return "none".to_string();
    }
    let mut rows = metrics.dynamic_filter_runtime_rows.iter().collect::<Vec<_>>();
    rows.sort_by_key(|row| row.to_stable_kv_row());
    rows.iter().map(|row| sanitize_nested(&row.to_stable_kv_row())).collect::<Vec<_>>().join("|")
}

fn format_task_operator_rows(metrics: &NativeDataFusionExecutionMetrics) -> String {
    if metrics.task_operator_rows.is_empty() {
        return "none".to_string();
    }
    let mut rows = metrics.task_operator_rows.iter().collect::<Vec<_>>();
    rows.sort_by_key(|row| row.to_stable_kv_row());
    rows.iter().map(|row| sanitize_nested(&row.to_stable_kv_row())).collect::<Vec<_>>().join("|")
}

fn format_task_runtime_rows(metrics: &NativeDataFusionExecutionMetrics) -> String {
    if metrics.task_runtime_rows.is_empty() {
        return "none".to_string();
    }
    let mut rows = metrics.task_runtime_rows.iter().collect::<Vec<_>>();
    rows.sort_by_key(|row| row.to_stable_kv_row());
    rows.iter().map(|row| sanitize_nested(&row.to_stable_kv_row())).collect::<Vec<_>>().join("|")
}

fn metric_field(
    metrics: Option<&NativeDataFusionExecutionMetrics>,
    value: impl Fn(&NativeDataFusionExecutionMetrics) -> usize,
) -> String {
    metrics.map_or_else(|| "not_run".to_string(), |metrics| value(metrics).to_string())
}

fn optional_f64_text(value: Option<f64>) -> String {
    value.map_or_else(|| "not_run".to_string(), |value| format!("{value:.3}"))
}

fn sanitize_nested(s: &str) -> String {
    s.replace(['\t', '\n', '\r'], " ").replace('|', "/")
}

fn max_optional_usize(left: Option<usize>, right: Option<usize>) -> Option<usize> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.max(right)),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn p1_tsv_header_matches_row_field_count_for_cpu_rows() {
        let query = TPCDS_QUERIES.iter().find(|query| query.id == "q85").unwrap();
        let outcome = IterOutcome {
            wall: Some(Duration::from_millis(1)),
            outcome: "succeeded",
            planning_outcome: "cpu_only".to_string(),
            output_batches: Some(1),
            output_rows: Some(2),
            output_bytes: Some(3),
            reservation_live_peak_bytes: None,
            reservation_denied_count: None,
            retry_count: None,
            native_runtime_started: false,
            native_metrics: None,
            timing_metrics: NativeTimingMetrics::default(),
            failure: None,
        };
        let p1 = P1BottleneckSummary::from_outcome(&outcome);
        assert_eq!(p1.runtime_operator_class, "not_run");
        assert_eq!(TSV_HEADER.split('\t').count(), row_fields(query, 0, false, &outcome, Engine::CpuDataFusion).len());
    }

    #[test]
    fn all_query_selector_runs_full_sweep_including_q39() {
        let queries = bench_queries_from_csv("all").unwrap();
        assert_eq!(queries.len(), TPCDS_QUERIES.len());
        assert!(queries.iter().any(|query| query.id == "q39"));
    }

    #[test]
    fn explicit_query_selector_rejects_unknown_ids() {
        let error = match bench_queries_from_csv("q39,no_such_query") {
            Ok(_) => panic!("unknown query id should be rejected"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("unknown TPC-DS query id: no_such_query"));
    }

    #[test]
    fn statistical_aggregate_contract_columns_are_visible_per_engine() {
        let query = TPCDS_QUERIES.iter().find(|query| query.id == "q39").unwrap();
        let outcome = IterOutcome {
            wall: Some(Duration::from_millis(1)),
            outcome: "succeeded",
            planning_outcome: "rewritten=1 rejected=0".to_string(),
            output_batches: Some(1),
            output_rows: Some(2),
            output_bytes: Some(3),
            reservation_live_peak_bytes: Some(4),
            reservation_denied_count: Some(0),
            retry_count: Some(0),
            native_runtime_started: true,
            native_metrics: None,
            timing_metrics: NativeTimingMetrics::default(),
            failure: None,
        };
        let header = TSV_HEADER.split('\t').collect::<Vec<_>>();
        let mode_index = header.iter().position(|field| *field == "statistical_aggregate_execution_mode").unwrap();
        let tolerance_index =
            header.iter().position(|field| *field == "statistical_aggregate_tolerance_contract").unwrap();

        for engine in [Engine::NativeGpu, Engine::HybridCpuPreferred] {
            let fields = row_fields(query, 0, false, &outcome, engine);
            assert_eq!(fields[mode_index], "gpu_native_tolerant");
            assert_eq!(fields[tolerance_index], "float_tolerant");
        }

        let cpu_fields = row_fields(query, 0, false, &outcome, Engine::CpuDataFusion);
        assert_eq!(cpu_fields[mode_index], "not_applicable");
        assert_eq!(cpu_fields[tolerance_index], "not_applicable");
    }

    #[test]
    fn native_no_runtime_not_run_rows_suppress_wall_time() {
        let query = TPCDS_QUERIES.iter().find(|query| query.id == "q67").unwrap();
        let outcome = IterOutcome {
            wall: None,
            outcome: "native_rejected_not_run",
            planning_outcome: "rewritten=0 rejected=1 reasons=native_bounded_join_unsupported".to_string(),
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

        let fields = row_fields(query, 0, false, &outcome, Engine::NativeGpu);

        let header = TSV_HEADER.split('\t').collect::<Vec<_>>();
        let wall_ms_index = header.iter().position(|field| *field == "wall_ms").unwrap();
        let outcome_index = header.iter().position(|field| *field == "outcome").unwrap();
        let native_runtime_started_index = header.iter().position(|field| *field == "native_runtime_started").unwrap();
        assert_eq!(fields[wall_ms_index], "");
        assert_eq!(fields[outcome_index], "native_rejected_not_run");
        assert_eq!(fields[native_runtime_started_index], "false");
    }

    #[test]
    fn p1_task_operator_classes_cover_native_runtime_rows() {
        assert_eq!(task_operator_class("source_read"), Some("source"));
        assert_eq!(task_operator_class("join_probe"), Some("join"));
        assert_eq!(task_operator_class("aggregate"), Some("aggregate"));
        assert_eq!(task_operator_class("sink_materialize"), Some("materialization"));
    }

    #[test]
    fn hybrid_cpu_preferred_engine_has_stable_label() {
        assert_eq!(Engine::HybridCpuPreferred.as_str(), "hybrid_cpu_preferred");
    }

    #[test]
    fn cpu_preferred_reason_codes_are_classified() {
        assert!(cpu_preferred_reason_code("native_small_selective_topk_cpu_preferred"));
        assert!(cpu_preferred_reason_code("native_small_selective_residual_join_cpu_preferred"));
        assert!(!cpu_preferred_reason_code("native_bounded_join_unsupported"));
    }
}

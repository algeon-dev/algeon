#[cfg(test)]
use super::config::bench_queries_from_csv;
use super::*;

pub(super) const TSV_HEADER: &str = concat!(
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
const TSV_COLUMNS: &[&str] = &[
    "engine",
    "query_id",
    "iteration",
    "warmup",
    "wall_ms",
    "outcome",
    "planning_outcome",
    "output_batches",
    "output_rows",
    "output_bytes",
    "reservation_live_peak_bytes",
    "reservation_denied_count",
    "retry_count",
    "native_runtime_started",
    "failure",
    "parquet_read_time_ms",
    "source_read_time_ms",
    "source_read_queue_wait_time_ms",
    "source_cache_lookup_time_ms",
    "host_fetch_budget_wait_time_ms",
    "source_fetch_concurrency_wait_time_ms",
    "source_fetch_time_ms",
    "source_decode_time_ms",
    "operator_time_ms",
    "chunk_operator_time_ms",
    "final_merge_time_ms",
    "final_sort_time_ms",
    "materialization_time_ms",
    "aggregate_libcudf_runtime_probe_count",
    "aggregate_libcudf_runtime_probe_elapsed_ns",
    "aggregate_libcudf_runtime_probe_input_rows",
    "aggregate_exact_cardinality_probe_count",
    "aggregate_exact_cardinality_probe_elapsed_ns",
    "aggregate_exact_cardinality_probe_input_rows",
    "aggregate_prepared_sort_groupby_elapsed_ns",
    "aggregate_encoded_key_partial_elapsed_ns",
    "runtime_operator_class",
    "runtime_operator_class_ms",
    "peak_memory_operator_class",
    "peak_memory_operator_class_bytes",
    "task_reservation_operator_class",
    "task_reservation_operator_class_bytes",
    "source_chunks_planned",
    "source_chunks_attempted",
    "source_chunks_committed",
    "source_chunks_retried",
    "source_read_prefetches_started",
    "source_read_prefetches_completed",
    "source_read_prefetches_skipped",
    "source_read_prefetches_abandoned",
    "source_cache_hits",
    "source_cache_misses",
    "source_cache_inserts",
    "source_cache_insert_skipped",
    "source_cache_retained_bytes",
    "source_fetch_bytes",
    "source_fetch_count",
    "row_groups_planned",
    "row_groups_committed",
    "hash_join_build_count",
    "filtered_join_build_count",
    "one_shot_join_count",
    "partitioned_join_count",
    "join_partitioning_status",
    "dynamic_filter_produced_count",
    "dynamic_filter_skipped_count",
    "dynamic_filter_applied_count",
    "dynamic_filter_row_groups_pruned",
    "dynamic_filter_source_bytes_pruned",
    "dynamic_filter_reader_predicate_rows_before",
    "dynamic_filter_reader_predicate_rows_after",
    "dynamic_filter_reader_predicate_rows_filtered",
    "dynamic_filter_post_decode_rows_before",
    "dynamic_filter_post_decode_rows_after",
    "dynamic_filter_post_decode_rows_filtered",
    "dynamic_filter_source_post_decode_rows_filtered",
    "dynamic_filter_filter_post_decode_rows_filtered",
    "dynamic_filter_join_probe_post_decode_rows_filtered",
    "dynamic_filter_post_decode_reduction_stage",
    "dynamic_filter_measured_reduction_status",
    "dynamic_filter_work_avoidance_status",
    "dynamic_filter_runtime_rows",
    "aggregate_strategy_rows",
    "join_strategy_rows",
    "task_operator_rows",
    "task_runtime_rows",
    "statistical_aggregate_execution_mode",
    "statistical_aggregate_tolerance_contract",
];

pub(super) fn print_metadata(cfg: &BenchConfig) {
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

pub(super) fn print_progress(cfg: &BenchConfig, query: &TpcdsQuery, iteration: usize, warmup: bool, stage: &str) {
    eprintln!(
        "[tpcds_three_way_bench] stage={} engine={} query_id={} iteration={} warmup={}",
        stage,
        cfg.engine.as_str(),
        query.id,
        iteration,
        warmup
    );
    let _ = io::stderr().flush();
}

pub(super) fn print_row(query: &TpcdsQuery, iteration: usize, warmup: bool, outcome: &IterOutcome, engine: Engine) {
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
    let fields = vec![
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
    ];
    debug_assert_eq!(fields.len(), TSV_COLUMNS.len());
    fields
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
pub(super) struct NativeTimingMetrics {
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
    pub(super) fn from_metrics(metrics: Option<&MetricsSet>) -> Self {
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
    metrics.join_strategy_rows.iter().map(|row| row.partitioned_hash_join_count).sum()
}

fn partitioned_join_site_count(metrics: &NativeDataFusionExecutionMetrics) -> usize {
    metrics.join_strategy_rows.iter().filter(|row| row.partitioned_hash_join_count > 0).count()
}

fn join_partitioning_status(metrics: &NativeDataFusionExecutionMetrics) -> String {
    let join_count = metrics.join_strategy_rows.len();
    let partitioned_sites = partitioned_join_site_count(metrics);
    match (join_count, partitioned_sites) {
        (0, _) => "no_joins",
        (_, 0) => "not_partitioned",
        (joins, partitioned) if joins == partitioned => "partitioned",
        _ => "mixed",
    }
    .to_string()
}

fn format_join_strategy_rows(metrics: &NativeDataFusionExecutionMetrics) -> String {
    datafusion_nexus::native_report::format_native_join_strategy_metric_rows(&metrics.join_strategy_rows)
}

fn format_aggregate_strategy_rows(metrics: &NativeDataFusionExecutionMetrics) -> String {
    datafusion_nexus::native_report::format_native_aggregate_strategy_metric_rows(&metrics.aggregate_strategy_rows)
}

fn format_dynamic_filter_runtime_rows(metrics: &NativeDataFusionExecutionMetrics) -> String {
    datafusion_nexus::native_report::format_native_dynamic_filter_runtime_metric_rows(
        &metrics.dynamic_filter_runtime_rows,
    )
}

fn format_task_operator_rows(metrics: &NativeDataFusionExecutionMetrics) -> String {
    datafusion_nexus::native_report::format_native_task_operator_metric_rows(&metrics.task_operator_rows)
}

fn format_task_runtime_rows(metrics: &NativeDataFusionExecutionMetrics) -> String {
    datafusion_nexus::native_report::format_native_task_runtime_metric_rows(&metrics.task_runtime_rows)
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
        assert_eq!(TSV_COLUMNS.join("\t"), TSV_HEADER);
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
        assert!(cpu_preferred_reason_code("native_small_selective_full_sort_cpu_preferred"));
        assert!(!cpu_preferred_reason_code("native_bounded_join_unsupported"));
    }
}

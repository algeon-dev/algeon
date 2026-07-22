use nexus_query_engine::harness::max_option as max_optional_metric;

use super::*;

#[derive(Debug, Clone)]
pub(super) struct QueryMeasurement {
    pub(super) output_batches: usize,
    pub(super) output_rows: usize,
    pub(super) output_bytes: usize,
    pub(super) output_fingerprint: String,
}

#[derive(Debug, Clone)]
pub(super) struct TimedQueryRun {
    pub(super) samples: Vec<Duration>,
    pub(super) final_measurement: QueryMeasurement,
}

#[derive(Clone)]
pub(super) struct PlannedCpuQuery {
    _ctx: SessionContext,
    plan: Arc<dyn ExecutionPlan>,
}

#[derive(Debug, Clone)]
pub(super) struct CpuRunFailure {
    pub(super) runtime_started: bool,
    pub(super) message: String,
    pub(super) details: FailureDetails,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct FailureDetails {
    pub(super) code: Option<String>,
    pub(super) kind: Option<String>,
    pub(super) status: Option<String>,
    pub(super) operation: Option<String>,
    pub(super) facts: BTreeMap<String, String>,
}

impl FailureDetails {
    pub(super) fn from_datafusion_error(error: &DataFusionError) -> Self {
        error.find_nexus_error().map(Self::from_adapter_error).unwrap_or_default()
    }

    fn from_adapter_error(error: &AdapterError) -> Self {
        Self {
            code: Some(error.code().as_str().to_string()),
            kind: Some(error.kind().as_str().to_string()),
            status: Some(error.status().as_str().to_string()),
            operation: Some(error.operation().to_string()),
            facts: error.facts().iter().map(|(key, value)| (key.to_string(), value.to_string())).collect(),
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct TimedNativeQueryRun {
    pub(super) samples: Vec<Duration>,
    pub(super) final_measurement: QueryMeasurement,
    pub(super) metrics: NativeDataFusionExecutionMetrics,
    pub(super) timing_metrics: NativeTpchTimingMetrics,
}

#[derive(Debug, Clone)]
pub(super) struct NativeRunFailure {
    pub(super) runtime_started: bool,
    pub(super) message: String,
    pub(super) details: FailureDetails,
}

#[derive(Debug, Clone)]
pub(super) struct PlannedNativeQuery {
    pub(super) plan: Arc<dyn ExecutionPlan>,
    pub(super) planning_row: NativeDataFusionPlanningRow,
}

pub(super) async fn plan_native_query(
    metadata: &TpchDatasetMetadata,
    query: &TpchQuery,
    config: &NativeTpchBenchmarkConfig,
) -> DataFusionResult<PlannedNativeQuery> {
    plan_native_query_with_resources(
        metadata,
        query,
        config,
        &NativeTpchRuntimeResources { runtime: None, source_cache: None },
    )
    .await
}

async fn plan_native_query_with_resources(
    metadata: &TpchDatasetMetadata,
    query: &TpchQuery,
    config: &NativeTpchBenchmarkConfig,
    resources: &NativeTpchRuntimeResources,
) -> DataFusionResult<PlannedNativeQuery> {
    let optimizer_config = native_optimizer_config(config, resources);
    let native_rule = Arc::new(NexusNativePhysicalOptimizerRule::new(optimizer_config.clone()));
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(native_session_config(config))
        .with_nexus_native_rule(Arc::clone(&native_rule))
        .build();
    let ctx = SessionContext::new_with_state(state);
    dataset::register_tpch_tables(&ctx, metadata).await?;
    let df = ctx.sql(query.sql).await?;
    let plan = df.create_physical_plan().await?;
    let planning_report =
        NativeDataFusionPlanningReport::from_retained_rule(query.id, Arc::clone(&plan), native_rule.as_ref());
    let planning_row = select_benchmark_planning_row(query.id, planning_report.rows)?;
    Ok(PlannedNativeQuery { plan, planning_row })
}

pub(super) fn native_optimizer_config(
    config: &NativeTpchBenchmarkConfig,
    resources: &NativeTpchRuntimeResources,
) -> NexusNativeOptimizerConfig {
    NexusNativeOptimizerConfig::new(config.native_fallback_policy)
        .with_native_execution_config(config.native_execution_config_with_resources(resources))
}

fn native_session_config(config: &NativeTpchBenchmarkConfig) -> SessionConfig {
    let mut session_config = tpch_report_session_config();
    if let Some(pushdown_filters) = config.parquet_pushdown_filters {
        session_config.options_mut().execution.parquet.pushdown_filters = pushdown_filters;
    }
    session_config
}

pub(super) fn select_benchmark_planning_row(
    query_id: &str,
    rows: Vec<NativeDataFusionPlanningRow>,
) -> DataFusionResult<NativeDataFusionPlanningRow> {
    match rows.len() {
        0 => Err(plan_error(format!("native planning report was empty for {query_id}"))),
        1 => {
            let row = rows.into_iter().next().expect("single planning row must exist");
            if row.query_id != query_id {
                return Err(plan_error(format!(
                    "native TPC-H benchmark planning row query_id mismatch; requested {query_id}, observed {}",
                    row.query_id
                )));
            }
            Ok(row)
        },
        count => {
            let mut whole_native_rows = Vec::new();
            for row in rows {
                if row.query_id != query_id {
                    return Err(plan_error(format!(
                        "native TPC-H benchmark planning row query_id mismatch; requested {query_id}, observed {}",
                        row.query_id
                    )));
                }
                if row.outcome == NativeDataFusionPlanningOutcome::AcceptedReplaced
                    && row.post_native_execution_path == NativeDataFusionPostNativeExecutionPath::Native
                    && row.native_plan_fragment_count == 1
                    && row.host_datafusion_boundary_count == 0
                    && !row.rejected_before_execution
                {
                    whole_native_rows.push(row);
                }
            }
            match whole_native_rows.len() {
                1 => Ok(whole_native_rows.pop().expect("one whole-native planning row should be present")),
                accepted_count => Err(plan_error(format!(
                    "native TPC-H benchmark requires exactly one whole-native accepted planning row when a query report has multiple rows; {query_id} produced {count} rows and {accepted_count} whole-native accepted rows"
                ))),
            }
        },
    }
}

pub(super) fn validate_decision_grade_native_planning_row(
    query_id: &str,
    row: &NativeDataFusionPlanningRow,
) -> DataFusionResult<()> {
    if row.outcome == NativeDataFusionPlanningOutcome::AcceptedReplaced
        && row.post_native_execution_path != NativeDataFusionPostNativeExecutionPath::Native
    {
        return Err(plan_error(format!(
            "native TPC-H benchmark requires post_native_execution_path=native for accepted query {query_id}; observed {}",
            row.post_native_execution_path.as_str()
        )));
    }
    if row.outcome == NativeDataFusionPlanningOutcome::AcceptedReplaced {
        if row.rejected_before_execution {
            return Err(plan_error(format!(
                "native TPC-H benchmark accepted planning row for {query_id} must not set rejected_before_execution=true"
            )));
        }
        if row.reason_code.as_deref().is_some_and(|reason_code| !reason_code.is_empty()) {
            return Err(plan_error(format!(
                "native TPC-H benchmark accepted planning row for {query_id} must not include a rejection reason_code"
            )));
        }
        if row.unsupported_category.as_deref().is_some_and(|unsupported_category| !unsupported_category.is_empty()) {
            return Err(plan_error(format!(
                "native TPC-H benchmark accepted planning row for {query_id} must not include an unsupported_category"
            )));
        }
    }
    Ok(())
}

pub(super) fn validate_decision_grade_native_rejection_row(
    query_id: &str,
    row: &NativeDataFusionPlanningRow,
) -> DataFusionResult<()> {
    if row.outcome == NativeDataFusionPlanningOutcome::AcceptedReplaced {
        return Ok(());
    }
    if !row.rejected_before_execution {
        return Err(plan_error(format!(
            "native TPC-H benchmark rejected planning row for {query_id} must set rejected_before_execution=true"
        )));
    }
    if row.reason_code.as_deref().is_none_or(str::is_empty) {
        return Err(plan_error(format!(
            "native TPC-H benchmark rejected planning row for {query_id} must include a stable reason_code"
        )));
    }
    if row.unsupported_category.as_deref().is_none_or(str::is_empty) {
        return Err(plan_error(format!(
            "native TPC-H benchmark rejected planning row for {query_id} must include a stable unsupported_category"
        )));
    }
    if row.post_native_execution_path == NativeDataFusionPostNativeExecutionPath::Native {
        return Err(plan_error(format!(
            "native TPC-H benchmark rejected planning row for {query_id} must not report post_native_execution_path=native"
        )));
    }
    Ok(())
}

fn validate_replanned_native_row(
    query_id: &str,
    expected: &NativeDataFusionPlanningRow,
    actual: &NativeDataFusionPlanningRow,
) -> DataFusionResult<()> {
    validate_decision_grade_native_planning_row(query_id, actual)?;
    if actual != expected {
        return Err(plan_error(format!(
            "native TPC-H benchmark planning row changed during execution for {query_id}; refusing to merge metrics with a different planning contract"
        )));
    }
    Ok(())
}

pub(super) async fn run_cpu_query_iterations(
    metadata: &TpchDatasetMetadata,
    query: &TpchQuery,
    config: &NativeTpchBenchmarkConfig,
) -> Result<TimedQueryRun, CpuRunFailure> {
    for _ in 0..config.warmup_iterations {
        let planned = plan_cpu_query(metadata, query).await?;
        let _ = collect_cpu_plan_once(planned).await?;
    }

    let mut samples = Vec::with_capacity(config.iterations);
    let mut final_measurement = None;
    for _ in 0..config.iterations {
        let planned = plan_cpu_query(metadata, query).await?;
        let schema = planned.plan.schema();
        let started = Instant::now();
        let batches = collect_cpu_plan_once(planned).await?;
        samples.push(started.elapsed());
        final_measurement = Some(measure_output(&schema, &batches).map_err(cpu_started)?);
    }
    Ok(TimedQueryRun {
        samples,
        final_measurement: final_measurement.expect("validated benchmark config requires iterations"),
    })
}

pub(super) async fn run_native_query_iterations(
    metadata: &TpchDatasetMetadata,
    query: &TpchQuery,
    config: &NativeTpchBenchmarkConfig,
    expected_planning_row: &NativeDataFusionPlanningRow,
    resources: &NativeTpchRuntimeResources,
) -> Result<TimedNativeQueryRun, NativeRunFailure> {
    for _ in 0..config.warmup_iterations {
        let planned =
            plan_native_query_with_resources(metadata, query, config, resources).await.map_err(native_not_started)?;
        if planned.planning_row.outcome == NativeDataFusionPlanningOutcome::AcceptedReplaced {
            validate_replanned_native_row(query.id, expected_planning_row, &planned.planning_row)
                .map_err(native_not_started)?;
            let _ = collect_native_plan_once(planned.plan).await.map_err(native_started)?;
        } else {
            return Err(NativeRunFailure {
                runtime_started: false,
                message: format!("{} stopped being a native accepted query during warmup", query.id),
                details: FailureDetails::default(),
            });
        }
    }

    let mut samples = Vec::with_capacity(config.iterations);
    let mut final_measurement = None;
    let mut measured_metrics = Vec::with_capacity(config.iterations);
    for _ in 0..config.iterations {
        let planned =
            plan_native_query_with_resources(metadata, query, config, resources).await.map_err(native_not_started)?;
        if planned.planning_row.outcome != NativeDataFusionPlanningOutcome::AcceptedReplaced {
            return Err(NativeRunFailure {
                runtime_started: false,
                message: format!("{} stopped being a native accepted query during execution", query.id),
                details: FailureDetails::default(),
            });
        }
        validate_replanned_native_row(query.id, expected_planning_row, &planned.planning_row)
            .map_err(native_not_started)?;
        let schema = planned.plan.schema();
        let started = Instant::now();
        let (batches, metrics) = collect_native_plan_once(Arc::clone(&planned.plan)).await.map_err(native_started)?;
        samples.push(started.elapsed());
        final_measurement = Some(measure_output(&schema, &batches).map_err(native_started)?);
        measured_metrics.push(metrics);
    }
    let metrics = aggregate_native_execution_metrics(&measured_metrics);
    let timing_metrics = NativeTpchTimingMetrics::mean_from_metrics(&measured_metrics);
    Ok(TimedNativeQueryRun {
        samples,
        final_measurement: final_measurement.expect("validated benchmark config requires iterations"),
        metrics,
        timing_metrics,
    })
}

fn aggregate_native_execution_metrics(metrics: &[MetricsSet]) -> NativeDataFusionExecutionMetrics {
    let mut aggregate = NativeDataFusionExecutionMetrics::default();
    for metrics in metrics {
        let current = NativeDataFusionExecutionMetrics::from_metrics(metrics);
        aggregate.gpu_resident_input_handoffs =
            aggregate.gpu_resident_input_handoffs.saturating_add(current.gpu_resident_input_handoffs);
        aggregate.host_to_device_uploads =
            aggregate.host_to_device_uploads.saturating_add(current.host_to_device_uploads);
        aggregate.host_to_device_uploaded_rows =
            aggregate.host_to_device_uploaded_rows.saturating_add(current.host_to_device_uploaded_rows);
        aggregate.host_to_device_uploaded_bytes =
            aggregate.host_to_device_uploaded_bytes.saturating_add(current.host_to_device_uploaded_bytes);
        aggregate.device_to_host_materializations =
            aggregate.device_to_host_materializations.saturating_add(current.device_to_host_materializations);
        aggregate.device_to_host_materialized_rows =
            aggregate.device_to_host_materialized_rows.saturating_add(current.device_to_host_materialized_rows);
        aggregate.device_to_host_materialized_bytes =
            aggregate.device_to_host_materialized_bytes.saturating_add(current.device_to_host_materialized_bytes);
        aggregate.device_to_device_copies =
            aggregate.device_to_device_copies.saturating_add(current.device_to_device_copies);
        aggregate.device_to_device_copied_bytes =
            aggregate.device_to_device_copied_bytes.saturating_add(current.device_to_device_copied_bytes);
        aggregate.parquet_files = aggregate.parquet_files.saturating_add(current.parquet_files);
        aggregate.parquet_bytes_known = aggregate.parquet_bytes_known.saturating_add(current.parquet_bytes_known);
        aggregate.row_groups_selected = aggregate.row_groups_selected.saturating_add(current.row_groups_selected);
        aggregate.read_columns = aggregate.read_columns.saturating_add(current.read_columns);
        aggregate.resident_dataframe_peak_bytes =
            max_optional_metric(aggregate.resident_dataframe_peak_bytes, current.resident_dataframe_peak_bytes);
        aggregate.allocator_high_water_bytes =
            max_optional_metric(aggregate.allocator_high_water_bytes, current.allocator_high_water_bytes);
        aggregate.reservation_requested_bytes =
            aggregate.reservation_requested_bytes.saturating_add(current.reservation_requested_bytes);
        aggregate.reservation_granted_bytes =
            aggregate.reservation_granted_bytes.saturating_add(current.reservation_granted_bytes);
        aggregate.reservation_live_peak_bytes =
            aggregate.reservation_live_peak_bytes.max(current.reservation_live_peak_bytes);
        aggregate.reservation_denied_count =
            aggregate.reservation_denied_count.saturating_add(current.reservation_denied_count);
        aggregate.max_task_reservation_bytes =
            aggregate.max_task_reservation_bytes.max(current.max_task_reservation_bytes);
        aggregate.max_task_observed_peak_bytes =
            max_optional_metric(aggregate.max_task_observed_peak_bytes, current.max_task_observed_peak_bytes);
        aggregate.max_task_retained_bytes =
            max_optional_metric(aggregate.max_task_retained_bytes, current.max_task_retained_bytes);
        aggregate.merge_graph_reservation_rows(current.graph_reservation_rows);
        aggregate.merge_graph_runtime_rows(current.graph_runtime_rows);
        aggregate.merge_task_operator_rows(current.task_operator_rows);
        aggregate.merge_task_runtime_rows(current.task_runtime_rows);
        aggregate.source_chunks_planned = aggregate.source_chunks_planned.saturating_add(current.source_chunks_planned);
        aggregate.source_chunks_attempted =
            aggregate.source_chunks_attempted.saturating_add(current.source_chunks_attempted);
        aggregate.source_chunks_committed =
            aggregate.source_chunks_committed.saturating_add(current.source_chunks_committed);
        aggregate.source_chunks_retried = aggregate.source_chunks_retried.saturating_add(current.source_chunks_retried);
        aggregate.source_read_prefetches_started =
            aggregate.source_read_prefetches_started.saturating_add(current.source_read_prefetches_started);
        aggregate.source_read_prefetches_completed =
            aggregate.source_read_prefetches_completed.saturating_add(current.source_read_prefetches_completed);
        aggregate.source_read_prefetches_skipped =
            aggregate.source_read_prefetches_skipped.saturating_add(current.source_read_prefetches_skipped);
        aggregate.source_read_prefetches_abandoned =
            aggregate.source_read_prefetches_abandoned.saturating_add(current.source_read_prefetches_abandoned);
        aggregate.source_cache_hits = aggregate.source_cache_hits.saturating_add(current.source_cache_hits);
        aggregate.source_cache_misses = aggregate.source_cache_misses.saturating_add(current.source_cache_misses);
        aggregate.source_cache_inserts = aggregate.source_cache_inserts.saturating_add(current.source_cache_inserts);
        aggregate.source_cache_insert_skipped =
            aggregate.source_cache_insert_skipped.saturating_add(current.source_cache_insert_skipped);
        aggregate.source_cache_retained_bytes =
            aggregate.source_cache_retained_bytes.max(current.source_cache_retained_bytes);
        aggregate.row_groups_planned = aggregate.row_groups_planned.saturating_add(current.row_groups_planned);
        aggregate.row_groups_committed = aggregate.row_groups_committed.saturating_add(current.row_groups_committed);
        aggregate.partial_aggregate_count =
            aggregate.partial_aggregate_count.saturating_add(current.partial_aggregate_count);
        aggregate.partial_aggregate_input_rows =
            aggregate.partial_aggregate_input_rows.saturating_add(current.partial_aggregate_input_rows);
        aggregate.partial_aggregate_rows =
            aggregate.partial_aggregate_rows.saturating_add(current.partial_aggregate_rows);
        aggregate.partial_aggregate_retained_bytes =
            aggregate.partial_aggregate_retained_bytes.max(current.partial_aggregate_retained_bytes);
        aggregate.final_merge_attempts = aggregate.final_merge_attempts.saturating_add(current.final_merge_attempts);
        aggregate.final_merge_levels = aggregate.final_merge_levels.saturating_add(current.final_merge_levels);
        aggregate.final_merge_max_fan_in = aggregate.final_merge_max_fan_in.max(current.final_merge_max_fan_in);
        aggregate.retry_count = aggregate.retry_count.saturating_add(current.retry_count);
        aggregate.retry_resource_exhausted_count =
            aggregate.retry_resource_exhausted_count.saturating_add(current.retry_resource_exhausted_count);
        aggregate.retry_reservation_failure_count =
            aggregate.retry_reservation_failure_count.saturating_add(current.retry_reservation_failure_count);
        aggregate.adaptive_retry_count = aggregate.adaptive_retry_count.saturating_add(current.adaptive_retry_count);
        aggregate.adaptive_spill_count = aggregate.adaptive_spill_count.saturating_add(current.adaptive_spill_count);
        aggregate.adaptive_downgrade_count =
            aggregate.adaptive_downgrade_count.saturating_add(current.adaptive_downgrade_count);
        aggregate.adaptive_memory_history_lookup_count =
            aggregate.adaptive_memory_history_lookup_count.saturating_add(current.adaptive_memory_history_lookup_count);
        aggregate.adaptive_memory_history_hit_count =
            aggregate.adaptive_memory_history_hit_count.saturating_add(current.adaptive_memory_history_hit_count);
        aggregate.adaptive_memory_history_miss_count =
            aggregate.adaptive_memory_history_miss_count.saturating_add(current.adaptive_memory_history_miss_count);
        aggregate.dynamic_filter_produced_count =
            aggregate.dynamic_filter_produced_count.saturating_add(current.dynamic_filter_produced_count);
        aggregate.dynamic_filter_skipped_count =
            aggregate.dynamic_filter_skipped_count.saturating_add(current.dynamic_filter_skipped_count);
        aggregate.dynamic_filter_applied_count =
            aggregate.dynamic_filter_applied_count.saturating_add(current.dynamic_filter_applied_count);
        aggregate.dynamic_filter_row_groups_pruned =
            aggregate.dynamic_filter_row_groups_pruned.saturating_add(current.dynamic_filter_row_groups_pruned);
        aggregate.dynamic_filter_source_bytes_pruned =
            aggregate.dynamic_filter_source_bytes_pruned.saturating_add(current.dynamic_filter_source_bytes_pruned);
        aggregate.dynamic_filter_reader_predicate_rows_before = aggregate
            .dynamic_filter_reader_predicate_rows_before
            .saturating_add(current.dynamic_filter_reader_predicate_rows_before);
        aggregate.dynamic_filter_reader_predicate_rows_after = aggregate
            .dynamic_filter_reader_predicate_rows_after
            .saturating_add(current.dynamic_filter_reader_predicate_rows_after);
        aggregate.dynamic_filter_reader_predicate_rows_filtered = aggregate
            .dynamic_filter_reader_predicate_rows_filtered
            .saturating_add(current.dynamic_filter_reader_predicate_rows_filtered);
        aggregate.dynamic_filter_post_decode_rows_before = aggregate
            .dynamic_filter_post_decode_rows_before
            .saturating_add(current.dynamic_filter_post_decode_rows_before);
        aggregate.dynamic_filter_post_decode_rows_after = aggregate
            .dynamic_filter_post_decode_rows_after
            .saturating_add(current.dynamic_filter_post_decode_rows_after);
        aggregate.dynamic_filter_post_decode_rows_filtered = aggregate
            .dynamic_filter_post_decode_rows_filtered
            .saturating_add(current.dynamic_filter_post_decode_rows_filtered);
        aggregate.dynamic_filter_source_post_decode_rows_filtered = aggregate
            .dynamic_filter_source_post_decode_rows_filtered
            .saturating_add(current.dynamic_filter_source_post_decode_rows_filtered);
        aggregate.dynamic_filter_filter_post_decode_rows_filtered = aggregate
            .dynamic_filter_filter_post_decode_rows_filtered
            .saturating_add(current.dynamic_filter_filter_post_decode_rows_filtered);
        aggregate.dynamic_filter_join_probe_post_decode_rows_filtered = aggregate
            .dynamic_filter_join_probe_post_decode_rows_filtered
            .saturating_add(current.dynamic_filter_join_probe_post_decode_rows_filtered);
        aggregate.merge_dynamic_filter_runtime_rows(current.dynamic_filter_runtime_rows);
        aggregate.hash_join_build_count = aggregate.hash_join_build_count.saturating_add(current.hash_join_build_count);
        aggregate.filtered_join_build_count =
            aggregate.filtered_join_build_count.saturating_add(current.filtered_join_build_count);
        aggregate.one_shot_join_count = aggregate.one_shot_join_count.saturating_add(current.one_shot_join_count);
        aggregate.hash_join_cache_hits = aggregate.hash_join_cache_hits.saturating_add(current.hash_join_cache_hits);
        aggregate.hash_join_cache_misses =
            aggregate.hash_join_cache_misses.saturating_add(current.hash_join_cache_misses);
        aggregate.hash_join_durable_cache_hits =
            aggregate.hash_join_durable_cache_hits.saturating_add(current.hash_join_durable_cache_hits);
        aggregate.hash_join_durable_cache_misses =
            aggregate.hash_join_durable_cache_misses.saturating_add(current.hash_join_durable_cache_misses);
        aggregate.hash_join_durable_cache_inserts =
            aggregate.hash_join_durable_cache_inserts.saturating_add(current.hash_join_durable_cache_inserts);
        aggregate.hash_join_hash_table_bytes =
            aggregate.hash_join_hash_table_bytes.saturating_add(current.hash_join_hash_table_bytes);
        aggregate.filtered_join_hash_table_bytes =
            aggregate.filtered_join_hash_table_bytes.saturating_add(current.filtered_join_hash_table_bytes);
        aggregate.filtered_join_probe_state_bytes =
            aggregate.filtered_join_probe_state_bytes.saturating_add(current.filtered_join_probe_state_bytes);
        aggregate.mixed_join_size_data_bytes =
            aggregate.mixed_join_size_data_bytes.saturating_add(current.mixed_join_size_data_bytes);
        aggregate.one_shot_join_index_bytes =
            aggregate.one_shot_join_index_bytes.saturating_add(current.one_shot_join_index_bytes);
        aggregate.one_shot_join_materialization_bytes =
            aggregate.one_shot_join_materialization_bytes.saturating_add(current.one_shot_join_materialization_bytes);
        aggregate.one_shot_join_output_bytes =
            aggregate.one_shot_join_output_bytes.saturating_add(current.one_shot_join_output_bytes);
        aggregate.join_gather_map_bytes = aggregate.join_gather_map_bytes.saturating_add(current.join_gather_map_bytes);
        aggregate.merge_aggregate_strategy_rows(current.aggregate_strategy_rows);
        aggregate.merge_join_strategy_rows(current.join_strategy_rows);
    }
    aggregate
}

fn native_started(error: DataFusionError) -> NativeRunFailure {
    let details = FailureDetails::from_datafusion_error(&error);
    NativeRunFailure { runtime_started: true, message: error.to_string(), details }
}

fn native_not_started(error: DataFusionError) -> NativeRunFailure {
    let details = FailureDetails::from_datafusion_error(&error);
    NativeRunFailure { runtime_started: false, message: error.to_string(), details }
}

async fn plan_cpu_query(metadata: &TpchDatasetMetadata, query: &TpchQuery) -> Result<PlannedCpuQuery, CpuRunFailure> {
    let ctx = SessionContext::new_with_config(cpu_session_config());
    dataset::register_tpch_tables(&ctx, metadata).await.map_err(cpu_not_started)?;
    let df = ctx.sql(query.sql).await.map_err(cpu_not_started)?;
    let plan = df.create_physical_plan().await.map_err(cpu_not_started)?;
    Ok(PlannedCpuQuery { _ctx: ctx, plan })
}

async fn collect_cpu_plan_once(planned: PlannedCpuQuery) -> Result<Vec<RecordBatch>, CpuRunFailure> {
    let PlannedCpuQuery { _ctx, plan } = planned;
    collect(plan, Arc::new(TaskContext::default())).await.map_err(cpu_started)
}

async fn collect_native_plan_once(plan: Arc<dyn ExecutionPlan>) -> DataFusionResult<(Vec<RecordBatch>, MetricsSet)> {
    let batches = collect(Arc::clone(&plan), Arc::new(TaskContext::default())).await?;
    let metrics = plan.metrics().unwrap_or_default();
    Ok((batches, metrics))
}

fn cpu_session_config() -> SessionConfig {
    match std::env::var("NEXUS_TPCH_CPU_PROFILE").as_deref().map(str::to_ascii_lowercase).as_deref() {
        Ok("untuned") | Ok("report") => tpch_report_session_config(),
        _ => crate::report_workload::tpch::tpch_tuned_cpu_session_config(),
    }
}

fn measure_output(schema: &arrow::datatypes::SchemaRef, batches: &[RecordBatch]) -> DataFusionResult<QueryMeasurement> {
    Ok(QueryMeasurement {
        output_batches: batches.len(),
        output_rows: batches.iter().map(RecordBatch::num_rows).sum(),
        output_bytes: batches.iter().map(get_record_batch_memory_size).sum(),
        output_fingerprint: validation::fingerprint_batches(schema, batches)?,
    })
}

fn cpu_started(error: DataFusionError) -> CpuRunFailure {
    let details = FailureDetails::from_datafusion_error(&error);
    CpuRunFailure { runtime_started: true, message: error.to_string(), details }
}

fn cpu_not_started(error: DataFusionError) -> CpuRunFailure {
    let details = FailureDetails::from_datafusion_error(&error);
    CpuRunFailure { runtime_started: false, message: error.to_string(), details }
}

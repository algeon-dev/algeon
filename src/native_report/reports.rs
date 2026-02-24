#![deny(missing_docs)]

use super::*;

/// In-memory fallback-pressure report.
///
/// Rows describe native candidate acceptance or rejection before execution.
/// TSV serialization writes [`NATIVE_FALLBACK_PRESSURE_TSV_SCHEMA_VERSION`],
/// optional workload metadata, then [`NATIVE_FALLBACK_PRESSURE_TSV_HEADER`].
/// Optional planning fields serialize as `none`; missing dataset content
/// fingerprints serialize as `not_computed`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeFallbackPressureReport {
    /// Candidate rows in workload order. One query can produce multiple rows.
    pub rows: Vec<NativeFallbackPressureRow>,
    /// Optional fixture metadata written as `# workload_*` TSV header lines.
    pub workload_dataset: Option<WorkloadDatasetMetadata>,
}

impl NativeFallbackPressureReport {
    /// Collects diagnostic fallback-pressure rows using the unbounded native
    /// execution config.
    ///
    /// This convenience path is for legacy shape-pressure diagnostics only.
    /// Production report surfaces should use
    /// [`Self::collect_with_native_execution_config`] with an explicit bounded
    /// or adaptive config so bounded diagnostics affect fallback-pressure rows.
    pub fn collect(items: impl IntoIterator<Item = NativeFallbackPressureWorkloadItem>) -> Self {
        let execution_config = NativeCudfExecutionConfig::unbounded();
        Self::collect_with_native_execution_config(items, &execution_config)
    }

    /// Collects fallback-pressure rows using an explicit native execution config.
    ///
    /// Adaptive mode can add adaptive pipeline diagnostics to otherwise
    /// rewritten candidates; strict-only evidence remains on strict report
    /// surfaces.
    pub fn collect_with_native_execution_config(
        items: impl IntoIterator<Item = NativeFallbackPressureWorkloadItem>,
        execution_config: &NativeCudfExecutionConfig,
    ) -> Self {
        let rows = items.into_iter().flat_map(|item| report_rows_for_item(item, execution_config)).collect();
        Self { rows, workload_dataset: None }
    }

    /// Attaches fixture metadata to comment lines before the TSV header.
    ///
    /// Free-form metadata text is sanitized for TSV comments by replacing tabs
    /// and line breaks with spaces.
    pub fn with_workload_dataset_metadata(mut self, metadata: WorkloadDatasetMetadata) -> Self {
        self.workload_dataset = Some(metadata);
        self
    }

    /// Number of accepted candidate rows, not distinct query ids.
    pub fn rewritten_count(&self) -> usize {
        self.rows.iter().filter(|row| row.outcome == NativeFallbackPressureOutcome::Rewritten).count()
    }

    /// Number of rejected candidate rows, not distinct query ids.
    pub fn rejected_count(&self) -> usize {
        self.rows.iter().filter(|row| row.outcome == NativeFallbackPressureOutcome::Rejected).count()
    }

    /// Serializes the report as TSV with metadata comments followed by rows.
    pub fn to_tsv(&self) -> String {
        let mut output = String::new();
        let _ = writeln!(
            output,
            "# native_fallback_pressure_tsv_schema_version={NATIVE_FALLBACK_PRESSURE_TSV_SCHEMA_VERSION}"
        );
        write_workload_dataset_metadata(&mut output, self.workload_dataset.as_ref());
        output.push_str(NATIVE_FALLBACK_PRESSURE_TSV_HEADER);
        output.push('\n');
        for row in &self.rows {
            let _ = writeln!(
                output,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                row.query_id,
                row.physical_root,
                row.physical_shape,
                row.native_candidate_shape.as_deref().unwrap_or("none"),
                row.native_candidate_shape_family.as_deref().unwrap_or("none"),
                row.outcome.as_str(),
                row.reason_code.as_deref().unwrap_or("none"),
                row.unsupported_category.as_deref().unwrap_or("none"),
                format_adaptive_pipeline_rows(&row.adaptive_pipeline_rows),
                format_join_diagnostics(&row.join_diagnostics),
                format_source_diagnostics(&row.source_diagnostics),
                row.rejected_before_execution,
            );
        }
        output
    }
}

/// In-memory DataFusion planning report after the native optimizer rule.
///
/// Rows describe post-optimizer native coverage and structured fallback
/// reasons. TSV serialization preserves header order for
/// [`NATIVE_DATAFUSION_PLANNING_TSV_SCHEMA_VERSION`].
/// Optional planning fields serialize as `none`; missing dataset content
/// fingerprints serialize as `not_computed`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeDataFusionPlanningReport {
    /// Per-query planning rows in workload order.
    pub rows: Vec<NativeDataFusionPlanningRow>,
    /// Optional fixture metadata written as `# workload_*` TSV header lines.
    pub workload_dataset: Option<WorkloadDatasetMetadata>,
}

impl NativeDataFusionPlanningReport {
    /// Collects planning rows from captured workload items.
    pub fn collect(items: impl IntoIterator<Item = NativeDataFusionPlanningWorkloadItem>) -> Self {
        let rows = items.into_iter().flat_map(datafusion_planning_rows_for_item).collect();
        Self { rows, workload_dataset: None }
    }

    /// Builds a one-query report from a native rule that retained diagnostics.
    ///
    /// Missing retained diagnostics are represented as a planning-failure row.
    pub fn from_retained_rule(
        query_id: impl Into<String>,
        planned_plan: Arc<dyn ExecutionPlan>,
        native_rule: &CudfNativePhysicalOptimizerRule,
    ) -> Self {
        let Some(diagnostics) = native_rule.last_planning_diagnostics() else {
            return Self::collect([NativeDataFusionPlanningWorkloadItem::planning_failure(
                query_id,
                "native_diagnostics",
                "missing_native_diagnostics",
                "native optimizer rule did not retain planning diagnostics",
            )]);
        };
        Self::collect([NativeDataFusionPlanningWorkloadItem::new(query_id, planned_plan, diagnostics)])
    }

    /// Attaches fixture metadata to comment lines before the TSV header.
    ///
    /// Free-form metadata text is sanitized for TSV comments by replacing tabs
    /// and line breaks with spaces.
    pub fn with_workload_dataset_metadata(mut self, metadata: WorkloadDatasetMetadata) -> Self {
        self.workload_dataset = Some(metadata);
        self
    }

    /// Number of planning rows where native planning replaced at least one fragment.
    pub fn accepted_count(&self) -> usize {
        self.rows.iter().filter(|row| row.outcome == NativeDataFusionPlanningOutcome::AcceptedReplaced).count()
    }

    /// Number of planning rows rejected before native execution.
    pub fn rejected_count(&self) -> usize {
        self.rows.iter().filter(|row| row.outcome == NativeDataFusionPlanningOutcome::RejectedBeforeExecution).count()
    }

    /// Serializes the report as TSV with metadata comments followed by rows.
    pub fn to_tsv(&self) -> String {
        let mut output = String::new();
        let _ = writeln!(
            output,
            "# native_datafusion_planning_tsv_schema_version={NATIVE_DATAFUSION_PLANNING_TSV_SCHEMA_VERSION}"
        );
        write_workload_dataset_metadata(&mut output, self.workload_dataset.as_ref());
        output.push_str(NATIVE_DATAFUSION_PLANNING_TSV_HEADER);
        output.push('\n');
        for row in &self.rows {
            let _ = writeln!(
                output,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                row.query_id,
                row.physical_root,
                row.physical_shape,
                row.native_candidate_shape.as_deref().unwrap_or("none"),
                row.native_candidate_shape_family.as_deref().unwrap_or("none"),
                row.outcome.as_str(),
                row.reason_code.as_deref().unwrap_or("none"),
                row.unsupported_category.as_deref().unwrap_or("none"),
                format_strict_bounded_contract_rows(&row.strict_bounded_contract_rows),
                format_strict_bounded_compiled_graphs(&row.strict_bounded_compiled_graphs),
                format_adaptive_pipeline_rows(&row.adaptive_pipeline_rows),
                format_native_optimizer_diagnostics(&row.native_optimizer),
                row.edge_normalize_inserted,
                format_join_diagnostics(&row.join_diagnostics),
                format_source_diagnostics(&row.source_diagnostics),
                row.post_native_plan_root,
                row.post_native_execution_path.as_str(),
                row.native_plan_fragment_count,
                row.host_datafusion_boundary_count,
                row.rejected_before_execution,
            );
        }
        output
    }
}

/// In-memory DataFusion execution report.
///
/// Rows combine native planning diagnostics, runtime outcome, output counters,
/// and projected native metrics. `not_run` marks output counters for rows
/// where native runtime did not start; optional metric observations use `none`.
/// Missing dataset content fingerprints serialize as `not_computed`.
#[derive(Debug, Clone)]
pub struct NativeDataFusionExecutionReport {
    /// Per-query execution rows in workload order.
    pub rows: Vec<NativeDataFusionExecutionRow>,
    /// Optional fixture metadata written as `# workload_*` TSV header lines.
    pub workload_dataset: Option<WorkloadDatasetMetadata>,
}

impl NativeDataFusionExecutionReport {
    /// Collects execution rows from captured planning/runtime workload items.
    pub fn collect(items: impl IntoIterator<Item = NativeDataFusionExecutionWorkloadItem>) -> Self {
        let rows = items.into_iter().flat_map(datafusion_execution_rows_for_item).collect();
        Self { rows, workload_dataset: None }
    }

    /// Attaches fixture metadata to comment lines before the TSV header.
    ///
    /// Free-form metadata text is sanitized for TSV comments by replacing tabs
    /// and line breaks with spaces.
    pub fn with_workload_dataset_metadata(mut self, metadata: WorkloadDatasetMetadata) -> Self {
        self.workload_dataset = Some(metadata);
        self
    }

    /// Number of rows where native runtime was invoked.
    pub fn runtime_started_count(&self) -> usize {
        self.rows.iter().filter(|row| row.native_runtime_started).count()
    }

    /// Number of rows rejected before native runtime started.
    pub fn rejected_before_execution_count(&self) -> usize {
        self.rows.iter().filter(|row| row.rejected_before_execution).count()
    }

    /// Serializes the report as TSV with metadata comments followed by rows.
    pub fn to_tsv(&self) -> String {
        let mut output = String::new();
        let _ = writeln!(
            output,
            "# native_datafusion_execution_tsv_schema_version={NATIVE_DATAFUSION_EXECUTION_TSV_SCHEMA_VERSION}"
        );
        write_workload_dataset_metadata(&mut output, self.workload_dataset.as_ref());
        output.push_str(NATIVE_DATAFUSION_EXECUTION_TSV_HEADER);
        output.push('\n');
        for row in &self.rows {
            let fields = [
                row.query_id.clone(),
                row.physical_root.clone(),
                row.physical_shape.clone(),
                row.native_candidate_shape.clone().unwrap_or_else(|| "none".to_string()),
                row.native_candidate_shape_family.clone().unwrap_or_else(|| "none".to_string()),
                row.planning_outcome.as_str().to_string(),
                row.reason_code.clone().unwrap_or_else(|| "none".to_string()),
                row.unsupported_category.clone().unwrap_or_else(|| "none".to_string()),
                format_strict_bounded_contract_rows(&row.strict_bounded_contract_rows),
                format_strict_bounded_compiled_graphs(&row.strict_bounded_compiled_graphs),
                format_adaptive_pipeline_rows(&row.adaptive_pipeline_rows),
                row.post_native_plan_root.clone(),
                row.post_native_execution_path.as_str().to_string(),
                row.native_plan_fragment_count.to_string(),
                row.host_datafusion_boundary_count.to_string(),
                row.rejected_before_execution.to_string(),
                row.native_runtime_started.to_string(),
                row.runtime_outcome.as_str().to_string(),
                row.runtime_failure.clone().unwrap_or_else(|| "none".to_string()),
                row.output_batches.map_or_else(|| "not_run".to_string(), |value| value.to_string()),
                row.output_rows.map_or_else(|| "not_run".to_string(), |value| value.to_string()),
                row.output_bytes.map_or_else(|| "not_run".to_string(), |value| value.to_string()),
                row.metrics.gpu_resident_input_handoffs.to_string(),
                row.metrics.host_to_device_uploads.to_string(),
                row.metrics.host_to_device_uploaded_rows.to_string(),
                row.metrics.host_to_device_uploaded_bytes.to_string(),
                row.metrics.device_to_host_materializations.to_string(),
                row.metrics.device_to_host_materialized_rows.to_string(),
                row.metrics.device_to_host_materialized_bytes.to_string(),
                row.metrics.device_to_device_copies.to_string(),
                row.metrics.device_to_device_copied_bytes.to_string(),
                row.metrics.parquet_files.to_string(),
                row.metrics.parquet_bytes_known.to_string(),
                row.metrics.row_groups_selected.to_string(),
                row.metrics.read_columns.to_string(),
                row.metrics.resident_dataframe_peak_bytes.map_or_else(|| "none".to_string(), |value| value.to_string()),
                row.metrics.allocator_high_water_bytes.map_or_else(|| "none".to_string(), |value| value.to_string()),
                row.metrics.reservation_requested_bytes.to_string(),
                row.metrics.reservation_granted_bytes.to_string(),
                row.metrics.reservation_live_peak_bytes.to_string(),
                row.metrics.reservation_denied_count.to_string(),
                row.metrics.max_task_reservation_bytes.to_string(),
                row.metrics.max_task_observed_peak_bytes.map_or_else(|| "none".to_string(), |value| value.to_string()),
                row.metrics.max_task_retained_bytes.map_or_else(|| "none".to_string(), |value| value.to_string()),
                format_graph_runtime_rows(&row.metrics.graph_runtime_rows),
                format_task_operator_rows(&row.metrics.task_operator_rows),
                format_task_runtime_rows(&row.metrics.task_runtime_rows),
                row.metrics.source_chunks_planned.to_string(),
                row.metrics.source_chunks_attempted.to_string(),
                row.metrics.source_chunks_committed.to_string(),
                row.metrics.source_chunks_retried.to_string(),
                row.metrics.source_read_prefetches_started.to_string(),
                row.metrics.source_read_prefetches_completed.to_string(),
                row.metrics.source_read_prefetches_skipped.to_string(),
                row.metrics.source_read_prefetches_abandoned.to_string(),
                row.metrics.source_cache_hits.to_string(),
                row.metrics.source_cache_misses.to_string(),
                row.metrics.source_cache_inserts.to_string(),
                row.metrics.source_cache_insert_skipped.to_string(),
                row.metrics.source_cache_retained_bytes.to_string(),
                row.metrics.row_groups_planned.to_string(),
                row.metrics.row_groups_committed.to_string(),
                row.metrics.partial_aggregate_count.to_string(),
                row.metrics.partial_aggregate_input_rows.to_string(),
                row.metrics.partial_aggregate_rows.to_string(),
                row.metrics.partial_aggregate_retained_bytes.to_string(),
                row.metrics.final_merge_attempts.to_string(),
                row.metrics.final_merge_levels.to_string(),
                row.metrics.final_merge_max_fan_in.to_string(),
                row.metrics.retry_count.to_string(),
                row.metrics.retry_resource_exhausted_count.to_string(),
                row.metrics.retry_reservation_failure_count.to_string(),
                row.metrics.adaptive_retry_count.to_string(),
                row.metrics.adaptive_spill_count.to_string(),
                row.metrics.adaptive_downgrade_count.to_string(),
                row.metrics.adaptive_memory_history_lookup_count.to_string(),
                row.metrics.adaptive_memory_history_hit_count.to_string(),
                row.metrics.adaptive_memory_history_miss_count.to_string(),
                row.metrics.hash_join_build_count.to_string(),
                row.metrics.filtered_join_build_count.to_string(),
                row.metrics.one_shot_join_count.to_string(),
                row.metrics.hash_join_cache_hits.to_string(),
                row.metrics.hash_join_cache_misses.to_string(),
                row.metrics.hash_join_durable_cache_hits.to_string(),
                row.metrics.hash_join_durable_cache_misses.to_string(),
                row.metrics.hash_join_durable_cache_inserts.to_string(),
                row.metrics.hash_join_hash_table_bytes.to_string(),
                row.metrics.filtered_join_hash_table_bytes.to_string(),
                row.metrics.filtered_join_probe_state_bytes.to_string(),
                row.metrics.mixed_join_size_data_bytes.to_string(),
                row.metrics.one_shot_join_index_bytes.to_string(),
                row.metrics.one_shot_join_materialization_bytes.to_string(),
                row.metrics.one_shot_join_output_bytes.to_string(),
                row.metrics.join_gather_map_bytes.to_string(),
                format_aggregate_strategy_rows(&row.metrics.aggregate_strategy_rows),
                format_join_strategy_rows(&row.metrics.join_strategy_rows),
                row.metrics.source_fetch_bytes.to_string(),
                row.metrics.source_fetch_count.to_string(),
                optional_duration_ms_text(row.metrics.host_fetch_budget_wait_time_ns),
                optional_duration_ms_text(row.metrics.source_fetch_concurrency_wait_time_ns),
                optional_duration_ms_text(row.metrics.source_fetch_time_ns),
                optional_duration_ms_text(row.metrics.source_decode_time_ns),
                format_source_diagnostics(&row.source_diagnostics),
            ];
            let _ = writeln!(output, "{}", fields.join("\t"));
        }
        output
    }
}

fn optional_duration_ms_text(value: Option<usize>) -> String {
    value.map_or_else(|| "none".to_string(), |value| format!("{:.3}", value as f64 / 1_000_000.0))
}

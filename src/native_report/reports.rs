use super::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeFallbackPressureReport {
    pub rows: Vec<NativeFallbackPressureRow>,
    pub tpch_dataset: Option<TpchDatasetMetadata>,
}

impl NativeFallbackPressureReport {
    pub fn collect(items: impl IntoIterator<Item = NativeFallbackPressureWorkloadItem>) -> Self {
        let rows = items.into_iter().flat_map(report_rows_for_item).collect();
        Self { rows, tpch_dataset: None }
    }

    pub fn with_tpch_dataset_metadata(mut self, metadata: TpchDatasetMetadata) -> Self {
        self.tpch_dataset = Some(metadata);
        self
    }

    pub fn rewritten_count(&self) -> usize {
        self.rows.iter().filter(|row| row.outcome == NativeFallbackPressureOutcome::Rewritten).count()
    }

    pub fn rejected_count(&self) -> usize {
        self.rows.iter().filter(|row| row.outcome == NativeFallbackPressureOutcome::Rejected).count()
    }

    pub fn to_tsv(&self) -> String {
        let mut output = String::new();
        let _ = writeln!(
            output,
            "# native_fallback_pressure_tsv_schema_version={NATIVE_FALLBACK_PRESSURE_TSV_SCHEMA_VERSION}"
        );
        write_tpch_dataset_metadata(&mut output, self.tpch_dataset.as_ref());
        output.push_str(NATIVE_FALLBACK_PRESSURE_TSV_HEADER);
        output.push('\n');
        for row in &self.rows {
            let _ = writeln!(
                output,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                row.query_id,
                row.physical_root,
                row.physical_shape,
                row.native_candidate_shape.as_deref().unwrap_or("none"),
                row.native_candidate_shape_family.as_deref().unwrap_or("none"),
                row.outcome.as_str(),
                row.reason_code.as_deref().unwrap_or("none"),
                row.unsupported_category.as_deref().unwrap_or("none"),
                format_join_diagnostics(&row.join_diagnostics),
                format_source_diagnostics(&row.source_diagnostics),
                row.rejected_before_execution,
            );
        }
        output
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeDataFusionPlanningReport {
    pub rows: Vec<NativeDataFusionPlanningRow>,
    pub tpch_dataset: Option<TpchDatasetMetadata>,
}

impl NativeDataFusionPlanningReport {
    pub fn collect(items: impl IntoIterator<Item = NativeDataFusionPlanningWorkloadItem>) -> Self {
        let rows = items.into_iter().flat_map(datafusion_planning_rows_for_item).collect();
        Self { rows, tpch_dataset: None }
    }

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

    pub fn with_tpch_dataset_metadata(mut self, metadata: TpchDatasetMetadata) -> Self {
        self.tpch_dataset = Some(metadata);
        self
    }

    pub fn accepted_count(&self) -> usize {
        self.rows.iter().filter(|row| row.outcome == NativeDataFusionPlanningOutcome::AcceptedReplaced).count()
    }

    pub fn rejected_count(&self) -> usize {
        self.rows.iter().filter(|row| row.outcome == NativeDataFusionPlanningOutcome::RejectedBeforeExecution).count()
    }

    pub fn to_tsv(&self) -> String {
        let mut output = String::new();
        let _ = writeln!(
            output,
            "# native_datafusion_planning_tsv_schema_version={NATIVE_DATAFUSION_PLANNING_TSV_SCHEMA_VERSION}"
        );
        write_tpch_dataset_metadata(&mut output, self.tpch_dataset.as_ref());
        output.push_str(NATIVE_DATAFUSION_PLANNING_TSV_HEADER);
        output.push('\n');
        for row in &self.rows {
            let _ = writeln!(
                output,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
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
                format_join_diagnostics(&row.join_diagnostics),
                format_source_diagnostics(&row.source_diagnostics),
                row.post_native_plan_root,
                row.post_native_execution_path.as_str(),
                row.rejected_before_execution,
            );
        }
        output
    }
}

#[derive(Debug, Clone)]
pub struct NativeDataFusionExecutionReport {
    pub rows: Vec<NativeDataFusionExecutionRow>,
    pub tpch_dataset: Option<TpchDatasetMetadata>,
}

impl NativeDataFusionExecutionReport {
    pub fn collect(items: impl IntoIterator<Item = NativeDataFusionExecutionWorkloadItem>) -> Self {
        let rows = items.into_iter().flat_map(datafusion_execution_rows_for_item).collect();
        Self { rows, tpch_dataset: None }
    }

    pub fn with_tpch_dataset_metadata(mut self, metadata: TpchDatasetMetadata) -> Self {
        self.tpch_dataset = Some(metadata);
        self
    }

    pub fn runtime_started_count(&self) -> usize {
        self.rows.iter().filter(|row| row.native_runtime_started).count()
    }

    pub fn rejected_before_execution_count(&self) -> usize {
        self.rows.iter().filter(|row| row.rejected_before_execution).count()
    }

    pub fn to_tsv(&self) -> String {
        let mut output = String::new();
        let _ = writeln!(
            output,
            "# native_datafusion_execution_tsv_schema_version={NATIVE_DATAFUSION_EXECUTION_TSV_SCHEMA_VERSION}"
        );
        write_tpch_dataset_metadata(&mut output, self.tpch_dataset.as_ref());
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
                row.post_native_plan_root.clone(),
                row.post_native_execution_path.as_str().to_string(),
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
                row.metrics.reservation_denied_count.to_string(),
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
                row.metrics.partial_aggregate_rows.to_string(),
                row.metrics.final_merge_attempts.to_string(),
                row.metrics.final_merge_levels.to_string(),
                row.metrics.retry_count.to_string(),
                row.metrics.retry_resource_exhausted_count.to_string(),
                row.metrics.retry_reservation_failure_count.to_string(),
                row.metrics.hash_join_build_count.to_string(),
                row.metrics.filtered_join_build_count.to_string(),
                row.metrics.hash_join_cache_hits.to_string(),
                row.metrics.hash_join_cache_misses.to_string(),
                row.metrics.hash_join_durable_cache_hits.to_string(),
                row.metrics.hash_join_durable_cache_misses.to_string(),
                row.metrics.hash_join_durable_cache_inserts.to_string(),
                row.metrics.hash_join_hash_table_bytes.to_string(),
                row.metrics.filtered_join_hash_table_bytes.to_string(),
                row.metrics.filtered_join_probe_state_bytes.to_string(),
                row.metrics.mixed_join_size_data_bytes.to_string(),
                row.metrics.join_gather_map_bytes.to_string(),
                format_source_diagnostics(&row.source_diagnostics),
            ];
            let _ = writeln!(output, "{}", fields.join("\t"));
        }
        output
    }
}

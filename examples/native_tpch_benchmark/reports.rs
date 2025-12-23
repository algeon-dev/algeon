use super::*;

pub(super) const NATIVE_TPCH_BENCHMARK_TSV_SCHEMA_VERSION: u32 = 1;
pub(super) const NATIVE_TPCH_BENCHMARK_TSV_HEADER: &str = "query_id\tplanning_outcome\treason_code\tunsupported_category\tnative_candidate_shape_family\tpost_native_execution_path\trejected_before_execution\tcpu_runtime_started\tnative_runtime_started\truntime_outcome\tcpu_failure\truntime_failure\tparity_status\tcpu_wall_ms\tnative_wall_ms\tspeedup_vs_cpu\tcpu_output_batches\tnative_output_batches\tcpu_output_rows\tnative_output_rows\tcpu_output_bytes\tnative_output_bytes\tcpu_output_fingerprint\tnative_output_fingerprint\toutput_fingerprint\tgpu_resident_input_handoffs\thost_to_device_uploads\thost_to_device_uploaded_rows\thost_to_device_uploaded_bytes\tdevice_to_host_materializations\tdevice_to_host_materialized_rows\tdevice_to_host_materialized_bytes\tdevice_to_device_copies\tdevice_to_device_copied_bytes\tparquet_files\tparquet_bytes_known\trow_groups_selected\tread_columns\tresident_dataframe_peak_bytes\tallocator_high_water_bytes\treservation_requested_bytes\treservation_granted_bytes\treservation_denied_count\tsource_chunks_planned\tsource_chunks_attempted\tsource_chunks_committed\tsource_chunks_retried\trow_groups_planned\trow_groups_committed\tretry_count\tretry_resource_exhausted_count\tretry_reservation_failure_count\thash_join_build_count\tfiltered_join_build_count\thash_join_hash_table_bytes\tfiltered_join_hash_table_bytes\tfiltered_join_probe_state_bytes\tjoin_gather_map_bytes\tparquet_read_time_ms\tsource_read_time_ms\toperator_time_ms\tchunk_operator_time_ms\tfinal_merge_time_ms\tfinal_sort_time_ms\tmaterialization_time_ms";

#[derive(Debug, Clone)]
pub(super) struct NativeTpchBenchmarkReport {
    pub(super) metadata: TpchDatasetMetadata,
    pub(super) config: NativeTpchBenchmarkConfig,
    pub(super) rows: Vec<NativeTpchBenchmarkRow>,
}

impl NativeTpchBenchmarkReport {
    pub(super) fn to_tsv(&self) -> String {
        let mut output = String::new();
        push_line(
            &mut output,
            format_args!("# native_tpch_benchmark_tsv_schema_version={NATIVE_TPCH_BENCHMARK_TSV_SCHEMA_VERSION}"),
        );
        write_tpch_dataset_metadata(&mut output, &self.metadata);
        push_line(&mut output, format_args!("# tpch_benchmark_query_ids={}", self.config.query_ids.join(",")));
        push_line(&mut output, format_args!("# tpch_benchmark_warmups={}", self.config.warmup_iterations));
        push_line(&mut output, format_args!("# tpch_benchmark_iterations={}", self.config.iterations));
        push_line(
            &mut output,
            format_args!("# tpch_benchmark_native_memory_policy={}", self.config.native_memory_policy.as_str()),
        );
        output.push_str(NATIVE_TPCH_BENCHMARK_TSV_HEADER);
        output.push('\n');
        for row in &self.rows {
            push_line(&mut output, format_args!("{}", row.to_tsv_line()));
        }
        output
    }
}

#[derive(Debug, Clone)]
pub(super) struct NativeTpchBenchmarkRow {
    pub(super) query_id: String,
    pub(super) planning_outcome: NativeDataFusionPlanningOutcome,
    pub(super) reason_code: Option<String>,
    pub(super) unsupported_category: Option<String>,
    pub(super) native_candidate_shape_family: Option<String>,
    pub(super) post_native_execution_path: NativeDataFusionPostNativeExecutionPath,
    pub(super) rejected_before_execution: bool,
    pub(super) cpu_runtime_started: bool,
    pub(super) native_runtime_started: bool,
    pub(super) runtime_outcome: NativeDataFusionRuntimeOutcome,
    pub(super) cpu_failure: Option<String>,
    pub(super) runtime_failure: Option<String>,
    pub(super) parity_status: validation::ParityStatus,
    pub(super) cpu_wall_ms: Option<f64>,
    pub(super) native_wall_ms: Option<f64>,
    pub(super) speedup_vs_cpu: Option<f64>,
    pub(super) cpu_output_batches: Option<usize>,
    pub(super) native_output_batches: Option<usize>,
    pub(super) cpu_output_rows: Option<usize>,
    pub(super) native_output_rows: Option<usize>,
    pub(super) cpu_output_bytes: Option<usize>,
    pub(super) native_output_bytes: Option<usize>,
    pub(super) cpu_output_fingerprint: Option<String>,
    pub(super) native_output_fingerprint: Option<String>,
    pub(super) output_fingerprint: Option<String>,
    pub(super) metrics: NativeDataFusionExecutionMetrics,
    pub(super) timing_metrics: NativeTpchTimingMetrics,
}

impl NativeTpchBenchmarkRow {
    pub(super) fn rejected(query_id: &str, planning_row: NativeDataFusionPlanningRow) -> Self {
        Self {
            query_id: query_id.to_string(),
            planning_outcome: planning_row.outcome,
            reason_code: planning_row.reason_code,
            unsupported_category: planning_row.unsupported_category,
            native_candidate_shape_family: planning_row.native_candidate_shape_family,
            post_native_execution_path: planning_row.post_native_execution_path,
            rejected_before_execution: planning_row.rejected_before_execution,
            cpu_runtime_started: false,
            native_runtime_started: false,
            runtime_outcome: NativeDataFusionRuntimeOutcome::NotStarted,
            cpu_failure: None,
            runtime_failure: None,
            parity_status: validation::ParityStatus::NotRun,
            cpu_wall_ms: None,
            native_wall_ms: None,
            speedup_vs_cpu: None,
            cpu_output_batches: None,
            native_output_batches: None,
            cpu_output_rows: None,
            native_output_rows: None,
            cpu_output_bytes: None,
            native_output_bytes: None,
            cpu_output_fingerprint: None,
            native_output_fingerprint: None,
            output_fingerprint: None,
            metrics: NativeDataFusionExecutionMetrics::default(),
            timing_metrics: NativeTpchTimingMetrics::default(),
        }
    }

    pub(super) fn planning_failure(query_id: &str, message: String) -> Self {
        Self {
            query_id: query_id.to_string(),
            planning_outcome: NativeDataFusionPlanningOutcome::RejectedBeforeExecution,
            reason_code: Some("physical_planning_failed".to_string()),
            unsupported_category: Some("planning".to_string()),
            native_candidate_shape_family: None,
            post_native_execution_path: NativeDataFusionPostNativeExecutionPath::Cpu,
            rejected_before_execution: true,
            cpu_runtime_started: false,
            native_runtime_started: false,
            runtime_outcome: NativeDataFusionRuntimeOutcome::NotStarted,
            cpu_failure: None,
            runtime_failure: Some(stable_tsv_field(&message)),
            parity_status: validation::ParityStatus::NotRun,
            cpu_wall_ms: None,
            native_wall_ms: None,
            speedup_vs_cpu: None,
            cpu_output_batches: None,
            native_output_batches: None,
            cpu_output_rows: None,
            native_output_rows: None,
            cpu_output_bytes: None,
            native_output_bytes: None,
            cpu_output_fingerprint: None,
            native_output_fingerprint: None,
            output_fingerprint: None,
            metrics: NativeDataFusionExecutionMetrics::default(),
            timing_metrics: NativeTpchTimingMetrics::default(),
        }
    }

    pub(super) fn accepted(
        query_id: &str,
        planning_row: NativeDataFusionPlanningRow,
        cpu: Result<execution::TimedQueryRun, String>,
        native: Result<execution::TimedNativeQueryRun, execution::NativeRunFailure>,
    ) -> Self {
        let (cpu, cpu_failure) = match cpu {
            Ok(run) => (Some(run), None),
            Err(error) => (None, Some(stable_tsv_field(&error))),
        };
        let (native, native_failure, native_runtime_started) = match native {
            Ok(run) => (Some(run), None, true),
            Err(error) => (None, Some(stable_tsv_field(&error.message)), error.runtime_started),
        };
        let parity_status = validation::parity_status(
            cpu.as_ref().map(|run| &run.final_measurement),
            native.as_ref().map(|run| &run.final_measurement),
        );
        let output_fingerprint = if parity_status == validation::ParityStatus::Matched {
            cpu.as_ref().map(|run| run.final_measurement.output_fingerprint.clone())
        } else {
            None
        };
        let cpu_wall_ms = cpu.as_ref().map(|run| duration_ms(mean_duration(&run.samples)));
        let native_wall_ms = native.as_ref().map(|run| duration_ms(mean_duration(&run.samples)));
        let speedup_vs_cpu = cpu_wall_ms.zip(native_wall_ms).and_then(|(cpu_ms, native_ms)| {
            if native_ms == 0.0 {
                Some(f64::INFINITY)
            } else if cpu_ms == 0.0 {
                None
            } else {
                Some(cpu_ms / native_ms)
            }
        });
        let metrics = native.as_ref().map(|run| run.metrics.clone()).unwrap_or_default();
        let timing_metrics =
            native.as_ref().map(|run| NativeTpchTimingMetrics::from_metrics(&run.raw_metrics)).unwrap_or_default();
        Self {
            query_id: query_id.to_string(),
            planning_outcome: planning_row.outcome,
            reason_code: planning_row.reason_code,
            unsupported_category: planning_row.unsupported_category,
            native_candidate_shape_family: planning_row.native_candidate_shape_family,
            post_native_execution_path: planning_row.post_native_execution_path,
            rejected_before_execution: planning_row.rejected_before_execution,
            cpu_runtime_started: cpu.is_some() || cpu_failure.is_some(),
            native_runtime_started,
            runtime_outcome: if native.is_some() {
                NativeDataFusionRuntimeOutcome::Succeeded
            } else {
                NativeDataFusionRuntimeOutcome::Failed
            },
            cpu_failure,
            runtime_failure: native_failure,
            parity_status,
            cpu_wall_ms,
            native_wall_ms,
            speedup_vs_cpu,
            cpu_output_batches: cpu.as_ref().map(|run| run.final_measurement.output_batches),
            native_output_batches: native.as_ref().map(|run| run.final_measurement.output_batches),
            cpu_output_rows: cpu.as_ref().map(|run| run.final_measurement.output_rows),
            native_output_rows: native.as_ref().map(|run| run.final_measurement.output_rows),
            cpu_output_bytes: cpu.as_ref().map(|run| run.final_measurement.output_bytes),
            native_output_bytes: native.as_ref().map(|run| run.final_measurement.output_bytes),
            cpu_output_fingerprint: cpu.as_ref().map(|run| run.final_measurement.output_fingerprint.clone()),
            native_output_fingerprint: native.as_ref().map(|run| run.final_measurement.output_fingerprint.clone()),
            output_fingerprint,
            metrics,
            timing_metrics,
        }
    }

    fn to_tsv_line(&self) -> String {
        let fields = [
            self.query_id.clone(),
            self.planning_outcome.as_str().to_string(),
            optional_text(self.reason_code.as_deref(), "none"),
            optional_text(self.unsupported_category.as_deref(), "none"),
            optional_text(self.native_candidate_shape_family.as_deref(), "none"),
            self.post_native_execution_path.as_str().to_string(),
            self.rejected_before_execution.to_string(),
            self.cpu_runtime_started.to_string(),
            self.native_runtime_started.to_string(),
            self.runtime_outcome.as_str().to_string(),
            optional_text(self.cpu_failure.as_deref(), "none"),
            optional_text(self.runtime_failure.as_deref(), "none"),
            self.parity_status.as_str().to_string(),
            optional_f64_text(self.cpu_wall_ms),
            optional_f64_text(self.native_wall_ms),
            optional_f64_text(self.speedup_vs_cpu),
            optional_usize_text(self.cpu_output_batches),
            optional_usize_text(self.native_output_batches),
            optional_usize_text(self.cpu_output_rows),
            optional_usize_text(self.native_output_rows),
            optional_usize_text(self.cpu_output_bytes),
            optional_usize_text(self.native_output_bytes),
            optional_text(self.cpu_output_fingerprint.as_deref(), "not_run"),
            optional_text(self.native_output_fingerprint.as_deref(), "not_run"),
            optional_text(self.output_fingerprint.as_deref(), "not_run"),
            self.metrics.gpu_resident_input_handoffs.to_string(),
            self.metrics.host_to_device_uploads.to_string(),
            self.metrics.host_to_device_uploaded_rows.to_string(),
            self.metrics.host_to_device_uploaded_bytes.to_string(),
            self.metrics.device_to_host_materializations.to_string(),
            self.metrics.device_to_host_materialized_rows.to_string(),
            self.metrics.device_to_host_materialized_bytes.to_string(),
            self.metrics.device_to_device_copies.to_string(),
            self.metrics.device_to_device_copied_bytes.to_string(),
            self.metrics.parquet_files.to_string(),
            self.metrics.parquet_bytes_known.to_string(),
            self.metrics.row_groups_selected.to_string(),
            self.metrics.read_columns.to_string(),
            optional_usize_text_none(self.metrics.resident_dataframe_peak_bytes),
            optional_usize_text_none(self.metrics.allocator_high_water_bytes),
            self.metrics.reservation_requested_bytes.to_string(),
            self.metrics.reservation_granted_bytes.to_string(),
            self.metrics.reservation_denied_count.to_string(),
            self.metrics.source_chunks_planned.to_string(),
            self.metrics.source_chunks_attempted.to_string(),
            self.metrics.source_chunks_committed.to_string(),
            self.metrics.source_chunks_retried.to_string(),
            self.metrics.row_groups_planned.to_string(),
            self.metrics.row_groups_committed.to_string(),
            self.metrics.retry_count.to_string(),
            self.metrics.retry_resource_exhausted_count.to_string(),
            self.metrics.retry_reservation_failure_count.to_string(),
            self.metrics.hash_join_build_count.to_string(),
            self.metrics.filtered_join_build_count.to_string(),
            self.metrics.hash_join_hash_table_bytes.to_string(),
            self.metrics.filtered_join_hash_table_bytes.to_string(),
            self.metrics.filtered_join_probe_state_bytes.to_string(),
            self.metrics.join_gather_map_bytes.to_string(),
            optional_f64_text(self.timing_metrics.parquet_read_time_ms),
            optional_f64_text(self.timing_metrics.source_read_time_ms),
            optional_f64_text(self.timing_metrics.operator_time_ms),
            optional_f64_text(self.timing_metrics.chunk_operator_time_ms),
            optional_f64_text(self.timing_metrics.final_merge_time_ms),
            optional_f64_text(self.timing_metrics.final_sort_time_ms),
            optional_f64_text(self.timing_metrics.materialization_time_ms),
        ];
        fields.join("\t")
    }
}

#[derive(Debug, Clone, Default)]
pub(super) struct NativeTpchTimingMetrics {
    pub(super) parquet_read_time_ms: Option<f64>,
    pub(super) source_read_time_ms: Option<f64>,
    pub(super) operator_time_ms: Option<f64>,
    pub(super) chunk_operator_time_ms: Option<f64>,
    pub(super) final_merge_time_ms: Option<f64>,
    pub(super) final_sort_time_ms: Option<f64>,
    pub(super) materialization_time_ms: Option<f64>,
}

impl NativeTpchTimingMetrics {
    fn from_metrics(metrics: &MetricsSet) -> Self {
        Self {
            parquet_read_time_ms: metric_duration_ms(metrics, "parquet_read_time"),
            source_read_time_ms: metric_duration_ms(metrics, "source_read_time"),
            operator_time_ms: metric_duration_ms(metrics, "operator_time"),
            chunk_operator_time_ms: metric_duration_ms(metrics, "chunk_operator_time"),
            final_merge_time_ms: metric_duration_ms(metrics, "final_merge_time"),
            final_sort_time_ms: metric_duration_ms(metrics, "final_sort_time"),
            materialization_time_ms: metric_duration_ms(metrics, "materialization_time"),
        }
    }
}

fn write_tpch_dataset_metadata(output: &mut String, metadata: &TpchDatasetMetadata) {
    push_line(output, format_args!("# tpch_dataset_root={}", metadata.root.display()));
    push_line(output, format_args!("# tpch_dataset_scale_factor={}", metadata_value(&metadata.scale_factor)));
    push_line(output, format_args!("# tpch_dataset_provenance={}", metadata_value(&metadata.provenance)));
    push_line(
        output,
        format_args!("# tpch_dataset_generation_command={}", metadata_value(&metadata.generation_command)),
    );
    push_line(output, format_args!("# tpch_dataset_table_count={}", metadata.table_count));
    push_line(output, format_args!("# tpch_dataset_file_count={}", metadata.file_count));
    push_line(output, format_args!("# tpch_dataset_total_rows={}", metadata.total_rows));
    push_line(output, format_args!("# tpch_dataset_total_bytes={}", metadata.total_bytes));
    push_line(output, format_args!("# tpch_dataset_metadata_fingerprint={}", metadata.metadata_fingerprint));
    push_line(
        output,
        format_args!(
            "# tpch_dataset_content_fingerprint={}",
            metadata.content_fingerprint.as_deref().unwrap_or("not_computed")
        ),
    );
    push_line(output, format_args!("# tpch_dataset_table_file_counts={}", metadata.table_file_counts()));
    push_line(output, format_args!("# tpch_dataset_table_row_counts={}", metadata.table_row_counts()));
    push_line(output, format_args!("# tpch_dataset_table_byte_sizes={}", metadata.table_byte_sizes()));
    push_line(output, format_args!("# tpch_dataset_table_row_group_counts={}", metadata.table_row_group_counts()));
}

fn metric_duration_ms(metrics: &MetricsSet, name: &str) -> Option<f64> {
    metrics
        .sum_by_name(name)
        .map(|value| duration_ms(Duration::from_nanos(value.as_usize().min(u64::MAX as usize) as u64)))
}

fn mean_duration(samples: &[Duration]) -> Duration {
    let total_nanos = samples.iter().map(Duration::as_nanos).sum::<u128>();
    Duration::from_nanos((total_nanos / samples.len().max(1) as u128).min(u128::from(u64::MAX)) as u64)
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn optional_f64_text(value: Option<f64>) -> String {
    value.map_or_else(|| "not_run".to_string(), |value| format!("{value:.3}"))
}

fn optional_usize_text(value: Option<usize>) -> String {
    value.map_or_else(|| "not_run".to_string(), |value| value.to_string())
}

fn optional_usize_text_none(value: Option<usize>) -> String {
    value.map_or_else(|| "none".to_string(), |value| value.to_string())
}

fn optional_text(value: Option<&str>, missing: &str) -> String {
    value.map_or_else(|| missing.to_string(), stable_tsv_field)
}

fn metadata_value(value: &str) -> String {
    value.replace(['\t', '\n', '\r'], " ")
}

fn stable_tsv_field(value: &str) -> String {
    value.replace(['\t', '\n', '\r'], " ")
}

fn push_line(output: &mut String, args: std::fmt::Arguments<'_>) {
    use std::fmt::Write;
    let _ = writeln!(output, "{args}");
}

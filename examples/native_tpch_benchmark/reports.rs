use cudf_query_engine::{AdaptivePipelineReportRow, StrictBoundedContractGateRow};
use datafusion_cudf::NativeStrictBoundedCompiledGraphDiagnostics;

use super::*;

#[path = "reports/formatting.rs"]
mod formatting;

use formatting::*;

pub(super) const NATIVE_TPCH_BENCHMARK_TSV_SCHEMA_VERSION: u32 = 13;
pub(super) const NATIVE_TPCH_BENCHMARK_TSV_HEADER: &str = "query_id\tplanning_outcome\tbounded_status\treason_code\tunsupported_category\tstrict_bounded_contract_rows\tstrict_bounded_compiled_graphs\tadaptive_pipeline_rows\tnative_candidate_shape_family\tpost_native_execution_path\trejected_before_execution\tcpu_runtime_started\tnative_runtime_started\truntime_outcome\tcpu_failure\tcpu_failure_code\tcpu_failure_kind\tcpu_failure_status\tcpu_failure_operation\tcpu_failure_facts\truntime_failure\truntime_failure_code\truntime_failure_kind\truntime_failure_status\truntime_failure_operation\truntime_failure_facts\tparity_status\tcpu_wall_ms\tnative_wall_ms\tspeedup_vs_cpu\tcpu_output_batches\tnative_output_batches\tcpu_output_rows\tnative_output_rows\tcpu_output_bytes\tnative_output_bytes\tcpu_output_fingerprint\tnative_output_fingerprint\toutput_fingerprint\tgpu_resident_input_handoffs\thost_to_device_uploads\thost_to_device_uploaded_rows\thost_to_device_uploaded_bytes\tdevice_to_host_materializations\tdevice_to_host_materialized_rows\tdevice_to_host_materialized_bytes\tdevice_to_device_copies\tdevice_to_device_copied_bytes\tparquet_files\tparquet_bytes_known\trow_groups_selected\tread_columns\tresident_dataframe_peak_bytes\tallocator_high_water_bytes\treservation_requested_bytes\treservation_granted_bytes\treservation_live_peak_bytes\treservation_denied_count\tmax_task_reservation_bytes\tmax_task_observed_peak_bytes\tmax_task_retained_bytes\ttask_operator_rows\tsource_chunks_planned\tsource_chunks_attempted\tsource_chunks_committed\tsource_chunks_retried\tsource_read_prefetches_started\tsource_read_prefetches_completed\tsource_read_prefetches_skipped\tsource_read_prefetches_abandoned\tsource_cache_hits\tsource_cache_misses\tsource_cache_inserts\tsource_cache_insert_skipped\tsource_cache_retained_bytes\trow_groups_planned\trow_groups_committed\tpartial_aggregate_count\tpartial_aggregate_rows\tfinal_merge_attempts\tfinal_merge_levels\tretry_count\tretry_resource_exhausted_count\tretry_reservation_failure_count\tadaptive_retry_count\tadaptive_spill_count\tadaptive_downgrade_count\tadaptive_memory_history_lookup_count\tadaptive_memory_history_hit_count\tadaptive_memory_history_miss_count\thash_join_build_count\tfiltered_join_build_count\tone_shot_join_count\thash_join_cache_hits\thash_join_cache_misses\thash_join_durable_cache_hits\thash_join_durable_cache_misses\thash_join_durable_cache_inserts\thash_join_hash_table_bytes\tfiltered_join_hash_table_bytes\tfiltered_join_probe_state_bytes\tmixed_join_size_data_bytes\tone_shot_join_index_bytes\tone_shot_join_materialization_bytes\tone_shot_join_output_bytes\tjoin_gather_map_bytes\tjoin_strategy_rows\tparquet_read_time_ms\tsource_read_time_ms\tsource_read_queue_wait_time_ms\tsource_cache_lookup_time_ms\toperator_time_ms\tchunk_operator_time_ms\tfinal_merge_time_ms\tfinal_sort_time_ms\tmaterialization_time_ms";

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
        push_line(&mut output, format_args!("# tpch_benchmark_wall_time_fields=per_measured_iteration_mean"));
        push_line(&mut output, format_args!("# tpch_benchmark_native_counter_fields=measured_iteration_totals"));
        push_line(
            &mut output,
            format_args!("# tpch_benchmark_native_peak_memory_fields=max_across_measured_iterations"),
        );
        push_line(&mut output, format_args!("# tpch_benchmark_native_timing_fields=per_measured_iteration_mean"));
        push_line(&mut output, format_args!("# tpch_benchmark_dataset_mode={}", self.config.dataset_mode.as_str()));
        push_line(
            &mut output,
            format_args!(
                "# tpch_benchmark_parquet_pushdown_filters={}",
                self.config
                    .parquet_pushdown_filters
                    .map_or_else(|| "session_default".to_string(), |value| value.to_string())
            ),
        );
        push_line(
            &mut output,
            format_args!(
                "# tpch_benchmark_native_pipeline_execution_mode={}",
                self.config.native_pipeline_execution_mode.as_str()
            ),
        );
        push_line(
            &mut output,
            format_args!("# tpch_benchmark_native_memory_policy={}", self.config.native_memory_policy.as_str()),
        );
        write_native_memory_policy_metadata(
            &mut output,
            self.config.native_memory_policy,
            self.config.max_source_reads_in_flight,
            self.config.native_source_cache_policy,
        );
        output.push_str(NATIVE_TPCH_BENCHMARK_TSV_HEADER);
        output.push('\n');
        for row in &self.rows {
            push_line(&mut output, format_args!("{}", row.to_tsv_line(self.config.native_memory_policy)));
        }
        output
    }
}

#[derive(Debug, Clone)]
pub(super) struct NativeTpchBenchmarkRow {
    pub(super) query_id: String,
    pub(super) planning_outcome: NativeDataFusionPlanningOutcome,
    pub(super) strict_bounded_contract_rows: Vec<StrictBoundedContractGateRow>,
    pub(super) strict_bounded_compiled_graphs: Vec<NativeStrictBoundedCompiledGraphDiagnostics>,
    pub(super) adaptive_pipeline_rows: Vec<AdaptivePipelineReportRow>,
    pub(super) reason_code: Option<String>,
    pub(super) unsupported_category: Option<String>,
    pub(super) native_candidate_shape_family: Option<String>,
    pub(super) post_native_execution_path: NativeDataFusionPostNativeExecutionPath,
    pub(super) rejected_before_execution: bool,
    pub(super) cpu_runtime_started: bool,
    pub(super) native_runtime_started: bool,
    pub(super) runtime_outcome: NativeDataFusionRuntimeOutcome,
    pub(super) cpu_failure: Option<String>,
    pub(super) cpu_failure_details: execution::FailureDetails,
    pub(super) runtime_failure: Option<String>,
    pub(super) runtime_failure_details: execution::FailureDetails,
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
            strict_bounded_contract_rows: planning_row.strict_bounded_contract_rows,
            strict_bounded_compiled_graphs: planning_row.strict_bounded_compiled_graphs,
            adaptive_pipeline_rows: planning_row.adaptive_pipeline_rows,
            reason_code: planning_row.reason_code,
            unsupported_category: planning_row.unsupported_category,
            native_candidate_shape_family: planning_row.native_candidate_shape_family,
            post_native_execution_path: planning_row.post_native_execution_path,
            rejected_before_execution: planning_row.rejected_before_execution,
            cpu_runtime_started: false,
            native_runtime_started: false,
            runtime_outcome: NativeDataFusionRuntimeOutcome::NotStarted,
            cpu_failure: None,
            cpu_failure_details: execution::FailureDetails::default(),
            runtime_failure: None,
            runtime_failure_details: execution::FailureDetails::default(),
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

    pub(super) fn planning_failure(query_id: &str, error: DataFusionError) -> Self {
        let details = execution::FailureDetails::from_datafusion_error(&error);
        Self {
            query_id: query_id.to_string(),
            planning_outcome: NativeDataFusionPlanningOutcome::RejectedBeforeExecution,
            strict_bounded_contract_rows: Vec::new(),
            strict_bounded_compiled_graphs: Vec::new(),
            adaptive_pipeline_rows: Vec::new(),
            reason_code: Some("physical_planning_failed".to_string()),
            unsupported_category: Some("planning".to_string()),
            native_candidate_shape_family: None,
            post_native_execution_path: NativeDataFusionPostNativeExecutionPath::Cpu,
            rejected_before_execution: true,
            cpu_runtime_started: false,
            native_runtime_started: false,
            runtime_outcome: NativeDataFusionRuntimeOutcome::NotStarted,
            cpu_failure: None,
            cpu_failure_details: execution::FailureDetails::default(),
            runtime_failure: Some(stable_tsv_field(&error.to_string())),
            runtime_failure_details: details,
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
        cpu: Result<execution::TimedQueryRun, execution::CpuRunFailure>,
        native: Result<execution::TimedNativeQueryRun, execution::NativeRunFailure>,
    ) -> Self {
        let (cpu, cpu_failure, cpu_failure_details, cpu_runtime_started) = match cpu {
            Ok(run) => (Some(run), None, execution::FailureDetails::default(), true),
            Err(error) => (None, Some(stable_tsv_field(&error.message)), error.details, error.runtime_started),
        };
        let (native, native_failure, runtime_failure_details, native_runtime_started) = match native {
            Ok(run) => (Some(run), None, execution::FailureDetails::default(), true),
            Err(error) => (None, Some(stable_tsv_field(&error.message)), error.details, error.runtime_started),
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
        let timing_metrics = native.as_ref().map(|run| run.timing_metrics.clone()).unwrap_or_default();
        Self {
            query_id: query_id.to_string(),
            planning_outcome: planning_row.outcome,
            strict_bounded_contract_rows: planning_row.strict_bounded_contract_rows,
            strict_bounded_compiled_graphs: planning_row.strict_bounded_compiled_graphs,
            adaptive_pipeline_rows: planning_row.adaptive_pipeline_rows,
            reason_code: planning_row.reason_code,
            unsupported_category: planning_row.unsupported_category,
            native_candidate_shape_family: planning_row.native_candidate_shape_family,
            post_native_execution_path: planning_row.post_native_execution_path,
            rejected_before_execution: planning_row.rejected_before_execution,
            cpu_runtime_started,
            native_runtime_started,
            runtime_outcome: if native.is_some() {
                NativeDataFusionRuntimeOutcome::Succeeded
            } else {
                NativeDataFusionRuntimeOutcome::Failed
            },
            cpu_failure,
            cpu_failure_details,
            runtime_failure: native_failure,
            runtime_failure_details,
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

    fn to_tsv_line(&self, memory_policy: NativeTpchMemoryPolicy) -> String {
        let fields = [
            self.query_id.clone(),
            self.planning_outcome.as_str().to_string(),
            bounded_status(memory_policy, self).to_string(),
            optional_text(self.reason_code.as_deref(), "none"),
            optional_text(self.unsupported_category.as_deref(), "none"),
            format_strict_bounded_contract_rows(&self.strict_bounded_contract_rows),
            format_strict_bounded_compiled_graphs(&self.strict_bounded_compiled_graphs),
            format_adaptive_pipeline_rows(&self.adaptive_pipeline_rows),
            optional_text(self.native_candidate_shape_family.as_deref(), "none"),
            self.post_native_execution_path.as_str().to_string(),
            self.rejected_before_execution.to_string(),
            self.cpu_runtime_started.to_string(),
            self.native_runtime_started.to_string(),
            self.runtime_outcome.as_str().to_string(),
            optional_text(self.cpu_failure.as_deref(), "none"),
            optional_text(self.cpu_failure_details.code.as_deref(), "none"),
            optional_text(self.cpu_failure_details.kind.as_deref(), "none"),
            optional_text(self.cpu_failure_details.status.as_deref(), "none"),
            optional_text(self.cpu_failure_details.operation.as_deref(), "none"),
            format_failure_facts(&self.cpu_failure_details.facts),
            optional_text(self.runtime_failure.as_deref(), "none"),
            optional_text(self.runtime_failure_details.code.as_deref(), "none"),
            optional_text(self.runtime_failure_details.kind.as_deref(), "none"),
            optional_text(self.runtime_failure_details.status.as_deref(), "none"),
            optional_text(self.runtime_failure_details.operation.as_deref(), "none"),
            format_failure_facts(&self.runtime_failure_details.facts),
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
            self.metrics.reservation_live_peak_bytes.to_string(),
            self.metrics.reservation_denied_count.to_string(),
            self.metrics.max_task_reservation_bytes.to_string(),
            optional_usize_text_none(self.metrics.max_task_observed_peak_bytes),
            optional_usize_text_none(self.metrics.max_task_retained_bytes),
            format_task_operator_rows(&self.metrics.task_operator_rows),
            self.metrics.source_chunks_planned.to_string(),
            self.metrics.source_chunks_attempted.to_string(),
            self.metrics.source_chunks_committed.to_string(),
            self.metrics.source_chunks_retried.to_string(),
            self.metrics.source_read_prefetches_started.to_string(),
            self.metrics.source_read_prefetches_completed.to_string(),
            self.metrics.source_read_prefetches_skipped.to_string(),
            self.metrics.source_read_prefetches_abandoned.to_string(),
            self.metrics.source_cache_hits.to_string(),
            self.metrics.source_cache_misses.to_string(),
            self.metrics.source_cache_inserts.to_string(),
            self.metrics.source_cache_insert_skipped.to_string(),
            self.metrics.source_cache_retained_bytes.to_string(),
            self.metrics.row_groups_planned.to_string(),
            self.metrics.row_groups_committed.to_string(),
            self.metrics.partial_aggregate_count.to_string(),
            self.metrics.partial_aggregate_rows.to_string(),
            self.metrics.final_merge_attempts.to_string(),
            self.metrics.final_merge_levels.to_string(),
            self.metrics.retry_count.to_string(),
            self.metrics.retry_resource_exhausted_count.to_string(),
            self.metrics.retry_reservation_failure_count.to_string(),
            self.metrics.adaptive_retry_count.to_string(),
            self.metrics.adaptive_spill_count.to_string(),
            self.metrics.adaptive_downgrade_count.to_string(),
            self.metrics.adaptive_memory_history_lookup_count.to_string(),
            self.metrics.adaptive_memory_history_hit_count.to_string(),
            self.metrics.adaptive_memory_history_miss_count.to_string(),
            self.metrics.hash_join_build_count.to_string(),
            self.metrics.filtered_join_build_count.to_string(),
            self.metrics.one_shot_join_count.to_string(),
            self.metrics.hash_join_cache_hits.to_string(),
            self.metrics.hash_join_cache_misses.to_string(),
            self.metrics.hash_join_durable_cache_hits.to_string(),
            self.metrics.hash_join_durable_cache_misses.to_string(),
            self.metrics.hash_join_durable_cache_inserts.to_string(),
            self.metrics.hash_join_hash_table_bytes.to_string(),
            self.metrics.filtered_join_hash_table_bytes.to_string(),
            self.metrics.filtered_join_probe_state_bytes.to_string(),
            self.metrics.mixed_join_size_data_bytes.to_string(),
            self.metrics.one_shot_join_index_bytes.to_string(),
            self.metrics.one_shot_join_materialization_bytes.to_string(),
            self.metrics.one_shot_join_output_bytes.to_string(),
            self.metrics.join_gather_map_bytes.to_string(),
            format_join_strategy_rows(&self.metrics.join_strategy_rows),
            optional_f64_text(self.timing_metrics.parquet_read_time_ms),
            optional_f64_text(self.timing_metrics.source_read_time_ms),
            optional_f64_text(self.timing_metrics.source_read_queue_wait_time_ms),
            optional_f64_text(self.timing_metrics.source_cache_lookup_time_ms),
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
    pub(super) source_read_queue_wait_time_ms: Option<f64>,
    pub(super) source_cache_lookup_time_ms: Option<f64>,
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
            source_read_queue_wait_time_ms: metric_duration_ms(metrics, "source_read_queue_wait_time"),
            source_cache_lookup_time_ms: metric_duration_ms(metrics, "source_cache_lookup_time"),
            operator_time_ms: metric_duration_ms(metrics, "operator_time"),
            chunk_operator_time_ms: metric_duration_ms(metrics, "chunk_operator_time"),
            final_merge_time_ms: metric_duration_ms(metrics, "final_merge_time"),
            final_sort_time_ms: metric_duration_ms(metrics, "final_sort_time"),
            materialization_time_ms: metric_duration_ms(metrics, "materialization_time"),
        }
    }

    pub(super) fn mean_from_metrics(metrics: &[MetricsSet]) -> Self {
        let values = metrics.iter().map(Self::from_metrics).collect::<Vec<_>>();
        Self {
            parquet_read_time_ms: mean_optional_f64(values.iter().map(|metrics| metrics.parquet_read_time_ms)),
            source_read_time_ms: mean_optional_f64(values.iter().map(|metrics| metrics.source_read_time_ms)),
            source_read_queue_wait_time_ms: mean_optional_f64(
                values.iter().map(|metrics| metrics.source_read_queue_wait_time_ms),
            ),
            source_cache_lookup_time_ms: mean_optional_f64(
                values.iter().map(|metrics| metrics.source_cache_lookup_time_ms),
            ),
            operator_time_ms: mean_optional_f64(values.iter().map(|metrics| metrics.operator_time_ms)),
            chunk_operator_time_ms: mean_optional_f64(values.iter().map(|metrics| metrics.chunk_operator_time_ms)),
            final_merge_time_ms: mean_optional_f64(values.iter().map(|metrics| metrics.final_merge_time_ms)),
            final_sort_time_ms: mean_optional_f64(values.iter().map(|metrics| metrics.final_sort_time_ms)),
            materialization_time_ms: mean_optional_f64(values.iter().map(|metrics| metrics.materialization_time_ms)),
        }
    }
}

pub(super) fn bounded_status(memory_policy: NativeTpchMemoryPolicy, row: &NativeTpchBenchmarkRow) -> &'static str {
    let accepted =
        row.planning_outcome == NativeDataFusionPlanningOutcome::AcceptedReplaced && !row.rejected_before_execution;
    let planning_failed = row.unsupported_category.as_deref() == Some("planning")
        || row.reason_code.as_deref() == Some("physical_planning_failed");
    match (memory_policy, accepted, planning_failed, row.parity_status) {
        (NativeTpchMemoryPolicy::Bounded { .. }, false, true, _) => "bounded_planning_failed",
        (NativeTpchMemoryPolicy::Unbounded, false, true, _) => "unbounded_planning_failed",
        (NativeTpchMemoryPolicy::Bounded { .. }, false, false, _) => "bounded_rejected",
        (NativeTpchMemoryPolicy::Unbounded, false, false, _) => "unbounded_rejected",
        (NativeTpchMemoryPolicy::Bounded { .. }, true, _, validation::ParityStatus::Matched) => {
            "bounded_accepted_parity_matched"
        },
        (NativeTpchMemoryPolicy::Bounded { .. }, true, _, validation::ParityStatus::Mismatched) => {
            "bounded_accepted_parity_mismatched"
        },
        (NativeTpchMemoryPolicy::Bounded { .. }, true, _, validation::ParityStatus::CpuFailed) => {
            "bounded_accepted_cpu_failed"
        },
        (NativeTpchMemoryPolicy::Bounded { .. }, true, _, validation::ParityStatus::NativeFailed) => {
            "bounded_accepted_native_failed"
        },
        (NativeTpchMemoryPolicy::Bounded { .. }, true, _, validation::ParityStatus::NotRun) => {
            "bounded_accepted_not_run"
        },
        (NativeTpchMemoryPolicy::Unbounded, true, _, validation::ParityStatus::Matched) => {
            "unbounded_accepted_parity_matched"
        },
        (NativeTpchMemoryPolicy::Unbounded, true, _, validation::ParityStatus::Mismatched) => {
            "unbounded_accepted_parity_mismatched"
        },
        (NativeTpchMemoryPolicy::Unbounded, true, _, validation::ParityStatus::CpuFailed) => {
            "unbounded_accepted_cpu_failed"
        },
        (NativeTpchMemoryPolicy::Unbounded, true, _, validation::ParityStatus::NativeFailed) => {
            "unbounded_accepted_native_failed"
        },
        (NativeTpchMemoryPolicy::Unbounded, true, _, validation::ParityStatus::NotRun) => "unbounded_accepted_not_run",
    }
}

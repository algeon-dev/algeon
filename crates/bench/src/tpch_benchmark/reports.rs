use std::collections::{BTreeMap, BTreeSet};

use algeon_datafusion::planner::NativeCapabilityCompiledGraphDiagnostics;
use query_engine::contract_report::CapabilityDecisionRow;

use super::*;

mod formatting;

use formatting::*;

/// Value of the leading `schema` column of every benchmark TSV row.
pub(super) const NATIVE_TPCH_BENCHMARK_TSV_SCHEMA: &str = "bench.native_tpch_benchmark_tsv.v2";
pub(super) const NATIVE_TPCH_BENCHMARK_TSV_HEADER: &str = "schema\tquery_id\tplanning_outcome\tbounded_status\treason_code\tunsupported_category\tcapability_decision_rows\tcapability_compiled_graphs\tnative_candidate_shape_family\tpost_native_execution_path\trejected_before_execution\tcpu_runtime_started\tnative_runtime_started\truntime_outcome\tcpu_failure\tcpu_failure_code\tcpu_failure_kind\tcpu_failure_status\tcpu_failure_operation\tcpu_failure_facts\truntime_failure\truntime_failure_code\truntime_failure_kind\truntime_failure_status\truntime_failure_operation\truntime_failure_facts\tparity_status\tcpu_wall_ms\tnative_wall_ms\tspeedup_vs_cpu\tcpu_output_batches\tnative_output_batches\tcpu_output_rows\tnative_output_rows\tcpu_output_bytes\tnative_output_bytes\tcpu_output_fingerprint\tnative_output_fingerprint\toutput_fingerprint\tgpu_resident_input_handoffs\thost_to_device_uploads\thost_to_device_uploaded_rows\thost_to_device_uploaded_bytes\tdevice_to_host_materializations\tdevice_to_host_materialized_rows\tdevice_to_host_materialized_bytes\tdevice_to_device_copies\tdevice_to_device_copied_bytes\tparquet_files\tparquet_bytes_known\trow_groups_selected\tread_columns\tresident_dataframe_peak_bytes\tallocator_high_water_bytes\tmax_task_observed_peak_bytes\tmax_task_retained_bytes\tgraph_runtime_rows\ttask_operator_rows\ttask_runtime_rows\tdynamic_filter_produced_count\tdynamic_filter_skipped_count\tdynamic_filter_applied_count\tdynamic_filter_row_groups_pruned\tdynamic_filter_source_bytes_pruned\tdynamic_filter_reader_predicate_rows_before\tdynamic_filter_reader_predicate_rows_after\tdynamic_filter_reader_predicate_rows_filtered\tdynamic_filter_post_decode_rows_before\tdynamic_filter_post_decode_rows_after\tdynamic_filter_post_decode_rows_filtered\tdynamic_filter_source_post_decode_rows_filtered\tdynamic_filter_filter_post_decode_rows_filtered\tdynamic_filter_join_probe_post_decode_rows_filtered\tdynamic_filter_post_decode_reduction_stage\tdynamic_filter_measured_reduction_status\tdynamic_filter_work_avoidance_status\tdynamic_filter_runtime_rows\ttask_high_water_rows\tpeak_memory_operator_class\tpeak_memory_operator_class_bytes\truntime_operator_class\truntime_operator_class_ms\tsource_predicate_pushdown_status\tpartitioned_join_count\tjoin_partitioning_status\taggregate_merge_status\tsource_chunks_planned\tsource_chunks_attempted\tsource_chunks_committed\tsource_chunks_retried\tsource_read_prefetches_started\tsource_read_prefetches_completed\tsource_read_prefetches_skipped\tsource_read_prefetches_abandoned\trow_groups_planned\trow_groups_committed\tpartial_aggregate_count\tpartial_aggregate_input_rows\tpartial_aggregate_rows\tpartial_aggregate_retained_bytes\tfinal_merge_attempts\tfinal_merge_levels\tfinal_merge_max_fan_in\taggregate_libcudf_runtime_probe_count\taggregate_libcudf_runtime_probe_elapsed_ns\taggregate_libcudf_runtime_probe_input_rows\taggregate_exact_cardinality_probe_count\taggregate_exact_cardinality_probe_elapsed_ns\taggregate_exact_cardinality_probe_input_rows\taggregate_prepared_sort_groupby_elapsed_ns\taggregate_encoded_key_partial_elapsed_ns\tretry_count\tretry_resource_exhausted_count\thash_join_build_count\tfiltered_join_build_count\tone_shot_join_count\thash_join_hash_table_bytes\tfiltered_join_hash_table_bytes\tfiltered_join_probe_state_bytes\tmixed_join_size_data_bytes\tone_shot_join_index_bytes\tone_shot_join_materialization_bytes\tone_shot_join_output_bytes\tjoin_gather_map_bytes\taggregate_strategy_rows\tjoin_strategy_rows\tparquet_read_time_ms\tsource_read_time_ms\tsource_read_queue_wait_time_ms\toperator_time_ms\tchunk_operator_time_ms\tfinal_merge_time_ms\tfinal_sort_time_ms\tmaterialization_time_ms";
pub(super) const NATIVE_TPCH_BENCHMARK_TSV_COLUMNS: &[&str] = &[
    "schema",
    "query_id",
    "planning_outcome",
    "bounded_status",
    "reason_code",
    "unsupported_category",
    "capability_decision_rows",
    "capability_compiled_graphs",
    "native_candidate_shape_family",
    "post_native_execution_path",
    "rejected_before_execution",
    "cpu_runtime_started",
    "native_runtime_started",
    "runtime_outcome",
    "cpu_failure",
    "cpu_failure_code",
    "cpu_failure_kind",
    "cpu_failure_status",
    "cpu_failure_operation",
    "cpu_failure_facts",
    "runtime_failure",
    "runtime_failure_code",
    "runtime_failure_kind",
    "runtime_failure_status",
    "runtime_failure_operation",
    "runtime_failure_facts",
    "parity_status",
    "cpu_wall_ms",
    "native_wall_ms",
    "speedup_vs_cpu",
    "cpu_output_batches",
    "native_output_batches",
    "cpu_output_rows",
    "native_output_rows",
    "cpu_output_bytes",
    "native_output_bytes",
    "cpu_output_fingerprint",
    "native_output_fingerprint",
    "output_fingerprint",
    "gpu_resident_input_handoffs",
    "host_to_device_uploads",
    "host_to_device_uploaded_rows",
    "host_to_device_uploaded_bytes",
    "device_to_host_materializations",
    "device_to_host_materialized_rows",
    "device_to_host_materialized_bytes",
    "device_to_device_copies",
    "device_to_device_copied_bytes",
    "parquet_files",
    "parquet_bytes_known",
    "row_groups_selected",
    "read_columns",
    "resident_dataframe_peak_bytes",
    "allocator_high_water_bytes",
    "max_task_observed_peak_bytes",
    "max_task_retained_bytes",
    "graph_runtime_rows",
    "task_operator_rows",
    "task_runtime_rows",
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
    "task_high_water_rows",
    "peak_memory_operator_class",
    "peak_memory_operator_class_bytes",
    "runtime_operator_class",
    "runtime_operator_class_ms",
    "source_predicate_pushdown_status",
    "partitioned_join_count",
    "join_partitioning_status",
    "aggregate_merge_status",
    "source_chunks_planned",
    "source_chunks_attempted",
    "source_chunks_committed",
    "source_chunks_retried",
    "source_read_prefetches_started",
    "source_read_prefetches_completed",
    "source_read_prefetches_skipped",
    "source_read_prefetches_abandoned",
    "row_groups_planned",
    "row_groups_committed",
    "partial_aggregate_count",
    "partial_aggregate_input_rows",
    "partial_aggregate_rows",
    "partial_aggregate_retained_bytes",
    "final_merge_attempts",
    "final_merge_levels",
    "final_merge_max_fan_in",
    "aggregate_libcudf_runtime_probe_count",
    "aggregate_libcudf_runtime_probe_elapsed_ns",
    "aggregate_libcudf_runtime_probe_input_rows",
    "aggregate_exact_cardinality_probe_count",
    "aggregate_exact_cardinality_probe_elapsed_ns",
    "aggregate_exact_cardinality_probe_input_rows",
    "aggregate_prepared_sort_groupby_elapsed_ns",
    "aggregate_encoded_key_partial_elapsed_ns",
    "retry_count",
    "retry_resource_exhausted_count",
    "hash_join_build_count",
    "filtered_join_build_count",
    "one_shot_join_count",
    "hash_join_hash_table_bytes",
    "filtered_join_hash_table_bytes",
    "filtered_join_probe_state_bytes",
    "mixed_join_size_data_bytes",
    "one_shot_join_index_bytes",
    "one_shot_join_materialization_bytes",
    "one_shot_join_output_bytes",
    "join_gather_map_bytes",
    "aggregate_strategy_rows",
    "join_strategy_rows",
    "parquet_read_time_ms",
    "source_read_time_ms",
    "source_read_queue_wait_time_ms",
    "operator_time_ms",
    "chunk_operator_time_ms",
    "final_merge_time_ms",
    "final_sort_time_ms",
    "materialization_time_ms",
];

#[derive(Debug, Clone)]
pub(super) struct NativeTpchBenchmarkReport {
    pub(super) metadata: TpchDatasetMetadata,
    pub(super) config: NativeTpchBenchmarkConfig,
    /// Devices the installed backend resolved for this run; empty when no
    /// backend was built (planning-only or test reports).
    pub(super) devices: Vec<DeviceProvenance>,
    pub(super) rows: Vec<NativeTpchBenchmarkRow>,
}

impl NativeTpchBenchmarkReport {
    pub(super) fn to_tsv(&self) -> String {
        let mut output = String::new();
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
        write_native_execution_policy_metadata(&mut output, self.config.native_execution);
        push_line(
            &mut output,
            format_args!("# tpch_benchmark_native_memory_policy={}", self.config.native_memory_policy.as_str()),
        );
        write_native_memory_policy_metadata(
            &mut output,
            self.config.native_memory_policy,
            self.config.max_source_reads_in_flight,
            self.devices.first().map(|device| device.managed_capacity_bytes),
        );
        // A loss flag cannot say how many detail rows were lost, only that the
        // rendered rows are not the full history.
        for row in self.rows.iter().filter(|row| row.metrics.detail_capture_lost) {
            push_line(
                &mut output,
                format_args!(
                    "# metric_detail_capture query_id={} status=incomplete lost_records=unknown",
                    row.query_id
                ),
            );
        }
        debug_assert_eq!(NATIVE_TPCH_BENCHMARK_TSV_COLUMNS.join("\t"), NATIVE_TPCH_BENCHMARK_TSV_HEADER);
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
    pub(super) planning_outcome: Option<CandidateOutcomeKind>,
    pub(super) capability_decision_rows: Vec<CapabilityDecisionRow>,
    pub(super) capability_compiled_graphs: Vec<NativeCapabilityCompiledGraphDiagnostics>,
    pub(super) reason_code: Option<String>,
    pub(super) unsupported_category: Option<String>,
    pub(super) native_candidate_shape_family: Option<String>,
    pub(super) final_disposition: Option<FinalPlanDisposition>,
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
    pub(super) source_predicate_pushdown_status: String,
    pub(super) metrics: NativeDataFusionExecutionMetrics,
    pub(super) timing_metrics: NativeTpchTimingMetrics,
}

impl NativeTpchBenchmarkRow {
    pub(super) fn not_selected(query_id: &str, planning_row: NativeDataFusionPlanningRow) -> Self {
        let source_predicate_pushdown_status = source_predicate_pushdown_status(&planning_row.source_diagnostics);
        Self {
            query_id: query_id.to_string(),
            planning_outcome: planning_row.outcome,
            capability_decision_rows: planning_row.capability_decision_rows,
            capability_compiled_graphs: planning_row.capability_compiled_graphs,
            reason_code: planning_row.reason_code,
            unsupported_category: planning_row.unsupported_category,
            native_candidate_shape_family: planning_row.native_candidate_shape_family,
            final_disposition: planning_row.final_disposition,
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
            source_predicate_pushdown_status,
            metrics: NativeDataFusionExecutionMetrics::default(),
            timing_metrics: NativeTpchTimingMetrics::default(),
        }
    }

    pub(super) fn planning_failure(query_id: &str, error: DataFusionError) -> Self {
        let details = execution::FailureDetails::from_datafusion_error(&error);
        Self {
            query_id: query_id.to_string(),
            planning_outcome: None,
            capability_decision_rows: Vec::new(),
            capability_compiled_graphs: Vec::new(),
            reason_code: Some("physical_planning_failed".to_string()),
            unsupported_category: Some("planning".to_string()),
            native_candidate_shape_family: None,
            final_disposition: None,
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
            source_predicate_pushdown_status: "planning_failed".to_string(),
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
            Err(error) => (None, Some(stable_tsv_field(&error.message)), *error.details, error.runtime_started),
        };
        let (native, native_failure, runtime_failure_details, native_runtime_started) = match native {
            Ok(run) => (Some(run), None, execution::FailureDetails::default(), true),
            Err(error) => (None, Some(stable_tsv_field(&error.message)), *error.details, error.runtime_started),
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
        let source_predicate_pushdown_status = source_predicate_pushdown_status(&planning_row.source_diagnostics);
        Self {
            query_id: query_id.to_string(),
            planning_outcome: planning_row.outcome,
            capability_decision_rows: planning_row.capability_decision_rows,
            capability_compiled_graphs: planning_row.capability_compiled_graphs,
            reason_code: planning_row.reason_code,
            unsupported_category: planning_row.unsupported_category,
            native_candidate_shape_family: planning_row.native_candidate_shape_family,
            final_disposition: planning_row.final_disposition,
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
            source_predicate_pushdown_status,
            metrics,
            timing_metrics,
        }
    }

    fn to_tsv_line(&self) -> String {
        self.fields().join("\t")
    }

    fn fields(&self) -> Vec<String> {
        let task_high_water_summaries = task_high_water_summaries(&self.metrics);
        let peak_memory = dominant_peak_memory_class(&task_high_water_summaries);
        let runtime = dominant_runtime_operator_class(self, &task_high_water_summaries);
        let fields = [
            NATIVE_TPCH_BENCHMARK_TSV_SCHEMA.to_string(),
            self.query_id.clone(),
            self.planning_outcome.map_or("none", CandidateOutcomeKind::as_str).to_string(),
            bounded_status(self).to_string(),
            optional_text(self.reason_code.as_deref(), "none"),
            optional_text(self.unsupported_category.as_deref(), "none"),
            format_capability_decision_rows(&self.capability_decision_rows),
            format_capability_compiled_graphs(&self.capability_compiled_graphs),
            optional_text(self.native_candidate_shape_family.as_deref(), "none"),
            self.final_disposition.map_or("none", FinalPlanDisposition::as_str).to_string(),
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
            optional_usize_text_none(self.metrics.max_task_observed_peak_bytes),
            optional_usize_text_none(self.metrics.max_task_retained_bytes),
            format_graph_runtime_rows(&self.metrics.graph_runtime_rows),
            format_task_operator_rows(&self.metrics.task_operator_rows),
            format_task_runtime_rows(&self.metrics.task_runtime_rows),
            self.metrics.dynamic_filter_produced_count.to_string(),
            self.metrics.dynamic_filter_skipped_count.to_string(),
            self.metrics.dynamic_filter_applied_count.to_string(),
            self.metrics.dynamic_filter_row_groups_pruned.to_string(),
            self.metrics.dynamic_filter_source_bytes_pruned.to_string(),
            self.metrics.dynamic_filter_reader_predicate_rows_before.to_string(),
            self.metrics.dynamic_filter_reader_predicate_rows_after.to_string(),
            self.metrics.dynamic_filter_reader_predicate_rows_filtered.to_string(),
            self.metrics.dynamic_filter_post_decode_rows_before.to_string(),
            self.metrics.dynamic_filter_post_decode_rows_after.to_string(),
            self.metrics.dynamic_filter_post_decode_rows_filtered.to_string(),
            self.metrics.dynamic_filter_source_post_decode_rows_filtered.to_string(),
            self.metrics.dynamic_filter_filter_post_decode_rows_filtered.to_string(),
            self.metrics.dynamic_filter_join_probe_post_decode_rows_filtered.to_string(),
            self.metrics.dynamic_filter_post_decode_reduction_stage().as_str().to_string(),
            self.metrics.dynamic_filter_measured_reduction_status().as_str().to_string(),
            self.metrics.dynamic_filter_work_avoidance_status().as_str().to_string(),
            format_dynamic_filter_runtime_rows(&self.metrics.dynamic_filter_runtime_rows),
            format_task_high_water_rows(&task_high_water_summaries),
            peak_memory.0.to_string(),
            peak_memory.1.to_string(),
            runtime.0.to_string(),
            optional_f64_text(runtime.1),
            self.source_predicate_pushdown_status.clone(),
            partitioned_join_count(&self.metrics).to_string(),
            join_partitioning_status(&self.metrics).to_string(),
            aggregate_merge_status(&self.metrics).to_string(),
            self.metrics.source_chunks_planned.to_string(),
            self.metrics.source_chunks_attempted.to_string(),
            self.metrics.source_chunks_committed.to_string(),
            self.metrics.source_chunks_retried.to_string(),
            self.metrics.source_read_prefetches_started.to_string(),
            self.metrics.source_read_prefetches_completed.to_string(),
            self.metrics.source_read_prefetches_skipped.to_string(),
            self.metrics.source_read_prefetches_abandoned.to_string(),
            self.metrics.row_groups_planned.to_string(),
            self.metrics.row_groups_committed.to_string(),
            self.metrics.partial_aggregate_count.to_string(),
            self.metrics.partial_aggregate_input_rows.to_string(),
            self.metrics.partial_aggregate_rows.to_string(),
            self.metrics.partial_aggregate_retained_bytes.to_string(),
            self.metrics.final_merge_attempts.to_string(),
            self.metrics.final_merge_levels.to_string(),
            self.metrics.final_merge_max_fan_in.to_string(),
            self.metrics.aggregate_libcudf_runtime_probe_count().to_string(),
            self.metrics.aggregate_libcudf_runtime_probe_elapsed_ns().to_string(),
            self.metrics.aggregate_libcudf_runtime_probe_input_rows().to_string(),
            self.metrics.aggregate_exact_cardinality_probe_count().to_string(),
            self.metrics.aggregate_exact_cardinality_probe_elapsed_ns().to_string(),
            self.metrics.aggregate_exact_cardinality_probe_input_rows().to_string(),
            self.metrics.aggregate_prepared_sort_groupby_elapsed_ns().to_string(),
            self.metrics.aggregate_encoded_key_partial_elapsed_ns().to_string(),
            self.metrics.retry_count.to_string(),
            self.metrics.retry_resource_exhausted_count.to_string(),
            self.metrics.hash_join_build_count.to_string(),
            self.metrics.filtered_join_build_count.to_string(),
            self.metrics.one_shot_join_count.to_string(),
            self.metrics.hash_join_hash_table_bytes.to_string(),
            self.metrics.filtered_join_hash_table_bytes.to_string(),
            self.metrics.filtered_join_probe_state_bytes.to_string(),
            self.metrics.mixed_join_size_data_bytes.to_string(),
            self.metrics.one_shot_join_index_bytes.to_string(),
            self.metrics.one_shot_join_materialization_bytes.to_string(),
            self.metrics.one_shot_join_output_bytes.to_string(),
            self.metrics.join_gather_map_bytes.to_string(),
            format_aggregate_strategy_rows(&self.metrics.aggregate_strategy_rows),
            format_join_strategy_rows(&self.metrics.join_strategy_rows),
            optional_f64_text(self.timing_metrics.parquet_read_time_ms),
            optional_f64_text(self.timing_metrics.source_read_time_ms),
            optional_f64_text(self.timing_metrics.source_read_queue_wait_time_ms),
            optional_f64_text(self.timing_metrics.operator_time_ms),
            optional_f64_text(self.timing_metrics.chunk_operator_time_ms),
            optional_f64_text(self.timing_metrics.final_merge_time_ms),
            optional_f64_text(self.timing_metrics.final_sort_time_ms),
            optional_f64_text(self.timing_metrics.materialization_time_ms),
        ];
        fields.into_iter().collect()
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct TaskHighWaterSummary {
    task_count: usize,
    max_observed_peak_bytes: Option<usize>,
    max_retained_bytes: Option<usize>,
}

impl TaskHighWaterSummary {
    fn merge_operator_row(&mut self, row: &NativeTaskOperatorMetricRow) {
        self.task_count = self.task_count.saturating_add(row.task_count);
        self.max_observed_peak_bytes = self.max_observed_peak_bytes.max(row.max_observed_peak_bytes);
        self.max_retained_bytes = self.max_retained_bytes.max(row.max_retained_bytes);
    }

    fn merge_runtime_row(&mut self, row: &NativeTaskRuntimeMetricRow) {
        self.max_observed_peak_bytes = self.max_observed_peak_bytes.max(row.max_observed_peak_bytes);
        self.max_retained_bytes = self.max_retained_bytes.max(row.max_retained_bytes);
    }

    fn merge_synthetic_high_water(&mut self, task_count: usize, bytes: Option<usize>) {
        self.task_count = self.task_count.saturating_add(task_count);
        if let Some(bytes) = bytes {
            self.max_observed_peak_bytes = self.max_observed_peak_bytes.max(Some(bytes));
            self.max_retained_bytes = self.max_retained_bytes.max(Some(bytes));
        }
    }

    fn to_stable_kv_row(self, class: &str) -> String {
        format!(
            "operator_class={},task_count={},max_observed_peak_bytes={},max_retained_bytes={}",
            class,
            self.task_count,
            optional_usize_text_none(self.max_observed_peak_bytes),
            optional_usize_text_none(self.max_retained_bytes),
        )
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
            summaries.entry(class).or_default().merge_runtime_row(row);
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

fn format_task_high_water_rows(summaries: &BTreeMap<&'static str, TaskHighWaterSummary>) -> String {
    TASK_HIGH_WATER_CLASSES
        .iter()
        .copied()
        .map(|class| summaries.get(class).copied().unwrap_or_default().to_stable_kv_row(class))
        .collect::<Vec<_>>()
        .join("|")
}

fn dominant_peak_memory_class(summaries: &BTreeMap<&'static str, TaskHighWaterSummary>) -> (&'static str, usize) {
    dominant_task_class_by(summaries, |summary| summary.max_observed_peak_bytes.unwrap_or(0))
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
    row: &NativeTpchBenchmarkRow,
    summaries: &BTreeMap<&'static str, TaskHighWaterSummary>,
) -> (&'static str, Option<f64>) {
    let operator_class = dominant_operator_runtime_class(row, summaries);
    [
        ("source", row.timing_metrics.source_read_time_ms),
        (operator_class, row.timing_metrics.operator_time_ms),
        ("final_merge", row.timing_metrics.final_merge_time_ms),
        ("final_sort", row.timing_metrics.final_sort_time_ms),
        ("materialization", row.timing_metrics.materialization_time_ms),
    ]
    .into_iter()
    .filter_map(|(class, value)| value.map(|value| (class, value)))
    .max_by(|left, right| left.1.partial_cmp(&right.1).unwrap_or(std::cmp::Ordering::Equal))
    .map(|(class, value)| (class, Some(value)))
    .unwrap_or(("not_run", None))
}

fn dominant_operator_runtime_class(
    row: &NativeTpchBenchmarkRow,
    summaries: &BTreeMap<&'static str, TaskHighWaterSummary>,
) -> &'static str {
    let join_tasks = summaries.get("join").copied().unwrap_or_default().task_count;
    let aggregate_tasks = summaries.get("aggregate").copied().unwrap_or_default().task_count;
    match (join_tasks > 0 || !row.metrics.join_strategy_rows.is_empty(), aggregate_tasks > 0) {
        (true, true) => "join_aggregate",
        (true, false) => "join",
        (false, true) => "aggregate",
        (false, false) => "native_operator",
    }
}

fn source_predicate_pushdown_status(diagnostics: &[NativeSourceDiagnosticSummary]) -> String {
    if diagnostics.is_empty() {
        return "no_sources".to_string();
    }
    let mut dispositions =
        diagnostics.iter().map(|diagnostic| diagnostic.predicate_disposition.as_str()).collect::<BTreeSet<_>>();
    if dispositions.contains("rejected") {
        return "rejected".to_string();
    }
    if dispositions.contains("reader_pushdown") {
        return if dispositions.len() == 1 { "reader_pushdown" } else { "mixed_reader_pushdown" }.to_string();
    }
    if dispositions.contains("native_post_scan_only") {
        return if dispositions.len() == 1 { "native_post_scan_only" } else { "mixed_native_post_scan_only" }
            .to_string();
    }
    if dispositions.len() == 1 {
        return dispositions.pop_first().unwrap_or("unknown").to_string();
    }
    dispositions.into_iter().collect::<Vec<_>>().join(",")
}

fn partitioned_join_count(metrics: &NativeDataFusionExecutionMetrics) -> usize {
    metrics.join_strategy_rows.iter().map(|row| row.partitioned_hash_join_count).sum()
}

fn partitioned_join_site_count(metrics: &NativeDataFusionExecutionMetrics) -> usize {
    metrics.join_strategy_rows.iter().filter(|row| row.partitioned_hash_join_count > 0).count()
}

fn join_partitioning_status(metrics: &NativeDataFusionExecutionMetrics) -> &'static str {
    let join_count = metrics.join_strategy_rows.len();
    let partitioned_sites = partitioned_join_site_count(metrics);
    match (join_count, partitioned_sites) {
        (0, _) => "no_joins",
        (_, 0) => "not_partitioned",
        (joins, partitioned) if joins == partitioned => "partitioned",
        _ => "mixed",
    }
}

fn aggregate_merge_status(metrics: &NativeDataFusionExecutionMetrics) -> &'static str {
    if metrics.final_merge_attempts == 0 {
        return "not_applicable";
    }
    if metrics.final_merge_max_fan_in > 0 && metrics.partial_aggregate_count > metrics.final_merge_max_fan_in {
        return "thresholded";
    }
    if metrics.final_merge_levels > 1 {
        return "thresholded";
    }
    "single_level"
}

#[derive(Debug, Clone, Default)]
pub(super) struct NativeTpchTimingMetrics {
    pub(super) parquet_read_time_ms: Option<f64>,
    pub(super) source_read_time_ms: Option<f64>,
    pub(super) source_read_queue_wait_time_ms: Option<f64>,
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
            operator_time_ms: mean_optional_f64(values.iter().map(|metrics| metrics.operator_time_ms)),
            chunk_operator_time_ms: mean_optional_f64(values.iter().map(|metrics| metrics.chunk_operator_time_ms)),
            final_merge_time_ms: mean_optional_f64(values.iter().map(|metrics| metrics.final_merge_time_ms)),
            final_sort_time_ms: mean_optional_f64(values.iter().map(|metrics| metrics.final_sort_time_ms)),
            materialization_time_ms: mean_optional_f64(values.iter().map(|metrics| metrics.materialization_time_ms)),
        }
    }
}

pub(super) fn bounded_status(row: &NativeTpchBenchmarkRow) -> &'static str {
    let accepted = row.planning_outcome == Some(CandidateOutcomeKind::Selected) && !row.rejected_before_execution;
    let planning_failed = row.unsupported_category.as_deref() == Some("planning")
        || row.reason_code.as_deref() == Some("physical_planning_failed");
    match (accepted, planning_failed, row.parity_status) {
        (false, true, _) => "bounded_planning_failed",
        (false, false, _) => "bounded_rejected",
        (true, _, validation::ParityStatus::Matched) => "bounded_accepted_parity_matched",
        (true, _, validation::ParityStatus::Mismatched) => "bounded_accepted_parity_mismatched",
        (true, _, validation::ParityStatus::CpuFailed) => "bounded_accepted_cpu_failed",
        (true, _, validation::ParityStatus::NativeFailed) => "bounded_accepted_native_failed",
        (true, _, validation::ParityStatus::NotRun) => "bounded_accepted_not_run",
    }
}

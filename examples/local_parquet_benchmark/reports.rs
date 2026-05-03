use super::*;

#[derive(Debug)]
pub(super) struct DirectParquetBenchmarkReport {
    pub(super) parquet_file_bytes: u64,
    pub(super) parquet_row_groups: usize,
    pub(super) input_rows: usize,
    pub(super) file_count: usize,
    pub(super) measure_columns: usize,
    pub(super) batch_size: usize,
    pub(super) row_group_size: usize,
    pub(super) group_count: usize,
    pub(super) warmup_iterations: usize,
    pub(super) iterations: usize,
    pub(super) limit: usize,
    pub(super) minimum_native_gpu_speedup: f64,
    pub(super) benchmark_paths: BenchmarkPathSet,
    pub(super) compression: BenchmarkCompression,
    pub(super) native_plan_shape: BenchmarkPlanShape,
    pub(super) expected_outcome: BenchmarkExpectedOutcome,
    pub(super) measured_run_order: String,
    pub(super) cpu_comparable: TimedBenchmarkRun,
    pub(super) cpu_default: TimedBenchmarkRun,
    pub(super) gpu: Option<TimedBenchmarkRun>,
    pub(super) datafusion_native_exec: TimedBenchmarkRun,
    pub(super) manual_native_lowering: TimedBenchmarkRun,
    pub(super) cpu_comparable_samples: Vec<Duration>,
    pub(super) cpu_default_samples: Vec<Duration>,
    pub(super) gpu_samples: Vec<Duration>,
    pub(super) datafusion_native_exec_samples: Vec<Duration>,
    pub(super) manual_native_lowering_samples: Vec<Duration>,
    pub(super) gpu_metric_samples: Vec<LargerIslandBenchmarkMetrics>,
    pub(super) datafusion_native_exec_metric_samples: Vec<LargerIslandBenchmarkMetrics>,
    pub(super) manual_native_lowering_metric_samples: Vec<LargerIslandBenchmarkMetrics>,
    pub(super) final_cpu_comparable: CollectMeasurement,
    pub(super) final_cpu_default: CollectMeasurement,
    pub(super) final_gpu: Option<GpuCollectMeasurement>,
    pub(super) final_datafusion_native_exec: GpuCollectMeasurement,
    pub(super) final_manual_native_lowering: GpuCollectMeasurement,
    pub(super) source_diagnostic: String,
}

impl DirectParquetBenchmarkReport {
    pub(super) fn to_text(&self) -> String {
        let comparable_speedup = self.gpu.as_ref().map(|gpu| speedup(self.cpu_comparable.mean, gpu.mean));
        let default_speedup = self.gpu.as_ref().map(|gpu| speedup(self.cpu_default.mean, gpu.mean));
        let datafusion_native_comparable_speedup = speedup(self.cpu_comparable.mean, self.datafusion_native_exec.mean);
        let datafusion_native_default_speedup = speedup(self.cpu_default.mean, self.datafusion_native_exec.mean);
        let manual_native_comparable_speedup = speedup(self.cpu_comparable.mean, self.manual_native_lowering.mean);
        let manual_native_default_speedup = speedup(self.cpu_default.mean, self.manual_native_lowering.mean);
        let native_metrics = &self.final_datafusion_native_exec.metrics;
        let mut text = String::new();

        push_line(&mut text, format_args!("case=gpu_friendly_local_parquet"));
        push_line(&mut text, format_args!("parquet_file_bytes={}", self.parquet_file_bytes));
        push_line(&mut text, format_args!("parquet_row_groups={}", self.parquet_row_groups));
        push_line(&mut text, format_args!("input_rows={}", self.input_rows));
        push_line(&mut text, format_args!("configured_parquet_files={}", self.file_count));
        push_line(&mut text, format_args!("measure_columns={}", self.measure_columns));
        push_line(&mut text, format_args!("batch_size={}", self.batch_size));
        push_line(&mut text, format_args!("row_group_size={}", self.row_group_size));
        push_line(&mut text, format_args!("group_count={}", self.group_count));
        push_line(&mut text, format_args!("compression={}", self.compression.name()));
        push_line(&mut text, format_args!("warmups={}", self.warmup_iterations));
        push_line(&mut text, format_args!("iterations={}", self.iterations));
        push_line(&mut text, format_args!("minimum_native_gpu_speedup={:.3}", self.minimum_native_gpu_speedup));
        push_line(&mut text, format_args!("benchmark_paths={}", self.benchmark_paths.name()));
        push_line(&mut text, format_args!("native_plan_shape={}", self.native_plan_shape.name()));
        push_line(&mut text, format_args!("expected_outcome={}", self.expected_outcome.name()));
        push_line(&mut text, format_args!("phase_c_decision_matrix_cases={}", phase_c_decision_matrix().len()));
        push_line(&mut text, format_args!("phase_b_runtime_gate=pass"));
        push_line(
            &mut text,
            format_args!("phase_c_memory_control_policy={}", native_metrics.phase_c_memory_control_policy),
        );
        push_line(&mut text, format_args!("phase_c_runtime_gate={}", native_metrics.phase_c_runtime_gate));
        push_line(&mut text, format_args!("measured_run_order={}", self.measured_run_order));
        push_line(&mut text, format_args!("cpu_comparable_wall_ms={:.3}", duration_ms(self.cpu_comparable.mean)));
        push_line(&mut text, format_args!("cpu_wall_ms={:.3}", duration_ms(self.cpu_comparable.mean)));
        push_line(&mut text, format_args!("cpu_default_wall_ms={:.3}", duration_ms(self.cpu_default.mean)));
        push_line(&mut text, format_args!("cpu_datafusion_wall_ms={:.3}", duration_ms(self.cpu_default.mean)));
        if let Some(gpu) = &self.gpu {
            push_line(&mut text, format_args!("gpu_wall_ms={:.3}", duration_ms(gpu.mean)));
        } else {
            push_line(&mut text, format_args!("gpu_wall_ms=not_run"));
        }
        push_line(
            &mut text,
            format_args!("datafusion_native_exec_wall_ms={:.3}", duration_ms(self.datafusion_native_exec.mean)),
        );
        push_line(
            &mut text,
            format_args!("manual_native_lowering_wall_ms={:.3}", duration_ms(self.manual_native_lowering.mean)),
        );
        push_line(&mut text, format_args!("gpu_speedup_vs_cpu_comparable={}", optional_f64_text(comparable_speedup)));
        push_line(&mut text, format_args!("gpu_speedup={}", optional_f64_text(comparable_speedup)));
        push_line(&mut text, format_args!("gpu_speedup_vs_cpu_default={}", optional_f64_text(default_speedup)));
        push_line(
            &mut text,
            format_args!("datafusion_native_exec_speedup_vs_cpu_comparable={datafusion_native_comparable_speedup:.3}"),
        );
        push_line(&mut text, format_args!("datafusion_native_exec_speedup={datafusion_native_comparable_speedup:.3}"));
        push_line(
            &mut text,
            format_args!("datafusion_native_exec_speedup_vs_cpu_datafusion={datafusion_native_default_speedup:.3}"),
        );
        push_line(
            &mut text,
            format_args!("manual_native_lowering_speedup_vs_cpu_comparable={manual_native_comparable_speedup:.3}"),
        );
        push_line(&mut text, format_args!("manual_native_lowering_speedup={manual_native_comparable_speedup:.3}"));
        push_line(
            &mut text,
            format_args!("manual_native_lowering_speedup_vs_cpu_datafusion={manual_native_default_speedup:.3}"),
        );
        push_line(
            &mut text,
            format_args!("cpu_comparable_wall_ms_samples={}", duration_samples_csv(&self.cpu_comparable_samples)),
        );
        push_line(
            &mut text,
            format_args!("cpu_default_wall_ms_samples={}", duration_samples_csv(&self.cpu_default_samples)),
        );
        push_line(&mut text, format_args!("gpu_wall_ms_samples={}", optional_duration_samples_csv(&self.gpu_samples)));
        push_line(
            &mut text,
            format_args!(
                "datafusion_native_exec_wall_ms_samples={}",
                duration_samples_csv(&self.datafusion_native_exec_samples)
            ),
        );
        push_line(
            &mut text,
            format_args!(
                "manual_native_lowering_wall_ms_samples={}",
                duration_samples_csv(&self.manual_native_lowering_samples)
            ),
        );
        if let Some(final_gpu) = &self.final_gpu {
            push_line(&mut text, format_args!("output_rows={}", final_gpu.output_rows));
            push_line(&mut text, format_args!("output_batches={}", final_gpu.output_batches));
            push_line(&mut text, format_args!("output_bytes={}", final_gpu.output_bytes));
        } else {
            push_line(&mut text, format_args!("output_rows=not_run"));
            push_line(&mut text, format_args!("output_batches=not_run"));
            push_line(&mut text, format_args!("output_bytes=not_run"));
        }
        push_line(
            &mut text,
            format_args!("datafusion_native_exec_output_rows={}", self.final_datafusion_native_exec.output_rows),
        );
        push_line(
            &mut text,
            format_args!("datafusion_native_exec_output_batches={}", self.final_datafusion_native_exec.output_batches),
        );
        push_line(
            &mut text,
            format_args!("datafusion_native_exec_output_bytes={}", self.final_datafusion_native_exec.output_bytes),
        );
        push_line(
            &mut text,
            format_args!("manual_native_lowering_output_rows={}", self.final_manual_native_lowering.output_rows),
        );
        push_line(
            &mut text,
            format_args!("manual_native_lowering_output_batches={}", self.final_manual_native_lowering.output_batches),
        );
        push_line(
            &mut text,
            format_args!("manual_native_lowering_output_bytes={}", self.final_manual_native_lowering.output_bytes),
        );
        push_line(&mut text, format_args!("cpu_comparable_output_rows={}", self.final_cpu_comparable.output_rows));
        push_line(
            &mut text,
            format_args!("cpu_comparable_output_batches={}", self.final_cpu_comparable.output_batches),
        );
        push_line(&mut text, format_args!("cpu_comparable_output_bytes={}", self.final_cpu_comparable.output_bytes));
        push_line(&mut text, format_args!("cpu_default_output_rows={}", self.final_cpu_default.output_rows));
        push_line(&mut text, format_args!("cpu_default_output_batches={}", self.final_cpu_default.output_batches));
        push_line(&mut text, format_args!("cpu_default_output_bytes={}", self.final_cpu_default.output_bytes));
        push_line(&mut text, format_args!("limit={}", self.limit));
        if let Some(final_gpu) = &self.final_gpu {
            let metrics = &final_gpu.metrics;
            let planning = &final_gpu.planning;
            push_line(&mut text, format_args!("parquet_read_ms={:.3}", duration_ms(metrics.parquet_read_time)));
            push_line(
                &mut text,
                format_args!(
                    "parquet_read_ms_mean={:.3}",
                    duration_ms(mean_metric_duration(&self.gpu_metric_samples, |metrics| metrics.parquet_read_time))
                ),
            );
            push_line(
                &mut text,
                format_args!(
                    "parquet_read_ms_samples={}",
                    metric_duration_samples_csv(&self.gpu_metric_samples, |metrics| metrics.parquet_read_time)
                ),
            );
            push_line(&mut text, format_args!("cudf_operator_ms={:.3}", duration_ms(metrics.cudf_operator_time)));
            push_line(
                &mut text,
                format_args!(
                    "cudf_operator_ms_mean={:.3}",
                    duration_ms(mean_metric_duration(&self.gpu_metric_samples, |metrics| metrics.cudf_operator_time))
                ),
            );
            push_line(
                &mut text,
                format_args!(
                    "cudf_operator_ms_samples={}",
                    metric_duration_samples_csv(&self.gpu_metric_samples, |metrics| metrics.cudf_operator_time)
                ),
            );
            push_line(
                &mut text,
                format_args!("record_batch_upload_ms={:.3}", duration_ms(metrics.record_batch_upload_time)),
            );
            push_line(
                &mut text,
                format_args!(
                    "record_batch_upload_ms_mean={:.3}",
                    duration_ms(mean_metric_duration(&self.gpu_metric_samples, |metrics| {
                        metrics.record_batch_upload_time
                    }))
                ),
            );
            push_line(
                &mut text,
                format_args!(
                    "record_batch_materialization_ms={:.3}",
                    duration_ms(metrics.record_batch_materialization_time)
                ),
            );
            push_line(
                &mut text,
                format_args!(
                    "record_batch_materialization_ms_mean={:.3}",
                    duration_ms(mean_metric_duration(&self.gpu_metric_samples, |metrics| {
                        metrics.record_batch_materialization_time
                    }))
                ),
            );
            push_line(&mut text, format_args!("host_to_device_uploads={}", metrics.host_to_device_uploads));
            push_line(&mut text, format_args!("host_to_device_uploaded_rows={}", metrics.host_to_device_uploaded_rows));
            push_line(
                &mut text,
                format_args!("device_to_host_materializations={}", metrics.device_to_host_materializations),
            );
            push_line(
                &mut text,
                format_args!("device_to_host_materialized_rows={}", metrics.device_to_host_materialized_rows),
            );
            push_line(&mut text, format_args!("parquet_files={}", metrics.parquet_files));
            push_line(&mut text, format_args!("parquet_bytes_known={}", metrics.parquet_bytes_known));
            push_line(&mut text, format_args!("row_groups_selected={}", metrics.row_groups_selected));
            push_line(&mut text, format_args!("read_column_count={}", metrics.read_columns));
            push_line(&mut text, format_args!("read_columns={}", final_gpu.read_columns.join(",")));
            push_line(&mut text, format_args!("fallback_candidate_count={}", planning.candidate_count));
            push_line(&mut text, format_args!("fallback_rewritten_count={}", planning.rewritten_count));
            push_line(&mut text, format_args!("fallback_rejected_count={}", planning.rejected_count));
            push_line(&mut text, format_args!("gpu_plan={}", final_gpu.plan_one_line));
        } else {
            push_line(&mut text, format_args!("gpu_direct_status=not_run"));
        }
        push_native_measurement_lines(
            &mut text,
            "datafusion_native_exec",
            &self.final_datafusion_native_exec,
            &self.datafusion_native_exec_metric_samples,
        );
        push_native_datafusion_planning_lines(
            &mut text,
            "datafusion_native_exec",
            self.final_datafusion_native_exec.native_datafusion_planning_report.as_ref(),
        );
        push_native_measurement_lines(
            &mut text,
            "manual_native_lowering",
            &self.final_manual_native_lowering,
            &self.manual_native_lowering_metric_samples,
        );
        push_line(&mut text, format_args!("source_diagnostic={}", self.source_diagnostic));
        text
    }
}

pub(super) fn push_native_measurement_lines(
    text: &mut String,
    prefix: &str,
    measurement: &GpuCollectMeasurement,
    metric_samples: &[LargerIslandBenchmarkMetrics],
) {
    let metrics = &measurement.metrics;
    let planning = &measurement.planning;
    push_line(text, format_args!("{prefix}_parquet_read_ms={:.3}", duration_ms(metrics.parquet_read_time)));
    push_line(
        text,
        format_args!(
            "{prefix}_parquet_read_ms_mean={:.3}",
            duration_ms(mean_metric_duration(metric_samples, |metrics| metrics.parquet_read_time))
        ),
    );
    push_line(
        text,
        format_args!(
            "{prefix}_parquet_read_ms_samples={}",
            metric_duration_samples_csv(metric_samples, |metrics| metrics.parquet_read_time)
        ),
    );
    push_line(text, format_args!("{prefix}_cudf_operator_ms={:.3}", duration_ms(metrics.cudf_operator_time)));
    push_line(
        text,
        format_args!(
            "{prefix}_cudf_operator_ms_mean={:.3}",
            duration_ms(mean_metric_duration(metric_samples, |metrics| metrics.cudf_operator_time))
        ),
    );
    push_line(
        text,
        format_args!(
            "{prefix}_cudf_operator_ms_samples={}",
            metric_duration_samples_csv(metric_samples, |metrics| metrics.cudf_operator_time)
        ),
    );
    push_line(
        text,
        format_args!(
            "{prefix}_record_batch_materialization_ms={:.3}",
            duration_ms(metrics.record_batch_materialization_time)
        ),
    );
    push_line(
        text,
        format_args!(
            "{prefix}_record_batch_materialization_ms_mean={:.3}",
            duration_ms(mean_metric_duration(metric_samples, |metrics| metrics.record_batch_materialization_time))
        ),
    );
    push_line(
        text,
        format_args!("{prefix}_peak_device_memory_bytes={}", optional_usize_text(metrics.peak_device_memory_bytes)),
    );
    push_line(
        text,
        format_args!(
            "{prefix}_resident_dataframe_peak_bytes={}",
            optional_usize_text(metrics.resident_dataframe_peak_bytes)
        ),
    );
    push_line(
        text,
        format_args!("{prefix}_allocator_high_water_bytes={}", optional_usize_text(metrics.allocator_high_water_bytes)),
    );
    push_line(text, format_args!("{prefix}_reservation_requested_bytes={}", metrics.reservation_requested_bytes));
    push_line(text, format_args!("{prefix}_reservation_granted_bytes={}", metrics.reservation_granted_bytes));
    push_line(text, format_args!("{prefix}_reservation_denied_count={}", metrics.reservation_denied_count));
    push_line(text, format_args!("{prefix}_source_chunks_planned={}", metrics.source_chunks_planned));
    push_line(text, format_args!("{prefix}_source_chunks_attempted={}", metrics.source_chunks_attempted));
    push_line(text, format_args!("{prefix}_source_chunks_committed={}", metrics.source_chunks_committed));
    push_line(text, format_args!("{prefix}_source_chunks_retried={}", metrics.source_chunks_retried));
    push_line(text, format_args!("{prefix}_row_groups_planned={}", metrics.row_groups_planned));
    push_line(text, format_args!("{prefix}_row_groups_committed={}", metrics.row_groups_committed));
    push_line(text, format_args!("{prefix}_partial_aggregate_count={}", metrics.partial_aggregate_count));
    push_line(text, format_args!("{prefix}_partial_aggregate_rows={}", metrics.partial_aggregate_rows));
    push_line(text, format_args!("{prefix}_retry_count={}", metrics.retry_count));
    push_line(text, format_args!("{prefix}_retry_resource_exhausted_count={}", metrics.retry_resource_exhausted_count));
    push_line(
        text,
        format_args!("{prefix}_retry_reservation_failure_count={}", metrics.retry_reservation_failure_count),
    );
    push_line(text, format_args!("{prefix}_final_merge_attempts={}", metrics.final_merge_attempts));
    push_line(text, format_args!("{prefix}_final_merge_levels={}", metrics.final_merge_levels));
    push_line(
        text,
        format_args!(
            "{prefix}_peak_device_memory_bytes_mean={}",
            optional_usize_text(mean_metric_optional_usize(metric_samples, |metrics| metrics.peak_device_memory_bytes))
        ),
    );
    push_line(
        text,
        format_args!(
            "{prefix}_peak_device_memory_bytes_samples={}",
            metric_optional_usize_samples_csv(metric_samples, |metrics| metrics.peak_device_memory_bytes)
        ),
    );
    push_line(text, format_args!("{prefix}_host_to_device_uploads={}", metrics.host_to_device_uploads));
    push_line(text, format_args!("{prefix}_host_to_device_uploaded_rows={}", metrics.host_to_device_uploaded_rows));
    push_line(
        text,
        format_args!("{prefix}_device_to_host_materializations={}", metrics.device_to_host_materializations),
    );
    push_line(
        text,
        format_args!("{prefix}_device_to_host_materialized_rows={}", metrics.device_to_host_materialized_rows),
    );
    push_line(text, format_args!("{prefix}_parquet_files={}", metrics.parquet_files));
    push_line(text, format_args!("{prefix}_parquet_bytes_known={}", metrics.parquet_bytes_known));
    push_line(text, format_args!("{prefix}_row_groups_selected={}", metrics.row_groups_selected));
    push_line(text, format_args!("{prefix}_read_column_count={}", metrics.read_columns));
    push_line(text, format_args!("{prefix}_read_columns={}", measurement.read_columns.join(",")));
    push_line(text, format_args!("{prefix}_fallback_candidate_count={}", planning.candidate_count));
    push_line(text, format_args!("{prefix}_fallback_rewritten_count={}", planning.rewritten_count));
    push_line(text, format_args!("{prefix}_fallback_rejected_count={}", planning.rejected_count));
    push_line(
        text,
        format_args!(
            "{prefix}_source_physical_plan={}",
            measurement.source_physical_plan_one_line.as_deref().unwrap_or("not_available")
        ),
    );
    push_line(
        text,
        format_args!("{prefix}_lowered_candidate_shapes={}", measurement.native_candidate_shapes.join(";")),
    );
    push_line(text, format_args!("{prefix}_plan={}", measurement.plan_one_line));
}

pub(super) fn push_native_datafusion_planning_lines(
    text: &mut String,
    prefix: &str,
    report: Option<&NativeDataFusionPlanningReport>,
) {
    let Some(report) = report else {
        push_line(text, format_args!("{prefix}_planning_report_status=not_available"));
        return;
    };
    push_line(text, format_args!("{prefix}_planning_report_status=available"));
    push_line(text, format_args!("{prefix}_planning_accepted_count={}", report.accepted_count()));
    push_line(text, format_args!("{prefix}_planning_rejected_count={}", report.rejected_count()));
    let Some(row) = report.rows.first() else {
        push_line(text, format_args!("{prefix}_planning_outcome=none"));
        return;
    };
    push_line(text, format_args!("{prefix}_planning_outcome={}", row.outcome.as_str()));
    push_line(text, format_args!("{prefix}_planning_reason_code={}", row.reason_code.as_deref().unwrap_or("none")));
    push_line(
        text,
        format_args!(
            "{prefix}_planning_unsupported_category={}",
            row.unsupported_category.as_deref().unwrap_or("none")
        ),
    );
    push_line(text, format_args!("{prefix}_planning_post_native_plan_root={}", row.post_native_plan_root));
    push_line(
        text,
        format_args!("{prefix}_planning_post_native_execution_path={}", row.post_native_execution_path.as_str()),
    );
    push_line(text, format_args!("{prefix}_planning_rejected_before_execution={}", row.rejected_before_execution));
    push_line(
        text,
        format_args!(
            "{prefix}_planning_source_diagnostics={}",
            native_source_diagnostics_text(&row.source_diagnostics)
        ),
    );
}

pub(super) fn native_source_diagnostics_text(diagnostics: &[datafusion_cudf::NativeSourceDiagnosticSummary]) -> String {
    if diagnostics.is_empty() {
        return "none".to_string();
    }

    diagnostics
        .iter()
        .map(|diagnostic| {
            format!(
                "format={},direct_device_read={},file_count={},ranged_file_count={},known_bytes={},selected_row_groups={},read_columns={},projected_output_columns={},pure_filter_columns={},predicate={},byte_range={},partition_columns={},zero_column_count={},schema_mapping={},rejected_features={}",
                diagnostic.format,
                diagnostic.direct_device_read,
                diagnostic.file_count,
                diagnostic.ranged_file_count,
                diagnostic.known_bytes.map_or_else(|| "unknown".to_string(), |value| value.to_string()),
                diagnostic.selected_row_groups.map_or_else(|| "all".to_string(), |value| value.to_string()),
                diagnostic.read_column_count,
                diagnostic.projected_output_column_count,
                diagnostic.pure_filter_column_count,
                diagnostic.predicate_disposition,
                diagnostic.byte_range_disposition,
                diagnostic.partition_column_disposition,
                diagnostic.zero_column_count_disposition,
                diagnostic.schema_mapping_disposition,
                if diagnostic.rejected_features.is_empty() {
                    "none".to_string()
                } else {
                    diagnostic.rejected_features.join(",")
                },
            )
        })
        .collect::<Vec<_>>()
        .join("|")
}

#[derive(Debug)]
pub(super) struct ExpectedPlanningRejectedReport {
    pub(super) parquet_file_bytes: u64,
    pub(super) parquet_row_groups: usize,
    pub(super) input_rows: usize,
    pub(super) file_count: usize,
    pub(super) compression: BenchmarkCompression,
    pub(super) native_plan_shape: BenchmarkPlanShape,
    pub(super) expected_outcome: BenchmarkExpectedOutcome,
    pub(super) final_plan_root: String,
    pub(super) source_diagnostic: String,
    pub(super) planning_report: NativeDataFusionPlanningReport,
}

impl ExpectedPlanningRejectedReport {
    pub(super) fn to_text(&self) -> String {
        let mut text = String::new();
        push_line(&mut text, format_args!("case=datafusion_native_exec_planning_rejected"));
        push_line(&mut text, format_args!("parquet_file_bytes={}", self.parquet_file_bytes));
        push_line(&mut text, format_args!("parquet_row_groups={}", self.parquet_row_groups));
        push_line(&mut text, format_args!("input_rows={}", self.input_rows));
        push_line(&mut text, format_args!("configured_parquet_files={}", self.file_count));
        push_line(&mut text, format_args!("compression={}", self.compression.name()));
        push_line(&mut text, format_args!("native_plan_shape={}", self.native_plan_shape.name()));
        push_line(&mut text, format_args!("expected_outcome={}", self.expected_outcome.name()));
        push_line(&mut text, format_args!("phase_c_runtime_gate=planning_rejected"));
        push_line(&mut text, format_args!("native_runtime_executed=false"));
        push_line(&mut text, format_args!("datafusion_native_exec_final_plan_root={}", self.final_plan_root));
        push_native_datafusion_planning_lines(&mut text, "datafusion_native_exec", Some(&self.planning_report));
        push_line(&mut text, format_args!("source_diagnostic={}", self.source_diagnostic));
        text
    }
}

#[derive(Debug)]
pub(super) struct ExpectedResourceFailureReport {
    pub(super) parquet_file_bytes: u64,
    pub(super) parquet_row_groups: usize,
    pub(super) input_rows: usize,
    pub(super) file_count: usize,
    pub(super) compression: BenchmarkCompression,
    pub(super) native_plan_shape: BenchmarkPlanShape,
    pub(super) expected_outcome: BenchmarkExpectedOutcome,
    pub(super) final_plan_root: String,
    pub(super) planning_report: NativeDataFusionPlanningReport,
    pub(super) native_candidate_shapes: Vec<String>,
    pub(super) source_physical_plan_one_line: String,
    pub(super) source_diagnostic: String,
    pub(super) error: AdapterRuntimeErrorReport,
}

impl ExpectedResourceFailureReport {
    pub(super) fn to_text(&self) -> String {
        let mut text = String::new();
        push_line(&mut text, format_args!("case=phase_c_expected_resource_failure"));
        push_line(&mut text, format_args!("parquet_file_bytes={}", self.parquet_file_bytes));
        push_line(&mut text, format_args!("parquet_row_groups={}", self.parquet_row_groups));
        push_line(&mut text, format_args!("input_rows={}", self.input_rows));
        push_line(&mut text, format_args!("configured_parquet_files={}", self.file_count));
        push_line(&mut text, format_args!("compression={}", self.compression.name()));
        push_line(&mut text, format_args!("native_plan_shape={}", self.native_plan_shape.name()));
        push_line(&mut text, format_args!("expected_outcome={}", self.expected_outcome.name()));
        push_line(&mut text, format_args!("phase_c_runtime_gate=expected_resource_failure"));
        push_line(&mut text, format_args!("no_silent_cpu_fallback=true"));
        push_line(&mut text, format_args!("native_runtime_executed=true"));
        push_line(&mut text, format_args!("datafusion_native_exec_final_plan_root={}", self.final_plan_root));
        push_native_datafusion_planning_lines(&mut text, "datafusion_native_exec", Some(&self.planning_report));
        push_line(&mut text, format_args!("native_error_code={}", self.error.code));
        push_line(&mut text, format_args!("native_error_kind={}", self.error.kind));
        push_line(&mut text, format_args!("native_error_status={}", self.error.status));
        push_line(&mut text, format_args!("native_error_operation={}", self.error.operation));
        push_line(&mut text, format_args!("native_error_facts={}", self.error.facts));
        push_line(
            &mut text,
            format_args!("native_lowered_candidate_shapes={}", self.native_candidate_shapes.join(";")),
        );
        push_line(&mut text, format_args!("native_default_sql_physical_plan={}", self.source_physical_plan_one_line));
        push_line(&mut text, format_args!("source_diagnostic={}", self.source_diagnostic));
        text
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct AdapterRuntimeErrorReport {
    pub(super) code: String,
    pub(super) kind: String,
    pub(super) status: String,
    pub(super) operation: &'static str,
    pub(super) facts: String,
}

impl AdapterRuntimeErrorReport {
    pub(super) fn from_adapter_error(error: &AdapterError) -> Self {
        Self {
            code: error.code().as_str().to_string(),
            kind: error.kind().as_str().to_string(),
            status: error.status().as_str().to_string(),
            operation: error.operation(),
            facts: adapter_error_facts_tsv(error),
        }
    }
}

pub(super) fn sanitize_report_value(value: &str) -> String {
    value.replace(['\t', '\n', '\r'], " ")
}

pub(super) fn tsv_escape(value: &str) -> String {
    sanitize_report_value(value)
}

pub(super) fn markdown_table_cell(value: &str) -> String {
    sanitize_report_value(value).replace('|', ";")
}

pub(super) fn push_line(text: &mut String, args: std::fmt::Arguments<'_>) {
    use std::fmt::Write as _;

    let _ = text.write_fmt(args);
    text.push('\n');
}

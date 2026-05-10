use super::*;

#[derive(Debug)]
pub(crate) struct DirectParquetBenchmarkReport {
    pub(crate) parquet_file_bytes: u64,
    pub(crate) parquet_row_groups: usize,
    pub(crate) input_rows: usize,
    pub(crate) file_count: usize,
    pub(crate) measure_columns: usize,
    pub(crate) batch_size: usize,
    pub(crate) row_group_size: usize,
    pub(crate) group_count: usize,
    pub(crate) warmup_iterations: usize,
    pub(crate) iterations: usize,
    pub(crate) limit: usize,
    pub(crate) benchmark_paths: BenchmarkPathSet,
    pub(crate) compression: BenchmarkCompression,
    pub(crate) native_plan_shape: BenchmarkPlanShape,
    pub(crate) expected_outcome: BenchmarkExpectedOutcome,
    pub(crate) measured_run_order: String,
    pub(crate) cpu_comparable: TimedBenchmarkRun,
    pub(crate) cpu_default: TimedBenchmarkRun,
    pub(crate) gpu: Option<TimedBenchmarkRun>,
    pub(crate) datafusion_native_exec: TimedBenchmarkRun,
    pub(crate) manual_native_lowering: TimedBenchmarkRun,
    pub(crate) cpu_comparable_samples: Vec<Duration>,
    pub(crate) cpu_default_samples: Vec<Duration>,
    pub(crate) gpu_samples: Vec<Duration>,
    pub(crate) datafusion_native_exec_samples: Vec<Duration>,
    pub(crate) manual_native_lowering_samples: Vec<Duration>,
    pub(crate) gpu_metric_samples: Vec<LargerIslandBenchmarkMetrics>,
    pub(crate) datafusion_native_exec_metric_samples: Vec<LargerIslandBenchmarkMetrics>,
    pub(crate) manual_native_lowering_metric_samples: Vec<LargerIslandBenchmarkMetrics>,
    pub(crate) final_cpu_comparable: CollectMeasurement,
    pub(crate) final_cpu_default: CollectMeasurement,
    pub(crate) final_gpu: Option<GpuCollectMeasurement>,
    pub(crate) final_datafusion_native_exec: GpuCollectMeasurement,
    pub(crate) final_manual_native_lowering: GpuCollectMeasurement,
    pub(crate) source_diagnostic: String,
}

impl DirectParquetBenchmarkReport {
    pub(crate) fn to_text(&self) -> String {
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
        push_line(&mut text, format_args!("benchmark_paths={}", self.benchmark_paths.name()));
        push_line(&mut text, format_args!("native_plan_shape={}", self.native_plan_shape.name()));
        push_line(&mut text, format_args!("expected_outcome={}", self.expected_outcome.name()));
        push_line(&mut text, format_args!("direct_native_runtime_contract=pass"));
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

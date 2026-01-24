use super::*;

pub(super) fn measure_cpu_output(wall_time: Duration, batches: Vec<RecordBatch>) -> CollectMeasurement {
    let output = measure_output(&batches);
    CollectMeasurement {
        wall_time,
        output_batches: output.output_batches,
        output_rows: output.output_rows,
        output_bytes: output.output_bytes,
        batches,
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct OutputShape {
    pub(super) output_batches: usize,
    pub(super) output_rows: usize,
    pub(super) output_bytes: usize,
}

pub(super) fn measure_output(batches: &[RecordBatch]) -> OutputShape {
    OutputShape {
        output_batches: batches.len(),
        output_rows: batches.iter().map(RecordBatch::num_rows).sum(),
        output_bytes: batches.iter().map(get_record_batch_memory_size).sum(),
    }
}

pub(super) fn benchmark_metrics_from_native(metrics: &NativeQueryMetrics) -> LargerIslandBenchmarkMetrics {
    LargerIslandBenchmarkMetrics {
        phase_c_memory_control_policy: metrics.phase_c_memory_control_policy.as_str(),
        phase_c_runtime_gate: phase_c_runtime_gate(metrics.phase_c_memory_control_policy),
        input_batches: u64_to_usize(metrics.input_batches),
        input_rows: u64_to_usize(metrics.input_rows),
        peak_device_memory_bytes: metrics.peak_device_memory_bytes.map(u64_to_usize),
        resident_dataframe_peak_bytes: metrics.resident_dataframe_peak_bytes.map(u64_to_usize),
        allocator_high_water_bytes: metrics.allocator_high_water_bytes.map(u64_to_usize),
        reservation_requested_bytes: u64_to_usize(metrics.reservation_requested_bytes),
        reservation_granted_bytes: u64_to_usize(metrics.reservation_granted_bytes),
        reservation_denied_count: u64_to_usize(metrics.reservation_denied_count),
        source_chunks_planned: u64_to_usize(metrics.source_chunks_planned),
        source_chunks_attempted: u64_to_usize(metrics.source_chunks_attempted),
        source_chunks_committed: u64_to_usize(metrics.source_chunks_committed),
        source_chunks_retried: u64_to_usize(metrics.source_chunks_retried),
        row_groups_planned: u64_to_usize(metrics.row_groups_planned),
        row_groups_committed: u64_to_usize(metrics.row_groups_committed),
        partial_aggregate_count: u64_to_usize(metrics.partial_aggregate_count),
        partial_aggregate_rows: u64_to_usize(metrics.partial_aggregate_rows),
        final_merge_attempts: u64_to_usize(metrics.final_merge_attempts),
        final_merge_levels: u64_to_usize(metrics.final_merge_levels),
        retry_count: u64_to_usize(metrics.retry_count),
        retry_resource_exhausted_count: u64_to_usize(metrics.retry_resource_exhausted_count),
        retry_reservation_failure_count: u64_to_usize(metrics.retry_reservation_failure_count),
        host_to_device_uploads: u64_to_usize(metrics.boundary.host_to_device_uploads),
        host_to_device_uploaded_rows: u64_to_usize(metrics.boundary.host_to_device_uploaded_rows),
        host_to_device_uploaded_bytes: u64_to_usize(metrics.boundary.host_to_device_uploaded_bytes),
        device_to_host_materializations: u64_to_usize(metrics.boundary.device_to_host_materializations),
        device_to_host_materialized_rows: u64_to_usize(metrics.boundary.device_to_host_materialized_rows),
        device_to_host_materialized_bytes: u64_to_usize(metrics.boundary.device_to_host_materialized_bytes),
        parquet_files: u64_to_usize(metrics.parquet_files),
        parquet_bytes_known: u64_to_usize(metrics.parquet_bytes_known),
        row_groups_selected: u64_to_usize(metrics.row_groups_selected),
        read_columns: u64_to_usize(metrics.read_columns),
        parquet_read_time: metrics.parquet_read_time,
        record_batch_upload_time: Duration::ZERO,
        cudf_operator_time: metrics.operator_time,
        record_batch_materialization_time: metrics.materialization_time,
        elapsed_compute: None,
        output_rows: Some(u64_to_usize(metrics.boundary.device_to_host_materialized_rows)),
    }
}

pub(super) fn benchmark_metrics_from_datafusion_native(
    metrics: &MetricsSet,
    memory_control_policy: NativeMemoryControlPolicy,
) -> LargerIslandBenchmarkMetrics {
    LargerIslandBenchmarkMetrics {
        phase_c_memory_control_policy: memory_control_policy.as_str(),
        phase_c_runtime_gate: phase_c_runtime_gate(memory_control_policy),
        input_batches: metric_count(metrics, "input_batches"),
        input_rows: metric_count(metrics, "input_rows"),
        peak_device_memory_bytes: metric_optional_count(metrics, "peak_device_memory_bytes"),
        resident_dataframe_peak_bytes: metric_optional_count(metrics, "resident_dataframe_peak_bytes"),
        allocator_high_water_bytes: metric_optional_count(metrics, "allocator_high_water_bytes"),
        reservation_requested_bytes: metric_count(metrics, "reservation_requested_bytes"),
        reservation_granted_bytes: metric_count(metrics, "reservation_granted_bytes"),
        reservation_denied_count: metric_count(metrics, "reservation_denied_count"),
        source_chunks_planned: metric_count(metrics, "source_chunks_planned"),
        source_chunks_attempted: metric_count(metrics, "source_chunks_attempted"),
        source_chunks_committed: metric_count(metrics, "source_chunks_committed"),
        source_chunks_retried: metric_count(metrics, "source_chunks_retried"),
        row_groups_planned: metric_count(metrics, "row_groups_planned"),
        row_groups_committed: metric_count(metrics, "row_groups_committed"),
        partial_aggregate_count: metric_count(metrics, "partial_aggregate_count"),
        partial_aggregate_rows: metric_count(metrics, "partial_aggregate_rows"),
        final_merge_attempts: metric_count(metrics, "final_merge_attempts"),
        final_merge_levels: metric_count(metrics, "final_merge_levels"),
        retry_count: metric_count(metrics, "retry_count"),
        retry_resource_exhausted_count: metric_count(metrics, "retry_resource_exhausted_count"),
        retry_reservation_failure_count: metric_count(metrics, "retry_reservation_failure_count"),
        host_to_device_uploads: metric_count(metrics, "host_to_device_uploads"),
        host_to_device_uploaded_rows: metric_count(metrics, "host_to_device_uploaded_rows"),
        host_to_device_uploaded_bytes: metric_count(metrics, "host_to_device_uploaded_bytes"),
        device_to_host_materializations: metric_count(metrics, "device_to_host_materializations"),
        device_to_host_materialized_rows: metric_count(metrics, "device_to_host_materialized_rows"),
        device_to_host_materialized_bytes: metric_count(metrics, "device_to_host_materialized_bytes"),
        parquet_files: metric_count(metrics, "parquet_files"),
        parquet_bytes_known: metric_count(metrics, "parquet_bytes_known"),
        row_groups_selected: metric_count(metrics, "row_groups_selected"),
        read_columns: metric_count(metrics, "read_columns"),
        parquet_read_time: metric_duration(metrics, "parquet_read_time"),
        record_batch_upload_time: Duration::ZERO,
        cudf_operator_time: metric_duration(metrics, "operator_time"),
        record_batch_materialization_time: metric_duration(metrics, "materialization_time"),
        elapsed_compute: metrics.elapsed_compute().map(duration_from_nanos),
        output_rows: metrics.output_rows(),
    }
}

pub(super) fn metric_count(metrics: &MetricsSet, name: &str) -> usize {
    metrics.sum_by_name(name).map(|value| value.as_usize()).unwrap_or(0)
}

pub(super) fn metric_optional_count(metrics: &MetricsSet, name: &str) -> Option<usize> {
    metrics.sum_by_name(name).map(|value| value.as_usize())
}

pub(super) fn metric_duration(metrics: &MetricsSet, name: &str) -> Duration {
    metrics.sum_by_name(name).map(|value| duration_from_nanos(value.as_usize())).unwrap_or(Duration::ZERO)
}

pub(super) fn duration_from_nanos(nanos: usize) -> Duration {
    Duration::from_nanos(nanos.min(u64::MAX as usize) as u64)
}

pub(super) fn speedup(cpu: Duration, gpu: Duration) -> f64 {
    if cpu.is_zero() {
        return 0.0;
    }
    if gpu.is_zero() {
        return f64::INFINITY;
    }
    cpu.as_secs_f64() / gpu.as_secs_f64()
}

pub(super) fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

pub(super) fn u64_to_usize(value: u64) -> usize {
    value.min(usize::MAX as u64) as usize
}

pub(super) fn duration_samples_csv(samples: &[Duration]) -> String {
    samples.iter().map(|duration| format!("{:.3}", duration_ms(*duration))).collect::<Vec<_>>().join(",")
}

pub(super) fn optional_duration_samples_csv(samples: &[Duration]) -> String {
    if samples.is_empty() { "not_run".to_string() } else { duration_samples_csv(samples) }
}

pub(super) fn optional_f64_text(value: Option<f64>) -> String {
    value.map_or_else(|| "not_run".to_string(), |value| format!("{value:.3}"))
}

pub(super) fn metric_duration_samples_csv(
    samples: &[LargerIslandBenchmarkMetrics],
    metric: impl Fn(&LargerIslandBenchmarkMetrics) -> Duration,
) -> String {
    samples.iter().map(|sample| format!("{:.3}", duration_ms(metric(sample)))).collect::<Vec<_>>().join(",")
}

pub(super) fn metric_optional_usize_samples_csv(
    samples: &[LargerIslandBenchmarkMetrics],
    metric: impl Fn(&LargerIslandBenchmarkMetrics) -> Option<usize>,
) -> String {
    samples.iter().map(|sample| optional_usize_text(metric(sample))).collect::<Vec<_>>().join(",")
}

pub(super) fn mean_metric_duration(
    samples: &[LargerIslandBenchmarkMetrics],
    metric: impl Fn(&LargerIslandBenchmarkMetrics) -> Duration,
) -> Duration {
    if samples.is_empty() {
        return Duration::ZERO;
    }
    let nanos = samples.iter().map(|sample| metric(sample).as_nanos()).sum::<u128>() / samples.len() as u128;
    Duration::from_nanos(nanos.min(u64::MAX as u128) as u64)
}

pub(super) fn mean_metric_optional_usize(
    samples: &[LargerIslandBenchmarkMetrics],
    metric: impl Fn(&LargerIslandBenchmarkMetrics) -> Option<usize>,
) -> Option<usize> {
    let mut total = 0u128;
    let mut count = 0u128;
    for sample in samples {
        let value = metric(sample)?;
        total += value as u128;
        count += 1;
    }
    total.checked_div(count).map(|mean| mean.min(usize::MAX as u128) as usize)
}

pub(super) fn optional_usize_text(value: Option<usize>) -> String {
    value.map_or_else(|| "unavailable".to_string(), |value| value.to_string())
}

pub(super) fn phase_c_runtime_gate(_policy: NativeMemoryControlPolicy) -> &'static str {
    "not_evaluated"
}

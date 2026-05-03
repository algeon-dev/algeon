use super::*;

pub(super) fn validate_gpu_direct_measurement(
    measurement: &GpuCollectMeasurement,
    parquet_file_bytes: u64,
    config: &LocalParquetBenchmarkConfig,
) -> DataFusionResult<()> {
    let metrics = &measurement.metrics;
    if !measurement.plan_one_line.contains("input=local_parquet_scan")
        || !measurement.plan_one_line.contains("input_boundary=cudf_parquet_reader")
    {
        return Err(plan_error(format!(
            "GPU plan did not use direct local Parquet input: {}",
            measurement.plan_one_line
        )));
    }
    if metrics.host_to_device_uploads != 0 || metrics.host_to_device_uploaded_rows != 0 {
        return Err(plan_error(format!(
            "direct GPU Parquet path crossed host-to-device upload boundary: uploads={}, uploaded_rows={}",
            metrics.host_to_device_uploads, metrics.host_to_device_uploaded_rows
        )));
    }
    if metrics.parquet_files != config.file_count {
        return Err(plan_error(format!(
            "direct GPU Parquet path expected {} files, got {}",
            config.file_count, metrics.parquet_files
        )));
    }
    if metrics.parquet_bytes_known < parquet_file_bytes as usize
        || metrics.parquet_bytes_known < config.minimum_file_bytes as usize
    {
        return Err(plan_error(format!(
            "direct GPU Parquet metrics did not account for the benchmark file size: parquet_bytes_known={}, parquet_file_bytes={}, minimum_file_bytes={}",
            metrics.parquet_bytes_known, parquet_file_bytes, config.minimum_file_bytes
        )));
    }
    if metrics.device_to_host_materialized_rows > config.limit {
        return Err(plan_error(format!(
            "direct GPU Parquet path materialized too many rows: materialized_rows={}, limit={}",
            metrics.device_to_host_materialized_rows, config.limit
        )));
    }

    let mut actual = measurement.read_columns.clone();
    actual.sort();
    let expected = config.expected_read_columns();
    if actual != expected {
        return Err(plan_error(format!(
            "direct GPU Parquet read columns did not match workload requirements: actual={}, expected={}",
            actual.join(","),
            expected.join(",")
        )));
    }
    Ok(())
}

pub(super) fn validate_phase_b_runtime_gate(
    cpu_comparable: &TimedBenchmarkRun,
    datafusion_native_exec: &TimedBenchmarkRun,
    native_measurement: &GpuCollectMeasurement,
    parquet_file_bytes: u64,
    dataset_layout: ParquetDatasetLayout,
    config: &LocalParquetBenchmarkConfig,
) -> DataFusionResult<()> {
    if parquet_file_bytes < config.minimum_file_bytes {
        return Err(plan_error(format!(
            "Phase B runtime gate requires a 3 GiB+ Parquet dataset: parquet_file_bytes={}, minimum_file_bytes={}",
            parquet_file_bytes, config.minimum_file_bytes
        )));
    }
    if dataset_layout.row_group_count <= dataset_layout.file_count {
        return Err(plan_error(format!(
            "Phase B runtime gate requires a multi-row-group Parquet dataset: files={}, row_groups={}",
            dataset_layout.file_count, dataset_layout.row_group_count
        )));
    }
    if native_measurement.metrics.host_to_device_uploads != 0
        || native_measurement.metrics.host_to_device_uploaded_rows != 0
    {
        return Err(plan_error(format!(
            "Phase B runtime gate requires zero native host-to-device uploads: uploads={}, uploaded_rows={}",
            native_measurement.metrics.host_to_device_uploads, native_measurement.metrics.host_to_device_uploaded_rows
        )));
    }
    if native_measurement.metrics.device_to_host_materializations != 1 {
        return Err(plan_error(format!(
            "Phase B runtime gate requires one final native host materialization: materializations={}",
            native_measurement.metrics.device_to_host_materializations
        )));
    }
    if native_measurement.metrics.row_groups_selected == 0
        || native_measurement.metrics.row_groups_selected >= dataset_layout.row_group_count
    {
        return Err(plan_error(format!(
            "Phase B runtime gate requires native row-group pruning: row_groups_selected={}, total_row_groups={}",
            native_measurement.metrics.row_groups_selected, dataset_layout.row_group_count
        )));
    }

    let native_speedup = speedup(cpu_comparable.mean, datafusion_native_exec.mean);
    if native_speedup <= config.minimum_native_gpu_speedup {
        return Err(plan_error(format!(
            "Phase B runtime gate did not preserve the DataFusion native exec win: datafusion_native_exec_speedup={native_speedup:.3}, minimum_native_gpu_speedup={:.3}, cpu_comparable_wall_ms={:.3}, datafusion_native_exec_wall_ms={:.3}",
            config.minimum_native_gpu_speedup,
            duration_ms(cpu_comparable.mean),
            duration_ms(datafusion_native_exec.mean)
        )));
    }
    Ok(())
}

pub(super) fn validate_phase_c_runtime_gate(
    native_measurement: &GpuCollectMeasurement,
    config: &LocalParquetBenchmarkConfig,
) -> DataFusionResult<()> {
    if !config.native_memory_control_policy.is_bounded() {
        return Ok(());
    }

    if native_measurement.metrics.phase_c_runtime_gate != "pass" {
        return Err(plan_error(format!(
            "Phase C runtime gate failed: gate={}, policy={}, source_chunks_planned={}, source_chunks_committed={}, row_groups_planned={}, row_groups_committed={}, host_to_device_uploads={}, device_to_host_materializations={}, resident_dataframe_peak_bytes={}",
            native_measurement.metrics.phase_c_runtime_gate,
            native_measurement.metrics.phase_c_memory_control_policy,
            native_measurement.metrics.source_chunks_planned,
            native_measurement.metrics.source_chunks_committed,
            native_measurement.metrics.row_groups_planned,
            native_measurement.metrics.row_groups_committed,
            native_measurement.metrics.host_to_device_uploads,
            native_measurement.metrics.device_to_host_materializations,
            optional_usize_text(native_measurement.metrics.resident_dataframe_peak_bytes),
        )));
    }

    Ok(())
}

pub(super) fn validate_datafusion_native_exec_measurement(
    measurement: &GpuCollectMeasurement,
    parquet_file_bytes: u64,
    config: &LocalParquetBenchmarkConfig,
) -> DataFusionResult<()> {
    if !measurement.plan_one_line.contains("NativeCudfExec") {
        return Err(plan_error(format!(
            "datafusion_native_exec benchmark did not execute through NativeCudfExec: {}",
            measurement.plan_one_line
        )));
    }
    let Some(report) = &measurement.native_datafusion_planning_report else {
        return Err(plan_error("datafusion_native_exec benchmark did not retain a native planning report"));
    };
    if report.accepted_count() != 1 || report.rejected_count() != 0 {
        return Err(plan_error(format!(
            "datafusion_native_exec planning did not accept exactly one native plan: accepted_count={}, rejected_count={}",
            report.accepted_count(),
            report.rejected_count()
        )));
    }
    let row = report.rows.first().ok_or_else(|| plan_error("datafusion_native_exec planning report was empty"))?;
    if row.post_native_plan_root != "NativeCudfExec" || row.post_native_execution_path.as_str() != "native" {
        return Err(plan_error(format!(
            "datafusion_native_exec planning did not route to native execution: root={}, path={}",
            row.post_native_plan_root,
            row.post_native_execution_path.as_str()
        )));
    }
    validate_native_direct_metrics("datafusion_native_exec", measurement, parquet_file_bytes, config)
}

pub(super) fn validate_manual_native_lowering_measurement(
    measurement: &GpuCollectMeasurement,
    parquet_file_bytes: u64,
    config: &LocalParquetBenchmarkConfig,
) -> DataFusionResult<()> {
    if !measurement.plan_one_line.contains("LocalParquetSource")
        || !measurement.plan_one_line.contains("format=Parquet")
        || !measurement.plan_one_line.contains("direct_device_read=true")
        || !measurement.plan_one_line.contains("sink=host_arrow")
    {
        return Err(plan_error(format!(
            "manual_native_lowering plan did not lower to the direct local Parquet host-arrow path: {}",
            measurement.plan_one_line
        )));
    }
    validate_native_direct_metrics("manual_native_lowering", measurement, parquet_file_bytes, config)
}

pub(super) fn validate_native_direct_metrics(
    label: &str,
    measurement: &GpuCollectMeasurement,
    parquet_file_bytes: u64,
    config: &LocalParquetBenchmarkConfig,
) -> DataFusionResult<()> {
    let metrics = &measurement.metrics;
    if metrics.host_to_device_uploads != 0 || metrics.host_to_device_uploaded_rows != 0 {
        return Err(plan_error(format!(
            "{label} direct Parquet path crossed host-to-device upload boundary: uploads={}, uploaded_rows={}",
            metrics.host_to_device_uploads, metrics.host_to_device_uploaded_rows
        )));
    }
    if metrics.device_to_host_materializations != 1 {
        return Err(plan_error(format!(
            "{label} direct Parquet path expected one final host materialization, got {}",
            metrics.device_to_host_materializations
        )));
    }
    if metrics.device_to_host_materialized_rows != measurement.output_rows {
        return Err(plan_error(format!(
            "{label} materialized row metrics did not match output rows: materialized_rows={}, output_rows={}",
            metrics.device_to_host_materialized_rows, measurement.output_rows
        )));
    }
    let Some(peak_device_memory_bytes) = metrics.peak_device_memory_bytes else {
        return Err(plan_error(format!("{label} direct Parquet path did not record peak device memory")));
    };
    if peak_device_memory_bytes == 0 {
        return Err(plan_error(format!("{label} direct Parquet path recorded zero peak device memory")));
    }
    if measurement.output_rows > config.limit {
        return Err(plan_error(format!(
            "{label} direct Parquet path returned too many rows: output_rows={}, limit={}",
            measurement.output_rows, config.limit
        )));
    }
    if metrics.parquet_files != config.file_count {
        return Err(plan_error(format!(
            "{label} direct Parquet path expected {} files, got {}",
            config.file_count, metrics.parquet_files
        )));
    }
    if metrics.parquet_bytes_known < parquet_file_bytes as usize
        || metrics.parquet_bytes_known < config.minimum_file_bytes as usize
    {
        return Err(plan_error(format!(
            "{label} direct Parquet metrics did not account for the benchmark file size: parquet_bytes_known={}, parquet_file_bytes={}, minimum_file_bytes={}",
            metrics.parquet_bytes_known, parquet_file_bytes, config.minimum_file_bytes
        )));
    }
    if measurement.planning.rejected_count != 0 {
        return Err(plan_error(format!(
            "{label} lowering rejected fallback candidates: rejected_count={}",
            measurement.planning.rejected_count
        )));
    }

    let mut actual = measurement.read_columns.clone();
    actual.sort();
    let expected = config.expected_read_columns();
    if actual != expected {
        return Err(plan_error(format!(
            "{label} direct Parquet read columns did not match workload requirements: actual={}, expected={}",
            actual.join(","),
            expected.join(",")
        )));
    }
    Ok(())
}

pub(super) fn compare_outputs(cpu: &[RecordBatch], gpu: &[RecordBatch]) -> DataFusionResult<()> {
    let cpu_rows = cpu.iter().map(RecordBatch::num_rows).sum::<usize>();
    let gpu_rows = gpu.iter().map(RecordBatch::num_rows).sum::<usize>();
    if cpu_rows != gpu_rows {
        return Err(plan_error(format!("CPU output rows ({cpu_rows}) did not match GPU output rows ({gpu_rows})")));
    }

    let cpu_values = batch_rows(cpu)?;
    let gpu_values = batch_rows(gpu)?;
    if cpu_values != gpu_values {
        return Err(plan_error("CPU and GPU benchmark outputs differed"));
    }
    Ok(())
}

pub(super) fn batch_rows(batches: &[RecordBatch]) -> DataFusionResult<Vec<Vec<String>>> {
    let mut rows = Vec::new();
    for batch in batches {
        for row in 0..batch.num_rows() {
            let mut values = Vec::with_capacity(batch.num_columns());
            for column in 0..batch.num_columns() {
                values.push(array_value_to_string(batch.column(column).as_ref(), row)?);
            }
            rows.push(values);
        }
    }
    Ok(rows)
}

//! Explicit execution-attempt lifecycle embedding example.
//!
//! Set `NEXUS_ATTEMPT_LIFECYCLE_PROFILING_ROWS` for an Nsight-sized input.

use std::{process::ExitCode, sync::Arc};

use anyhow::{Context, anyhow, ensure};
use arrow::{array::Int32Array, record_batch::RecordBatch};
use datafusion::prelude::SessionContext;
use datafusion_nexus::{
    attempt::AttemptOptions,
    backend::{GpuDeviceMemoryLimit, GpuMemoryOwnership, NexusGpuBackend, NexusGpuDeviceProfile},
    native_exec::NexusNativeExecutionConfig,
};
use futures::StreamExt;
use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};

const ATTEMPT_MEMORY_LIMIT_BYTES: u64 = 1024 * 1024 * 1024;
const DEVICE_MEMORY_LIMIT_BYTES: u64 = 4 * ATTEMPT_MEMORY_LIMIT_BYTES;
const DEFAULT_PROFILING_ROWS: usize = 1_000_000;
const PARQUET_BATCH_ROWS: usize = 1_000_000;
const PROFILING_ROWS_ENV: &str = "NEXUS_ATTEMPT_LIFECYCLE_PROFILING_ROWS";

#[tokio::main]
async fn main() -> ExitCode {
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("attempt lifecycle example: {error:#}");
            ExitCode::FAILURE
        },
    }
}

async fn run() -> anyhow::Result<()> {
    let backend = NexusGpuBackend::builder()
        .with_native_execution_config(NexusNativeExecutionConfig::bounded(
            ATTEMPT_MEMORY_LIMIT_BYTES,
            64 * 1024 * 1024,
            1,
            3,
            1,
        ))
        .device_profiles([NexusGpuDeviceProfile::new(0)
            .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: DEVICE_MEMORY_LIMIT_BYTES })
            .with_max_active_attempts(2)
            .with_installation_bounds(8, 2, 2)])
        .device_memory_ownership([(0, GpuMemoryOwnership::WholeDeviceExclusive)])
        .build()
        .map_err(|error| anyhow!("no usable CUDA device 0 for this example: {error}"))?;

    let ctx_a = SessionContext::new_with_state(
        backend.install_on(datafusion::execution::SessionStateBuilder::new_with_default_features())?.build(),
    );
    let ctx_b = SessionContext::new_with_state(
        backend.install_on(datafusion::execution::SessionStateBuilder::new_with_default_features())?.build(),
    );
    let directory = tempfile::tempdir()?;
    let fallback_parquet = directory.path().join("attempt_lifecycle_fallback.parquet");
    let profiling_parquet = directory.path().join("attempt_lifecycle_profiling.parquet");
    write_parquet(&fallback_parquet, 4)?;
    let profiling_rows = profiling_row_count()?;
    write_parquet(&profiling_parquet, profiling_rows)?;
    let fallback_url = fallback_parquet.to_string_lossy();
    let profiling_url = profiling_parquet.to_string_lossy();
    ctx_a
        .sql(&format!("CREATE EXTERNAL TABLE fallback_values STORED AS PARQUET LOCATION '{fallback_url}'"))
        .await?
        .collect()
        .await?;
    ctx_a
        .sql(&format!("CREATE EXTERNAL TABLE values_a STORED AS PARQUET LOCATION '{profiling_url}'"))
        .await?
        .collect()
        .await?;
    ctx_b
        .sql(&format!("CREATE EXTERNAL TABLE values_b STORED AS PARQUET LOCATION '{profiling_url}'"))
        .await?
        .collect()
        .await?;

    // CPU-only invocation: the aggregate is outside the native capability
    // catalogue, so the plan falls back to DataFusion CPU execution.
    run_and_print(
        &backend.begin_attempt(AttemptOptions::default())?,
        &ctx_a,
        "SELECT approx_distinct(value) FROM fallback_values",
    )
    .await?;

    run_same_device_native_attempts(&backend, &ctx_a, &ctx_b).await?;

    // Completion is independent of result consumption: dropping the stream
    // still produces a bounded terminal report.
    let dropped = backend.begin_attempt(AttemptOptions::default())?;
    let stream = dropped.sql(&ctx_a, "SELECT * FROM fallback_values").await?;
    drop(stream);
    print_report(dropped.completion().await.as_ref());
    Ok(())
}

/// Run two native cuDF attempts concurrently on the same explicitly owned
/// device.  This is a public-API profiling probe, not the Phase 6 runner.
async fn run_same_device_native_attempts(
    backend: &NexusGpuBackend,
    ctx_a: &SessionContext,
    ctx_b: &SessionContext,
) -> anyhow::Result<()> {
    let first = backend.begin_attempt(AttemptOptions::default())?;
    let second = backend.begin_attempt(AttemptOptions::default())?;
    let (first_report, second_report) = tokio::try_join!(
        run_and_capture(&first, ctx_a, "SELECT SUM(value) FROM values_a WHERE value > 0"),
        run_and_capture(&second, ctx_b, "SELECT SUM(value) FROM values_b WHERE value > 0"),
    )?;

    ensure!(first_report.attempt_id != second_report.attempt_id, "concurrent attempts must retain distinct identities");
    let first_admission = first_report.admission.as_ref().context("first native attempt was not admitted")?;
    let second_admission = second_report.admission.as_ref().context("second native attempt was not admitted")?;
    ensure!(
        first_report.disposition == nexus_query_engine::observability::AttemptDisposition::Native,
        "first profiling attempt did not reach native cuDF execution: {:?}",
        first_report.disposition
    );
    ensure!(
        second_report.disposition == nexus_query_engine::observability::AttemptDisposition::Native,
        "second profiling attempt did not reach native cuDF execution: {:?}",
        second_report.disposition
    );
    ensure!(
        first_admission.device_ordinal == 0 && second_admission.device_ordinal == 0,
        "profiling probe must place both attempts on device 0, got {} and {}",
        first_admission.device_ordinal,
        second_admission.device_ordinal
    );

    println!(
        "same_device_native_attempts device_ordinal=0 first_attempt_id={} first_grant_bytes={} \
         second_attempt_id={} second_grant_bytes={}",
        first_report.attempt_id, first_admission.grant_bytes, second_report.attempt_id, second_admission.grant_bytes,
    );
    print_report(first_report.as_ref());
    print_report(second_report.as_ref());
    Ok(())
}

async fn run_and_print(
    attempt: &datafusion_nexus::attempt::NexusAttempt,
    ctx: &SessionContext,
    sql: &str,
) -> anyhow::Result<()> {
    let report = run_and_capture(attempt, ctx, sql).await?;
    print_report(report.as_ref());
    Ok(())
}

async fn run_and_capture(
    attempt: &datafusion_nexus::attempt::NexusAttempt,
    ctx: &SessionContext,
    sql: &str,
) -> anyhow::Result<Arc<nexus_query_engine::observability::QueryAttemptReport>> {
    let stream = attempt.sql(ctx, sql).await?;
    for batch in stream.collect::<Vec<_>>().await {
        batch.context("consume attempt result stream")?;
    }
    Ok(attempt.completion().await)
}

fn print_report(report: &nexus_query_engine::observability::QueryAttemptReport) {
    println!(
        "attempt_id={} disposition={:?} outcome={:?} total_ns={:?} planning_ns={:?}",
        report.attempt_id,
        report.disposition,
        report.outcome,
        report.total_ns(),
        report.planning_ns(),
    );
}

fn profiling_row_count() -> anyhow::Result<usize> {
    let row_count = match std::env::var(PROFILING_ROWS_ENV) {
        Ok(value) => value
            .parse::<usize>()
            .with_context(|| format!("{PROFILING_ROWS_ENV} must be a positive decimal row count"))?,
        Err(std::env::VarError::NotPresent) => DEFAULT_PROFILING_ROWS,
        Err(error) => return Err(anyhow!("read {PROFILING_ROWS_ENV}: {error}")),
    };
    ensure!(row_count > 0, "{PROFILING_ROWS_ENV} must be greater than zero");
    Ok(row_count)
}

fn write_parquet(path: &std::path::Path, row_count: usize) -> anyhow::Result<()> {
    let schema = Arc::new(arrow::datatypes::Schema::new(vec![arrow::datatypes::Field::new(
        "value",
        arrow::datatypes::DataType::Int32,
        false,
    )]));
    let file = std::fs::File::create(path)?;
    let properties = WriterProperties::builder().set_max_row_group_row_count(Some(PARQUET_BATCH_ROWS)).build();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(properties))?;
    for offset in (0..row_count).step_by(PARQUET_BATCH_ROWS) {
        let upper_bound = (offset + PARQUET_BATCH_ROWS).min(row_count);
        let values = (offset..upper_bound).map(|value| (value % 1024) as i32).collect::<Vec<_>>();
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(values))])?;
        writer.write(&batch)?;
    }
    writer.close()?;
    Ok(())
}

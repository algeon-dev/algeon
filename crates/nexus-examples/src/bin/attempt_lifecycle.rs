//! Explicit execution-attempt lifecycle embedding example.

use std::{process::ExitCode, sync::Arc};

use anyhow::{Context, anyhow};
use arrow::{array::Int32Array, record_batch::RecordBatch};
use datafusion::prelude::SessionContext;
use datafusion_nexus::{
    attempt::AttemptOptions,
    backend::{GpuMemoryOwnership, NexusGpuBackend, NexusMemoryPolicy},
};
use futures::StreamExt;
use parquet::arrow::ArrowWriter;

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
        .memory_policy(NexusMemoryPolicy::Bounded {
            device_ordinals: vec![0],
            device_budget_bytes: 256 * 1024 * 1024,
            max_source_chunk_bytes: 64 * 1024 * 1024,
            max_row_groups_per_chunk: 2,
            max_retry_attempts: 3,
            min_retry_chunk_row_groups: 1,
        })
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
    let parquet = directory.path().join("attempt_lifecycle.parquet");
    write_parquet(&parquet)?;
    let url = parquet.to_string_lossy();
    ctx_a.sql(&format!("CREATE EXTERNAL TABLE values_a STORED AS PARQUET LOCATION '{url}'")).await?.collect().await?;
    ctx_b.sql(&format!("CREATE EXTERNAL TABLE values_b STORED AS PARQUET LOCATION '{url}'")).await?.collect().await?;

    // CPU-only invocation: the aggregate is outside the native capability
    // catalogue, so the plan falls back to DataFusion CPU execution.
    run_and_print(
        &backend.begin_attempt(AttemptOptions::default())?,
        &ctx_a,
        "SELECT approx_distinct(value) FROM values_a",
    )
    .await?;
    run_and_print(
        &backend.begin_attempt(AttemptOptions::default())?,
        &ctx_b,
        "SELECT SUM(value) FROM values_b WHERE value > 1",
    )
    .await?;

    // Completion is independent of result consumption: dropping the stream
    // still produces a bounded terminal report.
    let dropped = backend.begin_attempt(AttemptOptions::default())?;
    let stream = dropped.sql(&ctx_a, "SELECT * FROM values_a").await?;
    drop(stream);
    print_report(dropped.completion().await.as_ref());
    Ok(())
}

async fn run_and_print(
    attempt: &datafusion_nexus::attempt::NexusAttempt,
    ctx: &SessionContext,
    sql: &str,
) -> anyhow::Result<()> {
    let stream = attempt.sql(ctx, sql).await?;
    for batch in stream.collect::<Vec<_>>().await {
        batch.context("consume attempt result stream")?;
    }
    print_report(attempt.completion().await.as_ref());
    Ok(())
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

fn write_parquet(path: &std::path::Path) -> anyhow::Result<()> {
    let schema = Arc::new(arrow::datatypes::Schema::new(vec![arrow::datatypes::Field::new(
        "value",
        arrow::datatypes::DataType::Int32,
        false,
    )]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![1, 2, 3, 4]))])?;
    let file = std::fs::File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema, None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

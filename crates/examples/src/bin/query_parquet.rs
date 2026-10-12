//! Filter and aggregate a Parquet file on the GPU, then inspect the attempt report.

use std::{fs::File, time::Duration};

use algeon_datafusion::{
    attempt::{AttemptDisposition, AttemptOptions},
    backend::{AlgeonGpuBackend, AlgeonGpuDeviceConfig, AlgeonGpuDeviceProfile, GpuMemoryOwnership},
    planner::AlgeonGpuExecutionMode,
};
use anyhow::{Context, Result, ensure};
use arrow::util::pretty::print_batches;
use datafusion::{
    execution::SessionStateBuilder,
    prelude::{ParquetReadOptions, SessionContext},
};
use futures::TryStreamExt;
use parquet::arrow::ArrowWriter;

const SQL: &str = "
    SELECT region, SUM(amount) AS revenue
    FROM orders
    WHERE paid
    GROUP BY region
    ORDER BY revenue DESC
";

#[tokio::main]
async fn main() -> Result<()> {
    let device = AlgeonGpuDeviceConfig::new(AlgeonGpuDeviceProfile::new(0), GpuMemoryOwnership::WholeDeviceExclusive);
    let backend = AlgeonGpuBackend::builder()
        .devices([device])
        .with_execution_mode(AlgeonGpuExecutionMode::NativeRequired)
        .build()?;

    // Keep shutdown on the error path, after the session and streams leave scope.
    let result = run(&backend).await;
    backend.close_and_wait_async(Duration::from_secs(30)).await?;
    result
}

async fn run(backend: &AlgeonGpuBackend) -> Result<()> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("orders.parquet");
    let input = SessionContext::new()
        .sql(
            "SELECT * FROM (VALUES
            ('North', 120, true), ('South', 80, true),
            ('North', 30, true), ('South', 50, false)
        ) AS orders(region, amount, paid)",
        )
        .await?
        .collect()
        .await?;
    let mut writer = ArrowWriter::try_new(File::create(&path)?, input[0].schema(), None)?;
    for batch in &input {
        writer.write(batch)?;
    }
    writer.close()?;

    let state = backend.install_on(SessionStateBuilder::new_with_default_features())?.build();
    let ctx = SessionContext::new_with_state(state);
    let path = path.to_str().context("temporary Parquet path must be UTF-8")?;
    ctx.register_parquet("orders", path, ParquetReadOptions::default()).await?;

    let attempt = backend.begin_attempt(AttemptOptions::default())?;
    // Retain completion before planning to observe failures as well as successful queries.
    let completion = attempt.completion();
    let result = async { attempt.sql(&ctx, SQL).await?.try_collect::<Vec<_>>().await }.await;
    let report = completion.await;
    println!("execution={:?} outcome={:?}", report.disposition, report.outcome);
    let batches = result?;
    ensure!(report.disposition == AttemptDisposition::Native, "expected native GPU execution");
    print_batches(&batches)?;
    Ok(())
}

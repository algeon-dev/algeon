//! Find the nearest catalog vectors with cuVS and return item names through SQL.

use std::time::Duration;

use algeon_datafusion::{
    attempt::AttemptOptions,
    backend::{AlgeonGpuBackend, AlgeonGpuDeviceConfig, AlgeonGpuDeviceProfile, GpuMemoryOwnership},
    planner::AlgeonGpuExecutionMode,
    session::AlgeonSessionStateBuilderExt,
};
use anyhow::{Result, ensure};
use arrow::util::pretty::print_batches;
use datafusion::{execution::SessionStateBuilder, prelude::SessionContext};
use futures::TryStreamExt;

const SQL: &str = "
    SELECT n.rank, i.name, n.distance
    FROM cuvs_brute_force_knn(
        dataset => (SELECT id, x, y FROM items),
        queries => (SELECT 100::BIGINT AS id, 0::REAL AS x, 0::REAL AS y),
        k => 2, metric => 'l2_expanded'
    ) n
    JOIN items i ON i.id = n.neighbor_id
    ORDER BY n.rank
";

#[tokio::main]
async fn main() -> Result<()> {
    let device = AlgeonGpuDeviceConfig::new(AlgeonGpuDeviceProfile::new(0), GpuMemoryOwnership::WholeDeviceExclusive);
    let backend = AlgeonGpuBackend::builder()
        .devices([device])
        .with_execution_mode(AlgeonGpuExecutionMode::FunctionsOnly)
        .build()?;

    let result = run(&backend).await;
    backend.close_and_wait_async(Duration::from_secs(30)).await?;
    result
}

async fn run(backend: &AlgeonGpuBackend) -> Result<()> {
    let state = backend.install_on(SessionStateBuilder::new_with_default_features())?.try_with_cuvs_sql()?.build();
    let ctx = SessionContext::new_with_state(state);
    // The UDTF input contains an Int64 id followed by equally typed vector dimensions.
    ctx.register_table(
        "items",
        ctx.sql(
            "SELECT * FROM (VALUES
        (1::BIGINT, 'Notebook', 0::REAL, 0::REAL),
        (2, 'Pencil', 1::REAL, 0::REAL), (3, 'Backpack', 3::REAL, 4::REAL)
    ) AS items(id, name, x, y)",
        )
        .await?
        .into_view(),
    )?;

    let attempt = backend.begin_attempt(AttemptOptions::default())?;
    let completion = attempt.completion();
    let result = async { attempt.sql(&ctx, SQL).await?.try_collect::<Vec<_>>().await }.await;
    let report = completion.await;
    println!("execution={:?} outcome={:?}", report.disposition, report.outcome);
    let batches = result?;
    ensure!(report.admission.is_some(), "vector search must acquire GPU admission");
    print_batches(&batches)?;
    Ok(())
}

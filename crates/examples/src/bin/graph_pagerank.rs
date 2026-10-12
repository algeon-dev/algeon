//! Rank four linked pages with cuGraph and join their scores to page titles in SQL.

use std::time::Duration;

use algeon_datafusion::{
    attempt::AttemptOptions,
    backend::{AlgeonGpuBackend, AlgeonGpuDeviceConfig, AlgeonGpuDeviceProfile, GpuMemoryOwnership},
    cugraph_sql::CugraphSqlConfig,
    planner::AlgeonGpuExecutionMode,
    session::AlgeonSessionStateBuilderExt,
};
use anyhow::{Result, ensure};
use arrow::util::pretty::print_batches;
use datafusion::{execution::SessionStateBuilder, prelude::SessionContext};
use futures::TryStreamExt;

const SQL: &str = "
    SELECT p.title, r.value AS pagerank
    FROM cugraph_pagerank(edges => (SELECT src, dst FROM links)) r
    JOIN pages p ON p.id = r.vertex
    ORDER BY pagerank DESC, p.id
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
    let state = backend
        .install_on(SessionStateBuilder::new_with_default_features())?
        .try_with_cugraph_sql(CugraphSqlConfig::default())?
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_table(
        "pages",
        ctx.sql(
            "SELECT * FROM (VALUES
        (1::BIGINT, 'Home'), (2, 'Guide'), (3, 'API'), (4, 'FAQ')
    ) AS pages(id, title)",
        )
        .await?
        .into_view(),
    )?;
    ctx.register_table(
        "links",
        ctx.sql(
            "SELECT * FROM (VALUES
        (1::BIGINT, 2::BIGINT), (1, 3), (2, 3), (3, 1), (4, 3)
    ) AS links(src, dst)",
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
    ensure!(report.admission.is_some(), "PageRank must acquire GPU admission");
    print_batches(&batches)?;
    Ok(())
}

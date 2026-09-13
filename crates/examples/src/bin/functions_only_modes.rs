//! Run one CPU/GPU composition under a selected GPU execution mode.

use std::{env, sync::Arc, time::Duration};

use algeon_datafusion::{
    backend::{AlgeonGpuBackend, AlgeonGpuDeviceConfig, AlgeonGpuDeviceProfile, GpuMemoryOwnership},
    cugraph_sql::CugraphSqlConfig,
    planner::AlgeonGpuExecutionMode,
    session::AlgeonSessionStateBuilderExt,
};
use anyhow::{Context, bail};
use arrow::{
    array::{Float64Array, Int64Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use datafusion::{datasource::MemTable, execution::SessionStateBuilder, prelude::SessionContext};

const MODE_ENV: &str = "ALGEON_EXAMPLE_EXECUTION_MODE";
const SQL: &str = "SELECT vertex, value \
                   FROM cugraph_pagerank(edges => (SELECT src, dst, weight FROM edges)) \
                   WHERE value > 0 \
                   ORDER BY value DESC, vertex";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mode = execution_mode()?;
    let device = AlgeonGpuDeviceConfig::new(AlgeonGpuDeviceProfile::new(0), GpuMemoryOwnership::WholeDeviceExclusive);
    let backend = AlgeonGpuBackend::builder().with_execution_mode(mode).devices([device]).build()?;
    let builder = backend.install_on(SessionStateBuilder::new_with_default_features())?;
    let state = builder.try_with_cugraph_sql(CugraphSqlConfig::default())?.build();
    let ctx = SessionContext::new_with_state(state);
    register_edges(&ctx)?;

    let batches = ctx.sql(SQL).await?.collect().await?;
    println!("mode={mode:?} rows={}", batches.iter().map(RecordBatch::num_rows).sum::<usize>());

    // Release the session's backend handle before waiting for backend shutdown.
    drop(ctx);
    backend.close_and_wait_async(Duration::from_secs(30)).await.context("GPU backend did not close cleanly")?;
    Ok(())
}

fn execution_mode() -> anyhow::Result<AlgeonGpuExecutionMode> {
    match env::var(MODE_ENV).unwrap_or_else(|_| "functions_only".to_string()).as_str() {
        "functions_only" => Ok(AlgeonGpuExecutionMode::FunctionsOnly),
        "native_preferred" => Ok(AlgeonGpuExecutionMode::NativePreferred),
        "native_required" => Ok(AlgeonGpuExecutionMode::NativeRequired),
        value => bail!("{MODE_ENV} must be functions_only, native_preferred, or native_required; got {value}"),
    }
}

fn register_edges(ctx: &SessionContext) -> anyhow::Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
        Field::new("weight", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3, 3])),
            Arc::new(Int64Array::from(vec![2, 3, 1, 4])),
            Arc::new(Float64Array::from(vec![1.0, 1.0, 1.0, 1.0])),
        ],
    )?;
    ctx.register_table("edges", Arc::new(MemTable::try_new(schema, vec![vec![batch]])?))?;
    Ok(())
}

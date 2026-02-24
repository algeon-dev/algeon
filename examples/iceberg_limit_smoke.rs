//! One-off smoke binary that times `SELECT * FROM <table> LIMIT N` against a
//! Glue Iceberg table, used to validate the ranged-fetch scan-limit path.
//!
//! Requires the `iceberg` feature. AWS credentials are read from the standard
//! `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN` env
//! vars. Prints the query elapsed time and returned row count.

use std::time::Instant;

use datafusion::{execution::session_state::SessionStateBuilder, physical_plan::collect, prelude::SessionContext};
use datafusion_cudf::{CudfOptimizerConfig, CudfSessionStateBuilderExt, GlueIcebergTableConfig, register_glue_table};

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let warehouse = std::env::var("SMOKE_WAREHOUSE")
        .unwrap_or_else(|_| "s3://example-warehouse/demo_db/".to_string());
    let database = std::env::var("SMOKE_DATABASE").unwrap_or_else(|_| "demo_db".to_string());
    let table = std::env::var("SMOKE_TABLE").unwrap_or_else(|_| "tpch_sf100_customer".to_string());
    let limit: usize = std::env::var("SMOKE_LIMIT").ok().and_then(|raw| raw.parse().ok()).unwrap_or(10);
    let region = std::env::var("AWS_REGION")
        .or_else(|_| std::env::var("AWS_DEFAULT_REGION"))
        .unwrap_or_else(|_| "us-east-1".to_string());
    let catalog_name = std::env::var("SMOKE_CATALOG_NAME").unwrap_or_else(|_| "glue".to_string());

    eprintln!(
        "[smoke] warehouse={warehouse} database={database} table={table} limit={limit} region={region} catalog={catalog_name}"
    );

    let state =
        SessionStateBuilder::new().with_default_features().with_cudf_native(CudfOptimizerConfig::default()).build();
    let ctx = SessionContext::new_with_state(state);

    let config = GlueIcebergTableConfig::new(&catalog_name, &database, &table, &warehouse, &region);
    register_glue_table(&ctx, table.as_str(), config).await?;

    let sql = format!("SELECT * FROM {table} LIMIT {limit}");
    eprintln!("[smoke] running: {sql}");

    let started = Instant::now();
    let df = ctx.sql(&sql).await?;
    let task_ctx = df.task_ctx();
    let physical = df.clone().create_physical_plan().await?;
    eprintln!("[smoke] physical_plan:\n{}", datafusion::physical_plan::displayable(physical.as_ref()).indent(true));
    let batches = collect(physical.clone(), task_ctx.into()).await?;
    let elapsed = started.elapsed();

    let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    let metrics = physical.metrics();
    let source_fetch_bytes = metric_sum(&metrics, "source_fetch_bytes");
    let source_fetch_count = metric_sum(&metrics, "source_fetch_count");
    let source_fetch_time_ms = metric_duration_ms(&metrics, "source_fetch_time");
    let source_decode_time_ms = metric_duration_ms(&metrics, "source_decode_time");
    println!(
        "[smoke] elapsed_ms={} rows={} batches={} source_fetch_bytes={} source_fetch_count={} source_fetch_time_ms={} source_decode_time_ms={}",
        elapsed.as_millis(),
        rows,
        batches.len(),
        optional_u64(source_fetch_bytes),
        optional_u64(source_fetch_count),
        optional_f64(source_fetch_time_ms),
        optional_f64(source_decode_time_ms)
    );
    Ok(())
}

fn metric_sum(metrics: &Option<datafusion::physical_plan::metrics::MetricsSet>, name: &str) -> Option<u64> {
    metrics.as_ref()?.sum_by_name(name).map(|value| value.as_usize().min(u64::MAX as usize) as u64)
}

fn metric_duration_ms(metrics: &Option<datafusion::physical_plan::metrics::MetricsSet>, name: &str) -> Option<f64> {
    metric_sum(metrics, name).map(|nanos| std::time::Duration::from_nanos(nanos).as_secs_f64() * 1_000.0)
}

fn optional_u64(value: Option<u64>) -> String {
    value.map_or_else(|| "none".to_string(), |value| value.to_string())
}

fn optional_f64(value: Option<f64>) -> String {
    value.map_or_else(|| "none".to_string(), |value| format!("{value:.3}"))
}

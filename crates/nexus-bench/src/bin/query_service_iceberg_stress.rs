#[tokio::main]
async fn main() -> datafusion::error::Result<()> {
    nexus_bench::query_service_iceberg_stress::run().await
}

#[tokio::main]
async fn main() -> datafusion::error::Result<()> {
    bench::query_service_iceberg_stress::run().await
}

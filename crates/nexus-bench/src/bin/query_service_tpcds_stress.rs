#[tokio::main]
async fn main() -> datafusion::error::Result<()> {
    nexus_bench::query_service_tpcds_stress::run().await
}

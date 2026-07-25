#[tokio::main]
async fn main() -> datafusion::error::Result<()> {
    nexus_bench::query_service_tpch_stress::run().await
}

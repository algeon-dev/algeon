#[tokio::main]
async fn main() {
    datafusion_nexus_bench::iceberg_tpch_benchmark::run_main().await
}

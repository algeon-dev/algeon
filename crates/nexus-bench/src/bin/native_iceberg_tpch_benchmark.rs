#[tokio::main]
async fn main() {
    nexus_bench::iceberg_tpch_benchmark::run_main().await
}

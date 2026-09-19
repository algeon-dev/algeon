#[tokio::main]
async fn main() {
    bench::iceberg_tpch_benchmark::run_main().await
}

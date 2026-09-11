use datafusion::common::Result as DataFusionResult;

#[tokio::main]
async fn main() -> DataFusionResult<()> {
    bench::tpch_benchmark::run_main().await
}

use datafusion::common::Result as DataFusionResult;

#[tokio::main]
async fn main() -> DataFusionResult<()> {
    nexus_bench::tpch_benchmark::run_main().await
}

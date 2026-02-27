use datafusion::common::Result as DataFusionResult;

#[tokio::main]
async fn main() -> DataFusionResult<()> {
    datafusion_nexus_bench::tpch_benchmark::run_main().await
}

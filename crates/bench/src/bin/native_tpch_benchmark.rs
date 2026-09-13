use bench::tpch_benchmark::run_main;
use datafusion::common::Result as DataFusionResult;

#[tokio::main]
async fn main() -> DataFusionResult<()> {
    run_main().await
}

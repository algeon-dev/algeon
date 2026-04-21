use std::error::Error;

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<(), Box<dyn Error>> {
    datafusion_nexus_bench::tpcds_three_way_bench::run().await
}

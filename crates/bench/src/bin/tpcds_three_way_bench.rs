use std::error::Error;

use bench::tpcds_three_way_bench::run;

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<(), Box<dyn Error>> {
    run().await
}

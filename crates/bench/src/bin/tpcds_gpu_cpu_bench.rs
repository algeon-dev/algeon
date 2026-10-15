use std::error::Error;

use bench::tpcds_gpu_cpu_bench::run;

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<(), Box<dyn Error>> {
    run().await
}

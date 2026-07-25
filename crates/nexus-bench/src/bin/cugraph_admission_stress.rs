use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    nexus_bench::cugraph_admission_stress::run().await
}

use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    bench::cugraph_admission_stress::run().await
}

use std::error::Error;

use bench::cugraph_admission_stress::run;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    run().await
}

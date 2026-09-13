use std::error::Error;

use bench::cugraph_sql_construction_report::run;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    run().await
}

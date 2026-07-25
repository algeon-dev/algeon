use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    nexus_bench::cugraph_sql_construction_report::run().await
}

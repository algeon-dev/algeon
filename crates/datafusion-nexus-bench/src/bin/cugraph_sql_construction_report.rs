use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    datafusion_nexus_bench::cugraph_sql_construction_report::run().await
}

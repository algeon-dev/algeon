use std::process::ExitCode;

#[tokio::main]
async fn main() -> ExitCode {
    datafusion_nexus_bench::clickbench_benchmark::run_main().await
}

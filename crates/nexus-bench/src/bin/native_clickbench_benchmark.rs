use std::process::ExitCode;

#[tokio::main]
async fn main() -> ExitCode {
    nexus_bench::clickbench_benchmark::run_main().await
}

use std::process::ExitCode;

use bench::clickbench_benchmark::run_main;

#[tokio::main]
async fn main() -> ExitCode {
    run_main().await
}

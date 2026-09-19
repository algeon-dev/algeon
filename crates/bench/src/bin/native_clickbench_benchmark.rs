use std::process::ExitCode;

#[tokio::main]
async fn main() -> ExitCode {
    bench::clickbench_benchmark::run_main().await
}

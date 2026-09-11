//! Open-loop goodput and latency measurement for the fixed-cap admission model.
//!
//! Configuration is environment-variable driven; see
//! `bench::query_service_goodput`. GPU runs must be wrapped in
//! `flock /tmp/cudf-gpu.lock`.

use std::process::ExitCode;

#[tokio::main]
async fn main() -> ExitCode {
    match bench::query_service_goodput::run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("query_service_goodput: {error}");
            // Backend and engine errors carry their reason in the source chain
            // rather than in the outermost message, so printing only the top
            // level turns a specific refusal into "initialization failed".
            let mut source = error.source();
            while let Some(cause) = source {
                eprintln!("  caused by: {cause}");
                source = cause.source();
            }
            ExitCode::FAILURE
        },
    }
}

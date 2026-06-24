use std::{env, error::Error as StdError, io::IsTerminal, process::ExitCode};

use datafusion_nexus::server::{NEXUS_SERVER_LOG_ENV, ServerConfig, ServerError, serve};

fn main() -> ExitCode {
    // Precondition check FIRST — before the tokio runtime or any other code that
    // may have been emitted with elevated SIMD. Catches a CPU below the compiled
    // baseline and prints a clear message instead of an opaque SIGILL.
    ensure_cpu_baseline();
    serve_main()
}

/// Fail fast if the host CPU is below the `x86-64-v2` baseline this binary is
/// compiled for (`target-cpu=x86-64-v2` in `.cargo/config.toml`). The check
/// itself relies only on CPUID (baseline-safe). `ahash` and all other Rust
/// codegen select their instruction set at compile time from `target_feature`,
/// so a too-old CPU would otherwise crash with an illegal instruction.
#[cfg(target_arch = "x86_64")]
fn ensure_cpu_baseline() {
    // SSE4.2 + POPCNT are representative of the x86-64-v2 feature set.
    let supported =
        std::arch::is_x86_feature_detected!("sse4.2") && std::arch::is_x86_feature_detected!("popcnt");
    if !supported {
        eprintln!(
            "FATAL: datafusion-nexus was built for the x86-64-v2 CPU baseline (SSE4.2 + POPCNT) but \
             this host CPU does not support it. Run on a newer CPU, or rebuild with a lower \
             target-cpu floor in .cargo/config.toml."
        );
        std::process::exit(1);
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn ensure_cpu_baseline() {}

#[tokio::main]
async fn serve_main() -> ExitCode {
    if let Err(error) = init_tracing() {
        eprintln!("ERROR datafusion-nexus Flight SQL server failed to initialize logging: {error}");
        return ExitCode::FAILURE;
    }
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            if let Some(source) = StdError::source(&error) {
                tracing::debug!(source = %source, "datafusion-nexus Flight SQL server failure source");
            }
            tracing::error!(
                error_code = error.code().as_str(),
                error_kind = error.kind().as_str(),
                error_status = error.status().as_str(),
                error_operation = error.operation(),
                error = %error,
                "datafusion-nexus Flight SQL server failed"
            );
            ExitCode::FAILURE
        },
    }
}

fn init_tracing() -> Result<(), Box<dyn StdError + Send + Sync>> {
    let configured_log_filter = env::var(NEXUS_SERVER_LOG_ENV).unwrap_or_else(|_| "info".to_string());
    let log_filter = format!("{configured_log_filter},datafusion_nexus_server=error");
    tracing_subscriber::fmt().with_env_filter(log_filter).with_ansi(std::io::stderr().is_terminal()).try_init()?;
    Ok(())
}

async fn run() -> Result<(), ServerError> {
    let config = ServerConfig::from_env()?;
    serve(config).await?;
    Ok(())
}

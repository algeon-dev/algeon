use std::{env, error::Error as StdError, io::IsTerminal, process::ExitCode};

use algeon_datafusion::{
    backend::GpuMemoryOwnership,
    error::ErrorLogDiagnostics,
    server::{ALGEON_SERVER_LOG_ENV, ServerConfig, ServerError, serve},
};

fn main() -> ExitCode {
    serve_main()
}

#[tokio::main]
async fn serve_main() -> ExitCode {
    if let Err(error) = init_tracing() {
        eprintln!("ERROR Algeon Flight SQL server failed to initialize logging: {error}");
        return ExitCode::FAILURE;
    }
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            log_server_error(&error);
            ExitCode::FAILURE
        },
    }
}

fn log_server_error(error: &ServerError) {
    let diagnostics = ErrorLogDiagnostics::from_error(error);
    tracing::error!(
        target: "algeon_server_bin",
        error_code = error.code().as_str(),
        error_kind = error.kind().as_str(),
        error_status = error.status().as_str(),
        error_operation = error.operation(),
        error_message = error.message(),
        error_cause = diagnostics.cause().filter(|cause| *cause != error.message()),
        error_source_path = diagnostics.source_path(),
        error_diagnostics_truncated = diagnostics.truncated().then_some(true),
        "Algeon Flight SQL server failed"
    );
}

fn init_tracing() -> Result<(), Box<dyn StdError + Send + Sync>> {
    let configured_log_filter = env::var(ALGEON_SERVER_LOG_ENV).unwrap_or_else(|_| "info".to_string());
    let log_filter = format!("{configured_log_filter},algeon_server_bin=error");
    tracing_subscriber::fmt().with_env_filter(log_filter).with_ansi(std::io::stderr().is_terminal()).try_init()?;
    Ok(())
}

async fn run() -> Result<(), ServerError> {
    let config = ServerConfig::from_env()?;
    let ownership = config
        .gpu_devices
        .iter()
        .copied()
        .map(|ordinal| (ordinal, GpuMemoryOwnership::WholeDeviceExclusive))
        .collect::<Vec<_>>();
    let config = config.with_gpu_memory_ownership(ownership);
    serve(config).await?;
    Ok(())
}

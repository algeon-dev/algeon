use std::{env, error::Error as StdError, io::IsTerminal, process::ExitCode};

use datafusion_nexus::server::{NEXUS_SERVER_LOG_ENV, ServerConfig, ServerError, serve};

#[tokio::main]
async fn main() -> ExitCode {
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

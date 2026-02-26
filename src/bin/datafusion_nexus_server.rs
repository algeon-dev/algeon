use std::io::IsTerminal;

use datafusion_nexus::server::{ServerConfig, serve};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let config = ServerConfig::from_env()?;
    tracing_subscriber::fmt()
        .with_env_filter(config.log_filter.clone())
        .with_ansi(std::io::stderr().is_terminal())
        .try_init()?;
    serve(config).await?;
    Ok(())
}

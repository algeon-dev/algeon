use std::time::Duration;

use tokio::time::sleep;
use tonic::transport::{Channel, Endpoint, Error as TransportError};

pub(crate) type ConnectionResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Connection retry policy shared by the explicit operator commands: a fixed
/// number of connection attempts separated by a fixed backoff.
#[derive(Clone, Copy)]
pub(crate) struct ConnectionRetryConfig {
    attempts: usize,
    backoff: Duration,
}

impl ConnectionRetryConfig {
    pub(crate) fn fixed_attempts(attempts: usize, backoff: Duration) -> Self {
        assert!(attempts > 0, "connection retry attempts must be positive");
        Self { attempts, backoff }
    }
}

/// Connects a tonic channel under a caller-selected retry contract.
///
/// `map_channel` retains the caller's desired result type (a raw `Channel` or
/// a typed Flight client), and `connection_failure` preserves the
/// command-specific terminal error context.
pub(crate) async fn connect_with_retry<T>(
    url: &str,
    config: ConnectionRetryConfig,
    map_channel: impl FnOnce(Channel) -> T,
    connection_failure: impl Fn(&TransportError) -> String,
) -> ConnectionResult<T> {
    let endpoint = Endpoint::from_shared(url.to_string())?;
    let mut failed_attempts = 0;

    loop {
        match endpoint.connect().await {
            Ok(channel) => return Ok(map_channel(channel)),
            Err(error) => {
                failed_attempts += 1;
                // Callers historically slept after every failed attempt,
                // including the final one; keep that retry window intact.
                sleep(config.backoff).await;
                if failed_attempts >= config.attempts {
                    return Err(connection_failure(&error).into());
                }
            },
        }
    }
}

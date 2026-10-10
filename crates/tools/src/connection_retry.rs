use std::time::{Duration, Instant};

use tokio::time::sleep;
use tonic::transport::{Channel, Endpoint, Error as TransportError};

pub(crate) type ConnectionResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Connection retry policy shared by the explicit operator commands: attempts
/// separated by a fixed backoff, bounded by an attempt count or a total time
/// budget.
#[derive(Clone, Copy)]
pub(crate) struct ConnectionRetryConfig {
    limit: RetryLimit,
    backoff: Duration,
}

#[derive(Clone, Copy)]
enum RetryLimit {
    Attempts(usize),
    Budget(Duration),
}

impl ConnectionRetryConfig {
    pub(crate) fn fixed_attempts(attempts: usize, backoff: Duration) -> Self {
        assert!(attempts > 0, "connection retry attempts must be positive");
        Self { limit: RetryLimit::Attempts(attempts), backoff }
    }

    /// Retries until `budget` has elapsed since the first attempt. Each TCP
    /// connect is capped by the remaining budget, so an unreachable host
    /// cannot hold one attempt past the deadline.
    pub(crate) fn time_budget(budget: Duration, backoff: Duration) -> Self {
        assert!(!budget.is_zero(), "connection retry budget must be positive");
        Self { limit: RetryLimit::Budget(budget), backoff }
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
    connect_with_retry_reporting(url, config, map_channel, connection_failure, |_| {}).await
}

/// [`connect_with_retry`] that calls `on_retry` with the time elapsed since the
/// first attempt before each retry.
pub(crate) async fn connect_with_retry_reporting<T>(
    url: &str,
    config: ConnectionRetryConfig,
    map_channel: impl FnOnce(Channel) -> T,
    connection_failure: impl Fn(&TransportError) -> String,
    mut on_retry: impl FnMut(Duration),
) -> ConnectionResult<T> {
    let endpoint = Endpoint::from_shared(url.to_string())?;
    let started = Instant::now();
    let mut failed_attempts = 0;

    loop {
        let attempt = match config.limit {
            RetryLimit::Attempts(_) => endpoint.clone(),
            RetryLimit::Budget(budget) => endpoint.clone().connect_timeout(budget.saturating_sub(started.elapsed())),
        };
        match attempt.connect().await {
            Ok(channel) => return Ok(map_channel(channel)),
            Err(error) => {
                failed_attempts += 1;
                match config.limit {
                    RetryLimit::Attempts(attempts) => {
                        // Callers historically slept after every failed attempt,
                        // including the final one; keep that retry window intact.
                        sleep(config.backoff).await;
                        if failed_attempts >= attempts {
                            return Err(connection_failure(&error).into());
                        }
                    },
                    RetryLimit::Budget(budget) => {
                        if started.elapsed() + config.backoff >= budget {
                            return Err(connection_failure(&error).into());
                        }
                        sleep(config.backoff).await;
                    },
                }
                on_retry(started.elapsed());
            },
        }
    }
}

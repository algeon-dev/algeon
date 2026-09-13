use std::time::{Duration, Instant};

use tokio::time::sleep;
use tonic::transport::{Channel, Endpoint, Error as TransportError};

pub(crate) type ConnectionResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

#[derive(Clone, Copy)]
enum RetryLimit {
    Timeout(Duration),
    Attempts(usize),
}

#[derive(Clone, Copy)]
enum RetryBackoff {
    Fixed(Duration),
    Exponential { initial: Duration, maximum: Duration },
}

/// Connection retry policy shared by the explicit operator/E2E commands.
///
/// Timeout policies preserve deadline-based callers, while attempt policies
/// preserve commands whose retry budget is expressed as a fixed number of
/// connection attempts. Exponential backoff doubles up to the configured cap.
#[derive(Clone, Copy)]
pub(crate) struct ConnectionRetryConfig {
    limit: RetryLimit,
    backoff: RetryBackoff,
}

impl ConnectionRetryConfig {
    pub(crate) const fn fixed_timeout(timeout: Duration, backoff: Duration) -> Self {
        Self { limit: RetryLimit::Timeout(timeout), backoff: RetryBackoff::Fixed(backoff) }
    }

    pub(crate) fn fixed_attempts(attempts: usize, backoff: Duration) -> Self {
        assert!(attempts > 0, "connection retry attempts must be positive");
        Self { limit: RetryLimit::Attempts(attempts), backoff: RetryBackoff::Fixed(backoff) }
    }

    pub(crate) const fn exponential_timeout(timeout: Duration, initial: Duration, maximum: Duration) -> Self {
        Self { limit: RetryLimit::Timeout(timeout), backoff: RetryBackoff::Exponential { initial, maximum } }
    }

    fn initial_backoff(self) -> Duration {
        match self.backoff {
            RetryBackoff::Fixed(backoff) => backoff,
            RetryBackoff::Exponential { initial, .. } => initial,
        }
    }

    fn next_backoff(self, current: Duration) -> Duration {
        match self.backoff {
            RetryBackoff::Fixed(backoff) => backoff,
            RetryBackoff::Exponential { maximum, .. } => (current * 2).min(maximum),
        }
    }
}

pub(crate) enum BeforeConnect {
    Continue,
    Abort(String),
}

/// Connects a tonic channel under a caller-selected retry contract.
///
/// `before_connect` lets process-owning callers fail immediately when their
/// child exits. `map_channel` retains the caller's desired result type (a raw
/// `Channel` or a typed Flight client), and `connection_failure` preserves the
/// command-specific terminal error context.
pub(crate) async fn connect_with_retry<T>(
    url: &str,
    config: ConnectionRetryConfig,
    mut before_connect: impl FnMut() -> ConnectionResult<BeforeConnect>,
    map_channel: impl FnOnce(Channel) -> T,
    connection_failure: impl Fn(&TransportError) -> String,
) -> ConnectionResult<T> {
    let endpoint = Endpoint::from_shared(url.to_string())?;
    let deadline = match config.limit {
        RetryLimit::Timeout(timeout) => Some(Instant::now() + timeout),
        RetryLimit::Attempts(_) => None,
    };
    let mut failed_attempts = 0;
    let mut backoff = config.initial_backoff();

    loop {
        match before_connect()? {
            BeforeConnect::Continue => {},
            BeforeConnect::Abort(message) => return Err(message.into()),
        }

        match endpoint.connect().await {
            Ok(channel) => return Ok(map_channel(channel)),
            Err(error) => {
                failed_attempts += 1;
                let exhausted = match config.limit {
                    RetryLimit::Timeout(_) => deadline.is_some_and(|deadline| Instant::now() >= deadline),
                    RetryLimit::Attempts(attempts) => failed_attempts >= attempts,
                };
                if exhausted {
                    // The fixed-attempt callers historically slept after every
                    // failed attempt, including the final one. Keep that retry
                    // window intact; timeout callers fail immediately once the
                    // deadline has elapsed.
                    if matches!(config.limit, RetryLimit::Attempts(_)) {
                        sleep(backoff).await;
                    }
                    return Err(connection_failure(&error).into());
                }
            },
        }

        sleep(backoff).await;
        backoff = config.next_backoff(backoff);
    }
}

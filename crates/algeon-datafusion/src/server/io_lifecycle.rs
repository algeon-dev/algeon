use std::{
    result::Result as StdResult,
    sync::{
        Arc, OnceLock,
        atomic::{AtomicUsize, Ordering},
    },
    time::Instant,
};

use cudf::io::drain_kvikio_bounce_buffer_pools;
use query_planning::error::{QueryError, QueryErrorCode};
use query_runtime::runtime::IoLifecycle;
use tokio::task::spawn_blocking;

use crate::server::{
    adapter_error::Error as AdapterError,
    error::{Result, ServerError},
};

/// Owns one server generation's native Parquet I/O gate and KvikIO drain.
///
/// Native execution configs receive only the cloneable `IoLifecycle` handle.
/// Dropping this owner closes acquisition without blocking; the supervised
/// shutdown path additionally waits for every lease and performs one drain.
#[derive(Debug)]
pub(super) struct ServerIoLifecycle {
    lifecycle: IoLifecycle,
    drain_result: OnceLock<StdResult<u64, Arc<AdapterError>>>,
    drain_invocations: AtomicUsize,
    #[cfg(test)]
    kvikio_drain_delay: std::sync::Mutex<Option<std::time::Duration>>,
    #[cfg(test)]
    drain_attempts: AtomicUsize,
}

impl ServerIoLifecycle {
    pub(super) fn new() -> Self {
        Self {
            lifecycle: IoLifecycle::new(),
            drain_result: OnceLock::new(),
            drain_invocations: AtomicUsize::new(0),
            #[cfg(test)]
            kvikio_drain_delay: std::sync::Mutex::new(None),
            #[cfg(test)]
            drain_attempts: AtomicUsize::new(0),
        }
    }

    pub(super) fn handle(&self) -> IoLifecycle {
        self.lifecycle.clone()
    }

    pub(super) fn close(&self) {
        self.lifecycle.close();
    }

    pub(super) async fn close_wait_and_drain_until(self: &Arc<Self>, deadline: Instant) -> Result<u64> {
        self.close();
        let owner = Arc::clone(self);
        spawn_blocking(move || owner.close_wait_and_drain_blocking(deadline)).await.map_err(|error| {
            ServerError::shutdown_with_source(
                "parquet_io_cleanup_task_failed",
                "server Parquet I/O cleanup task failed",
                error,
            )
        })?
    }

    fn close_wait_and_drain_blocking(&self, deadline: Instant) -> Result<u64> {
        #[cfg(test)]
        self.drain_attempts.fetch_add(1, Ordering::AcqRel);
        self.lifecycle
            .close_and_wait_for_idle_until(deadline, "algeon_datafusion.server.parquet_io.shutdown")
            .map_err(|error| {
                ServerError::session("server Parquet I/O did not quiesce before shutdown deadline", error)
                    .with_fact("shutdown_component", "parquet_io")
            })?;
        let cleared_bytes = match self.drain_result.get_or_init(|| {
            self.drain_invocations.fetch_add(1, Ordering::AcqRel);
            #[cfg(test)]
            if let Some(delay) = *self.kvikio_drain_delay.lock().unwrap() {
                std::thread::sleep(delay);
            }
            drain_kvikio_bounce_buffer_pools().map_err(|error| {
                Arc::new(AdapterError::from_cudf(
                    "algeon_datafusion.server.parquet_io.drain",
                    "failed to drain KvikIO bounce buffers after server Parquet I/O quiesced",
                    error,
                ))
            })
        }) {
            Ok(cleared_bytes) => *cleared_bytes,
            Err(error) => {
                return Err(ServerError::shutdown_from_shared_adapter_error(
                    "kvikio_bounce_buffer_drain_failed",
                    "failed to drain KvikIO bounce buffers after server Parquet I/O quiesced",
                    Arc::clone(error),
                )
                .with_fact("shutdown_component", "parquet_io"));
            },
        };
        if Instant::now() >= deadline {
            return Err(kvikio_drain_timeout());
        }
        Ok(cleared_bytes)
    }

    #[cfg(test)]
    pub(super) fn drain_invocations(&self) -> usize {
        self.drain_invocations.load(Ordering::Acquire)
    }

    #[cfg(test)]
    fn set_kvikio_drain_delay_for_test(&self, delay: std::time::Duration) {
        *self.kvikio_drain_delay.lock().unwrap() = Some(delay);
    }

    #[cfg(test)]
    pub(super) fn drain_attempts(&self) -> usize {
        self.drain_attempts.load(Ordering::Acquire)
    }
}

fn kvikio_drain_timeout() -> ServerError {
    ServerError::session(
        "KvikIO drain exceeded the server shutdown deadline",
        QueryError::new(
            QueryErrorCode::ServiceOverloaded,
            "algeon_datafusion.server.parquet_io.drain",
            "timed out draining KvikIO bounce-buffer pools",
        )
        .with_fact("reason", "kvikio_drain_timeout"),
    )
    .with_fact("shutdown_component", "parquet_io")
}

impl Drop for ServerIoLifecycle {
    fn drop(&mut self) {
        self.lifecycle.close();
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{Arc, mpsc},
        time::{Duration, Instant},
    };

    use super::*;

    static KVIKIO_DRAIN_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn shutdown_timeout_preserves_structured_lifecycle_facts() {
        let owner = Arc::new(ServerIoLifecycle::new());
        let lease = owner.handle().acquire("algeon_datafusion.test.parquet_io").unwrap();

        let error = owner
            .close_wait_and_drain_blocking(Instant::now())
            .expect_err("a live Parquet I/O lease must consume the shared shutdown deadline");

        assert_eq!(error.fact_value("reason"), Some("io_lifecycle_shutdown_timeout"));
        assert_eq!(error.fact_value("active_io_leases"), Some("1"));
        assert_eq!(error.fact_value("shutdown_component"), Some("parquet_io"));
        assert_eq!(owner.drain_invocations(), 0, "KvikIO drain must not race active native I/O");
        drop(lease);
    }

    #[test]
    fn in_flight_lease_blocks_exactly_one_drain_and_replacement_generation_is_open() {
        let _serial = KVIKIO_DRAIN_TEST_LOCK.lock().unwrap();
        let owner = Arc::new(ServerIoLifecycle::new());
        let lease = owner.handle().acquire("algeon_datafusion.test.parquet_io").unwrap();
        owner.close();
        let closed = owner.handle().acquire("algeon_datafusion.test.parquet_io.after_close").unwrap_err();
        assert_eq!(closed.fact_value("reason"), Some("io_lifecycle_closed"));

        let (finished_tx, finished_rx) = mpsc::channel();
        let draining_owner = Arc::clone(&owner);
        let drain = std::thread::spawn(move || {
            let result = draining_owner.close_wait_and_drain_blocking(Instant::now() + Duration::from_secs(10));
            finished_tx.send(()).unwrap();
            result
        });
        assert!(
            finished_rx.recv_timeout(Duration::from_millis(20)).is_err(),
            "server shutdown must wait while pre-close Parquet I/O is live"
        );
        assert_eq!(owner.drain_invocations(), 0);

        drop(lease);
        finished_rx.recv_timeout(Duration::from_secs(10)).expect("drain must finish after the final lease drops");
        let first_cleared_bytes = drain.join().unwrap().expect("idle KvikIO drain succeeds");
        assert_eq!(owner.drain_invocations(), 1);
        assert_eq!(
            owner
                .close_wait_and_drain_blocking(Instant::now() + Duration::from_secs(10))
                .expect("repeated shutdown reuses the stored drain result"),
            first_cleared_bytes
        );
        assert_eq!(owner.drain_invocations(), 1, "one server generation drains KvikIO at most once");

        let replacement = ServerIoLifecycle::new();
        let replacement_lease = replacement
            .handle()
            .acquire("algeon_datafusion.test.parquet_io.replacement")
            .expect("a replacement server generation starts with an open I/O gate");
        drop(replacement_lease);
    }

    #[test]
    fn successful_kvikio_ffi_overrun_returns_structured_drain_timeout() {
        let _serial = KVIKIO_DRAIN_TEST_LOCK.lock().unwrap();
        let owner = Arc::new(ServerIoLifecycle::new());
        owner.set_kvikio_drain_delay_for_test(Duration::from_millis(40));

        let error = owner
            .close_wait_and_drain_blocking(Instant::now() + Duration::from_millis(10))
            .expect_err("successful non-preemptible KvikIO drain beyond the deadline must not report success");

        assert_eq!(error.fact_value("reason"), Some("kvikio_drain_timeout"));
        assert_eq!(error.fact_value("shutdown_component"), Some("parquet_io"));
        assert_eq!(owner.drain_invocations(), 1);
    }
}

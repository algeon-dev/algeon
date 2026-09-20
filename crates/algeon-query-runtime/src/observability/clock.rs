//! Clock abstraction for query-relative runtime observations.

use std::{
    sync::atomic::{AtomicU64, Ordering},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

/// Clock sampled by a runtime observation session while holding recorder order.
pub trait RuntimeObservationClock: Send + Sync + 'static {
    /// Query-relative monotonic nanoseconds.
    fn monotonic_ns(&self) -> u64;

    /// Optional Unix epoch anchor captured once for cross-artifact alignment.
    fn unix_epoch_ns(&self) -> Option<u64>;
}

/// System monotonic clock with one wall-clock anchor.
#[derive(Debug)]
pub(crate) struct SystemRuntimeObservationClock {
    origin: Instant,
    unix_epoch_ns: Option<u64>,
}

impl SystemRuntimeObservationClock {
    /// Captures a new monotonic origin and optional Unix epoch anchor.
    pub fn new() -> Self {
        let unix_epoch_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .ok()
            .and_then(|duration| u64::try_from(duration.as_nanos()).ok());
        Self { origin: Instant::now(), unix_epoch_ns }
    }
}

impl Default for SystemRuntimeObservationClock {
    fn default() -> Self {
        Self::new()
    }
}

impl RuntimeObservationClock for SystemRuntimeObservationClock {
    fn monotonic_ns(&self) -> u64 {
        u64::try_from(self.origin.elapsed().as_nanos()).unwrap_or(u64::MAX)
    }

    fn unix_epoch_ns(&self) -> Option<u64> {
        self.unix_epoch_ns
    }
}

/// Deterministic clock for contract tests and controlled simulations.
#[derive(Debug)]
pub struct DeterministicRuntimeObservationClock {
    monotonic_ns: AtomicU64,
    unix_epoch_ns: Option<u64>,
}

impl DeterministicRuntimeObservationClock {
    /// Creates a deterministic clock at zero query-relative nanoseconds.
    pub const fn new(unix_epoch_ns: Option<u64>) -> Self {
        Self { monotonic_ns: AtomicU64::new(0), unix_epoch_ns }
    }

    /// Sets the next sampled query-relative value.
    pub fn set_monotonic_ns(&self, monotonic_ns: u64) {
        self.monotonic_ns.store(monotonic_ns, Ordering::Release);
    }

    /// Advances query-relative time and returns the new value.
    pub fn advance(&self, nanoseconds: u64) -> u64 {
        self.monotonic_ns.fetch_add(nanoseconds, Ordering::AcqRel).saturating_add(nanoseconds)
    }
}

impl RuntimeObservationClock for DeterministicRuntimeObservationClock {
    fn monotonic_ns(&self) -> u64 {
        self.monotonic_ns.load(Ordering::Acquire)
    }

    fn unix_epoch_ns(&self) -> Option<u64> {
        self.unix_epoch_ns
    }
}

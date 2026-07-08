//! Signal-only device memory allocation-failure counters.

use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};

use cudf::{DeviceMemoryFailureAction, DeviceMemoryFailureSource, DeviceMemoryPoolUsage};

/// Per-device allocation-failure counters observed by the device memory
/// manager failure callback.
#[derive(Debug, Clone)]
pub struct DeviceMemoryFailureTracker {
    inner: Arc<DeviceMemoryFailureTrackerInner>,
}

#[derive(Debug)]
struct DeviceMemoryFailureTrackerInner {
    device_ordinal: usize,
    callback_installed: AtomicBool,
    callback_install_unavailable: AtomicBool,
    allocation_failure_count: AtomicU64,
    total_failed_allocation_bytes: AtomicU64,
    largest_failed_allocation_bytes: AtomicU64,
    trim_retry_count: AtomicU64,
    pool_usage_observation_count: AtomicU64,
    last_failed_allocation_bytes: AtomicU64,
    last_pool_reserved_current_bytes: AtomicU64,
    last_pool_used_current_bytes: AtomicU64,
}

/// Point-in-time device memory allocation-failure report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceMemoryFailureReport {
    pub device_ordinal: usize,
    pub callback_installed: bool,
    pub callback_install_unavailable: bool,
    pub allocation_failure_count: u64,
    pub total_failed_allocation_bytes: u64,
    pub largest_failed_allocation_bytes: u64,
    pub trim_retry_count: u64,
    pub pool_usage_observation_count: u64,
    pub last_failed_allocation_bytes: u64,
    pub last_pool_reserved_current_bytes: u64,
    pub last_pool_used_current_bytes: u64,
}

impl DeviceMemoryFailureTracker {
    /// Creates a zeroed tracker for one CUDA device ordinal.
    pub fn new(device_ordinal: usize) -> Self {
        Self {
            inner: Arc::new(DeviceMemoryFailureTrackerInner {
                device_ordinal,
                callback_installed: AtomicBool::new(false),
                callback_install_unavailable: AtomicBool::new(false),
                allocation_failure_count: AtomicU64::new(0),
                total_failed_allocation_bytes: AtomicU64::new(0),
                largest_failed_allocation_bytes: AtomicU64::new(0),
                trim_retry_count: AtomicU64::new(0),
                pool_usage_observation_count: AtomicU64::new(0),
                last_failed_allocation_bytes: AtomicU64::new(0),
                last_pool_reserved_current_bytes: AtomicU64::new(0),
                last_pool_used_current_bytes: AtomicU64::new(0),
            }),
        }
    }

    /// Returns the process-global tracker for one CUDA device ordinal.
    ///
    /// cudf-rs installs at most one callback-bearing device manager per
    /// process/device, so the Nexus counters must have the same lifetime and
    /// sharing semantics. Multiple QueryServices for one device therefore
    /// report the same callback counters.
    pub fn for_device(device_ordinal: usize) -> Self {
        static TRACKERS: OnceLock<Mutex<HashMap<usize, DeviceMemoryFailureTracker>>> = OnceLock::new();
        let trackers = TRACKERS.get_or_init(|| Mutex::new(HashMap::new()));
        match trackers.lock() {
            Ok(mut trackers) => trackers.entry(device_ordinal).or_insert_with(|| Self::new(device_ordinal)).clone(),
            Err(_error) => Self::new(device_ordinal),
        }
    }

    /// CUDA device ordinal this tracker describes.
    pub fn device_ordinal(&self) -> usize {
        self.inner.device_ordinal
    }

    /// Returns a re-entrancy-safe callback for cudf-rs/RMM.
    ///
    /// The callback records counters and asks cudf-rs to trim idle async-pool
    /// pages only when the failure payload shows enough reclaimable idle
    /// memory for one trim retry. It must remain allocation-free and lock-free
    /// because RMM invokes it inside allocation failure handling.
    pub(crate) fn failure_callback(
        &self,
    ) -> Arc<dyn Fn(cudf::DeviceMemoryFailureInfo) -> DeviceMemoryFailureAction + Send + Sync + 'static> {
        let tracker = self.clone();
        Arc::new(move |info| {
            if info.source != DeviceMemoryFailureSource::Physical {
                return DeviceMemoryFailureAction::Decline;
            }
            let trim_retry = info.should_trim_and_retry();
            tracker.record_allocation_failure(info.requested_bytes, info.pool_usage, trim_retry);
            if trim_retry { DeviceMemoryFailureAction::TrimAndRetry } else { DeviceMemoryFailureAction::Decline }
        })
    }

    pub(crate) fn mark_callback_installed(&self) {
        self.inner.callback_installed.store(true, Ordering::Release);
        self.inner.callback_install_unavailable.store(false, Ordering::Release);
    }

    pub(crate) fn mark_callback_install_unavailable(&self) {
        self.inner.callback_installed.store(false, Ordering::Release);
        self.inner.callback_install_unavailable.store(true, Ordering::Release);
    }

    fn record_allocation_failure(&self, bytes: u64, pool_usage: Option<DeviceMemoryPoolUsage>, trim_retry: bool) {
        self.inner.allocation_failure_count.fetch_add(1, Ordering::Relaxed);
        self.inner.total_failed_allocation_bytes.fetch_add(bytes, Ordering::Relaxed);
        self.inner.last_failed_allocation_bytes.store(bytes, Ordering::Relaxed);
        update_max(&self.inner.largest_failed_allocation_bytes, bytes);
        if let Some(pool_usage) = pool_usage {
            self.inner.pool_usage_observation_count.fetch_add(1, Ordering::Relaxed);
            self.inner.last_pool_reserved_current_bytes.store(pool_usage.reserved_current_bytes, Ordering::Relaxed);
            self.inner.last_pool_used_current_bytes.store(pool_usage.used_current_bytes, Ordering::Relaxed);
        }
        if trim_retry {
            self.inner.trim_retry_count.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Returns a point-in-time report.
    pub fn report(&self) -> DeviceMemoryFailureReport {
        DeviceMemoryFailureReport {
            device_ordinal: self.inner.device_ordinal,
            callback_installed: self.inner.callback_installed.load(Ordering::Acquire),
            callback_install_unavailable: self.inner.callback_install_unavailable.load(Ordering::Acquire),
            allocation_failure_count: self.inner.allocation_failure_count.load(Ordering::Acquire),
            total_failed_allocation_bytes: self.inner.total_failed_allocation_bytes.load(Ordering::Acquire),
            largest_failed_allocation_bytes: self.inner.largest_failed_allocation_bytes.load(Ordering::Acquire),
            trim_retry_count: self.inner.trim_retry_count.load(Ordering::Acquire),
            pool_usage_observation_count: self.inner.pool_usage_observation_count.load(Ordering::Acquire),
            last_failed_allocation_bytes: self.inner.last_failed_allocation_bytes.load(Ordering::Acquire),
            last_pool_reserved_current_bytes: self.inner.last_pool_reserved_current_bytes.load(Ordering::Acquire),
            last_pool_used_current_bytes: self.inner.last_pool_used_current_bytes.load(Ordering::Acquire),
        }
    }
}

fn update_max(target: &AtomicU64, candidate: u64) {
    let mut current = target.load(Ordering::Relaxed);
    while candidate > current {
        match target.compare_exchange_weak(current, candidate, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return,
            Err(next) => current = next,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failure_callback_records_counters_and_declines_retry() {
        let tracker = DeviceMemoryFailureTracker::new(3);

        tracker.record_allocation_failure(10, None, false);
        tracker.record_allocation_failure(
            40,
            Some(DeviceMemoryPoolUsage {
                reserved_current_bytes: 400,
                reserved_high_bytes: 800,
                used_current_bytes: 100,
                used_high_bytes: 700,
            }),
            true,
        );
        tracker.record_allocation_failure(20, None, false);

        let report = tracker.report();
        assert_eq!(report.device_ordinal, 3);
        assert!(!report.callback_installed);
        assert_eq!(report.allocation_failure_count, 3);
        assert_eq!(report.total_failed_allocation_bytes, 70);
        assert_eq!(report.largest_failed_allocation_bytes, 40);
        assert_eq!(report.trim_retry_count, 1);
        assert_eq!(report.pool_usage_observation_count, 1);
        assert_eq!(report.last_failed_allocation_bytes, 20);
        assert_eq!(report.last_pool_reserved_current_bytes, 400);
        assert_eq!(report.last_pool_used_current_bytes, 100);
    }

    #[test]
    fn failure_callback_declines_reservation_limit_without_recording_device_failure() {
        let tracker = DeviceMemoryFailureTracker::new(3);
        let callback = tracker.failure_callback();

        let action = callback(cudf::DeviceMemoryFailureInfo {
            device: cudf::DeviceId::DEFAULT,
            source: DeviceMemoryFailureSource::ReservationLimit,
            requested_bytes: 10,
            callback_attempt: 0,
            trim_retry_count: 0,
            pool_usage: Some(DeviceMemoryPoolUsage {
                reserved_current_bytes: 1_000,
                reserved_high_bytes: 1_000,
                used_current_bytes: 0,
                used_high_bytes: 0,
            }),
        });

        assert_eq!(action, DeviceMemoryFailureAction::Decline);
        let report = tracker.report();
        assert_eq!(report.allocation_failure_count, 0);
        assert_eq!(report.trim_retry_count, 0);
        assert_eq!(report.pool_usage_observation_count, 0);
    }

    #[test]
    fn failure_callback_trims_and_records_physical_failure_with_idle_pool() {
        let tracker = DeviceMemoryFailureTracker::new(3);
        let callback = tracker.failure_callback();

        let action = callback(cudf::DeviceMemoryFailureInfo {
            device: cudf::DeviceId::DEFAULT,
            source: DeviceMemoryFailureSource::Physical,
            requested_bytes: 10,
            callback_attempt: 0,
            trim_retry_count: 0,
            pool_usage: Some(DeviceMemoryPoolUsage {
                reserved_current_bytes: 1_000,
                reserved_high_bytes: 1_000,
                used_current_bytes: 0,
                used_high_bytes: 0,
            }),
        });

        assert_eq!(action, DeviceMemoryFailureAction::TrimAndRetry);
        let report = tracker.report();
        assert_eq!(report.allocation_failure_count, 1);
        assert_eq!(report.trim_retry_count, 1);
        assert_eq!(report.pool_usage_observation_count, 1);
        assert_eq!(report.last_failed_allocation_bytes, 10);
    }
}

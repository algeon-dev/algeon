//! Signal-only device memory allocation-failure counters.

use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};

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
    /// The callback records counters and always declines retry. It must remain
    /// allocation-free and lock-free because RMM invokes it inside allocation
    /// failure handling.
    pub(crate) fn failure_callback(&self) -> Arc<dyn Fn(usize) -> bool + Send + Sync + 'static> {
        let tracker = self.clone();
        Arc::new(move |bytes| {
            tracker.record_allocation_failure(bytes as u64);
            false
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

    fn record_allocation_failure(&self, bytes: u64) {
        self.inner.allocation_failure_count.fetch_add(1, Ordering::Relaxed);
        self.inner.total_failed_allocation_bytes.fetch_add(bytes, Ordering::Relaxed);
        update_max(&self.inner.largest_failed_allocation_bytes, bytes);
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
        let callback = tracker.failure_callback();

        assert!(!callback(10));
        assert!(!callback(40));
        assert!(!callback(20));

        let report = tracker.report();
        assert_eq!(report.device_ordinal, 3);
        assert!(!report.callback_installed);
        assert_eq!(report.allocation_failure_count, 3);
        assert_eq!(report.total_failed_allocation_bytes, 70);
        assert_eq!(report.largest_failed_allocation_bytes, 40);
    }
}

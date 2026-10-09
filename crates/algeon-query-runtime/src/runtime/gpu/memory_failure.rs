//! Signal-only device memory allocation-failure counters.
//!
//! Signal-only means the counters inform reports and never admission: the
//! ledger decides capacity from its own integer accounting, and a failed
//! allocation is already a typed error at its source. The counters are
//! written by the callback the device memory manager invokes inside RMM's
//! failure handling, so that callback allocates nothing and takes no lock.
//!
//! The manager and its callback are process-lifetime, while backends are
//! constructed per generation. The registry below therefore keeps one
//! tracker per ordinal before installation (`pending`) and rebinds it to the
//! manager's installation generation afterwards (`installed`). A later backend
//! recovers the tracker the live callback already points at, because the
//! callback would never update a second counter.

use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};

use cudf::{DeviceMemoryFailureSource, DeviceMemoryPoolUsage};
use query_planning::error::{QueryError, QueryErrorCode, Result};
use serde::Serialize;

const TRACKER_LOOKUP_OPERATION: &str = "query_engine.gpu.memory_failure_tracker.lookup";

#[derive(Debug, Default)]
struct DeviceMemoryFailureTrackerRegistryState {
    pending: HashMap<usize, DeviceMemoryFailureTracker>,
    installed: HashMap<(usize, u64), DeviceMemoryFailureTracker>,
}

type DeviceMemoryFailureTrackerRegistry = Mutex<DeviceMemoryFailureTrackerRegistryState>;

fn device_memory_failure_trackers() -> &'static DeviceMemoryFailureTrackerRegistry {
    static TRACKERS: OnceLock<DeviceMemoryFailureTrackerRegistry> = OnceLock::new();
    TRACKERS.get_or_init(|| Mutex::new(DeviceMemoryFailureTrackerRegistryState::default()))
}

/// Per-device allocation-failure counters observed by the device memory
/// manager failure callback.
#[derive(Debug, Clone)]
pub(crate) struct DeviceMemoryFailureTracker {
    inner: Arc<DeviceMemoryFailureTrackerInner>,
}

#[derive(Debug)]
struct DeviceMemoryFailureTrackerInner {
    device_ordinal: usize,
    /// Zero until the tracker is bound to a manager installation; bound once
    /// and never rebound to a different generation.
    installation_generation: AtomicU64,
    /// Whether the live callback updates this tracker. Mutually exclusive
    /// with `callback_install_unavailable`; both false before installation.
    callback_installed: AtomicBool,
    callback_install_unavailable: AtomicBool,
    allocation_failure_count: AtomicU64,
    total_failed_allocation_bytes: AtomicU64,
    largest_failed_allocation_bytes: AtomicU64,
    pool_usage_observation_count: AtomicU64,
    last_failed_allocation_bytes: AtomicU64,
    last_pool_reserved_current_bytes: AtomicU64,
    last_pool_used_current_bytes: AtomicU64,
}

/// Point-in-time device memory allocation-failure report.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DeviceMemoryFailureReport {
    /// CUDA ordinal monitored by the installed allocation-failure callback.
    pub device_ordinal: usize,
    /// Whether the native memory manager updates this report's tracker.
    pub callback_installed: bool,
    /// Whether callback installation was unavailable; excludes `callback_installed`.
    pub callback_install_unavailable: bool,
    /// Allocation failures observed over the memory-manager installation lifetime.
    pub allocation_failure_count: u64,
    /// Sum of failed allocation request sizes in bytes.
    pub total_failed_allocation_bytes: u64,
    /// Largest failed allocation request size in bytes; zero before any failure.
    pub largest_failed_allocation_bytes: u64,
    /// Always zero because this memory-manager callback declines allocator retries.
    pub trim_retry_count: u64,
    /// Failure callbacks carrying a CUDA async-pool usage sample.
    pub pool_usage_observation_count: u64,
    /// Most recent failed request size in bytes; zero before any failure.
    pub last_failed_allocation_bytes: u64,
    /// Pool-reserved bytes from the last callback carrying a pool sample.
    /// Zero before the first sample; consult `pool_usage_observation_count`.
    pub last_pool_reserved_current_bytes: u64,
    /// Pool-used bytes from the last callback carrying a pool sample.
    /// Zero before the first sample; consult `pool_usage_observation_count`.
    pub last_pool_used_current_bytes: u64,
}

impl DeviceMemoryFailureTracker {
    /// Creates a zeroed tracker for one CUDA device ordinal.
    pub(crate) fn new(device_ordinal: usize) -> Self {
        Self {
            inner: Arc::new(DeviceMemoryFailureTrackerInner {
                device_ordinal,
                installation_generation: AtomicU64::new(0),
                callback_installed: AtomicBool::new(false),
                callback_install_unavailable: AtomicBool::new(false),
                allocation_failure_count: AtomicU64::new(0),
                total_failed_allocation_bytes: AtomicU64::new(0),
                largest_failed_allocation_bytes: AtomicU64::new(0),
                pool_usage_observation_count: AtomicU64::new(0),
                last_failed_allocation_bytes: AtomicU64::new(0),
                last_pool_reserved_current_bytes: AtomicU64::new(0),
                last_pool_used_current_bytes: AtomicU64::new(0),
            }),
        }
    }

    /// Recovers the tracker referenced by an already-installed callback, or
    /// creates an unbound candidate only while the process manager is still
    /// uninstalled. This prevents a fresh counter from being substituted for
    /// the callback's original `(device, installation_generation)` owner.
    pub(crate) fn for_installation_state(
        device_ordinal: usize,
        state: cudf::DeviceMemoryManagerInstallationState,
    ) -> Result<Self> {
        Self::for_installation_state_in_registry(device_ordinal, state, device_memory_failure_trackers())
    }

    fn for_installation_state_in_registry(
        device_ordinal: usize,
        state: cudf::DeviceMemoryManagerInstallationState,
        trackers: &DeviceMemoryFailureTrackerRegistry,
    ) -> Result<Self> {
        let mut trackers = trackers.lock().map_err(|_error| tracker_registry_poisoned(device_ordinal))?;
        match state.status {
            cudf::DeviceMemoryManagerInstallationStatus::Uninstalled => {
                if trackers.installed.keys().any(|(installed_device, _)| *installed_device == device_ordinal) {
                    return Err(QueryError::new(
                        QueryErrorCode::Internal,
                        TRACKER_LOOKUP_OPERATION,
                        "tracker remains bound while the device manager reports uninstalled",
                    )
                    .with_fact("device_ordinal", device_ordinal)
                    .with_fact("reason", "tracker_generation_without_installation"));
                }
                let tracker = trackers.pending.entry(device_ordinal).or_insert_with(|| Self::new(device_ordinal));
                if tracker.installation_generation().is_some() {
                    return Err(QueryError::new(
                        QueryErrorCode::Internal,
                        TRACKER_LOOKUP_OPERATION,
                        "pending tracker is bound while the device manager reports uninstalled",
                    )
                    .with_fact("device_ordinal", device_ordinal)
                    .with_fact("reason", "pending_tracker_generation_without_installation"));
                }
                Ok(tracker.clone())
            },
            cudf::DeviceMemoryManagerInstallationStatus::Owned
            | cudf::DeviceMemoryManagerInstallationStatus::Incomplete => {
                let generation = state.installation_generation.ok_or_else(|| {
                    QueryError::new(
                        QueryErrorCode::Internal,
                        TRACKER_LOOKUP_OPERATION,
                        "owned device-manager callback has no installation generation",
                    )
                    .with_fact("device_ordinal", device_ordinal)
                    .with_fact("reason", "installed_callback_generation_missing")
                })?;
                if let Some(tracker) = trackers.installed.get(&(device_ordinal, generation)).cloned() {
                    tracker.require_installation_generation(generation)?;
                    return Ok(tracker);
                }
                if let Some((_, existing_generation)) =
                    trackers.installed.keys().find(|(installed_device, _)| *installed_device == device_ordinal).copied()
                {
                    return Err(QueryError::new(
                        QueryErrorCode::CudfPersistentFailure,
                        TRACKER_LOOKUP_OPERATION,
                        "installed device-manager callback generation does not match the recoverable tracker",
                    )
                    .with_fact("device_ordinal", device_ordinal)
                    .with_fact("expected_installation_generation", existing_generation)
                    .with_fact("observed_installation_generation", generation)
                    .with_fact("reason", "callback_tracker_generation_mismatch"));
                }
                let tracker = trackers.pending.remove(&device_ordinal).ok_or_else(|| {
                    QueryError::new(
                        QueryErrorCode::Internal,
                        TRACKER_LOOKUP_OPERATION,
                        "installed device-manager callback has no recoverable original tracker",
                    )
                    .with_fact("device_ordinal", device_ordinal)
                    .with_fact("installation_generation", generation)
                    .with_fact("reason", "installed_callback_tracker_missing")
                })?;
                if state.status == cudf::DeviceMemoryManagerInstallationStatus::Incomplete
                    && tracker.installation_generation().is_none()
                {
                    // The native/Rust cuDF installation record commits callback
                    // identity before attempting the current-resource switch.
                    // A retryable switch failure therefore binds the original
                    // pre-install candidate here before a later owner retries.
                    tracker.bind_installation_generation(generation)?;
                } else {
                    tracker.require_installation_generation(generation)?;
                }
                trackers.installed.insert((device_ordinal, generation), tracker.clone());
                Ok(tracker)
            },
            _ => Err(QueryError::new(
                QueryErrorCode::CudfPersistentFailure,
                TRACKER_LOOKUP_OPERATION,
                "device-manager callback installation is permanently unusable",
            )
            .with_fact("device_ordinal", device_ordinal)
            .with_fact("installation_status", format!("{:?}", state.status))
            .with_fact("installation_retry", format!("{:?}", state.retry))
            .with_fact("installation_cause", format!("{:?}", state.cause))
            .with_fact("reason", "installed_callback_unrecoverable")),
        }
    }

    #[cfg(test)]
    fn for_device_in_registry(device_ordinal: usize, trackers: &DeviceMemoryFailureTrackerRegistry) -> Result<Self> {
        let mut trackers = trackers.lock().map_err(|_error| tracker_registry_poisoned(device_ordinal))?;
        if let Some(tracker) = trackers
            .installed
            .iter()
            .find_map(|((installed_device, _), tracker)| (*installed_device == device_ordinal).then(|| tracker.clone()))
        {
            return Ok(tracker);
        }
        Ok(trackers.pending.entry(device_ordinal).or_insert_with(|| Self::new(device_ordinal)).clone())
    }

    /// Moves this tracker from the pending to the installed registry under
    /// `generation`, after a successful manager installation. Idempotent for
    /// the same tracker; a different tracker already bound to the generation,
    /// or a pending candidate that is not this tracker, is refused so the
    /// callback and the reported counters cannot diverge.
    pub(crate) fn commit_installation_generation(&self, generation: u64) -> Result<()> {
        self.bind_installation_generation(generation)?;
        let device_ordinal = self.device_ordinal();
        let mut trackers =
            device_memory_failure_trackers().lock().map_err(|_error| tracker_registry_poisoned(device_ordinal))?;
        if let Some(existing) = trackers.installed.get(&(device_ordinal, generation)) {
            if Arc::ptr_eq(&existing.inner, &self.inner) {
                return Ok(());
            }
            return Err(QueryError::new(
                QueryErrorCode::CudfPersistentFailure,
                TRACKER_LOOKUP_OPERATION,
                "installation generation is already bound to a different callback tracker",
            )
            .with_fact("device_ordinal", device_ordinal)
            .with_fact("installation_generation", generation)
            .with_fact("reason", "callback_tracker_identity_mismatch"));
        }
        let pending = trackers.pending.remove(&device_ordinal).ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::CudfPersistentFailure,
                TRACKER_LOOKUP_OPERATION,
                "installed callback tracker was not the registered pre-install candidate",
            )
            .with_fact("device_ordinal", device_ordinal)
            .with_fact("installation_generation", generation)
            .with_fact("reason", "pending_callback_tracker_missing")
        })?;
        if !Arc::ptr_eq(&pending.inner, &self.inner) {
            trackers.pending.insert(device_ordinal, pending);
            return Err(QueryError::new(
                QueryErrorCode::CudfPersistentFailure,
                TRACKER_LOOKUP_OPERATION,
                "pending callback tracker does not match the installed callback tracker",
            )
            .with_fact("device_ordinal", device_ordinal)
            .with_fact("installation_generation", generation)
            .with_fact("reason", "callback_tracker_identity_mismatch"));
        }
        trackers.installed.insert((device_ordinal, generation), self.clone());
        Ok(())
    }

    pub(crate) fn bind_installation_generation(&self, generation: u64) -> Result<()> {
        if generation == 0 {
            return Err(QueryError::new(
                QueryErrorCode::Internal,
                TRACKER_LOOKUP_OPERATION,
                "callback installation generation must be nonzero",
            )
            .with_fact("device_ordinal", self.device_ordinal())
            .with_fact("reason", "zero_installation_generation"));
        }
        match self.inner.installation_generation.compare_exchange(0, generation, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => Ok(()),
            Err(existing) if existing == generation => Ok(()),
            Err(existing) => Err(QueryError::new(
                QueryErrorCode::CudfPersistentFailure,
                TRACKER_LOOKUP_OPERATION,
                "callback tracker is bound to a different installation generation",
            )
            .with_fact("device_ordinal", self.device_ordinal())
            .with_fact("expected_installation_generation", existing)
            .with_fact("observed_installation_generation", generation)
            .with_fact("reason", "callback_tracker_generation_mismatch")),
        }
    }

    fn require_installation_generation(&self, generation: u64) -> Result<()> {
        match self.installation_generation() {
            Some(existing) if existing == generation => Ok(()),
            Some(existing) => Err(QueryError::new(
                QueryErrorCode::CudfPersistentFailure,
                TRACKER_LOOKUP_OPERATION,
                "installed callback generation does not match its original tracker",
            )
            .with_fact("device_ordinal", self.device_ordinal())
            .with_fact("expected_installation_generation", existing)
            .with_fact("observed_installation_generation", generation)
            .with_fact("reason", "callback_tracker_generation_mismatch")),
            None => Err(QueryError::new(
                QueryErrorCode::CudfPersistentFailure,
                TRACKER_LOOKUP_OPERATION,
                "installed callback's tracker is not bound to an installation generation",
            )
            .with_fact("device_ordinal", self.device_ordinal())
            .with_fact("observed_installation_generation", generation)
            .with_fact("reason", "callback_tracker_generation_unbound")),
        }
    }

    fn installation_generation(&self) -> Option<u64> {
        match self.inner.installation_generation.load(Ordering::Acquire) {
            0 => None,
            generation => Some(generation),
        }
    }

    /// CUDA device ordinal this tracker describes.
    pub fn device_ordinal(&self) -> usize {
        self.inner.device_ordinal
    }

    /// Returns a re-entrancy-safe callback for cudf/RMM.
    ///
    /// The callback records counters without deciding allocation retry or
    /// reclamation. It must remain allocation-free and lock-free because RMM
    /// invokes it inside allocation failure handling.
    pub(crate) fn failure_callback(&self) -> Arc<dyn Fn(cudf::DeviceMemoryFailureInfo) + Send + Sync + 'static> {
        let tracker = self.clone();
        Arc::new(move |info| {
            if info.source != DeviceMemoryFailureSource::Physical {
                return;
            }
            tracker.record_allocation_failure(info.requested_bytes, info.pool_usage);
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

    fn record_allocation_failure(&self, bytes: u64, pool_usage: Option<DeviceMemoryPoolUsage>) {
        self.inner.allocation_failure_count.fetch_add(1, Ordering::Relaxed);
        self.inner.total_failed_allocation_bytes.fetch_add(bytes, Ordering::Relaxed);
        self.inner.last_failed_allocation_bytes.store(bytes, Ordering::Relaxed);
        update_max(&self.inner.largest_failed_allocation_bytes, bytes);
        if let Some(pool_usage) = pool_usage {
            self.inner.pool_usage_observation_count.fetch_add(1, Ordering::Relaxed);
            self.inner.last_pool_reserved_current_bytes.store(pool_usage.reserved_current_bytes, Ordering::Relaxed);
            self.inner.last_pool_used_current_bytes.store(pool_usage.used_current_bytes, Ordering::Relaxed);
        }
    }

    pub(crate) fn report(&self) -> DeviceMemoryFailureReport {
        DeviceMemoryFailureReport {
            device_ordinal: self.inner.device_ordinal,
            callback_installed: self.inner.callback_installed.load(Ordering::Acquire),
            callback_install_unavailable: self.inner.callback_install_unavailable.load(Ordering::Acquire),
            allocation_failure_count: self.inner.allocation_failure_count.load(Ordering::Acquire),
            total_failed_allocation_bytes: self.inner.total_failed_allocation_bytes.load(Ordering::Acquire),
            largest_failed_allocation_bytes: self.inner.largest_failed_allocation_bytes.load(Ordering::Acquire),
            trim_retry_count: 0,
            pool_usage_observation_count: self.inner.pool_usage_observation_count.load(Ordering::Acquire),
            last_failed_allocation_bytes: self.inner.last_failed_allocation_bytes.load(Ordering::Acquire),
            last_pool_reserved_current_bytes: self.inner.last_pool_reserved_current_bytes.load(Ordering::Acquire),
            last_pool_used_current_bytes: self.inner.last_pool_used_current_bytes.load(Ordering::Acquire),
        }
    }
}

fn tracker_registry_poisoned(device_ordinal: usize) -> QueryError {
    QueryError::new(
        QueryErrorCode::Internal,
        TRACKER_LOOKUP_OPERATION,
        "device memory failure-tracker registry mutex was poisoned",
    )
    .with_fact("device_ordinal", device_ordinal)
    .with_fact("reason", "tracker_registry_poisoned")
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
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use super::*;

    #[test]
    fn failure_callback_records_counters_and_declines_retry() {
        let tracker = DeviceMemoryFailureTracker::new(3);

        tracker.record_allocation_failure(10, None);
        tracker.record_allocation_failure(
            40,
            Some(DeviceMemoryPoolUsage {
                reserved_current_bytes: 400,
                reserved_high_bytes: 800,
                used_current_bytes: 100,
                used_high_bytes: 700,
            }),
        );
        tracker.record_allocation_failure(20, None);

        let report = tracker.report();
        assert_eq!(report.device_ordinal, 3);
        assert!(!report.callback_installed);
        assert_eq!(report.allocation_failure_count, 3);
        assert_eq!(report.total_failed_allocation_bytes, 70);
        assert_eq!(report.largest_failed_allocation_bytes, 40);
        assert_eq!(report.trim_retry_count, 0);
        assert_eq!(report.pool_usage_observation_count, 1);
        assert_eq!(report.last_failed_allocation_bytes, 20);
        assert_eq!(report.last_pool_reserved_current_bytes, 400);
        assert_eq!(report.last_pool_used_current_bytes, 100);
    }

    #[test]
    fn failure_callback_declines_reservation_limit_without_recording_device_failure() {
        let tracker = DeviceMemoryFailureTracker::new(3);
        let callback = tracker.failure_callback();

        callback(cudf::DeviceMemoryFailureInfo {
            device: cudf::DeviceId::DEFAULT,
            source: DeviceMemoryFailureSource::ReservationLimit,
            allocation_domain: None,
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

        let report = tracker.report();
        assert_eq!(report.allocation_failure_count, 0);
        assert_eq!(report.trim_retry_count, 0);
        assert_eq!(report.pool_usage_observation_count, 0);
    }

    #[test]
    fn failure_callback_observes_physical_failure_with_idle_pool() {
        let tracker = DeviceMemoryFailureTracker::new(3);
        let callback = tracker.failure_callback();

        callback(cudf::DeviceMemoryFailureInfo {
            device: cudf::DeviceId::DEFAULT,
            source: DeviceMemoryFailureSource::Physical,
            allocation_domain: None,
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

        let report = tracker.report();
        assert_eq!(report.allocation_failure_count, 1);
        assert_eq!(report.trim_retry_count, 0);
        assert_eq!(report.pool_usage_observation_count, 1);
        assert_eq!(report.last_failed_allocation_bytes, 10);
    }

    #[test]
    fn compatible_generation_lookup_recovers_original_callback_tracker() {
        let trackers = Mutex::new(DeviceMemoryFailureTrackerRegistryState::default());
        let first_generation = DeviceMemoryFailureTracker::for_device_in_registry(3, &trackers).unwrap();
        let original_identity = Arc::downgrade(&first_generation.inner);
        let original_callback = first_generation.failure_callback();
        drop(first_generation);

        let second_generation = DeviceMemoryFailureTracker::for_device_in_registry(3, &trackers).unwrap();
        let original_inner = original_identity.upgrade().expect("registry must retain the original tracker");
        assert!(Arc::ptr_eq(&original_inner, &second_generation.inner));

        original_callback(cudf::DeviceMemoryFailureInfo {
            device: cudf::DeviceId::DEFAULT,
            source: DeviceMemoryFailureSource::Physical,
            allocation_domain: None,
            requested_bytes: 64,
            callback_attempt: 0,
            trim_retry_count: 0,
            pool_usage: None,
        });

        let report = second_generation.report();
        assert_eq!(report.allocation_failure_count, 1);
        assert_eq!(report.total_failed_allocation_bytes, 64);
        assert_eq!(report.last_failed_allocation_bytes, 64);
    }

    #[test]
    fn retryable_incomplete_install_binds_the_original_preinstall_tracker() {
        let trackers = Mutex::new(DeviceMemoryFailureTrackerRegistryState::default());
        let original = DeviceMemoryFailureTracker::for_device_in_registry(3, &trackers).unwrap();
        assert_eq!(original.installation_generation(), None);
        let state = cudf::DeviceMemoryManagerInstallationState {
            device_ordinal: 3,
            status: cudf::DeviceMemoryManagerInstallationStatus::Incomplete,
            retry: cudf::DeviceMemoryManagerInstallationRetry::Retryable,
            cause: cudf::DeviceMemoryManagerInstallationCause::CurrentResourceInstallFailed,
            effective_options: Some(cudf::DeviceMemoryManagerOptions::default()),
            identity: Some(cudf::DeviceMemoryManagerInstallationIdentity::new([1; 16], 1, [2; 16])),
            installation_generation: Some(41),
            failure_callback_installed: true,
            current_resource_installed: false,
            failure_callback_panicked: false,
        };

        let recovered = DeviceMemoryFailureTracker::for_installation_state_in_registry(3, state, &trackers).unwrap();

        assert!(Arc::ptr_eq(&original.inner, &recovered.inner));
        assert_eq!(recovered.installation_generation(), Some(41));
        let registry = trackers.lock().unwrap();
        assert!(!registry.pending.contains_key(&3));
        assert!(registry.installed.get(&(3, 41)).is_some_and(|tracker| Arc::ptr_eq(&tracker.inner, &original.inner)));
    }

    #[test]
    fn tracker_registry_poison_fails_closed() {
        let trackers = Mutex::new(DeviceMemoryFailureTrackerRegistryState::default());
        let poison = catch_unwind(AssertUnwindSafe(|| {
            let _guard = trackers.lock().unwrap();
            panic!("poison test tracker registry");
        }));
        assert!(poison.is_err());

        let error = DeviceMemoryFailureTracker::for_device_in_registry(7, &trackers).unwrap_err();

        assert_eq!(error.code(), QueryErrorCode::Internal);
        assert_eq!(error.operation(), TRACKER_LOOKUP_OPERATION);
        assert_eq!(error.fact_value("device_ordinal"), Some("7"));
        assert_eq!(error.fact_value("reason"), Some("tracker_registry_poisoned"));
    }
}

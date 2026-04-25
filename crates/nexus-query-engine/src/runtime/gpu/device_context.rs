//! Per-device service runtime context.

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use super::{
    DeviceFacts, DeviceMemoryFailureReport, DeviceMemoryFailureTracker, GrantRequest, MemoryGovernor, MemoryGrant,
};
use crate::{
    error::{QueryError, QueryErrorCode, Result},
    runtime::ServiceStreamPool,
};

/// Service runtime state for one CUDA device ordinal.
#[derive(Debug, Clone)]
pub struct DeviceContext {
    inner: Arc<DeviceContextInner>,
}

#[derive(Debug)]
pub(crate) struct DeviceActiveQuerySlot {
    device_ordinal: usize,
    active_queries: Arc<AtomicUsize>,
}

impl DeviceActiveQuerySlot {
    /// CUDA device ordinal whose per-device admission slot is held.
    pub(crate) fn device_ordinal(&self) -> usize {
        self.device_ordinal
    }

    #[cfg(test)]
    pub(crate) fn new_for_test(device_ordinal: usize, active_queries: Arc<AtomicUsize>) -> Self {
        Self { device_ordinal, active_queries }
    }
}

impl Drop for DeviceActiveQuerySlot {
    fn drop(&mut self) {
        self.active_queries.fetch_sub(1, Ordering::AcqRel);
    }
}

#[derive(Debug)]
struct DeviceContextInner {
    device: cudf::DeviceId,
    stream_pool: ServiceStreamPool,
    mr_manager: cudf::DeviceMemoryManager,
    memory_failure_tracker: DeviceMemoryFailureTracker,
    governor: MemoryGovernor,
    facts: DeviceFacts,
    active_queries: Arc<AtomicUsize>,
}

impl DeviceContext {
    /// Creates a per-device service context.
    pub fn new(
        device: cudf::DeviceId,
        stream_pool: ServiceStreamPool,
        mr_manager: cudf::DeviceMemoryManager,
        memory_failure_tracker: DeviceMemoryFailureTracker,
        governor: MemoryGovernor,
        facts: DeviceFacts,
        active_queries: Arc<AtomicUsize>,
    ) -> Result<Self> {
        if device.ordinal() != facts.device {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.gpu.device_context",
                "device context facts must describe the context device",
            )
            .with_fact("device_ordinal", device.ordinal())
            .with_fact("facts_device_ordinal", facts.device));
        }
        if stream_pool.device() != device {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.gpu.device_context",
                "device context stream pool must match the context device",
            )
            .with_fact("device_ordinal", device.ordinal())
            .with_fact("pool_device_ordinal", stream_pool.device().ordinal()));
        }
        if mr_manager.device() != device {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.gpu.device_context",
                "device context memory manager must match the context device",
            )
            .with_fact("device_ordinal", device.ordinal())
            .with_fact("memory_manager_device_ordinal", mr_manager.device().ordinal()));
        }
        if memory_failure_tracker.device_ordinal() != device.ordinal() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.gpu.device_context",
                "device context memory failure tracker must match the context device",
            )
            .with_fact("device_ordinal", device.ordinal())
            .with_fact("memory_failure_tracker_device_ordinal", memory_failure_tracker.device_ordinal()));
        }
        if governor.device_ordinal() != device.ordinal() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.gpu.device_context",
                "device context memory governor must match the context device",
            )
            .with_fact("device_ordinal", device.ordinal())
            .with_fact("memory_governor_device_ordinal", governor.device_ordinal()));
        }
        Ok(Self {
            inner: Arc::new(DeviceContextInner {
                device,
                stream_pool,
                mr_manager,
                memory_failure_tracker,
                governor,
                facts,
                active_queries,
            }),
        })
    }

    /// CUDA device served by this context.
    pub fn device(&self) -> cudf::DeviceId {
        self.inner.device
    }

    /// Service stream pool for this device.
    pub fn stream_pool(&self) -> &ServiceStreamPool {
        &self.inner.stream_pool
    }

    /// Handle to cudf-rs's process-global per-device memory manager.
    pub fn memory_manager(&self) -> &cudf::DeviceMemoryManager {
        &self.inner.mr_manager
    }

    /// Signal-only allocation-failure counters from the device memory manager callback.
    pub fn memory_failure_report(&self) -> DeviceMemoryFailureReport {
        self.inner.memory_failure_tracker.report()
    }

    /// Per-device soft memory grant ledger.
    pub fn memory_governor(&self) -> &MemoryGovernor {
        &self.inner.governor
    }

    /// Stable inventory facts captured when the registry was installed.
    pub fn facts(&self) -> &DeviceFacts {
        &self.inner.facts
    }

    /// Device capacity used for service-level granted accounting.
    pub fn capacity_bytes(&self) -> u64 {
        self.inner.governor.capacity_bytes()
    }

    /// Number of currently admitted queries on this CUDA device ordinal across
    /// every `QueryService` in this process.
    pub fn active_queries(&self) -> usize {
        self.inner.active_queries.load(Ordering::Acquire)
    }

    /// Attempts to acquire one per-device active-query slot.
    pub(crate) fn try_acquire_query_slot(
        &self,
        max_queries_per_device: Option<usize>,
    ) -> Option<DeviceActiveQuerySlot> {
        let mut current = self.inner.active_queries.load(Ordering::Acquire);
        loop {
            if max_queries_per_device.is_some_and(|cap| current >= cap) {
                return None;
            }
            match self.inner.active_queries.compare_exchange_weak(
                current,
                current.saturating_add(1),
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    return Some(DeviceActiveQuerySlot {
                        device_ordinal: self.inner.device.ordinal(),
                        active_queries: Arc::clone(&self.inner.active_queries),
                    });
                },
                Err(actual) => current = actual,
            }
        }
    }

    /// Bytes grantable to admitted queries, including evictable cache bytes
    /// that can be synchronously released by the device memory governor.
    pub fn available_bytes(&self) -> Result<u64> {
        self.inner.governor.available_bytes()
    }

    /// Attempts to grant a query budget against this device's governor.
    ///
    /// This only updates the memory-governor ledger. Per-device active-query
    /// slots are tracked separately by `try_acquire_query_slot`.
    pub fn try_grant_query(&self, request: GrantRequest) -> Result<Option<MemoryGrant>> {
        self.inner.governor.try_request(request)
    }
}

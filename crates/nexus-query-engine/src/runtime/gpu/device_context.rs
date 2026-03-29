//! Per-device service runtime context.

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use super::{DeviceFacts, GrantRequest, MemoryGovernor, MemoryGrant};
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
struct DeviceContextInner {
    device: cudf::DeviceId,
    stream_pool: ServiceStreamPool,
    mr_manager: cudf::DeviceMemoryManager,
    governor: MemoryGovernor,
    facts: DeviceFacts,
    active_queries: AtomicUsize,
}

impl DeviceContext {
    /// Creates a per-device service context.
    pub fn new(
        device: cudf::DeviceId,
        stream_pool: ServiceStreamPool,
        mr_manager: cudf::DeviceMemoryManager,
        governor: MemoryGovernor,
        facts: DeviceFacts,
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
                governor,
                facts,
                active_queries: AtomicUsize::new(0),
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

    /// Number of currently admitted queries on this device.
    pub fn active_queries(&self) -> usize {
        self.inner.active_queries.load(Ordering::Acquire)
    }

    /// Capacity bytes not currently granted to admitted queries.
    pub fn available_bytes(&self) -> Result<u64> {
        self.inner.governor.available_bytes()
    }

    /// Attempts to grant a query budget against this device's governor.
    pub fn try_grant_query(&self, request: GrantRequest) -> Result<Option<MemoryGrant>> {
        let Some(grant) = self.inner.governor.try_request(request)? else {
            return Ok(None);
        };
        let context = self.clone();
        grant.notify_on_release(Arc::new(move || {
            context.release_query_grant();
        }))?;
        self.inner.active_queries.fetch_add(1, Ordering::AcqRel);
        Ok(Some(grant))
    }

    fn release_query_grant(&self) {
        self.inner.active_queries.fetch_sub(1, Ordering::AcqRel);
    }
}

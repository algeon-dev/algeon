//! Process-level GPU device registry for service execution.

use std::sync::{Arc, OnceLock};

use super::{
    DeviceContext, DeviceMemoryFailureReport, DeviceMemoryFailureTracker, GpuInventory, MemoryGovernor,
    PlacementPolicy, PlacementRequest, device_budget_bytes_from_facts,
};
use crate::{
    error::{QueryError, QueryErrorCode, Result},
    runtime::ServiceStreamPool,
};

/// Configuration used when installing a device registry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceRegistryConfig {
    /// CUDA ordinals enabled for the service. Empty means all visible devices.
    pub device_ordinals: Vec<usize>,
    /// Stream-pool size installed for each enabled device.
    pub stream_pool_size: usize,
    /// Fixed capacity headroom subtracted from each device's total memory.
    pub capacity_headroom_bytes: u64,
    /// Fallback capacity used if the headroom calculation underflows.
    pub fallback_capacity_bytes: u64,
    /// Optional per-device governor capacity cap.
    ///
    /// The effective capacity is `min(inventory_capacity, device_capacity_bytes)`
    /// so a single config value cannot over-grant smaller GPUs.
    pub device_capacity_bytes: Option<u64>,
}

impl DeviceRegistryConfig {
    /// Creates a registry config for explicit device ordinals.
    pub fn new(device_ordinals: Vec<usize>, stream_pool_size: usize) -> Self {
        Self {
            device_ordinals,
            stream_pool_size,
            capacity_headroom_bytes: 500 * 1024 * 1024,
            fallback_capacity_bytes: 256 * 1024 * 1024,
            device_capacity_bytes: None,
        }
    }

    /// Sets the per-device capacity calculation parameters.
    pub fn with_capacity_policy(mut self, headroom_bytes: u64, fallback_bytes: u64) -> Self {
        self.capacity_headroom_bytes = headroom_bytes;
        self.fallback_capacity_bytes = fallback_bytes;
        self
    }

    /// Sets a maximum governor capacity for every installed device.
    pub fn with_device_capacity_bytes(mut self, device_capacity_bytes: u64) -> Self {
        self.device_capacity_bytes = Some(device_capacity_bytes);
        self
    }
}

/// Registry of service contexts by CUDA device ordinal.
#[derive(Debug, Clone)]
pub struct DeviceRegistry {
    contexts: Arc<[Option<DeviceContext>]>,
    device_ordinals: Arc<[usize]>,
}

impl DeviceRegistry {
    /// Installs service contexts for configured devices.
    pub fn install(inventory: &dyn GpuInventory, config: &DeviceRegistryConfig) -> Result<Self> {
        validate_registry_config(config)?;
        let ordinals = configured_ordinals(inventory, config)?;
        let max_ordinal = ordinals.iter().copied().max().unwrap_or(0);
        let mut contexts = vec![None; max_ordinal.saturating_add(1)];
        for ordinal in &ordinals {
            let device = cudf::DeviceId::new(*ordinal).map_err(|error| {
                QueryError::from_cudf(
                    "nexus_query_engine.gpu.device_registry.device",
                    "failed to resolve CUDA device while installing registry",
                    error,
                )
                .with_fact("device_ordinal", *ordinal)
            })?;
            let facts = inventory.facts(*ordinal)?;
            let inventory_capacity_bytes =
                device_budget_bytes_from_facts(&facts, config.capacity_headroom_bytes, config.fallback_capacity_bytes);
            let capacity_bytes =
                config.device_capacity_bytes.map_or(inventory_capacity_bytes, |cap| cap.min(inventory_capacity_bytes));
            let memory_failure_tracker = DeviceMemoryFailureTracker::for_device(*ordinal);
            let mr_manager = install_device_memory_manager(device, &memory_failure_tracker)
                .map_err(|error| error.with_fact("device_ordinal", *ordinal))?;
            let stream_pool = ServiceStreamPool::new(device, config.stream_pool_size)?;
            let governor = MemoryGovernor::new(*ordinal, capacity_bytes)?;
            contexts[*ordinal] =
                Some(DeviceContext::new(device, stream_pool, mr_manager, memory_failure_tracker, governor, facts)?);
        }
        Ok(Self { contexts: contexts.into(), device_ordinals: ordinals.into() })
    }

    /// Returns the process-global registry installed with default inventory.
    pub fn shared() -> Result<Arc<Self>> {
        static SHARED: OnceLock<Arc<DeviceRegistry>> = OnceLock::new();
        if let Some(registry) = SHARED.get() {
            return Ok(Arc::clone(registry));
        }
        let config = DeviceRegistryConfig::new(Vec::new(), crate::runtime::DEFAULT_SERVICE_STREAM_POOL_SIZE);
        let registry = Arc::new(Self::install(&super::CudfInventory, &config)?);
        let _ = SHARED.set(Arc::clone(&registry));
        Ok(SHARED.get().map(Arc::clone).unwrap_or(registry))
    }

    /// Looks up a context by CUDA device ordinal.
    pub fn context(&self, device: cudf::DeviceId) -> Result<&DeviceContext> {
        self.context_by_ordinal(device.ordinal())
    }

    /// Looks up a context by CUDA device ordinal.
    pub fn context_by_ordinal(&self, ordinal: usize) -> Result<&DeviceContext> {
        self.contexts.get(ordinal).and_then(Option::as_ref).ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.gpu.device_registry.context",
                "CUDA device is not installed in the device registry",
            )
            .with_fact("device_ordinal", ordinal)
        })
    }

    /// Iterates installed contexts in ordinal order.
    pub fn contexts(&self) -> impl Iterator<Item = &DeviceContext> {
        self.contexts.iter().filter_map(Option::as_ref)
    }

    /// Allocation-failure reports for all installed devices.
    pub fn memory_failure_reports(&self) -> Vec<DeviceMemoryFailureReport> {
        self.contexts().map(DeviceContext::memory_failure_report).collect()
    }

    /// Enabled CUDA ordinals in placement order.
    pub fn device_ordinals(&self) -> &[usize] {
        &self.device_ordinals
    }

    /// Number of installed device contexts.
    pub fn len(&self) -> usize {
        self.device_ordinals.len()
    }

    /// Returns whether the registry has no installed devices.
    pub fn is_empty(&self) -> bool {
        self.device_ordinals.is_empty()
    }

    /// Returns whether any candidate device has enough configured capacity for
    /// this query budget, ignoring current active grants.
    pub fn can_fit_query_budget(&self, policy: &dyn PlacementPolicy, request: &PlacementRequest) -> Result<bool> {
        for device in policy.candidates(self, request)? {
            if self.context(device)?.capacity_bytes() >= request.min_device_budget_bytes {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

fn install_device_memory_manager(
    device: cudf::DeviceId,
    memory_failure_tracker: &DeviceMemoryFailureTracker,
) -> Result<cudf::DeviceMemoryManager> {
    let callback_state = cudf::DeviceMemoryManager::failure_callback_state_for_device(device).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.gpu.device_registry.memory_manager",
            "failed to inspect device memory manager failure callback state",
            error,
        )
    })?;
    if callback_state.exists && !callback_state.failure_callback_installed {
        memory_failure_tracker.mark_callback_install_unavailable();
        return cudf::DeviceMemoryManager::install_for_device(device).map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.gpu.device_registry.memory_manager",
                "failed to install existing device memory manager after callback install was unavailable",
                error,
            )
        });
    }
    if callback_state.exists && callback_state.failure_callback_installed {
        let report = memory_failure_tracker.report();
        if report.callback_installed {
            return cudf::DeviceMemoryManager::install_for_device(device).map_err(|error| {
                QueryError::from_cudf(
                    "nexus_query_engine.gpu.device_registry.memory_manager",
                    "failed to resolve existing callback-installed device memory manager",
                    error,
                )
            });
        }
        memory_failure_tracker.mark_callback_install_unavailable();
        return cudf::DeviceMemoryManager::install_for_device(device).map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.gpu.device_registry.memory_manager",
                "failed to resolve existing foreign callback-installed device memory manager",
                error,
            )
        });
    }
    match cudf::DeviceMemoryManager::install_for_device_with_failure_callback(
        device,
        memory_failure_tracker.failure_callback(),
    ) {
        Ok(manager) => {
            memory_failure_tracker.mark_callback_installed();
            Ok(manager)
        },
        Err(error) => {
            if existing_plain_memory_manager(device)? {
                memory_failure_tracker.mark_callback_install_unavailable();
                cudf::DeviceMemoryManager::install_for_device(device).map_err(|fallback_error| {
                    QueryError::from_cudf(
                        "nexus_query_engine.gpu.device_registry.memory_manager",
                        "failed to install existing device memory manager after callback install was unavailable",
                        fallback_error,
                    )
                })
            } else {
                Err(QueryError::from_cudf(
                    "nexus_query_engine.gpu.device_registry.memory_manager",
                    "failed to install device memory manager while installing registry",
                    error,
                ))
            }
        },
    }
}

fn existing_plain_memory_manager(device: cudf::DeviceId) -> Result<bool> {
    let state = cudf::DeviceMemoryManager::failure_callback_state_for_device(device).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.gpu.device_registry.memory_manager",
            "failed to inspect device memory manager failure callback state after install error",
            error,
        )
    })?;
    Ok(state.exists && !state.failure_callback_installed)
}

fn validate_registry_config(config: &DeviceRegistryConfig) -> Result<()> {
    if config.stream_pool_size == 0 {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.gpu.device_registry.config",
            "device registry stream pool size must be greater than zero",
        )
        .with_fact("field", "stream_pool_size")
        .with_fact("reason", "zero"));
    }
    if config.fallback_capacity_bytes == 0 {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.gpu.device_registry.config",
            "device registry fallback capacity must be greater than zero",
        )
        .with_fact("field", "fallback_capacity_bytes")
        .with_fact("reason", "zero"));
    }
    if let Some(device_capacity_bytes) = config.device_capacity_bytes
        && device_capacity_bytes == 0
    {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.gpu.device_registry.config",
            "device registry capacity cap must be greater than zero",
        )
        .with_fact("field", "device_capacity_bytes")
        .with_fact("reason", "zero"));
    }
    Ok(())
}

fn configured_ordinals(inventory: &dyn GpuInventory, config: &DeviceRegistryConfig) -> Result<Vec<usize>> {
    let device_count = inventory.device_count()?;
    let mut ordinals = if config.device_ordinals.is_empty() {
        (0..device_count).collect::<Vec<_>>()
    } else {
        config.device_ordinals.clone()
    };
    if ordinals.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.gpu.device_registry.config",
            "device registry requires at least one CUDA device",
        )
        .with_fact("field", "device_ordinals")
        .with_fact("reason", "empty"));
    }
    let original_len = ordinals.len();
    ordinals.sort_unstable();
    ordinals.dedup();
    if !config.device_ordinals.is_empty() && ordinals.len() != original_len {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.gpu.device_registry.config",
            "device registry ordinals must not contain duplicates",
        )
        .with_fact("field", "device_ordinals")
        .with_fact("reason", "duplicate"));
    }
    for ordinal in &ordinals {
        if *ordinal >= device_count {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.gpu.device_registry.config",
                "device registry ordinal is outside the visible CUDA device range",
            )
            .with_fact("field", "device_ordinals")
            .with_fact("device_ordinal", *ordinal)
            .with_fact("device_count", device_count)
            .with_fact("reason", "out_of_range"));
        }
    }
    Ok(ordinals)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::DeviceFacts;

    #[derive(Debug)]
    struct FakeInventory {
        facts: Vec<DeviceFacts>,
    }

    impl GpuInventory for FakeInventory {
        fn device_count(&self) -> Result<usize> {
            Ok(self.facts.len())
        }

        fn facts(&self, device_ordinal: usize) -> Result<DeviceFacts> {
            self.facts.get(device_ordinal).cloned().ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "nexus_query_engine.gpu.device_registry.test_inventory",
                    "missing fake device facts",
                )
            })
        }
    }

    #[test]
    fn configured_ordinals_default_to_all_inventory_devices() {
        let inventory = FakeInventory {
            facts: vec![
                DeviceFacts { device: 0, total_bytes: 100, free_bytes: 50, sm_count: Some(1), utilization_pct: None },
                DeviceFacts { device: 1, total_bytes: 100, free_bytes: 50, sm_count: Some(1), utilization_pct: None },
            ],
        };
        let config = DeviceRegistryConfig::new(Vec::new(), 1);

        assert_eq!(configured_ordinals(&inventory, &config).unwrap(), vec![0, 1]);
    }

    #[test]
    fn configured_ordinals_reject_duplicates() {
        let inventory = FakeInventory {
            facts: vec![DeviceFacts {
                device: 0,
                total_bytes: 100,
                free_bytes: 50,
                sm_count: Some(1),
                utilization_pct: None,
            }],
        };
        let config = DeviceRegistryConfig::new(vec![0, 0], 1);

        let error = configured_ordinals(&inventory, &config).unwrap_err();

        assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
        assert_eq!(error.fact_value("field"), Some("device_ordinals"));
        assert_eq!(error.fact_value("reason"), Some("duplicate"));
    }

    #[test]
    fn configured_ordinals_reject_out_of_range_before_context_vector_allocation() {
        let inventory = FakeInventory {
            facts: vec![DeviceFacts {
                device: 0,
                total_bytes: 100,
                free_bytes: 50,
                sm_count: Some(1),
                utilization_pct: None,
            }],
        };
        let config = DeviceRegistryConfig::new(vec![usize::MAX], 1);

        let error = configured_ordinals(&inventory, &config).unwrap_err();

        assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
        assert_eq!(error.fact_value("field"), Some("device_ordinals"));
        assert_eq!(error.fact_value("reason"), Some("out_of_range"));
    }
}

use std::sync::Arc;

use nexus_query_engine::{
    MemoryControlPolicy, QueryHandle, QueryService, QueryServiceConfig, QueryServiceTenantConfig, TenantId,
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, TryAcquireError};

use super::{
    config::ServerConfig,
    error::{Result, ServerError},
};

const DEFAULT_SERVER_TENANT_ID: &str = "default";
const SERVER_ADMISSION_QUEUE_MULTIPLIER: usize = 64;
const QUERY_SERVICE_ADMISSION_POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(10);

#[derive(Clone, Debug)]
enum QueryAdmissionBackend {
    QueryService { service: QueryService, tenant_id: TenantId },
    Semaphore { permits: Arc<Semaphore> },
}

#[derive(Debug)]
pub struct QueryAdmission {
    backend: QueryAdmissionBackend,
    max_in_flight: usize,
}

#[derive(Debug)]
struct ServerQueryServiceAdmissionConfig {
    service_config: QueryServiceConfig,
    device_ordinals: Vec<usize>,
    tenant_config: QueryServiceTenantConfig,
    tenant_id: TenantId,
}

impl QueryAdmission {
    pub fn new(max_in_flight: usize) -> Result<Self> {
        if max_in_flight == 0 {
            return Err(ServerError::invalid_config("max_in_flight", "must be greater than zero"));
        }
        Ok(Self {
            backend: QueryAdmissionBackend::Semaphore { permits: Arc::new(Semaphore::new(max_in_flight)) },
            max_in_flight,
        })
    }

    pub fn from_server_config(config: &ServerConfig) -> Result<Self> {
        let Some(admission_config) = server_query_service_admission_config(config)? else {
            return Self::new(config.max_in_flight);
        };
        let tenant_id = admission_config.tenant_id.clone();
        let service = QueryService::new_with_device_ordinals(
            admission_config.service_config,
            admission_config.device_ordinals,
            vec![admission_config.tenant_config],
        )
        .map_err(|error| ServerError::session("failed to initialize query service admission", error))?;
        Ok(Self {
            backend: QueryAdmissionBackend::QueryService { service, tenant_id },
            max_in_flight: config.max_in_flight,
        })
    }

    pub fn max_in_flight(&self) -> usize {
        self.max_in_flight
    }

    /// Close the admission gate. After close, any pending or future
    /// [`Self::acquire`] call returns a session error so graceful shutdown can
    /// drain in-flight work without admitting more.
    pub fn close(&self) {
        match &self.backend {
            QueryAdmissionBackend::QueryService { service, .. } => {
                if let Err(error) = service.close() {
                    eprintln!("[datafusion-nexus] warning: failed to close query service admission: {error}");
                }
            },
            QueryAdmissionBackend::Semaphore { permits } => permits.close(),
        }
    }

    pub async fn acquire(&self) -> Result<QueryAdmissionPermit> {
        match &self.backend {
            QueryAdmissionBackend::QueryService { service, tenant_id } => {
                if let Some(handle) = service
                    .try_admit_query(tenant_id.clone())
                    .map_err(|error| ServerError::session("query service admission failed", error))?
                {
                    return Ok(QueryAdmissionPermit {
                        inner: QueryAdmissionPermitInner::QueryService(Arc::new(handle)),
                    });
                }
                let mut waiter = service
                    .enqueue_admission(tenant_id.clone())
                    .map_err(|error| ServerError::session("query service admission failed", error))?;
                loop {
                    match waiter.try_admit() {
                        Ok(Some(handle)) => {
                            return Ok(QueryAdmissionPermit {
                                inner: QueryAdmissionPermitInner::QueryService(Arc::new(handle)),
                            });
                        },
                        Ok(None) => {
                            let sleep_for = waiter.remaining_wait().min(QUERY_SERVICE_ADMISSION_POLL_INTERVAL);
                            tokio::time::sleep(sleep_for).await;
                        },
                        Err(error) => return Err(ServerError::session("query service admission failed", error)),
                    }
                }
            },
            QueryAdmissionBackend::Semaphore { permits } => {
                let permit = permits
                    .clone()
                    .acquire_owned()
                    .await
                    .map_err(|error| ServerError::session("query admission queue closed", error))?;
                Ok(QueryAdmissionPermit { inner: QueryAdmissionPermitInner::Semaphore { _permit: permit } })
            },
        }
    }

    pub fn try_acquire(&self) -> Result<Option<QueryAdmissionPermit>> {
        match &self.backend {
            QueryAdmissionBackend::QueryService { service, tenant_id } => service
                .try_admit_query(tenant_id.clone())
                .map(|handle| {
                    handle.map(|handle| QueryAdmissionPermit {
                        inner: QueryAdmissionPermitInner::QueryService(Arc::new(handle)),
                    })
                })
                .map_err(|error| ServerError::session("query service admission failed", error)),
            QueryAdmissionBackend::Semaphore { permits } => match permits.clone().try_acquire_owned() {
                Ok(permit) => {
                    Ok(Some(QueryAdmissionPermit { inner: QueryAdmissionPermitInner::Semaphore { _permit: permit } }))
                },
                Err(TryAcquireError::NoPermits) => Ok(None),
                Err(error @ TryAcquireError::Closed) => {
                    Err(ServerError::session("query admission queue closed", error))
                },
            },
        }
    }
}

#[derive(Debug)]
pub struct QueryAdmissionPermit {
    inner: QueryAdmissionPermitInner,
}

#[derive(Debug)]
enum QueryAdmissionPermitInner {
    QueryService(Arc<QueryHandle>),
    Semaphore { _permit: OwnedSemaphorePermit },
}

impl QueryAdmissionPermit {
    pub fn query_service_handle(&self) -> Option<Arc<QueryHandle>> {
        match &self.inner {
            QueryAdmissionPermitInner::QueryService(handle) => Some(Arc::clone(handle)),
            QueryAdmissionPermitInner::Semaphore { .. } => None,
        }
    }
}

impl Drop for QueryAdmissionPermit {
    fn drop(&mut self) {
        if let QueryAdmissionPermitInner::QueryService(handle) = &self.inner {
            handle.finish();
        }
    }
}

fn server_stream_pool_size(max_concurrent_queries: usize, max_source_reads_in_flight: usize) -> usize {
    let per_query_streams = 1usize.saturating_add(max_source_reads_in_flight.max(1));
    max_concurrent_queries.saturating_mul(per_query_streams).max(max_concurrent_queries)
}

fn server_admission_queue_len(max_concurrent_queries: usize) -> usize {
    max_concurrent_queries.saturating_mul(SERVER_ADMISSION_QUEUE_MULTIPLIER).max(SERVER_ADMISSION_QUEUE_MULTIPLIER)
}

fn server_query_service_admission_config(config: &ServerConfig) -> Result<Option<ServerQueryServiceAdmissionConfig>> {
    let native = config.native_optimizer_config.native_execution_config();
    let MemoryControlPolicy::Bounded {
        device_budget_bytes,
        max_source_chunk_bytes,
        max_row_groups_per_chunk,
        max_retry_attempts,
        min_retry_chunk_row_groups,
    } = native.memory_control_policy()
    else {
        return Ok(None);
    };
    let max_in_flight = config.max_in_flight;
    if max_in_flight == 0 {
        return Err(ServerError::invalid_config("max_in_flight", "must be greater than zero"));
    }
    let tenant_min_budget_bytes = partitioned_query_budget_bytes(device_budget_bytes, max_in_flight)?;
    let device_ordinal = *config.gpu_devices.first().unwrap_or(&0);
    // Phase 2 keeps the deadlock-sensitive order explicit:
    // admission permit -> root/source stream leases -> bounded memory grants.
    // The service pool covers every admitted query's root stream plus its
    // configured source-read fan-out, so an admitted bounded query cannot
    // consume a memory grant while waiting on an undersized stream pool.
    let stream_pool_size =
        server_stream_pool_size(max_in_flight, native.source_scheduler_policy().max_source_reads_in_flight());
    let service_config = QueryServiceConfig {
        device_ordinal,
        source_fetch: native.source_fetch_config().clone(),
        source_storage_profiles: native.source_storage_profiles().to_vec(),
        boundary_policy: native.boundary_policy(),
        source_scheduler_policy: native.source_scheduler_policy(),
        source_chunk_planning_mode: native.source_chunk_planning_mode(),
        native_pipeline_execution_mode: native.native_pipeline_execution_mode(),
        stddev_samp_execution_mode: native.stddev_samp_execution_mode(),
        dynamic_filter_execution_options: native.dynamic_filter_execution_options(),
        keyed_hash_join_strategy_options: native.keyed_hash_join_strategy_options(),
        aggregate_strategy_execution_options: native.aggregate_strategy_execution_options(),
        source_chunk_planning_options: native.source_chunk_planning_options(),
        stream_pool_size: Some(stream_pool_size),
        device_capacity_bytes: Some(device_budget_bytes),
        max_concurrent_queries: max_in_flight,
        max_admission_queue_len: server_admission_queue_len(max_in_flight),
        max_source_chunk_bytes,
        max_row_groups_per_chunk,
        max_retry_attempts,
        min_retry_chunk_row_groups,
        ..QueryServiceConfig::default()
    };
    let tenant_id = TenantId::new(DEFAULT_SERVER_TENANT_ID);
    let tenant_config =
        QueryServiceTenantConfig::with_grant_range(tenant_id.clone(), tenant_min_budget_bytes, device_budget_bytes);
    Ok(Some(ServerQueryServiceAdmissionConfig {
        service_config,
        device_ordinals: config.gpu_devices.clone(),
        tenant_config,
        tenant_id,
    }))
}

fn partitioned_query_budget_bytes(capacity_bytes: u64, max_concurrent_queries: usize) -> Result<u64> {
    capacity_bytes
        .checked_div(u64::try_from(max_concurrent_queries).unwrap_or(u64::MAX))
        .filter(|budget| *budget > 0)
        .ok_or_else(|| {
            ServerError::invalid_config(
                "native.device_budget_bytes",
                "is too small to partition across max concurrent queries",
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn server_stream_pool_size_covers_root_and_source_read_streams() {
        assert_eq!(server_stream_pool_size(1, 1), 2);
        assert_eq!(server_stream_pool_size(4, 1), 8);
        assert_eq!(server_stream_pool_size(4, 3), 16);
    }

    #[test]
    fn partitioned_query_budget_never_exceeds_capacity_sum() {
        let per_query = partitioned_query_budget_bytes(100, 3).unwrap();

        assert_eq!(per_query, 33);
        assert!(per_query * 3 <= 100);
    }

    #[test]
    fn server_query_service_config_partitions_budget_and_sizes_stream_pool() {
        let env = std::collections::BTreeMap::from([
            (super::super::config::NEXUS_SERVER_MAX_CONCURRENT_QUERIES_ENV, "3".to_string()),
            (super::super::config::NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV, "3072".to_string()),
            (super::super::config::NEXUS_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, "1024".to_string()),
            (super::super::config::NEXUS_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV, "1".to_string()),
            (super::super::config::NEXUS_SERVER_NATIVE_MAX_SOURCE_READS_IN_FLIGHT_ENV, "2".to_string()),
        ]);
        let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

        let admission = server_query_service_admission_config(&config)
            .unwrap()
            .expect("bounded native config should use QueryService admission");

        assert_eq!(admission.service_config.max_concurrent_queries, 3);
        assert_eq!(admission.device_ordinals, vec![0]);
        assert_eq!(admission.service_config.device_capacity_bytes, Some(3072));
        assert_eq!(admission.tenant_config.min_device_budget_bytes, 1024);
        assert_eq!(admission.tenant_config.device_budget_bytes, 3072);
        assert_eq!(admission.service_config.stream_pool_size, Some(9));
        assert_eq!(admission.service_config.source_scheduler_policy.max_source_reads_in_flight(), 2);
    }

    #[test]
    fn server_query_service_config_preserves_multi_gpu_device_subset() {
        let env = std::collections::BTreeMap::from([
            (super::super::config::NEXUS_SERVER_GPU_DEVICES_ENV, "0,1".to_string()),
            (super::super::config::NEXUS_SERVER_MAX_CONCURRENT_QUERIES_ENV, "2".to_string()),
            (super::super::config::NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV, "2048".to_string()),
            (super::super::config::NEXUS_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, "1024".to_string()),
            (super::super::config::NEXUS_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV, "1".to_string()),
        ]);
        let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

        let admission = server_query_service_admission_config(&config)
            .unwrap()
            .expect("bounded native config should use QueryService admission");

        assert_eq!(admission.device_ordinals, vec![0, 1]);
        assert_eq!(admission.service_config.device_ordinal, 0);
    }
}

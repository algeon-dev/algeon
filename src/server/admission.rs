use std::sync::Arc;

use nexus_query_engine::{QueryHandle, TenantId};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, TryAcquireError};

use super::{
    config::ServerConfig,
    error::{Result, ServerError},
};
use crate::{NexusAdmissionPolicy, NexusGpuBackend};

const DEFAULT_SERVER_TENANT_ID: &str = "default";

#[derive(Clone, Debug)]
enum QueryAdmissionBackend {
    QueryService { backend: NexusGpuBackend },
    Semaphore { permits: Arc<Semaphore> },
}

#[derive(Debug)]
pub struct QueryAdmission {
    backend: QueryAdmissionBackend,
    max_in_flight: usize,
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
        let backend = server_backend_from_config(config)?;
        Self::from_backend_option(config.max_in_flight, backend)
    }

    #[cfg(test)]
    fn from_server_config_admission_only(config: &ServerConfig) -> Result<Self> {
        let backend = server_backend_from_config_admission_only(config)?;
        Self::from_backend_option(config.max_in_flight, backend)
    }

    fn from_backend_option(max_in_flight: usize, backend: Option<NexusGpuBackend>) -> Result<Self> {
        let Some(backend) = backend else {
            return Self::new(max_in_flight);
        };
        Ok(Self { backend: QueryAdmissionBackend::QueryService { backend }, max_in_flight })
    }

    pub fn max_in_flight(&self) -> usize {
        self.max_in_flight
    }

    /// Close the admission gate. After close, any pending or future
    /// [`Self::acquire`] call returns a session error so graceful shutdown can
    /// drain in-flight work without admitting more.
    pub fn close(&self) {
        match &self.backend {
            QueryAdmissionBackend::QueryService { backend } => {
                if let Err(error) = backend.query_service().close() {
                    eprintln!("[datafusion-nexus] warning: failed to close query service admission: {error}");
                }
            },
            QueryAdmissionBackend::Semaphore { permits } => permits.close(),
        }
    }

    pub async fn acquire(&self) -> Result<QueryAdmissionPermit> {
        match &self.backend {
            QueryAdmissionBackend::QueryService { backend } => {
                let handle = backend
                    .query_service()
                    .admit_async(backend.tenant_id().clone())
                    .await
                    .map_err(|error| ServerError::session("query service admission failed", error))?;
                Ok(QueryAdmissionPermit { inner: QueryAdmissionPermitInner::QueryService(Arc::new(handle)) })
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
            QueryAdmissionBackend::QueryService { backend } => backend
                .query_service()
                .try_admit_query(backend.tenant_id().clone())
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

    pub fn log_query_service_state(&self, query_id: Option<&str>, event: &'static str) {
        let QueryAdmissionBackend::QueryService { backend } = &self.backend else {
            return;
        };
        let service = backend.query_service();
        match service.admission_snapshot() {
            Ok(snapshot) => {
                tracing::info!(
                    query_id = query_id.unwrap_or(""),
                    event,
                    active_queries = snapshot.active_queries,
                    queued_queries = snapshot.queued_queries,
                    next_tenant_id = snapshot.next_tenant_id.as_ref().map(TenantId::as_str).unwrap_or(""),
                    tenant_queue_depths = %format_tenant_queue_depths(&snapshot.queued_by_tenant),
                    "query service admission snapshot"
                );
            },
            Err(error) => tracing::warn!(
                query_id = query_id.unwrap_or(""),
                event,
                %error,
                "failed to read query service admission snapshot"
            ),
        }
        match service.memory_governor_reports() {
            Ok(reports) => {
                for report in reports {
                    tracing::info!(
                        query_id = query_id.unwrap_or(""),
                        event,
                        device_ordinal = report.device_ordinal,
                        capacity_bytes = report.capacity_bytes,
                        outstanding_bytes = report.outstanding_bytes,
                        available_bytes = report.available_bytes,
                        active_grants = report.grants.len(),
                        "query service memory governor report"
                    );
                }
            },
            Err(error) => tracing::warn!(
                query_id = query_id.unwrap_or(""),
                event,
                %error,
                "failed to read query service memory governor report"
            ),
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

fn server_backend_from_config(config: &ServerConfig) -> Result<Option<NexusGpuBackend>> {
    NexusGpuBackend::from_optimizer_config_for_admission(
        config.native_optimizer_config.clone(),
        config.gpu_devices.clone(),
        server_admission_policy(config),
        TenantId::new(DEFAULT_SERVER_TENANT_ID),
    )
    .map_err(|error| ServerError::session("failed to initialize query service admission", error))
}

#[cfg(test)]
fn server_backend_from_config_admission_only(config: &ServerConfig) -> Result<Option<NexusGpuBackend>> {
    NexusGpuBackend::from_optimizer_config_for_admission_only(
        config.native_optimizer_config.clone(),
        config.gpu_devices.clone(),
        server_admission_policy(config),
        TenantId::new(DEFAULT_SERVER_TENANT_ID),
    )
    .map_err(|error| ServerError::session("failed to initialize query service admission", error))
}

fn server_admission_policy(config: &ServerConfig) -> NexusAdmissionPolicy {
    config.admission_policy.clone()
}

fn format_tenant_queue_depths(depths: &[nexus_query_engine::TenantQueueDepth]) -> String {
    depths
        .iter()
        .map(|depth| format!("{}:{}", depth.tenant_id.as_str(), depth.queued_queries))
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn server_backend_partitions_budget_and_sizes_stream_pool() {
        let env = std::collections::BTreeMap::from([
            (super::super::config::NEXUS_SERVER_MAX_CONCURRENT_QUERIES_ENV, "3".to_string()),
            (super::super::config::NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV, "3072".to_string()),
            (super::super::config::NEXUS_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, "1024".to_string()),
            (super::super::config::NEXUS_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV, "1".to_string()),
            (super::super::config::NEXUS_SERVER_NATIVE_MAX_SOURCE_READS_IN_FLIGHT_ENV, "2".to_string()),
        ]);
        let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

        let admission = QueryAdmission::from_server_config_admission_only(&config).unwrap();
        let QueryAdmissionBackend::QueryService { backend } = &admission.backend else {
            panic!("bounded native config should use QueryService admission");
        };
        let service = backend.query_service();

        assert_eq!(service.config().max_concurrent_queries, 3);
        assert_eq!(service.config().device_ordinal, 0);
        assert_eq!(service.config().device_capacity_bytes, Some(3072));
        assert_eq!(service.config().stream_pool_size, Some(9));
        assert_eq!(service.config().source_scheduler_policy.max_source_reads_in_flight(), 2);
        assert_eq!(service.config().max_admission_queue_len, 192);
        assert_eq!(service.config().admission_wait_timeout, QueryServiceConfig::default().admission_wait_timeout);
        assert_eq!(service.tenant_budget(backend.tenant_id()).unwrap(), 1536);

        let permit = admission.try_acquire().unwrap().expect("first query should admit");
        let handle = permit.query_service_handle().expect("query service admission should return a handle");
        assert_eq!(handle.device_budget_bytes(), 1536);
    }

    #[test]
    fn server_backend_preserves_multi_gpu_device_subset() {
        let env = std::collections::BTreeMap::from([
            (super::super::config::NEXUS_SERVER_GPU_DEVICES_ENV, "0,1".to_string()),
            (super::super::config::NEXUS_SERVER_MAX_CONCURRENT_QUERIES_ENV, "2".to_string()),
            (super::super::config::NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV, "2048".to_string()),
            (super::super::config::NEXUS_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, "1024".to_string()),
            (super::super::config::NEXUS_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV, "1".to_string()),
        ]);
        let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

        let admission = QueryAdmission::from_server_config_admission_only(&config).unwrap();
        let QueryAdmissionBackend::QueryService { backend } = &admission.backend else {
            panic!("bounded native config should use QueryService admission");
        };

        assert_eq!(backend.device_ordinals(), &[0, 1]);
        assert_eq!(backend.query_service().config().device_ordinal, 0);
        let permit = admission.try_acquire().unwrap().expect("first query should admit");
        let handle = permit.query_service_handle().expect("query service admission should return a handle");
        assert_eq!(handle.device_ordinal(), 0);
        assert_eq!(handle.device_budget_bytes(), 1024);
    }
}

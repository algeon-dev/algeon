use std::{fmt::Write, sync::Arc};

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
    QueryService { backend: Box<NexusGpuBackend> },
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
        Ok(Self { backend: QueryAdmissionBackend::QueryService { backend: Box::new(backend) }, max_in_flight })
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
        for report in service.device_memory_failure_reports() {
            tracing::info!(
                query_id = query_id.unwrap_or(""),
                event,
                device_ordinal = report.device_ordinal,
                callback_installed = report.callback_installed,
                callback_install_unavailable = report.callback_install_unavailable,
                allocation_failure_count = report.allocation_failure_count,
                total_failed_allocation_bytes = report.total_failed_allocation_bytes,
                largest_failed_allocation_bytes = report.largest_failed_allocation_bytes,
                "query service device memory failure report"
            );
        }
    }

    pub fn diagnostics_payload(&self) -> Result<String> {
        let mut output = String::new();
        let _ = writeln!(output, "# datafusion_nexus_admission_stats=1");
        let _ = writeln!(output, "# max_in_flight={}", self.max_in_flight);
        match &self.backend {
            QueryAdmissionBackend::QueryService { backend } => {
                let service = backend.query_service();
                let config = service.config();
                let _ = writeln!(output, "# backend=query_service");
                let _ = writeln!(output, "# tenant_id={}", payload_value(backend.tenant_id().as_str()));
                let _ = writeln!(output, "# device_ordinals={}", payload_value(&join_usize(backend.device_ordinals())));
                let _ = writeln!(output, "# max_concurrent_queries={}", config.max_concurrent_queries);
                let _ = writeln!(output, "# max_admission_queue_len={}", config.max_admission_queue_len);
                let _ = writeln!(output, "# admission_wait_timeout_ms={}", config.admission_wait_timeout.as_millis());
                let _ = writeln!(output, "# slot_guarantee={}", slot_guarantee_name(config.slot_guarantee));

                let snapshot = service
                    .admission_snapshot()
                    .map_err(|error| ServerError::session("failed to read query service admission snapshot", error))?;
                output.push_str("\n[admission]\n");
                output.push_str("active_queries\tqueued_queries\tnext_tenant_id\n");
                let _ = writeln!(
                    output,
                    "{}\t{}\t{}",
                    snapshot.active_queries,
                    snapshot.queued_queries,
                    payload_value(snapshot.next_tenant_id.as_ref().map(TenantId::as_str).unwrap_or(""))
                );

                output.push_str("\n[tenant_queue_depths]\n");
                output.push_str("tenant_id\tqueued_queries\n");
                for depth in snapshot.queued_by_tenant {
                    let _ = writeln!(output, "{}\t{}", payload_value(depth.tenant_id.as_str()), depth.queued_queries);
                }

                let reports = service.memory_governor_reports().map_err(|error| {
                    ServerError::session("failed to read query service memory governor reports", error)
                })?;
                output.push_str("\n[memory_governor_reports]\n");
                output.push_str("device_ordinal\tcapacity_bytes\toutstanding_bytes\tavailable_bytes\tactive_grants\n");
                for report in &reports {
                    let _ = writeln!(
                        output,
                        "{}\t{}\t{}\t{}\t{}",
                        report.device_ordinal,
                        report.capacity_bytes,
                        report.outstanding_bytes,
                        report.available_bytes,
                        report.grants.len()
                    );
                }

                output.push_str("\n[memory_governor_grants]\n");
                output.push_str("device_ordinal\tgrant_id\tquery_id\tpriority\tbudget_bytes\n");
                for report in &reports {
                    for grant in &report.grants {
                        let _ = writeln!(
                            output,
                            "{}\t{}\t{}\t{}\t{}",
                            report.device_ordinal,
                            grant.grant_id,
                            grant.query_id,
                            payload_value(grant.priority),
                            grant.budget_bytes
                        );
                    }
                }

                output.push_str("\n[device_memory_failure_reports]\n");
                output.push_str(
                    "device_ordinal\tcallback_installed\tcallback_install_unavailable\tallocation_failure_count\ttotal_failed_allocation_bytes\tlargest_failed_allocation_bytes\n",
                );
                for report in service.device_memory_failure_reports() {
                    let _ = writeln!(
                        output,
                        "{}\t{}\t{}\t{}\t{}\t{}",
                        report.device_ordinal,
                        report.callback_installed,
                        report.callback_install_unavailable,
                        report.allocation_failure_count,
                        report.total_failed_allocation_bytes,
                        report.largest_failed_allocation_bytes
                    );
                }
            },
            QueryAdmissionBackend::Semaphore { permits } => {
                let available_permits = permits.available_permits();
                let active_queries = self.max_in_flight.saturating_sub(available_permits);
                let _ = writeln!(output, "# backend=semaphore");
                output.push_str("\n[admission]\n");
                output.push_str("active_queries\tqueued_queries\tnext_tenant_id\n");
                let _ = writeln!(output, "{active_queries}\t\t");

                output.push_str("\n[semaphore]\n");
                output.push_str("max_in_flight\tavailable_permits\n");
                let _ = writeln!(output, "{}\t{}", self.max_in_flight, available_permits);
            },
        }
        Ok(output)
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
    // `ServerConfig` keeps the legacy public `max_in_flight` field beside the
    // richer admission policy. Normalize here so direct test/operator config
    // mutation still controls the QueryService concurrency gate.
    config.admission_policy.clone().with_max_concurrent_queries(config.max_in_flight)
}

fn format_tenant_queue_depths(depths: &[nexus_query_engine::TenantQueueDepth]) -> String {
    depths
        .iter()
        .map(|depth| format!("{}:{}", depth.tenant_id.as_str(), depth.queued_queries))
        .collect::<Vec<_>>()
        .join(",")
}

fn slot_guarantee_name(slot_guarantee: nexus_query_engine::SlotGuarantee) -> &'static str {
    match slot_guarantee {
        nexus_query_engine::SlotGuarantee::GuaranteedMinimum => "guaranteed_minimum",
        nexus_query_engine::SlotGuarantee::Optimistic => "optimistic",
    }
}

fn join_usize(values: &[usize]) -> String {
    values.iter().map(usize::to_string).collect::<Vec<_>>().join(",")
}

fn payload_value(value: &str) -> String {
    value.replace(['\t', '\n', '\r'], " ")
}

#[cfg(test)]
mod tests {
    use nexus_query_engine::QueryServiceConfig;

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

    #[test]
    fn query_service_admission_diagnostics_payload_reports_snapshot_and_governors() {
        let env = std::collections::BTreeMap::from([
            (super::super::config::NEXUS_SERVER_MAX_CONCURRENT_QUERIES_ENV, "2".to_string()),
            (super::super::config::NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV, "1024".to_string()),
            (super::super::config::NEXUS_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, "512".to_string()),
            (super::super::config::NEXUS_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV, "1".to_string()),
            (super::super::config::NEXUS_SERVER_ADMISSION_GRANT_MIN_ENV, "256".to_string()),
            (super::super::config::NEXUS_SERVER_ADMISSION_GRANT_WANT_ENV, "512".to_string()),
        ]);
        let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();
        let admission = QueryAdmission::from_server_config_admission_only(&config).unwrap();
        let permit = admission.try_acquire().unwrap().expect("first query should admit");

        let payload = admission.diagnostics_payload().unwrap();

        assert!(payload.contains("# datafusion_nexus_admission_stats=1\n"), "{payload}");
        assert!(payload.contains("# backend=query_service\n"), "{payload}");
        assert!(payload.contains("# max_in_flight=2\n"), "{payload}");
        assert!(payload.contains("# max_concurrent_queries=2\n"), "{payload}");
        assert!(payload.contains("# slot_guarantee=guaranteed_minimum\n"), "{payload}");
        assert!(
            payload.contains("[admission]\nactive_queries\tqueued_queries\tnext_tenant_id\n1\t0\tdefault\n"),
            "{payload}"
        );
        assert!(payload.contains("[tenant_queue_depths]\ntenant_id\tqueued_queries\ndefault\t0\n"), "{payload}");
        assert!(
            payload.contains(
                "[memory_governor_reports]\ndevice_ordinal\tcapacity_bytes\toutstanding_bytes\tavailable_bytes\tactive_grants\n0\t1024\t512\t512\t1\n"
            ),
            "{payload}"
        );
        assert!(payload.contains("[memory_governor_grants]\n"), "{payload}");
        assert!(payload.contains("\t1\tnormal\t512\n"), "{payload}");
        assert!(
            payload.contains(
                "[device_memory_failure_reports]\ndevice_ordinal\tcallback_installed\tcallback_install_unavailable\tallocation_failure_count\ttotal_failed_allocation_bytes\tlargest_failed_allocation_bytes\n"
            ),
            "{payload}"
        );

        drop(permit);
    }

    #[test]
    fn semaphore_admission_diagnostics_payload_reports_permit_counts() {
        let admission = QueryAdmission::new(2).unwrap();
        let _permit = admission.try_acquire().unwrap().expect("first query should admit");

        let payload = admission.diagnostics_payload().unwrap();

        assert!(payload.contains("# datafusion_nexus_admission_stats=1\n"), "{payload}");
        assert!(payload.contains("# backend=semaphore\n"), "{payload}");
        assert!(payload.contains("# max_in_flight=2\n"), "{payload}");
        assert!(payload.contains("[admission]\nactive_queries\tqueued_queries\tnext_tenant_id\n1\t\t\n"), "{payload}");
        assert!(payload.contains("[semaphore]\nmax_in_flight\tavailable_permits\n2\t1\n"), "{payload}");
    }
}

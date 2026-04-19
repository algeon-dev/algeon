use std::{fmt::Write, num::NonZeroUsize, sync::Arc};

#[cfg(test)]
use nexus_query_engine::QueryHandle;
use nexus_query_engine::TenantId;
use tokio::sync::Semaphore;
#[cfg(test)]
use tokio::sync::{OwnedSemaphorePermit, TryAcquireError};

use super::{
    config::ServerConfig,
    error::{Result, ServerError},
};
use crate::{NexusAdmissionPolicy, NexusGpuBackend, native_exec::UnboundedNativeExecutionLimiter};

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

    #[cfg(test)]
    fn from_server_config_admission_only(config: &ServerConfig) -> Result<Self> {
        let backend = server_backend_from_config_admission_only(config)?;
        Self::from_backend_option(config.max_in_flight, backend)
    }

    pub(crate) fn from_backend(backend: Option<&NexusGpuBackend>, config: &ServerConfig) -> Result<Self> {
        Self::from_backend_option(config.max_in_flight, backend.cloned())
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

    pub(crate) fn unbounded_native_execution_limiter(&self) -> Option<UnboundedNativeExecutionLimiter> {
        let QueryAdmissionBackend::Semaphore { permits } = &self.backend else {
            return None;
        };
        let max_concurrent_queries =
            NonZeroUsize::new(self.max_in_flight).expect("QueryAdmission max_in_flight is validated nonzero");
        Some(UnboundedNativeExecutionLimiter::from_shared_semaphore(Arc::clone(permits), max_concurrent_queries))
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

    #[cfg(test)]
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
                        evictable_bytes = report.evictable_bytes,
                        free_bytes = report.free_bytes,
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
        for snapshot in service.device_admission_snapshots() {
            tracing::info!(
                query_id = query_id.unwrap_or(""),
                event,
                device_ordinal = snapshot.device_ordinal,
                process_active_query_slots = snapshot.process_active_query_slots,
                "query service device admission snapshot"
            );
        }
        for snapshot in service.device_telemetry_snapshots() {
            tracing::info!(
                query_id = query_id.unwrap_or(""),
                event,
                sampled_at_unix_ms = snapshot.sampled_at_unix_ms,
                device_ordinal = snapshot.device_ordinal,
                cuda_status = snapshot.cuda_status.as_str(),
                cuda_total_bytes = ?snapshot.cuda_total_bytes,
                cuda_free_bytes = ?snapshot.cuda_free_bytes,
                nvml_status = snapshot.nvml_status.as_str(),
                nvml_total_bytes = ?snapshot.nvml_total_bytes,
                nvml_reserved_bytes = ?snapshot.nvml_reserved_bytes,
                nvml_used_bytes = ?snapshot.nvml_used_bytes,
                nvml_free_bytes = ?snapshot.nvml_free_bytes,
                gpu_utilization_pct = ?snapshot.gpu_utilization_pct,
                memory_utilization_pct = ?snapshot.memory_utilization_pct,
                "query service device telemetry snapshot"
            );
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
                trim_retry_count = report.trim_retry_count,
                pool_usage_observation_count = report.pool_usage_observation_count,
                last_failed_allocation_bytes = report.last_failed_allocation_bytes,
                last_pool_reserved_current_bytes = report.last_pool_reserved_current_bytes,
                last_pool_used_current_bytes = report.last_pool_used_current_bytes,
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
                let _ = writeln!(
                    output,
                    "# max_queries_per_device={}",
                    config.max_queries_per_device.map_or_else(|| "none".to_string(), |value| value.to_string())
                );
                let _ = writeln!(output, "# placement_policy={}", config.placement_policy.as_str());
                if let nexus_query_engine::QueryPlacementPolicy::Pinned { device_ordinal } = config.placement_policy {
                    let _ = writeln!(output, "# pinned_device_ordinal={device_ordinal}");
                }
                let _ = writeln!(output, "# max_admission_queue_len={}", config.max_admission_queue_len);
                let _ = writeln!(output, "# admission_wait_timeout_ms={}", config.admission_wait_timeout.as_millis());
                let _ = writeln!(output, "# slot_guarantee={}", slot_guarantee_name(config.slot_guarantee));
                let _ = writeln!(
                    output,
                    "# local_parquet_source_cache_max_retained_bytes={}",
                    config
                        .local_parquet_source_cache_max_retained_bytes
                        .map_or_else(|| "none".to_string(), |value| value.to_string())
                );
                let _ = writeln!(
                    output,
                    "# resident_graph_cache_max_retained_bytes={}",
                    config
                        .resident_graph_cache_max_retained_bytes
                        .map_or_else(|| "none".to_string(), |value| value.to_string())
                );
                let _ = writeln!(
                    output,
                    "# query_memory_domain_ceiling_multiplier={}",
                    config
                        .query_memory_domain
                        .map_or_else(|| "none".to_string(), |policy| policy.ceiling_multiplier.to_string())
                );

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
                output.push_str(
                    "device_ordinal\tcapacity_bytes\toutstanding_bytes\tavailable_bytes\tactive_grants\tevictable_bytes\tfree_bytes\n",
                );
                for report in &reports {
                    let _ = writeln!(
                        output,
                        "{}\t{}\t{}\t{}\t{}\t{}\t{}",
                        report.device_ordinal,
                        report.capacity_bytes,
                        report.outstanding_bytes,
                        report.available_bytes,
                        report.grants.len(),
                        report.evictable_bytes,
                        report.free_bytes
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
                    "device_ordinal\tcallback_installed\tcallback_install_unavailable\tallocation_failure_count\ttotal_failed_allocation_bytes\tlargest_failed_allocation_bytes\ttrim_retry_count\tpool_usage_observation_count\tlast_failed_allocation_bytes\tlast_pool_reserved_current_bytes\tlast_pool_used_current_bytes\n",
                );
                for report in service.device_memory_failure_reports() {
                    let _ = writeln!(
                        output,
                        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                        report.device_ordinal,
                        report.callback_installed,
                        report.callback_install_unavailable,
                        report.allocation_failure_count,
                        report.total_failed_allocation_bytes,
                        report.largest_failed_allocation_bytes,
                        report.trim_retry_count,
                        report.pool_usage_observation_count,
                        report.last_failed_allocation_bytes,
                        report.last_pool_reserved_current_bytes,
                        report.last_pool_used_current_bytes
                    );
                }

                let query_execution = service.gpu_execution_query_report();
                output.push_str("\n[gpu_execution]\n");
                output.push_str("active_query_jobs\tpeak_active_query_jobs\n");
                let _ = writeln!(
                    output,
                    "{}\t{}",
                    query_execution.active_query_jobs, query_execution.peak_active_query_jobs
                );

                output.push_str("\n[gpu_execution_pools]\n");
                output.push_str(
                    "device_ordinal\tworker_count\tactive_jobs\tpeak_active_jobs\tqueued_jobs\tpeak_queued_jobs\tactive_query_jobs\tpeak_active_query_jobs\n",
                );
                for report in service.gpu_execution_pool_reports() {
                    let _ = writeln!(
                        output,
                        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                        report.device_ordinal,
                        report.worker_count,
                        report.active_jobs,
                        report.peak_active_jobs,
                        report.queued_jobs,
                        report.peak_queued_jobs,
                        report.active_query_jobs,
                        report.peak_active_query_jobs,
                    );
                }

                output.push_str("\n[device_admission_slots]\n");
                output.push_str("device_ordinal\tprocess_active_query_slots\n");
                for snapshot in service.device_admission_snapshots() {
                    let _ = writeln!(output, "{}\t{}", snapshot.device_ordinal, snapshot.process_active_query_slots);
                }

                output.push_str("\n[device_telemetry]\n");
                output.push_str(
                    "sampled_at_unix_ms\tdevice_ordinal\tcuda_status\tcuda_total_bytes\tcuda_free_bytes\tnvml_status\tnvml_total_bytes\tnvml_reserved_bytes\tnvml_used_bytes\tnvml_free_bytes\tgpu_utilization_pct\tmemory_utilization_pct\n",
                );
                for snapshot in service.device_telemetry_snapshots() {
                    let _ = writeln!(
                        output,
                        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                        snapshot.sampled_at_unix_ms,
                        snapshot.device_ordinal,
                        snapshot.cuda_status.as_str(),
                        optional_payload_value(snapshot.cuda_total_bytes),
                        optional_payload_value(snapshot.cuda_free_bytes),
                        snapshot.nvml_status.as_str(),
                        optional_payload_value(snapshot.nvml_total_bytes),
                        optional_payload_value(snapshot.nvml_reserved_bytes),
                        optional_payload_value(snapshot.nvml_used_bytes),
                        optional_payload_value(snapshot.nvml_free_bytes),
                        optional_payload_value(snapshot.gpu_utilization_pct),
                        optional_payload_value(snapshot.memory_utilization_pct),
                    );
                }

                output.push_str("\n[graph_cache]\n");
                output.push_str(
                    "device_ordinal\tenabled\tentries\tresident_bytes\tin_use_bytes\thits\tmisses\tevictions\tbuilds_in_flight\tbuild_waits\n",
                );
                for snapshot in service
                    .resident_graph_cache_snapshots()
                    .map_err(|error| ServerError::session("failed to read resident graph cache snapshots", error))?
                {
                    let _ = writeln!(
                        output,
                        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                        snapshot.device_ordinal,
                        snapshot.enabled,
                        snapshot.entries,
                        snapshot.resident_bytes,
                        snapshot.in_use_bytes,
                        snapshot.hits,
                        snapshot.misses,
                        snapshot.evictions,
                        snapshot.builds_in_flight,
                        snapshot.build_waits,
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

fn optional_payload_value(value: Option<impl ToString>) -> String {
    value.map_or_else(|| "none".to_string(), |value| value.to_string())
}

#[cfg(test)]
#[derive(Debug)]
pub struct QueryAdmissionPermit {
    inner: QueryAdmissionPermitInner,
}

#[cfg(test)]
#[derive(Debug)]
enum QueryAdmissionPermitInner {
    QueryService(Arc<QueryHandle>),
    Semaphore { _permit: OwnedSemaphorePermit },
}

#[cfg(test)]
impl QueryAdmissionPermit {
    pub fn query_service_handle(&self) -> Option<Arc<QueryHandle>> {
        match &self.inner {
            QueryAdmissionPermitInner::QueryService(handle) => Some(Arc::clone(handle)),
            QueryAdmissionPermitInner::Semaphore { .. } => None,
        }
    }
}

#[cfg(test)]
impl Drop for QueryAdmissionPermit {
    fn drop(&mut self) {
        if let QueryAdmissionPermitInner::QueryService(handle) = &self.inner {
            handle.finish();
        }
    }
}

impl NexusGpuBackend {
    /// Build the server's query-service backend from a validated server config.
    ///
    /// Returns `None` for unbounded native memory policy; the Flight SQL server
    /// then falls back to its legacy semaphore admission path.
    pub fn from_server_config(config: &ServerConfig) -> Result<Option<Self>> {
        Self::from_optimizer_config_for_admission(
            config.native_optimizer_config.clone(),
            config.gpu_devices.clone(),
            server_admission_policy(config),
            TenantId::new(DEFAULT_SERVER_TENANT_ID),
        )
        .map_err(|error| ServerError::session("failed to initialize query service admission", error))
    }
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

pub(crate) fn server_admission_policy(config: &ServerConfig) -> NexusAdmissionPolicy {
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
    use super::*;

    #[test]
    fn server_backend_partitions_budget_and_sizes_stream_pool() {
        let mut env = std::collections::BTreeMap::from([
            (super::super::config::NEXUS_SERVER_MAX_CONCURRENT_QUERIES_ENV, "3".to_string()),
            (super::super::config::NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV, "3072".to_string()),
            (super::super::config::NEXUS_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, "1024".to_string()),
            (super::super::config::NEXUS_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV, "1".to_string()),
            (super::super::config::NEXUS_SERVER_NATIVE_MAX_SOURCE_READS_IN_FLIGHT_ENV, "2".to_string()),
            (super::super::config::NEXUS_SERVER_NATIVE_SOURCE_CACHE_MAX_RETAINED_BYTES_ENV, "2048".to_string()),
        ]);
        if cfg!(feature = "cugraph") {
            env.insert(super::super::config::NEXUS_SERVER_CUGRAPH_ENABLED_ENV, "true".to_string());
            env.insert(
                super::super::config::NEXUS_SERVER_NATIVE_GRAPH_CACHE_MAX_RETAINED_BYTES_ENV,
                "4096".to_string(),
            );
        }
        let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

        let admission = QueryAdmission::from_server_config_admission_only(&config).unwrap();
        let QueryAdmissionBackend::QueryService { backend } = &admission.backend else {
            panic!("bounded native config should use QueryService admission");
        };
        let service = backend.query_service();

        assert_eq!(service.config().max_concurrent_queries, 3);
        assert_eq!(service.config().max_queries_per_device, None);
        assert_eq!(service.config().placement_policy, nexus_query_engine::QueryPlacementPolicy::LeastLoaded);
        assert_eq!(service.config().device_ordinal, 0);
        assert_eq!(service.config().device_capacity_bytes, Some(3072));
        assert_eq!(service.config().stream_pool_size, Some(9));
        assert_eq!(service.config().source_scheduler_policy.max_source_reads_in_flight(), 2);
        assert_eq!(service.config().max_admission_queue_len, 192);
        assert_eq!(service.config().admission_wait_timeout, std::time::Duration::from_secs(60));
        assert_eq!(service.config().local_parquet_source_cache_max_retained_bytes, Some(2048));
        #[cfg(feature = "cugraph")]
        assert_eq!(service.config().resident_graph_cache_max_retained_bytes, Some(4096));
        assert_eq!(service.tenant_budget(backend.tenant_id()).unwrap(), 3072);

        let permit = admission.try_acquire().unwrap().expect("first query should admit");
        let handle = permit.query_service_handle().expect("query service admission should return a handle");
        assert_eq!(handle.device_budget_bytes(), 2304);
    }

    #[test]
    fn server_backend_honors_admission_wait_timeout_env() {
        let env = std::collections::BTreeMap::from([
            (super::super::config::NEXUS_SERVER_ADMISSION_WAIT_TIMEOUT_MS_ENV, "250".to_string()),
            (super::super::config::NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV, "3072".to_string()),
            (super::super::config::NEXUS_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, "1024".to_string()),
            (super::super::config::NEXUS_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV, "1".to_string()),
        ]);
        let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

        let admission = QueryAdmission::from_server_config_admission_only(&config).unwrap();
        let QueryAdmissionBackend::QueryService { backend } = &admission.backend else {
            panic!("bounded native config should use QueryService admission");
        };

        assert_eq!(backend.query_service().config().admission_wait_timeout, std::time::Duration::from_millis(250));
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
        assert_eq!(handle.device_budget_bytes(), 1792);
    }

    #[test]
    fn query_service_admission_diagnostics_payload_reports_snapshot_and_governors() {
        let mut env = std::collections::BTreeMap::from([
            (super::super::config::NEXUS_SERVER_MAX_CONCURRENT_QUERIES_ENV, "2".to_string()),
            (super::super::config::NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV, "1024".to_string()),
            (super::super::config::NEXUS_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, "512".to_string()),
            (super::super::config::NEXUS_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV, "1".to_string()),
            (super::super::config::NEXUS_SERVER_ADMISSION_GRANT_MIN_ENV, "256".to_string()),
            (super::super::config::NEXUS_SERVER_ADMISSION_GRANT_WANT_ENV, "512".to_string()),
            (super::super::config::NEXUS_SERVER_NATIVE_SOURCE_CACHE_MAX_RETAINED_BYTES_ENV, "2048".to_string()),
        ]);
        if cfg!(feature = "cugraph") {
            env.insert(super::super::config::NEXUS_SERVER_CUGRAPH_ENABLED_ENV, "true".to_string());
            env.insert(
                super::super::config::NEXUS_SERVER_NATIVE_GRAPH_CACHE_MAX_RETAINED_BYTES_ENV,
                "4096".to_string(),
            );
        }
        let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();
        let admission = QueryAdmission::from_server_config_admission_only(&config).unwrap();
        let permit = admission.try_acquire().unwrap().expect("first query should admit");

        let payload = admission.diagnostics_payload().unwrap();

        assert!(payload.contains("# datafusion_nexus_admission_stats=1\n"), "{payload}");
        assert!(payload.contains("# backend=query_service\n"), "{payload}");
        assert!(payload.contains("# max_in_flight=2\n"), "{payload}");
        assert!(payload.contains("# max_concurrent_queries=2\n"), "{payload}");
        assert!(payload.contains("# max_queries_per_device=none\n"), "{payload}");
        assert!(payload.contains("# placement_policy=least_loaded\n"), "{payload}");
        assert!(payload.contains("# slot_guarantee=guaranteed_minimum\n"), "{payload}");
        assert!(payload.contains("# local_parquet_source_cache_max_retained_bytes=2048\n"), "{payload}");
        #[cfg(feature = "cugraph")]
        assert!(payload.contains("# resident_graph_cache_max_retained_bytes=4096\n"), "{payload}");
        #[cfg(not(feature = "cugraph"))]
        assert!(payload.contains("# resident_graph_cache_max_retained_bytes=none\n"), "{payload}");
        assert!(
            payload.contains("[admission]\nactive_queries\tqueued_queries\tnext_tenant_id\n1\t0\tdefault\n"),
            "{payload}"
        );
        assert!(payload.contains("[tenant_queue_depths]\ntenant_id\tqueued_queries\ndefault\t0\n"), "{payload}");
        assert!(payload.contains("[gpu_execution]\nactive_query_jobs\tpeak_active_query_jobs\n0\t0\n"), "{payload}");
        assert!(
            payload.contains(
                "[gpu_execution_pools]\ndevice_ordinal\tworker_count\tactive_jobs\tpeak_active_jobs\tqueued_jobs\tpeak_queued_jobs\tactive_query_jobs\tpeak_active_query_jobs\n"
            ),
            "{payload}"
        );
        assert!(
            payload.contains(
                "[memory_governor_reports]\ndevice_ordinal\tcapacity_bytes\toutstanding_bytes\tavailable_bytes\tactive_grants\tevictable_bytes\tfree_bytes\n0\t1024\t512\t512\t1\t0\t512\n"
            ),
            "{payload}"
        );
        assert!(payload.contains("[memory_governor_grants]\n"), "{payload}");
        assert!(payload.contains("\t1\tnormal\t512\n"), "{payload}");
        assert!(
            payload.contains(
                "[device_memory_failure_reports]\ndevice_ordinal\tcallback_installed\tcallback_install_unavailable\tallocation_failure_count\ttotal_failed_allocation_bytes\tlargest_failed_allocation_bytes\ttrim_retry_count\tpool_usage_observation_count\tlast_failed_allocation_bytes\tlast_pool_reserved_current_bytes\tlast_pool_used_current_bytes\n"
            ),
            "{payload}"
        );
        assert!(
            payload.find("[device_memory_failure_reports]").unwrap() < payload.find("[gpu_execution]").unwrap(),
            "new diagnostics sections must follow the pre-existing TSV section order: {payload}"
        );
        assert!(
            payload.contains("[device_admission_slots]\ndevice_ordinal\tprocess_active_query_slots\n"),
            "{payload}"
        );
        assert!(
            payload.contains(
                "[device_telemetry]\nsampled_at_unix_ms\tdevice_ordinal\tcuda_status\tcuda_total_bytes\tcuda_free_bytes\tnvml_status\tnvml_total_bytes\tnvml_reserved_bytes\tnvml_used_bytes\tnvml_free_bytes\tgpu_utilization_pct\tmemory_utilization_pct\n"
            ),
            "{payload}"
        );
        assert!(
            payload.find("[gpu_execution_pools]").unwrap() < payload.find("[device_admission_slots]").unwrap()
                && payload.find("[device_admission_slots]").unwrap() < payload.find("[device_telemetry]").unwrap()
                && payload.find("[device_telemetry]").unwrap() < payload.find("[graph_cache]").unwrap(),
            "appended diagnostics sections must follow every pre-existing section: {payload}"
        );
        assert!(
            payload.contains(
                "[graph_cache]\ndevice_ordinal\tenabled\tentries\tresident_bytes\tin_use_bytes\thits\tmisses\tevictions\tbuilds_in_flight\tbuild_waits\n"
            ),
            "{payload}"
        );
        #[cfg(feature = "cugraph")]
        assert!(payload.contains("\n0\ttrue\t0\t0\t0\t0\t0\t0\t0\t0\n"), "{payload}");
        #[cfg(not(feature = "cugraph"))]
        assert!(payload.contains("\n0\tfalse\t0\t0\t0\t0\t0\t0\t0\t0\n"), "{payload}");

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

#[cfg(test)]
use std::sync::atomic::{AtomicBool, Ordering};
use std::{fmt::Write, sync::Arc};

#[cfg(test)]
use nexus_query_engine::runtime::{QueryAdmissionRequest, QueryHandle};

use super::{
    config::ServerConfig,
    error::{Result, ServerError},
};
use crate::{
    attempt::{AttemptOptions, NexusAttempt},
    backend::{NexusGpuBackend, NexusGpuQueuePolicy},
};

#[derive(Clone, Debug)]
enum QueryAdmissionBackend {
    QueryService {
        backend: Box<NexusGpuBackend>,
    },
    #[cfg(test)]
    TestOnly {
        closed: Arc<AtomicBool>,
    },
}

#[derive(Debug)]
pub struct QueryAdmission {
    backend: QueryAdmissionBackend,
}

impl QueryAdmission {
    pub(crate) fn from_backend(backend: &NexusGpuBackend) -> Self {
        Self { backend: QueryAdmissionBackend::QueryService { backend: Box::new(backend.clone()) } }
    }

    #[cfg(test)]
    pub(crate) fn from_test_backend(backend: &NexusGpuBackend) -> Self {
        Self::from_backend(backend)
    }

    #[cfg(test)]
    pub(crate) fn new_for_tests() -> Self {
        Self { backend: QueryAdmissionBackend::TestOnly { closed: Arc::new(AtomicBool::new(false)) } }
    }

    /// Begin the single execution attempt owned by a production Flight
    /// invocation. The test-only admission projection intentionally has no
    /// runtime backend, so legacy CPU-only unit fixtures return `None`; every
    /// server-built admission facade returns one concrete attempt.
    pub(crate) fn begin_attempt(&self, options: AttemptOptions) -> crate::error::Result<Option<NexusAttempt>> {
        match &self.backend {
            QueryAdmissionBackend::QueryService { backend } => backend.begin_attempt(options).map(Some),
            #[cfg(test)]
            QueryAdmissionBackend::TestOnly { .. } => Ok(None),
        }
    }

    /// Whether this facade owns a real backend and therefore must route Flight
    /// execution through `NexusAttempt` rather than the CPU-only test seam.
    pub(crate) fn has_attempt_backend(&self) -> bool {
        matches!(self.backend, QueryAdmissionBackend::QueryService { .. })
    }

    /// Close the admission gate. After close, any pending or future
    /// [`Self::acquire`] call returns a session error so graceful shutdown can
    /// drain in-flight work without admitting more.
    pub fn close(&self) {
        match &self.backend {
            QueryAdmissionBackend::QueryService { backend } => {
                backend.close();
            },
            #[cfg(test)]
            QueryAdmissionBackend::TestOnly { closed } => {
                closed.store(true, Ordering::Release);
            },
        }
    }

    fn close_and_wait_for_quiescence_blocking_until(&self, deadline: std::time::Instant) -> Result<()> {
        match &self.backend {
            QueryAdmissionBackend::QueryService { backend } => {
                backend.close_and_wait_until(deadline).map(|_snapshot| ()).map_err(|error| {
                    ServerError::session("query service shutdown cleanup did not quiesce", error)
                        .with_fact("shutdown_component", "query_admission")
                })
            },
            #[cfg(test)]
            QueryAdmissionBackend::TestOnly { closed } => {
                closed.store(true, Ordering::Release);
                Ok(())
            },
        }
    }

    pub(crate) async fn close_and_wait_for_quiescence_until(
        self: &Arc<Self>,
        deadline: std::time::Instant,
    ) -> Result<()> {
        let admission = Arc::clone(self);
        tokio::task::spawn_blocking(move || admission.close_and_wait_for_quiescence_blocking_until(deadline))
            .await
            .map_err(|error| {
                ServerError::shutdown_with_source(
                    "query_admission_cleanup_task_failed",
                    "query admission shutdown cleanup task failed",
                    error,
                )
            })?
    }

    #[cfg(test)]
    pub fn try_acquire(&self) -> Result<Option<QueryAdmissionPermit>> {
        match &self.backend {
            QueryAdmissionBackend::QueryService { backend } => backend
                .session_capability()
                .try_admit_query_with_request(QueryAdmissionRequest::new())
                .map(|handle| {
                    handle.map(|handle| QueryAdmissionPermit {
                        inner: QueryAdmissionPermitInner::QueryService(Arc::new(handle)),
                    })
                })
                .map_err(|error| ServerError::session("query service admission failed", error)),
            QueryAdmissionBackend::TestOnly { closed } => {
                if closed.load(Ordering::Acquire) {
                    return Err(ServerError::shutdown("query_admission_closed", "query admission is closed"));
                }
                Ok(Some(QueryAdmissionPermit { inner: QueryAdmissionPermitInner::TestOnly }))
            },
        }
    }

    pub fn log_query_service_state(&self, query_id: Option<&str>, event: &'static str) {
        let QueryAdmissionBackend::QueryService { backend } = &self.backend else {
            return;
        };
        let session = backend.session_capability();
        match session.admission_snapshot() {
            Ok(snapshot) => {
                tracing::info!(
                    query_id = query_id.unwrap_or(""),
                    event,
                    active_queries = snapshot.active_queries,
                    queued_queries = snapshot.queued_queries,
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
        match session.device_ledger_reports() {
            Ok(reports) => {
                for report in reports {
                    tracing::info!(
                        query_id = query_id.unwrap_or(""),
                        event,
                        device_ordinal = report.device_ordinal,
                        managed_capacity_bytes = report.managed_capacity_bytes,
                        live_grants_bytes = report.live_grants_bytes,
                        evictable_bytes = report.evictable_bytes,
                        available_bytes = report.available_bytes,
                        active_grants = report.grants.len(),
                        "query service device ledger report"
                    );
                }
            },
            Err(error) => tracing::warn!(
                query_id = query_id.unwrap_or(""),
                event,
                %error,
                "failed to read query service device ledger report"
            ),
        }
        for snapshot in session.device_admission_snapshots().unwrap_or_default() {
            tracing::info!(
                query_id = query_id.unwrap_or(""),
                event,
                device_ordinal = snapshot.device_ordinal,
                process_active_query_slots = snapshot.process_active_query_slots,
                "query service device admission snapshot"
            );
        }
        for snapshot in session.device_telemetry_snapshots().unwrap_or_default() {
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
        for report in session.device_memory_failure_reports().unwrap_or_default() {
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
        match &self.backend {
            QueryAdmissionBackend::QueryService { backend } => {
                let session = backend.session_capability();
                let queue_bounds = session
                    .queue_bounds()
                    .map_err(|error| ServerError::session("failed to read backend queue policy", error))?;
                let _ = writeln!(output, "# backend=query_service");
                let _ = writeln!(output, "# device_ordinals={}", payload_value(&join_usize(backend.device_ordinals())));
                let _ = writeln!(output, "# device_profile_count={}", backend.device_profiles().len());
                for profile in backend.device_profiles() {
                    let ordinal = profile.device_ordinal();
                    let _ =
                        writeln!(output, "# device_{ordinal}_max_active_attempts={}", profile.max_active_attempts());
                    let _ = writeln!(
                        output,
                        "# device_{ordinal}_attempt_floor_bytes={}",
                        profile.attempt_floor_bytes().map_or_else(|| "none".to_string(), |value| value.to_string())
                    );
                }
                let _ = writeln!(output, "# max_queued_attempts={}", queue_bounds.max_waiters);
                let _ = writeln!(output, "# max_queue_wait_ms={}", queue_bounds.max_wait.as_millis());

                let snapshot = session
                    .admission_snapshot()
                    .map_err(|error| ServerError::session("failed to read query service admission snapshot", error))?;
                output.push_str("\n[admission]\n");
                output.push_str("active_queries\tqueued_queries\n");
                let _ = writeln!(output, "{}\t{}", snapshot.active_queries, snapshot.queued_queries);

                let reports = session.device_ledger_reports().map_err(|error| {
                    ServerError::session("failed to read query service device ledger reports", error)
                })?;
                output.push_str("\n[device_ledger_reports]\n");
                output.push_str(
                    "device_ordinal\tmanaged_capacity_bytes\tlive_grants_bytes\tavailable_bytes\tactive_grants\tevictable_bytes\n",
                );
                for report in &reports {
                    let _ = writeln!(
                        output,
                        "{}\t{}\t{}\t{}\t{}\t{}",
                        report.device_ordinal,
                        report.managed_capacity_bytes,
                        report.live_grants_bytes,
                        report.available_bytes,
                        report.grants.len(),
                        report.evictable_bytes
                    );
                }

                output.push_str("\n[device_ledger_grants]\n");
                output.push_str("device_ordinal\tgrant_id\tadmission_request_id\tgrant_bytes\tlive_reserved_bytes\n");
                for report in &reports {
                    for grant in &report.grants {
                        let _ = writeln!(
                            output,
                            "{}\t{}\t{}\t{}\t{}",
                            report.device_ordinal,
                            grant.grant_id,
                            grant.admission_request_id,
                            grant.grant_bytes,
                            grant.live_reserved_bytes
                        );
                    }
                }

                output.push_str("\n[device_memory_failure_reports]\n");
                output.push_str(
                    "device_ordinal\tcallback_installed\tcallback_install_unavailable\tallocation_failure_count\ttotal_failed_allocation_bytes\tlargest_failed_allocation_bytes\ttrim_retry_count\tpool_usage_observation_count\tlast_failed_allocation_bytes\tlast_pool_reserved_current_bytes\tlast_pool_used_current_bytes\n",
                );
                for report in session
                    .device_memory_failure_reports()
                    .map_err(|error| ServerError::session("failed to read device memory failure reports", error))?
                {
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

                let query_execution = session
                    .gpu_execution_query_report()
                    .map_err(|error| ServerError::session("failed to read GPU execution report", error))?;
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
                for report in session
                    .gpu_execution_pool_reports()
                    .map_err(|error| ServerError::session("failed to read GPU execution pool reports", error))?
                {
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
                for snapshot in session
                    .device_admission_snapshots()
                    .map_err(|error| ServerError::session("failed to read device admission snapshots", error))?
                {
                    let _ = writeln!(output, "{}\t{}", snapshot.device_ordinal, snapshot.process_active_query_slots);
                }

                output.push_str("\n[device_telemetry]\n");
                output.push_str(
                    "sampled_at_unix_ms\tdevice_ordinal\tcuda_status\tcuda_total_bytes\tcuda_free_bytes\tnvml_status\tnvml_total_bytes\tnvml_reserved_bytes\tnvml_used_bytes\tnvml_free_bytes\tgpu_utilization_pct\tmemory_utilization_pct\n",
                );
                for snapshot in session
                    .device_telemetry_snapshots()
                    .map_err(|error| ServerError::session("failed to read device telemetry snapshots", error))?
                {
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
            },
            #[cfg(test)]
            QueryAdmissionBackend::TestOnly { .. } => {
                let _ = writeln!(output, "# backend=test_only");
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
    TestOnly,
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
    /// Builds the server's mandatory bounded query-service backend.
    pub fn from_server_config(config: &ServerConfig) -> Result<Option<Self>> {
        config.validate_gpu_device_profiles()?;
        Self::from_optimizer_config_for_admission(
            config.native_optimizer_config.clone(),
            config.gpu_device_profiles.clone(),
            config.gpu_memory_ownership.clone(),
            server_queue_policy(config),
        )
        .map_err(|error| ServerError::session("failed to initialize query service admission", error))
    }
}

#[cfg(test)]
fn server_test_controller_projection_from_config(
    config: &ServerConfig,
) -> Result<Option<crate::backend::TestBackendControllerProjection>> {
    config.validate_gpu_device_profiles()?;
    crate::backend::test_controller_projection_from_optimizer_config(
        config.native_optimizer_config.clone(),
        config.gpu_device_profiles.clone(),
        config.gpu_memory_ownership.clone(),
        server_queue_policy(config),
    )
    .map_err(|error| ServerError::session("failed to project query service admission policy", error))
}

pub(crate) fn server_queue_policy(config: &ServerConfig) -> NexusGpuQueuePolicy {
    config.queue_policy.clone()
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

    fn server_config_with_test_device_profile(
        mut env: std::collections::BTreeMap<&'static str, String>,
        memory_limit_bytes: Option<u64>,
    ) -> crate::server::Result<ServerConfig> {
        let config_file = tempfile::NamedTempFile::new().expect("create server config fixture");
        let profile = match memory_limit_bytes {
            Some(memory_limit_bytes) => format!(
                "[[admission.device_profiles]]\ndevice_ordinal = 0\nmemory_limit_bytes = {memory_limit_bytes}\n"
            ),
            None => "[[admission.device_profiles]]\ndevice_ordinal = 0\n".to_string(),
        };
        std::fs::write(config_file.path(), profile).expect("write server config fixture");
        env.insert(
            super::super::config::NEXUS_SERVER_CONFIG_FILE_ENV,
            config_file.path().to_string_lossy().into_owned(),
        );
        ServerConfig::from_env_get(|name| env.get(name).cloned())
    }

    fn declare_test_gpu_ownership(mut config: ServerConfig) -> ServerConfig {
        config.gpu_memory_ownership = config
            .gpu_devices
            .iter()
            .copied()
            .map(|ordinal| (ordinal, crate::backend::GpuMemoryOwnership::WholeDeviceExclusive))
            .collect();
        config
    }

    #[test]
    fn server_backend_partitions_budget_and_sizes_stream_pool() {
        let env = std::collections::BTreeMap::from([
            (super::super::config::NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV, "3072".to_string()),
            (super::super::config::NEXUS_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, "1024".to_string()),
            (super::super::config::NEXUS_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV, "1".to_string()),
            (super::super::config::NEXUS_SERVER_NATIVE_MAX_SOURCE_READS_IN_FLIGHT_ENV, "2".to_string()),
        ]);
        let config = declare_test_gpu_ownership(server_config_with_test_device_profile(env, Some(3072)).unwrap());

        let backend = server_test_controller_projection_from_config(&config)
            .unwrap()
            .expect("bounded native config should project QueryService admission");
        let controller = backend.controller_config().unwrap();

        let profile = &backend.device_profiles()[0];
        assert_eq!(profile.max_active_attempts(), 1);
        assert_eq!(profile.device_ordinal(), 0);
        assert_eq!(profile.memory_limit(), crate::backend::GpuDeviceMemoryLimit::Explicit { limit_bytes: 3072 });
        assert_eq!(controller.source_scheduler_policy.max_source_reads_in_flight(), 2);
        assert_eq!(backend.controller_queue_bounds().unwrap().max_waiters, 64);
        assert_eq!(backend.controller_queue_bounds().unwrap().max_wait, std::time::Duration::from_secs(30));
        // Floor defaults to managed / max(8, A_d) = 3072 / 8.
        assert_eq!(backend.configured_floor_bytes().unwrap(), 384);

        let permit = backend.admit().expect("first query should admit");
        // A hint-less grant takes the full candidate maximum.
        assert_eq!(permit.device_budget_bytes(), 3072);
    }

    #[test]
    fn server_config_rejects_removed_p2_cache_profile_env_before_backend_projection() {
        let env = std::collections::BTreeMap::from([
            (super::super::config::NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV, "3072".to_string()),
            (super::super::config::NEXUS_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, "1024".to_string()),
            (super::super::config::NEXUS_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV, "1".to_string()),
            ("NEXUS_SERVER_NATIVE_SOURCE_CACHE_MAX_RETAINED_BYTES", "2048".to_string()),
        ]);
        let error = server_config_with_test_device_profile(env, None)
            .expect_err("P2 must reject cache profile configuration before backend construction");

        assert_eq!(error.code(), crate::error::ErrorCode::InvalidRequest);
        assert_eq!(error.operation(), "datafusion_nexus.server.config");
        assert_eq!(error.fact_value("config_field"), Some("NEXUS_SERVER_NATIVE_SOURCE_CACHE_MAX_RETAINED_BYTES"));
        assert!(error.message().contains("removed by the P2 resource-only admission cutover"));
    }

    #[test]
    fn server_backend_honors_queue_wait_bound_env() {
        let env = std::collections::BTreeMap::from([
            (super::super::config::NEXUS_SERVER_ADMISSION_MAX_QUEUE_WAIT_MS_ENV, "250".to_string()),
            (super::super::config::NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV, "3072".to_string()),
            (super::super::config::NEXUS_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, "1024".to_string()),
            (super::super::config::NEXUS_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV, "1".to_string()),
        ]);
        let config = declare_test_gpu_ownership(server_config_with_test_device_profile(env, Some(3072)).unwrap());

        let backend = server_test_controller_projection_from_config(&config)
            .unwrap()
            .expect("bounded native config should project QueryService admission");

        assert_eq!(backend.controller_queue_bounds().unwrap().max_wait, std::time::Duration::from_millis(250));
    }

    #[test]
    fn server_backend_preserves_multi_gpu_device_subset() {
        let env = std::collections::BTreeMap::from([
            (super::super::config::NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV, "2048".to_string()),
            (super::super::config::NEXUS_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, "1024".to_string()),
            (super::super::config::NEXUS_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV, "1".to_string()),
        ]);
        let mut config = server_config_with_test_device_profile(env, Some(2048)).unwrap();
        config.gpu_devices = vec![0, 1];
        config.gpu_device_profiles = vec![
            crate::backend::NexusGpuDeviceProfile::new(0)
                .with_memory_limit(crate::backend::GpuDeviceMemoryLimit::Explicit { limit_bytes: 2048 }),
            crate::backend::NexusGpuDeviceProfile::new(1)
                .with_memory_limit(crate::backend::GpuDeviceMemoryLimit::Explicit { limit_bytes: 4096 }),
        ];
        let config = declare_test_gpu_ownership(config);

        let backend = server_test_controller_projection_from_config(&config)
            .unwrap()
            .expect("bounded native config should project QueryService admission");

        assert_eq!(backend.device_ordinals(), &[0, 1]);
        assert_eq!(
            backend.device_profiles().iter().map(|profile| profile.device_ordinal()).collect::<Vec<_>>(),
            vec![0, 1]
        );
        let permit = backend.admit().expect("first query should admit");
        assert_eq!(permit.device_ordinal(), 0);
        assert_eq!(permit.device_budget_bytes(), 2048);
    }

    #[test]
    fn query_service_admission_diagnostics_payload_reports_snapshot_and_governors() {
        let env = std::collections::BTreeMap::from([
            (super::super::config::NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV, "1024".to_string()),
            (super::super::config::NEXUS_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, "512".to_string()),
            (super::super::config::NEXUS_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV, "1".to_string()),
        ]);
        let mut config = declare_test_gpu_ownership(server_config_with_test_device_profile(env, Some(1024)).unwrap());
        config.gpu_device_profiles[0] = config.gpu_device_profiles[0].clone().with_attempt_floor_bytes(Some(256));
        let backend = NexusGpuBackend::from_server_config(&config)
            .unwrap()
            .expect("bounded native config should build a runtime backend");
        let admission = QueryAdmission::from_backend(&backend);
        let permit = admission.try_acquire().unwrap().expect("first query should admit");

        let payload = admission.diagnostics_payload().unwrap();

        assert!(payload.contains("# datafusion_nexus_admission_stats=1\n"), "{payload}");
        assert!(payload.contains("# backend=query_service\n"), "{payload}");
        assert!(payload.contains("# device_profile_count=1\n"), "{payload}");
        assert!(payload.contains("# device_0_max_active_attempts=1\n"), "{payload}");
        assert!(payload.contains("# device_0_attempt_floor_bytes=256\n"), "{payload}");
        assert!(payload.contains("[admission]\nactive_queries\tqueued_queries\n1\t0\n"), "{payload}");
        assert!(payload.contains("[gpu_execution]\nactive_query_jobs\tpeak_active_query_jobs\n0\t0\n"), "{payload}");
        assert!(
            payload.contains(
                "[gpu_execution_pools]\ndevice_ordinal\tworker_count\tactive_jobs\tpeak_active_jobs\tqueued_jobs\tpeak_queued_jobs\tactive_query_jobs\tpeak_active_query_jobs\n"
            ),
            "{payload}"
        );
        assert!(
            payload.contains(
                "[device_ledger_reports]\ndevice_ordinal\tmanaged_capacity_bytes\tlive_grants_bytes\tavailable_bytes\tactive_grants\tevictable_bytes\n0\t1024\t1024\t0\t1\t0\n"
            ),
            "{payload}"
        );
        assert!(payload.contains("[device_ledger_grants]\n"), "{payload}");
        assert!(payload.contains("\t1\t1024\t"), "{payload}");
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
                && payload.find("[device_admission_slots]").unwrap() < payload.find("[device_telemetry]").unwrap(),
            "appended diagnostics sections must follow every pre-existing section: {payload}"
        );

        drop(permit);
    }

    #[test]
    fn semaphore_admission_diagnostics_payload_reports_permit_counts() {
        let admission = QueryAdmission::new_for_tests();
        let _permit = admission.try_acquire().unwrap().expect("test-only admission should construct a permit");

        let payload = admission.diagnostics_payload().unwrap();

        assert!(payload.contains("# datafusion_nexus_admission_stats=1\n"), "{payload}");
        assert!(payload.contains("# backend=test_only\n"), "{payload}");
    }
}

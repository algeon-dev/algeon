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
    /// runtime backend, so CPU-only unit fixtures return `None`; every
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
        let backend = match &self.backend {
            QueryAdmissionBackend::QueryService { backend } => backend,
            #[cfg(test)]
            QueryAdmissionBackend::TestOnly { .. } => return,
        };
        let snapshot = backend.backend_metrics_snapshot();
        let configured_devices = snapshot.devices.iter().flatten().count();
        let active_attempts =
            snapshot.devices.iter().flatten().map(|device| device.admission.active_attempts).sum::<u64>();

        tracing::info!(
            query_id = query_id.unwrap_or(""),
            event,
            backend_generation = snapshot.backend_generation,
            metrics_generation = snapshot.generation,
            configured_devices,
            active_attempts,
            "query service backend metrics snapshot"
        );
        for device in snapshot.devices.into_iter().flatten() {
            tracing::info!(
                query_id = query_id.unwrap_or(""),
                event,
                device_ordinal = device.device_ordinal,
                ledger_sequence = device.ledger_sequence,
                active_attempts = device.admission.active_attempts,
                live_grant_bytes = device.admission.live_grant_bytes,
                live_reserved_bytes = device.admission.live_reserved_bytes,
                max_live_reserved_bytes = device.admission.max_live_reserved_bytes,
                cache_evictable_resident_bytes = device.admission.cache_evictable_resident_bytes,
                cache_pinned_in_use_bytes = device.admission.cache_pinned_in_use_bytes,
                admission_health = ?device.admission.health,
                telemetry_refresh = ?device.telemetry.last_refresh,
                allocation_failure_count = device.telemetry.allocation_failure_count,
                allocation_failure_bytes = device.telemetry.allocation_failure_bytes,
                largest_allocation_failure_bytes = device.telemetry.largest_allocation_failure_bytes,
                "query service device backend metrics snapshot"
            );
        }
    }

    pub fn diagnostics_payload(&self) -> Result<String> {
        let mut output = String::new();
        let _ = writeln!(output, "# datafusion_nexus_admission_stats=1");
        match &self.backend {
            QueryAdmissionBackend::QueryService { backend } => {
                let queue_bounds = backend
                    .controller_queue_bounds()
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
                // This is the one dynamic observation for the whole payload.
                // It cannot contain grant ids, query ids, CUDA/NVML payloads,
                // or execution-pool state, because none of those are part of
                // the bounded backend metrics contract.
                let snapshot = backend.backend_metrics_snapshot();
                let _ = writeln!(output, "# backend_metrics_backend_generation={}", snapshot.backend_generation);
                let _ = writeln!(output, "# backend_metrics_generation={}", snapshot.generation);

                output.push_str("\n[backend_metrics]\n");
                output.push_str("backend_generation\tgeneration\tconfigured_devices\n");
                let _ = writeln!(
                    output,
                    "{}\t{}\t{}",
                    snapshot.backend_generation,
                    snapshot.generation,
                    snapshot.devices.iter().flatten().count()
                );

                output.push_str("\n[backend_device_admission]\n");
                output.push_str(
                    "device_ordinal\tledger_sequence\tactive_attempts\tlive_grant_bytes\tlive_reserved_bytes\tmax_live_reserved_bytes\tcache_evictable_resident_bytes\tcache_pinned_in_use_bytes\thealth\n",
                );
                for device in snapshot.devices.iter().flatten() {
                    let _ = writeln!(
                        output,
                        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:?}",
                        device.device_ordinal,
                        device.ledger_sequence,
                        device.admission.active_attempts,
                        device.admission.live_grant_bytes,
                        device.admission.live_reserved_bytes,
                        device.admission.max_live_reserved_bytes,
                        device.admission.cache_evictable_resident_bytes,
                        device.admission.cache_pinned_in_use_bytes,
                        device.admission.health,
                    );
                }

                output.push_str("\n[backend_device_telemetry]\n");
                output.push_str(
                    "device_ordinal\trefresh_source\tsampled_at_unix_ms\tage_ms\tavailability\tallocation_failure_count\tallocation_failure_bytes\tlargest_allocation_failure_bytes\n",
                );
                for device in snapshot.devices.iter().flatten() {
                    match device.telemetry.last_refresh {
                        Some(refresh) => {
                            let _ = writeln!(
                                output,
                                "{}\t{:?}\t{}\t{}\t{:?}\t{}\t{}\t{}",
                                device.device_ordinal,
                                refresh.source,
                                refresh.sampled_at_unix_ms,
                                refresh.age_ms,
                                refresh.availability,
                                device.telemetry.allocation_failure_count,
                                device.telemetry.allocation_failure_bytes,
                                device.telemetry.largest_allocation_failure_bytes,
                            );
                        },
                        None => {
                            let _ = writeln!(
                                output,
                                "{}\tnone\tnone\tnone\tnone\t{}\t{}\t{}",
                                device.device_ordinal,
                                device.telemetry.allocation_failure_count,
                                device.telemetry.allocation_failure_bytes,
                                device.telemetry.largest_allocation_failure_bytes,
                            );
                        },
                    }
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
fn server_test_validation_projection_from_config(
    config: &ServerConfig,
) -> Result<Option<crate::backend::TestBackendValidationProjection>> {
    config.validate_gpu_device_profiles()?;
    crate::backend::test_validation_projection_from_optimizer_config(
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

        let backend = server_test_validation_projection_from_config(&config)
            .unwrap()
            .expect("bounded native config should project QueryService admission");
        let controller = backend.query_execution_config().unwrap();

        let profile = &backend.device_profiles()[0];
        assert_eq!(profile.max_active_attempts(), 1);
        assert_eq!(profile.device_ordinal(), 0);
        assert_eq!(profile.memory_limit(), crate::backend::GpuDeviceMemoryLimit::Explicit { limit_bytes: 3072 });
        assert_eq!(
            profile.engine_config(crate::backend::GpuMemoryOwnership::WholeDeviceExclusive),
            nexus_query_engine::runtime::GpuBackendDeviceConfig::new(
                0,
                crate::backend::GpuMemoryOwnership::WholeDeviceExclusive,
            )
            .with_memory_limit(crate::backend::GpuDeviceMemoryLimit::Explicit { limit_bytes: 3072 }),
            "server profile defaults must be an engine construction projection"
        );
        assert_eq!(controller.source_scheduler_policy.max_source_reads_in_flight(), 2);
        assert_eq!(backend.queue_bounds().unwrap(), nexus_query_engine::runtime::QueueBounds::default());
        // The validation projection intentionally stops before runtime floor
        // resolution or grant allocation.
        assert_eq!(profile.attempt_floor_bytes(), None);
    }

    #[test]
    fn server_config_rejects_removed_legacy_cache_profile_env_before_backend_projection() {
        let env = std::collections::BTreeMap::from([
            (super::super::config::NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV, "3072".to_string()),
            (super::super::config::NEXUS_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, "1024".to_string()),
            (super::super::config::NEXUS_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV, "1".to_string()),
            ("NEXUS_SERVER_NATIVE_SOURCE_CACHE_MAX_RETAINED_BYTES", "2048".to_string()),
        ]);
        let error = server_config_with_test_device_profile(env, None)
            .expect_err("removed cache profile configuration must fail before backend construction");

        assert_eq!(error.code(), crate::error::ErrorCode::InvalidRequest);
        assert_eq!(error.operation(), "datafusion_nexus.server.config");
        assert_eq!(error.fact_value("config_field"), Some("NEXUS_SERVER_NATIVE_SOURCE_CACHE_MAX_RETAINED_BYTES"));
        assert!(error.message().contains("removed by the resource-only admission contract"));
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

        let backend = server_test_validation_projection_from_config(&config)
            .unwrap()
            .expect("bounded native config should project QueryService admission");

        assert_eq!(backend.queue_bounds().unwrap().max_wait, std::time::Duration::from_millis(250));
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

        let backend = server_test_validation_projection_from_config(&config)
            .unwrap()
            .expect("bounded native config should project QueryService admission");

        assert_eq!(backend.device_ordinals(), &[0, 1]);
        assert_eq!(
            backend.device_profiles().iter().map(|profile| profile.device_ordinal()).collect::<Vec<_>>(),
            vec![0, 1]
        );
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

        let snapshot = backend.backend_metrics_snapshot();
        let device = snapshot.device(0).expect("configured device must have one bounded metrics slot");
        assert_eq!(device.admission.active_attempts, 1);
        assert_eq!(device.admission.live_grant_bytes, 1024);
        assert_eq!(device.admission.live_reserved_bytes, 0);
        // The default zero-cap profile creates no cache allocation domain, so
        // its ledger-backed cache projections remain byte-for-byte zero.
        assert_eq!(device.admission.cache_evictable_resident_bytes, 0);
        assert_eq!(device.admission.cache_pinned_in_use_bytes, 0);

        let payload = admission.diagnostics_payload().unwrap();

        assert!(payload.contains("# datafusion_nexus_admission_stats=1\n"), "{payload}");
        assert!(payload.contains("# backend=query_service\n"), "{payload}");
        assert!(payload.contains("# device_profile_count=1\n"), "{payload}");
        assert!(payload.contains("# device_0_max_active_attempts=1\n"), "{payload}");
        assert!(payload.contains("# device_0_attempt_floor_bytes=256\n"), "{payload}");
        assert!(
            payload.contains("[backend_metrics]\nbackend_generation\tgeneration\tconfigured_devices\n"),
            "{payload}"
        );
        assert!(
            payload.contains(
                "[backend_device_admission]\ndevice_ordinal\tledger_sequence\tactive_attempts\tlive_grant_bytes\tlive_reserved_bytes\tmax_live_reserved_bytes\tcache_evictable_resident_bytes\tcache_pinned_in_use_bytes\thealth\n0\t"
            ),
            "{payload}"
        );
        assert!(payload.contains("\t1\t1024\t0\t0\t0\t0\tOpen\n"), "{payload}");
        assert!(
            payload.contains(
                "[backend_device_telemetry]\ndevice_ordinal\trefresh_source\tsampled_at_unix_ms\tage_ms\tavailability\tallocation_failure_count\tallocation_failure_bytes\tlargest_allocation_failure_bytes\n0\tnone\tnone\tnone\tnone\t0\t0\t0\n"
            ),
            "{payload}"
        );
        assert!(
            !payload.contains("[device_ledger_grants]") && !payload.contains("[gpu_execution]"),
            "diagnostics must not combine the backend snapshot with separate service reads: {payload}"
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

#[cfg(test)]
use std::sync::atomic::{AtomicBool, Ordering};
use std::{fmt::Write, sync::Arc};

use tokio::task::spawn_blocking;

use crate::{
    error::Result,
    server::{
        attempt::{AlgeonAttempt, AttemptOptions},
        backend::{AlgeonGpuBackend, AlgeonGpuDeviceConfig, AlgeonGpuQueuePolicy},
        config::ServerConfig,
        error::{Result as ServerResult, ServerError},
    },
};

#[derive(Clone, Debug)]
enum QueryAdmissionBackend {
    QueryService {
        backend: Box<AlgeonGpuBackend>,
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
    pub(crate) fn from_backend(backend: &AlgeonGpuBackend) -> Self {
        Self { backend: QueryAdmissionBackend::QueryService { backend: Box::new(backend.clone()) } }
    }

    #[cfg(test)]
    pub(crate) fn new_for_tests() -> Self {
        Self { backend: QueryAdmissionBackend::TestOnly { closed: Arc::new(AtomicBool::new(false)) } }
    }

    /// Begin the single execution attempt owned by a production Flight
    /// invocation. The test-only admission projection intentionally has no
    /// runtime backend, so CPU-only unit fixtures return `None`; every
    /// server-built admission facade returns one concrete attempt.
    pub(crate) fn begin_attempt(&self, options: AttemptOptions) -> Result<Option<AlgeonAttempt>> {
        match &self.backend {
            QueryAdmissionBackend::QueryService { backend } => backend.begin_attempt(options).map(Some),
            #[cfg(test)]
            QueryAdmissionBackend::TestOnly { .. } => Ok(None),
        }
    }

    /// Whether this facade owns a real backend and therefore must route Flight
    /// execution through `AlgeonAttempt` rather than the CPU-only test seam.
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

    fn close_and_wait_for_quiescence_blocking_until(&self, deadline: std::time::Instant) -> ServerResult<()> {
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
    ) -> ServerResult<()> {
        let admission = Arc::clone(self);
        spawn_blocking(move || admission.close_and_wait_for_quiescence_blocking_until(deadline)).await.map_err(
            |error| {
                ServerError::shutdown_with_source(
                    "query_admission_cleanup_task_failed",
                    "query admission shutdown cleanup task failed",
                    error,
                )
            },
        )?
    }

    #[cfg(test)]
    pub fn try_acquire(&self) -> ServerResult<Option<QueryAdmissionPermit>> {
        let QueryAdmissionBackend::TestOnly { closed } = &self.backend else {
            panic!("test admission permits only exist for the test-only backend");
        };
        if closed.load(Ordering::Acquire) {
            return Err(ServerError::shutdown("query_admission_closed", "query admission is closed"));
        }
        Ok(Some(QueryAdmissionPermit))
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
                resident_owner_bytes = device.admission.resident_owner_bytes,
                max_resident_owner_bytes = device.admission.max_resident_owner_bytes,
                cache_evictable_resident_bytes = device.admission.cache_evictable_resident_bytes,
                cache_pinned_in_use_bytes = device.admission.cache_pinned_in_use_bytes,
                admission_health = ?device.admission.health,
                "query service device backend metrics snapshot"
            );
        }
    }

    pub fn diagnostics_payload(&self) -> ServerResult<String> {
        let mut output = String::new();
        match &self.backend {
            QueryAdmissionBackend::QueryService { backend } => {
                let queue_policy = backend.queue_policy();
                let _ = writeln!(output, "# backend=query_service");
                let ordinals = backend.devices().iter().map(AlgeonGpuDeviceConfig::device_ordinal).collect::<Vec<_>>();
                let _ = writeln!(output, "# device_ordinals={}", payload_value(&join_usize(&ordinals)));
                let _ = writeln!(output, "# device_profile_count={}", backend.devices().len());
                for device in backend.devices() {
                    let profile = device.profile();
                    let ordinal = device.device_ordinal();
                    let _ =
                        writeln!(output, "# device_{ordinal}_max_active_attempts={}", profile.max_active_attempts());
                    let _ = writeln!(
                        output,
                        "# device_{ordinal}_default_query_limit_bytes={}",
                        profile
                            .default_query_limit_bytes()
                            .map_or_else(|| "none".to_string(), |value| value.to_string())
                    );
                }
                let _ = writeln!(output, "# max_queued_attempts={}", queue_policy.max_queued_attempts());
                let _ = writeln!(output, "# max_queue_wait_ms={}", queue_policy.max_queue_wait().as_millis());
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
                    "device_ordinal\tledger_sequence\tactive_attempts\tlive_grant_bytes\tresident_owner_bytes\tmax_resident_owner_bytes\tcache_evictable_resident_bytes\tcache_pinned_in_use_bytes\thealth\n",
                );
                for device in snapshot.devices.iter().flatten() {
                    let _ = writeln!(
                        output,
                        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:?}",
                        device.device_ordinal,
                        device.ledger_sequence,
                        device.admission.active_attempts,
                        device.admission.live_grant_bytes,
                        device.admission.resident_owner_bytes,
                        device.admission.max_resident_owner_bytes,
                        device.admission.cache_evictable_resident_bytes,
                        device.admission.cache_pinned_in_use_bytes,
                        device.admission.health,
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

#[cfg(test)]
#[derive(Debug)]
pub struct QueryAdmissionPermit;

pub(crate) fn server_queue_policy(config: &ServerConfig) -> AlgeonGpuQueuePolicy {
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
    use arrow::array::ArrayRef;
    use datafusion::{
        execution::SessionStateBuilder,
        prelude::{ParquetReadOptions, SessionContext},
    };
    use futures::StreamExt;
    use parquet::arrow::ArrowWriter;

    use super::*;
    use crate::server::{
        adapter_error::ErrorCode as ServerErrorCode,
        backend::{AlgeonGpuDeviceProfile, GpuDeviceMemoryLimit, GpuMemoryOwnership},
        backend_from_server_config,
        config::{
            ALGEON_SERVER_ADMISSION_MAX_QUEUE_WAIT_MS_ENV, ALGEON_SERVER_CONFIG_FILE_ENV,
            ALGEON_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV, ALGEON_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV,
            ALGEON_SERVER_NATIVE_MAX_SOURCE_READS_IN_FLIGHT_ENV,
        },
    };

    fn write_holder_parquet(path: &std::path::Path) {
        use arrow::{
            array::Int64Array,
            datatypes::{DataType, Field, Schema},
            record_batch::RecordBatch,
        };

        let schema = std::sync::Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
        let batch = RecordBatch::try_new(
            std::sync::Arc::clone(&schema),
            vec![std::sync::Arc::new(Int64Array::from(vec![1, 2])) as ArrayRef],
        )
        .expect("holder fixture batch");
        let file = std::fs::File::create(path).expect("holder fixture file");
        let mut writer = ArrowWriter::try_new(file, schema, None).expect("holder writer");
        writer.write(&batch).expect("write holder fixture");
        writer.close().expect("close holder fixture");
    }

    fn server_config_with_test_device_profile(
        mut env: std::collections::BTreeMap<&'static str, String>,
        memory_limit_bytes: Option<u64>,
    ) -> ServerResult<ServerConfig> {
        let config_file = tempfile::NamedTempFile::new().expect("create server config fixture");
        let profile = match memory_limit_bytes {
            Some(memory_limit_bytes) => format!(
                "[[admission.device_profiles]]\ndevice_ordinal = 0\nmemory_limit_bytes = {memory_limit_bytes}\nbackend_reserve_bytes = 0\n"
            ),
            None => "[[admission.device_profiles]]\ndevice_ordinal = 0\n".to_string(),
        };
        std::fs::write(config_file.path(), profile).expect("write server config fixture");
        env.insert(ALGEON_SERVER_CONFIG_FILE_ENV, config_file.path().to_string_lossy().into_owned());
        ServerConfig::from_env_get(|name| env.get(name).cloned())
    }

    fn declare_test_gpu_ownership(mut config: ServerConfig) -> ServerConfig {
        config.gpu_memory_ownership = config
            .gpu_devices
            .iter()
            .copied()
            .map(|ordinal| (ordinal, GpuMemoryOwnership::WholeDeviceExclusive))
            .collect();
        config
    }

    #[test]
    fn server_backend_partitions_budget_and_sizes_stream_pool() {
        let env = std::collections::BTreeMap::from([
            (ALGEON_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, "1024".to_string()),
            (ALGEON_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV, "1".to_string()),
            (ALGEON_SERVER_NATIVE_MAX_SOURCE_READS_IN_FLIGHT_ENV, "2".to_string()),
        ]);
        let config = declare_test_gpu_ownership(server_config_with_test_device_profile(env, Some(3072)).unwrap());

        let devices = config.resolve_gpu_devices().expect("server device mapping must resolve");
        let profile = devices[0].profile();
        assert_eq!(profile.max_active_attempts(), 2);
        assert_eq!(profile.device_ordinal(), 0);
        assert_eq!(profile.memory_limit(), GpuDeviceMemoryLimit::Explicit { limit_bytes: 3072 });
        assert_eq!(profile.default_query_limit_bytes(), None);
        assert_eq!(
            config
                .native_optimizer_config
                .native_execution_config()
                .source_scheduler_policy()
                .max_source_reads_in_flight(),
            2
        );
        assert_eq!(server_queue_policy(&config), AlgeonGpuQueuePolicy::default());
    }

    #[test]
    fn server_config_rejects_removed_legacy_cache_profile_env_before_backend_projection() {
        let env = std::collections::BTreeMap::from([
            (ALGEON_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, "1024".to_string()),
            (ALGEON_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV, "1".to_string()),
            ("ALGEON_SERVER_NATIVE_SOURCE_CACHE_MAX_RETAINED_BYTES", "2048".to_string()),
        ]);
        let error = server_config_with_test_device_profile(env, None)
            .expect_err("removed cache profile configuration must fail before backend construction");

        assert_eq!(error.code(), ServerErrorCode::InvalidConfiguration);
        assert_eq!(error.operation(), "algeon_datafusion.server.config");
        assert_eq!(error.fact_value("config_field"), Some("ALGEON_SERVER_NATIVE_SOURCE_CACHE_MAX_RETAINED_BYTES"));
        assert!(error.message().contains("removed by the resource-only admission contract"));
    }

    #[test]
    fn server_backend_honors_queue_wait_bound_env() {
        let env = std::collections::BTreeMap::from([
            (ALGEON_SERVER_ADMISSION_MAX_QUEUE_WAIT_MS_ENV, "250".to_string()),
            (ALGEON_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, "1024".to_string()),
            (ALGEON_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV, "1".to_string()),
        ]);
        let config = declare_test_gpu_ownership(server_config_with_test_device_profile(env, Some(3072)).unwrap());

        assert_eq!(server_queue_policy(&config).max_queue_wait(), std::time::Duration::from_millis(250));
    }

    #[test]
    fn server_backend_preserves_multi_gpu_device_subset() {
        let env = std::collections::BTreeMap::from([
            (ALGEON_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, "1024".to_string()),
            (ALGEON_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV, "1".to_string()),
        ]);
        let mut config = server_config_with_test_device_profile(env, Some(2048)).unwrap();
        config.gpu_devices = vec![0, 1];
        config.gpu_device_profiles = vec![
            AlgeonGpuDeviceProfile::new(0).with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 2048 }),
            AlgeonGpuDeviceProfile::new(1).with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 4096 }),
        ];
        let config = declare_test_gpu_ownership(config);

        let devices = config.resolve_gpu_devices().expect("server device mapping must resolve");
        assert_eq!(devices.iter().map(|device| device.device_ordinal()).collect::<Vec<_>>(), vec![0, 1]);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn query_service_admission_diagnostics_payload_reports_snapshot_and_governors() {
        const HOLDER_DEVICE_CAP_BYTES: u64 = 1024 * 1024 * 1024;
        const HOLDER_QUERY_CAP_BYTES: u64 = 512 * 1024 * 1024;

        let env = std::collections::BTreeMap::from([
            (ALGEON_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, "512".to_string()),
            (ALGEON_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV, "1".to_string()),
        ]);
        let mut config = declare_test_gpu_ownership(
            server_config_with_test_device_profile(env, Some(HOLDER_DEVICE_CAP_BYTES)).unwrap(),
        );
        config.gpu_device_profiles[0] =
            config.gpu_device_profiles[0].clone().with_default_query_limit_bytes(HOLDER_QUERY_CAP_BYTES);
        let backend = backend_from_server_config(&config).unwrap();
        let admission = QueryAdmission::from_backend(&backend);

        // Hold one admitted execution through the canonical attempt
        // lifecycle. The stream is polled once to admit, then left
        // suspended: the slot stays occupied until the stream drops.
        let holder_dir = tempfile::TempDir::new().expect("holder parquet fixture");
        let holder_path = holder_dir.path().join("holder.parquet");
        write_holder_parquet(&holder_path);
        let holder_state = backend
            .install_on(SessionStateBuilder::new_with_default_features())
            .expect("holder session should install")
            .build();
        let holder_ctx = SessionContext::new_with_state(holder_state);
        holder_ctx
            .register_parquet(
                "holder_t",
                holder_path.to_str().expect("utf8 holder path"),
                ParquetReadOptions::default(),
            )
            .await
            .expect("holder table should register");
        let holder_attempt = backend.begin_attempt(AttemptOptions::default()).expect("holder attempt should begin");
        let mut holder_stream =
            holder_attempt.sql(&holder_ctx, "select id from holder_t").await.expect("holder should plan");
        holder_stream.next().await.expect("holder admits on first poll").expect("holder batch");
        let _held = (holder_attempt, holder_stream);

        let snapshot = backend.backend_metrics_snapshot();
        let device = snapshot.device(0).expect("configured device must have one bounded metrics slot");
        assert_eq!(device.admission.active_attempts, 1);
        assert_eq!(device.admission.live_grant_bytes, HOLDER_QUERY_CAP_BYTES);
        assert_eq!(device.admission.resident_owner_bytes, 16);
        // The default zero-cap profile creates no cache allocation domain, so
        // its ledger-backed cache projections remain byte-for-byte zero.
        assert_eq!(device.admission.cache_evictable_resident_bytes, 0);
        assert_eq!(device.admission.cache_pinned_in_use_bytes, 0);

        let payload = admission.diagnostics_payload().unwrap();

        assert!(payload.contains("# backend=query_service\n"), "{payload}");
        assert!(payload.contains("# device_profile_count=1\n"), "{payload}");
        assert!(payload.contains("# device_0_max_active_attempts=2\n"), "{payload}");
        assert!(payload.contains("# device_0_default_query_limit_bytes=536870912\n"), "{payload}");
        assert!(
            payload.contains("[backend_metrics]\nbackend_generation\tgeneration\tconfigured_devices\n"),
            "{payload}"
        );
        assert!(
            payload.contains(
                "[backend_device_admission]\ndevice_ordinal\tledger_sequence\tactive_attempts\tlive_grant_bytes\tresident_owner_bytes\tmax_resident_owner_bytes\tcache_evictable_resident_bytes\tcache_pinned_in_use_bytes\thealth\n0\t"
            ),
            "{payload}"
        );
        assert!(payload.contains("\t1\t536870912\t16\t16\t0\t0\tOpen\n"), "{payload}");
        assert!(
            !payload.contains("[device_ledger_grants]") && !payload.contains("[gpu_execution]"),
            "diagnostics must not combine the backend snapshot with separate service reads: {payload}"
        );
    }

    #[test]
    fn semaphore_admission_diagnostics_payload_reports_permit_counts() {
        let admission = QueryAdmission::new_for_tests();
        let _permit = admission.try_acquire().unwrap().expect("test-only admission should construct a permit");

        let payload = admission.diagnostics_payload().unwrap();

        assert!(payload.contains("# backend=test_only\n"), "{payload}");
    }
}

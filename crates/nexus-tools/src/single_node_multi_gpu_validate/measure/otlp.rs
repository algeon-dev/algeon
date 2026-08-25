use super::*;

/// Runtime-only OTLP exporter input. Its full endpoint is deliberately never
/// copied into an artifact; only `endpoint_host_port` is retained by W8.
#[derive(Clone)]
pub(in crate::single_node_multi_gpu_validate) struct OtlpExportConfig {
    #[cfg(feature = "single-node-multi-gpu-validate-run-otlp")]
    pub(super) endpoint: String,
    pub(super) endpoint_host_port: String,
    pub(super) interval: Duration,
}

impl OtlpExportConfig {
    pub(in crate::single_node_multi_gpu_validate) fn new(endpoint: &str, interval: Duration) -> ToolResult<Self> {
        let parsed =
            url::Url::parse(endpoint).map_err(|_| "--otlp-metrics-endpoint must be an absolute http(s) URL")?;
        if !matches!(parsed.scheme(), "http" | "https") {
            return Err("--otlp-metrics-endpoint must use http or https".into());
        }
        if !parsed.username().is_empty() || parsed.password().is_some() {
            return Err("--otlp-metrics-endpoint must not contain credentials".into());
        }
        let host = parsed.host_str().ok_or("--otlp-metrics-endpoint must include a host")?;
        let port = parsed.port_or_known_default().ok_or("--otlp-metrics-endpoint must include a port")?;
        Ok(Self {
            #[cfg(feature = "single-node-multi-gpu-validate-run-otlp")]
            endpoint: endpoint.to_string(),
            endpoint_host_port: format!("{host}:{port}"),
            interval,
        })
    }
}

/// Runs the reference `nexus-examples` OTLP exporter beside live traffic.
/// It owns no execution state and obtains every value through the public
/// `backend_metrics_snapshot()` accessor supplied to the example adapter.
#[cfg(feature = "single-node-multi-gpu-validate-run-otlp")]
pub(super) struct OtlpExportWorker {
    pub(super) stop: Arc<(Mutex<bool>, Condvar)>,
    pub(super) handle: std::thread::JoinHandle<OtlpExportEvidence>,
    pub(super) endpoint_host_port: String,
    pub(super) interval_ms: u64,
}

#[cfg(feature = "single-node-multi-gpu-validate-run-otlp")]
impl OtlpExportWorker {
    pub(super) fn start(backend: &NexusGpuBackend, config: &OtlpExportConfig) -> Self {
        let stop = Arc::new((Mutex::new(false), Condvar::new()));
        let signal = Arc::clone(&stop);
        let endpoint = config.endpoint.clone();
        let endpoint_host_port = config.endpoint_host_port.clone();
        let fallback_endpoint_host_port = endpoint_host_port.clone();
        let interval = config.interval;
        let interval_ms = interval.as_millis().min(u128::from(u64::MAX)) as u64;
        let backend = backend.clone();
        let handle = std::thread::spawn(move || {
            let runtime = match tokio::runtime::Builder::new_current_thread().enable_all().build() {
                Ok(runtime) => runtime,
                Err(_) => return OtlpExportEvidence::requested(endpoint_host_port, interval_ms, 0, 0, 1),
            };
            let meter = opentelemetry::global::meter("datafusion-nexus-stress");
            let snapshot_backend = backend.clone();
            let exporter = nexus_examples::otlp_metrics::BackendMetricsOtlpExporter::new(&meter, move || {
                snapshot_backend.backend_metrics_snapshot()
            });
            let resource = opentelemetry_proto::tonic::resource::v1::Resource::default();
            let start_time_unix_nano = unix_epoch_ns();
            let mut export_attempts = 0_u64;
            let mut successful_exports = 0_u64;
            let mut failed_exports = 0_u64;

            loop {
                export_attempts += 1;
                match runtime.block_on(exporter.export_histograms(
                    &endpoint,
                    &resource,
                    start_time_unix_nano,
                    unix_epoch_ns(),
                )) {
                    Ok(()) => successful_exports += 1,
                    Err(_) => failed_exports += 1,
                }
                let (lock, condition) = &*signal;
                let should_stop = match lock.lock() {
                    Ok(stopped) => match condition.wait_timeout(stopped, interval) {
                        Ok((stopped, _)) => *stopped,
                        Err(_) => true,
                    },
                    Err(_) => true,
                };
                if should_stop {
                    break;
                }
            }
            OtlpExportEvidence::requested(
                endpoint_host_port,
                interval_ms,
                export_attempts,
                successful_exports,
                failed_exports,
            )
        });
        Self { stop, handle, endpoint_host_port: fallback_endpoint_host_port, interval_ms }
    }

    pub(super) fn stop(self) -> OtlpExportEvidence {
        let (lock, condition) = &*self.stop;
        if let Ok(mut stopped) = lock.lock() {
            *stopped = true;
            condition.notify_all();
        }
        let endpoint_host_port = self.endpoint_host_port;
        let interval_ms = self.interval_ms;
        self.handle.join().unwrap_or_else(|_| OtlpExportEvidence::requested(endpoint_host_port, interval_ms, 0, 0, 1))
    }
}

/// A requested OTLP cell cannot run unless the explicit optional feature
/// linked the reference exporter. `run_measured_run` rejects that CLI shape
/// before reaching this fallback; retaining it here keeps the failure honest
/// if a future caller bypasses the CLI parser.
#[cfg(not(feature = "single-node-multi-gpu-validate-run-otlp"))]
pub(super) struct OtlpExportWorker {
    pub(super) evidence: OtlpExportEvidence,
}

#[cfg(not(feature = "single-node-multi-gpu-validate-run-otlp"))]
impl OtlpExportWorker {
    pub(super) fn start(_backend: &NexusGpuBackend, config: &OtlpExportConfig) -> Self {
        Self {
            evidence: OtlpExportEvidence::requested(
                config.endpoint_host_port.clone(),
                config.interval.as_millis().min(u128::from(u64::MAX)) as u64,
                0,
                0,
                1,
            ),
        }
    }

    pub(super) fn stop(self) -> OtlpExportEvidence {
        self.evidence
    }
}

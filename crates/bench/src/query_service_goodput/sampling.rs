use std::{
    collections::BTreeMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use algeon_datafusion::backend::AlgeonGpuBackend;

/// One device-scope telemetry sample.
///
/// Pool counters are device facts shared by every query on that device. They
/// are never projected into an attempt's peak; the attempt peak comes from
/// the query domain's own aligned counters in the sealed report.
#[derive(Debug, Clone, Copy)]
pub(super) struct DeviceSample {
    pub(super) offset_ms: f64,
    pub(super) device_ordinal: usize,
    pub(super) active_queries: usize,
    pub(super) queued_queries: usize,
    pub(super) managed_capacity_bytes: u64,
    pub(super) live_grants_bytes: u64,
    pub(super) available_bytes: u64,
    pub(super) active_grants: usize,
    pub(super) pool_reserved_current_bytes: u64,
    pub(super) pool_reserved_high_bytes: u64,
    pub(super) pool_used_current_bytes: u64,
    pub(super) pool_used_high_bytes: u64,
    pub(super) allocation_failure_count: u64,
    pub(super) trim_retry_count: u64,
    pub(super) host_rss_bytes: u64,
    pub(super) host_rss_peak_bytes: u64,
}

pub(super) struct Sampler {
    stop: Arc<AtomicBool>,
    handle: std::thread::JoinHandle<()>,
    samples: Arc<Mutex<Vec<DeviceSample>>>,
}

impl Sampler {
    pub(super) fn start(
        backend: &AlgeonGpuBackend,
        device_ordinals: &[usize],
        interval: Duration,
        origin: Instant,
    ) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let samples = Arc::new(Mutex::new(Vec::new()));
        let backend = backend.clone();
        let device_ordinals = device_ordinals.to_vec();
        let thread_stop = Arc::clone(&stop);
        let thread_samples = Arc::clone(&samples);
        let handle = std::thread::spawn(move || {
            // Alias the manager the backend already installed. Installing one
            // here instead would be rejected whenever the backend's options
            // differ, and the sampler would then report an unrelated pool.
            let managers = device_ordinals
                .iter()
                .filter_map(|ordinal| {
                    let device = match cudf::DeviceId::new(*ordinal) {
                        Ok(device) => device,
                        Err(error) => {
                            eprintln!("[goodput] device {ordinal} is unavailable for pool telemetry: {error}");
                            return None;
                        },
                    };
                    match cudf::DeviceMemoryManager::for_device(device) {
                        Ok(manager) => Some((*ordinal, manager)),
                        Err(error) => {
                            eprintln!("[goodput] pool telemetry unavailable for device {ordinal}: {error}");
                            None
                        },
                    }
                })
                .collect::<BTreeMap<_, _>>();
            while !thread_stop.load(Ordering::SeqCst) {
                let collected = collect(&backend, &device_ordinals, &managers, origin);
                thread_samples.lock().unwrap_or_else(std::sync::PoisonError::into_inner).extend(collected);
                std::thread::sleep(interval);
            }
        });
        Self { stop, handle, samples }
    }

    pub(super) fn finish(self) -> Vec<DeviceSample> {
        self.stop.store(true, Ordering::SeqCst);
        let _ = self.handle.join();
        let mut guard = self.samples.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        std::mem::take(&mut guard)
    }
}

fn collect(
    backend: &AlgeonGpuBackend,
    device_ordinals: &[usize],
    managers: &BTreeMap<usize, cudf::DeviceMemoryManager>,
    origin: Instant,
) -> Vec<DeviceSample> {
    let offset_ms = origin.elapsed().as_nanos() as f64 / 1_000_000.0;
    let (active_queries, queued_queries) = backend
        .admission_snapshot()
        .map(|snapshot| (snapshot.active_queries, snapshot.queued_queries))
        .unwrap_or_default();
    let (host_rss_bytes, host_rss_peak_bytes) = host_memory_bytes();
    let ledger = backend
        .device_ledger_reports()
        .map(|reports| reports.into_iter().map(|report| (report.device_ordinal, report)).collect::<BTreeMap<_, _>>())
        .unwrap_or_default();
    let failures = backend
        .device_memory_failure_reports()
        .map(|reports| reports.into_iter().map(|report| (report.device_ordinal, report)).collect::<BTreeMap<_, _>>())
        .unwrap_or_default();
    device_ordinals
        .iter()
        .map(|ordinal| {
            let pool = managers.get(ordinal).and_then(|manager| manager.pool_usage().ok());
            let ledger = ledger.get(ordinal);
            let failure = failures.get(ordinal);
            DeviceSample {
                offset_ms,
                device_ordinal: *ordinal,
                active_queries,
                queued_queries,
                managed_capacity_bytes: ledger.map_or(0, |report| report.managed_capacity_bytes),
                live_grants_bytes: ledger.map_or(0, |report| report.live_grants_bytes),
                available_bytes: ledger.map_or(0, |report| report.available_bytes),
                active_grants: ledger.map_or(0, |report| report.grants.len()),
                pool_reserved_current_bytes: pool.map_or(0, |usage| usage.reserved_current_bytes),
                pool_reserved_high_bytes: pool.map_or(0, |usage| usage.reserved_high_bytes),
                pool_used_current_bytes: pool.map_or(0, |usage| usage.used_current_bytes),
                pool_used_high_bytes: pool.map_or(0, |usage| usage.used_high_bytes),
                allocation_failure_count: failure.map_or(0, |report| report.allocation_failure_count),
                trim_retry_count: failure.map_or(0, |report| report.trim_retry_count),
                host_rss_bytes,
                host_rss_peak_bytes,
            }
        })
        .collect()
}

/// Current and peak resident host memory of this process.
///
/// Host memory is reported alongside device memory because bounded streaming
/// moves device results through host Arrow buffers; an unbounded host
/// footprint is a real cost of a concurrency configuration.
fn host_memory_bytes() -> (u64, u64) {
    let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
        return (0, 0);
    };
    let read = |prefix: &str| -> u64 {
        status
            .lines()
            .find_map(|line| line.strip_prefix(prefix))
            .and_then(|value| value.split_whitespace().next())
            .and_then(|value| value.parse::<u64>().ok())
            .map_or(0, |kib| kib.saturating_mul(1024))
    };
    (read("VmRSS:"), read("VmHWM:"))
}

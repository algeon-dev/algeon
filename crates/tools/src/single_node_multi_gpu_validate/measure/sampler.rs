use std::{
    collections::BTreeMap,
    sync::{Arc, Condvar, Mutex},
    time::{Duration, Instant},
};

use query_runtime::runtime::{BACKEND_ATTEMPT_DURATION_BUCKET_COUNT, DeviceLedgerReport};

use super::{
    AlgeonGpuBackend, BackendAttemptDisposition, BackendAttemptDuration, BackendAttemptOutcome,
    BackendHistogramSnapshot, BackendMetric, BackendMetricsSnapshot, BackendSnapshotSample, DeviceOccupancyPeaks,
    DeviceSeriesRow, MAX_SNAPSHOT_SAMPLES, MetricSnapshotObservation, ProcessSeriesRow, SAMPLE_INTERVAL, Value,
    backend_metric_descriptors, sample_interval_for,
};

/// Polls the public snapshot accessors on the window-aware interval and folds
/// them into per-device peaks plus the compact time series.
pub(super) struct Sampler {
    pub(super) stop: Arc<(Mutex<bool>, Condvar)>,
    pub(super) handle: std::thread::JoinHandle<SamplerResult>,
}

pub(super) struct SamplerResult {
    pub(super) samples: Vec<BackendSnapshotSample>,
    pub(super) occupancy: BTreeMap<usize, DeviceOccupancyPeaks>,
    pub(super) metric_snapshots: Vec<MetricSnapshotObservation>,
    pub(super) raw_snapshot_retention_truncated: bool,
    pub(super) device_series: Vec<DeviceSeriesRow>,
    pub(super) process_series: Vec<ProcessSeriesRow>,
    pub(super) sample_interval: Duration,
}

impl Sampler {
    pub(super) fn start(backend: &AlgeonGpuBackend, window: Duration) -> Self {
        let sample_interval = sample_interval_for(window);
        let stop = Arc::new((Mutex::new(false), Condvar::new()));
        let signal = Arc::clone(&stop);
        let started = Instant::now();
        // Capture the baseline before the caller can submit work. Spawning a
        // sampler thread and relying on its first poll to win that race makes the
        // resulting CUDA-free delta unsuitable for claim-versus-VRAM
        // calibration.
        let mut initial = SamplerResult::empty(sample_interval);
        sample_once(
            backend,
            started,
            &mut initial.samples,
            &mut initial.occupancy,
            &mut initial.metric_snapshots,
            &mut initial.raw_snapshot_retention_truncated,
            &mut initial.device_series,
            &mut initial.process_series,
            false,
        );
        let backend = backend.clone();
        let handle = std::thread::spawn(move || {
            let mut samples = initial.samples;
            let mut peaks = initial.occupancy;
            let mut metric_snapshots = initial.metric_snapshots;
            let mut raw_snapshot_retention_truncated = initial.raw_snapshot_retention_truncated;
            let mut device_series = initial.device_series;
            let mut process_series = initial.process_series;
            loop {
                // Waiting on the condvar keeps `stop` prompt, so the last sample
                // lands right after the run ends. A plain sleep would defer the stop
                // check by up to one sample interval.
                let (lock, condition) = &*signal;
                let Ok(stopped) = lock.lock() else {
                    break;
                };
                let Ok((stopped, _)) = condition.wait_timeout(stopped, sample_interval) else {
                    break;
                };
                if *stopped {
                    // One final observation after the drive loop finished.
                    drop(stopped);
                    sample_once(
                        &backend,
                        started,
                        &mut samples,
                        &mut peaks,
                        &mut metric_snapshots,
                        &mut raw_snapshot_retention_truncated,
                        &mut device_series,
                        &mut process_series,
                        true,
                    );
                    break;
                }
                sample_once(
                    &backend,
                    started,
                    &mut samples,
                    &mut peaks,
                    &mut metric_snapshots,
                    &mut raw_snapshot_retention_truncated,
                    &mut device_series,
                    &mut process_series,
                    false,
                );
            }
            for entry in peaks.values_mut() {
                if entry.min_root_streams_available == usize::MAX {
                    entry.min_root_streams_available = 0;
                }
                if entry.min_root_worker_slots_available == usize::MAX {
                    entry.min_root_worker_slots_available = 0;
                }
            }
            SamplerResult {
                samples,
                occupancy: peaks,
                metric_snapshots,
                raw_snapshot_retention_truncated,
                device_series,
                process_series,
                sample_interval,
            }
        });
        Self { stop, handle }
    }

    pub(super) fn stop(self) -> SamplerResult {
        let (lock, condition) = &*self.stop;
        if let Ok(mut stopped) = lock.lock() {
            *stopped = true;
            condition.notify_all();
        }
        self.handle.join().unwrap_or_else(|_| SamplerResult::empty(SAMPLE_INTERVAL))
    }
}

impl SamplerResult {
    pub(super) fn empty(sample_interval: Duration) -> Self {
        Self {
            samples: Vec::new(),
            occupancy: BTreeMap::new(),
            metric_snapshots: Vec::new(),
            raw_snapshot_retention_truncated: false,
            device_series: Vec::new(),
            process_series: Vec::new(),
            sample_interval,
        }
    }
}

/// The ledger-owned portion of one graph-model calibration observation. Equality across the two
/// reads bracketing telemetry proves that the physical sample is not paired
/// with a claim from a different ledger transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct CalibrationClaimSnapshot {
    pub(super) sequence: u64,
    pub(super) live_grants_bytes: u64,
    pub(super) cache_evictable_resident_bytes: u64,
    pub(super) cache_pinned_in_use_bytes: u64,
}

impl CalibrationClaimSnapshot {
    pub(super) fn from_report(report: &DeviceLedgerReport) -> Self {
        Self {
            sequence: report.sequence,
            live_grants_bytes: report.live_grants_bytes,
            cache_evictable_resident_bytes: report.cache_evictable_resident_bytes,
            cache_pinned_in_use_bytes: report.cache_pinned_in_use_bytes,
        }
    }
}

pub(super) fn calibration_claim_stable_across_telemetry(
    before: CalibrationClaimSnapshot,
    after: Option<CalibrationClaimSnapshot>,
) -> bool {
    after == Some(before)
}

/// One polling pass: folds the peaks, retains a bounded raw sample, and
/// appends the compact series rows. Raw retention may displace its tail
/// for the terminal sample; the compact series instead keeps the whole run,
/// since whole-window drift coverage is its purpose and its rows stay small.
#[expect(
    clippy::too_many_arguments,
    reason = "sampling-strategy matrix helper; parameters are the independent sampling axes"
)]
pub(super) fn sample_once(
    backend: &AlgeonGpuBackend,
    started: Instant,
    samples: &mut Vec<BackendSnapshotSample>,
    peaks: &mut BTreeMap<usize, DeviceOccupancyPeaks>,
    metric_snapshots: &mut Vec<MetricSnapshotObservation>,
    raw_snapshot_retention_truncated: &mut bool,
    device_series: &mut Vec<DeviceSeriesRow>,
    process_series: &mut Vec<ProcessSeriesRow>,
    retain_terminal_sample: bool,
) {
    let observation_started = Instant::now();
    // Graph-model calibration compares a logical claim against device telemetry. A diagnostics
    // read and a telemetry sample are not one transaction, so bracket telemetry
    // with two diagnostics reads. A row whose ledger sequence changed cannot
    // supply the "same-sample" claim and is retained only for non-calibration
    // utilization/drift evidence.
    let ledgers = backend.diagnostics().map(|diagnostics| diagnostics.ledgers).unwrap_or_default();
    let telemetry = backend.device_telemetry_snapshots().unwrap_or_default();
    let after_telemetry = backend.diagnostics().ok();
    let admission = after_telemetry.as_ref().map(|diagnostics| diagnostics.admission.clone());
    let (ledgers_after_telemetry, cache_reports, pools, failures) = after_telemetry
        .map(|diagnostics| {
            (diagnostics.ledgers, diagnostics.caches, diagnostics.execution_pools, diagnostics.memory_failures)
        })
        .unwrap_or_default();
    let cugraph_bfs_singleflight = cugraph_bfs_singleflight_snapshot(backend);
    let backend_metrics = backend.backend_metrics_snapshot();
    let completed_attempts = BackendAttemptDisposition::ALL
        .into_iter()
        .flat_map(|disposition| {
            BackendAttemptOutcome::ALL
                .into_iter()
                .map(move |outcome| backend_metrics.attempts.completed(disposition, outcome))
        })
        .fold(0_u64, u64::saturating_add);
    let successful_native_attempts = [BackendAttemptDisposition::PartialNative, BackendAttemptDisposition::Native]
        .into_iter()
        .map(|disposition| backend_metrics.attempts.completed(disposition, BackendAttemptOutcome::Succeeded))
        .fold(0_u64, u64::saturating_add);
    let observation_pull_ns = observation_started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
    for ledger in &ledgers {
        let entry = peaks.entry(ledger.device_ordinal).or_insert_with(|| DeviceOccupancyPeaks {
            min_root_streams_available: usize::MAX,
            min_root_worker_slots_available: usize::MAX,
            ..DeviceOccupancyPeaks::default()
        });
        entry.peak_active_attempts = entry.peak_active_attempts.max(ledger.active_attempts);
        entry.peak_live_grants_bytes = entry.peak_live_grants_bytes.max(ledger.live_grants_bytes);
        entry.peak_resident_owner_bytes = entry.peak_resident_owner_bytes.max(ledger.resident_owner_bytes);
        entry.min_root_streams_available = entry.min_root_streams_available.min(ledger.root_streams_available);
        entry.min_root_worker_slots_available =
            entry.min_root_worker_slots_available.min(ledger.root_worker_slots_available);
    }
    for pool in &pools {
        let entry = peaks.entry(pool.device_ordinal).or_default();
        entry.peak_active_query_jobs = entry.peak_active_query_jobs.max(pool.peak_active_query_jobs);
    }
    for failure in &failures {
        let entry = peaks.entry(failure.device_ordinal).or_default();
        entry.allocation_failure_count = entry.allocation_failure_count.max(failure.allocation_failure_count);
    }
    // Raw snapshots and metric-audit DTOs have a bounded artifact retention
    // budget, but the compact series must span the entire load window.
    // Keep sampling the same public observations after the raw budget fills.
    let retain_raw_snapshot =
        reserve_snapshot_slot(samples, metric_snapshots, raw_snapshot_retention_truncated, retain_terminal_sample);
    let offset_ns = started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
    process_series.push(ProcessSeriesRow {
        offset_ns,
        active_queries: admission.as_ref().map_or(0, |snapshot| snapshot.active_queries),
        queued_queries: admission.as_ref().map_or(0, |snapshot| snapshot.queued_queries),
        backend_generation: backend_metrics.backend_generation,
        metric_generation: backend_metrics.generation,
        observation_pull_ns,
        completed_attempts,
        successful_native_attempts,
    });
    for ledger in &ledgers {
        let ledger_after_telemetry =
            ledgers_after_telemetry.iter().find(|after| after.device_ordinal == ledger.device_ordinal);
        let claim_snapshot_end_sequence = ledger_after_telemetry.map(|after| after.sequence);
        let claim_snapshot_stable = calibration_claim_stable_across_telemetry(
            CalibrationClaimSnapshot::from_report(ledger),
            ledger_after_telemetry.map(CalibrationClaimSnapshot::from_report),
        );
        let cache_report = cache_reports.iter().find(|report| report.device_ordinal == ledger.device_ordinal);
        let device_telemetry = telemetry.iter().find(|snapshot| snapshot.device_ordinal == ledger.device_ordinal);
        let pool = pools.iter().find(|report| report.device_ordinal == ledger.device_ordinal);
        let failure = failures.iter().find(|report| report.device_ordinal == ledger.device_ordinal);
        device_series.push(DeviceSeriesRow {
            offset_ns,
            device_ordinal: ledger.device_ordinal,
            health_open: ledger.health == "open",
            gpu_utilization_pct: device_telemetry.and_then(|snapshot| snapshot.gpu_utilization_pct),
            memory_utilization_pct: device_telemetry.and_then(|snapshot| snapshot.memory_utilization_pct),
            cuda_free_bytes: device_telemetry.and_then(|snapshot| snapshot.cuda_free_bytes),
            nvml_used_bytes: device_telemetry.and_then(|snapshot| snapshot.nvml_used_bytes),
            active_attempts: ledger.active_attempts,
            managed_capacity_bytes: ledger.managed_capacity_bytes,
            live_grants_bytes: ledger.live_grants_bytes,
            live_grant_ids: ledger.grants.iter().map(|grant| grant.grant_id).collect(),
            resident_owner_bytes: ledger.resident_owner_bytes,
            cache_evictable_resident_bytes: ledger.cache_evictable_resident_bytes,
            cache_pinned_in_use_bytes: ledger.cache_pinned_in_use_bytes,
            cache_physical_current_bytes: cache_report.map(|report| report.physical_current_bytes),
            cache_physical_peak_bytes: cache_report.map(|report| report.physical_peak_bytes),
            allocation_failure_count: failure.map_or(0, |report| report.allocation_failure_count),
            ledger_sequence: ledger.sequence,
            claim_snapshot_end_sequence,
            claim_snapshot_stable,
            active_query_jobs: pool.map_or(0, |report| report.active_query_jobs),
            queued_jobs: pool.map_or(0, |report| report.queued_jobs),
        });
    }
    if !retain_raw_snapshot {
        return;
    }
    metric_snapshots.push(metric_snapshot_observation(offset_ns, &backend_metrics));
    samples.push(BackendSnapshotSample {
        offset_ns,
        backend_metrics_snapshot: serde_json::to_value(backend_metrics).unwrap_or(Value::Null),
        admission: admission.and_then(|snapshot| serde_json::to_value(snapshot).ok()).unwrap_or(Value::Null),
        device_ledger_reports: serde_json::to_value(&ledgers).unwrap_or(Value::Null),
        device_cache_reports: serde_json::to_value(&cache_reports).unwrap_or(Value::Null),
        device_memory_failure_reports: serde_json::to_value(&failures).unwrap_or(Value::Null),
        gpu_execution_pool_reports: serde_json::to_value(&pools).unwrap_or(Value::Null),
        device_telemetry_snapshots: serde_json::to_value(&telemetry).unwrap_or(Value::Null),
        cugraph_bfs_singleflight_snapshot: cugraph_bfs_singleflight,
    });
}

#[cfg(feature = "single-node-multi-gpu-validate-run-cugraph")]
pub(super) fn cugraph_bfs_singleflight_snapshot(backend: &AlgeonGpuBackend) -> Value {
    let snapshot = backend.cugraph_bfs_singleflight_snapshot();
    serde_json::json!({
        "eligible_requests": snapshot.eligible_requests,
        "physical_traversals": snapshot.physical_traversals,
        "follower_copies": snapshot.follower_copies,
        "groups_formed": snapshot.groups_formed,
        "overflow_groups": snapshot.overflow_groups,
        "leader_failures": snapshot.leader_failures,
        "cancelled_waiters": snapshot.cancelled_waiters,
        "peak_width": snapshot.peak_width,
        "events": snapshot.events.iter().map(|event| serde_json::json!({
            "sequence": event.sequence,
            "device_ordinal": event.device_ordinal,
            "singleflight_key_fingerprint": event.singleflight_key_fingerprint,
            "correlation_fingerprint": event.correlation_fingerprint,
            "generation": event.generation,
            "action": event.action,
        })).collect::<Vec<_>>(),
    })
}

#[cfg(not(feature = "single-node-multi-gpu-validate-run-cugraph"))]
pub(super) fn cugraph_bfs_singleflight_snapshot(_backend: &AlgeonGpuBackend) -> Value {
    Value::Null
}

/// Reserves one bounded raw/audit snapshot slot. The two vectors always move
/// together so an artifact sample and the corresponding metrics-audit DTO
/// cannot refer to different public backend-metric pulls.
pub(super) fn reserve_snapshot_slot<T, U>(
    samples: &mut Vec<T>,
    metric_snapshots: &mut Vec<U>,
    raw_snapshot_retention_truncated: &mut bool,
    retain_terminal_sample: bool,
) -> bool {
    debug_assert_eq!(samples.len(), metric_snapshots.len());
    if samples.len() < MAX_SNAPSHOT_SAMPLES {
        return true;
    }
    *raw_snapshot_retention_truncated = true;
    if !retain_terminal_sample {
        return false;
    }
    // The bounded artifact cannot retain every intermediate snapshot, but
    // backend-metric report equality must use the post-drain pull. Drop the former tail and
    // replace it with the terminal snapshot in lockstep with the audit DTO;
    // cadence remains explicitly not-evidenced after truncation.
    samples.pop();
    metric_snapshots.pop();
    true
}

/// Projects one public backend snapshot into the pure metrics-audit DTO. The raw
/// snapshot is retained separately in the artifact; this only records the
/// bounded checks needed to prove its cadence, cardinality, and exact terminal
/// count.
pub(super) fn metric_snapshot_observation(
    offset_ns: u64,
    snapshot: &BackendMetricsSnapshot,
) -> MetricSnapshotObservation {
    let configured_device_ordinals =
        snapshot.devices.iter().flatten().map(|device| device.device_ordinal).collect::<Vec<_>>();
    let (projected_series_count, expected_series_count) =
        backend_metric_descriptors().iter().fold((Some(0_usize), 0_usize), |(projected, expected), descriptor| {
            let projected = projected.and_then(|count| {
                snapshot.project_backend_metric(descriptor).ok().and_then(|samples| count.checked_add(samples.len()))
            });
            let expected = expected
                .saturating_add(expected_descriptor_series(descriptor.metric, configured_device_ordinals.len()));
            (projected, expected)
        });
    let terminal_completed = BackendAttemptDisposition::ALL
        .into_iter()
        .flat_map(|disposition| {
            BackendAttemptOutcome::ALL.into_iter().map(move |outcome| snapshot.attempts.completed(disposition, outcome))
        })
        .sum();
    let mut total_duration_bucket_counts = vec![0_u64; BACKEND_ATTEMPT_DURATION_BUCKET_COUNT];
    let mut total_duration_count = 0_u64;
    for disposition in BackendAttemptDisposition::ALL {
        for outcome in BackendAttemptOutcome::ALL {
            let histogram = snapshot.attempts.duration(disposition, outcome, BackendAttemptDuration::Total);
            accumulate_histogram(&mut total_duration_bucket_counts, &mut total_duration_count, histogram);
        }
    }
    MetricSnapshotObservation {
        offset_ns,
        backend_generation: snapshot.backend_generation,
        generation: snapshot.generation,
        configured_device_ordinals,
        projected_series_count,
        expected_series_count,
        terminal_completed,
        total_duration_bucket_counts,
        total_duration_count,
    }
}

pub(super) fn expected_descriptor_series(metric: BackendMetric, configured_device_count: usize) -> usize {
    match metric {
        BackendMetric::attempts_completed => BackendAttemptDisposition::COUNT * BackendAttemptOutcome::COUNT,
        BackendMetric::attempts_duration => {
            BackendAttemptDisposition::COUNT * BackendAttemptOutcome::COUNT * BackendAttemptDuration::COUNT
        },
        _ => configured_device_count,
    }
}

pub(super) fn accumulate_histogram(
    total_bucket_counts: &mut [u64],
    total_count: &mut u64,
    histogram: BackendHistogramSnapshot,
) {
    for (total, value) in total_bucket_counts.iter_mut().zip(histogram.bucket_counts) {
        *total = total.saturating_add(value);
    }
    *total_count = total_count.saturating_add(histogram.count);
}

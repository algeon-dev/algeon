//! Open-loop goodput and latency harness for the fixed-cap admission model.
//!
//! The harness drives one process-scoped [`NexusGpuBackend`] through the
//! adapter's attempt API — the same path `nexus-server` uses beneath its
//! Flight transport — replays a seeded arrival trace, and reports, for one
//! observation window:
//!
//! - offered, completed, succeeded and SLO-met counts with one shared
//!   denominator, plus goodput in queries and in work units per second,
//! - the closed outcome taxonomy (queue rejection, query-cap failure,
//!   physical OOM, unattributed allocation failure, cancellation, support
//!   rejection, other) so a configuration cannot look better by failing fast,
//! - queue/execution/cleanup and end-to-end latency percentiles,
//! - the attempt's own aligned allocator peak from its sealed report, kept
//!   separate from device-scope pool telemetry and host memory.
//!
//! Correctness is not assumed: every completed query is compared against an
//! output fingerprint frozen by a single-flight calibration run, and the SLO
//! deadlines come from that same frozen calibration rather than from the
//! configuration under test.
//!
//! Everything is environment-variable driven. GPU runs must be wrapped in
//! `flock /tmp/cudf-gpu.lock`.

use std::{
    collections::BTreeMap,
    error::Error,
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};

use datafusion::prelude::SessionContext;
use datafusion_nexus::{
    attempt::QueryAttemptReport,
    backend::{
        GpuDeviceMemoryLimit, GpuMemoryOwnership, NexusGpuBackend, NexusGpuDeviceConfig,
        NexusGpuDeviceInstallationBounds, NexusGpuDeviceProfile, NexusGpuQueuePolicy,
    },
    native_exec::NexusNativeExecutionConfig,
};
use nexus_query_engine::observability::{AttemptFailureFactKey, SealedAttemptProjection};
use nexus_workloads::{tpcds::describe_tpcds_dataset, tpch::describe_tpch_dataset};

mod arrival;
mod calibration;
mod config;
mod execution;
mod record;
mod report;
mod sampling;
mod workload;

use arrival::{Arrival, Phase, build_trace};
use calibration::{CalibrationIdentity, CalibrationRow, CalibrationStore};
use config::{GoodputConfig, Mode};
use execution::{AttemptOutcome, duration_ms, optional_ns_ms, run_attempt};
use record::{AttemptRecord, OutcomeClass, WindowSummary, classify, percentile, sorted_samples};
use report::RunReport;
use sampling::Sampler;
use workload::{
    FamilySessions, WorkloadFamily, WorkloadItem, build_sessions, discover_graph_seed, mix_families, resolve_item,
};

#[cfg(test)]
mod tests;

/// Qualification label carried by every summary row.
///
/// The harness exercises the same backend, attempt, ledger, allocator and
/// native path as the qualified `E2E-Flight` service, but not the gRPC
/// transport or the Flight host-Arrow streaming above it.
const QUALIFICATION: &str = "integration-engine";

pub async fn run() -> Result<(), Box<dyn Error>> {
    let config = Arc::new(GoodputConfig::from_env()?);
    std::fs::create_dir_all(config.run_dir())?;
    let backend = build_backend(&config)?;
    let result = run_with_backend(&config, &backend).await;
    let close = backend.close_and_wait(Duration::from_secs(120));
    result?;
    close?;
    Ok(())
}

async fn run_with_backend(config: &Arc<GoodputConfig>, backend: &NexusGpuBackend) -> Result<(), Box<dyn Error>> {
    let sessions = build_sessions(backend, config).await?;
    let families = mix_families(&config.mix)?;
    let graph_seed = match sessions.get(&WorkloadFamily::Graph) {
        Some(ctx) => discover_graph_seed(ctx).await?,
        None => 0,
    };
    let items =
        config.mix.iter().map(|entry| resolve_item(&entry.item_id, graph_seed)).collect::<Result<Vec<_>, _>>()?;
    let identity = live_identity(config, backend)?;
    match config.mode {
        Mode::Calibrate => calibrate(config, backend, &sessions, &items, &identity).await,
        Mode::Measure => measure(config, backend, &sessions, &items, &identity, &families).await,
    }
}

fn build_backend(config: &GoodputConfig) -> Result<NexusGpuBackend, Box<dyn Error>> {
    let execution_config = NexusNativeExecutionConfig::bounded(
        config.query_cap_bytes,
        config.max_source_chunk_bytes,
        config.max_row_groups_per_chunk,
        3,
        1,
    );
    let devices = config.device_ordinals.iter().copied().map(|ordinal| {
        let profile = NexusGpuDeviceProfile::new(ordinal)
            .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: config.device_capacity_bytes })
            .with_default_query_limit_bytes(config.query_cap_bytes)
            .with_max_active_attempts(config.max_active_attempts)
            .with_installation_bounds(NexusGpuDeviceInstallationBounds::new(
                config.stream_pool_size(),
                config.max_active_attempts,
                config.lane_count(),
            ));
        NexusGpuDeviceConfig::new(profile, GpuMemoryOwnership::WholeDeviceExclusive)
    });
    let queue_policy = NexusGpuQueuePolicy::default()
        .with_max_queued_attempts(config.max_queued_attempts)
        .with_max_queue_wait(config.max_queue_wait);
    Ok(NexusGpuBackend::builder()
        .with_native_execution_config(execution_config)
        .devices(devices)
        .queue_policy(queue_policy)
        .build()?)
}

fn live_identity(config: &GoodputConfig, backend: &NexusGpuBackend) -> Result<CalibrationIdentity, Box<dyn Error>> {
    // NVML product names need the optional `nvml` feature, which this package
    // does not enable; the CUDA capacity fact is always present and is enough
    // to catch a calibration carried over from different hardware.
    let gpu_model = backend
        .device_startup_records()
        .iter()
        .find(|record| Some(&record.device_ordinal) == config.device_ordinals.first())
        .map(|record| {
            record.model_name.clone().unwrap_or_else(|| format!("cuda_total_bytes:{}", record.facts.total_bytes))
        })
        .unwrap_or_else(|| "unknown".to_string());
    let tpch_fingerprint = match config.tpch_root.as_deref() {
        Some(root) => {
            describe_tpch_dataset(root, config.tpch_scale.clone(), "nexus_bench.goodput", "unspecified")
                .map_err(|error| format!("failed to describe the TPC-H dataset: {error}"))?
                .metadata_fingerprint
        },
        None => String::new(),
    };
    let tpcds_fingerprint = match config.tpcds_root.as_deref() {
        Some(root) => {
            describe_tpcds_dataset(root, config.tpcds_scale.clone(), "nexus_bench.goodput", "unspecified")
                .map_err(|error| format!("failed to describe the TPC-DS dataset: {error}"))?
                .metadata_fingerprint
        },
        None => String::new(),
    };
    let graph_fingerprint = match config.graph_edges.as_deref() {
        Some(edges) => graph_fingerprint(edges)?,
        None => String::new(),
    };
    Ok(CalibrationIdentity { gpu_model, tpch_fingerprint, tpcds_fingerprint, graph_fingerprint })
}

/// Identifies the graph fixture by the content hash its generator recorded,
/// falling back to the file length when the manifest is absent.
fn graph_fingerprint(edges: &Path) -> Result<String, Box<dyn Error>> {
    let manifest = edges.parent().map(|parent| parent.join("_generation.json"));
    if let Some(manifest) = manifest.filter(|path| path.exists()) {
        let text = std::fs::read_to_string(&manifest)?;
        let value: serde_json::Value = serde_json::from_str(&text)?;
        if let Some(sha256) = value.get("sha256").and_then(serde_json::Value::as_str) {
            return Ok(format!("sha256:{sha256}"));
        }
    }
    let length = std::fs::metadata(edges)?.len();
    Ok(format!("bytes:{length}"))
}

async fn calibrate(
    config: &Arc<GoodputConfig>,
    backend: &NexusGpuBackend,
    sessions: &FamilySessions,
    items: &[WorkloadItem],
    identity: &CalibrationIdentity,
) -> Result<(), Box<dyn Error>> {
    let mut rows = Vec::with_capacity(items.len());
    for item in items {
        let ctx = session_for(sessions, item)?;
        let mut latencies = Vec::with_capacity(config.calibration_reps);
        let mut peaks = Vec::with_capacity(config.calibration_reps);
        let mut disposition = "unavailable".to_string();
        let mut output_rows = 0_usize;
        let mut output_fingerprint = String::new();
        for rep in 0..(config.calibration_warmup_reps + config.calibration_reps) {
            let outcome = run_attempt(backend, ctx, config, item, format!("calibrate-{}-{rep}", item.id)).await;
            let facts = ReportFacts::from(outcome.report.as_deref());
            if outcome.error_message.is_some() || facts.terminal_outcome != "succeeded" {
                return Err(format!(
                    "calibration of {} failed on rep {rep}: outcome={} failure={} error={}",
                    item.id,
                    facts.terminal_outcome,
                    facts.failure_code.as_deref().unwrap_or("none"),
                    outcome.error_message.as_deref().unwrap_or("none"),
                )
                .into());
            }
            let fingerprint = outcome.output_fingerprint.clone().unwrap_or_default();
            if rep < config.calibration_warmup_reps {
                continue;
            }
            if output_fingerprint.is_empty() {
                output_fingerprint = fingerprint;
                output_rows = outcome.output_rows.unwrap_or_default();
                disposition = facts.disposition.clone();
            } else if output_fingerprint != fingerprint {
                return Err(format!("calibration of {} produced unstable output between reps", item.id).into());
            }
            latencies.push(duration_ms(outcome.finished.duration_since(outcome.started)));
            if let Some(peak) = facts.attempt_peak_aligned_bytes {
                peaks.push(peak as f64);
            }
        }
        let latencies = sorted_samples(latencies.into_iter());
        let peaks = sorted_samples(peaks.into_iter());
        rows.push(CalibrationRow {
            item_id: item.id.clone(),
            family: item.family,
            work_units: item.work_units,
            reps: config.calibration_reps,
            p50_ms: percentile(&latencies, 0.50).ok_or("calibration produced no measured reps")?,
            p95_ms: percentile(&latencies, 0.95).ok_or("calibration produced no measured reps")?,
            attempt_peak_aligned_bytes_p50: percentile(&peaks, 0.50).map(|value| value as u64),
            attempt_peak_aligned_bytes_max: peaks.last().map(|value| *value as u64),
            disposition,
            output_rows,
            output_fingerprint,
        });
        eprintln!("[goodput] calibrated {}", item.id);
    }
    let mut header = BTreeMap::new();
    header.insert("gpu_model".to_string(), identity.gpu_model.clone());
    header.insert("tpch_fingerprint".to_string(), identity.tpch_fingerprint.clone());
    header.insert("tpcds_fingerprint".to_string(), identity.tpcds_fingerprint.clone());
    header.insert("graph_fingerprint".to_string(), identity.graph_fingerprint.clone());
    header.insert("tpch_scale".to_string(), config.tpch_scale.clone());
    header.insert("tpcds_scale".to_string(), config.tpcds_scale.clone());
    header.insert("graph_dataset".to_string(), config.graph_dataset.clone());
    header.insert("device_capacity_bytes".to_string(), config.device_capacity_bytes.to_string());
    header.insert("query_cap_bytes".to_string(), config.query_cap_bytes.to_string());
    header.insert("pool_state".to_string(), "hot".to_string());
    header.insert("reps".to_string(), config.calibration_reps.to_string());
    header.insert("source_identity".to_string(), config.source_identity.clone());
    let store = CalibrationStore { identity: identity.clone(), header, rows };
    store.write_to(&config.slo_path)?;
    eprintln!("[goodput] wrote calibration to {}", config.slo_path.display());
    Ok(())
}

async fn measure(
    config: &Arc<GoodputConfig>,
    backend: &NexusGpuBackend,
    sessions: &FamilySessions,
    items: &[WorkloadItem],
    identity: &CalibrationIdentity,
    families: &[WorkloadFamily],
) -> Result<(), Box<dyn Error>> {
    let store = CalibrationStore::load(&config.slo_path)?;
    store.assert_identity(identity, families)?;
    for item in items {
        if store.row(&item.id).is_none() {
            return Err(format!("calibration has no frozen row for mix item {}; re-run calibration", item.id).into());
        }
    }
    let weights = config.mix.iter().map(|entry| entry.weight).collect::<Vec<_>>();
    let rate_qps = match config.rate_qps {
        Some(rate) => rate,
        None => config.rate_multiplier * single_flight_rate(config, items, &store)?,
    };
    let trace = build_trace(config, &weights, rate_qps);
    if trace.is_empty() {
        return Err("the arrival trace is empty; increase the rate or the window".into());
    }
    eprintln!(
        "[goodput] group={} arrivals={} rate_qps={rate_qps:.4} max_active={} cap_bytes={}",
        config.group,
        trace.len(),
        config.max_active_attempts,
        config.query_cap_bytes
    );

    let origin = Instant::now();
    let sampler = Sampler::start(backend, &config.device_ordinals, config.sample_interval, origin);
    let mut tasks = Vec::with_capacity(trace.len());
    for arrival in trace {
        let config = Arc::clone(config);
        let backend = backend.clone();
        let item = items[arrival.item_index].clone();
        let ctx = Arc::clone(session_for(sessions, &item)?);
        tasks.push(tokio::spawn(async move {
            let due = origin + arrival.offset;
            tokio::time::sleep_until(tokio::time::Instant::from_std(due)).await;
            let outcome =
                run_attempt(&backend, &ctx, &config, &item, format!("goodput-{}-{}", config.group, arrival.sequence))
                    .await;
            (arrival, item, outcome)
        }));
    }

    let mut records = Vec::with_capacity(tasks.len());
    for task in tasks {
        let (arrival, item, outcome) = task.await.map_err(|error| format!("arrival task failed: {error}"))?;
        records.push(build_record(config, &store, origin, arrival, &item, outcome));
    }
    records.sort_by_key(|record| record.sequence);
    let samples = sampler.finish();

    let window_records = records.iter().filter(|record| record.phase == Phase::Window).cloned().collect::<Vec<_>>();
    let window = WindowSummary::from_records(&window_records);
    let sensitivity = config
        .slo_sensitivity_multipliers
        .iter()
        .map(|multiplier| {
            let met = window_records
                .iter()
                .filter(|record| {
                    record.outcome == OutcomeClass::Succeeded
                        && store.row(&record.item_id).is_some_and(|row| record.end_to_end_ms <= row.p50_ms * multiplier)
                })
                .count();
            (*multiplier, met)
        })
        .collect::<Vec<_>>();

    let device_health = backend
        .device_ledger_reports()
        .map(|reports| {
            reports
                .into_iter()
                .map(|report| format!("{}:{}", report.device_ordinal, report.health))
                .collect::<Vec<_>>()
                .join("+")
        })
        .unwrap_or_else(|_| "unavailable".to_string());
    let report = RunReport {
        config,
        identity,
        qualification: QUALIFICATION,
        rate_qps,
        records: &records,
        window: &window,
        window_records: &window_records,
        samples: &samples,
        sensitivity: &sensitivity,
        device_health: &device_health,
    };
    report.write_all()?;
    println!("{}", report::SUMMARY_COLUMNS.join("\t"));
    println!("{}", report.summary_fields().join("\t"));
    eprintln!(
        "[goodput] group={} offered={} succeeded={} slo_met={} goodput_qps={:.4} memory_failure_ratio={:.4}",
        config.group,
        window.offered,
        window.succeeded,
        window.slo_met,
        window.goodput_qps(config.window_seconds),
        window.memory_failure_ratio()
    );
    Ok(())
}

/// Mean single-flight completion rate of the mix, from the frozen calibration.
///
/// This is the reference the rate ladder multiplies. It never reads the
/// configuration under test.
fn single_flight_rate(
    config: &GoodputConfig,
    items: &[WorkloadItem],
    store: &CalibrationStore,
) -> Result<f64, Box<dyn Error>> {
    let mut weighted_seconds = 0.0_f64;
    let mut total_weight = 0.0_f64;
    for (entry, item) in config.mix.iter().zip(items) {
        let row = store.row(&item.id).ok_or_else(|| format!("calibration has no row for {}", item.id))?;
        let weight = f64::from(entry.weight);
        weighted_seconds += weight * row.p50_ms / 1000.0;
        total_weight += weight;
    }
    if weighted_seconds <= 0.0 {
        return Err("calibration service time is not positive".into());
    }
    Ok(total_weight / weighted_seconds)
}

fn session_for<'a>(
    sessions: &'a FamilySessions,
    item: &WorkloadItem,
) -> Result<&'a Arc<SessionContext>, Box<dyn Error>> {
    sessions
        .get(&item.family)
        .ok_or_else(|| format!("no session registered for {} item {}", item.family.as_str(), item.id).into())
}

/// Stable labels lifted out of one sealed attempt report.
struct ReportFacts {
    disposition: String,
    terminal_outcome: String,
    device_ordinal: Option<usize>,
    grant_bytes: Option<u64>,
    queue_wait_ms: Option<f64>,
    planning_ms: Option<f64>,
    execution_ms: Option<f64>,
    cleanup_ms: Option<f64>,
    total_ms: Option<f64>,
    attempt_peak_aligned_bytes: Option<u64>,
    attempt_terminal_usage_provenance: String,
    attempt_pool_reserved_high_bytes: Option<u64>,
    unattributed_allocator_provenance: String,
    cleanup_disposition: String,
    capacity_disposition: String,
    failure_code: Option<String>,
    failure_status: Option<String>,
    failure_operation: Option<String>,
    failure_reason: Option<String>,
}

impl ReportFacts {
    fn from(report: Option<&QueryAttemptReport>) -> Self {
        let Some(report) = report else {
            return Self {
                disposition: "unavailable".to_string(),
                terminal_outcome: "unavailable".to_string(),
                device_ordinal: None,
                grant_bytes: None,
                queue_wait_ms: None,
                planning_ms: None,
                execution_ms: None,
                cleanup_ms: None,
                total_ms: None,
                attempt_peak_aligned_bytes: None,
                attempt_terminal_usage_provenance: "unavailable".to_string(),
                attempt_pool_reserved_high_bytes: None,
                unattributed_allocator_provenance: "unavailable".to_string(),
                cleanup_disposition: "unavailable".to_string(),
                capacity_disposition: "unavailable".to_string(),
                failure_code: None,
                failure_status: None,
                failure_operation: None,
                failure_reason: None,
            };
        };
        let evidence = report.resource_evidence.as_ref();
        // `SealedAttemptLabels` is the sole stable enum-to-string mapping for
        // post-seal consumers; this harness must not define a second one.
        let labels = SealedAttemptProjection::from(report).labels();
        Self {
            disposition: labels.disposition.to_string(),
            terminal_outcome: labels.outcome.to_string(),
            device_ordinal: report.device_ordinal,
            grant_bytes: report.admission.as_ref().map(|admission| admission.grant_bytes),
            queue_wait_ms: optional_ns_ms(report.queue_wait_ns()),
            planning_ms: optional_ns_ms(report.planning_ns()),
            execution_ms: optional_ns_ms(report.physical_execution_ns()),
            cleanup_ms: optional_ns_ms(report.cleanup_ns()),
            total_ms: optional_ns_ms(report.total_ns()),
            // The domain's aligned peak counter is monotonic over the attempt,
            // so the post-cleanup final sample carries the whole-attempt peak.
            // Its presence is the engine's own trustworthiness signal: absent
            // means no final sample completed, never a zero peak.
            attempt_peak_aligned_bytes: report.final_domain_usage.map(|usage| usage.peak_aligned_bytes),
            // Provenance of the engine's separate pre-cleanup terminal sample.
            // It qualifies that sample, not the peak above.
            attempt_terminal_usage_provenance: evidence.map_or_else(
                || "unavailable".to_string(),
                |evidence| label(&evidence.outcome_terminal_usage.provenance),
            ),
            attempt_pool_reserved_high_bytes: evidence
                .and_then(|evidence| evidence.outcome_terminal_usage.value)
                .and_then(|usage| usage.pool_usage)
                .map(|pool| pool.reserved_high_bytes),
            unattributed_allocator_provenance: evidence.map_or_else(
                || "unavailable".to_string(),
                |evidence| label(&evidence.unattributed_device_allocator_event.provenance),
            ),
            cleanup_disposition: labels.cleanup_disposition.to_string(),
            capacity_disposition: labels.capacity_disposition.to_string(),
            failure_code: report.primary_failure.as_ref().map(|failure| failure.code().as_str().to_string()),
            failure_status: report.primary_failure.as_ref().map(|failure| label(&failure.status())),
            failure_operation: report.primary_failure.as_ref().map(|failure| label(&failure.operation())),
            failure_reason: report.primary_failure.as_ref().and_then(|failure| {
                failure
                    .facts()
                    .iter()
                    .find(|fact| fact.key() == AttemptFailureFactKey::Reason)
                    .map(|fact| label(&fact.value()))
            }),
        }
    }
}

fn build_record(
    config: &GoodputConfig,
    store: &CalibrationStore,
    origin: Instant,
    arrival: Arrival,
    item: &WorkloadItem,
    outcome: AttemptOutcome,
) -> AttemptRecord {
    let facts = ReportFacts::from(outcome.report.as_deref());
    let calibration = store.row(&item.id);
    let fingerprint_match = match (&outcome.output_fingerprint, calibration) {
        (Some(observed), Some(row)) if !row.output_fingerprint.is_empty() => Some(*observed == row.output_fingerprint),
        _ => None,
    };
    let class = classify(
        Some(facts.terminal_outcome.as_str()),
        facts.failure_code.as_deref(),
        facts.failure_reason.as_deref(),
        outcome.adapter_error_code.as_deref(),
        fingerprint_match,
    );
    let submit_offset_ms = duration_ms(outcome.started.duration_since(origin));
    let scheduled_offset_ms = duration_ms(arrival.offset);
    // Response time is measured from the scheduled arrival, not from the
    // moment the harness got around to submitting, so any harness scheduling
    // delay counts against the configuration instead of being hidden.
    let end_to_end_ms = (duration_ms(outcome.finished.duration_since(origin)) - scheduled_offset_ms).max(0.0);
    let slo_deadline_ms = calibration.map(|row| row.p50_ms * config.slo_multiplier);
    let mut record = AttemptRecord {
        sequence: arrival.sequence,
        phase: arrival.phase,
        item_id: item.id.clone(),
        family: item.family,
        work_units: item.work_units,
        scheduled_offset_ms,
        submit_offset_ms,
        submit_delay_ms: (submit_offset_ms - scheduled_offset_ms).max(0.0),
        end_to_end_ms,
        queue_wait_ms: facts.queue_wait_ms,
        planning_ms: facts.planning_ms,
        execution_ms: facts.execution_ms,
        cleanup_ms: facts.cleanup_ms,
        attempt_total_ms: facts.total_ms,
        device_ordinal: facts.device_ordinal,
        grant_bytes: facts.grant_bytes,
        disposition: facts.disposition,
        terminal_outcome: facts.terminal_outcome,
        outcome: class,
        attempt_peak_aligned_bytes: facts.attempt_peak_aligned_bytes,
        attempt_terminal_usage_provenance: facts.attempt_terminal_usage_provenance,
        attempt_pool_reserved_high_bytes: facts.attempt_pool_reserved_high_bytes,
        unattributed_allocator_provenance: facts.unattributed_allocator_provenance,
        cleanup_disposition: facts.cleanup_disposition,
        capacity_disposition: facts.capacity_disposition,
        failure_code: facts.failure_code,
        failure_status: facts.failure_status,
        failure_operation: facts.failure_operation,
        failure_reason: facts.failure_reason,
        output_rows: outcome.output_rows,
        output_fingerprint: outcome.output_fingerprint,
        fingerprint_match,
        slo_deadline_ms,
        slo_met: false,
        harness_timeout: outcome.timed_out,
        error_message: outcome.error_message,
    };
    record.slo_met = record.meets(slo_deadline_ms);
    record
}

/// Renders an engine enum through its own `serde` snake_case rename.
///
/// This is used only for the closed enums that have no entry in
/// `SealedAttemptLabels` (evidence provenance, failure status and operation,
/// failure fact values). It reads the declared rename rather than restating a
/// mapping, so it does not create a second label source.
fn label<T: serde::Serialize>(value: &T) -> String {
    match serde_json::to_value(value) {
        Ok(serde_json::Value::String(text)) => text,
        Ok(other) => other.to_string(),
        Err(_) => "unknown".to_string(),
    }
}

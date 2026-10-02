use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::PathBuf,
    process::Command,
    sync::atomic::{AtomicUsize, Ordering},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use arrow::record_batch::RecordBatch;
use arrow_flight::FlightData;
use prost::Message as _;
use query_engine::observability::RUNTIME_OBSERVATION_JSONL_SCHEMA;
use serde::Serialize;
use serde_json::{Value, json};

use super::{
    E2eResult,
    params::{EVIDENCE_SCHEMA, EVIDENCE_VERSION, ObservabilityLimits, RuntimeParams},
    validation::{parse_u64_field, tsv_section, tsv_section_single_row},
};

pub(super) const CONTROL_SCHEMA: &str = "algeon_datafusion.query_runtime_artifact.control";
/// Every runtime artifact control carries exactly these fields, whatever its
/// state.
const CONTROL_FIELDS: [&str; 8] =
    ["schema", "query_id", "state", "retryable", "reason", "incomplete", "runtime_jsonl_follows", "attempt"];
pub(super) const QUERY_ID_METADATA: &str = "x-algeon-query-id";
pub(super) const CORRELATION_HEADER: &str = "x-algeon-correlation-id";
pub(super) const FLIGHT_MAX_ENCODED_FRAME_BYTES: usize = 4 * 1024 * 1024;
pub(super) const BROAD_PROJECTION_ROWS: usize = 262_144;
// Each dedicated probe may own at most 2 * 64 KiB. The 192 KiB global cap therefore admits
// every artifact individually but requires eviction across the set.
pub(super) const GLOBAL_BYTE_PRESSURE_PER_QUERY_BYTES: u64 = 64 * 1024;
pub(super) const GLOBAL_BYTE_PRESSURE_RETAINED_BYTES: u64 = 192 * 1024;
pub(super) const GLOBAL_BYTE_PRESSURE_RETAINED_COUNT: usize = 64;
pub(super) const GLOBAL_BYTE_PRESSURE_TTL_SECS: u64 = 60 * 60;
pub(super) const GLOBAL_BYTE_PRESSURE_MAX_QUERIES: usize = 16;
pub(super) const GLOBAL_BYTE_PRESSURE_SETUP_QUERIES: usize = 2;

pub(super) const FATAL_PATTERNS: &[&str] = &[
    "panicked at",
    "sigsegv",
    "sigabrt",
    "segmentation fault",
    "core dumped",
    "cuda_error_illegal_address",
    "illegal memory access",
    "an illegal memory access",
    "unspecified launch failure",
    "launch failed",
    "device lost",
    " xid ",
    "out of memory",
    "outofmemory",
    "allocation failed",
    "fatal runtime",
    "poisoned",
];

#[derive(Debug, Serialize)]
pub(super) struct RunEvidence {
    pub(super) schema: &'static str,
    pub(super) version: u32,
    pub(super) status: String,
    pub(super) failure: Option<String>,
    pub(super) started_unix_ms: u128,
    pub(super) elapsed_ms: u64,
    pub(super) hardware: Value,
    pub(super) config: Value,
    pub(super) scenarios: Vec<ScenarioEvidence>,
    pub(super) waves: Vec<WaveEvidence>,
    pub(super) retention: Option<RetentionEvidence>,
    pub(super) restarts: Vec<RestartEvidence>,
    pub(super) processes: Vec<ProcessEvidence>,
    pub(super) graph: Value,
    pub(super) server_logs: Vec<PathBuf>,
    pub(super) fatal_patterns: FatalPatternEvidence,
}

impl RunEvidence {
    pub(super) fn new(params: &RuntimeParams) -> Self {
        Self {
            schema: EVIDENCE_SCHEMA,
            version: EVIDENCE_VERSION,
            status: "running".to_string(),
            failure: None,
            started_unix_ms: SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |value| value.as_millis()),
            elapsed_ms: 0,
            hardware: hardware_evidence(),
            config: json!({
                "fixture_root": params.fixture_root,
                "catalog_path": params.catalog_path,
                "lineitem_path": params.lineitem_path(),
                "orders_path": params.orders_path(),
                "server_binary": params.server_binary,
                "device": params.device,
                "concurrency": params.concurrency,
                "waves": params.waves,
                "restart_iterations": params.restart_iterations,
                "scenario_selection": params.scenario_selection.label(),
                "native_once": params.native_once,
                "timeouts_secs": {
                    "connect": params.connect_timeout.as_secs(),
                    "query": params.query_timeout.as_secs(),
                    "poll": params.poll_timeout.as_secs(),
                    "shutdown": params.shutdown_timeout.as_secs(),
                },
                "observability": {
                    "per_query_records": params.observability.per_query_records,
                    "per_query_bytes": params.observability.per_query_bytes,
                    "retained_count": params.observability.retained_count,
                    "retained_bytes": params.observability.retained_bytes,
                    "ttl_secs": params.observability.ttl_secs,
                    "shutdown_grace_secs": params.observability.shutdown_grace_secs,
                },
            }),
            scenarios: Vec::new(),
            waves: Vec::new(),
            retention: None,
            restarts: Vec::new(),
            processes: Vec::new(),
            graph: json!({
                "covered": false,
                "reason": "not requested by this local Parquet matrix; use the explicit Flight graph smoke against a graph-enabled server"
            }),
            server_logs: Vec::new(),
            fatal_patterns: FatalPatternEvidence::default(),
        }
    }
}

#[derive(Debug, Serialize)]
pub(super) struct WaveEvidence {
    pub(super) wave: usize,
    pub(super) started: usize,
    pub(super) completed: usize,
    pub(super) passed: usize,
    pub(super) failed: usize,
    pub(super) elapsed_ms: u64,
    pub(super) throughput_queries_per_second: f64,
    pub(super) peak_active_attempts: u64,
    pub(super) overlap_proven: bool,
}

#[derive(Debug, Serialize)]
pub(super) struct ScenarioEvidence {
    pub(super) scenario: String,
    pub(super) execution_path: String,
    pub(super) query_id: String,
    pub(super) correlation_id: String,
    pub(super) outcome: String,
    pub(super) output_batches: usize,
    pub(super) output_rows: usize,
    pub(super) encoded_frame_count: Option<usize>,
    pub(super) record_batch_chunks: Option<usize>,
    pub(super) max_encoded_frame_bytes: Option<usize>,
    pub(super) stats_assertions: Vec<String>,
    pub(super) artifact: ArtifactEvidence,
    pub(super) elapsed_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct ArtifactEvidence {
    pub(super) state: String,
    pub(super) reason: Option<String>,
    pub(super) frame_count: usize,
    pub(super) wire_bytes: usize,
    pub(super) jsonl_records: usize,
    pub(super) topology_records: usize,
    pub(super) timeline_events: usize,
    pub(super) plan_execution_ids: Vec<String>,
    pub(super) terminal_kind: Option<String>,
    pub(super) retained_records: Option<u64>,
    pub(super) retained_bytes: Option<u64>,
    pub(super) truncated: Option<bool>,
    pub(super) degraded: Option<bool>,
    pub(super) poison_recoveries: Option<u64>,
    pub(super) dropped_by_class: Option<Value>,
    pub(super) dropped_by_reason: Option<Value>,
}

#[derive(Debug, Serialize)]
pub(super) struct RetentionEvidence {
    pub(super) generated_query_ids: Vec<String>,
    pub(super) live_query_id: String,
    pub(super) live_state_during_pressure: String,
    pub(super) sealed_count: usize,
    pub(super) unavailable: BTreeMap<String, String>,
    pub(super) not_found: Vec<String>,
    pub(super) generated_wire_bytes: usize,
    pub(super) capture_budget: CaptureBudgetEvidence,
    pub(super) global_byte_pressure: Option<GlobalBytePressureEvidence>,
    pub(super) ttl_query_id: String,
    pub(super) ttl_terminal_state: String,
    pub(super) ttl_unavailable_elapsed_ms: u64,
    pub(super) ttl_not_found_elapsed_ms: u64,
    pub(super) no_silent_loss: bool,
}

#[derive(Debug, Serialize)]
pub(super) struct CaptureBudgetEvidence {
    pub(super) query_id: String,
    pub(super) input_bytes: usize,
    pub(super) per_query_capture_limit: u64,
    pub(super) state: String,
    pub(super) reason: String,
    pub(super) frame_count: usize,
    pub(super) stats_incomplete: bool,
    pub(super) retained_byte_loss_recorded: bool,
}

#[derive(Debug, Serialize)]
pub(super) struct GlobalBytePressureEvidence {
    pub(super) server_label: String,
    pub(super) per_query_bytes: u64,
    pub(super) retained_count_limit: usize,
    pub(super) retained_byte_limit: u64,
    pub(super) ttl_secs: u64,
    pub(super) generated_query_ids: Vec<String>,
    pub(super) artifact_charge_bytes: BTreeMap<String, u64>,
    pub(super) sealed_query_ids_after_pressure: Vec<String>,
    pub(super) evicted: BTreeMap<String, String>,
    pub(super) generated_charge_bytes: u64,
    pub(super) max_observed_generated_retained_bytes: u64,
    pub(super) retained_bytes_after_pressure: u64,
    pub(super) setup_artifacts_known_evicted: bool,
    pub(super) exact_retained_set_proven: bool,
    pub(super) count_limit_not_reached: bool,
    pub(super) byte_limit_respected: bool,
    pub(super) no_silent_loss: bool,
}

#[derive(Debug, Serialize)]
pub(super) struct RestartEvidence {
    pub(super) iteration: usize,
    pub(super) termination: String,
    pub(super) in_flight_query_id: Option<String>,
    pub(super) in_flight_query_ids: Vec<String>,
    pub(super) client_terminal_results: Vec<String>,
    pub(super) exit: ProcessEvidence,
    pub(super) recovery_query_id: String,
    pub(super) recovery_artifact_state: String,
    pub(super) recovery_idle: bool,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct ProcessEvidence {
    pub(super) label: String,
    pub(super) pid: Option<u32>,
    pub(super) stdout_log: PathBuf,
    pub(super) stderr_log: PathBuf,
    pub(super) exit_success: bool,
    pub(super) exit_code: Option<i32>,
    pub(super) exit_signal: Option<i32>,
    pub(super) exit_display: String,
    pub(super) elapsed_ms: u64,
}

#[derive(Debug, Default, Serialize)]
pub(super) struct FatalPatternEvidence {
    pub(super) scanned_files: usize,
    pub(super) matches: Vec<Value>,
}

pub(super) struct CollectedFlight {
    pub(super) batches: Vec<RecordBatch>,
    pub(super) encoded_frame_count: usize,
    pub(super) max_encoded_frame_bytes: usize,
}

impl CollectedFlight {
    pub(super) fn record_batch_chunks(&self) -> usize {
        self.batches.len()
    }

    pub(super) fn frame_evidence(&self) -> VariableFrameEvidence {
        VariableFrameEvidence {
            encoded_frame_count: self.encoded_frame_count,
            record_batch_chunks: self.record_batch_chunks(),
            max_encoded_frame_bytes: self.max_encoded_frame_bytes,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct VariableFrameEvidence {
    pub(super) encoded_frame_count: usize,
    pub(super) record_batch_chunks: usize,
    pub(super) max_encoded_frame_bytes: usize,
}

pub(super) fn validate_variable_frame_evidence(evidence: VariableFrameEvidence) -> E2eResult<()> {
    if evidence.record_batch_chunks < 2 {
        return Err("variable-width projection did not exercise Flight record-batch chunking".into());
    }
    if evidence.max_encoded_frame_bytes > FLIGHT_MAX_ENCODED_FRAME_BYTES {
        return Err(format!(
            "variable-width projection emitted a {} byte Flight frame above the {} byte limit",
            evidence.max_encoded_frame_bytes, FLIGHT_MAX_ENCODED_FRAME_BYTES
        )
        .into());
    }
    if evidence.encoded_frame_count < evidence.record_batch_chunks.saturating_add(1) {
        return Err("variable-width projection did not observe schema plus record-batch Flight frames".into());
    }
    Ok(())
}

pub(super) fn observe_encoded_frame(frame: &FlightData, frame_count: &AtomicUsize, max_frame_bytes: &AtomicUsize) {
    frame_count.fetch_add(1, Ordering::Relaxed);
    max_frame_bytes.fetch_max(frame.encoded_len(), Ordering::Relaxed);
}

#[derive(Debug)]
pub(super) struct GlobalBytePressureSnapshot {
    pub(super) sealed_query_ids: Vec<String>,
    pub(super) sealed_charge_bytes: BTreeMap<String, u64>,
    pub(super) evicted: BTreeMap<String, String>,
    pub(super) retained_bytes: u64,
}

pub(super) fn artifact_charge(runtime_wire_bytes: usize, stats_bytes: usize) -> E2eResult<u64> {
    u64::try_from(runtime_wire_bytes)
        .ok()
        .and_then(|runtime| u64::try_from(stats_bytes).ok().and_then(|stats| runtime.checked_add(stats)))
        .ok_or_else(|| "runtime artifact exact byte charge overflowed".into())
}

fn parse_control(query_id: &str, frames: &[Vec<u8>]) -> E2eResult<Value> {
    let first = frames.first().ok_or("runtime artifact action returned no frames")?;
    let control: Value = serde_json::from_slice(first)?;
    let has_control_shape = control.as_object().is_some_and(|fields| {
        fields.len() == CONTROL_FIELDS.len() && CONTROL_FIELDS.iter().all(|field| fields.contains_key(*field))
    });
    if !has_control_shape
        || control.get("schema").and_then(Value::as_str) != Some(CONTROL_SCHEMA)
        || control.get("query_id").and_then(Value::as_str) != Some(query_id)
    {
        return Err(format!("invalid runtime artifact control for {query_id}: {control}").into());
    }
    Ok(control)
}

pub(super) fn control_state(query_id: &str, frames: &[Vec<u8>]) -> E2eResult<String> {
    let control = parse_control(query_id, frames)?;
    Ok(control
        .get("state")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("runtime artifact control lacks state: {control}"))?
        .to_string())
}

pub(super) fn require_stats_identity(stats: &str, query_id: &str, correlation_id: &str) -> E2eResult<()> {
    require_stats_contains(stats, &format!("# query_id={query_id}\n"), "query id")?;
    require_stats_contains(stats, &format!("# correlation_id={correlation_id}\n"), "correlation id")?;
    require_stats_contains(stats, "# sealed=true\n", "sealed stats")?;
    if stats.to_ascii_lowercase().contains("multiple_sessions") {
        return Err(format!("query {query_id} stats reported multiple_sessions").into());
    }
    Ok(())
}

pub(super) fn require_stats_contains(stats: &str, needle: &str, label: &str) -> E2eResult<()> {
    if !stats.contains(needle) {
        return Err(format!("query stats missing {label} {needle:?}:\n{stats}").into());
    }
    Ok(())
}

pub(super) fn require_stats_field(stats: &str, section: &str, field: &str, expected: &str) -> E2eResult<()> {
    let row = tsv_section_single_row(stats, section)?;
    let actual = row.get(field).ok_or_else(|| format!("query stats [{section}] missing field {field}: {row:?}"))?;
    if actual != expected {
        return Err(format!("query stats [{section}] expected {field}={expected:?}, got {actual:?}").into());
    }
    Ok(())
}

pub(super) fn parse_artifact(
    query_id: &str,
    expected_correlation_id: Option<&str>,
    frames: &[Vec<u8>],
    expected_terminal: Option<&str>,
    expected_limits: Option<&ObservabilityLimits>,
) -> E2eResult<ArtifactEvidence> {
    let wire_bytes = frames.iter().map(Vec::len).sum();
    let control = parse_control(query_id, frames)?;
    let state = control
        .get("state")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("runtime artifact control lacks state: {control}"))?;
    let retryable = control
        .get("retryable")
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("runtime artifact control lacks retryable: {control}"))?;
    let reason = control.get("reason").and_then(Value::as_str).map(str::to_string);
    let runtime_jsonl_follows = control
        .get("runtime_jsonl_follows")
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("runtime artifact control lacks runtime_jsonl_follows: {control}"))?;
    if runtime_jsonl_follows && state != "sealed" {
        return Err(format!("runtime artifact {query_id} state {state} announces a runtime JSONL frame").into());
    }
    let expected_frame_count = 1 + usize::from(runtime_jsonl_follows);
    if frames.len() != expected_frame_count {
        return Err(format!(
            "runtime artifact {query_id} state {state} returned {} frames, expected {expected_frame_count}",
            frames.len()
        )
        .into());
    }
    match state {
        "pending" | "active" if !retryable => {
            return Err(format!("runtime artifact {query_id} state {state} must be retryable").into());
        },
        "sealed" | "unavailable" if retryable => {
            return Err(format!("runtime artifact {query_id} state {state} must not be retryable").into());
        },
        "pending" | "active" | "sealed" | "unavailable" => {},
        other => return Err(format!("runtime artifact {query_id} has unknown state {other:?}").into()),
    }
    if state == "unavailable" && reason.is_none() {
        return Err(format!("runtime artifact {query_id} unavailable control lacks reason").into());
    }
    // A runtime observation belongs to an attempt, so a control followed by
    // its JSONL frame always projects that attempt's sealed report.
    match control.get("attempt") {
        Some(Value::Null) if runtime_jsonl_follows => {
            return Err(format!("sealed runtime artifact control lacks attempt projection: {control}").into());
        },
        Some(Value::Null) => {},
        Some(Value::Object(attempt)) => {
            let retry_advice = attempt
                .get("retry_advice")
                .and_then(Value::as_object)
                .ok_or_else(|| format!("runtime artifact control attempt lacks retry advice: {control}"))?;
            if retry_advice.get("recovery").and_then(Value::as_str).is_none()
                || retry_advice.get("prerequisite").and_then(Value::as_str).is_none()
            {
                return Err(format!("runtime artifact control attempt has incomplete retry advice: {control}").into());
            }
        },
        _ => return Err(format!("runtime artifact control attempt is not an object or null: {control}").into()),
    }
    if !runtime_jsonl_follows {
        if expected_terminal.is_some() {
            let observation = if state == "sealed" { " without a runtime observation" } else { "" };
            return Err(format!("runtime artifact {query_id} is {state}{observation}, expected sealed terminal").into());
        }
        return Ok(ArtifactEvidence {
            state: state.to_string(),
            reason,
            frame_count: frames.len(),
            wire_bytes,
            jsonl_records: 0,
            topology_records: 0,
            timeline_events: 0,
            plan_execution_ids: Vec::new(),
            terminal_kind: None,
            retained_records: None,
            retained_bytes: None,
            truncated: None,
            degraded: None,
            poison_recoveries: None,
            dropped_by_class: None,
            dropped_by_reason: None,
        });
    }

    let jsonl = std::str::from_utf8(&frames[1])?;
    if !jsonl.ends_with('\n') {
        return Err(format!("runtime artifact {query_id} JSONL lacks a final newline").into());
    }
    if jsonl.to_ascii_lowercase().contains("multiple_sessions") {
        return Err(format!("runtime artifact {query_id} reported multiple_sessions").into());
    }
    let records = jsonl
        .lines()
        .enumerate()
        .map(|(index, line)| {
            serde_json::from_str::<Value>(line)
                .map_err(|error| format!("runtime artifact {query_id} JSONL line {index} is invalid: {error}").into())
        })
        .collect::<E2eResult<Vec<_>>>()?;
    if records.len() < 2 {
        return Err(format!("runtime artifact {query_id} needs manifest and footer records").into());
    }
    let manifest_indices = record_indices(&records, "manifest");
    let footer_indices = record_indices(&records, "footer");
    if manifest_indices != [0] || footer_indices != [records.len() - 1] {
        return Err(format!(
            "runtime artifact {query_id} requires one first manifest and one last footer; manifest={manifest_indices:?} footer={footer_indices:?}"
        )
        .into());
    }
    let manifest = records[0].get("data").ok_or("manifest lacks data")?;
    if manifest.get("schema").and_then(Value::as_str) != Some(RUNTIME_OBSERVATION_JSONL_SCHEMA)
        || manifest.get("query_id").and_then(Value::as_str) != Some(query_id)
    {
        return Err(format!("runtime artifact {query_id} has invalid manifest: {manifest}").into());
    }
    if let Some(expected) = expected_correlation_id
        && manifest.get("trace_id").and_then(Value::as_str) != Some(expected)
    {
        return Err(format!(
            "runtime artifact {query_id} trace_id {:?} did not match correlation id {expected:?}",
            manifest.get("trace_id")
        )
        .into());
    }
    let manifest_record_limit = manifest
        .pointer("/limits/max_retained_records")
        .and_then(Value::as_u64)
        .ok_or("manifest lacks max_retained_records")?;
    let manifest_byte_limit = manifest
        .pointer("/limits/max_retained_bytes")
        .and_then(Value::as_u64)
        .ok_or("manifest lacks max_retained_bytes")?;
    if let Some(limits) = expected_limits
        && (manifest_record_limit != limits.per_query_records || manifest_byte_limit != limits.per_query_bytes)
    {
        return Err(format!(
            "runtime artifact {query_id} manifest limits records={manifest_record_limit} bytes={manifest_byte_limit}, expected records={} bytes={}",
            limits.per_query_records, limits.per_query_bytes
        )
        .into());
    }

    const TOPOLOGY_TYPES: &[&str] = &[
        "plan_node",
        "plan_edge",
        "task_definition",
        "compiled_edge",
        "barrier_definition",
        "sink_definition",
        "job_definition",
        "worker_definition",
        "resource",
    ];
    let mut topology_records = 0usize;
    let mut timeline_events = 0usize;
    let mut timeline_started = false;
    let mut last_topology_rank = None;
    let mut last_timeline_sequence = None;
    let mut retained_sequences = BTreeSet::new();
    let mut plan_execution_ids = BTreeSet::new();
    for (index, record) in records.iter().enumerate().skip(1).take(records.len() - 2) {
        let record_type = record
            .get("record_type")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("runtime artifact {query_id} record {index} lacks record_type"))?;
        if TOPOLOGY_TYPES.contains(&record_type) {
            if timeline_started {
                return Err(format!("runtime artifact {query_id} topology record follows timeline at {index}").into());
            }
            let rank = TOPOLOGY_TYPES
                .iter()
                .position(|candidate| *candidate == record_type)
                .expect("known topology record type");
            if last_topology_rank.is_some_and(|previous| rank < previous) {
                return Err(format!(
                    "runtime artifact {query_id} topology record type {record_type} is outside canonical enum order"
                )
                .into());
            }
            last_topology_rank = Some(rank);
            topology_records += 1;
        } else if record_type == "timeline_event" {
            timeline_started = true;
            timeline_events += 1;
        } else {
            return Err(format!("runtime artifact {query_id} has unexpected record type {record_type:?}").into());
        }
        let sequence = record
            .pointer("/data/sequence")
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("runtime artifact {query_id} record {index} lacks numeric sequence"))?;
        if record_type == "timeline_event" {
            if last_timeline_sequence.is_some_and(|previous| sequence <= previous) {
                return Err(format!(
                    "runtime artifact {query_id} timeline sequence is not strictly increasing: {sequence} after {last_timeline_sequence:?}"
                )
                .into());
            }
            last_timeline_sequence = Some(sequence);
        }
        if !retained_sequences.insert(sequence) {
            return Err(format!("runtime artifact {query_id} repeats sequence {sequence}").into());
        }
        collect_plan_execution_ids(record, &mut plan_execution_ids)?;
    }

    let footer = records.last().and_then(|record| record.get("data")).ok_or("footer lacks data")?;
    if footer.get("complete").and_then(Value::as_bool) != Some(true) {
        return Err(format!("runtime artifact {query_id} footer is incomplete: {footer}").into());
    }
    let terminal_kind = footer.pointer("/terminal/kind").and_then(Value::as_str).map(str::to_string);
    if let Some(expected) = expected_terminal
        && terminal_kind.as_deref() != Some(expected)
    {
        return Err(format!(
            "runtime artifact {query_id} terminal {:?} did not match stats outcome {expected:?}",
            terminal_kind
        )
        .into());
    }
    let retained_records =
        footer.get("retained_records").and_then(Value::as_u64).ok_or("footer lacks retained_records")?;
    let retained_bytes = footer.get("retained_bytes").and_then(Value::as_u64).ok_or("footer lacks retained_bytes")?;
    let observed_retained = u64::try_from(topology_records + timeline_events)?;
    // The footer also counts the metrics detail rows (operator, strategy,
    // dynamic-filter, and budget rows) retained for query stats; the JSONL
    // carries only topology and timeline records.
    if retained_records < observed_retained
        || retained_records > manifest_record_limit
        || retained_bytes > manifest_byte_limit
        || (retained_records == 0) != (retained_bytes == 0)
    {
        return Err(format!(
            "runtime artifact {query_id} footer retention inconsistent: records={retained_records}/{observed_retained}/{manifest_record_limit}, bytes={retained_bytes}/{manifest_byte_limit}"
        )
        .into());
    }
    let truncated = footer.get("truncated").and_then(Value::as_bool).ok_or("footer lacks truncated")?;
    let degraded = footer.get("degraded").and_then(Value::as_bool).ok_or("footer lacks degraded")?;
    let poison_recoveries =
        footer.get("poison_recoveries").and_then(Value::as_u64).ok_or("footer lacks poison_recoveries")?;
    validate_footer_sequences(query_id, footer, retained_records, &retained_sequences, truncated)?;
    let plan_execution_ids = plan_execution_ids.into_iter().collect::<Vec<_>>();
    Ok(ArtifactEvidence {
        state: state.to_string(),
        reason,
        frame_count: frames.len(),
        wire_bytes,
        jsonl_records: records.len(),
        topology_records,
        timeline_events,
        plan_execution_ids,
        terminal_kind,
        retained_records: Some(retained_records),
        retained_bytes: Some(retained_bytes),
        truncated: Some(truncated),
        degraded: Some(degraded),
        poison_recoveries: Some(poison_recoveries),
        dropped_by_class: footer.get("dropped_by_class").cloned(),
        dropped_by_reason: footer.get("dropped_by_reason").cloned(),
    })
}

pub(super) fn record_indices(records: &[Value], record_type: &str) -> Vec<usize> {
    records
        .iter()
        .enumerate()
        .filter_map(|(index, record)| {
            (record.get("record_type").and_then(Value::as_str) == Some(record_type)).then_some(index)
        })
        .collect()
}

pub(super) fn validate_footer_sequences(
    query_id: &str,
    footer: &Value,
    retained_records: u64,
    retained_sequences: &BTreeSet<u64>,
    truncated: bool,
) -> E2eResult<()> {
    let first_dropped = optional_u64_field(footer, "first_dropped_sequence")?;
    let last_attempted = optional_u64_field(footer, "last_attempted_sequence")?;
    let dropped_by_class = loss_count(footer, "dropped_by_class")?;
    let dropped_by_reason = loss_count(footer, "dropped_by_reason")?;
    if truncated != first_dropped.is_some() || first_dropped.is_some() != (dropped_by_class > 0) {
        return Err(format!(
            "runtime artifact {query_id} footer loss flags disagree: truncated={truncated} first_dropped={first_dropped:?} dropped_by_class={dropped_by_class}"
        )
        .into());
    }
    if dropped_by_reason < dropped_by_class {
        return Err(format!(
            "runtime artifact {query_id} reason loss {dropped_by_reason} is below class loss {dropped_by_class}"
        )
        .into());
    }
    match last_attempted {
        Some(last) => {
            if retained_records.saturating_add(dropped_by_class) != last {
                return Err(format!(
                    "runtime artifact {query_id} attempted sequence {last} does not reconcile retained={retained_records} + class_dropped={dropped_by_class}"
                )
                .into());
            }
            if retained_sequences.last().is_some_and(|sequence| *sequence > last) {
                return Err(
                    format!("runtime artifact {query_id} retained sequence exceeds last attempted {last}").into()
                );
            }
            if let Some(first) = first_dropped
                && (first == 0 || first > last || retained_sequences.contains(&first))
            {
                return Err(format!(
                    "runtime artifact {query_id} invalid first dropped sequence {first} for last attempted {last}"
                )
                .into());
            }
        },
        None if retained_records == 0 && dropped_by_class == 0 => {},
        None => {
            return Err(format!(
                "runtime artifact {query_id} lacks last_attempted_sequence with retained={retained_records} dropped={dropped_by_class}"
            )
            .into());
        },
    }
    Ok(())
}

pub(super) fn optional_u64_field(value: &Value, field: &str) -> E2eResult<Option<u64>> {
    match value.get(field) {
        Some(Value::Null) | None => Ok(None),
        Some(value) => value
            .as_u64()
            .map(Some)
            .ok_or_else(|| format!("footer field {field} is not an unsigned integer: {value}").into()),
    }
}

pub(super) fn loss_count(footer: &Value, field: &str) -> E2eResult<u64> {
    let rows =
        footer.get(field).and_then(Value::as_array).ok_or_else(|| format!("footer field {field} is not an array"))?;
    rows.iter().try_fold(0u64, |total, row| {
        let count =
            row.get("count").and_then(Value::as_u64).ok_or_else(|| format!("footer {field} row lacks count: {row}"))?;
        Ok(total.saturating_add(count))
    })
}

pub(super) fn collect_plan_execution_ids(value: &Value, output: &mut BTreeSet<String>) -> E2eResult<()> {
    match value {
        Value::Object(object) => {
            for (key, value) in object {
                if key == "plan_execution_id"
                    && let Some(id) = value.as_str()
                {
                    if id.is_empty() || id.len() > 256 || id.chars().any(char::is_control) {
                        return Err(format!("illegal plan_execution_id {id:?}").into());
                    }
                    output.insert(id.to_string());
                }
                collect_plan_execution_ids(value, output)?;
            }
        },
        Value::Array(values) => {
            for value in values {
                collect_plan_execution_ids(value, output)?;
            }
        },
        _ => {},
    }
    Ok(())
}

/// Sum of the per-device `active_attempts` ledger counters in a `GetAdmissionStats` payload.
pub(super) fn active_attempts(payload: &str) -> E2eResult<u64> {
    tsv_section(payload, "backend_device_admission")?.iter().try_fold(0u64, |total, row| {
        Ok(total.saturating_add(parse_u64_field(row, "active_attempts", "runtime observability wave")?))
    })
}

pub(super) fn millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

pub(super) fn scan_fatal_patterns(paths: &[PathBuf]) -> FatalPatternEvidence {
    let mut evidence = FatalPatternEvidence::default();
    for path in paths {
        let Ok(contents) = fs::read_to_string(path) else { continue };
        evidence.scanned_files += 1;
        for (line_index, line) in contents.lines().enumerate() {
            let lower = line.to_ascii_lowercase();
            let mut matched =
                FATAL_PATTERNS.iter().copied().filter(|pattern| lower.contains(pattern)).collect::<BTreeSet<_>>();
            if lower.contains("thread") && lower.contains("panicked") {
                matched.insert("thread .* panicked");
            }
            for pattern in matched {
                evidence.matches.push(json!({
                    "path": path,
                    "line": line_index + 1,
                    "pattern": pattern,
                    "excerpt": line.chars().take(500).collect::<String>(),
                }));
            }
        }
    }
    evidence
}

pub(super) fn hardware_evidence() -> Value {
    let query = Command::new("nvidia-smi")
        .args(["--query-gpu=index,name,uuid,memory.total,driver_version", "--format=csv,noheader,nounits"])
        .output();
    let (nvidia_smi_status, gpus, nvidia_smi_stderr) = match query {
        Ok(output) => (
            output.status.to_string(),
            String::from_utf8_lossy(&output.stdout)
                .lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .map(str::to_string)
                .collect::<Vec<_>>(),
            String::from_utf8_lossy(&output.stderr).trim().to_string(),
        ),
        Err(error) => ("spawn_failed".to_string(), Vec::new(), error.to_string()),
    };
    let kernel = Command::new("uname")
        .arg("-r")
        .output()
        .ok()
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string());
    json!({
        "nvidia_smi_status": nvidia_smi_status,
        "nvidia_smi_stderr": nvidia_smi_stderr,
        "gpus": gpus,
        "kernel": kernel,
        "cuda_visible_devices": std::env::var("CUDA_VISIBLE_DEVICES").ok(),
    })
}

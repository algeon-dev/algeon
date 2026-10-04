use std::{
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
    time::Duration,
};

use arrow_flight::FlightData;
use prost::Message as _;
use query_runtime::observability::{
    AttemptCapacityDisposition, AttemptCleanupDisposition, AttemptDisposition, AttemptId, AttemptRetryAdvice,
    RUNTIME_OBSERVATION_JSONL_SCHEMA, RuntimeTerminalOutcome, SEALED_ATTEMPT_PROJECTION_SCHEMA,
    SealedAttemptProjection,
};
use serde_json::{Value, json};

use super::{
    evidence::{
        ArtifactEvidence, CONTROL_SCHEMA, FLIGHT_MAX_ENCODED_FRAME_BYTES, GLOBAL_BYTE_PRESSURE_MAX_QUERIES,
        GLOBAL_BYTE_PRESSURE_PER_QUERY_BYTES, GLOBAL_BYTE_PRESSURE_RETAINED_BYTES, GLOBAL_BYTE_PRESSURE_RETAINED_COUNT,
        GLOBAL_BYTE_PRESSURE_SETUP_QUERIES, RunEvidence, ScenarioEvidence, VariableFrameEvidence, active_attempts,
        artifact_charge, observe_encoded_frame, parse_artifact, validate_variable_frame_evidence,
    },
    params::{EVIDENCE_SCHEMA, ObservabilityLimits, RuntimeParams, ScenarioSelection},
    validation::diagnostics_are_idle,
};

const QUERY_ID: &str = "parser-fixture-query";
const TRACE_ID: &str = "parser-fixture-correlation";

/// A control frame in the one shape the server emits for every state.
fn control(state: &str, retryable: bool, reason: Option<&str>, attempt: Option<Value>) -> Value {
    let runtime_jsonl_follows = state == "sealed" && attempt.is_some();
    json!({
        "schema": CONTROL_SCHEMA,
        "query_id": QUERY_ID,
        "state": state,
        "retryable": retryable,
        "reason": reason,
        "incomplete": state == "unavailable",
        "runtime_jsonl_follows": runtime_jsonl_follows,
        "attempt": attempt,
    })
}

fn control_frame(control: &Value) -> Vec<u8> {
    serde_json::to_vec(control).unwrap()
}

fn attempt_projection() -> Value {
    serde_json::to_value(SealedAttemptProjection {
        schema: SEALED_ATTEMPT_PROJECTION_SCHEMA,
        attempt_id: AttemptId::mint(),
        disposition: AttemptDisposition::Native,
        outcome: RuntimeTerminalOutcome::Succeeded,
        cleanup_disposition: AttemptCleanupDisposition::Completed,
        cleanup_failure_count: 0,
        cleanup_failure_count_complete: true,
        capacity_disposition: AttemptCapacityDisposition::Released,
        health_transition: None,
        retry_advice: AttemptRetryAdvice::no_retry_needed(),
        observation_truncated: false,
        observation_degraded: false,
        native_fault_evidence_present: false,
        resource_evidence_present: false,
    })
    .unwrap()
}

/// The sealed control of an attempt-owned statement, which a runtime JSONL
/// frame follows.
fn sealed_control_frame() -> Vec<u8> {
    control_frame(&control("sealed", false, None, Some(attempt_projection())))
}

fn manifest() -> Value {
    json!({
        "record_type": "manifest",
        "data": {
            "schema": RUNTIME_OBSERVATION_JSONL_SCHEMA,
            "query_id": QUERY_ID,
            "trace_id": TRACE_ID,
            "limits": {
                "max_retained_records": 16,
                "max_retained_bytes": 8192,
            }
        }
    })
}

fn footer(
    retained_records: u64,
    retained_bytes: u64,
    truncated: bool,
    first_dropped_sequence: Option<u64>,
    last_attempted_sequence: Option<u64>,
    dropped_by_class: Vec<Value>,
    dropped_by_reason: Vec<Value>,
) -> Value {
    json!({
        "record_type": "footer",
        "data": {
            "complete": true,
            "terminal": {"kind": "succeeded"},
            "retained_records": retained_records,
            "retained_bytes": retained_bytes,
            "truncated": truncated,
            "degraded": false,
            "poison_recoveries": 0,
            "first_dropped_sequence": first_dropped_sequence,
            "last_attempted_sequence": last_attempted_sequence,
            "dropped_by_class": dropped_by_class,
            "dropped_by_reason": dropped_by_reason,
        }
    })
}

fn sealed_frames(middle: Vec<Value>, footer: Value) -> Vec<Vec<u8>> {
    let mut records = vec![manifest()];
    records.extend(middle);
    records.push(footer);
    let mut jsonl = records
        .into_iter()
        .map(|record| serde_json::to_string(&record).unwrap())
        .collect::<Vec<_>>()
        .join("\n")
        .into_bytes();
    jsonl.push(b'\n');
    vec![sealed_control_frame(), jsonl]
}

fn canonical_middle() -> Vec<Value> {
    vec![
        json!({
            "record_type": "plan_node",
            "data": {"sequence": 2, "plan_execution_id": "exec-1"}
        }),
        json!({"record_type": "task_definition", "data": {"sequence": 1}}),
        json!({"record_type": "timeline_event", "data": {"sequence": 3}}),
        json!({"record_type": "timeline_event", "data": {"sequence": 4}}),
    ]
}

#[test]
fn sealed_two_frame_fixture_accepts_canonical_topology_independent_of_topology_sequence() {
    let frames = sealed_frames(canonical_middle(), footer(4, 512, false, None, Some(4), vec![], vec![]));

    let artifact = parse_artifact(QUERY_ID, Some(TRACE_ID), &frames, Some("succeeded"), None).unwrap();

    assert_eq!(artifact.state, "sealed");
    assert_eq!(artifact.frame_count, 2);
    assert_eq!(artifact.jsonl_records, 6);
    assert_eq!(artifact.topology_records, 2);
    assert_eq!(artifact.timeline_events, 2);
    assert_eq!(artifact.plan_execution_ids, ["exec-1"]);
    assert_eq!(artifact.retained_records, Some(4));
    assert_eq!(artifact.truncated, Some(false));
}

#[test]
fn control_fixture_accepts_pending_active_and_unavailable_states() {
    for (state, retryable, reason) in [
        ("pending", true, None),
        ("active", true, None),
        ("unavailable", false, Some("capture_budget")),
        ("unavailable", false, Some("evicted")),
        ("unavailable", false, Some("global_budget")),
    ] {
        let frames = vec![control_frame(&control(state, retryable, reason, None))];
        let artifact = parse_artifact(QUERY_ID, None, &frames, None, None).unwrap();
        assert_eq!(artifact.state, state);
        assert_eq!(artifact.reason.as_deref(), reason);
        assert_eq!(artifact.frame_count, 1);
        assert_eq!(artifact.jsonl_records, 0);
    }
    // A capture-truncated attempt keeps its attempt projection without
    // exposing its JSONL frame.
    let truncated = control("unavailable", false, Some("capture_budget"), Some(attempt_projection()));
    let artifact = parse_artifact(QUERY_ID, None, &[control_frame(&truncated)], None, None).unwrap();
    assert_eq!(artifact.state, "unavailable");
    assert_eq!(artifact.frame_count, 1);
}

#[test]
fn a_statement_without_runtime_observation_seals_in_one_control_frame() {
    let sealed = control("sealed", false, None, None);
    let frames = vec![control_frame(&sealed)];

    let artifact = parse_artifact(QUERY_ID, None, &frames, None, None).unwrap();
    assert_eq!(artifact.state, "sealed");
    assert_eq!(artifact.frame_count, 1);
    assert_eq!(artifact.jsonl_records, 0);
    assert!(
        parse_artifact(QUERY_ID, None, &frames, Some("succeeded"), None).is_err(),
        "a sealed terminal with a runtime observation was expected"
    );
    let mut announcing = sealed;
    announcing["runtime_jsonl_follows"] = json!(true);
    assert!(
        parse_artifact(QUERY_ID, None, &[control_frame(&announcing)], None, None).is_err(),
        "a control announcing a JSONL frame needs that frame"
    );
}

fn runtime_params_fixture() -> RuntimeParams {
    RuntimeParams {
        fixture_root: PathBuf::from("/fixture"),
        catalog_path: PathBuf::from("/catalog"),
        server_binary: PathBuf::from("/server"),
        device: 0,
        concurrency: 2,
        waves: 1,
        restart_iterations: 0,
        scenario_selection: ScenarioSelection::Matrix,
        evidence_path: PathBuf::from("/evidence.json"),
        server_log_path: PathBuf::from("/server.log"),
        connect_timeout: Duration::from_secs(1),
        query_timeout: Duration::from_secs(1),
        poll_timeout: Duration::from_secs(1),
        shutdown_timeout: Duration::from_secs(1),
        native_once: false,
        observability: ObservabilityLimits {
            per_query_records: 1,
            per_query_bytes: 1,
            retained_count: 2,
            retained_bytes: 2,
            ttl_secs: 1,
            shutdown_grace_secs: 1,
        },
    }
}

#[test]
fn evidence_pins_datafusion_baseline_payload_shape() {
    assert_eq!(EVIDENCE_SCHEMA, "algeon_datafusion.server_runtime_observability_e2e");
    const {
        assert!(GLOBAL_BYTE_PRESSURE_PER_QUERY_BYTES < GLOBAL_BYTE_PRESSURE_RETAINED_BYTES);
        assert!(
            GLOBAL_BYTE_PRESSURE_MAX_QUERIES + GLOBAL_BYTE_PRESSURE_SETUP_QUERIES < GLOBAL_BYTE_PRESSURE_RETAINED_COUNT
        );
    }
    assert_eq!(artifact_charge(38_000, 12_000).unwrap(), 50_000);

    let mut evidence = RunEvidence::new(&runtime_params_fixture());
    evidence.scenarios.push(ScenarioEvidence {
        scenario: "datafusion_baseline".to_string(),
        execution_path: "datafusion".to_string(),
        query_id: QUERY_ID.to_string(),
        correlation_id: TRACE_ID.to_string(),
        outcome: "succeeded".to_string(),
        output_batches: 1,
        output_rows: 1,
        encoded_frame_count: None,
        record_batch_chunks: None,
        max_encoded_frame_bytes: None,
        stats_assertions: Vec::new(),
        artifact: ArtifactEvidence {
            state: "sealed".to_string(),
            reason: None,
            frame_count: 0,
            wire_bytes: 0,
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
        },
        elapsed_ms: 0,
    });
    let payload = serde_json::to_value(evidence).unwrap();
    assert_eq!(payload["schema"], EVIDENCE_SCHEMA);
    let config = payload["config"].as_object().unwrap();
    assert_eq!(config.get("scenario_selection"), Some(&json!("matrix")));
    assert_eq!(config.get("native_once"), Some(&json!(false)));
    assert!(!config.contains_key("profile"));
    assert!(!config.contains_key("bounded_mode"));
    let scenario = payload["scenarios"][0].as_object().unwrap();
    assert_eq!(scenario.get("scenario"), Some(&json!("datafusion_baseline")));
    assert_eq!(scenario.get("execution_path"), Some(&json!("datafusion")));
    assert!(!scenario.contains_key("profile"));
}

#[test]
fn datafusion_baseline_is_the_only_non_native_scenario_selector() {
    assert_eq!(ScenarioSelection::parse("datafusion_baseline").unwrap().label(), "datafusion_baseline");
    for legacy_selector in ["cpu_fallback", "cpu-fallback"] {
        assert!(ScenarioSelection::parse(legacy_selector).is_err(), "{legacy_selector} must be rejected");
    }
}

#[test]
fn parser_rejects_noncanonical_topology_and_nonincreasing_timeline_sequence() {
    let noncanonical = vec![
        json!({"record_type": "task_definition", "data": {"sequence": 1}}),
        json!({"record_type": "plan_node", "data": {"sequence": 2}}),
    ];
    let error = parse_artifact(
        QUERY_ID,
        None,
        &sealed_frames(noncanonical, footer(2, 128, false, None, Some(2), vec![], vec![])),
        Some("succeeded"),
        None,
    )
    .unwrap_err();
    assert!(error.to_string().contains("outside canonical enum order"), "{error}");

    let timeline = vec![
        json!({"record_type": "timeline_event", "data": {"sequence": 3}}),
        json!({"record_type": "timeline_event", "data": {"sequence": 2}}),
    ];
    let error = parse_artifact(
        QUERY_ID,
        None,
        &sealed_frames(timeline, footer(2, 128, false, None, Some(2), vec![], vec![])),
        Some("succeeded"),
        None,
    )
    .unwrap_err();
    assert!(error.to_string().contains("timeline sequence is not strictly increasing"), "{error}");
}

#[test]
fn footer_loss_fixture_reconciles_and_rejects_inconsistent_reason_loss() {
    let middle = vec![
        json!({"record_type": "plan_node", "data": {"sequence": 1}}),
        json!({"record_type": "timeline_event", "data": {"sequence": 2}}),
    ];
    let valid_footer = footer(
        2,
        128,
        true,
        Some(3),
        Some(3),
        vec![json!({"class": "timeline", "count": 1})],
        vec![json!({"reason": "record_limit", "count": 1})],
    );
    let artifact =
        parse_artifact(QUERY_ID, None, &sealed_frames(middle.clone(), valid_footer), Some("succeeded"), None).unwrap();
    assert_eq!(artifact.truncated, Some(true));
    assert_eq!(artifact.dropped_by_class, Some(json!([{"class": "timeline", "count": 1}])));

    let invalid_footer = footer(2, 128, true, Some(3), Some(3), vec![json!({"class": "timeline", "count": 1})], vec![]);
    let error =
        parse_artifact(QUERY_ID, None, &sealed_frames(middle, invalid_footer), Some("succeeded"), None).unwrap_err();
    assert!(error.to_string().contains("reason loss 0 is below class loss 1"), "{error}");
}

#[test]
fn parser_rejects_malformed_multiple_sessions_and_oversized_plan_execution_id() {
    let malformed = vec![sealed_control_frame(), b"{not-json}\n".to_vec()];
    let error = parse_artifact(QUERY_ID, None, &malformed, Some("succeeded"), None).unwrap_err();
    assert!(error.to_string().contains("JSONL line 0 is invalid"), "{error}");

    let multiple_sessions = vec![json!({
        "record_type": "timeline_event",
        "data": {"sequence": 1, "diagnostic": "multiple_sessions"}
    })];
    let error = parse_artifact(
        QUERY_ID,
        None,
        &sealed_frames(multiple_sessions, footer(1, 64, false, None, Some(1), vec![], vec![])),
        Some("succeeded"),
        None,
    )
    .unwrap_err();
    assert!(error.to_string().contains("reported multiple_sessions"), "{error}");

    let oversized = vec![json!({
        "record_type": "plan_node",
        "data": {"sequence": 1, "plan_execution_id": "x".repeat(257)}
    })];
    let error = parse_artifact(
        QUERY_ID,
        None,
        &sealed_frames(oversized, footer(1, 64, false, None, Some(1), vec![], vec![])),
        Some("succeeded"),
        None,
    )
    .unwrap_err();
    assert!(error.to_string().contains("illegal plan_execution_id"), "{error}");
}

#[test]
fn variable_frame_evidence_aggregates_and_rejects_transport_oversize() {
    let count = AtomicUsize::new(0);
    let max = AtomicUsize::new(0);
    let frames =
        [8usize, 32, 16].map(|bytes| FlightData { data_body: vec![b'x'; bytes].into(), ..FlightData::default() });
    for frame in &frames {
        observe_encoded_frame(frame, &count, &max);
    }
    let evidence = VariableFrameEvidence {
        encoded_frame_count: count.load(Ordering::Relaxed),
        record_batch_chunks: 2,
        max_encoded_frame_bytes: max.load(Ordering::Relaxed),
    };
    assert_eq!(evidence.encoded_frame_count, 3);
    assert_eq!(evidence.max_encoded_frame_bytes, frames[1].encoded_len());
    validate_variable_frame_evidence(evidence).unwrap();

    let oversized = VariableFrameEvidence { max_encoded_frame_bytes: FLIGHT_MAX_ENCODED_FRAME_BYTES + 1, ..evidence };
    let error = validate_variable_frame_evidence(oversized).unwrap_err();
    assert!(error.to_string().contains("above the 4194304 byte limit"), "{error}");
}

fn admission_payload(rows: &[(usize, u64, u64)]) -> String {
    let mut payload = String::from(
        "# backend=query_service\n# device_ordinals=0,1\n# max_queued_attempts=64\n\n\
         [backend_metrics]\nbackend_generation\tgeneration\tconfigured_devices\n1\t9\t2\n\n\
         [backend_device_admission]\ndevice_ordinal\tledger_sequence\tactive_attempts\tlive_grant_bytes\tresident_owner_bytes\tmax_resident_owner_bytes\tcache_evictable_resident_bytes\tcache_pinned_in_use_bytes\thealth\n",
    );
    for (device, attempts, grant_bytes) in rows {
        payload.push_str(&format!("{device}\t4\t{attempts}\t{grant_bytes}\t0\t0\t0\t0\tOpen\n"));
    }
    payload
}

#[test]
fn admission_sections_drive_idle_and_overlap_checks() {
    let idle = admission_payload(&[(0, 0, 0), (1, 0, 0)]);
    assert!(diagnostics_are_idle(&idle).unwrap());
    assert_eq!(active_attempts(&idle).unwrap(), 0);

    let overlapping = admission_payload(&[(0, 1, 4096), (1, 2, 8192)]);
    assert!(!diagnostics_are_idle(&overlapping).unwrap());
    assert_eq!(active_attempts(&overlapping).unwrap(), 3);

    let grant_outlives_attempt = admission_payload(&[(0, 0, 4096), (1, 0, 0)]);
    assert!(!diagnostics_are_idle(&grant_outlives_attempt).unwrap());
}

use std::{collections::BTreeMap, path::PathBuf, time::Duration};

use super::{
    arrival::{Phase, build_trace},
    calibration::{CalibrationIdentity, CalibrationRow, CalibrationStore},
    cap::CapAssignment,
    config::{ArrivalProcess, GoodputConfig, Mode, PoolState},
    record::{AttemptRecord, OutcomeClass, WindowSummary, classify, percentile},
    report::{REQUEST_MANIFEST_COLUMNS, request_manifest_tsv},
    workload::{MixEntry, WorkloadFamily, parse_mix},
};

fn test_config() -> GoodputConfig {
    GoodputConfig {
        mode: Mode::Measure,
        mix: vec![
            MixEntry { item_id: "tpch.q06".to_string(), weight: 3 },
            MixEntry { item_id: "tpch.q18".to_string(), weight: 1 },
        ],
        tpch_root: None,
        tpcds_root: None,
        graph_edges: None,
        tpch_scale: "sf100".to_string(),
        tpcds_scale: "sf100".to_string(),
        graph_dataset: "none".to_string(),
        arrival: ArrivalProcess::Poisson,
        rate_qps: Some(4.0),
        rate_multiplier: 1.0,
        wave_size: 0,
        warmup_seconds: 10.0,
        window_seconds: 20.0,
        seed: 42,
        device_ordinals: vec![0],
        device_capacity_bytes: 1024,
        query_cap_bytes: 512,
        request_cap_manifest: None,
        cap_assignment: CapAssignment::default(),
        max_active_attempts: 2,
        max_queued_attempts: 32,
        max_queue_wait: Duration::from_secs(10),
        max_source_chunk_bytes: 1024,
        max_row_groups_per_chunk: 1,
        slo_path: PathBuf::from("unused"),
        slo_multiplier: 3.0,
        slo_sensitivity_multipliers: vec![1.5, 6.0],
        pool_state: PoolState::Hot,
        calibration_reps: 5,
        calibration_warmup_reps: 1,
        query_timeout: Duration::from_secs(60),
        fingerprint_max_rows: 1000,
        sample_interval: Duration::from_millis(250),
        out_dir: PathBuf::from("target/goodput"),
        run_id: "test".to_string(),
        group: "test".to_string(),
        source_identity: "test".to_string(),
    }
}

fn record(outcome: OutcomeClass, slo_met: bool, work_units: u64, phase: Phase) -> AttemptRecord {
    AttemptRecord {
        sequence: 0,
        phase,
        item_id: "tpch.q06".to_string(),
        family: WorkloadFamily::Tpch,
        work_units,
        requested_query_cap_bytes: 512,
        scheduled_offset_ms: 0.0,
        submit_offset_ms: 0.0,
        submit_delay_ms: 0.0,
        end_to_end_ms: 100.0,
        queue_wait_ms: Some(1.0),
        planning_ms: Some(1.0),
        execution_ms: Some(90.0),
        cleanup_ms: Some(1.0),
        attempt_total_ms: Some(100.0),
        device_ordinal: Some(0),
        grant_bytes: Some(512),
        disposition: "native".to_string(),
        terminal_outcome: "succeeded".to_string(),
        outcome,
        attempt_peak_aligned_bytes: Some(256),
        attempt_terminal_usage_provenance: "sampled_at_terminal_boundary".to_string(),
        attempt_pool_reserved_high_bytes: Some(1024),
        unattributed_allocator_provenance: "unavailable".to_string(),
        cleanup_disposition: "completed".to_string(),
        capacity_disposition: "released".to_string(),
        failure_code: None,
        failure_status: None,
        failure_operation: None,
        failure_reason: None,
        output_rows: Some(1),
        output_fingerprint: Some("output-fnv1a64:0".to_string()),
        fingerprint_match: Some(true),
        slo_deadline_ms: Some(300.0),
        slo_met,
        harness_timeout: false,
        error_message: None,
    }
}

#[test]
fn mix_parsing_reads_weights_and_rejects_repeated_items() {
    let mix = parse_mix("tpch.q06@3, tpch.q18 ,tpcds.q07@2").unwrap();
    assert_eq!(mix.len(), 3);
    assert_eq!(mix[0].weight, 3);
    assert_eq!(mix[1].weight, 1);
    assert_eq!(mix[2].item_id, "tpcds.q07");

    assert!(parse_mix("tpch.q06,tpch.q06").is_err(), "a repeated mix item must be rejected");
    assert!(parse_mix("tpch.q06@0").is_err(), "a zero weight must be rejected");
    assert!(parse_mix("").is_err(), "an empty mix must be rejected");
}

#[test]
fn a_physically_exhausted_device_is_not_classified_as_a_query_cap_failure() {
    assert_eq!(
        classify(Some("failed"), Some("device_out_of_memory"), None, None, None),
        OutcomeClass::FailedPhysicalOom
    );
    assert_eq!(
        classify(Some("failed"), Some("query_memory_limit_exceeded"), None, None, None),
        OutcomeClass::FailedQueryCap
    );
    assert_eq!(
        classify(Some("failed"), Some("device_allocation_failed"), None, None, None),
        OutcomeClass::FailedUnknownAllocation
    );
}

#[test]
fn queue_rejection_is_split_by_its_admission_reason() {
    assert_eq!(
        classify(Some("failed"), Some("service_overloaded"), Some("admission_queue_full"), None, None),
        OutcomeClass::RejectedQueueFull
    );
    assert_eq!(
        classify(Some("failed"), Some("service_overloaded"), Some("queue_wait_elapsed"), None, None),
        OutcomeClass::RejectedQueueTimeout
    );
}

#[test]
fn a_terminal_success_with_a_different_output_is_not_a_success() {
    assert_eq!(classify(Some("succeeded"), None, None, None, Some(true)), OutcomeClass::Succeeded);
    assert_eq!(classify(Some("succeeded"), None, None, None, Some(false)), OutcomeClass::OutputMismatch);
    assert_eq!(classify(Some("succeeded"), None, None, None, None), OutcomeClass::Succeeded);
}

#[test]
fn shed_and_failed_arrivals_stay_in_the_goodput_denominator() {
    let records = vec![
        record(OutcomeClass::Succeeded, true, 1, Phase::Window),
        record(OutcomeClass::Succeeded, false, 1, Phase::Window),
        record(OutcomeClass::RejectedQueueFull, false, 1, Phase::Window),
        record(OutcomeClass::FailedQueryCap, false, 1, Phase::Window),
        record(OutcomeClass::FailedUnsupported, false, 1, Phase::Window),
    ];
    let summary = WindowSummary::from_records(&records);

    assert_eq!(summary.offered, 5, "every window arrival is offered load");
    assert_eq!(summary.completed, 4, "only the queue rejection never reached execution");
    assert_eq!(summary.succeeded, 2);
    assert_eq!(summary.slo_met, 1);
    assert_eq!(summary.slo_missed, 1);
    assert!((summary.goodput_qps(10.0) - 0.1).abs() < f64::EPSILON, "only SLO-met successes are goodput");
    assert_eq!(summary.memory_failures, 1);
    assert_eq!(summary.unsupported, 1);
    assert!(
        (summary.memory_failure_ratio() - 0.2).abs() < f64::EPSILON,
        "queue rejections and unsupported plans must not inflate the memory failure ratio"
    );
}

#[test]
fn work_unit_goodput_credits_only_completed_work() {
    let records = vec![
        record(OutcomeClass::Succeeded, true, 256, Phase::Window),
        record(OutcomeClass::FailedQueryCap, false, 256, Phase::Window),
    ];
    let summary = WindowSummary::from_records(&records);

    assert_eq!(summary.work_units_offered, 512);
    assert_eq!(summary.work_units_slo_met, 256);
    assert!((summary.goodput_work_units_per_second(2.0) - 128.0).abs() < f64::EPSILON);
}

#[test]
fn percentiles_use_nearest_rank_over_the_sorted_sample() {
    let sorted = [1.0, 2.0, 3.0, 4.0];
    assert_eq!(percentile(&sorted, 0.50), Some(2.0));
    assert_eq!(percentile(&sorted, 0.95), Some(4.0));
    assert_eq!(percentile(&sorted, 0.0), Some(1.0));
    assert_eq!(percentile(&[], 0.5), None);
}

#[test]
fn the_arrival_trace_is_reproducible_and_splits_warmup_from_the_window() {
    let config = test_config();
    let weights = [3, 1];
    let first = build_trace(&config, &weights, 4.0);
    let second = build_trace(&config, &weights, 4.0);

    assert!(!first.is_empty());
    assert_eq!(first.len(), second.len(), "the same seed must produce the same trace length");
    for (left, right) in first.iter().zip(&second) {
        assert_eq!(left.offset, right.offset);
        assert_eq!(left.item_index, right.item_index);
        assert_eq!(left.phase, right.phase);
        assert_eq!(left.requested_query_cap_bytes, right.requested_query_cap_bytes);
    }
    assert!(
        first.iter().any(|arrival| arrival.phase == Phase::Warmup)
            && first.iter().any(|arrival| arrival.phase == Phase::Window),
        "a 10s warmup and 20s window must both receive arrivals"
    );
    assert!(
        first
            .iter()
            .all(|arrival| (arrival.phase == Phase::Warmup) == (arrival.offset.as_secs_f64() < config.warmup_seconds)),
        "phase must follow the warmup boundary"
    );
    assert!(
        first.iter().all(|arrival| arrival.requested_query_cap_bytes == config.query_cap_bytes),
        "every paired arrival must carry its explicit query cap"
    );
    let mut different_seed = test_config();
    different_seed.seed = 43;
    let other = build_trace(&different_seed, &weights, 4.0);
    assert!(
        other.len() != first.len() || other.iter().zip(&first).any(|(left, right)| left.offset != right.offset),
        "a different seed must produce a different trace"
    );
}

#[test]
fn request_cap_assignments_are_reproducible_and_use_specificity_order() {
    let assignment = CapAssignment::parse(
        "# schema=algeon-goodput-cap-assignment-v1\nkind\tkey\tcap_bytes\ndefault\t\t128\nfamily\ttpch\t256\nitem\ttpch.q06\t512\nsequence\t9\t1024\n",
    )
    .unwrap();

    assert_eq!(assignment.cap_for(9, "tpch.q06", 64), 1024, "sequence overrides item and family");
    assert_eq!(assignment.cap_for(8, "tpch.q06", 64), 512, "item overrides family");
    assert_eq!(assignment.cap_for(8, "tpch.q18", 64), 256, "family overrides default");
    assert_eq!(assignment.cap_for(8, "tpcds.q01", 64), 128, "default covers an unassigned family");
    assert_eq!(assignment.cap_for(8, "unknown.item", 64), 128, "default covers an unqualified item");
    assert_eq!(assignment.descriptor(), "default:128,family=tpch:256,item=tpch.q06:512,sequence=9:1024");
}

#[test]
fn one_arrival_trace_can_carry_multiple_explicit_caps() {
    let mut config = test_config();
    config.cap_assignment = CapAssignment::parse(
        "# schema=algeon-goodput-cap-assignment-v1\nkind\tkey\tcap_bytes\ndefault\t\t768\nsequence\t0\t256\n",
    )
    .unwrap();

    let first = build_trace(&config, &[3, 1], 4.0);
    let second = build_trace(&config, &[3, 1], 4.0);
    assert!(first.len() > 1);
    assert_eq!(first[0].requested_query_cap_bytes, 256);
    assert!(first.iter().any(|arrival| arrival.requested_query_cap_bytes == 768));
    assert_eq!(
        first.iter().map(|arrival| arrival.requested_query_cap_bytes).collect::<Vec<_>>(),
        second.iter().map(|arrival| arrival.requested_query_cap_bytes).collect::<Vec<_>>(),
        "the same seed and assignment must reproduce request caps"
    );
}

#[test]
fn sequence_only_assignments_cannot_silently_choose_a_calibration_cap() {
    let assignment =
        CapAssignment::parse("# schema=algeon-goodput-cap-assignment-v1\nkind\tkey\tcap_bytes\nsequence\t0\t256\n")
            .unwrap();
    let item = super::workload::resolve_item("tpch.q06", 0).unwrap();
    assert!(assignment.calibration_caps(&[item], 512).is_err());
}

#[test]
fn request_manifest_joins_configured_cap_to_the_observed_grant() {
    let config = test_config();
    let mut admitted = record(OutcomeClass::Succeeded, true, 1, Phase::Window);
    admitted.sequence = 7;
    admitted.scheduled_offset_ms = 125.0;
    let mut rejected = record(OutcomeClass::RejectedQueueFull, false, 1, Phase::Window);
    rejected.sequence = 8;
    rejected.grant_bytes = None;

    let manifest = request_manifest_tsv(&config, &[admitted, rejected]);
    let mut lines = manifest.lines();
    let header = REQUEST_MANIFEST_COLUMNS.join("\t");
    assert_eq!(lines.next(), Some(header.as_str()));
    assert_eq!(lines.next(), Some("7\twindow\tgoodput-test-7\ttpch.q06\ttpch\t125.000\t512\t512\ttrue\tsucceeded"));
    assert_eq!(
        lines.next(),
        Some("8\twindow\tgoodput-test-8\ttpch.q06\ttpch\t0.000\t512\tunavailable\tunavailable\trejected_queue_full")
    );
    assert!(lines.next().is_none());

    let mut mismatched = record(OutcomeClass::Succeeded, true, 1, Phase::Window);
    mismatched.requested_query_cap_bytes = 256;
    let mismatch = request_manifest_tsv(&config, &[mismatched]);
    assert!(mismatch.lines().nth(1).is_some_and(|line| line.ends_with("\tfalse\tsucceeded")));
}

#[test]
fn a_wave_trace_releases_every_request_of_a_wave_at_one_offset() {
    let mut config = test_config();
    config.arrival = ArrivalProcess::Wave;
    config.wave_size = 8;
    let trace = build_trace(&config, &[3, 1], 4.0);

    assert!(!trace.is_empty());
    assert_eq!(trace.len() % config.wave_size, 0, "waves are released whole");
    for wave in trace.chunks(config.wave_size) {
        assert!(wave.iter().all(|arrival| arrival.offset == wave[0].offset), "one wave shares one release offset");
    }
}

fn calibration_store() -> CalibrationStore {
    let mut header = BTreeMap::new();
    header.insert("gpu_model".to_string(), "TEST GPU".to_string());
    header.insert("tpch_fingerprint".to_string(), "tpch-abc".to_string());
    header.insert("tpcds_fingerprint".to_string(), "tpcds-def".to_string());
    header.insert("graph_fingerprint".to_string(), String::new());
    header.insert("calibration_caps".to_string(), "tpch.q06=512".to_string());
    CalibrationStore {
        identity: CalibrationIdentity {
            gpu_model: "TEST GPU".to_string(),
            tpch_fingerprint: "tpch-abc".to_string(),
            tpcds_fingerprint: "tpcds-def".to_string(),
            graph_fingerprint: String::new(),
        },
        header,
        rows: vec![CalibrationRow {
            item_id: "tpch.q06".to_string(),
            family: WorkloadFamily::Tpch,
            work_units: 1,
            reps: 5,
            p50_ms: 120.5,
            p95_ms: 140.25,
            attempt_peak_aligned_bytes_p50: Some(4096),
            attempt_peak_aligned_bytes_max: None,
            disposition: "native".to_string(),
            output_rows: 1,
            output_fingerprint: "output-fnv1a64:00000000deadbeef".to_string(),
        }],
    }
}

#[test]
fn the_calibration_store_round_trips_through_its_tsv() {
    let store = calibration_store();
    let parsed = CalibrationStore::parse(&store.to_tsv()).unwrap();

    assert_eq!(parsed.identity, store.identity);
    let row = parsed.row("tpch.q06").expect("the calibrated item survives the round trip");
    assert!((row.p50_ms - 120.5).abs() < f64::EPSILON);
    assert_eq!(row.attempt_peak_aligned_bytes_p50, Some(4096));
    assert_eq!(row.attempt_peak_aligned_bytes_max, None, "an unavailable peak must not become zero");
    assert_eq!(row.output_fingerprint, "output-fnv1a64:00000000deadbeef");
    assert_eq!(parsed.header.get("calibration_caps").map(String::as_str), Some("tpch.q06=512"));
}

#[test]
fn a_measurement_refuses_a_calibration_from_another_environment() {
    let store = calibration_store();
    let live = CalibrationIdentity {
        gpu_model: "TEST GPU".to_string(),
        tpch_fingerprint: "tpch-abc".to_string(),
        tpcds_fingerprint: "tpcds-def".to_string(),
        graph_fingerprint: "graph-xyz".to_string(),
    };

    store.assert_identity(&live, &[WorkloadFamily::Tpch, WorkloadFamily::Tpcds]).expect("matching families pass");
    assert!(
        store.assert_identity(&live, &[WorkloadFamily::Graph]).is_err(),
        "a family the calibration never exercised must fail closed"
    );

    let other_gpu = CalibrationIdentity { gpu_model: "OTHER GPU".to_string(), ..live.clone() };
    assert!(store.assert_identity(&other_gpu, &[WorkloadFamily::Tpch]).is_err(), "a different device must fail closed");

    let other_data = CalibrationIdentity { tpch_fingerprint: "tpch-zzz".to_string(), ..live };
    assert!(
        store.assert_identity(&other_data, &[WorkloadFamily::Tpch]).is_err(),
        "a different dataset must fail closed"
    );
}

#[test]
fn ad_hoc_items_resolve_and_graph_items_report_their_work_units() {
    let item = super::workload::resolve_item("adhoc.skew_join", 0).unwrap();
    assert_eq!(item.family, WorkloadFamily::Adhoc);
    assert!(item.sql.contains("JOIN orders"));
    assert_eq!(item.work_units, 1);

    assert!(super::workload::resolve_item("tpch.q06", 0).is_ok());
    assert!(super::workload::resolve_item("tpch.q99", 0).is_err());
    assert!(super::workload::resolve_item("nosuch.item", 0).is_err());
}

#[cfg(feature = "cugraph")]
#[test]
fn graph_items_carry_their_seed_count_as_work_units() {
    let batched = super::workload::resolve_item("graph.betweenness_k256", 7).unwrap();
    let split = super::workload::resolve_item("graph.betweenness_k64", 7).unwrap();

    assert_eq!(batched.work_units, 256);
    assert_eq!(split.work_units, 64);
    assert!(batched.sql.contains("k => 256"));

    let bfs = super::workload::resolve_item("graph.bfs", 7).unwrap();
    assert!(bfs.sql.contains("source_vertex => 7"), "the discovered seed must reach the traversal item: {}", bfs.sql);
}

/// Every vertex-keyed algorithm returns one row per vertex, so a fingerprint
/// taken over the row count alone is the same number whether the algorithm
/// computed anything or not, and goodput would count wrong results as correct.
#[cfg(feature = "cugraph")]
#[test]
fn graph_items_fingerprint_computed_values_and_not_only_row_counts() {
    for item_id in [
        "graph.pagerank",
        "graph.bfs",
        "graph.sssp",
        "graph.wcc",
        "graph.triangles",
        "graph.betweenness_k64",
        "graph.betweenness_k256",
    ] {
        let item = super::workload::resolve_item(item_id, 7).unwrap();
        assert!(
            item.sql.contains("SUM(") || item.sql.contains("COUNT(DISTINCT"),
            "{item_id} fingerprints only its cardinality, so a wrong result would still match: {}",
            item.sql
        );
    }
}

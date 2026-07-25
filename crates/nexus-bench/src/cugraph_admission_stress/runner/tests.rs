use arrow::{
    array::{ArrayRef, Int64Array},
    datatypes::{DataType, Field, Schema},
};
use datafusion_nexus::native_report::{NativeGraphReservationMetricRow, NativeGraphRuntimeMetricRow};

use super::*;

#[test]
fn output_fingerprint_compares_full_row_multisets() {
    let schema = Arc::new(Schema::new(vec![Field::new("vertex", DataType::Int64, false)]));
    let batch = |values: Vec<i64>| {
        RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int64Array::from(values)) as ArrayRef]).unwrap()
    };

    let ordered = fingerprint_batches(&schema, &[batch(vec![1, 2, 3])]).unwrap();
    let reordered = fingerprint_batches(&schema, &[batch(vec![3]), batch(vec![1, 2])]).unwrap();
    let changed = fingerprint_batches(&schema, &[batch(vec![1, 2, 4])]).unwrap();

    assert_eq!(ordered, reordered);
    assert_ne!(ordered, changed);
}

#[test]
fn query_tsv_row_contract_matches_columns() {
    let config = HarnessConfig {
        edges_parquet: "edges.parquet".into(),
        papers_parquet: None,
        fos_parquet: None,
        workload: vec![("pagerank".into(), 1)],
        iterations: 1,
        stagger_ms: 0,
        max_concurrent_queries: 2,
        slot_guarantee: SlotGuarantee::GuaranteedMinimum,
        grant_min: None,
        grant_want: None,
        wait_timeout_ms: 1000,
        budget_fraction: None,
        device_budget_bytes: None,
        requested_grant_min_bytes: None,
        requested_grant_bytes: None,
        raw_input_edges: None,
        sample_interval_ms: 100,
        query_timeout_ms: 1000,
        out_dir: PathBuf::from("."),
        run_id: "test".into(),
        bfs_seed: None,
        batch_size: None,
        device_ordinals: vec![0],
        source_cache_max_retained_bytes: Some(1024),
        graph_cache_max_retained_bytes: Some(2048),
        warmup_iterations: 0,
        cuda_profiler_capture: false,
    };
    let request = Request {
        task_index: 0,
        spec: QuerySpec { label: "pagerank".into(), sql: "SELECT 1".into(), needs_papers: false, needs_fos: false },
    };
    let row = QueryRow::new(&config, &request, 0, 0).finish_failed(Instant::now(), "x\ty".into());
    assert_eq!(row.to_tsv().split('\t').count(), QUERY_COLUMNS.len());
    assert_eq!(QUERY_COLUMNS.first(), Some(&"run_id"));
    assert!(QUERY_COLUMNS.contains(&"source_cache_hits"));
    assert!(QUERY_COLUMNS.contains(&"source_cache_retained_bytes"));
    let error_index = QUERY_COLUMNS.iter().position(|column| *column == "error").expect("error column is present");
    let source_cache_index = QUERY_COLUMNS
        .iter()
        .position(|column| *column == "source_cache_hits")
        .expect("source cache columns are present");
    assert_eq!(QUERY_COLUMNS[error_index.saturating_sub(1)], "native_lowered");
    assert!(
        source_cache_index > error_index,
        "new source-cache columns must append after the legacy query row columns"
    );
}

#[test]
fn legacy_sample_tsv_schema_and_order_are_unchanged() {
    assert_eq!(
        SAMPLE_COLUMNS,
        &[
            "epoch_ms",
            "governor_capacity_bytes",
            "governor_outstanding_bytes",
            "governor_available_bytes",
            "active_grants",
            "active_queries",
            "queued_queries",
            "pool_reserved_current_bytes",
            "pool_reserved_high_bytes",
            "pool_used_current_bytes",
            "pool_used_high_bytes",
            "allocation_failure_count",
            "trim_retry_count",
        ]
    );
    let row = SampleRow {
        epoch_ms: 1,
        governor_capacity_bytes: 2,
        governor_outstanding_bytes: 3,
        governor_available_bytes: 4,
        active_grants: 5,
        active_queries: 6,
        queued_queries: 7,
        pool_reserved_current_bytes: 8,
        pool_reserved_high_bytes: 9,
        pool_used_current_bytes: 10,
        pool_used_high_bytes: 11,
        allocation_failure_count: 12,
        trim_retry_count: 13,
    };
    assert_eq!(row.to_tsv().split('\t').count(), SAMPLE_COLUMNS.len());
}

#[test]
fn device_samples_cover_every_configured_ordinal_by_identity() {
    let facts = BTreeMap::from([
        (
            3,
            DeviceSampleFacts {
                governor_capacity_bytes: 300,
                pool_used_high_bytes: 30,
                gpu_execution_active_query_jobs: 3,
                ..Default::default()
            },
        ),
        (
            1,
            DeviceSampleFacts {
                governor_capacity_bytes: 100,
                pool_used_high_bytes: 10,
                gpu_execution_active_query_jobs: 1,
                ..Default::default()
            },
        ),
    ]);

    let rows = device_sample_rows(42, &[3, 1], &facts, 4, 2);

    assert_eq!(rows.iter().map(|row| row.device_ordinal).collect::<Vec<_>>(), vec![3, 1]);
    assert!(rows.iter().all(|row| row.epoch_ms == 42));
    assert!(rows.iter().all(|row| row.active_queries == 4 && row.queued_queries == 2));
    assert_eq!(rows[0].facts.governor_capacity_bytes, 300);
    assert_eq!(rows[1].facts.governor_capacity_bytes, 100);
    assert!(rows.iter().all(|row| row.to_tsv().split('\t').count() == DEVICE_SAMPLE_COLUMNS.len()));
}

#[test]
fn device_peaks_are_reported_separately_per_ordinal() {
    let facts = |pool_used_high_bytes, active_query_jobs| DeviceSampleFacts {
        pool_used_high_bytes,
        gpu_execution_active_query_jobs: active_query_jobs,
        ..Default::default()
    };
    let rows = vec![
        DeviceSampleRow { epoch_ms: 1, device_ordinal: 0, active_queries: 2, queued_queries: 0, facts: facts(10, 1) },
        DeviceSampleRow { epoch_ms: 2, device_ordinal: 0, active_queries: 2, queued_queries: 0, facts: facts(20, 2) },
        DeviceSampleRow { epoch_ms: 2, device_ordinal: 2, active_queries: 2, queued_queries: 0, facts: facts(40, 3) },
    ];

    let peaks = device_peaks(&[0, 2], &rows);

    assert_eq!(peaks.get("0").unwrap()["pool_used_high_bytes"], serde_json::json!(20));
    assert_eq!(peaks.get("2").unwrap()["gpu_execution_active_query_jobs_max"], serde_json::json!(3));
}

#[test]
fn query_row_absorbs_graph_calibration_fields() {
    let config = HarnessConfig {
        edges_parquet: "edges.parquet".into(),
        papers_parquet: None,
        fos_parquet: None,
        workload: vec![("pagerank".into(), 1)],
        iterations: 1,
        stagger_ms: 0,
        max_concurrent_queries: 2,
        slot_guarantee: SlotGuarantee::GuaranteedMinimum,
        grant_min: None,
        grant_want: None,
        wait_timeout_ms: 1000,
        budget_fraction: None,
        device_budget_bytes: None,
        requested_grant_min_bytes: None,
        requested_grant_bytes: None,
        raw_input_edges: None,
        sample_interval_ms: 100,
        query_timeout_ms: 1000,
        out_dir: PathBuf::from("."),
        run_id: "test".into(),
        bfs_seed: None,
        batch_size: None,
        device_ordinals: vec![0],
        source_cache_max_retained_bytes: None,
        graph_cache_max_retained_bytes: None,
        warmup_iterations: 0,
        cuda_profiler_capture: false,
    };
    let request = Request {
        task_index: 0,
        spec: QuerySpec { label: "pagerank".into(), sql: "SELECT 1".into(), needs_papers: false, needs_fos: false },
    };
    let mut row = QueryRow::new(&config, &request, 0, 0);
    row.absorb_metrics(&NativeDataFusionExecutionMetrics {
        source_cache_hits: 2,
        source_cache_misses: 1,
        source_cache_inserts: 1,
        source_cache_insert_skipped: 0,
        source_cache_retained_bytes: 8192,
        graph_reservation_rows: vec![NativeGraphReservationMetricRow {
            node_id: "g0".into(),
            algorithm: "triangle_count_all".into(),
            class: "workspace_heavy_data_dependent".into(),
            outcome: "admitted".into(),
            raw_input_edges: Some(45),
            input_edges: 90,
            vertex_upper_bound: 180,
            vertex_estimate: 12,
            vertex_estimate_source: "runtime_exact_distinct".into(),
            vertex_width_bytes: 8,
            weight_width_bytes: Some(8),
            pair_rows: None,
            pair_vertex_width_bytes: None,
            pair_cardinality_source: "none".into(),
            pair_cardinality_admission_mode: "none".into(),
            has_edge_id: false,
            directed: false,
            renumber: true,
            construction_policy: "raw_libcugraph".into(),
            edge_normalize_inserted: true,
            calibration_source: "raw_input_edges".into(),
            seed_count: None,
            seed_mode: "none".into(),
            data_dependent: true,
            model_requested_bytes: 2048,
            requested_bytes: 4096,
            budget_bytes: Some(8192),
            graph_model_error_ratio: Some("2.000".into()),
            measured_peak_bytes: Some(1024),
            graph_statistics_exact_vertices: Some(12),
            graph_statistics_degree_max: Some(8),
            graph_statistics_degree_p99: Some(8),
            graph_statistics_sum_min_degree_oriented: Some(123),
            graph_statistics_oriented_edge_count: Some(45),
            graph_statistics_peak_chunk_min_degree_sum: Some(45),
            graph_statistics_sum_out_degree: Some(90),
            reason: "none".into(),
        }],
        graph_runtime_rows: vec![NativeGraphRuntimeMetricRow {
            algorithm: "triangle_count_all".into(),
            elapsed_ns: 1_000_000,
            graph_peak_bytes: Some(1024),
            algorithm_metadata: "iterations=1,graph_cache=hit".into(),
            ..Default::default()
        }],
        ..Default::default()
    });

    let tsv = row.to_tsv();
    let fields = tsv.split('\t').collect::<Vec<_>>();
    let field = |name: &str| fields[QUERY_COLUMNS.iter().position(|column| *column == name).unwrap()];
    assert_eq!(field("source_cache_hits"), "2");
    assert_eq!(field("source_cache_misses"), "1");
    assert_eq!(field("source_cache_inserts"), "1");
    assert_eq!(field("source_cache_insert_skipped"), "0");
    assert_eq!(field("source_cache_retained_bytes"), "8192");
    assert_eq!(field("graph_raw_input_edges"), "45");
    assert_eq!(field("graph_vertex_width_bytes"), "8");
    assert_eq!(field("graph_weight_width_bytes"), "8");
    assert_eq!(field("graph_directed"), "false");
    assert_eq!(field("graph_edge_normalize_inserted"), "true");
    assert_eq!(field("graph_calibration_source"), "raw_input_edges");
    assert_eq!(field("graph_seed_mode"), "none");
    assert_eq!(field("graph_model_requested_bytes"), "2048");
    assert_eq!(field("graph_model_error_ratio"), "2.000");
    assert_eq!(field("graph_statistics_exact_vertices"), "12");
    assert_eq!(field("graph_statistics_degree_max"), "8");
    assert_eq!(field("graph_statistics_degree_p99"), "8");
    assert_eq!(field("graph_statistics_sum_min_degree_oriented"), "123");
    assert_eq!(field("graph_statistics_oriented_edge_count"), "45");
    assert_eq!(field("graph_statistics_peak_chunk_min_degree_sum"), "45");
    assert_eq!(field("graph_statistics_sum_out_degree"), "90");
    assert_eq!(field("graph_peak_bytes"), "1024");
    assert_eq!(field("graph_cache"), "hit");
}

#[test]
fn workload_parsing_accepts_counts_and_bare_labels() {
    let parsed = parse_workload("pagerank:4, louvain:2 ,bfs").unwrap();
    assert_eq!(parsed, vec![("pagerank".into(), 4), ("louvain".into(), 2), ("bfs".into(), 1)]);
}

#[test]
fn graph_cache_metadata_value_extracts_optional_token() {
    assert_eq!(graph_cache_metadata_value("graph_cache=miss_inserted"), Some("miss_inserted".into()));
    assert_eq!(
        graph_cache_metadata_value("depth_limit=3,graph_cache=hit;retained_device_bytes=1024,null_edge_rows_dropped=0"),
        Some("hit;retained_device_bytes=1024".into())
    );
    assert_eq!(graph_cache_metadata_value("depth_limit=3"), None);
    assert_eq!(graph_cache_metadata_value("graph_cache="), None);
}

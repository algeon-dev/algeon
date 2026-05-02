use std::{
    fs::{self, File},
    path::Path,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use arrow::{
    array::Int32Array,
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use cudf_query_engine::{
    AggregateExpr, AggregateFunction, AggregateSpec, BinaryOp, BoundaryMetrics, DataPlacement, ExecutionContext, Expr,
    LimitSpec, LiteralValue, LocalParquetFile, LocalParquetSource, MemoryControlPolicy, MemoryReservationRequest,
    PlanNode, PredicateDisposition, QueryError, QueryErrorCode, QueryErrorKind, QueryErrorStatus, QueryMetrics,
    QueryPlanBuilder, ReservationKind, ResultSink, RetryFacts, SchemaBehavior, SortExpr, SortSpec,
    SourceChunkPlanningPolicy, SourceNode, plan_local_parquet_source_chunks, validate_local_parquet_source,
};
use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};

#[test]
fn native_plan_can_represent_proven_benchmark_shape() {
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt32, false),
        Field::new("payload", DataType::Int64, false),
        Field::new("m00", DataType::Float64, false),
        Field::new("m01", DataType::Float64, false),
    ]));
    let projected_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt32, false),
        Field::new("x00", DataType::Float64, false),
    ]));
    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt32, false),
        Field::new("sum_x00", DataType::Float64, true),
    ]));

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new("/tmp/phase4.parquet").with_file_size_bytes(3_235_196_281)],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "payload".to_string(), "m00".to_string(), "m01".to_string()],
    );
    let filter = Expr::binary(Expr::column("payload"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int64(10)));
    let projection =
        vec![Expr::column("key"), Expr::binary(Expr::column("m00"), BinaryOp::Add, Expr::column("m01")).alias("x00")];
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("x00"), "sum_x00")],
    );
    let sort = SortSpec::new(vec![
        SortExpr::new(Expr::column("sum_x00"), true, false),
        SortExpr::new(Expr::column("key"), false, true),
    ]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let filter = builder.add_filter(source, filter, Arc::clone(&input_schema));
    let projection = builder.add_projection(filter, projection, projected_schema);
    let aggregate = builder.add_aggregate(projection, aggregate, Arc::clone(&aggregated_schema));
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&aggregated_schema));
    let limit = builder.add_limit(sort, LimitSpec::fetch(100), Arc::clone(&aggregated_schema));
    let plan = builder.finish_with_host_sink(limit);

    assert_eq!(plan.node_count(), 6);
    assert_eq!(plan.output_schema().fields().len(), 2);
    assert_eq!(plan.root_id(), limit);
    assert!(matches!(plan.sink(), ResultSink::HostArrow { input, .. } if *input == limit));
    assert!(plan.sink().materializes_host_boundary());
    assert!(plan.nodes().iter().all(|node| node.facts.placement == DataPlacement::Device));
    assert_eq!(plan.node(source).unwrap().facts.schema_behavior, SchemaBehavior::SourceOutput);
    assert_eq!(plan.node(filter).unwrap().facts.schema_behavior, SchemaBehavior::PreservesInput);
    assert_eq!(plan.node(projection).unwrap().facts.schema_behavior, SchemaBehavior::Projection);
    assert_eq!(plan.node(aggregate).unwrap().facts.schema_behavior, SchemaBehavior::Aggregate);
    assert_eq!(plan.node(sort).unwrap().facts.output_ordering.len(), 2);
    assert_eq!(plan.node(limit).unwrap().facts.output_ordering.len(), 2);
    assert!(matches!(plan.root_node().kind, PlanNode::Limit(_)));
    assert!(plan.to_string().contains("LocalParquetSource"));
    assert!(plan.to_string().contains("Limit input=n4 skip=0 fetch=100"));
    assert!(plan.to_string().contains("Sink id=sink input=n5 mode=host_arrow materialization=device_to_host"));
}

#[test]
fn query_error_codes_and_facts_are_stable_without_datafusion() {
    let error = QueryError::new(
        QueryErrorCode::UnsupportedExpression,
        "cudf_query_engine.plan.validate",
        "expression is not supported by native cudf query engine",
    )
    .with_fact("expression", "Utf8 LIKE")
    .with_fact("reason", "unsupported_expression");

    assert_eq!(error.code(), QueryErrorCode::UnsupportedExpression);
    assert_eq!(error.code().as_str(), "unsupported_expression");
    assert_eq!(error.kind(), QueryErrorKind::Unsupported);
    assert_eq!(error.status(), QueryErrorStatus::Permanent);
    assert_eq!(error.fact_value("reason"), Some("unsupported_expression"));
    assert!(error.to_string().contains("code=unsupported_expression"));
    assert!(error.to_string().contains("expression=\"Utf8 LIKE\""));
}

#[test]
fn memory_reservation_failure_is_temporary_resource_exhaustion() {
    let request = MemoryReservationRequest::new(ReservationKind::SourceRead, 1024, 512)
        .with_memory_boundary("source_chunk_read")
        .with_node_id("n0")
        .with_stage("source_read")
        .with_chunk_index(3)
        .with_file_index(1)
        .with_path("/tmp/input.parquet")
        .with_row_groups(vec![4, 5])
        .with_attempt(2, 4);

    let error = request.denied_error(256);

    assert_eq!(error.code(), QueryErrorCode::MemoryReservationFailed);
    assert_eq!(error.code().as_str(), "memory_reservation_failed");
    assert_eq!(error.kind(), QueryErrorKind::ResourceExhausted);
    assert_eq!(error.status(), QueryErrorStatus::Temporary);
    assert_eq!(error.fact_value("memory_boundary"), Some("source_chunk_read"));
    assert_eq!(error.fact_value("reservation_kind"), Some("source_read"));
    assert_eq!(error.fact_value("requested_bytes"), Some("1024"));
    assert_eq!(error.fact_value("granted_bytes"), Some("256"));
    assert_eq!(error.fact_value("budget_bytes"), Some("512"));
    assert_eq!(error.fact_value("node_id"), Some("n0"));
    assert_eq!(error.fact_value("chunk_index"), Some("3"));
    assert_eq!(error.fact_value("file_index"), Some("1"));
    assert_eq!(error.fact_value("path"), Some("/tmp/input.parquet"));
    assert_eq!(error.fact_value("row_groups"), Some("4,5"));
    assert_eq!(error.fact_value("attempt"), Some("2"));
    assert_eq!(error.fact_value("max_attempts"), Some("4"));
}

#[test]
fn memory_control_policy_names_bounded_and_unbounded_modes() {
    let unbounded = MemoryControlPolicy::Unbounded;
    let bounded = MemoryControlPolicy::Bounded {
        device_budget_bytes: 4 * 1024 * 1024 * 1024,
        max_source_chunk_bytes: 512 * 1024 * 1024,
        max_row_groups_per_chunk: 2,
        max_retry_attempts: 3,
        min_retry_chunk_row_groups: 1,
    };

    assert_eq!(unbounded.as_str(), "unbounded");
    assert!(!unbounded.is_bounded());
    assert_eq!(bounded.as_str(), "bounded");
    assert!(bounded.is_bounded());
    assert_eq!(ReservationKind::FinalMerge.as_str(), "final_merge");

    let chunk_policy = SourceChunkPlanningPolicy::from(bounded);
    assert_eq!(chunk_policy.max_source_chunk_bytes, Some(512 * 1024 * 1024));
    assert_eq!(chunk_policy.max_row_groups_per_chunk, 2);
}

#[test]
fn bounded_memory_control_policy_rejects_zero_fields_before_runtime_initialization() {
    let error = ExecutionContext::builder()
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 0,
            max_source_chunk_bytes: 512 * 1024 * 1024,
            max_row_groups_per_chunk: 2,
            max_retry_attempts: 3,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap_err();

    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.operation(), "cudf_query_engine.memory_policy.validate");
    assert_eq!(error.fact_value("policy"), Some("bounded"));
    assert_eq!(error.fact_value("field"), Some("device_budget_bytes"));
    assert_eq!(error.fact_value("reason"), Some("zero"));
}

#[test]
fn source_chunk_planning_policy_rejects_zero_row_group_limit() {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new("/tmp/part-000.parquet").with_file_size_bytes(300).with_row_groups(vec![0])],
        Arc::clone(&schema),
        vec!["key".to_string()],
    );
    let mut builder = QueryPlanBuilder::new();
    let source_id = builder.add_source(SourceNode::LocalParquet(source.clone()));

    let error = plan_local_parquet_source_chunks(source_id, &source, SourceChunkPlanningPolicy::new(Some(1024), 0))
        .unwrap_err();

    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.operation(), "cudf_query_engine.source_chunk.policy");
    assert_eq!(error.fact_value("field"), Some("max_row_groups_per_chunk"));
    assert_eq!(error.fact_value("reason"), Some("zero"));
}

#[test]
fn retry_facts_attach_stable_resume_and_merge_context() {
    let request = MemoryReservationRequest::new(ReservationKind::FinalMerge, 2048, 1024);
    let error = RetryFacts::new()
        .with_retry_action("reduce_merge_fan_in")
        .with_resume_boundary("final_merge")
        .with_committed_chunks(8)
        .with_committed_partials(8)
        .with_merge_level(1)
        .with_merge_fan_in(2)
        .attach_to_error(request.denied_error(1024));

    assert_eq!(error.fact_value("retry_action"), Some("reduce_merge_fan_in"));
    assert_eq!(error.fact_value("resume_boundary"), Some("final_merge"));
    assert_eq!(error.fact_value("committed_chunks"), Some("8"));
    assert_eq!(error.fact_value("committed_partials"), Some("8"));
    assert_eq!(error.fact_value("merge_level"), Some("1"));
    assert_eq!(error.fact_value("merge_fan_in"), Some("2"));
}

#[test]
fn query_metrics_make_host_device_boundaries_explicit() {
    let mut metrics = QueryMetrics::default();
    metrics.record_parquet_source(Some(3_235_196_281), 12, 34);
    metrics.boundary.record_device_to_host_materialization(100, 8192);

    assert_eq!(metrics.parquet_files, 1);
    assert_eq!(metrics.parquet_bytes_known, 3_235_196_281);
    assert_eq!(metrics.row_groups_selected, 12);
    assert_eq!(metrics.read_columns, 34);
    assert_eq!(metrics.boundary.host_to_device_uploads, 0);
    assert_eq!(metrics.boundary.device_to_host_materializations, 1);
    assert_eq!(metrics.boundary.device_to_host_materialized_rows, 100);

    let mut boundary = BoundaryMetrics::default();
    boundary.record_host_to_device_upload(10, 1024);
    assert_eq!(boundary.host_to_device_uploads, 1);
    assert_eq!(boundary.host_to_device_uploaded_bytes, 1024);
}

#[test]
fn local_parquet_source_tracks_direct_source_diagnostics() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt32, false),
        Field::new("payload", DataType::Int64, false),
    ]));
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/phase4.parquet")
                .with_file_size_bytes(3_235_196_281)
                .with_row_groups(vec![0, 2, 4]),
        ],
        schema,
        vec!["key".to_string(), "payload".to_string()],
    );

    validate_local_parquet_source(&source).unwrap();
    assert!(source.is_direct_device_read());
    assert_eq!(source.read_column_count(), 2);
    assert_eq!(source.selected_row_group_count(), 3);
    assert_eq!(source.diagnostics.file_count, 1);
    assert_eq!(source.diagnostics.ranged_file_count, 0);
    assert_eq!(source.diagnostics.known_bytes, Some(3_235_196_281));
    assert_eq!(source.diagnostics.selected_row_groups, Some(3));
    assert_eq!(source.diagnostics.predicate_disposition, PredicateDisposition::Absent);
    assert!(source.diagnostics.rejected_features.is_empty());
}

#[test]
fn predicate_disposition_names_are_stable_manifest_values() {
    let dispositions = [
        (PredicateDisposition::Absent, "absent"),
        (PredicateDisposition::PruningOnly, "pruning_only"),
        (PredicateDisposition::Rejected, "rejected"),
    ];

    for (disposition, expected) in dispositions {
        assert_eq!(disposition.as_str(), expected);
    }
}

#[test]
fn local_parquet_source_validation_rejects_invalid_reader_contracts() {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::UInt32, false)]));
    let empty_columns = LocalParquetSource::new("/tmp/phase4.parquet", Arc::clone(&schema), Vec::new());

    let error = validate_local_parquet_source(&empty_columns).unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("reason"), Some("empty_read_columns"));

    let overflow_row_group = LocalParquetSource::from_files(
        vec![LocalParquetFile::new("/tmp/phase4.parquet").with_row_groups(vec![usize::MAX])],
        schema,
        vec!["key".to_string()],
    );

    let error = validate_local_parquet_source(&overflow_row_group).unwrap_err();
    let expected_row_group = usize::MAX.to_string();
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("row_group"), Some(expected_row_group.as_str()));
}

#[test]
fn source_chunk_planning_preserves_selected_row_groups_and_file_order() {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/part-000.parquet").with_file_size_bytes(300).with_row_groups(vec![0, 2, 4]),
            LocalParquetFile::new("/tmp/part-001.parquet").with_file_size_bytes(200).with_row_groups(vec![1, 3]),
        ],
        Arc::clone(&schema),
        vec!["key".to_string()],
    );
    let mut builder = QueryPlanBuilder::new();
    let source_id = builder.add_source(SourceNode::LocalParquet(source.clone()));

    let plan =
        plan_local_parquet_source_chunks(source_id, &source, SourceChunkPlanningPolicy::new(Some(1024), 2)).unwrap();

    assert_eq!(plan.source_node_id, source_id);
    assert_eq!(plan.chunk_count(), 3);
    assert_eq!(plan.row_group_count(), 5);
    assert!(!plan.row_group_metadata_available);
    assert!(plan.size_estimate_based);
    assert_eq!(plan.metadata_unavailable_files, 2);
    assert_eq!(plan.chunks[0].chunk_index, 0);
    assert_eq!(plan.chunks[0].file_index, 0);
    assert_eq!(plan.chunks[0].row_group_indexes, vec![0, 2]);
    assert_eq!(plan.chunks[0].known_file_bytes, Some(300));
    assert_eq!(plan.chunks[0].known_row_group_bytes, None);
    assert_eq!(plan.chunks[0].estimated_chunk_bytes, Some(200));
    assert_eq!(plan.chunks[0].read_columns, vec!["key".to_string()]);
    assert!(!plan.chunks[0].reads_all_row_groups);
    assert_eq!(plan.chunks[1].file_index, 0);
    assert_eq!(plan.chunks[1].row_group_indexes, vec![4]);
    assert_eq!(plan.chunks[1].estimated_chunk_bytes, Some(100));
    assert_eq!(plan.chunks[2].file_index, 1);
    assert_eq!(plan.chunks[2].row_group_indexes, vec![1, 3]);
    assert_eq!(plan.chunks[2].estimated_chunk_bytes, Some(200));
}

#[test]
fn source_chunk_planning_expands_all_row_groups_from_parquet_metadata() {
    let dir = unique_temp_dir("cudf-query-engine-source-chunks");
    let path = dir.join("input.parquet");
    let schema = write_row_grouped_parquet(&path, 6, 2);
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        schema,
        vec!["key".to_string()],
    );
    let mut builder = QueryPlanBuilder::new();
    let source_id = builder.add_source(SourceNode::LocalParquet(source.clone()));

    let plan = plan_local_parquet_source_chunks(source_id, &source, SourceChunkPlanningPolicy::new(None, 2)).unwrap();

    assert_eq!(plan.chunk_count(), 2);
    assert_eq!(plan.row_group_count(), 3);
    assert!(plan.row_group_metadata_available);
    assert!(!plan.size_estimate_based);
    assert_eq!(plan.metadata_unavailable_files, 0);
    assert_eq!(plan.chunks[0].row_group_indexes, vec![0, 1]);
    assert_eq!(plan.chunks[1].row_group_indexes, vec![2]);
    assert!(plan.chunks.iter().all(|chunk| chunk.known_row_group_bytes.unwrap() > 0));
    assert!(plan.chunks.iter().all(|chunk| !chunk.reads_all_row_groups));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn source_chunk_planning_uses_single_all_row_groups_chunk_when_metadata_is_unavailable() {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new("/tmp/metadata-unavailable.parquet").with_file_size_bytes(4096)],
        schema,
        vec!["key".to_string()],
    );
    let mut builder = QueryPlanBuilder::new();
    let source_id = builder.add_source(SourceNode::LocalParquet(source.clone()));

    let plan =
        plan_local_parquet_source_chunks(source_id, &source, SourceChunkPlanningPolicy::new(Some(1024), 2)).unwrap();

    assert_eq!(plan.chunk_count(), 1);
    assert_eq!(plan.row_group_count(), 0);
    assert!(!plan.row_group_metadata_available);
    assert!(plan.size_estimate_based);
    assert_eq!(plan.metadata_unavailable_files, 1);
    assert!(plan.chunks[0].reads_all_row_groups);
    assert!(plan.chunks[0].row_group_indexes.is_empty());
    assert_eq!(plan.chunks[0].known_file_bytes, Some(4096));
    assert_eq!(plan.chunks[0].known_row_group_bytes, None);
    assert_eq!(plan.chunks[0].estimated_chunk_bytes, Some(4096));
}

#[test]
fn source_chunk_planning_honors_max_source_chunk_bytes_when_row_group_sizes_are_known() {
    let dir = unique_temp_dir("cudf-query-engine-byte-source-chunks");
    let path = dir.join("input.parquet");
    let schema = write_row_grouped_parquet(&path, 6, 2);
    let source = LocalParquetSource::from_files(vec![LocalParquetFile::new(&path)], schema, vec!["key".to_string()]);
    let mut builder = QueryPlanBuilder::new();
    let source_id = builder.add_source(SourceNode::LocalParquet(source.clone()));

    let plan =
        plan_local_parquet_source_chunks(source_id, &source, SourceChunkPlanningPolicy::new(Some(1), 128)).unwrap();

    assert_eq!(plan.chunk_count(), 3);
    assert_eq!(plan.chunks[0].row_group_indexes, vec![0]);
    assert_eq!(plan.chunks[1].row_group_indexes, vec![1]);
    assert_eq!(plan.chunks[2].row_group_indexes, vec![2]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn source_chunk_planning_rejects_selected_row_groups_outside_metadata() {
    let dir = unique_temp_dir("cudf-query-engine-source-chunk-rejects");
    let path = dir.join("input.parquet");
    let schema = write_row_grouped_parquet(&path, 4, 2);
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_row_groups(vec![0, 2])],
        schema,
        vec!["key".to_string()],
    );
    let mut builder = QueryPlanBuilder::new();
    let source_id = builder.add_source(SourceNode::LocalParquet(source.clone()));

    let error = plan_local_parquet_source_chunks(source_id, &source, SourceChunkPlanningPolicy::default()).unwrap_err();

    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("file_index"), Some("0"));
    assert_eq!(error.fact_value("row_group_position"), Some("1"));
    assert_eq!(error.fact_value("row_group"), Some("2"));
    assert_eq!(error.fact_value("max_row_group"), Some("1"));

    fs::remove_dir_all(dir).unwrap();
}

fn write_row_grouped_parquet(path: &Path, row_count: usize, rows_per_group: usize) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
    let values = (0..row_count).map(|value| value as i32).collect::<Vec<_>>();
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(values))]).unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(rows_per_group)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    schema
}

fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let path = std::env::temp_dir().join(format!("{prefix}-{}-{nonce}", std::process::id()));
    fs::create_dir_all(&path).unwrap();
    path
}

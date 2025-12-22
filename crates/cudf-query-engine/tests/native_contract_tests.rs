use std::{
    fs::{self, File},
    path::Path,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use arrow::{
    array::Int32Array,
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use cudf::{DataType as CudfDataType, TypeId};
use cudf_query_engine::{
    AggregateExpr, AggregateFunction, AggregateSpec, BinaryOp, BoundaryMetrics, ByteRangeDisposition, ColumnRef,
    DataPlacement, ExecutionContext, Expr, JoinKey, JoinNullEquality, JoinOperatorMetrics, JoinOutputColumn, JoinSide,
    JoinSpec, JoinType, LimitSpec, LiteralValue, LocalParquetFile, LocalParquetSource, MemoryControlPolicy,
    MemoryReservationRequest, OperatorMetrics, PartitionColumn, PartitionColumnDisposition, PlanNode,
    PredicateDisposition, QueryError, QueryErrorCode, QueryErrorKind, QueryErrorStatus, QueryMetrics, QueryPlanBuilder,
    RejectedSourceFeature, ReservationKind, ResultSink, RetryFacts, SchemaBehavior, SchemaMappingDisposition, SortExpr,
    SortSpec, SourceChunkPlanningPolicy, SourceColumnRef, SourceNode, ZERO_COLUMN_COUNT_COLUMN,
    ZeroColumnCountDisposition, cudf_dtype_from_arrow_type,
    expr::{CaseWhen, DatePart},
    is_supported_arrow_cast, is_supported_arrow_implicit_cast, is_supported_arrow_type,
    plan_local_parquet_source_chunks, validate_local_parquet_source,
};
use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};

#[test]
fn decimal128_type_support_policy_accepts_valid_arrow_decimals() {
    let decimal_with_fractional_scale = DataType::Decimal128(38, 2);
    let decimal_with_negative_scale = DataType::Decimal128(20, -1);

    assert!(is_supported_arrow_type(&decimal_with_fractional_scale));
    assert_eq!(
        cudf_dtype_from_arrow_type(&decimal_with_fractional_scale),
        Some(CudfDataType::fixed_point(TypeId::Decimal128, -2))
    );
    assert!(is_supported_arrow_type(&decimal_with_negative_scale));
    assert_eq!(
        cudf_dtype_from_arrow_type(&decimal_with_negative_scale),
        Some(CudfDataType::fixed_point(TypeId::Decimal128, 1))
    );
}

#[test]
fn decimal128_type_support_policy_rejects_invalid_arrow_decimals() {
    let invalid_precision = DataType::Decimal128(39, 2);
    let invalid_scale = DataType::Decimal128(10, 11);

    assert!(!is_supported_arrow_type(&invalid_precision));
    assert_eq!(cudf_dtype_from_arrow_type(&invalid_precision), None);
    assert!(!is_supported_arrow_type(&invalid_scale));
    assert_eq!(cudf_dtype_from_arrow_type(&invalid_scale), None);
}

#[test]
fn decimal128_cast_support_uses_type_support_policy() {
    let valid_source = DataType::Decimal128(20, -1);
    let valid_target = DataType::Decimal128(38, 2);
    let float64_target = DataType::Float64;
    let float64_source = DataType::Float64;
    let int32_source = DataType::Int32;
    let invalid_decimal = DataType::Decimal128(39, 2);

    assert!(is_supported_arrow_cast(&valid_source, &valid_target));
    assert!(is_supported_arrow_cast(&valid_source, &float64_target));
    assert!(is_supported_arrow_cast(&float64_source, &valid_target));
    assert!(is_supported_arrow_cast(&int32_source, &valid_target));
    assert!(is_supported_arrow_implicit_cast(&valid_source, &valid_target));
    assert!(!is_supported_arrow_implicit_cast(&valid_source, &float64_target));
    assert!(!is_supported_arrow_implicit_cast(&float64_source, &valid_target));
    assert!(!is_supported_arrow_cast(&invalid_decimal, &invalid_decimal));
    assert!(!is_supported_arrow_cast(&invalid_decimal, &valid_target));
    assert!(!is_supported_arrow_cast(&invalid_decimal, &float64_target));
    assert!(!is_supported_arrow_implicit_cast(&invalid_decimal, &float64_target));
    assert!(!is_supported_arrow_cast(&valid_source, &invalid_decimal));
    assert!(!is_supported_arrow_cast(&float64_source, &invalid_decimal));
}

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
fn native_expr_can_represent_boolean_not() {
    let expr = !Expr::binary(Expr::column("payload"), BinaryOp::Lt, Expr::literal(LiteralValue::Int64(10)));

    assert_eq!(expr.to_string(), "NOT (payload < 10)");
}

#[test]
fn native_expr_can_represent_numeric_cast() {
    let expr = Expr::cast(Expr::column("payload"), DataType::Int64).alias("payload_i64");

    assert_eq!(expr.to_string(), "CAST(payload AS Int64) AS payload_i64");
}

#[test]
fn native_expr_can_represent_phase2_core_expression_surface() {
    let modulo = Expr::binary(Expr::column("payload"), BinaryOp::Modulo, Expr::literal(LiteralValue::Int64(10)));
    let like = Expr::like(Expr::column("name"), "A%");
    let not_like = Expr::not_like_escape(Expr::column("name"), "A!_%", "!");
    let substring = Expr::substring(Expr::column("name"), 2, Some(3));
    let in_list = Expr::in_list(
        Expr::column("status"),
        vec![Expr::literal(LiteralValue::Utf8("O".to_string())), Expr::literal(LiteralValue::Utf8("F".to_string()))],
    );
    let not_in_list = Expr::not_in_list(
        Expr::column("status"),
        vec![Expr::literal(LiteralValue::Utf8("R".to_string())), Expr::literal(LiteralValue::Utf8("X".to_string()))],
    );
    let case = Expr::case(
        vec![
            CaseWhen::new(
                Expr::binary(Expr::column("score"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int64(90))),
                Expr::literal(LiteralValue::Utf8("A".to_string())),
            ),
            CaseWhen::new(
                Expr::binary(Expr::column("score"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int64(80))),
                Expr::literal(LiteralValue::Utf8("B".to_string())),
            ),
        ],
        Expr::literal(LiteralValue::Utf8("C".to_string())),
    );
    let extract = Expr::extract(DatePart::Year, Expr::column("event_ts"));
    let is_null = Expr::is_null(Expr::column("discount"));
    let is_not_null = Expr::is_not_null(Expr::column("discount"));
    let timestamp_plus_duration = Expr::binary(
        Expr::column("event_ts"),
        BinaryOp::Add,
        Expr::literal(LiteralValue::Duration { raw: 10, unit: TimeUnit::Nanosecond }),
    );
    let date_plus_interval = Expr::binary(
        Expr::column("shipdate"),
        BinaryOp::Add,
        Expr::literal(LiteralValue::IntervalMonthDayNano { months: 1, days: 2, nanoseconds: 0 }),
    );

    assert_eq!(modulo.to_string(), "(payload % 10)");
    assert_eq!(like.to_string(), "name LIKE \"A%\"");
    assert_eq!(not_like.to_string(), "name NOT LIKE \"A!_%\" ESCAPE \"!\"");
    assert_eq!(substring.to_string(), "SUBSTRING(name FROM 2 FOR 3)");
    assert_eq!(in_list.to_string(), "status IN (\"O\", \"F\")");
    assert_eq!(not_in_list.to_string(), "status NOT IN (\"R\", \"X\")");
    assert_eq!(case.to_string(), "CASE WHEN (score >= 90) THEN \"A\" WHEN (score >= 80) THEN \"B\" ELSE \"C\" END");
    assert_eq!(extract.to_string(), "EXTRACT(YEAR FROM event_ts)");
    assert_eq!(is_null.to_string(), "discount IS NULL");
    assert_eq!(is_not_null.to_string(), "discount IS NOT NULL");
    assert_eq!(timestamp_plus_duration.to_string(), "(event_ts + DURATION(10, Nanosecond))");
    assert_eq!(date_plus_interval.to_string(), "(shipdate + INTERVAL(months=1, days=2, nanoseconds=0))");
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
    assert_eq!(ReservationKind::BuildSide.as_str(), "build_side");
    assert_eq!(ReservationKind::ProbeSide.as_str(), "probe_side");
    assert_eq!(ReservationKind::HashTable.as_str(), "hash_table");
    assert_eq!(ReservationKind::GatherMap.as_str(), "gather_map");
    assert_eq!(ReservationKind::JoinOutput.as_str(), "join_output");

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
fn native_expr_preserves_qualified_column_identity_separate_from_display_name() {
    let left = ColumnRef::input(0, "id", "left_id");
    let right = ColumnRef::input(1, "id", "right_id");

    assert_ne!(left, right);
    assert_eq!(left.name(), "id");
    assert_eq!(left.display_name(), "left_id");
    assert!(left.is_qualified());
    assert_eq!(Expr::Column(left.clone()).to_string(), "input0.id");
    assert_eq!(Expr::Column(right.clone()).to_string(), "input1.id");
}

#[test]
fn native_plan_graph_represents_binary_join_with_same_named_inputs() {
    let left_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let right_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("left_id", DataType::Int32, false),
        Field::new("right_id", DataType::Int32, false),
    ]));
    let left_source = LocalParquetSource::new("/tmp/left.parquet", Arc::clone(&left_schema), vec!["id".into()]);
    let right_source = LocalParquetSource::new("/tmp/right.parquet", Arc::clone(&right_schema), vec!["id".into()]);

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(left_source));
    let right = builder.add_source(SourceNode::LocalParquet(right_source));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "id", "left_id"), ColumnRef::input(1, "id", "right_id"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "id", "left_id"), "left_id"),
            JoinOutputColumn::right(ColumnRef::input(1, "id", "right_id"), "right_id"),
        ])
        .with_null_equality(JoinNullEquality::NullsDoNotMatch)
        .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(join);

    let PlanNode::Join(join_node) = &plan.root_node().kind else {
        panic!("root node should be a native join contract");
    };
    assert_eq!(plan.root_node().input_ids(), &[left, right]);
    assert_eq!(join_node.spec.keys[0].left.name(), "id");
    assert_eq!(join_node.spec.keys[0].left.display_name(), "left_id");
    assert_eq!(join_node.spec.keys[0].right.name(), "id");
    assert_eq!(join_node.spec.keys[0].right.display_name(), "right_id");
    assert_eq!(plan.root_node().facts.schema_behavior, SchemaBehavior::Join);
    assert!(plan.to_string().contains("Join left=n0 right=n1 type=inner key_count=1"));
}

#[test]
fn native_query_validation_rejects_unreachable_dag_nodes() {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let mut builder = QueryPlanBuilder::new();
    let root = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/root.parquet",
        Arc::clone(&schema),
        vec!["id".into()],
    )));
    let _unreachable = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/unreachable.parquet",
        Arc::clone(&schema),
        vec!["id".into()],
    )));
    let plan = builder.finish_with_host_sink(root);
    let unreachable = plan.unreachable_node_ids().iter().map(ToString::to_string).collect::<Vec<_>>();
    assert_eq!(unreachable, vec!["n1"]);
}

#[test]
fn query_metrics_have_join_operator_contract() {
    let mut metrics = QueryMetrics::default();
    metrics.record_operator_metrics(OperatorMetrics::join(
        "n2",
        JoinOperatorMetrics {
            join_type: "inner".into(),
            key_count: 2,
            residual_predicate: true,
            null_equality: "nulls_do_not_match".into(),
            build_side: "right".into(),
            probe_side: "left".into(),
            build_rows: 10,
            build_bytes: 1024,
            probe_rows: 20,
            probe_bytes: 2048,
            hash_table_bytes: 4096,
            output_rows: 5,
            output_bytes: 512,
        },
    ));

    assert_eq!(metrics.operators.len(), 1);
    let join = metrics.operators[0].join.as_ref().expect("join metrics should be attached");
    assert_eq!(join.join_type, "inner");
    assert_eq!(join.key_count, 2);
    assert_eq!(join.hash_table_bytes, 4096);
    assert_eq!(join.build_side, "right");
    assert_eq!(join.probe_side, "left");

    metrics.record_hash_join_cache_miss();
    metrics.record_hash_join_build();
    metrics.record_hash_join_cache_hit();
    assert_eq!(metrics.hash_join_cache_misses, 1);
    assert_eq!(metrics.hash_join_build_count, 1);
    assert_eq!(metrics.hash_join_cache_hits, 1);
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
                .with_stable_id("phase4:3235196281")
                .with_file_size_bytes(3_235_196_281)
                .with_row_groups(vec![0, 2, 4])
                .with_row_group_bytes(vec![10, 20, 30, 40, 50]),
        ],
        schema,
        vec!["key".to_string(), "payload".to_string()],
    )
    .with_column_roles(vec!["key".to_string()], vec!["payload".to_string()]);

    validate_local_parquet_source(&source).unwrap();
    assert_eq!(source.files[0].stable_id, "phase4:3235196281");
    assert!(source.is_direct_device_read());
    assert_eq!(source.read_column_count(), 2);
    assert_eq!(source.read_column_file_indices, vec![0, 1]);
    assert_eq!(source.projected_output_column_count(), 1);
    assert_eq!(source.projected_output_column_source_refs, vec![SourceColumnRef::File(0)]);
    assert_eq!(source.pure_filter_column_count(), 1);
    assert_eq!(source.pure_filter_column_source_refs, vec![SourceColumnRef::File(1)]);
    assert_eq!(source.selected_row_group_count(), 3);
    assert_eq!(source.diagnostics.file_count, 1);
    assert_eq!(source.diagnostics.ranged_file_count, 0);
    assert_eq!(source.diagnostics.known_bytes, Some(3_235_196_281));
    assert_eq!(source.diagnostics.selected_row_groups, Some(3));
    assert_eq!(source.diagnostics.read_column_count, 2);
    assert_eq!(source.diagnostics.projected_output_column_count, 1);
    assert_eq!(source.diagnostics.pure_filter_column_count, 1);
    assert_eq!(source.diagnostics.predicate_disposition, PredicateDisposition::Absent);
    assert_eq!(source.diagnostics.byte_range_disposition, ByteRangeDisposition::FullFiles);
    assert_eq!(source.diagnostics.partition_column_disposition, PartitionColumnDisposition::Absent);
    assert_eq!(source.diagnostics.zero_column_count_disposition, ZeroColumnCountDisposition::NotRequired);
    assert_eq!(source.diagnostics.schema_mapping_disposition, SchemaMappingDisposition::DirectByName);
    assert!(source.diagnostics.rejected_features.is_empty());
}

#[test]
fn local_parquet_source_represents_partition_columns_as_synthetic_source_columns() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt32, false),
        Field::new("payload", DataType::Int64, false),
        Field::new("part", DataType::Int32, false),
    ]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new("/tmp/phase4.parquet").with_partition_values(vec![LiteralValue::Int32(7)])],
        schema,
        vec!["key".to_string(), "payload".to_string()],
    )
    .with_partition_columns(vec![PartitionColumn::new("part", DataType::Int32, false)])
    .with_column_roles(vec!["key".to_string(), "part".to_string()], vec!["payload".to_string(), "part".to_string()]);

    validate_local_parquet_source(&source).unwrap();
    assert_eq!(source.partition_column_count(), 1);
    assert_eq!(source.read_columns, vec!["key".to_string(), "payload".to_string()]);
    assert_eq!(
        source.projected_output_column_source_refs,
        vec![SourceColumnRef::File(0), SourceColumnRef::Partition(0)]
    );
    assert_eq!(source.pure_filter_column_source_refs, vec![SourceColumnRef::File(1), SourceColumnRef::Partition(0)]);
    assert_eq!(source.diagnostics.partition_column_disposition, PartitionColumnDisposition::Represented);
    assert!(source.diagnostics.rejected_features.is_empty());
}

#[test]
fn local_parquet_source_rejects_invalid_partition_column_contracts() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt32, false),
        Field::new("part", DataType::Int32, false),
    ]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new("/tmp/phase4.parquet")],
        Arc::clone(&schema),
        vec!["key".to_string()],
    )
    .with_partition_columns(vec![PartitionColumn::new("part", DataType::Int32, false)]);

    let error = validate_local_parquet_source(&source).unwrap_err();
    assert_eq!(error.fact_value("reason"), Some("partition_value_count_mismatch"));
    assert_eq!(error.fact_value("partition_columns"), Some("1"));
    assert_eq!(error.fact_value("partition_values"), Some("0"));

    let wrong_type = LocalParquetSource::from_files(
        vec![LocalParquetFile::new("/tmp/phase4.parquet").with_partition_values(vec![LiteralValue::Int64(7)])],
        schema,
        vec!["key".to_string()],
    )
    .with_partition_columns(vec![PartitionColumn::new("part", DataType::Int32, false)]);

    let error = validate_local_parquet_source(&wrong_type).unwrap_err();
    assert_eq!(error.fact_value("reason"), Some("partition_value_type_mismatch"));
    assert_eq!(error.fact_value("partition_column"), Some("part"));
}

#[test]
fn source_disposition_names_are_stable_manifest_values() {
    let dispositions = [
        (PredicateDisposition::Absent, "absent"),
        (PredicateDisposition::PruningOnly, "pruning_only"),
        (PredicateDisposition::Rejected, "rejected"),
    ];

    for (disposition, expected) in dispositions {
        assert_eq!(disposition.as_str(), expected);
    }

    assert_eq!(ByteRangeDisposition::FullFiles.as_str(), "full_files");
    assert_eq!(ByteRangeDisposition::CoalescedFullFiles.as_str(), "coalesced_full_files");
    assert_eq!(ByteRangeDisposition::Rejected.as_str(), "rejected");
    assert_eq!(PartitionColumnDisposition::Absent.as_str(), "absent");
    assert_eq!(PartitionColumnDisposition::Rejected.as_str(), "rejected");
    assert_eq!(PartitionColumnDisposition::Represented.as_str(), "represented");
    assert_eq!(ZeroColumnCountDisposition::NotRequired.as_str(), "not_required");
    assert_eq!(ZeroColumnCountDisposition::Rejected.as_str(), "rejected");
    assert_eq!(ZeroColumnCountDisposition::Represented.as_str(), "represented");
    assert_eq!(SchemaMappingDisposition::DirectByName.as_str(), "direct_by_name");
    assert_eq!(SchemaMappingDisposition::Rejected.as_str(), "rejected");
}

#[test]
fn local_parquet_source_validation_rejects_invalid_reader_contracts() {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::UInt32, false)]));
    let empty_columns = LocalParquetSource::new("/tmp/phase4.parquet", Arc::clone(&schema), Vec::new());

    let error = validate_local_parquet_source(&empty_columns).unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("reason"), Some("empty_read_columns"));
    assert_eq!(empty_columns.diagnostics.zero_column_count_disposition, ZeroColumnCountDisposition::Rejected);
    assert!(empty_columns.diagnostics.rejected_features.iter().any(|feature| feature.as_str() == "zero_column_count"));

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
fn zero_column_count_source_contract_is_represented_without_data_columns() {
    let source = LocalParquetSource::zero_column_count(vec![
        LocalParquetFile::new("/tmp/phase4.parquet")
            .with_stable_id("phase4:count")
            .with_file_size_bytes(300)
            .with_row_groups(vec![0, 2])
            .with_row_group_rows(vec![10, 20, 30]),
    ]);

    validate_local_parquet_source(&source).unwrap();
    assert!(source.is_zero_column_count_source());
    assert!(source.is_direct_device_read());
    assert_eq!(source.read_column_count(), 0);
    assert_eq!(source.projected_output_column_count(), 0);
    assert_eq!(source.pure_filter_column_count(), 0);
    assert!(source.read_column_file_indices.is_empty());
    assert!(source.projected_output_column_source_refs.is_empty());
    assert!(source.pure_filter_column_source_refs.is_empty());
    assert!(source.schema.field_with_name(ZERO_COLUMN_COUNT_COLUMN).is_ok());
    assert_eq!(source.diagnostics.read_column_count, 0);
    assert_eq!(source.diagnostics.projected_output_column_count, 0);
    assert_eq!(source.diagnostics.pure_filter_column_count, 0);
    assert_eq!(source.diagnostics.zero_column_count_disposition, ZeroColumnCountDisposition::Represented);
    assert!(source.diagnostics.rejected_features.is_empty());
}

#[test]
fn local_parquet_source_validation_rejects_unhonorable_source_facts_before_execution() {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::UInt32, false)]));
    let base = LocalParquetSource::new("/tmp/phase4.parquet", Arc::clone(&schema), vec!["key".to_string()]);

    let mut rejected_byte_range = base.clone();
    rejected_byte_range.diagnostics.byte_range_disposition = ByteRangeDisposition::Rejected;
    let error = validate_local_parquet_source(&rejected_byte_range).unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("reason"), Some("unsupported_source_contract_fact"));
    assert_eq!(error.fact_value("field"), Some("byte_range"));
    assert_eq!(error.fact_value("disposition"), Some("rejected"));

    let mut inconsistent_partition_columns = base.clone();
    inconsistent_partition_columns.diagnostics.partition_column_disposition = PartitionColumnDisposition::Represented;
    let error = validate_local_parquet_source(&inconsistent_partition_columns).unwrap_err();
    assert_eq!(error.fact_value("reason"), Some("empty_partition_columns"));

    let mut rejected_zero_column_count = base.clone();
    rejected_zero_column_count.diagnostics.zero_column_count_disposition = ZeroColumnCountDisposition::Represented;
    let error = validate_local_parquet_source(&rejected_zero_column_count).unwrap_err();
    assert_eq!(error.fact_value("reason"), Some("invalid_zero_column_count_columns"));
    assert_eq!(error.fact_value("read_columns"), Some("1"));

    let mut rejected_schema_mapping = base.clone();
    rejected_schema_mapping.diagnostics.schema_mapping_disposition = SchemaMappingDisposition::Rejected;
    let error = validate_local_parquet_source(&rejected_schema_mapping).unwrap_err();
    assert_eq!(error.fact_value("field"), Some("schema_mapping"));
    assert_eq!(error.fact_value("disposition"), Some("rejected"));

    let mut rejected_feature = base;
    rejected_feature.diagnostics.rejected_features.push(RejectedSourceFeature::RowSelection);
    let error = validate_local_parquet_source(&rejected_feature).unwrap_err();
    assert_eq!(error.fact_value("reason"), Some("rejected_source_features"));
    assert_eq!(error.fact_value("rejected_features"), Some("row_selection"));
}

#[test]
fn local_parquet_source_validation_rejects_inconsistent_column_and_row_group_contracts() {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::UInt32, false)]));
    let two_column_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt32, false),
        Field::new("payload", DataType::UInt32, false),
    ]));

    let mut missing_projected_read =
        LocalParquetSource::new("/tmp/phase4.parquet", two_column_schema, vec!["key".to_string()])
            .with_column_roles(vec!["payload".to_string()], Vec::new());
    missing_projected_read.diagnostics.rejected_features.clear();
    missing_projected_read.diagnostics.direct_device_read = true;
    missing_projected_read.diagnostics.schema_mapping_disposition = SchemaMappingDisposition::DirectByName;
    missing_projected_read.projected_output_column_source_refs.push(SourceColumnRef::File(1));
    let error = validate_local_parquet_source(&missing_projected_read).unwrap_err();
    assert_eq!(error.fact_value("reason"), Some("projected_column_not_available"));
    assert_eq!(error.fact_value("column"), Some("payload"));

    let mut mismatched_diagnostics =
        LocalParquetSource::new("/tmp/phase4.parquet", Arc::clone(&schema), vec!["key".to_string()]);
    mismatched_diagnostics.diagnostics.read_column_count = 2;
    let error = validate_local_parquet_source(&mismatched_diagnostics).unwrap_err();
    assert_eq!(error.fact_value("reason"), Some("diagnostic_column_count_mismatch"));
    assert_eq!(error.fact_value("diagnostic_read_columns"), Some("2"));
    assert_eq!(error.fact_value("read_columns"), Some("1"));

    let missing_schema_mapping =
        LocalParquetSource::new("/tmp/phase4.parquet", Arc::clone(&schema), vec!["missing".to_string()]);
    assert_eq!(missing_schema_mapping.diagnostics.schema_mapping_disposition, SchemaMappingDisposition::Rejected);
    let error = validate_local_parquet_source(&missing_schema_mapping).unwrap_err();
    assert_eq!(error.fact_value("reason"), Some("rejected_source_features"));
    assert_eq!(error.fact_value("rejected_features"), Some("schema_mapping"));

    let row_group_outside_contract = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/phase4.parquet")
                .with_row_groups(vec![0, 2])
                .with_row_group_bytes(vec![100, 200]),
        ],
        schema,
        vec!["key".to_string()],
    );
    let error = validate_local_parquet_source(&row_group_outside_contract).unwrap_err();
    assert_eq!(error.fact_value("reason"), None);
    assert_eq!(error.fact_value("row_group"), Some("2"));
    assert_eq!(error.fact_value("max_row_group"), Some("1"));
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
fn source_chunk_planning_uses_contract_row_group_bytes_without_file_metadata() {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/part-000.parquet")
                .with_file_size_bytes(600)
                .with_row_groups(vec![0, 2])
                .with_row_group_bytes(vec![100, 200, 300]),
        ],
        Arc::clone(&schema),
        vec!["key".to_string()],
    );
    let mut builder = QueryPlanBuilder::new();
    let source_id = builder.add_source(SourceNode::LocalParquet(source.clone()));

    let plan =
        plan_local_parquet_source_chunks(source_id, &source, SourceChunkPlanningPolicy::new(Some(250), 8)).unwrap();

    assert_eq!(plan.chunk_count(), 2);
    assert_eq!(plan.row_group_count(), 2);
    assert!(plan.row_group_metadata_available);
    assert!(!plan.size_estimate_based);
    assert_eq!(plan.metadata_unavailable_files, 0);
    assert_eq!(plan.chunks[0].row_group_indexes, vec![0]);
    assert_eq!(plan.chunks[0].known_row_group_bytes, Some(100));
    assert_eq!(plan.chunks[1].row_group_indexes, vec![2]);
    assert_eq!(plan.chunks[1].known_row_group_bytes, Some(300));
}

#[test]
fn source_chunk_planning_carries_zero_column_count_row_counts() {
    let source = LocalParquetSource::zero_column_count(vec![
        LocalParquetFile::new("/tmp/part-000.parquet")
            .with_file_size_bytes(600)
            .with_row_groups(vec![0, 2])
            .with_row_group_bytes(vec![100, 200, 300])
            .with_row_group_rows(vec![5, 7, 11]),
    ]);
    let mut builder = QueryPlanBuilder::new();
    let source_id = builder.add_source(SourceNode::LocalParquet(source.clone()));

    let plan =
        plan_local_parquet_source_chunks(source_id, &source, SourceChunkPlanningPolicy::new(Some(250), 8)).unwrap();

    assert_eq!(plan.chunk_count(), 2);
    assert_eq!(plan.row_group_count(), 2);
    assert!(plan.chunks.iter().all(|chunk| chunk.zero_column_count));
    assert!(plan.chunks.iter().all(|chunk| chunk.read_columns.is_empty()));
    assert_eq!(plan.chunks[0].row_group_indexes, vec![0]);
    assert_eq!(plan.chunks[0].known_row_count, Some(5));
    assert_eq!(plan.chunks[1].row_group_indexes, vec![2]);
    assert_eq!(plan.chunks[1].known_row_count, Some(11));
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

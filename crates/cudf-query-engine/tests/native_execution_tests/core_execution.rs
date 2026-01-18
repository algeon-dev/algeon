use super::*;

#[test]
fn native_query_executes_direct_parquet_pipeline_to_host_sink() {
    let dir = unique_temp_dir("cudf-query-engine-native-execution");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_input_parquet(&runtime, &path);

    let projected_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("x00", DataType::Int64, false),
    ]));
    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("sum_x00", DataType::Int64, true),
    ]));

    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(&path)
                .with_file_size_bytes(fs::metadata(&path).unwrap().len())
                .with_row_group_bytes(vec![4096, 4096, 4096]),
        ],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "m00".to_string(), "m01".to_string(), "payload".to_string()],
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
        SortExpr::new(Expr::column("key"), false, false),
    ]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let filter = builder.add_filter(source, filter, Arc::clone(&input_schema));
    let projection = builder.add_projection(filter, projection_outputs(projection, projected_schema));
    let aggregate = builder.add_aggregate(projection, aggregate, Arc::clone(&aggregated_schema));
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&aggregated_schema));
    let limit = builder.add_limit(sort, LimitSpec::fetch(2), aggregated_schema);
    let plan = builder.finish_with_host_sink(limit);

    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.parquet_files, 1);
    assert_eq!(metrics.read_columns, 4);
    assert_eq!(metrics.input_batches, 1);
    assert_eq!(metrics.input_rows, 4);
    assert_eq!(metrics.boundary.host_to_device_uploads, 0);
    assert_eq!(metrics.boundary.device_to_host_materializations, 1);
    assert_eq!(metrics.boundary.device_to_host_materialized_rows, 2);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 2);
    assert_eq!(batch.num_columns(), 2);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[3, 1]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[404, 303]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_execution_boundary_policy_reports_direct_source_and_explicit_host_materialization() {
    let dir = unique_temp_dir("cudf-query-engine-native-boundary-policy-source-host");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_input_parquet(&runtime, &path);
    let policy = ExecutionBoundaryPolicy::default();

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "payload".to_string(), "m00".to_string(), "m01".to_string()],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let plan = builder.finish_with_host_sink(source);
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_boundary_policy(policy)
        .build()
        .unwrap();

    assert_eq!(context.boundary_policy(), policy);
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.execution_boundary_policy, policy);
    assert_eq!(metrics.boundary.host_to_device_uploads, 0);
    assert_eq!(metrics.boundary.device_to_host_materializations, 1);
    assert_eq!(metrics.boundary.device_to_host_materialized_rows, 4);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("explicit host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 4);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_local_parquet_source_scan_limit_applies_before_parent_operators() {
    let dir = unique_temp_dir("cudf-query-engine-native-source-scan-limit");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_input_parquet(&runtime, &path);
    let projected_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
    ]));

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "payload".to_string()],
    )
    .with_scan_limit(Some(2));
    let projection = vec![Expr::column("key"), Expr::column("payload")];

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let projection = builder.add_projection(source, projection_outputs(projection, projected_schema));
    let plan = builder.finish_with_host_sink(projection);

    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));
    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 2);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[5, 10]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_local_parquet_source_materializes_missing_nullable_columns() {
    let dir = unique_temp_dir("cudf-query-engine-native-missing-nullable-source-columns");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    write_input_parquet(&runtime, &path);

    let source_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("missing_payload", DataType::Int64, true),
    ]));
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(&path)
                .with_file_size_bytes(fs::metadata(&path).unwrap().len())
                .with_row_group_rows(vec![4]),
        ],
        Arc::clone(&source_schema),
        vec!["key".to_string()],
    )
    .with_missing_columns(vec![MissingColumn::new("missing_payload", DataType::Int64, true)])
    .with_column_roles(vec!["key".to_string(), "missing_payload".to_string()], Vec::new());

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let plan = builder.finish_with_host_sink(source);

    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));
    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 4);
    assert_eq!(batch.num_columns(), 2);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2, 1, 3]);
    let missing = batch.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
    assert_eq!(missing.null_count(), 4);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_local_parquet_source_materializes_missing_nullable_columns() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-missing-nullable-source-columns");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    write_input_parquet(&runtime, &path);

    let source_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("missing_payload", DataType::Int64, true),
    ]));
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(&path)
                .with_file_size_bytes(fs::metadata(&path).unwrap().len())
                .with_row_group_bytes(vec![4096])
                .with_row_group_rows(vec![4]),
        ],
        Arc::clone(&source_schema),
        vec!["key".to_string()],
    )
    .with_missing_columns(vec![MissingColumn::new("missing_payload", DataType::Int64, true)])
    .with_column_roles(vec!["key".to_string(), "missing_payload".to_string()], Vec::new());

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let plan = builder.finish_with_host_sink(source);

    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 64 * 1024 * 1024,
            max_source_chunk_bytes: 16 * 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 2,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();
    let result = execute_query_plan(&context, &plan).unwrap();
    assert_eq!(result.metrics().source_chunks_committed, 1);
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 4);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2, 1, 3]);
    let missing = batch.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
    assert_eq!(missing.null_count(), 4);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_generic_local_parquet_pipeline_uses_ordered_source_scheduler() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-generic-source-scheduler");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_row_grouped_input_parquet(&path);

    let projected_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("x00", DataType::Int64, false),
    ]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "m00".to_string(), "m01".to_string(), "payload".to_string()],
    );
    let filter = Expr::binary(Expr::column("payload"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int64(10)));
    let projection =
        vec![Expr::column("key"), Expr::binary(Expr::column("m00"), BinaryOp::Add, Expr::column("m01")).alias("x00")];
    let sort = SortSpec::new(vec![
        SortExpr::new(Expr::column("key"), false, false),
        SortExpr::new(Expr::column("x00"), true, false),
    ]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let filter = builder.add_filter(source, filter, Arc::clone(&input_schema));
    let projection = builder.add_projection(filter, projection_outputs(projection, Arc::clone(&projected_schema)));
    let sort = builder.add_sort(projection, sort, Arc::clone(&projected_schema));
    let limit = builder.add_limit(sort, LimitSpec::fetch(4), projected_schema);
    let plan = builder.finish_with_host_sink(limit);

    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .with_source_scheduler_policy(SourceSchedulerPolicy::new(3))
        .build()
        .unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.source_chunks_planned, 3);
    assert_eq!(metrics.source_chunks_attempted, 3);
    assert_eq!(metrics.source_chunks_committed, 3);
    assert_eq!(metrics.source_read_prefetches_started, 2);
    assert_eq!(metrics.source_read_prefetches_completed, 2);
    assert_eq!(metrics.source_read_prefetches_skipped, 0);
    assert_eq!(metrics.source_read_prefetches_abandoned, 0);
    let first_generic_operator = metrics
        .reservation_events
        .iter()
        .position(|event| event.outcome.as_str() == "granted" && event.memory_boundary == "bounded_generic_filter")
        .expect("bounded generic filter reservation should be recorded");
    let prefetches_before_first_operator = metrics.reservation_events[..first_generic_operator]
        .iter()
        .filter(|event| {
            event.outcome.as_str() == "granted" && event.memory_boundary == "bounded_generic_source_read_prefetch"
        })
        .count();
    assert_eq!(prefetches_before_first_operator, 2);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 4);
    assert_eq!(batch.num_columns(), 2);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 1, 2, 2]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[303, 101, 505, 202]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_generic_local_parquet_source_scan_limit_stops_after_required_chunks() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-source-scan-limit");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let _input_schema = write_row_grouped_input_parquet(&path);
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
    ]));

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        source_schema,
        vec!["key".to_string(), "payload".to_string()],
    )
    .with_scan_limit(Some(3));

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let plan = builder.finish_with_host_sink(source);

    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .with_source_scheduler_policy(SourceSchedulerPolicy::new(3))
        .build()
        .unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.source_chunks_planned, 2);
    assert_eq!(metrics.source_chunks_attempted, 2);
    assert_eq!(metrics.source_chunks_committed, 2);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2, 1]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[10, 10, 20]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_local_parquet_source_cache_reuses_equivalent_relowered_scan_contract() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-source-cache-reuse");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let _input_schema = write_row_grouped_input_parquet(&path);
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
    ]));
    let cache = Arc::new(GpuLocalParquetSourceCache::new(64 * 1024 * 1024));
    let file_size = fs::metadata(&path).unwrap().len();

    let build_plan = || {
        let source = LocalParquetSource::from_files(
            vec![LocalParquetFile::new(&path).with_stable_id("stable-input").with_file_size_bytes(file_size)],
            Arc::clone(&source_schema),
            vec!["key".to_string(), "payload".to_string()],
        );

        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(source));
        builder.finish_with_host_sink(source)
    };
    let build_context = || {
        ExecutionContext::builder()
            .with_shared_runtime(Arc::clone(&runtime))
            .with_memory_control_policy(MemoryControlPolicy::Bounded {
                device_budget_bytes: 8 * 1024 * 1024 * 1024,
                max_source_chunk_bytes: 1024 * 1024,
                max_row_groups_per_chunk: 1,
                max_retry_attempts: 1,
                min_retry_chunk_row_groups: 1,
            })
            .with_local_parquet_source_cache(Arc::clone(&cache))
            .build()
            .unwrap()
    };

    let first = execute_query_plan(&build_context(), &build_plan()).unwrap();
    let first_metrics = first.metrics();
    assert_eq!(first_metrics.source_chunks_planned, 3);
    assert_eq!(first_metrics.source_cache_hits, 0);
    assert_eq!(first_metrics.source_cache_misses, 3);
    assert_eq!(first_metrics.source_cache_inserts, 3);
    assert_eq!(first_metrics.source_cache_insert_skipped, 0);
    assert!(first_metrics.source_cache_retained_bytes > 0);
    assert_eq!(cache.len().unwrap(), 3);
    let retained_bytes = cache.retained_device_bytes().unwrap();
    assert!(retained_bytes > 0);

    let second = execute_query_plan(&build_context(), &build_plan()).unwrap();
    let second_metrics = second.metrics();
    assert_eq!(second_metrics.source_chunks_planned, 3);
    assert_eq!(second_metrics.source_cache_hits, 3);
    assert_eq!(second_metrics.source_cache_misses, 0);
    assert_eq!(second_metrics.source_cache_inserts, 0);
    assert_eq!(second_metrics.source_cache_insert_skipped, 0);
    assert_eq!(second_metrics.source_cache_retained_bytes, retained_bytes);
    assert!(second_metrics.reservation_events.iter().any(|event| {
        event.outcome.as_str() == "granted"
            && event.reservation_kind == ReservationKind::SourceCacheRetained
            && event.memory_boundary == "source_cache_retained"
            && event.requested_bytes == retained_bytes
    }));

    let QueryOutput::HostArrow(batch) = second.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 6);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2, 1, 3, 2, 3]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[10, 10, 20, 20, 30, 5]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_local_parquet_source_cache_binds_execution_frame_schema_with_pure_filter_columns() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-source-cache-execution-frame-schema");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    write_row_grouped_input_parquet(&path);
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("payload", DataType::Int64, false),
        Field::new("key", DataType::Int32, false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
    let cache = Arc::new(GpuLocalParquetSourceCache::new(64 * 1024 * 1024));
    let file_size = fs::metadata(&path).unwrap().len();

    let build_plan = || {
        let source = LocalParquetSource::from_files(
            vec![LocalParquetFile::new(&path).with_stable_id("stable-input").with_file_size_bytes(file_size)],
            Arc::clone(&source_schema),
            vec!["payload".to_string(), "key".to_string()],
        )
        .with_column_roles(vec!["key".to_string()], vec!["payload".to_string()]);

        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(source));
        let filter = builder.add_filter(
            source,
            Expr::binary(Expr::column("payload"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int64(20))),
            Arc::clone(&source_schema),
        );
        let projection =
            builder.add_projection(filter, projection_outputs(vec![Expr::column("key")], Arc::clone(&output_schema)));
        builder.finish_with_host_sink(projection)
    };
    let build_context = || {
        ExecutionContext::builder()
            .with_shared_runtime(Arc::clone(&runtime))
            .with_memory_control_policy(MemoryControlPolicy::Bounded {
                device_budget_bytes: 8 * 1024 * 1024 * 1024,
                max_source_chunk_bytes: 1024 * 1024,
                max_row_groups_per_chunk: 1,
                max_retry_attempts: 1,
                min_retry_chunk_row_groups: 1,
            })
            .with_local_parquet_source_cache(Arc::clone(&cache))
            .build()
            .unwrap()
    };

    let first = execute_query_plan(&build_context(), &build_plan()).unwrap();
    assert_eq!(first.metrics().source_cache_misses, 3);
    assert_eq!(first.metrics().source_cache_inserts, 3);

    let second = execute_query_plan(&build_context(), &build_plan()).unwrap();
    assert_eq!(second.metrics().source_cache_hits, 3);
    assert_eq!(second.metrics().source_cache_misses, 0);

    let QueryOutput::HostArrow(batch) = second.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 3, 2]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_local_parquet_source_cache_retained_bytes_are_admitted_before_cache_hits() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-source-cache-admission");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let _input_schema = write_row_grouped_input_parquet(&path);
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
    ]));
    let cache = Arc::new(GpuLocalParquetSourceCache::new(64 * 1024 * 1024));
    let file_size = fs::metadata(&path).unwrap().len();

    let build_plan = || {
        let source = LocalParquetSource::from_files(
            vec![LocalParquetFile::new(&path).with_stable_id("stable-input").with_file_size_bytes(file_size)],
            Arc::clone(&source_schema),
            vec!["key".to_string(), "payload".to_string()],
        );

        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(source));
        builder.finish_with_host_sink(source)
    };
    let warm_context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .with_local_parquet_source_cache(Arc::clone(&cache))
        .build()
        .unwrap();
    execute_query_plan(&warm_context, &build_plan()).unwrap();
    let retained_bytes = cache.retained_device_bytes().unwrap();
    assert!(retained_bytes > 0);

    let constrained_context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: retained_bytes.saturating_sub(1),
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .with_local_parquet_source_cache(Arc::clone(&cache))
        .build()
        .unwrap();
    let error = match execute_query_plan(&constrained_context, &build_plan()) {
        Ok(_) => panic!("source-cache retained bytes must be reserved before cache hits can be reused"),
        Err(error) => error,
    };
    assert_eq!(error.code(), QueryErrorCode::MemoryReservationFailed);
    assert_eq!(error.fact_value("memory_boundary"), Some("source_cache_retained"));
    assert_eq!(error.fact_value("reservation_kind"), Some("source_cache_retained"));
    let requested_bytes = error.fact_value("requested_bytes").and_then(|value| value.parse::<u64>().ok());
    assert_eq!(requested_bytes, Some(retained_bytes));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_local_parquet_source_cache_reclaims_retained_bytes_under_reservation_pressure() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-source-cache-pressure");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let _input_schema = write_row_grouped_input_parquet(&path);
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
    ]));
    let cache = Arc::new(GpuLocalParquetSourceCache::new(64 * 1024 * 1024));
    let file_size = fs::metadata(&path).unwrap().len();

    let build_plan = |source_path: &std::path::Path, stable_id: &str| {
        let source = LocalParquetSource::from_files(
            vec![
                LocalParquetFile::new(source_path)
                    .with_stable_id(stable_id)
                    .with_file_size_bytes(file_size)
                    .with_row_group_bytes(vec![8192, 8192, 8192])
                    .with_projected_row_group_bytes(vec![8192, 8192, 8192], vec![8192, 8192, 8192]),
            ],
            Arc::clone(&source_schema),
            vec!["key".to_string(), "payload".to_string()],
        );

        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(source));
        builder.finish_gpu_resident(source)
    };
    let build_context = |device_budget_bytes, max_row_groups_per_chunk| {
        ExecutionContext::builder()
            .with_shared_runtime(Arc::clone(&runtime))
            .with_memory_control_policy(MemoryControlPolicy::Bounded {
                device_budget_bytes,
                max_source_chunk_bytes: 1024 * 1024,
                max_row_groups_per_chunk,
                max_retry_attempts: 1,
                min_retry_chunk_row_groups: 1,
            })
            .with_local_parquet_source_cache(Arc::clone(&cache))
            .build()
            .unwrap()
    };

    let mut max_source_read_request = None;
    for index in 0..3 {
        let warm_path = dir.join(format!("warm-{index}.parquet"));
        fs::copy(&path, &warm_path).unwrap();
        let stable_id = format!("stable-input-a{index}");
        let warm =
            execute_query_plan(&build_context(8 * 1024 * 1024 * 1024, 3), &build_plan(&warm_path, &stable_id)).unwrap();
        assert_eq!(warm.metrics().source_cache_inserts, 1);
        max_source_read_request = warm
            .metrics()
            .reservation_events
            .iter()
            .filter(|event| event.reservation_kind == ReservationKind::SourceRead)
            .map(|event| event.requested_bytes)
            .max();
    }
    let retained_bytes = cache.retained_device_bytes().unwrap();
    assert!(retained_bytes > 0);
    let max_source_read_request =
        max_source_read_request.expect("warm source read should record a reservation request");

    let pressured = execute_query_plan(
        &build_context(retained_bytes.saturating_add(max_source_read_request).saturating_sub(1), 3),
        &build_plan(&path, "stable-input-b"),
    )
    .unwrap();
    assert_eq!(pressured.metrics().source_cache_hits, 0);
    assert_eq!(pressured.metrics().source_cache_misses, 1);
    assert!(pressured.metrics().source_cache_inserts > 0);
    assert_eq!(pressured.metrics().reservation_denied_count, 0);

    let QueryOutput::GpuResident(handle) = pressured.output() else {
        panic!("gpu sink should retain one GPU result handle");
    };
    assert_eq!(handle.num_rows(), 6);
    assert_eq!(handle.num_columns(), 2);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_query_executes_group_only_aggregate_as_distinct_keys() {
    let dir = unique_temp_dir("cudf-query-engine-native-group-only-aggregate");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_input_parquet(&runtime, &path);
    let output_schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string()],
    );
    let aggregate = AggregateSpec::new(vec![Expr::column("key")], vec![]);
    let sort = SortSpec::new(vec![SortExpr::new(Expr::column("key"), false, false)]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, Arc::clone(&output_schema));
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&output_schema));
    let plan = builder.finish_with_host_sink(sort);

    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));
    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(batch.num_columns(), 1);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2, 3]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_executes_group_only_aggregate_with_budgeted_distinct() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-group-only-aggregate");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_final_merge_retry_input_parquet(&path, 4);
    let output_schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string()],
    );
    let aggregate = AggregateSpec::new(vec![Expr::column("key")], vec![]);
    let sort = SortSpec::new(vec![SortExpr::new(Expr::column("key"), false, false)]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, Arc::clone(&output_schema));
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&output_schema));
    let plan = builder.finish_with_host_sink(sort);

    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.source_chunks_committed, 3);
    assert_eq!(metrics.final_merge_attempts, 1);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 4);
    assert_eq!(batch.num_columns(), 1);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[0, 1, 2, 3]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_query_executes_multi_file_direct_parquet_source() {
    let dir = unique_temp_dir("cudf-query-engine-native-execution-multi-file");
    let first_path = dir.join("part-000.parquet");
    let second_path = dir.join("part-001.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_input_parquet(&runtime, &first_path);
    write_input_parquet(&runtime, &second_path);

    let projected_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("x00", DataType::Int64, false),
    ]));
    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("sum_x00", DataType::Int64, true),
    ]));
    let files = vec![
        LocalParquetFile::new(&first_path).with_file_size_bytes(fs::metadata(&first_path).unwrap().len()),
        LocalParquetFile::new(&second_path).with_file_size_bytes(fs::metadata(&second_path).unwrap().len()),
    ];
    let source = LocalParquetSource::from_files(
        files,
        Arc::clone(&input_schema),
        vec!["key".to_string(), "m00".to_string(), "m01".to_string(), "payload".to_string()],
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
        SortExpr::new(Expr::column("key"), false, false),
    ]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let filter = builder.add_filter(source, filter, Arc::clone(&input_schema));
    let projection = builder.add_projection(filter, projection_outputs(projection, projected_schema));
    let aggregate = builder.add_aggregate(projection, aggregate, Arc::clone(&aggregated_schema));
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&aggregated_schema));
    let limit = builder.add_limit(sort, LimitSpec::fetch(2), aggregated_schema);
    let plan = builder.finish_with_host_sink(limit);

    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.parquet_files, 2);
    assert_eq!(metrics.read_columns, 8);
    assert_eq!(metrics.input_batches, 2);
    assert_eq!(metrics.input_rows, 8);
    assert_eq!(metrics.boundary.host_to_device_uploads, 0);
    assert_eq!(metrics.boundary.device_to_host_materializations, 1);
    assert_eq!(metrics.boundary.device_to_host_materialized_rows, 2);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[3, 1]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[808, 606]);

    fs::remove_dir_all(dir).unwrap();
}

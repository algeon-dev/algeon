#[test]
fn native_local_parquet_source_scan_limit_applies_before_parent_operators() {
    let dir = unique_temp_dir("nexus-query-engine-native-source-scan-limit");
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

    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime)).unwrap();
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
    let dir = unique_temp_dir("nexus-query-engine-native-missing-nullable-source-columns");
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

    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime)).unwrap();
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
    let dir = unique_temp_dir("nexus-query-engine-bounded-missing-nullable-source-columns");
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
fn bounded_generic_local_parquet_pipeline_uses_required_only_source_scheduler() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-generic-source-scheduler");
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
    )
    .with_column_roles(vec!["key".to_string(), "m00".to_string(), "m01".to_string()], vec!["payload".to_string()]);
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
    assert_eq!(metrics.source_read_prefetches_started, 0);
    assert_eq!(metrics.source_read_prefetches_completed, 0);
    assert_eq!(metrics.source_read_prefetches_skipped, 0);
    assert_eq!(metrics.source_read_prefetches_abandoned, 0);
    let production_prefetches = metrics
        .reservation_events
        .iter()
        .filter(|event| {
            event.outcome.as_str() == "granted"
                && matches!(event.memory_boundary.as_str(), "source_chunk_prefetch" | "join_source_read_prefetch")
        })
        .count();
    assert_eq!(production_prefetches, 0);
    assert!(metrics.reservation_events.iter().any(|event| {
        event.outcome.as_str() == "granted"
            && event.memory_boundary == "bounded_generic_source_read"
            && event.action == MemoryReservationEventAction::Reserve
    }));
    assert!(
        metrics.reservation_events.iter().all(|event| {
            !(event.memory_boundary == "bounded_generic_source_read"
                && event.action == MemoryReservationEventAction::Grow)
        }),
        "source-adjacent generic filtering/projection should not grow raw source-read reservations to observed bytes"
    );
    assert!(
        metrics.reservation_events.iter().any(|event| {
            event.memory_boundary == "bounded_generic_source_filter_output"
                && event.action == MemoryReservationEventAction::Transfer
        }),
        "source-adjacent generic filtering should transfer ownership to the filtered output boundary"
    );
    let first_filter_transfer = metrics
        .reservation_events
        .iter()
        .position(|event| {
            event.memory_boundary == "bounded_generic_source_filter_output"
                && event.action == MemoryReservationEventAction::Transfer
        })
        .expect("source-adjacent generic filtering should transfer each chunk before source concat");
    let source_read_reserves_before_first_filter = metrics.reservation_events[..first_filter_transfer]
        .iter()
        .filter(|event| {
            event.outcome.as_str() == "granted"
                && event.memory_boundary == "bounded_generic_source_read"
                && event.action == MemoryReservationEventAction::Reserve
        })
        .count();
    assert_eq!(
        source_read_reserves_before_first_filter, 1,
        "generic source-adjacent streaming should filter/project the first chunk before reading later chunks"
    );
    assert!(
        metrics.reservation_events.iter().any(|event| {
            event.memory_boundary == "bounded_generic_source_projection_output"
                && event.action == MemoryReservationEventAction::Transfer
        }),
        "source-adjacent generic projection should transfer ownership to the projected output boundary"
    );
    assert!(
        metrics.reservation_events.iter().any(|event| {
            event.memory_boundary == "bounded_generic_source_concat_output"
                && event.action == MemoryReservationEventAction::Reserve
        }),
        "multi-chunk generic source concat output should be traced separately from raw source reads"
    );

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
fn bounded_generic_local_parquet_source_batches_adjacent_files() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-generic-source-batch");
    let paths = [dir.join("input-0.parquet"), dir.join("input-1.parquet"), dir.join("input-2.parquet")];
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    write_input_parquet(&runtime, &paths[0]);
    write_input_parquet(&runtime, &paths[1]);
    write_input_parquet(&runtime, &paths[2]);
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
    ]));

    let files = paths
        .iter()
        .map(|path| {
            LocalParquetFile::new(path)
                .with_file_size_bytes(fs::metadata(path).unwrap().len())
                .with_row_group_rows(vec![4])
        })
        .collect::<Vec<_>>();
    let source = LocalParquetSource::from_files(files, source_schema, vec!["key".to_string(), "payload".to_string()]);

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
        .build()
        .unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.source_chunks_planned, 3);
    assert_eq!(metrics.source_chunks_attempted, 3);
    assert_eq!(metrics.source_chunks_committed, 3);
    assert_eq!(metrics.row_groups_planned, 3);
    assert_eq!(metrics.row_groups_committed, 3);
    assert_eq!(metrics.source_read_runtime_rows.len(), 1);
    let source_read_row = &metrics.source_read_runtime_rows[0];
    assert_eq!(source_read_row.read_kind, "chunk_batch");
    assert_eq!(source_read_row.read_strategy, "local_kvikio");
    assert_eq!(source_read_row.chunk_count, 3);
    assert_eq!(source_read_row.file_count, 3);
    assert_eq!(source_read_row.row_group_count, 3);
    assert_eq!(source_read_row.read_columns_per_file, 2);
    assert_eq!(source_read_row.read_columns_total, 6);
    assert_eq!(source_read_row.parquet_read_evidence.input_kind, "local_path_multi");
    assert_eq!(source_read_row.parquet_read_evidence.remote_read_mode, "none");
    assert_eq!(source_read_row.parquet_read_evidence.backend, "libcudf_path_reader");

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 12);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2, 1, 3, 1, 2, 1, 3, 1, 2, 1, 3]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[5, 10, 20, 20, 5, 10, 20, 20, 5, 10, 20, 20]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_generic_local_parquet_source_scan_limit_trims_to_required_row_group_prefix() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-source-scan-limit");
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
            max_source_chunk_bytes: 8 * 1024 * 1024 * 1024,
            max_row_groups_per_chunk: 8,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .with_source_scheduler_policy(SourceSchedulerPolicy::new(3))
        .build()
        .unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.source_chunks_planned, 1);
    assert_eq!(metrics.source_chunks_attempted, 1);
    assert_eq!(metrics.source_chunks_committed, 1);
    assert_eq!(metrics.row_groups_planned, 2);
    assert_eq!(metrics.row_groups_committed, 2);

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
    let dir = unique_temp_dir("nexus-query-engine-bounded-source-cache-reuse");
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
    let build_context = |runtime: &Arc<cudf::Runtime>| {
        ExecutionContext::builder()
            .with_shared_runtime(Arc::clone(runtime))
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

    let first = execute_query_plan(&build_context(&runtime), &build_plan()).unwrap();
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

    let second = execute_query_plan(&build_context(&runtime), &build_plan()).unwrap();
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

    let other_runtime = Arc::new(cudf::Runtime::new().unwrap());
    let third = execute_query_plan(&build_context(&other_runtime), &build_plan()).unwrap();
    let third_metrics = third.metrics();
    assert_eq!(third_metrics.source_cache_hits, 0);
    assert_eq!(third_metrics.source_cache_misses, 3);
    assert_eq!(third_metrics.source_cache_inserts, 3);
    assert_eq!(cache.len().unwrap(), 6);
    let retained_after_other_runtime = cache.retained_device_bytes().unwrap();
    assert_eq!(retained_after_other_runtime, retained_bytes.saturating_mul(2));

    let fourth = execute_query_plan(&build_context(&runtime), &build_plan()).unwrap();
    let fourth_metrics = fourth.metrics();
    assert_eq!(fourth_metrics.source_cache_hits, 3);
    assert_eq!(fourth_metrics.source_cache_misses, 0);
    assert_eq!(fourth_metrics.source_cache_inserts, 0);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn query_service_shutdown_drains_source_cache_and_waits_for_escaped_pinned_output() {
    use std::{
        sync::mpsc,
        time::{Duration, Instant},
    };

    let dir = unique_temp_dir("nexus-query-engine-service-source-cache-shutdown");
    let path = dir.join("input.parquet");
    write_row_grouped_input_parquet(&path);
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
    ]));
    let config = QueryServiceConfig {
        device_capacity_bytes: Some(128 * 1024 * 1024),
        max_concurrent_queries: 1,
        max_queries_per_device: Some(1),
        max_row_groups_per_chunk: 8,
        local_parquet_source_cache_max_retained_bytes: Some(32 * 1024 * 1024),
        slot_guarantee: SlotGuarantee::GuaranteedMinimum,
        ..QueryServiceConfig::default()
    };
    let tenants = vec![QueryServiceTenantConfig::with_grant_range(
        TenantId::new("cache-shutdown"),
        8 * 1024 * 1024,
        32 * 1024 * 1024,
    )];
    let service = QueryService::new(config.clone(), tenants.clone()).unwrap();
    let file_size = fs::metadata(&path).unwrap().len();
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(&path).with_stable_id("service-cache-shutdown-input").with_file_size_bytes(file_size),
        ],
        Arc::clone(&source_schema),
        vec!["key".to_string(), "payload".to_string()],
    );
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let plan = builder.finish_gpu_resident(source);

    let query = service.admit_query("cache-shutdown").unwrap();
    let tenant_context = query.acquire_context().unwrap();
    let cache = tenant_context.context().local_parquet_source_cache().expect("service source cache");
    let result = execute_query_plan(tenant_context.context(), &plan).unwrap();
    assert_eq!(result.metrics().source_cache_inserts, 1);
    let (output, _metrics) = result.into_parts();
    let QueryOutput::GpuResident(escaped) = output else {
        panic!("GPU-resident source plan must return an escaped pinned handle");
    };
    assert!(!cache.is_empty().unwrap());
    assert!(cache.retained_device_bytes().unwrap() > 0);
    query.finish();
    drop(tenant_context);
    drop(query);

    let (finished_tx, finished_rx) = mpsc::channel();
    let closing_service = service.clone();
    let close = std::thread::spawn(move || {
        let result = closing_service.close_and_wait_for_quiescence(Duration::from_secs(10));
        finished_tx.send(()).unwrap();
        result
    });
    let cache_deadline = Instant::now() + Duration::from_secs(10);
    while !cache.is_empty().unwrap() {
        assert!(Instant::now() < cache_deadline, "shutdown did not drain the populated source cache");
        std::thread::yield_now();
    }
    assert_eq!(cache.retained_device_bytes().unwrap(), 0);
    assert!(
        finished_rx.recv_timeout(Duration::from_millis(20)).is_err(),
        "escaped GPU output must keep the invalidated service epoch live after cache drain"
    );

    drop(escaped);
    finished_rx
        .recv_timeout(Duration::from_secs(10))
        .expect("service shutdown must finish after the escaped output drops");
    close.join().unwrap().expect("query service closes cleanly");
    drop(cache);
    drop(service);

    let replacement = QueryService::new(config, tenants).expect("replacement service acquires the released claim");
    replacement.close_and_wait_for_quiescence(Duration::from_secs(10)).unwrap();
    drop(replacement);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn whole_source_local_parquet_source_cache_does_not_use_active_grant_capacity() {
    let dir = unique_temp_dir("nexus-query-engine-whole-source-cache-grant");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    write_row_grouped_input_parquet(&path);
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
    ]));
    let governor = MemoryGovernor::new(runtime.device().ordinal(), 64 * 1024 * 1024).unwrap();
    let grant = governor
        .request(GrantRequest::new(AdmissionRequestId::new(200), governor.capacity_bytes(), governor.capacity_bytes()))
        .unwrap();
    assert_eq!(governor.free_bytes().unwrap(), 0);
    let cache = Arc::new(GpuLocalParquetSourceCache::new_governor_backed(64 * 1024 * 1024, governor.clone()));
    GpuLocalParquetSourceCache::register_governor_pressure_observer(&cache).unwrap();
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
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_grant(grant)
        .with_local_parquet_source_cache(Arc::clone(&cache))
        .build()
        .unwrap();

    let cold = execute_query_plan(&context, &build_plan()).unwrap();
    assert_eq!(cold.metrics().source_cache_hits, 0);
    assert_eq!(cold.metrics().source_cache_misses, 1);
    assert_eq!(cold.metrics().source_cache_inserts, 0);
    assert_eq!(cold.metrics().source_cache_insert_skipped, 1);
    assert_eq!(cache.retained_device_bytes().unwrap(), 0);
    let report = governor.report().unwrap();
    assert_eq!(report.evictable_bytes, 0);
    assert_eq!(report.outstanding_bytes, governor.capacity_bytes());
    assert_eq!(report.free_bytes, 0);

    let warm = execute_query_plan(&context, &build_plan()).unwrap();
    assert_eq!(warm.metrics().source_cache_hits, 0);
    assert_eq!(warm.metrics().source_cache_misses, 1);
    assert_eq!(warm.metrics().source_cache_inserts, 0);
    assert_eq!(warm.metrics().source_cache_insert_skipped, 1);
    assert_eq!(cache.retained_device_bytes().unwrap(), 0);

    let QueryOutput::HostArrow(batch) = warm.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 6);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2, 1, 3, 2, 3]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn whole_source_local_parquet_source_cache_miss_projection_pins_gpu_result_until_drop() {
    let dir = unique_temp_dir("nexus-query-engine-whole-source-cache-projection-pin");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    write_row_grouped_input_parquet(&path);
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
    let governor = MemoryGovernor::new(runtime.device().ordinal(), 64 * 1024 * 1024).unwrap();
    let cache = Arc::new(GpuLocalParquetSourceCache::new_governor_backed(64 * 1024 * 1024, governor.clone()));
    GpuLocalParquetSourceCache::register_governor_pressure_observer(&cache).unwrap();
    let file_size = fs::metadata(&path).unwrap().len();

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_stable_id("stable-input").with_file_size_bytes(file_size)],
        Arc::clone(&source_schema),
        vec!["key".to_string(), "payload".to_string()],
    );
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let projection =
        builder.add_projection(source, projection_outputs(vec![Expr::column("key")], Arc::clone(&output_schema)));
    let plan = builder.finish_gpu_resident(projection);
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_local_parquet_source_cache(Arc::clone(&cache))
        .build()
        .unwrap();

    let cold = execute_query_plan(&context, &plan).unwrap();
    assert_eq!(cold.metrics().source_cache_hits, 0);
    assert_eq!(cold.metrics().source_cache_misses, 1);
    assert_eq!(cold.metrics().source_cache_inserts, 1);
    let QueryOutput::GpuResident(handle) = cold.output() else {
        panic!("gpu sink should retain the projected cached frame");
    };
    assert_eq!(handle.num_rows(), 6);
    assert_eq!(handle.num_columns(), 1);
    let retained_bytes = cache.retained_device_bytes().unwrap();
    assert!(retained_bytes > 0);
    assert!(
        governor
            .try_request(GrantRequest::new(
                AdmissionRequestId::new(201),
                governor.capacity_bytes(),
                governor.capacity_bytes()
            ))
            .unwrap()
            .is_none(),
        "pressure admission must not evict a cold-miss source-cache entry pinned by projected GPU output"
    );
    assert_eq!(cache.retained_device_bytes().unwrap(), retained_bytes);
    assert_eq!(governor.evictable_bytes().unwrap(), retained_bytes);

    drop(cold);
    let grant = governor
        .try_request(GrantRequest::new(
            AdmissionRequestId::new(202),
            governor.capacity_bytes(),
            governor.capacity_bytes(),
        ))
        .unwrap()
        .expect("unpinned cold-miss cache entry should be evicted under pressure");
    assert_eq!(grant.budget_bytes(), governor.capacity_bytes());
    assert_eq!(cache.retained_device_bytes().unwrap(), 0);
    assert_eq!(governor.evictable_bytes().unwrap(), 0);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn whole_source_local_parquet_source_cache_capacity_skip_returns_uncached_result() {
    let dir = unique_temp_dir("nexus-query-engine-whole-source-cache-capacity-skip");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    write_row_grouped_input_parquet(&path);
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
    ]));
    let cache = Arc::new(GpuLocalParquetSourceCache::new(1));
    let file_size = fs::metadata(&path).unwrap().len();
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_stable_id("stable-input").with_file_size_bytes(file_size)],
        Arc::clone(&source_schema),
        vec!["key".to_string(), "payload".to_string()],
    );
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let plan = builder.finish_with_host_sink(source);
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_local_parquet_source_cache(Arc::clone(&cache))
        .build()
        .unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    assert_eq!(result.metrics().source_cache_hits, 0);
    assert_eq!(result.metrics().source_cache_misses, 1);
    assert_eq!(result.metrics().source_cache_inserts, 0);
    assert_eq!(result.metrics().source_cache_insert_skipped, 1);
    assert_eq!(cache.retained_device_bytes().unwrap(), 0);
    assert!(cache.is_empty().unwrap());

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should still materialize the uncached cold-miss result");
    };
    assert_eq!(batch.num_rows(), 6);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2, 1, 3, 2, 3]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn whole_source_local_parquet_source_cache_single_input_union_pins_gpu_result_until_drop() {
    let dir = unique_temp_dir("nexus-query-engine-whole-source-cache-union-pin");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    write_row_grouped_input_parquet(&path);
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
    ]));
    let governor = MemoryGovernor::new(runtime.device().ordinal(), 64 * 1024 * 1024).unwrap();
    let cache = Arc::new(GpuLocalParquetSourceCache::new_governor_backed(64 * 1024 * 1024, governor.clone()));
    GpuLocalParquetSourceCache::register_governor_pressure_observer(&cache).unwrap();
    let file_size = fs::metadata(&path).unwrap().len();
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_stable_id("stable-input").with_file_size_bytes(file_size)],
        Arc::clone(&source_schema),
        vec!["key".to_string(), "payload".to_string()],
    );
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let union = builder.add_union(vec![source], Arc::clone(&source_schema));
    let plan = builder.finish_gpu_resident(union);
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_local_parquet_source_cache(Arc::clone(&cache))
        .build()
        .unwrap();

    let cold = execute_query_plan(&context, &plan).unwrap();
    assert_eq!(cold.metrics().source_cache_hits, 0);
    assert_eq!(cold.metrics().source_cache_misses, 1);
    assert_eq!(cold.metrics().source_cache_inserts, 1);
    let QueryOutput::GpuResident(handle) = cold.output() else {
        panic!("gpu sink should retain the union output");
    };
    assert_eq!(handle.num_rows(), 6);
    assert_eq!(handle.num_columns(), 2);
    let retained_bytes = cache.retained_device_bytes().unwrap();
    assert!(retained_bytes > 0);
    assert!(
        governor
            .try_request(GrantRequest::new(
                AdmissionRequestId::new(203),
                governor.capacity_bytes(),
                governor.capacity_bytes()
            ))
            .unwrap()
            .is_none(),
        "pressure admission must not evict a source-cache entry pinned through single-input union"
    );

    drop(cold);
    let grant = governor
        .try_request(GrantRequest::new(
            AdmissionRequestId::new(204),
            governor.capacity_bytes(),
            governor.capacity_bytes(),
        ))
        .unwrap()
        .expect("unpinned union output should release source-cache pressure protection");
    assert_eq!(grant.budget_bytes(), governor.capacity_bytes());
    assert_eq!(cache.retained_device_bytes().unwrap(), 0);
    assert_eq!(governor.evictable_bytes().unwrap(), 0);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn whole_source_local_parquet_source_cache_edge_normalize_pins_alias_until_gpu_result_drop() {
    let dir = unique_temp_dir("nexus-query-engine-whole-source-cache-edge-normalize-pin");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    write_row_grouped_input_parquet(&path);
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
    ]));
    let governor = MemoryGovernor::new(runtime.device().ordinal(), 64 * 1024 * 1024).unwrap();
    let cache = Arc::new(GpuLocalParquetSourceCache::new_governor_backed(64 * 1024 * 1024, governor.clone()));
    GpuLocalParquetSourceCache::register_governor_pressure_observer(&cache).unwrap();
    let file_size = fs::metadata(&path).unwrap().len();
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_stable_id("stable-input").with_file_size_bytes(file_size)],
        Arc::clone(&source_schema),
        vec!["key".to_string(), "payload".to_string()],
    );
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let edge_normalize = builder.add_edge_normalize(
        source,
        nexus_query_engine::plan::EdgeNormalizeSpec {
            edge_columns: nexus_query_engine::plan::GraphEdgeColumnBindings::new("key", "payload", None::<&str>),
            add_reverse: false,
            dedup: nexus_query_engine::plan::EdgeDedupStrategy::None,
            weight_combiner: nexus_query_engine::plan::EdgeWeightCombiner::None,
        },
        Arc::clone(&source_schema),
    );
    let plan = builder.finish_gpu_resident(edge_normalize);
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_local_parquet_source_cache(Arc::clone(&cache))
        .build()
        .unwrap();

    let cold = execute_query_plan(&context, &plan).unwrap();
    assert_eq!(cold.metrics().source_cache_hits, 0);
    assert_eq!(cold.metrics().source_cache_misses, 1);
    assert_eq!(cold.metrics().source_cache_inserts, 1);
    let QueryOutput::GpuResident(handle) = cold.output() else {
        panic!("gpu sink should retain the edge-normalized output");
    };
    assert_eq!(handle.num_rows(), 6);
    assert_eq!(handle.num_columns(), 2);
    let retained_bytes = cache.retained_device_bytes().unwrap();
    assert!(retained_bytes > 0);
    assert!(
        governor
            .try_request(GrantRequest::new(
                AdmissionRequestId::new(206),
                governor.capacity_bytes(),
                governor.capacity_bytes()
            ))
            .unwrap()
            .is_none(),
        "pressure admission must not evict source-cache entry while edge-normalized GPU output aliases it"
    );

    drop(cold);
    let grant = governor
        .try_request(GrantRequest::new(
            AdmissionRequestId::new(207),
            governor.capacity_bytes(),
            governor.capacity_bytes(),
        ))
        .unwrap()
        .expect("unpinned edge-normalize source-cache entry should be evicted under pressure");
    assert_eq!(grant.budget_bytes(), governor.capacity_bytes());
    assert_eq!(cache.retained_device_bytes().unwrap(), 0);
    assert_eq!(governor.evictable_bytes().unwrap(), 0);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_local_parquet_source_cache_governor_owns_evictable_bytes() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-source-cache-governor");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let _input_schema = write_row_grouped_input_parquet(&path);
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
    ]));
    let governor = MemoryGovernor::new(runtime.device().ordinal(), 64 * 1024 * 1024).unwrap();
    let cache = Arc::new(GpuLocalParquetSourceCache::new_governor_backed(64 * 1024 * 1024, governor.clone()));
    GpuLocalParquetSourceCache::register_governor_pressure_observer(&cache).unwrap();
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
    assert_eq!(first.metrics().source_cache_hits, 0);
    assert_eq!(first.metrics().source_cache_inserts, 3);
    assert!(
        first.metrics().reservation_events.iter().all(|event| {
            event.reservation_kind != ReservationKind::SourceCacheRetained || event.charged_delta_bytes == 0
        }),
        "governor-backed cache inserts must not charge a per-query source-cache baseline"
    );
    let retained_bytes = cache.retained_device_bytes().unwrap();
    assert!(retained_bytes > 0);
    let report = governor.report().unwrap();
    assert_eq!(report.evictable_bytes, retained_bytes);
    assert_eq!(report.outstanding_bytes, 0);
    assert_eq!(report.free_bytes, report.capacity_bytes.saturating_sub(retained_bytes));

    let second = execute_query_plan(&build_context(), &build_plan()).unwrap();
    assert_eq!(second.metrics().source_cache_hits, 3);
    assert_eq!(second.metrics().source_cache_misses, 0);
    assert_eq!(governor.evictable_bytes().unwrap(), retained_bytes);

    drop(first);
    drop(second);
    let pressure_grant = governor
        .try_request(GrantRequest::new(
            AdmissionRequestId::new(99),
            governor.capacity_bytes(),
            governor.capacity_bytes(),
        ))
        .unwrap()
        .expect("admission should evict unpinned source-cache entries");
    assert_eq!(pressure_grant.budget_bytes(), governor.capacity_bytes());
    assert_eq!(cache.retained_device_bytes().unwrap(), 0);
    assert_eq!(governor.evictable_bytes().unwrap(), 0);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_local_parquet_source_cache_does_not_use_active_grant_capacity_for_inserts() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-source-cache-no-grant-fallback");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let _input_schema = write_row_grouped_input_parquet(&path);
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
    ]));
    let governor = MemoryGovernor::new(runtime.device().ordinal(), 64 * 1024 * 1024).unwrap();
    let grant = governor
        .request(GrantRequest::new(AdmissionRequestId::new(205), governor.capacity_bytes(), governor.capacity_bytes()))
        .unwrap();
    assert_eq!(governor.free_bytes().unwrap(), 0);
    let cache = Arc::new(GpuLocalParquetSourceCache::new_governor_backed(64 * 1024 * 1024, governor.clone()));
    GpuLocalParquetSourceCache::register_governor_pressure_observer(&cache).unwrap();
    let file_size = fs::metadata(&path).unwrap().len();

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_stable_id("stable-input").with_file_size_bytes(file_size)],
        Arc::clone(&source_schema),
        vec!["key".to_string(), "payload".to_string()],
    );
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let plan = builder.finish_with_host_sink(source);
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_grant(grant)
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

    let result = execute_query_plan(&context, &plan).unwrap();
    assert_eq!(result.metrics().source_cache_hits, 0);
    assert_eq!(result.metrics().source_cache_misses, 3);
    assert_eq!(result.metrics().source_cache_inserts, 0);
    assert_eq!(result.metrics().source_cache_insert_skipped, 3);
    assert_eq!(cache.retained_device_bytes().unwrap(), 0);
    assert_eq!(governor.evictable_bytes().unwrap(), 0);
    let report = governor.report().unwrap();
    assert_eq!(report.outstanding_bytes, governor.capacity_bytes());
    assert_eq!(report.free_bytes, 0);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize even when bounded source-cache inserts are skipped");
    };
    assert_eq!(batch.num_rows(), 6);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2, 1, 3, 2, 3]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_local_parquet_source_cache_pressure_skips_gpu_resident_hit_until_drop() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-source-cache-governor-pinned");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let _input_schema = write_row_grouped_input_parquet(&path);
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
    ]));
    let governor = MemoryGovernor::new(runtime.device().ordinal(), 64 * 1024 * 1024).unwrap();
    let cache = Arc::new(GpuLocalParquetSourceCache::new_governor_backed(64 * 1024 * 1024, governor.clone()));
    GpuLocalParquetSourceCache::register_governor_pressure_observer(&cache).unwrap();
    let file_size = fs::metadata(&path).unwrap().len();

    let build_plan = || {
        let source = LocalParquetSource::from_files(
            vec![LocalParquetFile::new(&path).with_stable_id("stable-input").with_file_size_bytes(file_size)],
            Arc::clone(&source_schema),
            vec!["key".to_string(), "payload".to_string()],
        );

        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(source));
        builder.finish_gpu_resident(source)
    };
    let build_context = || {
        ExecutionContext::builder()
            .with_shared_runtime(Arc::clone(&runtime))
            .with_memory_control_policy(MemoryControlPolicy::Bounded {
                device_budget_bytes: 8 * 1024 * 1024 * 1024,
                max_source_chunk_bytes: 1024 * 1024,
                max_row_groups_per_chunk: 3,
                max_retry_attempts: 1,
                min_retry_chunk_row_groups: 1,
            })
            .with_local_parquet_source_cache(Arc::clone(&cache))
            .build()
            .unwrap()
    };

    let cold = execute_query_plan(&build_context(), &build_plan()).unwrap();
    assert_eq!(cold.metrics().source_cache_inserts, 1);
    drop(cold);
    let retained_bytes = cache.retained_device_bytes().unwrap();
    assert!(retained_bytes > 0);

    let warm = execute_query_plan(&build_context(), &build_plan()).unwrap();
    assert_eq!(warm.metrics().source_cache_hits, 1);
    let QueryOutput::GpuResident(_handle) = warm.output() else {
        panic!("test plan should retain the cached frame as a GPU-resident output");
    };
    assert!(
        governor
            .try_request(GrantRequest::new(
                AdmissionRequestId::new(100),
                governor.capacity_bytes(),
                governor.capacity_bytes()
            ))
            .unwrap()
            .is_none(),
        "pressure admission must not evict a pinned cache entry"
    );
    assert_eq!(cache.retained_device_bytes().unwrap(), retained_bytes);
    assert_eq!(governor.evictable_bytes().unwrap(), retained_bytes);

    drop(warm);
    let grant = governor
        .try_request(GrantRequest::new(
            AdmissionRequestId::new(101),
            governor.capacity_bytes(),
            governor.capacity_bytes(),
        ))
        .unwrap()
        .expect("unpinned cache entry should be evicted under pressure");
    assert_eq!(grant.budget_bytes(), governor.capacity_bytes());
    assert_eq!(cache.retained_device_bytes().unwrap(), 0);
    assert_eq!(governor.evictable_bytes().unwrap(), 0);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_local_parquet_source_cache_source_adjacent_aggregate_warm_hit_skips_decode_and_reinsert() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-source-cache-scheduled-aggregate");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_final_merge_retry_input_parquet(&path, 4);
    let output_schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
    let governor = MemoryGovernor::new(runtime.device().ordinal(), 64 * 1024 * 1024).unwrap();
    let cache = Arc::new(GpuLocalParquetSourceCache::new_governor_backed(64 * 1024 * 1024, governor));
    let file_size = fs::metadata(&path).unwrap().len();

    let build_plan = || {
        let source = LocalParquetSource::from_files(
            vec![LocalParquetFile::new(&path).with_file_size_bytes(file_size)],
            Arc::clone(&input_schema),
            vec!["key".to_string()],
        );
        let aggregate = AggregateSpec::new(vec![Expr::column("key")], vec![]);
        let sort = SortSpec::new(vec![SortExpr::new(Expr::column("key"), false, false)]);

        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(source));
        let filter = builder.add_filter(
            source,
            Expr::binary(Expr::column("key"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int32(2))),
            Arc::clone(&input_schema),
        );
        let aggregate = builder.add_aggregate(filter, aggregate, Arc::clone(&output_schema));
        let sort = builder.add_sort(aggregate, sort, Arc::clone(&output_schema));
        builder.finish_with_host_sink(sort)
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
            .with_source_scheduler_policy(SourceSchedulerPolicy::new(2))
            .with_local_parquet_source_cache(Arc::clone(&cache))
            .build()
            .unwrap()
    };

    let cold = execute_query_plan(&build_context(), &build_plan()).unwrap();
    assert_eq!(cold.metrics().source_cache_hits, 0);
    assert!(cold.metrics().source_cache_inserts > 0);

    let warm = execute_query_plan(&build_context(), &build_plan()).unwrap();
    let warm_metrics = warm.metrics();
    assert!(warm_metrics.source_cache_hits > 0);
    assert_eq!(warm_metrics.source_cache_misses, 0);
    assert_eq!(warm_metrics.source_cache_inserts, 0);
    assert_eq!(warm_metrics.source_cache_insert_skipped, 0);

    let QueryOutput::HostArrow(batch) = warm.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[2, 3]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_local_parquet_source_cache_q06_like_global_aggregate_reuses_all_source_chunks() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-source-cache-q06-like-global-aggregate");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_key_payload_row_grouped_parquet(&path, 1);
    let projected_schema = Arc::new(Schema::new(vec![Field::new("discounted_payload", DataType::Int64, false)]));
    let output_schema = Arc::new(Schema::new(vec![Field::new("revenue", DataType::Int64, true)]));
    let governor = MemoryGovernor::new(runtime.device().ordinal(), 64 * 1024 * 1024).unwrap();
    let cache = Arc::new(GpuLocalParquetSourceCache::new_governor_backed(64 * 1024 * 1024, governor));
    let file_size = fs::metadata(&path).unwrap().len();

    let build_plan = || {
        let source = LocalParquetSource::from_files(
            vec![LocalParquetFile::new(&path).with_file_size_bytes(file_size)],
            Arc::clone(&input_schema),
            vec!["key".to_string(), "payload".to_string()],
        );
        let aggregate = AggregateSpec::new(
            vec![],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("discounted_payload"), "revenue")],
        );

        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(source));
        let filter = builder.add_filter(
            source,
            Expr::binary(Expr::column("key"), BinaryOp::LtEq, Expr::literal(LiteralValue::Int32(2))),
            Arc::clone(&input_schema),
        );
        let projection = builder.add_projection(
            filter,
            projection_outputs(
                vec![
                    Expr::binary(Expr::column("payload"), BinaryOp::Multiply, Expr::literal(LiteralValue::Int64(2)))
                        .alias("discounted_payload"),
                ],
                Arc::clone(&projected_schema),
            ),
        );
        let aggregate = builder.add_aggregate(projection, aggregate, Arc::clone(&output_schema));
        builder.finish_with_host_sink(aggregate)
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
            .with_source_scheduler_policy(SourceSchedulerPolicy::new(2))
            .with_local_parquet_source_cache(Arc::clone(&cache))
            .build()
            .unwrap()
    };

    let cold = execute_query_plan(&build_context(), &build_plan()).unwrap();
    let cold_metrics = cold.metrics();
    assert_eq!(cold_metrics.source_cache_hits, 0);
    assert_eq!(cold_metrics.source_cache_misses, 4);
    assert_eq!(cold_metrics.source_cache_inserts, 4);
    assert_eq!(cold_metrics.source_chunks_committed, 4);

    let warm = execute_query_plan(&build_context(), &build_plan()).unwrap();
    let warm_metrics = warm.metrics();
    assert_eq!(warm_metrics.source_cache_hits, 4);
    assert_eq!(warm_metrics.source_cache_misses, 0);
    assert_eq!(warm_metrics.source_cache_inserts, 0);
    assert_eq!(
        warm_metrics.source_cache_insert_skipped, 4,
        "scheduled warm hits may re-attempt admission, but must resolve as already present without decoding"
    );
    assert_eq!(warm_metrics.source_chunks_committed, 4);

    let QueryOutput::HostArrow(batch) = warm.output() else {
        panic!("host sink should materialize the q06-like global aggregate");
    };
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(int64_array(batch, 0).values().as_ref(), &[70]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_local_parquet_source_cache_binds_execution_frame_schema_with_pure_filter_columns() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-source-cache-execution-frame-schema");
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
    let second_metrics = second.metrics();
    assert_eq!(second_metrics.source_cache_hits, 3);
    assert_eq!(second_metrics.source_cache_misses, 0);
    assert!(
        second_metrics.reservation_events.iter().any(|event| {
            event.memory_boundary == "source_cache_borrowed"
                && event.action == MemoryReservationEventAction::AliasAcquire
        }),
        "cache-aware source-adjacent streaming should borrow retained raw chunks before filtering"
    );
    assert!(
        second_metrics.reservation_events.iter().any(|event| {
            event.memory_boundary == "bounded_generic_source_filter_output"
                && event.action == MemoryReservationEventAction::Transfer
        }),
        "cache-aware source-adjacent filtering should transfer ownership to the filtered chunk boundary"
    );
    assert!(
        second_metrics.reservation_events.iter().any(|event| {
            event.memory_boundary == "bounded_generic_source_projection_output"
                && event.action == MemoryReservationEventAction::Transfer
        }),
        "cache-aware source-adjacent projection should transfer ownership to the projected chunk boundary"
    );
    assert!(
        second_metrics.reservation_events.iter().all(|event| event.memory_boundary != "bounded_generic_filter"),
        "cache-aware source-adjacent streaming should not materialize the cached raw source before filtering"
    );

    let QueryOutput::HostArrow(batch) = second.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 3, 2]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_local_parquet_source_cache_retained_bytes_are_admitted_before_cache_hits() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-source-cache-admission");
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
    let dir = unique_temp_dir("nexus-query-engine-bounded-source-cache-pressure");
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

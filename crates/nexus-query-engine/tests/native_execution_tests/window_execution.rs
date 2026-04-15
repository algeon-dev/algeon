use super::*;

#[test]
fn native_cumulative_aggregate_window_restores_input_order_and_carries_null_rows() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("grp", DataType::Int32, false),
        Field::new("ord", DataType::Int32, false),
        Field::new("value", DataType::Int64, true),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("grp", DataType::Int32, false),
        Field::new("ord", DataType::Int32, false),
        Field::new("value", DataType::Int64, true),
        Field::new("running_sum", DataType::Int64, true),
        Field::new("running_max", DataType::Int64, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&input_schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 1, 2, 1, 2, 2])),
            Arc::new(Int32Array::from(vec![2, 1, 2, 3, 1, 3])),
            Arc::new(Int64Array::from(vec![None, Some(10), None, Some(5), Some(7), Some(3)])),
        ],
    )
    .unwrap();
    let frame = DataFrame::from_record_batch(&runtime, &batch).unwrap();

    let mut builder = QueryPlanBuilder::new();
    let source =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("window_input", Arc::clone(&input_schema))));
    let window = builder.add_window(
        source,
        WindowSpec::new(
            vec![Expr::column("grp")],
            vec![SortExpr::new(Expr::column("ord"), false, false)],
            vec![
                WindowFunction::cumulative_aggregate(AggregateFunction::Sum, Expr::column("value"), "running_sum"),
                WindowFunction::cumulative_aggregate(AggregateFunction::Max, Expr::column("value"), "running_max"),
            ],
        ),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(window);
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime)).unwrap();
    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "window_input".to_string(),
        GpuResultHandle::new(&context, GpuDataFrame::new(frame), Arc::clone(&input_schema)).unwrap(),
    );

    let result = execute_query_plan_with_gpu_inputs(&context, &plan, &gpu_inputs).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };

    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 1, 2, 1, 2, 2]);
    assert_eq!(int32_array(batch, 1).values().as_ref(), &[2, 1, 2, 3, 1, 3]);
    assert_eq!(int64_array(batch, 2).iter().collect::<Vec<_>>(), vec![None, Some(10), None, Some(5), Some(7), Some(3)]);
    assert_eq!(
        int64_array(batch, 3).iter().collect::<Vec<_>>(),
        vec![Some(10), Some(10), Some(7), Some(15), Some(7), Some(10)]
    );
    assert_eq!(
        int64_array(batch, 4).iter().collect::<Vec<_>>(),
        vec![Some(10), Some(10), Some(7), Some(10), Some(7), Some(7)]
    );
}

#[test]
fn native_lag_window_shifts_within_partition_and_restores_input_order() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("grp", DataType::Int32, false),
        Field::new("ord", DataType::Int32, false),
        Field::new("value", DataType::Int64, true),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("grp", DataType::Int32, false),
        Field::new("ord", DataType::Int32, false),
        Field::new("value", DataType::Int64, true),
        Field::new("prev_value", DataType::Int64, true),
    ]));
    // Input rows are deliberately not sorted by (grp, ord) so that the
    // sort-and-restore step inside execute_lag_lead_window has work to do.
    let batch = RecordBatch::try_new(
        Arc::clone(&input_schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 1, 2, 1, 2, 2])),
            Arc::new(Int32Array::from(vec![2, 1, 2, 3, 1, 3])),
            Arc::new(Int64Array::from(vec![Some(10), Some(20), Some(100), Some(30), Some(200), Some(300)])),
        ],
    )
    .unwrap();
    let frame = DataFrame::from_record_batch(&runtime, &batch).unwrap();

    let mut builder = QueryPlanBuilder::new();
    let source =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("window_input", Arc::clone(&input_schema))));
    let window = builder.add_window(
        source,
        WindowSpec::new(
            vec![Expr::column("grp")],
            vec![SortExpr::new(Expr::column("ord"), false, false)],
            vec![WindowFunction::lag(Expr::column("value"), 1, None, "prev_value")],
        ),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(window);
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime)).unwrap();
    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "window_input".to_string(),
        GpuResultHandle::new(&context, GpuDataFrame::new(frame), Arc::clone(&input_schema)).unwrap(),
    );

    let result = execute_query_plan_with_gpu_inputs(&context, &plan, &gpu_inputs).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };

    // Pass-through columns must keep input row order verbatim.
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 1, 2, 1, 2, 2]);
    assert_eq!(int32_array(batch, 1).values().as_ref(), &[2, 1, 2, 3, 1, 3]);
    assert_eq!(
        int64_array(batch, 2).iter().collect::<Vec<_>>(),
        vec![Some(10), Some(20), Some(100), Some(30), Some(200), Some(300)]
    );
    // Within each partition rows are reasoned about in `ord` order:
    //   grp=1: ord=1 (value=20) -> ord=2 (value=10) -> ord=3 (value=30)
    //   grp=2: ord=1 (value=200) -> ord=2 (value=100) -> ord=3 (value=300)
    // So LAG(value, 1) in input row order is:
    //   row 0 (grp=1, ord=2): 20
    //   row 1 (grp=1, ord=1): NULL (first in partition)
    //   row 2 (grp=2, ord=2): 200
    //   row 3 (grp=1, ord=3): 10
    //   row 4 (grp=2, ord=1): NULL (first in partition)
    //   row 5 (grp=2, ord=3): 100
    assert_eq!(
        int64_array(batch, 3).iter().collect::<Vec<_>>(),
        vec![Some(20), None, Some(200), Some(10), None, Some(100)]
    );
}

#[test]
fn native_lag_window_groups_null_partition_keys_together() {
    // SQL window `PARTITION BY` treats NULL keys as a single group, so the
    // grouped shift must pass `include_null_keys` through to libcudf;
    // otherwise null-keyed rows are dropped and the row-index reattach
    // misaligns the output column.
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("grp", DataType::Int32, true),
        Field::new("ord", DataType::Int32, false),
        Field::new("value", DataType::Int64, true),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("grp", DataType::Int32, true),
        Field::new("ord", DataType::Int32, false),
        Field::new("value", DataType::Int64, true),
        Field::new("prev_value", DataType::Int64, true),
    ]));
    let grp = Int32Array::from(vec![Some(1), None, Some(1), None, Some(1)]);
    let ord = Int32Array::from(vec![2, 1, 1, 2, 3]);
    let value = Int64Array::from(vec![Some(20), Some(100), Some(10), Some(200), Some(30)]);
    let batch =
        RecordBatch::try_new(Arc::clone(&input_schema), vec![Arc::new(grp), Arc::new(ord), Arc::new(value)]).unwrap();
    let frame = DataFrame::from_record_batch(&runtime, &batch).unwrap();

    let mut builder = QueryPlanBuilder::new();
    let source =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("window_input", Arc::clone(&input_schema))));
    let window = builder.add_window(
        source,
        WindowSpec::new(
            vec![Expr::column("grp")],
            vec![SortExpr::new(Expr::column("ord"), false, false)],
            vec![WindowFunction::lag(Expr::column("value"), 1, None, "prev_value")],
        ),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(window);
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime)).unwrap();
    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "window_input".to_string(),
        GpuResultHandle::new(&context, GpuDataFrame::new(frame), Arc::clone(&input_schema)).unwrap(),
    );

    let result = execute_query_plan_with_gpu_inputs(&context, &plan, &gpu_inputs).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };

    // Output keeps input row count.
    assert_eq!(batch.num_rows(), 5);
    // Within-partition logical order:
    //   grp=Some(1) sorted by ord: ord=1 (value=10) -> ord=2 (value=20) -> ord=3 (value=30)
    //   grp=None  sorted by ord: ord=1 (value=100) -> ord=2 (value=200)
    // LAG(value, 1) in input row order:
    //   row 0 (grp=Some(1), ord=2): 10
    //   row 1 (grp=None,    ord=1): NULL (first in null partition)
    //   row 2 (grp=Some(1), ord=1): NULL (first in Some(1) partition)
    //   row 3 (grp=None,    ord=2): 100
    //   row 4 (grp=Some(1), ord=3): 20
    assert_eq!(int64_array(batch, 3).iter().collect::<Vec<_>>(), vec![Some(10), None, None, Some(100), Some(20)]);
}

#[test]
fn native_lead_window_with_explicit_default_fills_partition_edges() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("grp", DataType::Int32, false),
        Field::new("ord", DataType::Int32, false),
        Field::new("value", DataType::Int64, false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("grp", DataType::Int32, false),
        Field::new("ord", DataType::Int32, false),
        Field::new("value", DataType::Int64, false),
        // LEAD with a non-null default returns a non-nullable column when the
        // input is non-nullable and the default is non-null. The validator
        // enforces this contract.
        Field::new("next_value", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&input_schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 1, 2, 1, 2, 2])),
            Arc::new(Int32Array::from(vec![2, 1, 2, 3, 1, 3])),
            Arc::new(Int64Array::from(vec![10, 20, 100, 30, 200, 300])),
        ],
    )
    .unwrap();
    let frame = DataFrame::from_record_batch(&runtime, &batch).unwrap();

    let mut builder = QueryPlanBuilder::new();
    let source =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("window_input", Arc::clone(&input_schema))));
    let window = builder.add_window(
        source,
        WindowSpec::new(
            vec![Expr::column("grp")],
            vec![SortExpr::new(Expr::column("ord"), false, false)],
            vec![WindowFunction::lead(
                Expr::column("value"),
                1,
                Some(nexus_query_engine::LiteralValue::Int64(-1)),
                "next_value",
            )],
        ),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(window);
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime)).unwrap();
    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "window_input".to_string(),
        GpuResultHandle::new(&context, GpuDataFrame::new(frame), Arc::clone(&input_schema)).unwrap(),
    );

    let result = execute_query_plan_with_gpu_inputs(&context, &plan, &gpu_inputs).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };

    // LEAD(value, 1, -1) in input row order:
    //   grp=1 sorted by ord: 20 → 10 → 30 → end
    //   grp=2 sorted by ord: 200 → 100 → 300 → end
    //   row 0 (grp=1, ord=2, value=10): next = 30
    //   row 1 (grp=1, ord=1, value=20): next = 10
    //   row 2 (grp=2, ord=2, value=100): next = 300
    //   row 3 (grp=1, ord=3, value=30): next = -1 (end of partition)
    //   row 4 (grp=2, ord=1, value=200): next = 100
    //   row 5 (grp=2, ord=3, value=300): next = -1 (end of partition)
    assert_eq!(int64_array(batch, 3).values().as_ref(), &[30, 10, 300, -1, 100, -1]);
}

#[test]
fn native_row_number_window_counts_one_based_rows_per_partition() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("grp", DataType::Int32, false),
        Field::new("ord", DataType::Int32, false),
        Field::new("value", DataType::Int64, true),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("grp", DataType::Int32, false),
        Field::new("ord", DataType::Int32, false),
        Field::new("value", DataType::Int64, true),
        Field::new("rn", DataType::UInt64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&input_schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 1, 2, 1, 2, 2])),
            Arc::new(Int32Array::from(vec![2, 1, 2, 3, 1, 3])),
            Arc::new(Int64Array::from(vec![Some(20), Some(10), None, Some(30), Some(7), Some(3)])),
        ],
    )
    .unwrap();
    let frame = DataFrame::from_record_batch(&runtime, &batch).unwrap();

    let mut builder = QueryPlanBuilder::new();
    let source =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("window_input", Arc::clone(&input_schema))));
    let sort = builder.add_sort(
        source,
        SortSpec::new(vec![
            SortExpr::new(Expr::column("grp"), false, false),
            SortExpr::new(Expr::column("ord"), false, false),
        ]),
        Arc::clone(&input_schema),
    );
    let window = builder.add_window(
        sort,
        WindowSpec::new(
            vec![Expr::column("grp")],
            vec![SortExpr::new(Expr::column("ord"), false, false)],
            vec![WindowFunction::row_number("rn")],
        ),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(window);
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime)).unwrap();
    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "window_input".to_string(),
        GpuResultHandle::new(&context, GpuDataFrame::new(frame), Arc::clone(&input_schema)).unwrap(),
    );

    let result = execute_query_plan_with_gpu_inputs(&context, &plan, &gpu_inputs).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };

    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 1, 1, 2, 2, 2]);
    assert_eq!(int32_array(batch, 1).values().as_ref(), &[1, 2, 3, 1, 2, 3]);
    assert_eq!(uint64_array(batch, 3).values().as_ref(), &[1, 2, 3, 1, 2, 3]);
}

#[test]
fn whole_source_cache_window_output_pins_alias_until_gpu_result_drop() {
    let dir = unique_temp_dir("nexus-query-engine-whole-source-cache-window-pin");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    write_row_grouped_input_parquet(&path);
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
        Field::new("rn", DataType::UInt64, false),
    ]));
    let governor = MemoryGovernor::new(runtime.device().ordinal(), 64 * 1024 * 1024).unwrap();
    let cache = Arc::new(GpuLocalParquetSourceCache::new_governor_backed(64 * 1024 * 1024, governor.clone()));
    GpuLocalParquetSourceCache::register_governor_pressure_observer(&cache).unwrap();
    let file_size = fs::metadata(&path).unwrap().len();

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_stable_id("stable-window-input").with_file_size_bytes(file_size)],
        Arc::clone(&source_schema),
        vec!["key".to_string(), "payload".to_string()],
    );
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let window = builder.add_window(
        source,
        WindowSpec::new(
            vec![Expr::column("key")],
            vec![SortExpr::new(Expr::column("payload"), false, false)],
            vec![WindowFunction::row_number("rn")],
        ),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_gpu_resident(window);
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_local_parquet_source_cache(Arc::clone(&cache))
        .build()
        .unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    assert_eq!(result.metrics().source_cache_misses, 1);
    assert_eq!(result.metrics().source_cache_inserts, 1);
    let QueryOutput::GpuResident(handle) = result.output() else {
        panic!("gpu sink should retain the window output");
    };
    assert_eq!(handle.num_rows(), 6);
    assert_eq!(handle.num_columns(), 3);
    let retained_bytes = cache.retained_device_bytes().unwrap();
    assert!(retained_bytes > 0);
    assert!(
        governor
            .try_request(GrantRequest::new(301, governor.capacity_bytes(), governor.capacity_bytes()))
            .unwrap()
            .is_none(),
        "pressure admission must not evict source-cache entry while window GPU output aliases it"
    );

    drop(result);
    let grant = governor
        .try_request(GrantRequest::new(302, governor.capacity_bytes(), governor.capacity_bytes()))
        .unwrap()
        .expect("unpinned window source-cache entry should be evicted under pressure");
    assert_eq!(grant.budget_bytes(), governor.capacity_bytes());
    assert_eq!(cache.retained_device_bytes().unwrap(), 0);

    fs::remove_dir_all(dir).unwrap();
}

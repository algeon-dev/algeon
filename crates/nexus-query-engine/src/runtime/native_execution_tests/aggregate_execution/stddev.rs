use super::*;

#[test]
fn grouped_stddev_samp_can_use_gpu_native_tolerant_contract() {
    let dir = unique_temp_dir("nexus-query-engine-gpu-native-stddev-samp");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_stddev_input_parquet(&runtime, &path);
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("stdev_i32", DataType::Float64, true),
        Field::new("avg_i32", DataType::Float64, true),
    ]));

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "v_i32".to_string()],
    );
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![
            AggregateExpr::new(AggregateFunction::StddevSamp, Expr::column("v_i32"), "stdev_i32"),
            AggregateExpr::new(AggregateFunction::Avg, Expr::column("v_i32"), "avg_i32"),
        ],
    );
    let sort = SortSpec::new(vec![SortExpr::new(Expr::column("key"), false, false)]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, Arc::clone(&output_schema));
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&output_schema));
    let plan = builder.finish_with_host_sink(sort);

    let context = ExecutionContext::builder()
        .with_shared_runtime(runtime)
        .with_stddev_samp_execution_mode(StddevSampExecutionMode::GpuNativeTolerant)
        .build()
        .unwrap();
    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };

    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2, 3, 4]);
    assert_eq!(float64_array(batch, 1).value(0), 2.0_f64.sqrt());
    assert!(float64_array(batch, 1).is_null(1));
    assert!(float64_array(batch, 1).is_null(2));
    assert!(float64_array(batch, 1).is_null(3));
    assert_eq!(float64_array(batch, 2).value(0), 3.0);
    assert!(float64_array(batch, 2).is_null(3));
    assert_eq!(
        aggregate_strategy_fact(result.metrics(), "selected_strategy").as_deref(),
        Some("stddev_samp_gpu_native_tolerant")
    );
    assert_eq!(
        aggregate_strategy_fact(result.metrics(), "stddev_samp_execution_mode").as_deref(),
        Some("gpu_native_tolerant")
    );
    assert_eq!(aggregate_strategy_fact(result.metrics(), "floating_statistical_aggregate").as_deref(), Some("true"));
    assert_eq!(
        aggregate_strategy_fact(result.metrics(), "statistical_aggregate_execution_mode").as_deref(),
        Some("gpu_native_tolerant")
    );
    assert_eq!(
        aggregate_strategy_fact(result.metrics(), "statistical_aggregate_tolerance_contract").as_deref(),
        Some("float_tolerant")
    );
    assert_eq!(
        aggregate_strategy_fact(result.metrics(), "statistical_aggregate_gpu_native_formula").as_deref(),
        Some("algebraic_moments")
    );
    assert_eq!(
        aggregate_strategy_fact(result.metrics(), "statistical_aggregate_ddof").as_deref(),
        Some("stddev_samp:1")
    );
    assert_eq!(aggregate_strategy_fact(result.metrics(), "stddev_samp_ddof").as_deref(), Some("1"));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn grouped_tolerant_statistical_aggregate_preserves_decimal_avg_scale() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let dir = unique_temp_dir("nexus-query-engine-gpu-native-statistical-decimal-avg");
    let path = dir.join("input.parquet");
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("v", DataType::Float64, true),
        Field::new("amount", DataType::Decimal128(7, 2), true),
    ]));
    let input = RecordBatch::try_new(
        Arc::clone(&input_schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 1, 2])),
            Arc::new(Float64Array::from(vec![Some(2.0), Some(4.0), Some(7.0)])),
            Arc::new(Decimal128Array::from_iter_values([3465, 0, 6930]).with_precision_and_scale(7, 2).unwrap()),
        ],
    )
    .unwrap();
    DataFrame::from_record_batch(&runtime, &input).unwrap().write_parquet(&path).unwrap();
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("stdev_v", DataType::Float64, true),
        Field::new("avg_amount", DataType::Decimal128(11, 6), true),
    ]));

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "v".to_string(), "amount".to_string()],
    );
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![
            AggregateExpr::new(AggregateFunction::StddevSamp, Expr::column("v"), "stdev_v"),
            AggregateExpr::new(AggregateFunction::Avg, Expr::column("amount"), "avg_amount"),
        ],
    );
    let sort = SortSpec::new(vec![SortExpr::new(Expr::column("key"), false, false)]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, Arc::clone(&output_schema));
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&output_schema));
    let plan = builder.finish_with_host_sink(sort);

    let context = ExecutionContext::builder()
        .with_shared_runtime(runtime)
        .with_stddev_samp_execution_mode(StddevSampExecutionMode::GpuNativeTolerant)
        .build()
        .unwrap();
    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };

    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2]);
    assert_eq!(float64_array(batch, 1).value(0), 2.0_f64.sqrt());
    assert!(float64_array(batch, 1).is_null(1));
    let avg_amount = decimal128_array(batch, 2);
    assert_eq!(avg_amount.precision(), 11);
    assert_eq!(avg_amount.scale(), 6);
    assert_eq!(avg_amount.value(0), 17_325_000);
    assert_eq!(avg_amount.value(1), 69_300_000);
    assert_eq!(
        aggregate_strategy_fact(result.metrics(), "statistical_aggregate_gpu_native_formula").as_deref(),
        Some("algebraic_moments")
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn grouped_floating_statistical_family_can_use_gpu_native_tolerant_contract() {
    let dir = unique_temp_dir("nexus-query-engine-gpu-native-statistical-aggregates");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_stddev_input_parquet(&runtime, &path);
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("stddev_pop_i32", DataType::Float64, true),
        Field::new("var_samp_i32", DataType::Float64, true),
        Field::new("var_pop_i32", DataType::Float64, true),
    ]));

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "v_i32".to_string()],
    );
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![
            AggregateExpr::new(AggregateFunction::StddevPop, Expr::column("v_i32"), "stddev_pop_i32"),
            AggregateExpr::new(AggregateFunction::VarSamp, Expr::column("v_i32"), "var_samp_i32"),
            AggregateExpr::new(AggregateFunction::VarPop, Expr::column("v_i32"), "var_pop_i32"),
        ],
    );
    let sort = SortSpec::new(vec![SortExpr::new(Expr::column("key"), false, false)]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, Arc::clone(&output_schema));
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&output_schema));
    let plan = builder.finish_with_host_sink(sort);

    let context = ExecutionContext::builder()
        .with_shared_runtime(runtime)
        .with_statistical_aggregate_execution_mode(StddevSampExecutionMode::GpuNativeTolerant)
        .build()
        .unwrap();
    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };

    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2, 3, 4]);
    assert_eq!(float64_array(batch, 1).value(0), 1.0);
    assert_eq!(float64_array(batch, 2).value(0), 2.0);
    assert_eq!(float64_array(batch, 3).value(0), 1.0);
    assert_eq!(float64_array(batch, 1).value(1), 0.0);
    assert!(float64_array(batch, 2).is_null(1));
    assert_eq!(float64_array(batch, 3).value(1), 0.0);
    assert!(float64_array(batch, 1).is_null(3));
    assert!(float64_array(batch, 2).is_null(3));
    assert!(float64_array(batch, 3).is_null(3));
    assert_eq!(
        aggregate_strategy_fact(result.metrics(), "selected_strategy").as_deref(),
        Some("floating_statistical_gpu_native_tolerant")
    );
    assert_eq!(
        aggregate_strategy_fact(result.metrics(), "statistical_aggregate_functions").as_deref(),
        Some("stddev_pop,var_samp,var_pop")
    );
    assert_eq!(
        aggregate_strategy_fact(result.metrics(), "statistical_aggregate_ddof").as_deref(),
        Some("stddev_pop:0,var_samp:1,var_pop:0")
    );
    assert_eq!(
        aggregate_strategy_fact(result.metrics(), "statistical_aggregate_gpu_native_formula").as_deref(),
        Some("algebraic_moments")
    );
    assert_eq!(aggregate_strategy_fact(result.metrics(), "stddev_samp_execution_mode"), None);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_tolerant_grouped_statistical_aggregate_streams_partial_moments() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let dir = unique_temp_dir("nexus-query-engine-bounded-tolerant-statistical-moments");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("v", DataType::Float64, true),
        Field::new("d", DataType::Decimal128(7, 2), true),
    ]));
    let right_schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("grp", DataType::Int32, false)]));
    let left_batch = RecordBatch::try_new(
        Arc::clone(&left_schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 1, 2, 2, 3])),
            Arc::new(Float64Array::from(vec![Some(2.0), Some(4.0), Some(10.0), Some(14.0), None])),
            Arc::new(
                Decimal128Array::from_iter_values([3465_i128, 0, 6930, 6930, 1234])
                    .with_precision_and_scale(7, 2)
                    .unwrap(),
            ),
        ],
    )
    .unwrap();
    let right_batch = RecordBatch::try_new(
        Arc::clone(&right_schema),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3])), Arc::new(Int32Array::from(vec![10, 20, 30]))],
    )
    .unwrap();
    write_batch_with_row_group_size(Arc::clone(&left_schema), &left_path, &left_batch, 1);
    DataFrame::from_record_batch(&runtime, &right_batch).unwrap().write_parquet(&right_path).unwrap();

    let join_schema = Arc::new(Schema::new(vec![
        Field::new("grp", DataType::Int32, false),
        Field::new("v", DataType::Float64, true),
        Field::new("d", DataType::Decimal128(7, 2), true),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("grp", DataType::Int32, false),
        Field::new("stdev_v", DataType::Float64, true),
        Field::new("avg_v", DataType::Float64, true),
        Field::new("avg_d", DataType::Decimal128(11, 6), true),
    ]));
    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(&left_path)
                .with_file_size_bytes(fs::metadata(&left_path).unwrap().len())
                .with_projected_row_group_bytes(vec![2048; 5], vec![2048; 5])
                .with_row_group_rows(vec![1; 5]),
        ],
        Arc::clone(&left_schema),
        vec!["k".to_string(), "v".to_string(), "d".to_string()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&right_path).with_file_size_bytes(fs::metadata(&right_path).unwrap().len())],
        Arc::clone(&right_schema),
        vec!["k".to_string(), "grp".to_string()],
    )));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                nexus_query_engine::expr::ColumnRef::input(0, "k", "k"),
                nexus_query_engine::expr::ColumnRef::input(1, "k", "k"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::right(nexus_query_engine::expr::ColumnRef::input(1, "grp", "grp"), "grp"),
            JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "v", "v"), "v"),
            JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "d", "d"), "d"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&join_schema),
    );
    let aggregate = builder.add_aggregate(
        join,
        AggregateSpec::new(
            vec![Expr::column("grp")],
            vec![
                AggregateExpr::new(AggregateFunction::StddevSamp, Expr::column("v"), "stdev_v"),
                AggregateExpr::new(AggregateFunction::Avg, Expr::column("v"), "avg_v"),
                AggregateExpr::new(AggregateFunction::Avg, Expr::column("d"), "avg_d"),
            ],
        ),
        Arc::clone(&aggregate_schema),
    );
    let sort = builder.add_sort(
        aggregate,
        SortSpec::new(vec![SortExpr::new(Expr::column("grp"), false, false)]),
        Arc::clone(&aggregate_schema),
    );
    let plan = builder.finish_with_host_sink(sort);

    let context = ExecutionContext::builder()
        .with_shared_runtime(runtime)
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 128 * 1024 * 1024,
            max_source_chunk_bytes: 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .with_statistical_aggregate_execution_mode(StddevSampExecutionMode::GpuNativeTolerant)
        .build()
        .unwrap();
    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };

    assert_eq!(int32_array(batch, 0).values().as_ref(), &[10, 20, 30]);
    assert_eq!(float64_array(batch, 1).value(0), 2.0_f64.sqrt());
    assert_eq!(float64_array(batch, 1).value(1), 8.0_f64.sqrt());
    assert!(float64_array(batch, 1).is_null(2));
    assert_eq!(float64_array(batch, 2).value(0), 3.0);
    assert_eq!(float64_array(batch, 2).value(1), 12.0);
    assert!(float64_array(batch, 2).is_null(2));
    let avg_d = decimal128_array(batch, 3);
    assert_eq!(avg_d.precision(), 11);
    assert_eq!(avg_d.scale(), 6);
    assert_eq!(avg_d.value(0), 17_325_000);
    assert_eq!(avg_d.value(1), 69_300_000);
    assert_eq!(avg_d.value(2), 12_340_000);
    assert!(aggregate_strategy_fact_matches(
        result.metrics(),
        "selected_strategy",
        "floating_statistical_gpu_native_tolerant_partial_moments"
    ));
    assert!(aggregate_strategy_fact_matches(
        result.metrics(),
        "selected_strategy",
        "floating_statistical_gpu_native_tolerant_final_moments"
    ));
    assert!(aggregate_strategy_fact_matches(
        result.metrics(),
        "statistical_aggregate_gpu_native_formula",
        "algebraic_moments"
    ));
    assert!(
        result.metrics().partial_aggregate_count > 1,
        "bounded join stream should execute more than one statistical moment partial"
    );
    let statistical_partial_work_reserves = result
        .metrics()
        .reservation_events
        .iter()
        .filter(|event| {
            event.memory_boundary == "compiled_statistical_aggregate_partial_state_work"
                && event.action == MemoryReservationEventAction::Reserve
        })
        .collect::<Vec<_>>();
    assert!(
        statistical_partial_work_reserves.len() > 1,
        "bounded statistical moments should reserve every partial-state upper bound before execution"
    );
    assert!(
        statistical_partial_work_reserves.iter().all(|event| event.requested_bytes > 0),
        "statistical partial-state work reservations must carry a closed-form nonzero bound"
    );
    assert!(
        result.metrics().reservation_events.iter().all(|event| {
            event.memory_boundary != "compiled_statistical_aggregate_partial_state"
                || event.action != MemoryReservationEventAction::Grow
        }),
        "statistical partial-state retention must not grow after observing output"
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_tolerant_filtered_grouped_statistical_aggregate_uses_filter_preparation() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let dir = unique_temp_dir("nexus-query-engine-bounded-filtered-statistical-moments");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("v", DataType::Float64, true),
        Field::new("keep", DataType::Boolean, false),
    ]));
    let right_schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("grp", DataType::Int32, false)]));
    let left_batch = RecordBatch::try_new(
        Arc::clone(&left_schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 1, 1, 2, 2, 3])),
            Arc::new(Float64Array::from(vec![Some(2.0), Some(4.0), Some(100.0), Some(10.0), Some(14.0), Some(7.0)])),
            Arc::new(BooleanArray::from(vec![true, true, false, true, false, false])),
        ],
    )
    .unwrap();
    let right_batch = RecordBatch::try_new(
        Arc::clone(&right_schema),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3])), Arc::new(Int32Array::from(vec![10, 20, 30]))],
    )
    .unwrap();
    write_batch_with_row_group_size(Arc::clone(&left_schema), &left_path, &left_batch, 1);
    DataFrame::from_record_batch(&runtime, &right_batch).unwrap().write_parquet(&right_path).unwrap();

    let join_schema = Arc::new(Schema::new(vec![
        Field::new("grp", DataType::Int32, false),
        Field::new("v", DataType::Float64, true),
        Field::new("keep", DataType::Boolean, false),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("grp", DataType::Int32, false),
        Field::new("stdev_v", DataType::Float64, true),
        Field::new("avg_v", DataType::Float64, true),
    ]));
    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(&left_path)
                .with_file_size_bytes(fs::metadata(&left_path).unwrap().len())
                .with_projected_row_group_bytes(vec![2048; 6], vec![2048; 6])
                .with_row_group_rows(vec![1; 6]),
        ],
        Arc::clone(&left_schema),
        vec!["k".to_string(), "v".to_string(), "keep".to_string()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&right_path).with_file_size_bytes(fs::metadata(&right_path).unwrap().len())],
        Arc::clone(&right_schema),
        vec!["k".to_string(), "grp".to_string()],
    )));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                nexus_query_engine::expr::ColumnRef::input(0, "k", "k"),
                nexus_query_engine::expr::ColumnRef::input(1, "k", "k"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::right(nexus_query_engine::expr::ColumnRef::input(1, "grp", "grp"), "grp"),
            JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "v", "v"), "v"),
            JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "keep", "keep"), "keep"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&join_schema),
    );
    let aggregate = builder.add_aggregate(
        join,
        AggregateSpec::new(
            vec![Expr::column("grp")],
            vec![
                AggregateExpr::new(AggregateFunction::StddevSamp, Expr::column("v"), "stdev_v")
                    .with_filter(Some(Expr::column("keep"))),
                AggregateExpr::new(AggregateFunction::Avg, Expr::column("v"), "avg_v")
                    .with_filter(Some(Expr::column("keep"))),
            ],
        ),
        Arc::clone(&aggregate_schema),
    );
    let sort = builder.add_sort(
        aggregate,
        SortSpec::new(vec![SortExpr::new(Expr::column("grp"), false, false)]),
        Arc::clone(&aggregate_schema),
    );
    let plan = builder.finish_with_host_sink(sort);

    let context = ExecutionContext::builder()
        .with_shared_runtime(runtime)
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 128 * 1024 * 1024,
            max_source_chunk_bytes: 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .with_statistical_aggregate_execution_mode(StddevSampExecutionMode::GpuNativeTolerant)
        .build()
        .unwrap();
    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };

    assert_eq!(int32_array(batch, 0).values().as_ref(), &[10, 20, 30]);
    assert_eq!(float64_array(batch, 1).value(0), 2.0_f64.sqrt());
    assert!(float64_array(batch, 1).is_null(1));
    assert!(float64_array(batch, 1).is_null(2));
    assert_eq!(float64_array(batch, 2).value(0), 3.0);
    assert_eq!(float64_array(batch, 2).value(1), 10.0);
    assert!(float64_array(batch, 2).is_null(2));
    assert!(aggregate_strategy_fact_matches(result.metrics(), "selected_strategy", "stddev_samp_gpu_native_tolerant"));
    assert!(!aggregate_strategy_fact_matches(
        result.metrics(),
        "selected_strategy",
        "floating_statistical_gpu_native_tolerant_partial_moments"
    ));
    assert!(!aggregate_strategy_fact_matches(
        result.metrics(),
        "selected_strategy",
        "floating_statistical_gpu_native_tolerant_final_moments"
    ));

    fs::remove_dir_all(dir).unwrap();
}

fn write_stddev_input_parquet(runtime: &cudf::Runtime, path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("v_i32", DataType::Int32, true),
        Field::new("v_i64", DataType::Int64, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 1, 1, 2, 2, 3, 4])),
            Arc::new(Int32Array::from(vec![Some(2), Some(4), None, Some(10), None, Some(7), None])),
            Arc::new(Int64Array::from(vec![Some(20), Some(40), None, Some(100), None, Some(70), None])),
        ],
    )
    .unwrap();
    DataFrame::from_record_batch(runtime, &batch).unwrap().write_parquet(path).unwrap();
    schema
}

fn write_batch_with_row_group_size(schema: Arc<Schema>, path: &Path, batch: &RecordBatch, rows_per_group: usize) {
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(rows_per_group)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(batch).unwrap();
    writer.close().unwrap();
}

fn aggregate_strategy_fact(metrics: &QueryMetrics, key: &str) -> Option<String> {
    metrics
        .aggregate_strategy_decisions
        .iter()
        .flat_map(|decision| decision.facts.iter())
        .find_map(|(fact_key, value)| (fact_key == key).then(|| value.clone()))
}

fn aggregate_strategy_fact_matches(metrics: &QueryMetrics, key: &str, expected: &str) -> bool {
    metrics
        .aggregate_strategy_decisions
        .iter()
        .flat_map(|decision| decision.facts.iter())
        .any(|(fact_key, value)| fact_key == key && value == expected)
}

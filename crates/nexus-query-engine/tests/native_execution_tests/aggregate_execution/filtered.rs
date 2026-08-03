use super::*;

#[test]
fn native_grouped_aggregate_filter_masks_false_and_null_predicates() {
    let (runtime, context, schema, inputs) = filtered_aggregate_input();
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("sum_payload", DataType::Int64, true),
        Field::new("count_payload", DataType::Int64, false),
        Field::new("count_rows", DataType::Int64, false),
        Field::new("avg_payload", DataType::Float64, true),
    ]));
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![
            AggregateExpr::new(AggregateFunction::Sum, Expr::column("payload"), "sum_payload")
                .with_filter(Some(Expr::column("flag"))),
            AggregateExpr::new(AggregateFunction::Count, Expr::column("payload"), "count_payload")
                .with_filter(Some(Expr::column("flag"))),
            AggregateExpr::new(AggregateFunction::Count, Expr::literal(LiteralValue::Int64(1)), "count_rows")
                .with_filter(Some(Expr::column("flag"))),
            AggregateExpr::new(AggregateFunction::Avg, Expr::column("payload"), "avg_payload")
                .with_filter(Some(Expr::column("flag"))),
        ],
    );
    let sort = SortSpec::new(vec![SortExpr::new(Expr::column("key"), false, false)]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("filtered", Arc::clone(&schema))));
    let aggregate = builder.add_aggregate(source, aggregate, Arc::clone(&output_schema));
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&output_schema));
    let plan = builder.finish_with_host_sink(sort);

    let result = execute_query_plan_with_gpu_inputs(&context, &plan, &inputs).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 2);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[10, 5]);
    assert_eq!(int64_array(batch, 2).values().as_ref(), &[1, 1]);
    assert_eq!(int64_array(batch, 3).values().as_ref(), &[1, 2]);
    assert_eq!(float64_array(batch, 4).values().as_ref(), &[10.0, 5.0]);
    assert_eq!(result.metrics().boundary.host_to_device_uploads, 0);
    assert_eq!(runtime.device().ordinal(), context.runtime().unwrap().device().ordinal());
}

#[test]
fn bounded_native_grouped_filtered_aggregate_single_pass_reports_materialization() {
    let context = backend_context_with_memory_control_policy(MemoryControlPolicy::Bounded {
        device_budget_bytes: 512 * 1024 * 1024,
        max_source_chunk_bytes: 8 * 1024 * 1024,
        max_row_groups_per_chunk: 1,
        max_retry_attempts: 1,
        min_retry_chunk_row_groups: 1,
    });
    let runtime = Arc::new(context.runtime().unwrap());
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, true),
        Field::new("flag", DataType::Boolean, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 1, 1, 2, 2, 2])),
            Arc::new(Int64Array::from(vec![Some(10), Some(20), Some(30), Some(5), None, Some(7)])),
            Arc::new(BooleanArray::from(vec![Some(true), Some(false), None, Some(true), Some(true), None])),
        ],
    )
    .unwrap();
    let frame = DataFrame::from_record_batch(&runtime, &batch).unwrap();
    let mut inputs = GpuResidentInputMap::new();
    inputs.insert(
        "filtered".to_string(),
        GpuResultHandle::new(&context, GpuDataFrame::new(frame), Arc::clone(&schema)).unwrap(),
    );
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("sum_payload", DataType::Int64, true),
        Field::new("count_rows", DataType::Int64, false),
    ]));
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![
            AggregateExpr::new(AggregateFunction::Sum, Expr::column("payload"), "sum_payload")
                .with_filter(Some(Expr::column("flag"))),
            AggregateExpr::new(AggregateFunction::Count, Expr::literal(LiteralValue::Int64(1)), "count_rows")
                .with_filter(Some(Expr::column("flag"))),
        ],
    );
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("filtered", Arc::clone(&schema))));
    let aggregate = builder.add_aggregate(source, aggregate, Arc::clone(&output_schema));
    let plan = builder.finish_with_host_sink(aggregate);

    let result = execute_query_plan_with_gpu_inputs(&context, &plan, &inputs).unwrap();
    let strategy = result
        .metrics()
        .aggregate_strategy_decisions
        .iter()
        .find(|decision| {
            decision.facts.get("filtered_input_materialization_strategy").map(String::as_str)
                == Some("single_pass_with_columns")
        })
        .expect("single-pass grouped filtered materialization should be reported");
    assert_eq!(strategy.facts.get("aggregate_filter_count").map(String::as_str), Some("2"));
    assert!(
        strategy
            .facts
            .get("filtered_input_materialization_estimated_bytes")
            .and_then(|bytes| bytes.parse::<u64>().ok())
            .is_some_and(|bytes| bytes > 0)
    );
}

#[test]
fn bounded_native_grouped_filtered_aggregate_segments_low_cardinality_groups() {
    // The aligned domain pool charge (~8x raw bytes for many small
    // allocations) sizes the pressure point above the legacy arena path.
    let context = backend_context_with_memory_control_policy(MemoryControlPolicy::Bounded {
        device_budget_bytes: 48 * 1024 * 1024,
        max_source_chunk_bytes: 8 * 1024 * 1024,
        max_row_groups_per_chunk: 1,
        max_retry_attempts: 1,
        min_retry_chunk_row_groups: 1,
    });
    let runtime = Arc::new(context.runtime().unwrap());
    let row_count = 100_000usize;
    let schema = Arc::new(
        Schema::new(vec![
            Field::new("key", DataType::Int32, false),
            Field::new("payload", DataType::Int64, false),
            Field::new("flag", DataType::Boolean, false),
        ])
        .with_metadata(HashMap::from([(
            "__nexus_filtered_grouped_segment_target_bytes".to_string(),
            "1024".to_string(),
        )])),
    );
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from_iter_values((0..row_count).map(|value| (value % 4) as i32))),
            Arc::new(Int64Array::from_iter_values((0..row_count).map(|value| (value % 100) as i64))),
            Arc::new(BooleanArray::from_iter((0..row_count).map(|value| Some(value % 2 == 0)))),
        ],
    )
    .unwrap();
    let frame = DataFrame::from_record_batch(&runtime, &batch).unwrap();
    let mut inputs = GpuResidentInputMap::new();
    inputs.insert(
        "filtered".to_string(),
        GpuResultHandle::new(&context, GpuDataFrame::new(frame), Arc::clone(&schema)).unwrap(),
    );

    let mut fields = vec![Field::new("key", DataType::Int32, false)];
    let mut aggr_exprs = Vec::new();
    for bucket in 0..10 {
        fields.push(Field::new(format!("count_rows_{bucket}"), DataType::Int64, false));
        fields.push(Field::new(format!("sum_payload_{bucket}"), DataType::Int64, true));
        fields.push(Field::new(format!("avg_payload_{bucket}"), DataType::Float64, true));
        aggr_exprs.push(
            AggregateExpr::new(
                AggregateFunction::Count,
                Expr::literal(LiteralValue::Int64(1)),
                format!("count_rows_{bucket}"),
            )
            .with_filter(Some(Expr::column("flag"))),
        );
        aggr_exprs.push(
            AggregateExpr::new(AggregateFunction::Sum, Expr::column("payload"), format!("sum_payload_{bucket}"))
                .with_filter(Some(Expr::column("flag"))),
        );
        aggr_exprs.push(
            AggregateExpr::new(AggregateFunction::Avg, Expr::column("payload"), format!("avg_payload_{bucket}"))
                .with_filter(Some(Expr::column("flag"))),
        );
    }
    let output_schema = Arc::new(Schema::new(fields));
    let aggregate = AggregateSpec::new(vec![Expr::column("key")], aggr_exprs);
    let sort = SortSpec::new(vec![SortExpr::new(Expr::column("key"), false, false)]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("filtered", Arc::clone(&schema))));
    let aggregate = builder.add_aggregate(source, aggregate, Arc::clone(&output_schema));
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&output_schema));
    let plan = builder.finish_with_host_sink(sort);

    let result = execute_query_plan_with_gpu_inputs(&context, &plan, &inputs).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[0, 1, 2, 3]);
    for bucket in 0..10 {
        let base = 1 + bucket * 3;
        assert_eq!(int64_array(batch, base).value(0), 25_000);
        assert_eq!(int64_array(batch, base + 1).value(0), 1_200_000);
        assert_eq!(float64_array(batch, base + 2).value(0), 48.0);
        assert_eq!(int64_array(batch, base).value(2), 25_000);
        assert_eq!(int64_array(batch, base + 1).value(2), 1_250_000);
        assert_eq!(float64_array(batch, base + 2).value(2), 50.0);
        assert_eq!(int64_array(batch, base).value(1), 0);
        assert!(int64_array(batch, base + 1).is_null(1));
        assert!(float64_array(batch, base + 2).is_null(1));
        assert_eq!(int64_array(batch, base).value(3), 0);
        assert!(int64_array(batch, base + 1).is_null(3));
        assert!(float64_array(batch, base + 2).is_null(3));
    }
    let strategy = result
        .metrics()
        .aggregate_strategy_decisions
        .iter()
        .find(|decision| {
            decision.facts.get("selected_strategy").map(String::as_str) == Some("filtered_grouped_segmented_state")
        })
        .expect("segmented filtered grouped strategy should be reported");
    assert_eq!(
        strategy.facts.get("filtered_input_materialization_strategy").map(String::as_str),
        Some("row_chunked_grouped_segmented_state")
    );
    assert_eq!(strategy.facts.get("aggregate_filter_count").map(String::as_str), Some("30"));
    assert!(
        strategy
            .facts
            .get("filtered_grouped_actual_segment_count")
            .and_then(|count| count.parse::<u64>().ok())
            .is_some_and(|count| count > 1),
        "test must exercise grouped expression segmentation"
    );
    assert!(
        strategy
            .facts
            .get("filtered_grouped_actual_chunk_count")
            .and_then(|count| count.parse::<u64>().ok())
            .is_some_and(|count| count > 1),
        "test must exercise grouped row chunking"
    );
}

#[test]
fn bounded_native_partial_grouped_filtered_aggregate_segments_high_cardinality_groups() {
    let dir = unique_temp_dir("nexus-query-engine-filtered-grouped-high-cardinality");
    let path = dir.join("input.parquet");
    let context = backend_context_with_memory_control_policy(MemoryControlPolicy::Bounded {
        device_budget_bytes: 32 * 1024 * 1024,
        max_source_chunk_bytes: 8 * 1024 * 1024,
        max_row_groups_per_chunk: 1,
        max_retry_attempts: 1,
        min_retry_chunk_row_groups: 1,
    });
    let row_count = 20_000usize;
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new("payload", DataType::Int64, false),
        Field::new("flag", DataType::Boolean, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from_iter_values((0..row_count).map(|value| value as i64))),
            Arc::new(Int64Array::from_iter_values((0..row_count).map(|value| (value % 100) as i64))),
            Arc::new(BooleanArray::from_iter((0..row_count).map(|_| Some(true)))),
        ],
    )
    .unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(row_count)).build();
    let file = File::create(&path).unwrap();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&schema),
        vec!["key".to_string(), "payload".to_string(), "flag".to_string()],
    );
    let mut fields = vec![Field::new("key", DataType::Int64, false)];
    let mut aggr_exprs = Vec::new();
    for bucket in 0..30 {
        fields.push(Field::new(format!("count_rows_{bucket}"), DataType::Int64, false));
        fields.push(Field::new(format!("sum_payload_{bucket}"), DataType::Int64, true));
        fields.push(Field::new(format!("max_payload_{bucket}"), DataType::Int64, true));
        aggr_exprs.push(
            AggregateExpr::new(
                AggregateFunction::Count,
                Expr::literal(LiteralValue::Int64(1)),
                format!("count_rows_{bucket}"),
            )
            .with_filter(Some(Expr::column("flag"))),
        );
        aggr_exprs.push(
            AggregateExpr::new(AggregateFunction::Sum, Expr::column("payload"), format!("sum_payload_{bucket}"))
                .with_filter(Some(Expr::column("flag"))),
        );
        aggr_exprs.push(
            AggregateExpr::new(AggregateFunction::Max, Expr::column("payload"), format!("max_payload_{bucket}"))
                .with_filter(Some(Expr::column("flag"))),
        );
    }
    let output_schema = Arc::new(Schema::new(fields));
    let aggregate = AggregateSpec::new(vec![Expr::column("key")], aggr_exprs);
    let sort = SortSpec::new(vec![SortExpr::new(Expr::column("key"), false, false)]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, Arc::clone(&output_schema));
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&output_schema));
    let plan = builder.finish_with_host_sink(sort);

    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), row_count);
    assert_eq!(int64_array(batch, 0).value(0), 0);
    assert_eq!(int64_array(batch, 1).value(0), 1);
    assert_eq!(int64_array(batch, 2).value(0), 0);
    assert_eq!(int64_array(batch, 3).value(0), 0);
    assert_eq!(int64_array(batch, 0).value(99), 99);
    assert_eq!(int64_array(batch, 2).value(99), 99);
    assert_eq!(int64_array(batch, 3).value(99), 99);

    let strategy = result
        .metrics()
        .aggregate_strategy_decisions
        .iter()
        .find(|decision| {
            decision.facts.get("execution_stage").map(String::as_str) == Some("partial")
                && decision.facts.get("selected_strategy").map(String::as_str)
                    == Some("filtered_grouped_segmented_state")
        })
        .expect("segmented filtered grouped partial strategy should be reported");
    assert_eq!(
        strategy.facts.get("filtered_grouped_high_cardinality_admission").map(String::as_str),
        Some("chunk_local_partial_final_merge")
    );
    assert_eq!(strategy.facts.get("aggregate_filter_count").map(String::as_str), Some("90"));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_global_aggregate_filter_masks_false_and_null_predicates() {
    let (_runtime, context, schema, inputs) = filtered_aggregate_input();
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("sum_payload", DataType::Int64, true),
        Field::new("count_payload", DataType::Int64, false),
        Field::new("count_rows", DataType::Int64, false),
        Field::new("avg_payload", DataType::Float64, true),
    ]));
    let aggregate = AggregateSpec::new(
        Vec::new(),
        vec![
            AggregateExpr::new(AggregateFunction::Sum, Expr::column("payload"), "sum_payload")
                .with_filter(Some(Expr::column("flag"))),
            AggregateExpr::new(AggregateFunction::Count, Expr::column("payload"), "count_payload")
                .with_filter(Some(Expr::column("flag"))),
            AggregateExpr::new(AggregateFunction::Count, Expr::literal(LiteralValue::Int64(1)), "count_rows")
                .with_filter(Some(Expr::column("flag"))),
            AggregateExpr::new(AggregateFunction::Avg, Expr::column("payload"), "avg_payload")
                .with_filter(Some(Expr::column("flag"))),
        ],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("filtered", Arc::clone(&schema))));
    let aggregate = builder.add_aggregate(source, aggregate, output_schema);
    let plan = builder.finish_with_host_sink(aggregate);

    let result = execute_query_plan_with_gpu_inputs(&context, &plan, &inputs).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(int64_array(batch, 0).values().as_ref(), &[15]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[2]);
    assert_eq!(int64_array(batch, 2).values().as_ref(), &[3]);
    assert_eq!(float64_array(batch, 3).values().as_ref(), &[7.5]);
}

#[test]
fn native_global_aggregate_filter_masks_count_distinct_inputs() {
    let (_runtime, context, schema, inputs) = filtered_aggregate_input();
    let output_schema = Arc::new(Schema::new(vec![Field::new("distinct_payload", DataType::Int64, false)]));
    let aggregate = AggregateSpec::new(
        Vec::new(),
        vec![
            AggregateExpr::new(AggregateFunction::Count, Expr::column("payload"), "distinct_payload")
                .with_distinct(true)
                .with_filter(Some(Expr::column("flag"))),
        ],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("filtered", Arc::clone(&schema))));
    let aggregate = builder.add_aggregate(source, aggregate, output_schema);
    let plan = builder.finish_with_host_sink(aggregate);

    let result = execute_query_plan_with_gpu_inputs(&context, &plan, &inputs).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(int64_array(batch, 0).values().as_ref(), &[2]);
}

#[test]
fn bounded_native_global_filtered_aggregate_segments_large_materialization() {
    // The aligned domain pool charge (~8x raw bytes for many small
    // allocations) sizes the pressure point above the legacy arena path.
    let context = backend_context_with_memory_control_policy(MemoryControlPolicy::Bounded {
        device_budget_bytes: 40 * 1024 * 1024,
        max_source_chunk_bytes: 8 * 1024 * 1024,
        max_row_groups_per_chunk: 1,
        max_retry_attempts: 1,
        min_retry_chunk_row_groups: 1,
    });
    let runtime = Arc::new(context.runtime().unwrap());
    let schema = Arc::new(
        Schema::new(vec![
            Field::new("payload", DataType::Int64, false),
            Field::new("nullable_payload", DataType::Int64, true),
            Field::new("flag", DataType::Boolean, true),
            Field::new("all_flag", DataType::Boolean, false),
            Field::new("empty_flag", DataType::Boolean, false),
        ])
        .with_metadata(HashMap::from([(
            "__nexus_filtered_global_segment_target_bytes".to_string(),
            "1024".to_string(),
        )])),
    );
    let payload = (0..100_000).map(|value| i64::from(value % 100)).collect::<Vec<_>>();
    let nullable_payload = vec![None::<i64>; 100_000];
    let flag = (0..100_000).map(|value| Some(value % 2 == 0)).collect::<Vec<_>>();
    let all_flag = vec![true; 100_000];
    let empty_flag = vec![false; 100_000];
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(payload)),
            Arc::new(Int64Array::from(nullable_payload)),
            Arc::new(BooleanArray::from(flag)),
            Arc::new(BooleanArray::from(all_flag)),
            Arc::new(BooleanArray::from(empty_flag)),
        ],
    )
    .unwrap();
    let frame = DataFrame::from_record_batch(&runtime, &batch).unwrap();
    let mut inputs = GpuResidentInputMap::new();
    inputs.insert(
        "filtered".to_string(),
        GpuResultHandle::new(&context, GpuDataFrame::new(frame), Arc::clone(&schema)).unwrap(),
    );

    let mut fields = Vec::new();
    let mut aggr_exprs = Vec::new();
    for bucket in 0..5 {
        fields.push(Field::new(format!("count_rows_{bucket}"), DataType::Int64, false));
        fields.push(Field::new(format!("sum_payload_{bucket}"), DataType::Int64, true));
        fields.push(Field::new(format!("avg_payload_{bucket}"), DataType::Float64, true));
        aggr_exprs.push(
            AggregateExpr::new(
                AggregateFunction::Count,
                Expr::literal(LiteralValue::Int64(1)),
                format!("count_rows_{bucket}"),
            )
            .with_filter(Some(Expr::column("flag"))),
        );
        aggr_exprs.push(
            AggregateExpr::new(AggregateFunction::Sum, Expr::column("payload"), format!("sum_payload_{bucket}"))
                .with_filter(Some(Expr::column("flag"))),
        );
        aggr_exprs.push(
            AggregateExpr::new(AggregateFunction::Avg, Expr::column("payload"), format!("avg_payload_{bucket}"))
                .with_filter(Some(Expr::column("flag"))),
        );
    }
    for (alias, function, value, filter) in [
        ("min_payload_filtered", AggregateFunction::Min, "payload", "flag"),
        ("max_payload_filtered", AggregateFunction::Max, "payload", "flag"),
        ("min_payload_all", AggregateFunction::Min, "payload", "all_flag"),
        ("max_payload_all", AggregateFunction::Max, "payload", "all_flag"),
        ("min_payload_empty", AggregateFunction::Min, "payload", "empty_flag"),
        ("max_payload_empty", AggregateFunction::Max, "payload", "empty_flag"),
        ("min_null_payload_all", AggregateFunction::Min, "nullable_payload", "all_flag"),
        ("max_null_payload_all", AggregateFunction::Max, "nullable_payload", "all_flag"),
    ] {
        fields.push(Field::new(alias, DataType::Int64, true));
        aggr_exprs
            .push(AggregateExpr::new(function, Expr::column(value), alias).with_filter(Some(Expr::column(filter))));
    }
    let output_schema = Arc::new(Schema::new(fields));
    let aggregate = AggregateSpec::new(Vec::new(), aggr_exprs);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("filtered", Arc::clone(&schema))));
    let aggregate = builder.add_aggregate(source, aggregate, Arc::clone(&output_schema));
    let plan = builder.finish_with_host_sink(aggregate);

    let result = execute_query_plan_with_gpu_inputs(&context, &plan, &inputs).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1);
    for bucket in 0..5 {
        assert_eq!(int64_array(batch, bucket * 3).values().as_ref(), &[50_000]);
        assert_eq!(int64_array(batch, bucket * 3 + 1).values().as_ref(), &[2_450_000]);
        assert_eq!(float64_array(batch, bucket * 3 + 2).values().as_ref(), &[49.0]);
    }
    let extra_base = 5 * 3;
    assert_eq!(int64_array(batch, extra_base).value(0), 0);
    assert_eq!(int64_array(batch, extra_base + 1).value(0), 98);
    assert_eq!(int64_array(batch, extra_base + 2).value(0), 0);
    assert_eq!(int64_array(batch, extra_base + 3).value(0), 99);
    assert!(int64_array(batch, extra_base + 4).is_null(0));
    assert!(int64_array(batch, extra_base + 5).is_null(0));
    assert!(int64_array(batch, extra_base + 6).is_null(0));
    assert!(int64_array(batch, extra_base + 7).is_null(0));
    let strategy = result
        .metrics()
        .aggregate_strategy_decisions
        .iter()
        .find(|decision| {
            decision.facts.get("selected_strategy").map(String::as_str) == Some("filtered_global_segmented_state")
        })
        .expect("segmented filtered global strategy should be reported");
    assert_eq!(
        strategy.facts.get("filtered_input_materialization_strategy").map(String::as_str),
        Some("row_chunked_segmented_state")
    );
    assert_eq!(strategy.facts.get("aggregate_filter_count").map(String::as_str), Some("23"));
    assert!(
        strategy
            .facts
            .get("filtered_global_actual_chunk_count")
            .and_then(|count| count.parse::<u64>().ok())
            .is_some_and(|count| count > 1),
        "test must exercise row-chunked state merge"
    );
    assert!(
        strategy
            .facts
            .get("filtered_global_actual_segment_count")
            .and_then(|count| count.parse::<u64>().ok())
            .is_some_and(|count| count > 1),
        "test must exercise filtered expression segmentation"
    );
    assert_eq!(
        strategy.facts.get("filtered_global_state_rows_merged").map(String::as_str),
        strategy.facts.get("filtered_global_actual_chunk_count").map(String::as_str)
    );
    assert!(
        strategy
            .facts
            .get("filtered_global_observed_temporary_peak_bytes")
            .and_then(|count| count.parse::<u64>().ok())
            .is_some_and(|bytes| bytes > 0),
        "segmented strategy should surface lower-level temporary peak evidence"
    );
    assert!(
        strategy
            .facts
            .get("filtered_global_observed_retained_peak_bytes")
            .and_then(|count| count.parse::<u64>().ok())
            .is_some_and(|bytes| bytes > 0),
        "segmented strategy should surface lower-level retained peak evidence"
    );
}

#[test]
fn bounded_native_global_filtered_aggregate_segments_q09_decimal_avg_shape() {
    let context = backend_context_with_memory_control_policy(MemoryControlPolicy::Bounded {
        device_budget_bytes: 32 * 1024 * 1024,
        max_source_chunk_bytes: 8 * 1024 * 1024,
        max_row_groups_per_chunk: 1,
        max_retry_attempts: 1,
        min_retry_chunk_row_groups: 1,
    });
    let runtime = Arc::new(context.runtime().unwrap());
    let repeats_per_quantity = 700usize;
    let row_count = 100 * repeats_per_quantity;
    let schema = Arc::new(
        Schema::new(vec![
            Field::new("ss_quantity", DataType::Int32, false),
            Field::new("ss_ext_discount_amt", DataType::Decimal128(7, 2), false),
            Field::new("ss_net_paid", DataType::Decimal128(7, 2), false),
        ])
        .with_metadata(HashMap::from([(
            "__nexus_filtered_global_segment_target_bytes".to_string(),
            (1024 * 1024).to_string(),
        )])),
    );
    let quantity = (0..row_count).map(|row| i32::try_from(row % 100 + 1).unwrap()).collect::<Vec<_>>();
    let discount = quantity.iter().map(|value| i128::from(*value) * 100).collect::<Vec<_>>();
    let net_paid = quantity.iter().map(|value| i128::from(*value) * 200).collect::<Vec<_>>();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(quantity)),
            Arc::new(Decimal128Array::from_iter_values(discount).with_precision_and_scale(7, 2).unwrap()),
            Arc::new(Decimal128Array::from_iter_values(net_paid).with_precision_and_scale(7, 2).unwrap()),
        ],
    )
    .unwrap();
    let frame = DataFrame::from_record_batch(&runtime, &batch).unwrap();
    let mut inputs = GpuResidentInputMap::new();
    inputs.insert(
        "filtered".to_string(),
        GpuResultHandle::new(&context, GpuDataFrame::new(frame), Arc::clone(&schema)).unwrap(),
    );

    let mut fields = Vec::new();
    let mut aggr_exprs = Vec::new();
    for (bucket, lower) in [1, 21, 41, 61, 81].into_iter().enumerate() {
        let upper = lower + 19;
        let filter = q09_quantity_between(lower, upper);
        fields.push(Field::new(format!("b{bucket}_cnt"), DataType::Int64, false));
        fields.push(Field::new(format!("b{bucket}_avg_discount"), DataType::Decimal128(11, 6), true));
        fields.push(Field::new(format!("b{bucket}_avg_net_paid"), DataType::Decimal128(11, 6), true));
        aggr_exprs.push(
            AggregateExpr::new(
                AggregateFunction::Count,
                Expr::literal(LiteralValue::Int64(1)),
                format!("b{bucket}_cnt"),
            )
            .with_filter(Some(filter.clone())),
        );
        aggr_exprs.push(
            AggregateExpr::new(
                AggregateFunction::Avg,
                Expr::column("ss_ext_discount_amt"),
                format!("b{bucket}_avg_discount"),
            )
            .with_filter(Some(filter.clone())),
        );
        aggr_exprs.push(
            AggregateExpr::new(AggregateFunction::Avg, Expr::column("ss_net_paid"), format!("b{bucket}_avg_net_paid"))
                .with_filter(Some(filter)),
        );
    }
    let output_schema = Arc::new(Schema::new(fields));
    let aggregate = AggregateSpec::new(Vec::new(), aggr_exprs);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("filtered", Arc::clone(&schema))));
    let aggregate = builder.add_aggregate(source, aggregate, Arc::clone(&output_schema));
    let plan = builder.finish_with_host_sink(aggregate);

    let result = execute_query_plan_with_gpu_inputs(&context, &plan, &inputs).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.schema(), output_schema);
    assert_eq!(batch.num_rows(), 1);
    let expected_discount = [10_500_000, 30_500_000, 50_500_000, 70_500_000, 90_500_000];
    let expected_net_paid = [21_000_000, 61_000_000, 101_000_000, 141_000_000, 181_000_000];
    for bucket in 0..5 {
        assert_eq!(int64_array(batch, bucket * 3).value(0), i64::try_from(20 * repeats_per_quantity).unwrap());
        let avg_discount = decimal128_array(batch, bucket * 3 + 1);
        assert_eq!(avg_discount.scale(), 6);
        assert_eq!(avg_discount.value(0), expected_discount[bucket]);
        let avg_net_paid = decimal128_array(batch, bucket * 3 + 2);
        assert_eq!(avg_net_paid.scale(), 6);
        assert_eq!(avg_net_paid.value(0), expected_net_paid[bucket]);
    }
    let strategy = result
        .metrics()
        .aggregate_strategy_decisions
        .iter()
        .find(|decision| {
            decision.facts.get("selected_strategy").map(String::as_str) == Some("filtered_global_segmented_state")
        })
        .expect("segmented filtered global strategy should be reported");
    assert_eq!(
        strategy.facts.get("filtered_input_materialization_strategy").map(String::as_str),
        Some("row_chunked_segmented_state")
    );
    assert_eq!(strategy.facts.get("aggregate_filter_count").map(String::as_str), Some("15"));
    assert_eq!(strategy.facts.get("filtered_input_unique_predicate_count").map(String::as_str), Some("5"));
    assert_eq!(strategy.facts.get("filtered_global_retained_predicate_bytes_per_row").map(String::as_str), Some("10"));
    let actual_chunk_count = strategy
        .facts
        .get("filtered_global_actual_chunk_count")
        .and_then(|count| count.parse::<u64>().ok())
        .expect("segmented strategy should record actual chunk count");
    let actual_segment_count = strategy
        .facts
        .get("filtered_global_actual_segment_count")
        .and_then(|count| count.parse::<u64>().ok())
        .expect("segmented strategy should record actual segment count");
    let inputs_per_segment = strategy
        .facts
        .get("filtered_global_inputs_per_segment")
        .and_then(|count| count.parse::<usize>().ok())
        .expect("segmented strategy should record inputs per segment");
    assert!(actual_chunk_count > 1, "test must exercise row-chunked state merge");
    assert!(
        actual_segment_count > actual_chunk_count,
        "test must exercise filtered expression segmentation beyond row chunking"
    );
    assert!(inputs_per_segment < 15, "test should segment the 15 filtered aggregate inputs");
}

fn q09_quantity_between(lower: i32, upper: i32) -> Expr {
    Expr::binary(
        Expr::binary(Expr::column("ss_quantity"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int32(lower))),
        BinaryOp::And,
        Expr::binary(Expr::column("ss_quantity"), BinaryOp::LtEq, Expr::literal(LiteralValue::Int32(upper))),
    )
}

#[test]
fn bounded_native_partial_global_filtered_aggregate_segments_large_materialization() {
    let dir = unique_temp_dir("nexus-query-engine-filtered-global-partial-segmented");
    let path = dir.join("input.parquet");
    let context = backend_context_with_memory_control_policy(MemoryControlPolicy::Bounded {
        device_budget_bytes: 8 * 1024 * 1024,
        max_source_chunk_bytes: 8 * 1024 * 1024,
        max_row_groups_per_chunk: 1,
        max_retry_attempts: 1,
        min_retry_chunk_row_groups: 1,
    });
    let schema = Arc::new(
        Schema::new(vec![
            Field::new("payload", DataType::Int64, false),
            Field::new("nullable_payload", DataType::Int64, true),
            Field::new("flag", DataType::Boolean, true),
            Field::new("all_flag", DataType::Boolean, false),
            Field::new("empty_flag", DataType::Boolean, false),
        ])
        .with_metadata(HashMap::from([(
            "__nexus_filtered_global_segment_target_bytes".to_string(),
            "1024".to_string(),
        )])),
    );
    let row_count = 20_000usize;
    let payload = (0..row_count).map(|value| (value % 100) as i64).collect::<Vec<_>>();
    let nullable_payload = vec![None::<i64>; row_count];
    let flag = (0..row_count).map(|value| Some(value % 2 == 0)).collect::<Vec<_>>();
    let all_flag = vec![true; row_count];
    let empty_flag = vec![false; row_count];
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(payload)),
            Arc::new(Int64Array::from(nullable_payload)),
            Arc::new(BooleanArray::from(flag)),
            Arc::new(BooleanArray::from(all_flag)),
            Arc::new(BooleanArray::from(empty_flag)),
        ],
    )
    .unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(row_count)).build();
    let file = File::create(&path).unwrap();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&schema),
        vec![
            "payload".to_string(),
            "nullable_payload".to_string(),
            "flag".to_string(),
            "all_flag".to_string(),
            "empty_flag".to_string(),
        ],
    );
    let mut fields = Vec::new();
    let mut aggr_exprs = Vec::new();
    for bucket in 0..10 {
        fields.push(Field::new(format!("count_rows_{bucket}"), DataType::Int64, false));
        fields.push(Field::new(format!("sum_payload_{bucket}"), DataType::Int64, true));
        fields.push(Field::new(format!("avg_payload_{bucket}"), DataType::Float64, true));
        aggr_exprs.push(
            AggregateExpr::new(
                AggregateFunction::Count,
                Expr::literal(LiteralValue::Int64(1)),
                format!("count_rows_{bucket}"),
            )
            .with_filter(Some(Expr::column("flag"))),
        );
        aggr_exprs.push(
            AggregateExpr::new(AggregateFunction::Sum, Expr::column("payload"), format!("sum_payload_{bucket}"))
                .with_filter(Some(Expr::column("flag"))),
        );
        aggr_exprs.push(
            AggregateExpr::new(AggregateFunction::Avg, Expr::column("payload"), format!("avg_payload_{bucket}"))
                .with_filter(Some(Expr::column("flag"))),
        );
    }
    for (alias, function, value, filter) in [
        ("min_payload_filtered", AggregateFunction::Min, "payload", "flag"),
        ("max_payload_filtered", AggregateFunction::Max, "payload", "flag"),
        ("min_payload_all", AggregateFunction::Min, "payload", "all_flag"),
        ("max_payload_all", AggregateFunction::Max, "payload", "all_flag"),
        ("min_payload_empty", AggregateFunction::Min, "payload", "empty_flag"),
        ("max_payload_empty", AggregateFunction::Max, "payload", "empty_flag"),
        ("min_null_payload_all", AggregateFunction::Min, "nullable_payload", "all_flag"),
        ("max_null_payload_all", AggregateFunction::Max, "nullable_payload", "all_flag"),
    ] {
        fields.push(Field::new(alias, DataType::Int64, true));
        aggr_exprs
            .push(AggregateExpr::new(function, Expr::column(value), alias).with_filter(Some(Expr::column(filter))));
    }
    let output_schema = Arc::new(Schema::new(fields));
    let aggregate = AggregateSpec::new(Vec::new(), aggr_exprs);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, Arc::clone(&output_schema));
    let plan = builder.finish_with_host_sink(aggregate);

    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1);
    for bucket in 0..10 {
        assert_eq!(int64_array(batch, bucket * 3).values().as_ref(), &[10_000]);
        assert_eq!(int64_array(batch, bucket * 3 + 1).values().as_ref(), &[490_000]);
        assert_eq!(float64_array(batch, bucket * 3 + 2).values().as_ref(), &[49.0]);
    }
    let extra_base = 10 * 3;
    assert_eq!(int64_array(batch, extra_base).value(0), 0);
    assert_eq!(int64_array(batch, extra_base + 1).value(0), 98);
    assert_eq!(int64_array(batch, extra_base + 2).value(0), 0);
    assert_eq!(int64_array(batch, extra_base + 3).value(0), 99);
    assert!(int64_array(batch, extra_base + 4).is_null(0));
    assert!(int64_array(batch, extra_base + 5).is_null(0));
    assert!(int64_array(batch, extra_base + 6).is_null(0));
    assert!(int64_array(batch, extra_base + 7).is_null(0));

    let partial_strategy = result
        .metrics()
        .aggregate_strategy_decisions
        .iter()
        .find(|decision| {
            decision.facts.get("execution_stage").map(String::as_str) == Some("partial")
                && decision.facts.get("selected_strategy").map(String::as_str)
                    == Some("filtered_global_segmented_state")
        })
        .expect("segmented filtered global partial strategy should be reported");
    assert_eq!(
        partial_strategy.facts.get("filtered_input_materialization_strategy").map(String::as_str),
        Some("row_chunked_segmented_state")
    );
    assert_eq!(partial_strategy.facts.get("aggregate_filter_count").map(String::as_str), Some("38"));
    assert!(
        partial_strategy
            .facts
            .get("filtered_global_actual_chunk_count")
            .and_then(|count| count.parse::<u64>().ok())
            .is_some_and(|count| count > 1),
        "partial path must exercise row-chunked state merge"
    );
    assert!(
        partial_strategy
            .facts
            .get("filtered_global_actual_segment_count")
            .and_then(|count| count.parse::<u64>().ok())
            .is_some_and(|count| count > 1),
        "partial path must exercise filtered expression segmentation"
    );
    assert!(
        partial_strategy
            .facts
            .get("filtered_global_observed_temporary_peak_bytes")
            .and_then(|count| count.parse::<u64>().ok())
            .is_some_and(|bytes| bytes > 0),
        "partial path should surface lower-level temporary peak evidence"
    );
    assert_eq!(
        partial_strategy.facts.get("filtered_global_state_rows_merged").map(String::as_str),
        partial_strategy.facts.get("filtered_global_actual_chunk_count").map(String::as_str)
    );

    fs::remove_dir_all(dir).unwrap();
}

fn filtered_aggregate_input() -> (Arc<cudf::Runtime>, BackendQueryContext, Arc<Schema>, GpuResidentInputMap) {
    let context = backend_context();
    let runtime = Arc::new(context.runtime().unwrap());
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, true),
        Field::new("flag", DataType::Boolean, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 1, 1, 2, 2, 2])),
            Arc::new(Int64Array::from(vec![Some(10), Some(20), Some(30), Some(5), None, Some(7)])),
            Arc::new(BooleanArray::from(vec![Some(true), Some(false), None, Some(true), Some(true), None])),
        ],
    )
    .unwrap();
    let frame = DataFrame::from_record_batch(&runtime, &batch).unwrap();
    let mut inputs = GpuResidentInputMap::new();
    inputs.insert(
        "filtered".to_string(),
        GpuResultHandle::new(&context, GpuDataFrame::new(frame), Arc::clone(&schema)).unwrap(),
    );
    (runtime, context, schema, inputs)
}

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
    assert_eq!(runtime.device().ordinal(), context.runtime().device().ordinal());
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

fn filtered_aggregate_input() -> (Arc<cudf::Runtime>, ExecutionContext, Arc<Schema>, GpuResidentInputMap) {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let context = ExecutionContext::builder().with_shared_runtime(Arc::clone(&runtime)).build().unwrap();
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

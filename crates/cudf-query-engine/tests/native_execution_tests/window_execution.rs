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
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));
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

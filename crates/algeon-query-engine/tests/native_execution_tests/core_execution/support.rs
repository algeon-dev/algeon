fn assert_string_list_value(array: &ListArray, index: usize, expected: &[Option<&str>]) {
    let values = array.value(index);
    let strings = values.as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(strings.len(), expected.len());
    for (value_index, expected_value) in expected.iter().enumerate() {
        match expected_value {
            Some(expected_value) => assert_eq!(strings.value(value_index), *expected_value),
            None => assert!(strings.is_null(value_index), "value {value_index} should be null"),
        }
    }
}

fn assert_large_string_list_value(array: &LargeListArray, index: usize, expected: &[Option<&str>]) {
    let values = array.value(index);
    let strings = values.as_any().downcast_ref::<LargeStringArray>().unwrap();
    assert_eq!(strings.len(), expected.len());
    for (value_index, expected_value) in expected.iter().enumerate() {
        match expected_value {
            Some(expected_value) => assert_eq!(strings.value(value_index), *expected_value),
            None => assert!(strings.is_null(value_index), "value {value_index} should be null"),
        }
    }
}

fn execute_gpu_resident_filter_ids(
    context: &BackendQueryContext,
    schema: Arc<Schema>,
    batch: &RecordBatch,
    predicate: Expr,
) -> Vec<i32> {
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", Arc::clone(&schema))));
    let filter = builder.add_filter(source, predicate, Arc::clone(&schema));
    let plan = builder.finish_with_host_sink(filter);
    let admitted = context.admit_plan(&plan).unwrap();
    let runtime = Arc::new(admitted.runtime().unwrap());
    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "input".to_string(),
        GpuResultHandle::new(
            &admitted,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, batch).unwrap()),
            Arc::clone(&schema),
        )
        .unwrap(),
    );

    let result = execute_query_plan_with_gpu_inputs(&admitted, &plan, &gpu_inputs).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    let ids = int32_array(batch, 0).values().to_vec();
    admitted.finish();
    ids
}

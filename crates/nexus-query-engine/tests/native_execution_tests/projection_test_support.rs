use super::*;

pub(super) fn assert_duplicate_host_sink_schema_restoration(bounded: bool) {
    let dir = unique_temp_dir(if bounded {
        "nexus-query-engine-bounded-duplicate-host-sink-schema"
    } else {
        "nexus-query-engine-duplicate-host-sink-schema"
    });
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_input_parquet(&runtime, &path);
    let executable_schema = Arc::new(Schema::new(vec![
        Field::new("exec_left_value", DataType::Int64, false),
        Field::new("exec_right_value", DataType::Int64, false),
    ]));
    let frontend_sink_schema = Arc::new(Schema::new(vec![
        Field::new("value", DataType::Int64, false),
        Field::new("value", DataType::Int64, false),
    ]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["m00".to_string(), "m01".to_string()],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let projection = builder.add_projection(
        source,
        projection_outputs(
            vec![Expr::column("m00").alias("exec_left_value"), Expr::column("m01").alias("exec_right_value")],
            executable_schema,
        ),
    );
    let plan = builder.finish_with_host_sink_schema(projection, Arc::clone(&frontend_sink_schema));
    let mut context_builder = ExecutionContext::builder().with_shared_runtime(Arc::clone(&runtime));
    if bounded {
        context_builder = context_builder.with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        });
    }
    let context = context_builder.build().unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.schema().as_ref(), frontend_sink_schema.as_ref());
    assert_eq!(batch.schema().field(0).name(), "value");
    assert_eq!(batch.schema().field(1).name(), "value");
    assert_eq!(int64_array(batch, 0).values().as_ref(), &[100, 200, 300, 400]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[1, 2, 3, 4]);

    fs::remove_dir_all(dir).unwrap();
}

pub(super) fn host_sink_schema_validation_error(
    root_schema: Arc<Schema>,
    sink_schema: Arc<Schema>,
) -> nexus_query_engine::error::QueryError {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let mut builder = QueryPlanBuilder::new();
    let source =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("missing_input", Arc::clone(&root_schema))));
    let plan = builder.finish_with_host_sink_schema(source, sink_schema);
    let context = ExecutionContext::from_shared_runtime(runtime).unwrap();

    match execute_query_plan(&context, &plan) {
        Ok(_) => panic!("native validation should reject host sink schema drift before source read"),
        Err(error) => error,
    }
}

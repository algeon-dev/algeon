use super::*;

#[test]
fn native_query_rejects_duplicate_projection_output_names_generically_before_runtime_materialization() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
    ]));
    let duplicate_projection_schema = Arc::new(Schema::new(vec![
        Field::new("dup", DataType::Int32, false),
        Field::new("dup", DataType::Int64, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let source =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("missing_input", Arc::clone(&input_schema))));
    let projection = builder.add_projection(
        source,
        projection_outputs(
            vec![Expr::column("key").alias("dup"), Expr::column("payload").alias("dup")],
            duplicate_projection_schema,
        ),
    );
    let plan = builder.finish_with_host_sink(projection);
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));

    let error = match execute_query_plan(&context, &plan) {
        Ok(_) => {
            panic!("native validation should reject duplicate projection output names before runtime materialization")
        },
        Err(error) => error,
    };
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.operation(), "cudf_query_engine.plan.validate_output_names");
    assert_eq!(error.fact_value("duplicate_output_name"), Some("dup"));
    assert_eq!(error.fact_value("first_output_index"), Some("0"));
    assert_eq!(error.fact_value("duplicate_output_index"), Some("1"));
    assert_eq!(error.fact_value("output_names"), Some("dup,dup"));
    assert_eq!(error.fact_value("node_kind"), Some("projection"));
    assert_eq!(error.fact_value("schema_behavior"), Some("projection"));
    assert!(error.fact_value("input_name").is_none());
}

#[test]
fn native_query_restores_duplicate_host_sink_schema_without_swapping_columns() {
    assert_duplicate_host_sink_schema_restoration(false);
}

#[test]
fn bounded_native_query_restores_duplicate_host_sink_schema_without_swapping_columns() {
    assert_duplicate_host_sink_schema_restoration(true);
}

#[test]
fn native_query_rejects_host_sink_schema_nullability_drift_before_source_read() {
    let root_schema = Arc::new(Schema::new(vec![Field::new("exec_key", DataType::Int32, false)]));
    let sink_schema = Arc::new(Schema::new(vec![Field::new("frontend_key", DataType::Int32, true)]));

    let error = host_sink_schema_validation_error(root_schema, sink_schema);

    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.operation(), "cudf_query_engine.plan.validate_sink");
    assert_eq!(error.fact_value("field_index"), Some("0"));
    assert_eq!(error.fact_value("sink_nullable"), Some("true"));
    assert_eq!(error.fact_value("root_nullable"), Some("false"));
    assert!(error.fact_value("input_name").is_none());
}

#[test]
fn native_query_rejects_host_sink_schema_metadata_drift_before_source_read() {
    let root_schema = Arc::new(Schema::new(vec![Field::new("exec_key", DataType::Int32, false)]));
    let sink_field = Field::new("frontend_key", DataType::Int32, false)
        .with_metadata(HashMap::from([("field_role".to_string(), "frontend".to_string())]));
    let sink_schema = Arc::new(Schema::new(vec![sink_field]));

    let error = host_sink_schema_validation_error(root_schema, sink_schema);

    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.operation(), "cudf_query_engine.plan.validate_sink");
    assert_eq!(error.fact_value("field_index"), Some("0"));
    assert!(error.fact_value("sink_metadata").is_some_and(|metadata| metadata.contains("field_role")));
    assert!(error.fact_value("root_metadata").is_some_and(|metadata| metadata == "{}"));
    assert!(error.fact_value("input_name").is_none());
}

#[test]
fn native_gpu_result_handle_rejects_sink_schema_name_order_mismatch_before_host_materialization() {
    let dir = unique_temp_dir("cudf-query-engine-native-execution-schema-mismatch");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_input_parquet(&runtime, &path);
    let mismatched_schema = Arc::new(Schema::new(vec![
        Field::new("payload", DataType::Int64, false),
        Field::new("key", DataType::Int32, false),
        Field::new("m00", DataType::Int64, false),
        Field::new("m01", DataType::Int64, false),
    ]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "payload".to_string(), "m00".to_string(), "m01".to_string()],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let limit = builder.add_limit(source, LimitSpec::fetch(2), mismatched_schema);
    let plan = builder.finish_with_host_sink(limit);

    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));
    let error = match execute_query_plan(&context, &plan) {
        Ok(_) => panic!("GPU result handle should reject sink schema name/order mismatches"),
        Err(error) => error,
    };
    assert_eq!(error.code(), QueryErrorCode::CudfInvalidInput);
    assert_eq!(error.operation(), "cudf_query_engine.gpu_result_handle");
    assert_eq!(error.fact_value("cudf_operation"), Some("dataframe::GpuRecordBatch::new"));
    assert_eq!(error.fact_value("cudf_status"), Some("permanent"));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_projection_schema_coercion_rejects_unsupported_output_type_mismatch() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
    let output_schema = Arc::new(Schema::new(vec![Field::new("bad_key", DataType::Utf8, false)]));
    let batch =
        RecordBatch::try_new(Arc::clone(&input_schema), vec![Arc::new(Int32Array::from(vec![1, 2, 3]))]).unwrap();
    let frame = DataFrame::from_record_batch(&runtime, &batch).unwrap();

    let mut builder = QueryPlanBuilder::new();
    let source =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("int_input", Arc::clone(&input_schema))));
    let projection = builder.add_projection(
        source,
        projection_outputs(vec![Expr::column("key").alias("bad_key")], Arc::clone(&output_schema)),
    );
    let plan = builder.finish_gpu_resident(projection);
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));
    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "int_input".to_string(),
        GpuResultHandle::new(&context, GpuDataFrame::new(frame), Arc::clone(&input_schema)).unwrap(),
    );

    let error = match execute_query_plan_with_gpu_inputs(&context, &plan, &gpu_inputs) {
        Ok(_) => panic!("projection schema coercion should reject unsupported Int32 to Utf8 mismatch"),
        Err(error) => error,
    };
    assert_eq!(error.code(), QueryErrorCode::UnsupportedType);
    assert_eq!(error.operation(), "cudf_query_engine.projection.schema_coerce");
    assert_eq!(error.fact_value("reason"), Some("projection_output_schema_mismatch"));
    assert_eq!(error.fact_value("reason_detail"), Some("implicit_cast_not_supported"));
    assert_eq!(error.fact_value("field_name"), Some("bad_key"));
    assert_eq!(error.fact_value("field_index"), Some("0"));
    assert_eq!(error.fact_value("actual_type"), Some("Int32"));
    assert_eq!(error.fact_value("expected_type"), Some("Utf8"));
}

#[test]
fn bounded_projection_schema_coercion_rejects_unsupported_output_type_mismatch_with_budget_facts() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-projection-schema-coerce-reject");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_row_grouped_input_parquet(&path);
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("sum_m00", DataType::Int64, true),
    ]));
    let projection_schema = Arc::new(Schema::new(vec![Field::new("bad_key", DataType::Utf8, false)]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "m00".to_string()],
    );
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("m00"), "sum_m00")],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, aggregate_schema);
    let projection = builder.add_projection(
        aggregate,
        projection_outputs(vec![Expr::column("key").alias("bad_key")], Arc::clone(&projection_schema)),
    );
    let plan = builder.finish_gpu_resident(projection);
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

    let error = match execute_query_plan(&context, &plan) {
        Ok(_) => panic!("bounded projection schema coercion should reject unsupported Int32 to Utf8 mismatch"),
        Err(error) => error,
    };
    assert_eq!(error.code(), QueryErrorCode::UnsupportedType);
    assert_eq!(error.operation(), "cudf_query_engine.projection.schema_coerce");
    assert_eq!(error.fact_value("reason"), Some("projection_output_schema_mismatch"));
    assert_eq!(error.fact_value("reason_detail"), Some("implicit_cast_not_supported"));
    assert_eq!(error.fact_value("field_name"), Some("bad_key"));
    assert_eq!(error.fact_value("field_index"), Some("0"));
    assert_eq!(error.fact_value("actual_type"), Some("Int32"));
    assert_eq!(error.fact_value("expected_type"), Some("Utf8"));
    assert_eq!(error.fact_value("budget_boundary"), Some("post_aggregate_projection"));
    assert!(error.fact_value("budget_bytes").is_some());
    assert_eq!(error.fact_value("budget_enforcement"), Some("hard_memory_resource_limit"));

    fs::remove_dir_all(dir).unwrap();
}

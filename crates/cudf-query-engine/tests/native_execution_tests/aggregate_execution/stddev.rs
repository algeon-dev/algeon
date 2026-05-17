use super::*;

#[test]
fn grouped_stddev_samp_matches_datafusion_welford_contract() {
    let dir = unique_temp_dir("cudf-query-engine-stddev-samp");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_stddev_input_parquet(&runtime, &path);
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("stdev_i32", DataType::Float64, true),
        Field::new("stdev_i64", DataType::Float64, true),
        Field::new("avg_i32", DataType::Float64, true),
        Field::new("cov_i32", DataType::Float64, true),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("stdev_i32", DataType::Float64, true),
        Field::new("stdev_i64", DataType::Float64, true),
        Field::new("avg_i32", DataType::Float64, true),
    ]));

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "v_i32".to_string(), "v_i64".to_string()],
    );
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![
            AggregateExpr::new(AggregateFunction::StddevSamp, Expr::column("v_i32"), "stdev_i32"),
            AggregateExpr::new(AggregateFunction::StddevSamp, Expr::column("v_i64"), "stdev_i64"),
            AggregateExpr::new(AggregateFunction::Avg, Expr::column("v_i32"), "avg_i32"),
        ],
    );
    let projection = projection_outputs(
        vec![
            Expr::column("key"),
            Expr::column("stdev_i32"),
            Expr::column("stdev_i64"),
            Expr::column("avg_i32"),
            Expr::binary(Expr::column("stdev_i32"), BinaryOp::Divide, Expr::column("avg_i32")).alias("cov_i32"),
        ],
        Arc::clone(&output_schema),
    );
    let sort = SortSpec::new(vec![SortExpr::new(Expr::column("key"), false, false)]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, aggregate_schema);
    let projection = builder.add_projection(aggregate, projection);
    let sort = builder.add_sort(projection, sort, Arc::clone(&output_schema));
    let plan = builder.finish_with_host_sink(sort);

    let context = ExecutionContext::builder().with_shared_runtime(runtime).build().unwrap();
    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };

    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2, 3, 4]);
    assert_eq!(float64_array(batch, 1).value(0), 2.0_f64.sqrt());
    assert!(float64_array(batch, 1).is_null(1));
    assert!(float64_array(batch, 1).is_null(2));
    assert!(float64_array(batch, 1).is_null(3));
    assert_eq!(float64_array(batch, 2).value(0), 200.0_f64.sqrt());
    assert!(float64_array(batch, 2).is_null(1));
    assert!(float64_array(batch, 2).is_null(2));
    assert!(float64_array(batch, 2).is_null(3));
    assert_eq!(float64_array(batch, 3).value(0), 3.0);
    assert!(float64_array(batch, 3).is_null(3));
    assert_eq!(float64_array(batch, 4).value(0), 2.0_f64.sqrt() / 3.0);
    assert!(float64_array(batch, 4).is_null(1));
    assert!(float64_array(batch, 4).is_null(2));
    assert!(float64_array(batch, 4).is_null(3));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn global_stddev_samp_count_zero_and_one_return_null() {
    let dir = unique_temp_dir("cudf-query-engine-global-stddev-samp");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, true)]));
    let batch =
        RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![Some(7), None]))]).unwrap();
    DataFrame::from_record_batch(&runtime, &batch).unwrap().write_parquet(&path).unwrap();
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("stdev", DataType::Float64, true),
        Field::new("avg_v", DataType::Float64, true),
    ]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&schema),
        vec!["v".to_string()],
    );
    let aggregate = AggregateSpec::new(
        Vec::new(),
        vec![
            AggregateExpr::new(AggregateFunction::StddevSamp, Expr::column("v"), "stdev"),
            AggregateExpr::new(AggregateFunction::Avg, Expr::column("v"), "avg_v"),
        ],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, output_schema);
    let plan = builder.finish_with_host_sink(aggregate);

    let context = ExecutionContext::builder().with_shared_runtime(runtime).build().unwrap();
    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };

    assert_eq!(batch.num_rows(), 1);
    assert!(float64_array(batch, 0).is_null(0));
    assert_eq!(float64_array(batch, 1).value(0), 7.0);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn global_stddev_samp_preserves_multifile_input_order_bits() {
    let dir = unique_temp_dir("cudf-query-engine-ordered-stddev-samp");
    let first_path = dir.join("first.parquet");
    let second_path = dir.join("second.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, true)]));
    write_f64_parquet(&runtime, &schema, &first_path, vec![Some(1.0e16), Some(1.0e-16)]);
    write_f64_parquet(&runtime, &schema, &second_path, vec![Some(1.0), Some(2.0), Some(3.0)]);

    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(&first_path).with_file_size_bytes(fs::metadata(&first_path).unwrap().len()),
            LocalParquetFile::new(&second_path).with_file_size_bytes(fs::metadata(&second_path).unwrap().len()),
        ],
        Arc::clone(&schema),
        vec!["v".to_string()],
    );
    let output_schema = Arc::new(Schema::new(vec![Field::new("stdev", DataType::Float64, true)]));
    let aggregate = AggregateSpec::new(
        Vec::new(),
        vec![AggregateExpr::new(AggregateFunction::StddevSamp, Expr::column("v"), "stdev")],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, Arc::clone(&output_schema));
    let plan = builder.finish_with_host_sink(aggregate);

    let context = ExecutionContext::builder().with_shared_runtime(runtime).build().unwrap();
    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };

    let expected = welford_stddev_samp([1.0e16, 1.0e-16, 1.0, 2.0, 3.0]).unwrap();
    let reversed = welford_stddev_samp([3.0, 2.0, 1.0, 1.0e-16, 1.0e16]).unwrap();
    assert_ne!(expected.to_bits(), reversed.to_bits(), "test values must expose Welford order sensitivity");
    assert_eq!(float64_array(batch, 0).value(0).to_bits(), expected.to_bits());

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

fn write_f64_parquet(runtime: &cudf::Runtime, schema: &Arc<Schema>, path: &Path, values: Vec<Option<f64>>) {
    let batch = RecordBatch::try_new(Arc::clone(schema), vec![Arc::new(Float64Array::from(values))]).unwrap();
    DataFrame::from_record_batch(runtime, &batch).unwrap().write_parquet(path).unwrap();
}

fn welford_stddev_samp(values: impl IntoIterator<Item = f64>) -> Option<f64> {
    let mut count = 0u64;
    let mut mean = 0.0;
    let mut m2 = 0.0;
    for value in values {
        let new_count = count + 1;
        let delta1 = value - mean;
        let new_mean = delta1 / new_count as f64 + mean;
        let delta2 = value - new_mean;
        m2 += delta1 * delta2;
        mean = new_mean;
        count = new_count;
    }
    (count > 1).then(|| (m2 / (count - 1) as f64).sqrt())
}

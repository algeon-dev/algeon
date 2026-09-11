use std::{fs::File, path::Path, sync::Arc};

use query_engine::mode::StatisticalAggregateExecutionMode;

use super::*;

#[test]
fn grouped_stddev_samp_matches_datafusion_welford_contract() {
    let dir = unique_temp_dir("query-engine-stddev-samp");
    let path = dir.join("input.parquet");
    let context =
        backend_context_with_statistical_aggregate_mode(StatisticalAggregateExecutionMode::DataFusionExactHostWelford);
    let runtime = Arc::new(context.runtime().unwrap());
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
    assert_eq!(
        aggregate_strategy_fact(result.metrics(), "selected_strategy").as_deref(),
        Some("datafusion_exact_host_welford")
    );
    assert_eq!(
        aggregate_strategy_fact(result.metrics(), "statistical_aggregate_execution_mode").as_deref(),
        Some("datafusion_exact_host_welford")
    );
    assert_eq!(
        aggregate_strategy_fact(result.metrics(), "statistical_aggregate_tolerance_contract").as_deref(),
        Some("exact")
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn grouped_stddev_samp_handles_repeated_string_multi_key_groups() {
    let dir = unique_temp_dir("query-engine-string-multi-key-stddev-samp");
    let path = dir.join("input.parquet");
    let context =
        backend_context_with_statistical_aggregate_mode(StatisticalAggregateExecutionMode::DataFusionExactHostWelford);
    let runtime = Arc::new(context.runtime().unwrap());
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("w_name", DataType::Utf8, false),
        Field::new("item_sk", DataType::Int64, false),
        Field::new("warehouse_sk", DataType::Int64, false),
        Field::new("month", DataType::Int64, false),
        Field::new("qty", DataType::Float64, true),
    ]));
    let input = RecordBatch::try_new(
        Arc::clone(&input_schema),
        vec![
            Arc::new(StringArray::from(vec!["w1", "w1", "w1", "w2", "w2"])),
            Arc::new(Int64Array::from(vec![10, 10, 10, 20, 20])),
            Arc::new(Int64Array::from(vec![1, 1, 1, 2, 2])),
            Arc::new(Int64Array::from(vec![1, 1, 2, 1, 1])),
            Arc::new(Float64Array::from(vec![Some(10.0), Some(14.0), Some(5.0), Some(3.0), Some(7.0)])),
        ],
    )
    .unwrap();
    DataFrame::from_record_batch(&runtime, &input).unwrap().write_parquet(&path).unwrap();
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("w_name", DataType::Utf8, false),
        Field::new("item_sk", DataType::Int64, false),
        Field::new("warehouse_sk", DataType::Int64, false),
        Field::new("month", DataType::Int64, false),
        Field::new("stdev_qty", DataType::Float64, true),
        Field::new("avg_qty", DataType::Float64, true),
    ]));

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec![
            "w_name".to_string(),
            "item_sk".to_string(),
            "warehouse_sk".to_string(),
            "month".to_string(),
            "qty".to_string(),
        ],
    );
    let aggregate = AggregateSpec::new(
        vec![Expr::column("w_name"), Expr::column("item_sk"), Expr::column("warehouse_sk"), Expr::column("month")],
        vec![
            AggregateExpr::new(AggregateFunction::StddevSamp, Expr::column("qty"), "stdev_qty"),
            AggregateExpr::new(AggregateFunction::Avg, Expr::column("qty"), "avg_qty"),
        ],
    );
    let sort = SortSpec::new(vec![
        SortExpr::new(Expr::column("w_name"), false, false),
        SortExpr::new(Expr::column("item_sk"), false, false),
        SortExpr::new(Expr::column("warehouse_sk"), false, false),
        SortExpr::new(Expr::column("month"), false, false),
    ]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, Arc::clone(&output_schema));
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&output_schema));
    let plan = builder.finish_with_host_sink(sort);

    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };

    assert_eq!(string_array(batch, 0).iter().collect::<Vec<_>>(), vec![Some("w1"), Some("w1"), Some("w2")]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[10, 10, 20]);
    assert_eq!(int64_array(batch, 2).values().as_ref(), &[1, 1, 2]);
    assert_eq!(int64_array(batch, 3).values().as_ref(), &[1, 2, 1]);
    assert_eq!(float64_array(batch, 4).value(0), 8.0_f64.sqrt());
    assert!(float64_array(batch, 4).is_null(1));
    assert_eq!(float64_array(batch, 4).value(2), 8.0_f64.sqrt());
    assert_eq!(float64_array(batch, 5).values().as_ref(), &[12.0, 5.0, 5.0]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn global_stddev_samp_count_zero_and_one_return_null() {
    assert_global_stddev_samp_count_zero_and_one_return_null(
        StatisticalAggregateExecutionMode::DataFusionExactHostWelford,
    );
}

#[test]
fn tolerant_global_stddev_samp_count_zero_and_one_return_null() {
    assert_global_stddev_samp_count_zero_and_one_return_null(StatisticalAggregateExecutionMode::GpuNativeTolerant);
}

fn assert_global_stddev_samp_count_zero_and_one_return_null(mode: StatisticalAggregateExecutionMode) {
    let dir = unique_temp_dir("query-engine-global-stddev-samp");
    let path = dir.join("input.parquet");
    let context = backend_context_with_statistical_aggregate_mode(mode);
    let runtime = Arc::new(context.runtime().unwrap());
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
    let dir = unique_temp_dir("query-engine-ordered-stddev-samp");
    let first_path = dir.join("first.parquet");
    let second_path = dir.join("second.parquet");
    let context =
        backend_context_with_statistical_aggregate_mode(StatisticalAggregateExecutionMode::DataFusionExactHostWelford);
    let runtime = Arc::new(context.runtime().unwrap());
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

#[test]
fn global_stddev_samp_preserves_row_group_filter_projection_order_bits() {
    let context =
        backend_context_with_statistical_aggregate_mode(StatisticalAggregateExecutionMode::DataFusionExactHostWelford);
    let dir = unique_temp_dir("query-engine-row-group-ordered-stddev-samp");
    let first_path = dir.join("first.parquet");
    let second_path = dir.join("second.parquet");
    let schema = Arc::new(Schema::new(vec![
        Field::new("v", DataType::Float64, true),
        Field::new("keep", DataType::Int32, false),
    ]));
    write_f64_keep_row_grouped_parquet(
        Arc::clone(&schema),
        &first_path,
        vec![(Some(1.0e16), 1), (Some(999.0), 0), (Some(1.0e-16), 1)],
        1,
    );
    write_f64_keep_row_grouped_parquet(
        Arc::clone(&schema),
        &second_path,
        vec![(Some(1.0), 1), (Some(888.0), 0), (Some(2.0), 1), (Some(3.0), 1)],
        2,
    );

    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(&first_path).with_file_size_bytes(fs::metadata(&first_path).unwrap().len()),
            LocalParquetFile::new(&second_path).with_file_size_bytes(fs::metadata(&second_path).unwrap().len()),
        ],
        Arc::clone(&schema),
        vec!["v".to_string(), "keep".to_string()],
    );
    let projected_schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, true)]));
    let output_schema = Arc::new(Schema::new(vec![Field::new("stdev", DataType::Float64, true)]));
    let aggregate = AggregateSpec::new(
        Vec::new(),
        vec![AggregateExpr::new(AggregateFunction::StddevSamp, Expr::column("v"), "stdev")],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let filtered = builder.add_filter(
        source,
        Expr::binary(Expr::column("keep"), BinaryOp::Gt, Expr::literal(LiteralValue::Int32(0))),
        Arc::clone(&schema),
    );
    let projected =
        builder.add_projection(filtered, projection_outputs(vec![Expr::column("v")], Arc::clone(&projected_schema)));
    let aggregate = builder.add_aggregate(projected, aggregate, Arc::clone(&output_schema));
    let plan = builder.finish_with_host_sink(aggregate);

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

#[test]
fn join_grouped_stddev_samp_uses_datafusion_probe_major_order_bits() {
    let dir = unique_temp_dir("query-engine-join-ordered-stddev-samp");
    let left_paths = [dir.join("left-0.parquet"), dir.join("left-1.parquet"), dir.join("left-2.parquet")];
    let right_path = dir.join("right.parquet");
    let context =
        backend_context_with_statistical_aggregate_mode(StatisticalAggregateExecutionMode::DataFusionExactHostWelford);
    let runtime = Arc::new(context.runtime().unwrap());
    let left_schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("v", DataType::Float64, true)]));
    let right_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("grp", DataType::Int32, false),
        Field::new("keep", DataType::Int32, false),
    ]));
    let left_parts = [vec![1.0e16, 1.0e-16], vec![1.0, 2.0], vec![3.0]];
    let left_values = left_parts.iter().flat_map(|part| part.iter().copied()).collect::<Vec<_>>();
    for (path, values) in left_paths.iter().zip(left_parts.iter()) {
        let left_batch = RecordBatch::try_new(
            Arc::clone(&left_schema),
            vec![Arc::new(Int32Array::from(vec![1; values.len()])), Arc::new(Float64Array::from(values.clone()))],
        )
        .unwrap();
        DataFrame::from_record_batch(&runtime, &left_batch).unwrap().write_parquet(path).unwrap();
    }
    let right_batch = RecordBatch::try_new(
        Arc::clone(&right_schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 1, 1])),
            Arc::new(Int32Array::from(vec![7, 7, 7])),
            Arc::new(Int32Array::from(vec![1, 0, 1])),
        ],
    )
    .unwrap();
    DataFrame::from_record_batch(&runtime, &right_batch).unwrap().write_parquet(&right_path).unwrap();

    let left = LocalParquetSource::from_files(
        left_paths
            .iter()
            .zip(left_parts.iter())
            .map(|(path, values)| {
                LocalParquetFile::new(path)
                    .with_file_size_bytes(fs::metadata(path).unwrap().len())
                    .with_row_group_rows(vec![values.len() as u64])
            })
            .collect(),
        Arc::clone(&left_schema),
        vec!["k".to_string(), "v".to_string()],
    );
    let right = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&right_path).with_file_size_bytes(fs::metadata(&right_path).unwrap().len())],
        Arc::clone(&right_schema),
        vec!["k".to_string(), "grp".to_string(), "keep".to_string()],
    );
    let projected_right_schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("grp", DataType::Int32, false)]));
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("grp", DataType::Int32, false),
        Field::new("v", DataType::Float64, true),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("grp", DataType::Int32, false),
        Field::new("stdev", DataType::Float64, true),
        Field::new("avg_v", DataType::Float64, true),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("grp", DataType::Int32, false),
        Field::new("stdev", DataType::Float64, true),
        Field::new("avg_v", DataType::Float64, true),
        Field::new("cov", DataType::Float64, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(left));
    let right = builder.add_source(SourceNode::LocalParquet(right));
    let right = builder.add_filter(
        right,
        Expr::binary(Expr::column("keep"), BinaryOp::Gt, Expr::literal(LiteralValue::Int32(0))),
        Arc::clone(&right_schema),
    );
    let right = builder.add_projection(
        right,
        projection_outputs(vec![Expr::column("k"), Expr::column("grp")], Arc::clone(&projected_right_schema)),
    );
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                query_engine::expr::ColumnRef::input(0, "k", "k"),
                query_engine::expr::ColumnRef::input(1, "k", "k"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::right(query_engine::expr::ColumnRef::input(1, "grp", "grp"), "grp"),
            JoinOutputColumn::left(query_engine::expr::ColumnRef::input(0, "v", "v"), "v"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&join_schema),
    );
    let aggregate = builder.add_aggregate(
        join,
        AggregateSpec::new(
            vec![Expr::column("grp")],
            vec![
                AggregateExpr::new(AggregateFunction::StddevSamp, Expr::column("v"), "stdev"),
                AggregateExpr::new(AggregateFunction::Avg, Expr::column("v"), "avg_v"),
            ],
        ),
        Arc::clone(&aggregate_schema),
    );
    let projection = builder.add_projection(
        aggregate,
        projection_outputs(
            vec![
                Expr::column("grp"),
                Expr::column("stdev"),
                Expr::column("avg_v"),
                Expr::binary(Expr::column("stdev"), BinaryOp::Divide, Expr::column("avg_v")).alias("cov"),
            ],
            Arc::clone(&output_schema),
        ),
    );
    let plan = builder.finish_with_host_sink(projection);

    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };

    let expected_values = left_values.iter().copied().chain(left_values.iter().copied()).collect::<Vec<_>>();
    let reversed_values =
        left_values.iter().rev().copied().chain(left_values.iter().rev().copied()).collect::<Vec<_>>();
    let expected = welford_stddev_samp(expected_values.iter().copied()).unwrap();
    let reversed = welford_stddev_samp(reversed_values).unwrap();
    let expected_avg = ordered_avg(expected_values.iter().copied()).unwrap();
    assert_ne!(expected.to_bits(), reversed.to_bits(), "test values must expose join-order sensitivity");
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[7]);
    assert_eq!(float64_array(batch, 1).value(0).to_bits(), expected.to_bits());
    assert_eq!(float64_array(batch, 3).value(0).to_bits(), (expected / expected_avg).to_bits());
    assert!(
        result
            .metrics()
            .source_read_runtime_rows
            .iter()
            .any(|row| row.read_kind == "chunk_batch" && row.chunk_count > 1),
        "order-tracked join stddev should keep multi-file source batching enabled"
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_join_grouped_stddev_samp_uses_datafusion_probe_major_order_bits() {
    // Bounded leaf source reads run on spawned worker threads. The exact-order
    // scope travels with the execution context clone, so worker-read chunks
    // must carry the same hidden order columns as caller-thread reads.
    // A one-shot hash join over a large probe does not preserve probe order,
    // so without restored order columns the pre-aggregate sort is a no-op and
    // Welford input order is lost at the bit level.
    let dir = unique_temp_dir("query-engine-bounded-join-ordered-stddev-samp");
    const PROBE_FILES: usize = 10;
    const PROBE_ROWS_PER_FILE: usize = 10_000;
    const PROBE_ROWS: usize = PROBE_FILES * PROBE_ROWS_PER_FILE;
    const SENSITIVE_PATTERN: [f64; 5] = [1.0e16, 1.0e-16, 1.0, 2.0, 3.0];
    let left_paths = (0..PROBE_FILES).map(|file| dir.join(format!("left-{file}.parquet"))).collect::<Vec<_>>();
    let right_path = dir.join("right.parquet");
    let context = backend_context_with_grant_and_statistical_aggregate_mode(
        16 * 1024 * 1024,
        StatisticalAggregateExecutionMode::DataFusionExactHostWelford,
    );
    let runtime = Arc::new(context.runtime().unwrap());
    let left_schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("v", DataType::Float64, true)]));
    let right_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("grp", DataType::Int32, false),
        Field::new("keep", DataType::Int32, false),
    ]));
    let right_batch = RecordBatch::try_new(
        Arc::clone(&right_schema),
        vec![
            Arc::new(Int32Array::from_iter_values(0..10)),
            Arc::new(Int32Array::from(vec![7; 10])),
            Arc::new(Int32Array::from(vec![1; 10])),
        ],
    )
    .unwrap();
    DataFrame::from_record_batch(&runtime, &right_batch).unwrap().write_parquet(&right_path).unwrap();
    let mut left_values = Vec::with_capacity(PROBE_ROWS);
    for (file_ordinal, path) in left_paths.iter().enumerate() {
        let base = file_ordinal * PROBE_ROWS_PER_FILE;
        let keys = (0..PROBE_ROWS_PER_FILE).map(|row| ((base + row) * 7 % 10) as i32).collect::<Vec<_>>();
        let values = (0..PROBE_ROWS_PER_FILE)
            .map(|row| SENSITIVE_PATTERN[(base + row) % SENSITIVE_PATTERN.len()])
            .collect::<Vec<_>>();
        left_values.extend(values.iter().copied());
        let left_batch = RecordBatch::try_new(
            Arc::clone(&left_schema),
            vec![Arc::new(Int32Array::from(keys)), Arc::new(Float64Array::from(values))],
        )
        .unwrap();
        DataFrame::from_record_batch(&runtime, &left_batch).unwrap().write_parquet(path).unwrap();
    }

    // Probe files deliberately carry no byte estimates so chunk batching keeps
    // every chunk a singleton batch, which routes leaf reads through the
    // spawned source-read workers instead of serial batch reads.
    let left = LocalParquetSource::from_files(
        left_paths.iter().map(LocalParquetFile::new).collect(),
        Arc::clone(&left_schema),
        vec!["k".to_string(), "v".to_string()],
    );
    let right = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&right_path).with_file_size_bytes(fs::metadata(&right_path).unwrap().len())],
        Arc::clone(&right_schema),
        vec!["k".to_string(), "grp".to_string(), "keep".to_string()],
    );
    let projected_right_schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("grp", DataType::Int32, false)]));
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("grp", DataType::Int32, false),
        Field::new("v", DataType::Float64, true),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("grp", DataType::Int32, false),
        Field::new("stdev", DataType::Float64, true),
        Field::new("avg_v", DataType::Float64, true),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("grp", DataType::Int32, false),
        Field::new("stdev", DataType::Float64, true),
        Field::new("avg_v", DataType::Float64, true),
        Field::new("cov", DataType::Float64, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(left));
    let right = builder.add_source(SourceNode::LocalParquet(right));
    let right = builder.add_filter(
        right,
        Expr::binary(Expr::column("keep"), BinaryOp::Gt, Expr::literal(LiteralValue::Int32(0))),
        Arc::clone(&right_schema),
    );
    let right = builder.add_projection(
        right,
        projection_outputs(vec![Expr::column("k"), Expr::column("grp")], Arc::clone(&projected_right_schema)),
    );
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                query_engine::expr::ColumnRef::input(0, "k", "k"),
                query_engine::expr::ColumnRef::input(1, "k", "k"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::right(query_engine::expr::ColumnRef::input(1, "grp", "grp"), "grp"),
            JoinOutputColumn::left(query_engine::expr::ColumnRef::input(0, "v", "v"), "v"),
        ])
        .with_build_side(JoinSide::Right)
        .with_datafusion_order_preserved_side(JoinSide::Left),
        Arc::clone(&join_schema),
    );
    let aggregate = builder.add_aggregate(
        join,
        AggregateSpec::new(
            vec![Expr::column("grp")],
            vec![
                AggregateExpr::new(AggregateFunction::StddevSamp, Expr::column("v"), "stdev"),
                AggregateExpr::new(AggregateFunction::Avg, Expr::column("v"), "avg_v"),
            ],
        ),
        Arc::clone(&aggregate_schema),
    );
    let projection = builder.add_projection(
        aggregate,
        projection_outputs(
            vec![
                Expr::column("grp"),
                Expr::column("stdev"),
                Expr::column("avg_v"),
                Expr::binary(Expr::column("stdev"), BinaryOp::Divide, Expr::column("avg_v")).alias("cov"),
            ],
            Arc::clone(&output_schema),
        ),
    );
    let plan = builder.finish_with_host_sink(projection);

    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };

    assert_eq!(result.metrics().memory_control_policy.as_str(), "bounded");
    assert_eq!(result.metrics().compiled_pipeline_attempts, 1);
    let expected = welford_stddev_samp(left_values.iter().copied()).unwrap();
    let reversed = welford_stddev_samp(left_values.iter().rev().copied()).unwrap();
    let expected_avg = ordered_avg(left_values.iter().copied()).unwrap();
    assert_ne!(expected.to_bits(), reversed.to_bits(), "test values must expose join-order sensitivity");
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[7]);
    assert_eq!(float64_array(batch, 1).value(0).to_bits(), expected.to_bits());
    assert_eq!(float64_array(batch, 3).value(0).to_bits(), (expected / expected_avg).to_bits());

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn join_grouped_stddev_samp_can_preserve_left_probe_order_bits() {
    let dir = unique_temp_dir("query-engine-left-probe-ordered-stddev-samp");
    let fact_path = dir.join("fact.parquet");
    let date_path = dir.join("date.parquet");
    let context =
        backend_context_with_statistical_aggregate_mode(StatisticalAggregateExecutionMode::DataFusionExactHostWelford);
    let runtime = Arc::new(context.runtime().unwrap());
    let fact_schema = Arc::new(Schema::new(vec![
        Field::new("date_sk", DataType::Int64, false),
        Field::new("grp", DataType::Int32, false),
        Field::new("qty", DataType::Int32, true),
    ]));
    let date_schema = Arc::new(Schema::new(vec![Field::new("date_sk", DataType::Int64, false)]));
    let fact_values = [378, 487, 1, 0];
    let fact_batch = RecordBatch::try_new(
        Arc::clone(&fact_schema),
        vec![
            Arc::new(Int64Array::from(vec![2_451_928, 2_451_935, 2_451_914, 2_451_921])),
            Arc::new(Int32Array::from(vec![7; 4])),
            Arc::new(Int32Array::from(fact_values.to_vec())),
        ],
    )
    .unwrap();
    let date_batch = RecordBatch::try_new(
        Arc::clone(&date_schema),
        vec![Arc::new(Int64Array::from(vec![2_451_914, 2_451_921, 2_451_928, 2_451_935]))],
    )
    .unwrap();
    DataFrame::from_record_batch(&runtime, &fact_batch).unwrap().write_parquet(&fact_path).unwrap();
    DataFrame::from_record_batch(&runtime, &date_batch).unwrap().write_parquet(&date_path).unwrap();

    let fact = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&fact_path).with_file_size_bytes(fs::metadata(&fact_path).unwrap().len())],
        Arc::clone(&fact_schema),
        vec!["date_sk".to_string(), "grp".to_string(), "qty".to_string()],
    );
    let date = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&date_path).with_file_size_bytes(fs::metadata(&date_path).unwrap().len())],
        Arc::clone(&date_schema),
        vec!["date_sk".to_string()],
    );
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("grp", DataType::Int32, false),
        Field::new("qty", DataType::Int32, true),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("grp", DataType::Int32, false),
        Field::new("stdev", DataType::Float64, true),
        Field::new("avg_qty", DataType::Float64, true),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("grp", DataType::Int32, false),
        Field::new("stdev", DataType::Float64, true),
        Field::new("avg_qty", DataType::Float64, true),
        Field::new("cov", DataType::Float64, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(SourceNode::LocalParquet(fact));
    let date = builder.add_source(SourceNode::LocalParquet(date));
    let join = builder.add_join(
        fact,
        date,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                query_engine::expr::ColumnRef::input(0, "date_sk", "date_sk"),
                query_engine::expr::ColumnRef::input(1, "date_sk", "date_sk"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(query_engine::expr::ColumnRef::input(0, "grp", "grp"), "grp"),
            JoinOutputColumn::left(query_engine::expr::ColumnRef::input(0, "qty", "qty"), "qty"),
        ])
        .with_build_side(JoinSide::Right)
        .with_datafusion_order_preserved_side(JoinSide::Left),
        Arc::clone(&join_schema),
    );
    let aggregate = builder.add_aggregate(
        join,
        AggregateSpec::new(
            vec![Expr::column("grp")],
            vec![
                AggregateExpr::new(AggregateFunction::StddevSamp, Expr::column("qty"), "stdev"),
                AggregateExpr::new(AggregateFunction::Avg, Expr::column("qty"), "avg_qty"),
            ],
        ),
        Arc::clone(&aggregate_schema),
    );
    let projection = builder.add_projection(
        aggregate,
        projection_outputs(
            vec![
                Expr::column("grp"),
                Expr::column("stdev"),
                Expr::column("avg_qty"),
                Expr::binary(Expr::column("stdev"), BinaryOp::Divide, Expr::column("avg_qty")).alias("cov"),
            ],
            Arc::clone(&output_schema),
        ),
    );
    let plan = builder.finish_with_host_sink(projection);

    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };

    let expected = welford_stddev_samp(fact_values.iter().map(|value| *value as f64)).unwrap();
    let right_order = welford_stddev_samp([1.0, 0.0, 378.0, 487.0]).unwrap();
    let expected_avg = ordered_avg(fact_values.iter().map(|value| *value as f64)).unwrap();
    assert_ne!(expected.to_bits(), right_order.to_bits(), "test values must expose preserved-side sensitivity");
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[7]);
    assert_eq!(float64_array(batch, 1).value(0).to_bits(), expected.to_bits());
    assert_eq!(float64_array(batch, 3).value(0).to_bits(), (expected / expected_avg).to_bits());

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_residual_join_grouped_stddev_samp_keeps_order_bits_after_filter() {
    let dir = unique_temp_dir("query-engine-bounded-residual-join-ordered-stddev-samp");
    let fact_path = dir.join("fact.parquet");
    let date_path = dir.join("date.parquet");
    let context =
        backend_context_with_statistical_aggregate_mode(StatisticalAggregateExecutionMode::DataFusionExactHostWelford);
    let runtime = Arc::new(context.runtime().unwrap());
    let fact_schema = Arc::new(Schema::new(vec![
        Field::new("date_sk", DataType::Int64, false),
        Field::new("grp", DataType::Int32, false),
        Field::new("qty", DataType::Int32, true),
    ]));
    let date_schema = Arc::new(Schema::new(vec![Field::new("date_sk", DataType::Int64, false)]));
    let fact_values = [378, 487, 1, 0];
    let fact_batch = RecordBatch::try_new(
        Arc::clone(&fact_schema),
        vec![
            Arc::new(Int64Array::from(vec![2_451_928, 2_451_935, 2_451_914, 2_451_921])),
            Arc::new(Int32Array::from(vec![7; 4])),
            Arc::new(Int32Array::from(fact_values.to_vec())),
        ],
    )
    .unwrap();
    let date_batch = RecordBatch::try_new(
        Arc::clone(&date_schema),
        vec![Arc::new(Int64Array::from(vec![2_451_914, 2_451_921, 2_451_928, 2_451_935]))],
    )
    .unwrap();
    DataFrame::from_record_batch(&runtime, &fact_batch).unwrap().write_parquet(&fact_path).unwrap();
    DataFrame::from_record_batch(&runtime, &date_batch).unwrap().write_parquet(&date_path).unwrap();

    let fact = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&fact_path).with_file_size_bytes(fs::metadata(&fact_path).unwrap().len())],
        Arc::clone(&fact_schema),
        vec!["date_sk".to_string(), "grp".to_string(), "qty".to_string()],
    );
    let date = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&date_path).with_file_size_bytes(fs::metadata(&date_path).unwrap().len())],
        Arc::clone(&date_schema),
        vec!["date_sk".to_string()],
    );
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("grp", DataType::Int32, false),
        Field::new("qty", DataType::Int32, true),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("grp", DataType::Int32, false),
        Field::new("stdev", DataType::Float64, true),
        Field::new("avg_qty", DataType::Float64, true),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("grp", DataType::Int32, false),
        Field::new("stdev", DataType::Float64, true),
        Field::new("avg_qty", DataType::Float64, true),
        Field::new("cov", DataType::Float64, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(SourceNode::LocalParquet(fact));
    let date = builder.add_source(SourceNode::LocalParquet(date));
    let join = builder.add_join(
        fact,
        date,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                query_engine::expr::ColumnRef::input(0, "date_sk", "date_sk"),
                query_engine::expr::ColumnRef::input(1, "date_sk", "date_sk"),
            )],
        )
        .with_residual_predicate(Expr::case(
            vec![query_engine::expr::CaseWhen::new(
                Expr::binary(
                    Expr::input_column(0, "date_sk", "date_sk"),
                    BinaryOp::Eq,
                    Expr::input_column(1, "date_sk", "date_sk"),
                ),
                Expr::literal(LiteralValue::Boolean(true)),
            )],
            Expr::literal(LiteralValue::Boolean(false)),
        ))
        .with_output_projection(vec![
            JoinOutputColumn::left(query_engine::expr::ColumnRef::input(0, "grp", "grp"), "grp"),
            JoinOutputColumn::left(query_engine::expr::ColumnRef::input(0, "qty", "qty"), "qty"),
        ])
        .with_build_side(JoinSide::Right)
        .with_datafusion_order_preserved_side(JoinSide::Left),
        Arc::clone(&join_schema),
    );
    let aggregate = builder.add_aggregate(
        join,
        AggregateSpec::new(
            vec![Expr::column("grp")],
            vec![
                AggregateExpr::new(AggregateFunction::StddevSamp, Expr::column("qty"), "stdev"),
                AggregateExpr::new(AggregateFunction::Avg, Expr::column("qty"), "avg_qty"),
            ],
        ),
        Arc::clone(&aggregate_schema),
    );
    let projection = builder.add_projection(
        aggregate,
        projection_outputs(
            vec![
                Expr::column("grp"),
                Expr::column("stdev"),
                Expr::column("avg_qty"),
                Expr::binary(Expr::column("stdev"), BinaryOp::Divide, Expr::column("avg_qty")).alias("cov"),
            ],
            Arc::clone(&output_schema),
        ),
    );
    let plan = builder.finish_with_host_sink(projection);

    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };

    let expected = welford_stddev_samp(fact_values.iter().map(|value| *value as f64)).unwrap();
    let right_order = welford_stddev_samp([1.0, 0.0, 378.0, 487.0]).unwrap();
    let expected_avg = ordered_avg(fact_values.iter().map(|value| *value as f64)).unwrap();
    let join_strategies = result
        .metrics()
        .operators
        .iter()
        .filter_map(|operator| operator.join.as_ref())
        .map(|join| (join.execution_strategy.as_str(), join.residual_predicate))
        .collect::<Vec<_>>();
    assert!(
        join_strategies.iter().any(|(strategy, residual)| *strategy == "bounded_one_shot_equi_join" && *residual),
        "{join_strategies:?}"
    );
    assert_ne!(expected.to_bits(), right_order.to_bits(), "test values must expose preserved-side sensitivity");
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[7]);
    assert_eq!(float64_array(batch, 1).value(0).to_bits(), expected.to_bits());
    assert_eq!(float64_array(batch, 3).value(0).to_bits(), (expected / expected_avg).to_bits());

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_grouped_stddev_samp_over_empty_join_stream_returns_empty_batch() {
    let context =
        backend_context_with_statistical_aggregate_mode(StatisticalAggregateExecutionMode::DataFusionExactHostWelford);
    let runtime = Arc::new(context.runtime().unwrap());
    let dir = unique_temp_dir("query-engine-bounded-empty-join-stddev");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    let left_schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("v", DataType::Float64, true)]));
    let right_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("i_item_id", DataType::Int32, false),
        Field::new("i_item_desc", DataType::Utf8, false),
        Field::new("s_state", DataType::Utf8, false),
    ]));
    let left_batch = RecordBatch::try_new(
        Arc::clone(&left_schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(Float64Array::from(vec![Some(10.0), Some(20.0), Some(30.0)])),
        ],
    )
    .unwrap();
    let right_batch = RecordBatch::try_new(
        Arc::clone(&right_schema),
        vec![
            Arc::new(Int32Array::from(vec![99])),
            Arc::new(Int32Array::from(vec![7])),
            Arc::new(StringArray::from(vec!["no-match-item"])),
            Arc::new(StringArray::from(vec!["ZZ"])),
        ],
    )
    .unwrap();
    write_batch_with_row_group_size(Arc::clone(&left_schema), &left_path, &left_batch, 1);
    DataFrame::from_record_batch(&runtime, &right_batch).unwrap().write_parquet(&right_path).unwrap();

    let join_schema = Arc::new(Schema::new(vec![
        Field::new("i_item_id", DataType::Int32, false),
        Field::new("i_item_desc", DataType::Utf8, false),
        Field::new("s_state", DataType::Utf8, false),
        Field::new("v", DataType::Float64, true),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("i_item_id", DataType::Int32, false),
        Field::new("i_item_desc", DataType::Utf8, false),
        Field::new("s_state", DataType::Utf8, false),
        Field::new("count_v", DataType::Int64, false),
        Field::new("avg_v", DataType::Float64, true),
        Field::new("stdev_v", DataType::Float64, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&left_path).with_file_size_bytes(fs::metadata(&left_path).unwrap().len())],
        Arc::clone(&left_schema),
        vec!["k".to_string(), "v".to_string()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&right_path).with_file_size_bytes(fs::metadata(&right_path).unwrap().len())],
        Arc::clone(&right_schema),
        vec!["k".to_string(), "i_item_id".to_string(), "i_item_desc".to_string(), "s_state".to_string()],
    )));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                query_engine::expr::ColumnRef::input(0, "k", "k"),
                query_engine::expr::ColumnRef::input(1, "k", "k"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::right(query_engine::expr::ColumnRef::input(1, "i_item_id", "i_item_id"), "i_item_id"),
            JoinOutputColumn::right(
                query_engine::expr::ColumnRef::input(1, "i_item_desc", "i_item_desc"),
                "i_item_desc",
            ),
            JoinOutputColumn::right(query_engine::expr::ColumnRef::input(1, "s_state", "s_state"), "s_state"),
            JoinOutputColumn::left(query_engine::expr::ColumnRef::input(0, "v", "v"), "v"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&join_schema),
    );
    let aggregate = builder.add_aggregate(
        join,
        AggregateSpec::new(
            vec![Expr::column("i_item_id"), Expr::column("i_item_desc"), Expr::column("s_state")],
            vec![
                AggregateExpr::new(AggregateFunction::Count, Expr::column("v"), "count_v"),
                AggregateExpr::new(AggregateFunction::Avg, Expr::column("v"), "avg_v"),
                AggregateExpr::new(AggregateFunction::StddevSamp, Expr::column("v"), "stdev_v"),
            ],
        ),
        Arc::clone(&aggregate_schema),
    );
    let plan = builder.finish_with_host_sink(aggregate);

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.compiled_pipeline_attempts, 1);
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.schema(), aggregate_schema);
    assert_eq!(batch.num_rows(), 0);

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

fn write_f64_parquet(runtime: &cudf::Runtime, schema: &Arc<Schema>, path: &Path, values: Vec<Option<f64>>) {
    let batch = RecordBatch::try_new(Arc::clone(schema), vec![Arc::new(Float64Array::from(values))]).unwrap();
    DataFrame::from_record_batch(runtime, &batch).unwrap().write_parquet(path).unwrap();
}

fn write_f64_keep_row_grouped_parquet(
    schema: Arc<Schema>,
    path: &Path,
    rows: Vec<(Option<f64>, i32)>,
    rows_per_group: usize,
) {
    let values = rows.iter().map(|(value, _keep)| *value).collect::<Vec<_>>();
    let keep = rows.iter().map(|(_value, keep)| *keep).collect::<Vec<_>>();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Float64Array::from(values)), Arc::new(Int32Array::from(keep))],
    )
    .unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(rows_per_group)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
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

fn ordered_avg(values: impl IntoIterator<Item = f64>) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0u64;
    for value in values {
        sum += value;
        count += 1;
    }
    (count > 0).then_some(sum / count as f64)
}

#[test]
fn grouped_stddev_samp_can_use_gpu_native_tolerant_contract() {
    let dir = unique_temp_dir("query-engine-gpu-native-stddev-samp");
    let path = dir.join("input.parquet");
    let context = backend_context_with_statistical_aggregate_mode(StatisticalAggregateExecutionMode::GpuNativeTolerant);
    let runtime = Arc::new(context.runtime().unwrap());
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

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn grouped_tolerant_statistical_aggregate_preserves_decimal_avg_scale() {
    let context = backend_context_with_statistical_aggregate_mode(StatisticalAggregateExecutionMode::GpuNativeTolerant);
    let runtime = Arc::new(context.runtime().unwrap());
    let dir = unique_temp_dir("query-engine-gpu-native-statistical-decimal-avg");
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
    let dir = unique_temp_dir("query-engine-gpu-native-statistical-aggregates");
    let path = dir.join("input.parquet");
    let context = backend_context_with_statistical_aggregate_mode(StatisticalAggregateExecutionMode::GpuNativeTolerant);
    let runtime = Arc::new(context.runtime().unwrap());
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
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_tolerant_grouped_statistical_aggregate_streams_partial_moments() {
    let context = backend_context_with_memory_control_policy_and_statistical_aggregate_mode(
        MemoryControlPolicy::Bounded {
            device_budget_bytes: 128 * 1024 * 1024,
            max_source_chunk_bytes: 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        },
        StatisticalAggregateExecutionMode::GpuNativeTolerant,
    );
    let runtime = Arc::new(context.runtime().unwrap());
    let dir = unique_temp_dir("query-engine-bounded-tolerant-statistical-moments");
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
                query_engine::expr::ColumnRef::input(0, "k", "k"),
                query_engine::expr::ColumnRef::input(1, "k", "k"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::right(query_engine::expr::ColumnRef::input(1, "grp", "grp"), "grp"),
            JoinOutputColumn::left(query_engine::expr::ColumnRef::input(0, "v", "v"), "v"),
            JoinOutputColumn::left(query_engine::expr::ColumnRef::input(0, "d", "d"), "d"),
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

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_tolerant_filtered_grouped_statistical_aggregate_uses_filter_preparation() {
    let context = backend_context_with_memory_control_policy_and_statistical_aggregate_mode(
        MemoryControlPolicy::Bounded {
            device_budget_bytes: 128 * 1024 * 1024,
            max_source_chunk_bytes: 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        },
        StatisticalAggregateExecutionMode::GpuNativeTolerant,
    );
    let runtime = Arc::new(context.runtime().unwrap());
    let dir = unique_temp_dir("query-engine-bounded-filtered-statistical-moments");
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
                query_engine::expr::ColumnRef::input(0, "k", "k"),
                query_engine::expr::ColumnRef::input(1, "k", "k"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::right(query_engine::expr::ColumnRef::input(1, "grp", "grp"), "grp"),
            JoinOutputColumn::left(query_engine::expr::ColumnRef::input(0, "v", "v"), "v"),
            JoinOutputColumn::left(query_engine::expr::ColumnRef::input(0, "keep", "keep"), "keep"),
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

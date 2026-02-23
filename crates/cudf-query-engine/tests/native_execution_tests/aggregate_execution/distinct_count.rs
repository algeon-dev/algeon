use super::*;

#[test]
fn bounded_global_count_distinct_counts_cross_chunk_duplicates_once() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-global-count-distinct");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_distinct_count_input_parquet(&path);

    let aggregated_schema = Arc::new(Schema::new(vec![Field::new("distinct_values", DataType::Int64, false)]));
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(&path)
                .with_file_size_bytes(fs::metadata(&path).unwrap().len())
                .with_row_groups(vec![0, 1, 2]),
        ],
        Arc::clone(&input_schema),
        vec!["value".to_string()],
    );
    let aggregate = AggregateSpec::new(
        Vec::new(),
        vec![AggregateExpr::new(AggregateFunction::NUnique, Expr::column("value"), "distinct_values")],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, Arc::clone(&aggregated_schema));
    let plan = builder.finish_with_host_sink(aggregate);

    let context = distinct_count_context(runtime);
    let result = execute_query_plan(&context, &plan).unwrap();
    assert_eq!(result.metrics().source_chunks_planned, 3);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(int64_array(batch, 0).value(0), 3);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_global_nunique_casts_to_declared_count_dtype() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-global-nunique-int32");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_distinct_count_input_parquet(&path);

    let aggregated_schema = Arc::new(Schema::new(vec![Field::new("distinct_values", DataType::Int32, false)]));
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(&path)
                .with_file_size_bytes(fs::metadata(&path).unwrap().len())
                .with_row_groups(vec![0, 1, 2]),
        ],
        Arc::clone(&input_schema),
        vec!["value".to_string()],
    );
    let aggregate = AggregateSpec::new(
        Vec::new(),
        vec![AggregateExpr::new(AggregateFunction::NUnique, Expr::column("value"), "distinct_values")],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, Arc::clone(&aggregated_schema));
    let plan = builder.finish_with_host_sink(aggregate);

    let context = distinct_count_context(runtime);
    let result = execute_query_plan(&context, &plan).unwrap();
    assert_eq!(result.metrics().source_chunks_planned, 3);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.schema().field(0).data_type(), &DataType::Int32);
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(int32_array(batch, 0).value(0), 3);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_grouped_count_distinct_ignores_null_values_and_keeps_null_groups() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-grouped-count-distinct");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_distinct_count_input_parquet(&path);

    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, true),
        Field::new("distinct_values", DataType::Int64, false),
    ]));
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(&path)
                .with_file_size_bytes(fs::metadata(&path).unwrap().len())
                .with_row_groups(vec![0, 1, 2]),
        ],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "value".to_string()],
    );
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![AggregateExpr::new(AggregateFunction::NUnique, Expr::column("value"), "distinct_values")],
    );
    let sort = SortSpec::new(vec![SortExpr::new(Expr::column("key"), false, false)]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, Arc::clone(&aggregated_schema));
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&aggregated_schema));
    let plan = builder.finish_with_host_sink(sort);

    let context = distinct_count_context(runtime);
    let result = execute_query_plan(&context, &plan).unwrap();
    assert_eq!(result.metrics().source_chunks_planned, 3);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 4);
    let rows = (0..batch.num_rows())
        .map(|row| {
            let key = if int32_array(batch, 0).is_null(row) { None } else { Some(int32_array(batch, 0).value(row)) };
            (key, int64_array(batch, 1).value(row))
        })
        .collect::<Vec<_>>();
    assert_eq!(rows, vec![(Some(1), 1), (Some(2), 2), (Some(3), 0), (None, 1)]);

    fs::remove_dir_all(dir).unwrap();
}

/// Regression test: per-chunk NUnique values must NOT be Sum-merged when a
/// grouped COUNT(DISTINCT) is combined with other aggregates and the source
/// spans multiple Parquet files.  This is the q10/q23 bug pattern.
///
/// File 1 data:  (key=1, value=10), (key=1, value=20), (key=2, value=30)
/// File 2 data:  (key=1, value=10), (key=1, value=40), (key=2, value=30), (key=2, value=50)
///
/// Correct result (NUnique across all data):
///   key=1: COUNT(DISTINCT value)=3 (10, 20, 40), COUNT(key)=4 (2 rows + 2 rows)
///   key=2: COUNT(DISTINCT value)=2 (30, 50),     COUNT(key)=3 (1 row + 2 rows)
///
/// Old buggy result (per-chunk NUnique then Sum):
///   key=1: COUNT(DISTINCT value)=2+2=4 (WRONG), COUNT(key)=4
///   key=2: COUNT(DISTINCT value)=1+2=3 (WRONG), COUNT(key)=3
#[test]
fn bounded_grouped_count_distinct_with_count_star_across_multiple_files_is_correct() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-multifile-distinct");
    let file1 = dir.join("part1.parquet");
    let file2 = dir.join("part2.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_multi_file_distinct_count_files(&file1, &file2);

    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("distinct_values", DataType::Int64, false),
        Field::new("total_count", DataType::Int64, false),
    ]));
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(&file1).with_file_size_bytes(fs::metadata(&file1).unwrap().len()),
            LocalParquetFile::new(&file2).with_file_size_bytes(fs::metadata(&file2).unwrap().len()),
        ],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "value".to_string()],
    );
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![
            AggregateExpr::new(AggregateFunction::NUnique, Expr::column("value"), "distinct_values"),
            AggregateExpr::new(AggregateFunction::Count, Expr::column("key"), "total_count"),
        ],
    );
    let sort = SortSpec::new(vec![SortExpr::new(Expr::column("key"), false, false)]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, Arc::clone(&aggregated_schema));
    let sort_node = builder.add_sort(aggregate, sort, Arc::clone(&aggregated_schema));
    let plan = builder.finish_with_host_sink(sort_node);

    let context = distinct_count_context(runtime);
    let result = execute_query_plan(&context, &plan).unwrap();
    assert!(
        result.metrics().source_chunks_planned >= 2,
        "test requires multiple source chunks to exercise the multi-file distinct path; got {}",
        result.metrics().source_chunks_planned
    );

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 2, "expected 2 groups");
    let rows = (0..batch.num_rows())
        .map(|row| {
            (int32_array(batch, 0).value(row), int64_array(batch, 1).value(row), int64_array(batch, 2).value(row))
        })
        .collect::<Vec<_>>();
    assert_eq!(
        rows,
        vec![
            (1, 3, 4), // key=1: distinct values {10,20,40}=3, COUNT(key)=2+2=4
            (2, 2, 3), // key=2: distinct values {30,50}=2,   COUNT(key)=1+2=3
        ],
        "COUNT(DISTINCT) must count unique values across all source chunks, not sum per-chunk counts"
    );

    fs::remove_dir_all(dir).unwrap();
}

/// Same as above but with COUNT(DISTINCT) as the only aggregate, verifying the
/// fix does not regress the sole-distinct-aggregate case.
#[test]
fn bounded_grouped_count_distinct_only_across_multiple_files_is_correct() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-multifile-distinct-only");
    let file1 = dir.join("part1.parquet");
    let file2 = dir.join("part2.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_multi_file_distinct_count_files(&file1, &file2);

    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("distinct_values", DataType::Int64, false),
    ]));
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(&file1).with_file_size_bytes(fs::metadata(&file1).unwrap().len()),
            LocalParquetFile::new(&file2).with_file_size_bytes(fs::metadata(&file2).unwrap().len()),
        ],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "value".to_string()],
    );
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![AggregateExpr::new(AggregateFunction::NUnique, Expr::column("value"), "distinct_values")],
    );
    let sort = SortSpec::new(vec![SortExpr::new(Expr::column("key"), false, false)]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, Arc::clone(&aggregated_schema));
    let sort_node = builder.add_sort(aggregate, sort, Arc::clone(&aggregated_schema));
    let plan = builder.finish_with_host_sink(sort_node);

    let context = distinct_count_context(runtime);
    let result = execute_query_plan(&context, &plan).unwrap();
    assert!(
        result.metrics().source_chunks_planned >= 2,
        "test requires multiple source chunks to exercise the multi-file distinct path; got {}",
        result.metrics().source_chunks_planned
    );

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 2, "expected 2 groups");
    let rows = (0..batch.num_rows())
        .map(|row| (int32_array(batch, 0).value(row), int64_array(batch, 1).value(row)))
        .collect::<Vec<_>>();
    assert_eq!(
        rows,
        vec![
            (1, 3), // key=1: distinct values {10,20,40}=3
            (2, 2), // key=2: distinct values {30,50}=2
        ],
        "sole COUNT(DISTINCT) must also be correct across multiple source chunks"
    );

    fs::remove_dir_all(dir).unwrap();
}

/// Write two Parquet files with intentional cross-file duplicate (key, value) pairs
/// so that a naive per-chunk NUnique + Sum would overcount.
///
/// File 1:  (1,10), (1,20), (2,30)
/// File 2:  (1,10), (1,40), (2,30), (2,50)
///
/// Correct grouped COUNT(DISTINCT value): key=1 → 3, key=2 → 2
/// Correct grouped COUNT(key):            key=1 → 4, key=2 → 3
fn write_multi_file_distinct_count_files(file1: &std::path::Path, file2: &std::path::Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("value", DataType::Int64, false),
    ]));
    let write_batch = |path: &std::path::Path, keys: Vec<Option<i32>>, values: Vec<Option<i64>>| {
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from(keys)), Arc::new(Int64Array::from(values))],
        )
        .unwrap();
        let file = File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    };
    write_batch(file1, vec![Some(1), Some(1), Some(2)], vec![Some(10), Some(20), Some(30)]);
    write_batch(file2, vec![Some(1), Some(1), Some(2), Some(2)], vec![Some(10), Some(40), Some(30), Some(50)]);
    schema
}

fn distinct_count_context(runtime: Arc<cudf::Runtime>) -> ExecutionContext {
    ExecutionContext::builder()
        .with_shared_runtime(runtime)
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap()
}

fn write_distinct_count_input_parquet(path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, true),
        Field::new("value", DataType::Int64, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![
                Some(1),
                Some(1),
                Some(1),
                Some(2),
                Some(2),
                Some(2),
                Some(3),
                Some(3),
                None,
                None,
            ])),
            Arc::new(Int64Array::from(vec![
                Some(10),
                Some(10),
                None,
                Some(10),
                Some(20),
                Some(20),
                None,
                None,
                Some(30),
                Some(30),
            ])),
        ],
    )
    .unwrap();
    let file = File::create(path).unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(4)).build();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    schema
}

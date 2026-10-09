use std::{fs::File, path::Path, sync::Arc};

use query_engine::{dimension_pruning::local_dimension_pruning_resolver, rapids_backend::rapids_analysis_context};
use query_planning::optimizer::optimize_query_plan;

use super::*;

#[test]
fn bounded_global_nunique_casts_to_declared_count_dtype() {
    let dir = unique_temp_dir("query-engine-bounded-global-nunique-int32");
    let path = dir.join("input.parquet");
    let context = backend_context();
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
    let aggregate = builder.add_aggregate(source, aggregate, Arc::clone(&aggregated_schema)).unwrap();
    let plan = builder.finish_with_host_sink(aggregate).unwrap();

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
fn bounded_global_count_star_and_nunique_match_controls_on_zero_row_component() {
    assert_bounded_global_count_and_nunique_matches_controls("zero-row", &[&[]], (0, 0), 0);
}

#[test]
fn bounded_global_count_star_and_nunique_match_controls_on_single_component() {
    assert_bounded_global_count_and_nunique_matches_controls("single", &[&[10, 20, 10, 30]], (4, 3), 1);
}

#[test]
fn bounded_global_count_star_and_nunique_match_controls_across_components() {
    assert_bounded_global_count_and_nunique_matches_controls("multi", &[&[10, 20, 30], &[10, 40, 30, 50]], (7, 5), 2);
}

fn assert_bounded_global_count_and_nunique_matches_controls(
    fixture: &str,
    parts: &[&[i64]],
    expected: (i64, i64),
    minimum_chunks: u64,
) {
    let dir = unique_temp_dir(&format!("query-engine-bounded-global-count-nunique-{fixture}"));
    let input_schema = Arc::new(Schema::new(vec![Field::new("label", DataType::Int64, false)]));
    let files = parts
        .iter()
        .enumerate()
        .map(|(index, values)| {
            let path = dir.join(format!("part-{index}.parquet"));
            let batch =
                RecordBatch::try_new(Arc::clone(&input_schema), vec![Arc::new(Int64Array::from(values.to_vec()))])
                    .unwrap();
            write_record_batch_with_row_group_rows(&path, Arc::clone(&input_schema), batch, values.len().max(1));
            LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(path).unwrap().len())
        })
        .collect::<Vec<_>>();
    let source = LocalParquetSource::from_files(files, Arc::clone(&input_schema), vec!["label".to_owned()]);
    let build_plan = |expressions: Vec<AggregateExpr>| {
        let output_schema = Arc::new(Schema::new(
            expressions.iter().map(|expr| Field::new(&expr.alias, DataType::Int64, false)).collect::<Vec<_>>(),
        ));
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(source.clone()));
        let aggregate =
            builder.add_aggregate(source, AggregateSpec::new(Vec::new(), expressions), output_schema).unwrap();
        builder.finish_with_host_sink(aggregate).unwrap()
    };
    let count = AggregateExpr::new(AggregateFunction::Count, Expr::literal(LiteralValue::Int64(1)), "row_count");
    let distinct = AggregateExpr::new(AggregateFunction::NUnique, Expr::column("label"), "distinct_labels");
    let context = backend_context();
    let count_result = execute_query_plan(&context, &build_plan(vec![count.clone()])).unwrap();
    let distinct_result = execute_query_plan(&context, &build_plan(vec![distinct.clone()])).unwrap();
    let mixed_result = execute_query_plan(&context, &build_plan(vec![count, distinct])).unwrap();
    assert!(mixed_result.metrics().source_chunks_planned >= minimum_chunks);

    let QueryOutput::HostArrow(count_batch) = count_result.output() else {
        panic!("COUNT(*) control should materialize one RecordBatch");
    };
    let QueryOutput::HostArrow(distinct_batch) = distinct_result.output() else {
        panic!("NUnique control should materialize one RecordBatch");
    };
    let QueryOutput::HostArrow(mixed_batch) = mixed_result.output() else {
        panic!("mixed aggregate should materialize one RecordBatch");
    };
    assert_eq!(mixed_batch.num_rows(), 1);
    assert_eq!(int64_array(count_batch, 0).value(0), expected.0);
    assert_eq!(int64_array(distinct_batch, 0).value(0), expected.1);
    assert_eq!(int64_array(mixed_batch, 0).value(0), int64_array(count_batch, 0).value(0));
    assert_eq!(int64_array(mixed_batch, 1).value(0), int64_array(distinct_batch, 0).value(0));
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_grouped_count_distinct_ignores_null_values_and_keeps_null_groups() {
    let dir = unique_temp_dir("query-engine-bounded-grouped-count-distinct");
    let path = dir.join("input.parquet");
    let context = backend_context();
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
    let aggregate = builder.add_aggregate(source, aggregate, Arc::clone(&aggregated_schema)).unwrap();
    let sort = builder.add_sort(aggregate, sort).unwrap();
    let plan = builder.finish_with_host_sink(sort).unwrap();

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

/// This regression test covers per-chunk NUnique values, which must NOT be
/// Sum-merged when a grouped COUNT(DISTINCT) is combined with other aggregates
/// and the source spans multiple Parquet files.  This is the q10/q23 bug pattern.
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
    let dir = unique_temp_dir("query-engine-bounded-multifile-distinct");
    let file1 = dir.join("part1.parquet");
    let file2 = dir.join("part2.parquet");
    let context = backend_context();
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
    let aggregate = builder.add_aggregate(source, aggregate, Arc::clone(&aggregated_schema)).unwrap();
    let sort_node = builder.add_sort(aggregate, sort).unwrap();
    let plan = builder.finish_with_host_sink(sort_node).unwrap();

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

/// Same as above but with COUNT(DISTINCT) as the only aggregate, which guards
/// the sole-distinct-aggregate case against regressions.
#[test]
fn bounded_grouped_count_distinct_only_across_multiple_files_is_correct() {
    let dir = unique_temp_dir("query-engine-bounded-multifile-distinct-only");
    let file1 = dir.join("part1.parquet");
    let file2 = dir.join("part2.parquet");
    let context = backend_context();
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
    let aggregate = builder.add_aggregate(source, aggregate, Arc::clone(&aggregated_schema)).unwrap();
    let sort_node = builder.add_sort(aggregate, sort).unwrap();
    let plan = builder.finish_with_host_sink(sort_node).unwrap();

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

/// An aggregate holding an exact distinct count reads its whole input and
/// concatenates it before aggregating. After the single-distinct split both
/// aggregates merge chunk partials, so the same query fits a grant that holds
/// the input once but not beside its concatenation.
#[test]
fn single_distinct_split_streams_grouped_count_distinct_beside_sum_and_count_within_a_small_grant() {
    let dir = unique_temp_dir("query-engine-single-distinct-split-grant");
    let path = dir.join("input.parquet");
    let row_groups = 64;
    let rows_per_row_group = 65_536;
    let rows = row_groups * rows_per_row_group;
    let input_schema = write_single_distinct_split_input_parquet(&path, rows, rows_per_row_group);
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(&path)
                .with_file_size_bytes(fs::metadata(&path).unwrap().len())
                .with_row_groups((0..row_groups).collect()),
        ],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "x".to_string(), "y".to_string()],
    );
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![
            AggregateExpr::new(AggregateFunction::NUnique, Expr::column("x"), "distinct_x"),
            AggregateExpr::new(AggregateFunction::Sum, Expr::column("y"), "sum_y"),
            AggregateExpr::new(AggregateFunction::Count, Expr::column("y"), "count_y"),
        ],
    );
    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("distinct_x", DataType::Int64, false),
        Field::new("sum_y", DataType::Int64, true),
        Field::new("count_y", DataType::Int64, false),
    ]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, aggregated_schema).unwrap();
    let sort =
        builder.add_sort(aggregate, SortSpec::new(vec![SortExpr::new(Expr::column("key"), false, false)])).unwrap();
    let plan = builder.finish_with_host_sink(sort).unwrap();
    let optimized = optimize_query_plan(
        plan.clone(),
        &Default::default(),
        rapids_analysis_context(PlanningPolicy::default()),
        local_dimension_pruning_resolver(),
    )
    .unwrap();
    assert_eq!(optimized.diagnostics.single_distinct_aggregate_split.applied_count, 1);

    // The 80 MiB input fits the grant once; its concatenation does not.
    let cap = 128 * 1024 * 1024;
    let context = backend_context_with_memory_control_policy(MemoryControlPolicy {
        device_budget_bytes: cap,
        max_source_chunk_bytes: 8 * 1024 * 1024,
        max_row_groups_per_chunk: 1,
        max_retry_attempts: 1,
        min_retry_chunk_row_groups: 1,
    });

    let error = execute_query_plan(&context, &plan).err().expect("the whole-input concatenation exceeds the grant");
    assert_eq!(error.code(), QueryErrorCode::QueryMemoryLimitExceeded);
    assert_eq!(error.operation(), "query_engine.bounded.union.concat");

    let result = execute_query_plan(&context, &optimized.plan).unwrap();
    assert_eq!(result.metrics().source_chunks_planned, row_groups as u64);
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    let mut expected = (0..16).map(|key| (key, 4, 0, (rows / 16) as i64)).collect::<Vec<_>>();
    for row in 0..rows {
        expected[row % 16].2 += (row % 1000) as i64;
    }
    let actual = (0..batch.num_rows())
        .map(|row| {
            (
                int32_array(batch, 0).value(row),
                int64_array(batch, 1).value(row),
                int64_array(batch, 2).value(row),
                int64_array(batch, 3).value(row),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);

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

/// `key = i % 16`, `x = i % 64` and `y = i % 1000`, so each key has four
/// distinct `x` values repeated across every row group.
fn write_single_distinct_split_input_parquet(path: &Path, rows: usize, rows_per_row_group: usize) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("x", DataType::Int64, false),
        Field::new("y", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from_iter_values((0..rows).map(|row| (row % 16) as i32))),
            Arc::new(Int64Array::from_iter_values((0..rows).map(|row| (row % 64) as i64))),
            Arc::new(Int64Array::from_iter_values((0..rows).map(|row| (row % 1000) as i64))),
        ],
    )
    .unwrap();
    write_record_batch_with_row_group_rows(path, Arc::clone(&schema), batch, rows_per_row_group);
    schema
}

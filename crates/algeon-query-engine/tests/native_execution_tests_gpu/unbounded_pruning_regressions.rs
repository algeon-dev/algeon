use std::{fs::File, path::Path, sync::Arc};

use super::*;

fn admitted_execution_context() -> BackendQueryContext {
    backend_context()
}

#[test]
fn unbounded_single_stage_grouped_median_preserves_pruning_regression() {
    let dir = unique_temp_dir("query-engine-unbounded-median-pruning");
    let path = dir.join("input.parquet");
    let schema = write_median_pruning_input(&path);
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&schema),
        vec!["key".to_string(), "payload".to_string(), "m00".to_string()],
    );
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("median_m00", DataType::Float64, true),
    ]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let filtered = builder
        .add_filter(
            source,
            Expr::binary(Expr::column("payload"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int32(10))),
        )
        .unwrap();
    let aggregate = builder
        .add_aggregate(
            filtered,
            AggregateSpec::new(
                vec![Expr::column("key")],
                vec![AggregateExpr::new(AggregateFunction::Median, Expr::column("m00"), "median_m00")],
            ),
            Arc::clone(&aggregate_schema),
        )
        .unwrap();
    let sorted = builder
        .add_sort(
            aggregate,
            SortSpec::new(vec![
                SortExpr::new(Expr::column("median_m00"), true, false),
                SortExpr::new(Expr::column("key"), false, false),
            ]),
        )
        .unwrap();
    let plan = builder.finish_with_host_sink(sorted).unwrap();

    let context = admitted_execution_context();
    let result = execute_query_plan(&context, &plan).unwrap();
    assert_eq!(result.metrics().partial_aggregate_count, 0, "median must remain single-stage");
    let QueryOutput::HostArrow(batch) = result.output() else { panic!("median must materialize to Arrow") };
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[2, 1]);
    assert_eq!(float64_array(batch, 1).values().as_ref(), &[70.0, 60.0]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn unbounded_physically_empty_parquet_count_and_count_star_return_zero() {
    let dir = unique_temp_dir("query-engine-unbounded-physical-empty-parquet");
    let zero_row_file = dir.join("zero-rows.parquet");
    let zero_row_group_file = dir.join("zero-row-group.parquet");
    let schema = empty_parquet_schema();
    write_physically_empty_parquet(&zero_row_file, Arc::clone(&schema));
    write_zero_row_group_parquet(&zero_row_group_file, Arc::clone(&schema));

    let context = admitted_execution_context();
    for path in [&zero_row_file, &zero_row_group_file] {
        let file = LocalParquetFile::new(path).with_file_size_bytes(fs::metadata(path).unwrap().len());

        let source = LocalParquetSource::from_files(vec![file.clone()], Arc::clone(&schema), vec!["key".to_string()]);
        let count_schema = Arc::new(Schema::new(vec![Field::new("count_key", DataType::Int64, false)]));
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(source));
        let aggregate = builder
            .add_aggregate(
                source,
                AggregateSpec::new(
                    Vec::new(),
                    vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("key"), "count_key")],
                ),
                count_schema,
            )
            .unwrap();
        let result = execute_query_plan(&context, &builder.finish_with_host_sink(aggregate).unwrap()).unwrap();
        let QueryOutput::HostArrow(batch) = result.output() else { panic!("COUNT must materialize to Arrow") };
        assert_eq!(batch.num_rows(), 1);
        assert_eq!(int64_array(batch, 0).value(0), 0);

        let probe = LocalParquetSource::count_star_probe(vec![file], Arc::clone(&schema), vec!["key".to_string()])
            .with_column_roles(Vec::new(), vec!["key".to_string()]);
        let count_star_schema = Arc::new(Schema::new(vec![Field::new("count_rows", DataType::Int64, true)]));
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(probe));
        let aggregate = builder
            .add_aggregate(
                source,
                AggregateSpec::new(
                    Vec::new(),
                    vec![AggregateExpr::new(
                        AggregateFunction::Sum,
                        Expr::column(ZERO_COLUMN_COUNT_COLUMN),
                        "count_rows",
                    )],
                ),
                count_star_schema,
            )
            .unwrap();
        let result = execute_query_plan(&context, &builder.finish_with_host_sink(aggregate).unwrap()).unwrap();
        let QueryOutput::HostArrow(batch) = result.output() else { panic!("COUNT(*) must materialize to Arrow") };
        assert_eq!(batch.num_rows(), 1);
        assert_eq!(int64_array(batch, 0).value(0), 0);
    }

    fs::remove_dir_all(dir).unwrap();
}

fn write_median_pruning_input(path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int32, false),
        Field::new("m00", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 1, 2, 1, 2, 1, 2])),
            Arc::new(Int32Array::from(vec![0, 1, 2, 3, 10, 11, 20, 21])),
            Arc::new(Int64Array::from(vec![10, 20, 30, 40, 50, 60, 70, 80])),
        ],
    )
    .unwrap();
    let file = File::create(path).unwrap();
    let properties = WriterProperties::builder().set_max_row_group_row_count(Some(2)).build();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(properties)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    schema
}

fn empty_parquet_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int32, false),
        Field::new("m00", DataType::Int64, false),
        Field::new("m01", DataType::Int64, false),
    ]))
}

fn write_physically_empty_parquet(path: &Path, schema: Arc<Schema>) {
    let file = File::create(path).unwrap();
    let writer = ArrowWriter::try_new(file, schema, Some(WriterProperties::builder().build())).unwrap();
    writer.close().unwrap();
}

fn write_zero_row_group_parquet(path: &Path, schema: Arc<Schema>) {
    use parquet::{
        arrow::ArrowSchemaConverter,
        data_type::{Int32Type, Int64Type},
        file::writer::SerializedFileWriter,
    };

    let file = File::create(path).unwrap();
    let properties = Arc::new(WriterProperties::builder().build());
    let parquet_schema = ArrowSchemaConverter::new().convert(&schema).unwrap();
    let mut writer =
        SerializedFileWriter::new(file, parquet_schema.root_schema_ptr(), Arc::clone(&properties)).unwrap();
    let mut row_group = writer.next_row_group().unwrap();
    for column_index in 0..4 {
        let mut column = row_group.next_column().unwrap().expect("zero-row group must contain every schema column");
        match column_index {
            0 | 1 => column.typed::<Int32Type>().write_batch(&[], None, None).unwrap(),
            2 | 3 => column.typed::<Int64Type>().write_batch(&[], None, None).unwrap(),
            _ => unreachable!(),
        };
        column.close().unwrap();
    }
    row_group.close().unwrap();
    writer.close().unwrap();
}

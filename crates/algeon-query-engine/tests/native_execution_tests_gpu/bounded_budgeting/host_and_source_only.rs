use std::sync::Arc;

use query_planning::plan::NodeId;

use super::*;

#[test]
fn bounded_native_query_executes_non_aggregate_source_plan() {
    let dir = unique_temp_dir("query-engine-bounded-source-only");
    let path = dir.join("input.parquet");
    let schema = write_row_grouped_input_parquet(&path);
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&schema),
        vec!["key".into(), "payload".into(), "m00".into(), "m01".into()],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let plan = builder.finish_with_host_sink(source).unwrap();
    let context = backend_context_with_grant(1024 * 1024);

    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should return a RecordBatch");
    };
    assert_eq!(batch.num_rows(), 6);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2, 1, 3, 2, 3]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[10, 10, 20, 20, 30, 5]);
    assert_eq!(result.metrics().parquet_files, 1);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_preserves_multi_file_source_order() {
    let dir = unique_temp_dir("query-engine-bounded-multi-file-source-order");
    let first_path = dir.join("part-000.parquet");
    let second_path = dir.join("part-001.parquet");
    let third_path = dir.join("part-002.parquet");
    let schema = write_i32_parquet_with_row_group_rows(&first_path, &[1, 2], 1);
    write_i32_parquet_with_row_group_rows(&second_path, &[10, 11], 1);
    write_i32_parquet_with_row_group_rows(&third_path, &[20], 1);
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(&second_path)
                .with_file_size_bytes(fs::metadata(&second_path).unwrap().len())
                .with_row_group_rows(vec![1, 1]),
            LocalParquetFile::new(&first_path)
                .with_file_size_bytes(fs::metadata(&first_path).unwrap().len())
                .with_row_group_rows(vec![1, 1]),
            LocalParquetFile::new(&third_path)
                .with_file_size_bytes(fs::metadata(&third_path).unwrap().len())
                .with_row_group_rows(vec![1]),
        ],
        Arc::clone(&schema),
        vec!["id".to_string()],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let plan = builder.finish_with_host_sink(source).unwrap();
    let context = backend_context_with_grant(1024 * 1024);

    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should return a RecordBatch");
    };
    assert_eq!(batch.num_rows(), 5);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[10, 11, 1, 2, 20]);
    assert_eq!(result.metrics().parquet_files, 3);
    assert_eq!(result.metrics().source_chunks_committed, 5);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_executes_union_plan_without_legacy_aggregate_path() {
    let context = backend_context_with_grant(1024 * 1024);
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
    let left_batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![1, 2]))]).unwrap();
    let right_batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![3, 4]))]).unwrap();

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("left", Arc::clone(&schema))));
    let right = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("right", Arc::clone(&schema))));
    let union = builder.add_union(vec![left, right], Arc::clone(&schema)).unwrap();
    let plan = builder.finish_with_host_sink(union).unwrap();
    let admitted = context.admit_plan(&plan).unwrap();
    let runtime = Arc::new(admitted.runtime().unwrap());
    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "left".to_string(),
        GpuResultHandle::new(
            &admitted,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &left_batch).unwrap()),
            Arc::clone(&schema),
        )
        .unwrap(),
    );
    gpu_inputs.insert(
        "right".to_string(),
        GpuResultHandle::new(
            &admitted,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &right_batch).unwrap()),
            Arc::clone(&schema),
        )
        .unwrap(),
    );

    let result = execute_query_plan_with_gpu_inputs(&admitted, &plan, &gpu_inputs).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should return a RecordBatch");
    };
    assert_eq!(batch.num_rows(), 4);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2, 3, 4]);
    admitted.finish();
}

const SELECTIVE_SCAN_ROWS: i64 = 1_000_000;

/// Writes `SELECTIVE_SCAN_ROWS` rows in 125k-row groups: row `i` has
/// `a = rows - 1 - i`, `b = i`, `grp = i % 3`, and `keep = 1` on every
/// hundredth row. The decoded scan is about 53 MiB.
fn write_selective_scan_parquet(path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int64, false),
        Field::new("b", DataType::Int64, false),
        Field::new("grp", DataType::Int64, false),
        Field::new("keep", DataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from_iter_values((0..SELECTIVE_SCAN_ROWS).rev())),
            Arc::new(Int64Array::from_iter_values(0..SELECTIVE_SCAN_ROWS)),
            Arc::new(Int64Array::from_iter_values((0..SELECTIVE_SCAN_ROWS).map(|row| row % 3))),
            Arc::new(Int32Array::from_iter_values((0..SELECTIVE_SCAN_ROWS).map(|row| i32::from(row % 100 == 0)))),
        ],
    )
    .unwrap();
    write_record_batch_with_row_group_rows(path, Arc::clone(&schema), batch, 125_000);
    schema
}

fn int64_column<'a>(batch: &'a RecordBatch, name: &str) -> &'a Int64Array {
    batch.column(batch.schema().index_of(name).unwrap()).as_any().downcast_ref::<Int64Array>().unwrap()
}

#[test]
fn bounded_native_query_filters_source_chunks_before_frame_only_consumers() {
    let dir = unique_temp_dir("query-engine-bounded-selective-scan-frame-consumers");
    let path = dir.join("input.parquet");
    let schema = write_selective_scan_parquet(&path);
    let read_columns = schema.fields().iter().map(|field| field.name().clone()).collect::<Vec<_>>();
    let plan = |consumer: &dyn Fn(&mut QueryPlanBuilder, NodeId) -> NodeId| {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_path(
            &path,
            Arc::clone(&schema),
            read_columns.clone(),
        )));
        let filter = builder
            .add_filter(source, Expr::binary(Expr::column("keep"), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(1))))
            .unwrap();
        let root = consumer(&mut builder, filter);
        builder.finish_with_host_sink(root).unwrap()
    };
    let sink = plan(&|_, filter| filter);
    let sort = plan(&|builder, filter| {
        builder.add_sort(filter, SortSpec::new(vec![SortExpr::new(Expr::column("a"), false, false)])).unwrap()
    });
    let window = plan(&|builder, filter| {
        let total = Field::new("grp_total", DataType::Int64, true);
        let function = WindowFunction::aggregate(AggregateFunction::Sum, Expr::column("b"), total);
        builder.add_window(filter, WindowSpec::new(vec![Expr::column("grp")], Vec::new(), vec![function])).unwrap()
    });
    let kept = (0..SELECTIVE_SCAN_ROWS).step_by(100).collect::<Vec<_>>();
    let mut grp_totals = [0_i64; 3];
    for row in &kept {
        grp_totals[(row % 3) as usize] += row;
    }

    // The grant holds the kept rows but not the decoded scan together with its
    // concatenation, so each consumer must receive only per-chunk filter outputs.
    let grant = 32 * 1024 * 1024;
    let run = |label: &str, plan: &QueryPlan| {
        let result = execute_query_plan(&backend_context_with_grant(grant), plan)
            .unwrap_or_else(|error| panic!("{label}: the filtered scan fits the grant: {error}"));
        let QueryOutput::HostArrow(batch) = result.output() else {
            panic!("{label}: host sink should return a RecordBatch");
        };
        batch.clone()
    };

    let batch = run("sink", &sink);
    assert_eq!(int64_column(&batch, "b").values().as_ref(), kept.as_slice(), "chunk outputs keep file order");
    let batch = run("sort", &sort);
    let ascending_a = kept.iter().rev().map(|row| SELECTIVE_SCAN_ROWS - 1 - row).collect::<Vec<_>>();
    assert_eq!(int64_column(&batch, "a").values().as_ref(), ascending_a.as_slice());
    let batch = run("window", &window);
    assert_eq!(batch.num_rows(), kept.len());
    let grp = int64_column(&batch, "grp");
    let totals = int64_column(&batch, "grp_total");
    assert!((0..batch.num_rows()).all(|row| totals.value(row) == grp_totals[grp.value(row) as usize]));

    fs::remove_dir_all(dir).unwrap();
}

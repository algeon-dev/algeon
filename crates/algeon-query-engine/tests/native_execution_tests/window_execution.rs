use std::sync::Arc;

use arrow::{compute::take, datatypes::UInt64Type};

use super::{
    core_execution::{assert_basic_list_payload_rows, basic_list_payload_batch},
    *,
};

#[test]
fn native_scalar_window_carries_basic_list_payloads_from_gpu_input() {
    let context = backend_context();
    let input = basic_list_payload_batch(&[3, 1, 4, 2]);
    let input_schema = input.schema();
    let mut output_fields = input_schema.fields().iter().map(|field| field.as_ref().clone()).collect::<Vec<_>>();
    output_fields.push(Field::new("rn", DataType::UInt64, false));
    output_fields.push(Field::new("rn_i32", DataType::Int32, false));
    let output_schema = Arc::new(Schema::new(output_fields));
    let mut builder = QueryPlanBuilder::new();
    let source = builder
        .add_source(SourceNode::GpuResident(GpuResidentSource::new("list_window_input", Arc::clone(&input_schema))));
    let sort = builder.add_sort(source, SortSpec::new(vec![SortExpr::new(Expr::column("key"), false, false)])).unwrap();
    let window = builder
        .add_window(
            sort,
            WindowSpec::new(
                Vec::new(),
                vec![SortExpr::new(Expr::column("key"), false, false)],
                vec![WindowFunction::row_number("rn"), WindowFunction::row_number_i32("rn_i32")],
            ),
        )
        .unwrap();
    let plan = builder.finish_with_host_sink(window).unwrap();
    let admitted = context.admit_plan(&plan).unwrap();
    let runtime = Arc::new(admitted.runtime().unwrap());
    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "list_window_input".to_string(),
        GpuResultHandle::new(
            &admitted,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &input).unwrap()),
            Arc::clone(&input_schema),
        )
        .unwrap(),
    );

    let result = execute_query_plan_with_gpu_inputs(&admitted, &plan, &gpu_inputs).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize window list payloads");
    };
    assert_eq!(batch.schema(), output_schema);
    assert_basic_list_payload_rows(batch, 0, 2, &[1, 2, 3, 4]);
    assert_eq!(batch.column(6).as_primitive::<UInt64Type>().values().as_ref(), &[1, 2, 3, 4]);
    assert_eq!(int32_array(batch, 7).values().as_ref(), &[1, 2, 3, 4]);
    admitted.finish();
}

#[test]
fn native_cumulative_aggregate_window_restores_input_order_and_carries_null_rows() {
    let context = backend_context();
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("grp", DataType::Int32, false),
        Field::new("ord", DataType::Int32, false),
        Field::new("value", DataType::Int64, true),
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
    let mut builder = QueryPlanBuilder::new();
    let source =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("window_input", Arc::clone(&input_schema))));
    let window = builder
        .add_window(
            source,
            WindowSpec::new(
                vec![Expr::column("grp")],
                vec![SortExpr::new(Expr::column("ord"), false, false)],
                vec![
                    WindowFunction::cumulative_aggregate(
                        AggregateFunction::Sum,
                        Expr::column("value"),
                        Field::new("running_sum", DataType::Int64, true),
                    ),
                    WindowFunction::cumulative_aggregate(
                        AggregateFunction::Max,
                        Expr::column("value"),
                        Field::new("running_max", DataType::Int64, true),
                    ),
                ],
            ),
        )
        .unwrap();
    let plan = builder.finish_with_host_sink(window).unwrap();
    let admitted = context.admit_plan(&plan).unwrap();
    let runtime = Arc::new(admitted.runtime().unwrap());
    let frame = DataFrame::from_record_batch(&runtime, &batch).unwrap();

    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "window_input".to_string(),
        GpuResultHandle::new(&admitted, GpuDataFrame::new(frame), Arc::clone(&input_schema)).unwrap(),
    );

    let result = execute_query_plan_with_gpu_inputs(&admitted, &plan, &gpu_inputs).unwrap();
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
    admitted.finish();
}

#[test]
fn native_lag_window_shifts_within_partition_and_restores_input_order() {
    let context = backend_context();
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("grp", DataType::Int32, false),
        Field::new("ord", DataType::Int32, false),
        Field::new("value", DataType::Int64, true),
    ]));
    // Input rows are deliberately not sorted by (grp, ord) so that the
    // sort-and-restore step inside execute_lag_lead_window has work to do.
    let batch = RecordBatch::try_new(
        Arc::clone(&input_schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 1, 2, 1, 2, 2])),
            Arc::new(Int32Array::from(vec![2, 1, 2, 3, 1, 3])),
            Arc::new(Int64Array::from(vec![Some(10), Some(20), Some(100), Some(30), Some(200), Some(300)])),
        ],
    )
    .unwrap();
    let mut builder = QueryPlanBuilder::new();
    let source =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("window_input", Arc::clone(&input_schema))));
    let window = builder
        .add_window(
            source,
            WindowSpec::new(
                vec![Expr::column("grp")],
                vec![SortExpr::new(Expr::column("ord"), false, false)],
                vec![WindowFunction::lag(
                    Expr::column("value"),
                    1,
                    None,
                    Field::new("prev_value", DataType::Int64, true),
                )],
            ),
        )
        .unwrap();
    let plan = builder.finish_with_host_sink(window).unwrap();
    let admitted = context.admit_plan(&plan).unwrap();
    let runtime = Arc::new(admitted.runtime().unwrap());
    let frame = DataFrame::from_record_batch(&runtime, &batch).unwrap();

    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "window_input".to_string(),
        GpuResultHandle::new(&admitted, GpuDataFrame::new(frame), Arc::clone(&input_schema)).unwrap(),
    );

    let result = execute_query_plan_with_gpu_inputs(&admitted, &plan, &gpu_inputs).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };

    // Pass-through columns must keep input row order verbatim.
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 1, 2, 1, 2, 2]);
    assert_eq!(int32_array(batch, 1).values().as_ref(), &[2, 1, 2, 3, 1, 3]);
    assert_eq!(
        int64_array(batch, 2).iter().collect::<Vec<_>>(),
        vec![Some(10), Some(20), Some(100), Some(30), Some(200), Some(300)]
    );
    // Within each partition rows are reasoned about in `ord` order:
    //   grp=1: ord=1 (value=20) -> ord=2 (value=10) -> ord=3 (value=30)
    //   grp=2: ord=1 (value=200) -> ord=2 (value=100) -> ord=3 (value=300)
    // So LAG(value, 1) in input row order is:
    //   row 0 (grp=1, ord=2): 20
    //   row 1 (grp=1, ord=1): NULL (first in partition)
    //   row 2 (grp=2, ord=2): 200
    //   row 3 (grp=1, ord=3): 10
    //   row 4 (grp=2, ord=1): NULL (first in partition)
    //   row 5 (grp=2, ord=3): 100
    assert_eq!(
        int64_array(batch, 3).iter().collect::<Vec<_>>(),
        vec![Some(20), None, Some(200), Some(10), None, Some(100)]
    );
}

#[test]
fn native_lag_window_groups_null_partition_keys_together() {
    // SQL window `PARTITION BY` treats NULL keys as a single group, so the
    // grouped shift must pass `include_null_keys` through to libcudf;
    // otherwise null-keyed rows are dropped and the row-index reattach
    // misaligns the output column.
    let context = backend_context();
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("grp", DataType::Int32, true),
        Field::new("ord", DataType::Int32, false),
        Field::new("value", DataType::Int64, true),
    ]));
    let grp = Int32Array::from(vec![Some(1), None, Some(1), None, Some(1)]);
    let ord = Int32Array::from(vec![2, 1, 1, 2, 3]);
    let value = Int64Array::from(vec![Some(20), Some(100), Some(10), Some(200), Some(30)]);
    let batch =
        RecordBatch::try_new(Arc::clone(&input_schema), vec![Arc::new(grp), Arc::new(ord), Arc::new(value)]).unwrap();
    let mut builder = QueryPlanBuilder::new();
    let source =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("window_input", Arc::clone(&input_schema))));
    let window = builder
        .add_window(
            source,
            WindowSpec::new(
                vec![Expr::column("grp")],
                vec![SortExpr::new(Expr::column("ord"), false, false)],
                vec![WindowFunction::lag(
                    Expr::column("value"),
                    1,
                    None,
                    Field::new("prev_value", DataType::Int64, true),
                )],
            ),
        )
        .unwrap();
    let plan = builder.finish_with_host_sink(window).unwrap();
    let admitted = context.admit_plan(&plan).unwrap();
    let runtime = Arc::new(admitted.runtime().unwrap());
    let frame = DataFrame::from_record_batch(&runtime, &batch).unwrap();

    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "window_input".to_string(),
        GpuResultHandle::new(&admitted, GpuDataFrame::new(frame), Arc::clone(&input_schema)).unwrap(),
    );

    let result = execute_query_plan_with_gpu_inputs(&admitted, &plan, &gpu_inputs).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };

    // Output keeps input row count.
    assert_eq!(batch.num_rows(), 5);
    // Within-partition logical order:
    //   grp=Some(1) sorted by ord: ord=1 (value=10) -> ord=2 (value=20) -> ord=3 (value=30)
    //   grp=None  sorted by ord: ord=1 (value=100) -> ord=2 (value=200)
    // LAG(value, 1) in input row order:
    //   row 0 (grp=Some(1), ord=2): 10
    //   row 1 (grp=None,    ord=1): NULL (first in null partition)
    //   row 2 (grp=Some(1), ord=1): NULL (first in Some(1) partition)
    //   row 3 (grp=None,    ord=2): 100
    //   row 4 (grp=Some(1), ord=3): 20
    assert_eq!(int64_array(batch, 3).iter().collect::<Vec<_>>(), vec![Some(10), None, None, Some(100), Some(20)]);
}

#[test]
fn native_lead_window_with_explicit_default_fills_partition_edges() {
    let context = backend_context();
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("grp", DataType::Int32, false),
        Field::new("ord", DataType::Int32, false),
        Field::new("value", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&input_schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 1, 2, 1, 2, 2])),
            Arc::new(Int32Array::from(vec![2, 1, 2, 3, 1, 3])),
            Arc::new(Int64Array::from(vec![10, 20, 100, 30, 200, 300])),
        ],
    )
    .unwrap();
    let mut builder = QueryPlanBuilder::new();
    let source =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("window_input", Arc::clone(&input_schema))));
    let window = builder
        .add_window(
            source,
            WindowSpec::new(
                vec![Expr::column("grp")],
                vec![SortExpr::new(Expr::column("ord"), false, false)],
                vec![WindowFunction::lead(
                    Expr::column("value"),
                    1,
                    Some(LiteralValue::Int64(-1)),
                    Field::new("next_value", DataType::Int64, false),
                )],
            ),
        )
        .unwrap();
    let plan = builder.finish_with_host_sink(window).unwrap();
    let admitted = context.admit_plan(&plan).unwrap();
    let runtime = Arc::new(admitted.runtime().unwrap());
    let frame = DataFrame::from_record_batch(&runtime, &batch).unwrap();

    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "window_input".to_string(),
        GpuResultHandle::new(&admitted, GpuDataFrame::new(frame), Arc::clone(&input_schema)).unwrap(),
    );

    let result = execute_query_plan_with_gpu_inputs(&admitted, &plan, &gpu_inputs).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };

    // LEAD(value, 1, -1) in input row order:
    //   grp=1 sorted by ord: 20 → 10 → 30 → end
    //   grp=2 sorted by ord: 200 → 100 → 300 → end
    //   row 0 (grp=1, ord=2, value=10): next = 30
    //   row 1 (grp=1, ord=1, value=20): next = 10
    //   row 2 (grp=2, ord=2, value=100): next = 300
    //   row 3 (grp=1, ord=3, value=30): next = -1 (end of partition)
    //   row 4 (grp=2, ord=1, value=200): next = 100
    //   row 5 (grp=2, ord=3, value=300): next = -1 (end of partition)
    assert_eq!(int64_array(batch, 3).values().as_ref(), &[30, 10, 300, -1, 100, -1]);
}

#[test]
fn native_row_number_window_counts_one_based_rows_per_partition() {
    let context = backend_context();
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("grp", DataType::Int32, false),
        Field::new("ord", DataType::Int32, false),
        Field::new("value", DataType::Int64, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&input_schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 1, 2, 1, 2, 2])),
            Arc::new(Int32Array::from(vec![2, 1, 2, 3, 1, 3])),
            Arc::new(Int64Array::from(vec![Some(20), Some(10), None, Some(30), Some(7), Some(3)])),
        ],
    )
    .unwrap();
    let mut builder = QueryPlanBuilder::new();
    let source =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("window_input", Arc::clone(&input_schema))));
    let sort = builder
        .add_sort(
            source,
            SortSpec::new(vec![
                SortExpr::new(Expr::column("grp"), false, false),
                SortExpr::new(Expr::column("ord"), false, false),
            ]),
        )
        .unwrap();
    let window = builder
        .add_window(
            sort,
            WindowSpec::new(
                vec![Expr::column("grp")],
                vec![SortExpr::new(Expr::column("ord"), false, false)],
                vec![
                    WindowFunction::row_number("rn"),
                    WindowFunction::row_number_i64("rn_i64"),
                    WindowFunction::row_number_i32("rn_i32"),
                ],
            ),
        )
        .unwrap();
    let plan = builder.finish_with_host_sink(window).unwrap();
    let admitted = context.admit_plan(&plan).unwrap();
    let runtime = Arc::new(admitted.runtime().unwrap());
    let frame = DataFrame::from_record_batch(&runtime, &batch).unwrap();

    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "window_input".to_string(),
        GpuResultHandle::new(&admitted, GpuDataFrame::new(frame), Arc::clone(&input_schema)).unwrap(),
    );

    let result = execute_query_plan_with_gpu_inputs(&admitted, &plan, &gpu_inputs).unwrap();
    // The sort establishes the window's partition and order keys, so the
    // window runs in the compiled pipeline.
    assert_eq!(result.metrics().compiled_pipeline_attempts, 1);
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };

    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 1, 1, 2, 2, 2]);
    assert_eq!(int32_array(batch, 1).values().as_ref(), &[1, 2, 3, 1, 2, 3]);
    assert_eq!(uint64_array(batch, 3).values().as_ref(), &[1, 2, 3, 1, 2, 3]);
    assert_eq!(int64_array(batch, 4).values().as_ref(), &[1, 2, 3, 1, 2, 3]);
    assert_eq!(int32_array(batch, 5).values().as_ref(), &[1, 2, 3, 1, 2, 3]);
    admitted.finish();
}

#[test]
fn native_row_number_window_refuses_input_ordered_only_by_its_order_key() {
    // Sorted by `ord` alone, the rows of each `grp` partition interleave, and
    // positional numbering would return [1, 1, 3, 3, 5, 5] without error.
    let context = backend_context();
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("grp", DataType::Int32, false),
        Field::new("ord", DataType::Int32, false),
    ]));
    let mut builder = QueryPlanBuilder::new();
    let source =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("window_input", Arc::clone(&input_schema))));
    let sort = builder.add_sort(source, SortSpec::new(vec![SortExpr::new(Expr::column("ord"), false, false)])).unwrap();
    let window = builder
        .add_window(
            sort,
            WindowSpec::new(
                vec![Expr::column("grp")],
                vec![SortExpr::new(Expr::column("ord"), false, false)],
                vec![WindowFunction::row_number("rn")],
            ),
        )
        .unwrap();
    let plan = builder.finish_with_host_sink(window).unwrap();

    let Err(error) = context.admit_plan(&plan) else {
        panic!("a window over input not grouped by its partition must not run");
    };
    assert_eq!(error.fact_value("compiled_pipeline_rejection_reason"), Some("window_input_ordering_not_established"));
}

#[test]
fn native_lead_window_attaches_output_to_filtered_parquet_source_frame() {
    // A Parquet chunk is decoded under a stage budget token, so its frame
    // carries the token's stream and memory resource instead of the
    // runtime's. The LEAD output column is built on the runtime's resources,
    // and the window must still attach it to that frame.
    let dir = unique_temp_dir("query-engine-lead-parquet-source");
    let path = dir.join("input.parquet");
    let schema = Arc::new(Schema::new(vec![
        Field::new("grp", DataType::Int32, false),
        Field::new("ord", DataType::Int32, false),
        Field::new("value", DataType::Int64, false),
        Field::new("keep", DataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 1, 1, 2, 1, 2, 2])),
            Arc::new(Int32Array::from(vec![2, 1, 1, 9, 3, 3, 2, 9])),
            Arc::new(Int64Array::from(vec![20, 200, 10, 999, 300, 30, 100, 888])),
            Arc::new(Int32Array::from(vec![1, 1, 1, 0, 1, 1, 1, 0])),
        ],
    )
    .unwrap();
    write_record_batch_with_row_group_rows(&path, Arc::clone(&schema), batch, 8);
    let read_columns = schema.fields().iter().map(|field| field.name().clone()).collect();
    let source = LocalParquetSource::from_path(&path, Arc::clone(&schema), read_columns);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let filter = builder
        .add_filter(source, Expr::binary(Expr::column("keep"), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(1))))
        .unwrap();
    let sort = builder
        .add_sort(
            filter,
            SortSpec::new(vec![
                SortExpr::new(Expr::column("grp"), false, false),
                SortExpr::new(Expr::column("ord"), false, false),
            ]),
        )
        .unwrap();
    let window = builder
        .add_window(
            sort,
            WindowSpec::new(
                vec![Expr::column("grp")],
                vec![SortExpr::new(Expr::column("ord"), false, false)],
                vec![WindowFunction::lead(
                    Expr::column("value"),
                    1,
                    None,
                    Field::new("next_value", DataType::Int64, true),
                )],
            ),
        )
        .unwrap();
    let plan = builder.finish_with_host_sink(window).unwrap();

    let context = backend_context();
    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };

    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 1, 1, 2, 2, 2]);
    assert_eq!(int32_array(batch, 1).values().as_ref(), &[1, 2, 3, 1, 2, 3]);
    assert_eq!(int64_array(batch, 2).values().as_ref(), &[10, 20, 30, 200, 100, 300]);
    assert_eq!(
        int64_array(batch, 4).iter().collect::<Vec<_>>(),
        vec![Some(20), Some(30), None, Some(100), Some(300), None]
    );

    fs::remove_dir_all(dir).unwrap();
}

struct ResidentWindowRun {
    output: RecordBatch,
    input_bytes: u64,
    peak_bytes: u64,
}

/// Runs `spec` over `batch` as a GPU-resident input, behind a sort on `sort_by`
/// when that is not empty, and reports the attempt's device bytes before the
/// run and at its peak.
fn run_resident_window(batch: &RecordBatch, sort_by: &[&str], spec: WindowSpec) -> ResidentWindowRun {
    let input_schema = batch.schema();
    let mut builder = QueryPlanBuilder::new();
    let mut input =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("window_input", Arc::clone(&input_schema))));
    if !sort_by.is_empty() {
        let keys = sort_by.iter().map(|name| SortExpr::new(Expr::column(*name), false, false)).collect();
        input = builder.add_sort(input, SortSpec::new(keys)).unwrap();
    }
    let window = builder.add_window(input, spec).unwrap();
    let plan = builder.finish_with_host_sink(window).unwrap();

    let admitted = backend_context().admit_plan(&plan).unwrap();
    let runtime = Arc::new(admitted.runtime().unwrap());
    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "window_input".to_string(),
        GpuResultHandle::new(
            &admitted,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, batch).unwrap()),
            input_schema,
        )
        .unwrap(),
    );
    let input_bytes = admitted.attempt_allocation_domain_usage().unwrap().unwrap().current_bytes;

    let result = execute_query_plan_with_gpu_inputs(&admitted, &plan, &gpu_inputs).unwrap();
    let QueryOutput::HostArrow(output) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    let output = output.clone();
    let peak_bytes = admitted.attempt_allocation_domain_usage().unwrap().unwrap().peak_bytes;
    admitted.finish();
    ResidentWindowRun { output, input_bytes, peak_bytes }
}

/// Eight rows in no particular order, with nullable partition and value
/// columns and string, float and boolean payload columns that include nulls
/// and an empty string.
fn wide_window_batch() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("grp", DataType::Int32, true),
        Field::new("ord", DataType::Int32, false),
        Field::new("value", DataType::Int64, true),
        Field::new("label", DataType::Utf8, true),
        Field::new("note", DataType::Utf8, true),
        Field::new("amount", DataType::Float64, true),
        Field::new("flag", DataType::Boolean, true),
        Field::new("tag", DataType::Int32, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![Some(1), None, Some(2), Some(1), None, Some(2), Some(1), Some(2)])),
            Arc::new(Int32Array::from(vec![3, 2, 1, 1, 1, 2, 2, 3])),
            Arc::new(Int64Array::from(vec![Some(30), Some(200), None, Some(10), Some(100), Some(20), None, Some(40)])),
            Arc::new(StringArray::from(vec![
                Some("alpha"),
                None,
                Some("gamma"),
                Some(""),
                Some("epsilon"),
                None,
                Some("eta"),
                Some("theta"),
            ])),
            Arc::new(StringArray::from(vec![
                Some("n-0"),
                Some("n-1"),
                None,
                Some("a note that is much longer than the others"),
                Some("n-4"),
                Some("n-5"),
                None,
                Some("n-7"),
            ])),
            Arc::new(Float64Array::from(vec![
                Some(1.5),
                None,
                Some(-2.25),
                Some(0.0),
                Some(4.5),
                None,
                Some(6.75),
                Some(7.0),
            ])),
            Arc::new(BooleanArray::from(vec![
                Some(true),
                Some(false),
                None,
                Some(true),
                None,
                Some(false),
                Some(true),
                Some(false),
            ])),
            Arc::new(Int32Array::from(vec![10, 11, 12, 13, 14, 15, 16, 17])),
        ],
    )
    .unwrap()
}

/// Output row `row` must carry the input columns of input row `input_rows[row]`.
fn assert_payload_columns_follow_rows(output: &RecordBatch, input: &RecordBatch, input_rows: &[u32]) {
    let indices = UInt32Array::from(input_rows.to_vec());
    for (index, field) in input.schema().fields().iter().enumerate() {
        let expected = take(input.column(index).as_ref(), &indices, None).unwrap();
        assert_eq!(output.schema().field(index), field.as_ref());
        assert_eq!(output.column(index).as_ref(), expected.as_ref(), "payload column {} changed", field.name());
    }
}

#[test]
fn native_ordered_windows_return_wide_payload_columns_unchanged_in_input_order() {
    let input = wide_window_batch();
    let run = run_resident_window(
        &input,
        &[],
        WindowSpec::new(
            vec![Expr::column("grp")],
            vec![SortExpr::new(Expr::column("ord"), false, false)],
            vec![
                WindowFunction::lag(Expr::column("value"), 1, None, Field::new("prev_value", DataType::Int64, true)),
                WindowFunction::lead(
                    Expr::column("value"),
                    1,
                    Some(LiteralValue::Int64(-1)),
                    Field::new("next_value", DataType::Int64, true),
                ),
                WindowFunction::cumulative_aggregate(
                    AggregateFunction::Sum,
                    Expr::column("value"),
                    Field::new("running_sum", DataType::Int64, true),
                ),
                WindowFunction::cumulative_aggregate(
                    AggregateFunction::Max,
                    Expr::column("value"),
                    Field::new("running_max", DataType::Int64, true),
                ),
            ],
        ),
    );
    let output = &run.output;

    assert_payload_columns_follow_rows(output, &input, &[0, 1, 2, 3, 4, 5, 6, 7]);
    // Partitions in `ord` order: grp=1 is rows 3, 6, 0; grp=NULL is rows 4, 1;
    // grp=2 is rows 2, 5, 7. Values are 10, NULL, 30 / 100, 200 / NULL, 20, 40.
    assert_eq!(
        int64_array(output, 8).iter().collect::<Vec<_>>(),
        vec![None, Some(100), None, None, None, None, Some(10), Some(20)]
    );
    assert_eq!(
        int64_array(output, 9).iter().collect::<Vec<_>>(),
        vec![Some(-1), Some(-1), Some(20), None, Some(200), Some(40), Some(30), Some(-1)]
    );
    assert_eq!(
        int64_array(output, 10).iter().collect::<Vec<_>>(),
        vec![Some(40), Some(300), None, Some(10), Some(100), Some(20), Some(10), Some(60)]
    );
    assert_eq!(
        int64_array(output, 11).iter().collect::<Vec<_>>(),
        vec![Some(30), Some(200), None, Some(10), Some(100), Some(20), Some(10), Some(40)]
    );
}

#[test]
fn native_whole_partition_aggregate_window_returns_wide_payload_columns_unchanged_in_input_order() {
    let input = wide_window_batch();
    let run = run_resident_window(
        &input,
        &[],
        WindowSpec::new(
            vec![Expr::column("grp")],
            Vec::new(),
            vec![
                WindowFunction::aggregate(
                    AggregateFunction::Sum,
                    Expr::column("value"),
                    Field::new("total", DataType::Int64, true),
                ),
                WindowFunction::aggregate(
                    AggregateFunction::Max,
                    Expr::column("value"),
                    Field::new("peak", DataType::Int64, true),
                ),
            ],
        ),
    );
    let output = &run.output;

    assert_payload_columns_follow_rows(output, &input, &[0, 1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(
        int64_array(output, 8).iter().collect::<Vec<_>>(),
        vec![Some(40), Some(300), Some(60), Some(40), Some(300), Some(60), Some(40), Some(60)]
    );
    assert_eq!(
        int64_array(output, 9).iter().collect::<Vec<_>>(),
        vec![Some(30), Some(200), Some(40), Some(30), Some(200), Some(40), Some(30), Some(40)]
    );
}

/// An empty partition list makes the whole input one partition: every row
/// carries the aggregate of all rows, and a null value does not count.
#[test]
fn native_whole_input_aggregate_window_broadcasts_global_aggregates_in_input_order() {
    let input = wide_window_batch();
    let run = run_resident_window(
        &input,
        &[],
        WindowSpec::new(
            Vec::new(),
            Vec::new(),
            vec![
                WindowFunction::aggregate(
                    AggregateFunction::Sum,
                    Expr::column("value"),
                    Field::new("total", DataType::Int64, true),
                ),
                WindowFunction::aggregate(
                    AggregateFunction::Max,
                    Expr::column("value"),
                    Field::new("peak", DataType::Int64, true),
                ),
                WindowFunction::aggregate(
                    AggregateFunction::Count,
                    Expr::column("value"),
                    Field::new("present", DataType::Int64, false),
                ),
            ],
        ),
    );
    let output = &run.output;

    assert_payload_columns_follow_rows(output, &input, &[0, 1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(int64_array(output, 8).iter().collect::<Vec<_>>(), vec![Some(400); 8]);
    assert_eq!(int64_array(output, 9).iter().collect::<Vec<_>>(), vec![Some(200); 8]);
    assert_eq!(int64_array(output, 10).iter().collect::<Vec<_>>(), vec![Some(6); 8]);
}

#[test]
fn native_whole_input_aggregate_window_over_no_rows_returns_no_rows() {
    let input = wide_window_batch().slice(0, 0);
    let run = run_resident_window(
        &input,
        &[],
        WindowSpec::new(
            Vec::new(),
            Vec::new(),
            vec![WindowFunction::aggregate(
                AggregateFunction::Count,
                Expr::column("value"),
                Field::new("present", DataType::Int64, false),
            )],
        ),
    );

    assert_eq!(run.output.num_rows(), 0);
    assert_eq!(run.output.num_columns(), input.num_columns() + 1);
}

#[test]
fn native_rank_and_row_number_windows_return_wide_payload_columns_unchanged_in_input_order() {
    let input = wide_window_batch();
    let run = run_resident_window(
        &input,
        &["grp", "ord"],
        WindowSpec::new(
            vec![Expr::column("grp")],
            vec![SortExpr::new(Expr::column("ord"), false, false)],
            vec![WindowFunction::rank("rk"), WindowFunction::row_number("rn")],
        ),
    );
    let output = &run.output;

    // The sort puts NULL partition keys last: grp=1 rows 3, 6, 0, then grp=2
    // rows 2, 5, 7, then grp=NULL rows 4, 1.
    assert_payload_columns_follow_rows(output, &input, &[3, 6, 0, 2, 5, 7, 4, 1]);
    assert_eq!(uint64_array(output, 8).values().as_ref(), &[1, 2, 3, 1, 2, 3, 1, 2]);
    assert_eq!(uint64_array(output, 9).values().as_ref(), &[1, 2, 3, 1, 2, 3, 1, 2]);
}

#[test]
fn native_rank_and_row_number_windows_accept_an_order_key_that_is_also_the_partition_key() {
    let input = wide_window_batch();
    let run = run_resident_window(
        &input,
        &["grp"],
        WindowSpec::new(
            vec![Expr::column("grp")],
            vec![SortExpr::new(Expr::column("grp"), false, false)],
            vec![WindowFunction::rank("rk"), WindowFunction::row_number("rn")],
        ),
    );
    let output = &run.output;

    // Every row of a partition ties on the order key, so rank is 1 throughout
    // while row_number still counts the partition's rows. The sort puts NULL
    // keys last.
    assert_eq!(
        int32_array(output, 0).iter().collect::<Vec<_>>(),
        vec![Some(1), Some(1), Some(1), Some(2), Some(2), Some(2), None, None]
    );
    assert_eq!(uint64_array(output, 8).values().as_ref(), &[1, 1, 1, 1, 1, 1, 1, 1]);
    assert_eq!(uint64_array(output, 9).values().as_ref(), &[1, 2, 3, 1, 2, 3, 1, 2]);
}

#[test]
fn native_window_paths_allocate_less_than_the_payload_columns_they_do_not_read() {
    const ROWS: usize = 1 << 17;
    const PAYLOAD_COLUMNS: usize = 48;
    let mut fields = vec![
        Field::new("grp", DataType::Int32, false),
        Field::new("ord", DataType::Int32, false),
        Field::new("value", DataType::Int64, false),
    ];
    let mut columns: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from_iter_values((0..ROWS).map(|row| (row % 97) as i32))),
        // 40503 is odd, so the order key is a permutation of 0..ROWS and the
        // rows of every partition arrive out of order.
        Arc::new(Int32Array::from_iter_values((0..ROWS).map(|row| (row * 40503 % ROWS) as i32))),
        Arc::new(Int64Array::from_iter_values((0..ROWS).map(|row| row as i64))),
    ];
    for column in 0..PAYLOAD_COLUMNS {
        fields.push(Field::new(format!("payload_{column}"), DataType::Int64, false));
        columns.push(Arc::new(Int64Array::from_iter_values((0..ROWS).map(|row| (row * (column + 1)) as i64))));
    }
    let input = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap();
    let payload_bytes = (ROWS * PAYLOAD_COLUMNS * size_of::<i64>()) as u64;

    let order_by_ord = vec![SortExpr::new(Expr::column("ord"), false, false)];
    // The window reads the keys and `value` only. Sorting, joining or copying
    // the payload columns would add at least one payload copy, so a path that
    // stays below half of it never touches them. Rank and row_number need
    // input grouped by the partition key, so their plans sort the whole input
    // first; that sort and the move of its output into the runtime's memory
    // domain are two payload copies that belong to the plan, and the window
    // itself must stay under one more.
    let cases = [
        (
            "lag_lead",
            Vec::new(),
            WindowSpec::new(
                vec![Expr::column("grp")],
                order_by_ord.clone(),
                vec![
                    WindowFunction::lag(Expr::column("value"), 1, None, Field::new("prev", DataType::Int64, true)),
                    WindowFunction::lead(Expr::column("value"), 1, None, Field::new("next", DataType::Int64, true)),
                ],
            ),
            payload_bytes / 2,
        ),
        (
            "cumulative_aggregate",
            Vec::new(),
            WindowSpec::new(
                vec![Expr::column("grp")],
                order_by_ord.clone(),
                vec![WindowFunction::cumulative_aggregate(
                    AggregateFunction::Sum,
                    Expr::column("value"),
                    Field::new("running", DataType::Int64, true),
                )],
            ),
            payload_bytes / 2,
        ),
        (
            "whole_partition_aggregate",
            Vec::new(),
            WindowSpec::new(
                vec![Expr::column("grp")],
                Vec::new(),
                vec![WindowFunction::aggregate(
                    AggregateFunction::Sum,
                    Expr::column("value"),
                    Field::new("total", DataType::Int64, true),
                )],
            ),
            payload_bytes / 2,
        ),
        (
            "rank_row_number",
            vec!["grp", "ord"],
            WindowSpec::new(
                vec![Expr::column("grp")],
                order_by_ord,
                vec![WindowFunction::rank("rk"), WindowFunction::row_number("rn")],
            ),
            3 * payload_bytes,
        ),
    ];

    let mut violations = Vec::new();
    for (name, sort_by, spec, allowed_bytes) in cases {
        let run = run_resident_window(&input, &sort_by, spec);
        assert_eq!(run.output.num_rows(), ROWS, "{name}");
        let allocated_bytes = run.peak_bytes.saturating_sub(run.input_bytes);
        if allocated_bytes >= allowed_bytes {
            violations.push(format!(
                "{name} allocated {allocated_bytes} bytes beyond its input, which must stay below {allowed_bytes} \
                 (payload columns: {payload_bytes} bytes)"
            ));
        }
    }
    assert!(violations.is_empty(), "{violations:#?}");
}

use std::{fs::File, path::Path, sync::Arc};

use arrow::record_batch::RecordBatchOptions;
use query_planning::expr::ColumnRef;

use super::*;

type RollupAvgRow = (Option<String>, Option<String>, Option<String>, Option<String>, u32, f64);

#[test]
fn bounded_native_query_emits_rollup_grand_total_row_for_empty_literal_input() {
    let context = backend_context();
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
    ]));
    let empty_batch = RecordBatch::try_new_with_options(
        Arc::clone(&input_schema),
        vec![Arc::new(Int32Array::from(Vec::<i32>::new())) as _, Arc::new(Int64Array::from(Vec::<i64>::new())) as _],
        &RecordBatchOptions::new().with_row_count(Some(0)),
    )
    .unwrap();
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, true),
        Field::new("__grouping_id", DataType::UInt32, false),
        Field::new("s", DataType::Int64, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let source = builder
        .add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(&input_schema), vec![empty_batch])));
    let aggregate = builder
        .add_aggregate(
            source,
            AggregateSpec::new(
                vec![Expr::column("key")],
                vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("payload"), "s")],
            )
            .with_grouping_sets(GroupingSetsSpec::new(vec![vec![false], vec![true]], "__grouping_id")),
            Arc::clone(&output_schema),
        )
        .unwrap();
    let plan = builder.finish_with_host_sink(aggregate).unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1, "empty-input ROLLUP must keep the grand-total row");
    assert!(int32_array(batch, 0).is_null(0));
    assert!(int64_array(batch, 2).is_null(0));
}

#[test]
fn bounded_native_query_emits_rollup_grand_total_row_for_runtime_filtered_empty_input() {
    let dir = unique_temp_dir("query-engine-bounded-native-filtered-empty-rollup");
    let path = dir.join("bench.parquet");
    let context = backend_context();
    let input_schema = write_filtered_empty_rollup_input_parquet(&path);

    let output_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, true),
        Field::new("__grouping_id", DataType::UInt32, false),
        Field::new("s", DataType::Int64, true),
    ]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "payload".to_string()],
    );
    let filter = Expr::binary(Expr::column("key"), BinaryOp::Gt, Expr::literal(LiteralValue::Int32(100_000)));

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let filtered = builder.add_filter(source, filter).unwrap();
    let aggregate = builder
        .add_aggregate(
            filtered,
            AggregateSpec::new(
                vec![Expr::column("key")],
                vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("payload"), "s")],
            )
            .with_grouping_sets(GroupingSetsSpec::new(vec![vec![false], vec![true]], "__grouping_id")),
            Arc::clone(&output_schema),
        )
        .unwrap();
    let plan = builder.finish_with_host_sink(aggregate).unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics().clone();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };

    assert_eq!(
        batch.num_rows(),
        1,
        "runtime-filtered empty ROLLUP must keep the grand-total row; metrics partials={} final_merges={} source_chunks_committed={} input_rows={}",
        metrics.partial_aggregate_count,
        metrics.final_merge_attempts,
        metrics.source_chunks_committed,
        metrics.input_rows
    );
    assert_eq!(batch.schema().field(0).data_type(), &DataType::Int32);
    assert_eq!(batch.schema().field(1).data_type(), &DataType::UInt32);
    assert_eq!(batch.schema().field(2).data_type(), &DataType::Int64);
    assert!(int32_array(batch, 0).is_null(0), "grand-total key must be NULL");
    let grouping_ids = batch.column(1).as_any().downcast_ref::<UInt32Array>().unwrap();
    assert_eq!(grouping_ids.value(0), 1, "grand-total grouping id must mark null key");
    assert!(int64_array(batch, 2).is_null(0), "grand-total SUM over empty input must be NULL");

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn rollup_grand_total_aggregates_read_grouping_key_input_values() {
    // AVG keeps the plan off the prefix-rollup shortcut, so the grand total runs its own
    // groupby. Its output key is NULL, but SUM, COUNT and MAX of the key read the input rows.
    let context = backend_context();
    let input_schema =
        Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, true), Field::new("y", DataType::Float64, false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&input_schema),
        vec![
            Arc::new(Int32Array::from(vec![Some(1), Some(1), Some(2), None])) as _,
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0])) as _,
        ],
    )
    .unwrap();
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int32, true),
        Field::new("__grouping_id", DataType::UInt32, false),
        Field::new("s", DataType::Int64, true),
        Field::new("c", DataType::Int64, false),
        Field::new("mx", DataType::Int32, true),
        Field::new("m", DataType::Float64, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let source =
        builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(&input_schema), vec![batch])));
    let aggregate = builder
        .add_aggregate(
            source,
            AggregateSpec::new(
                vec![Expr::column("a")],
                vec![
                    AggregateExpr::new(AggregateFunction::Sum, Expr::column("a"), "s"),
                    AggregateExpr::new(AggregateFunction::Count, Expr::column("a"), "c"),
                    AggregateExpr::new(AggregateFunction::Max, Expr::column("a"), "mx"),
                    AggregateExpr::new(AggregateFunction::Avg, Expr::column("y"), "m"),
                ],
            )
            .with_grouping_sets(GroupingSetsSpec::new(vec![vec![false], vec![true]], "__grouping_id")),
            Arc::clone(&output_schema),
        )
        .unwrap();
    let sort = builder
        .add_sort(
            aggregate,
            SortSpec::new(vec![
                SortExpr::new(Expr::column("__grouping_id"), false, false),
                SortExpr::new(Expr::column("a"), false, false),
            ]),
        )
        .unwrap();
    let plan = builder.finish_with_host_sink(sort).unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    let grouping_ids = batch.column(1).as_any().downcast_ref::<UInt32Array>().unwrap();
    let rows = (0..batch.num_rows())
        .map(|row| {
            let value = |array: &Int32Array| (!array.is_null(row)).then(|| array.value(row));
            let sum = int64_array(batch, 2);
            (
                value(int32_array(batch, 0)),
                grouping_ids.value(row),
                (!sum.is_null(row)).then(|| sum.value(row)),
                int64_array(batch, 3).value(row),
                value(int32_array(batch, 4)),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        rows,
        vec![
            (Some(1), 0, Some(2), 2, Some(1)),
            (Some(2), 0, Some(2), 1, Some(2)),
            (None, 0, None, 0, None),
            (None, 1, Some(4), 3, Some(2)),
        ],
        "rows are (a, grouping id, SUM(a), COUNT(a), MAX(a))"
    );
}

#[test]
fn prefix_rollup_count_levels_concatenate_with_per_set_output_types() {
    // SUM, COUNT, MIN and MAX keep the plan on the prefix-rollup shortcut, where coarser levels
    // re-aggregate the finer level's COUNT with SUM.
    let context = backend_context();
    let input_schema = prefix_rollup_count_input_schema();
    let output_schema = prefix_rollup_count_output_schema();

    let batch = prefix_rollup_count_input_batch(&input_schema);
    let source = SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(&input_schema), vec![batch]));
    let plan = prefix_rollup_count_plan(source, &output_schema);
    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_prefix_rollup_count_schema(batch, &output_schema);
    assert_eq!(prefix_rollup_count_rows(batch), expected_prefix_rollup_count_rows());

    let empty_batch = RecordBatch::new_empty(Arc::clone(&input_schema));
    let source = SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(&input_schema), vec![empty_batch]));
    let plan = prefix_rollup_count_plan(source, &output_schema);
    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_prefix_rollup_count_schema(batch, &output_schema);
    assert_eq!(
        prefix_rollup_count_rows(batch),
        vec![(None, None, 3, 0, 0, None, None, None)],
        "empty input keeps only the grand-total identity row"
    );
}

#[test]
fn bounded_prefix_rollup_count_partials_concatenate_with_per_set_output_types() {
    let dir = unique_temp_dir("query-engine-bounded-prefix-rollup-count-partials");
    let path = dir.join("input.parquet");
    // One two-row row group per chunk gives the aggregate several partial frames.
    let context = backend_context_with_memory_control_policy(MemoryControlPolicy {
        device_budget_bytes: 8 * 1024 * 1024 * 1024,
        max_source_chunk_bytes: 1,
        max_row_groups_per_chunk: 1,
        max_retry_attempts: 1,
        min_retry_chunk_row_groups: 1,
    });
    let input_schema = prefix_rollup_count_input_schema();
    let output_schema = prefix_rollup_count_output_schema();
    write_record_batch_with_row_group_rows(
        &path,
        Arc::clone(&input_schema),
        prefix_rollup_count_input_batch(&input_schema),
        2,
    );
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["a".to_string(), "b".to_string(), "y".to_string()],
    );
    let plan = prefix_rollup_count_plan(SourceNode::LocalParquet(source), &output_schema);

    let result = execute_query_plan(&context, &plan).unwrap();
    assert!(
        result.metrics().partial_aggregate_count >= 2,
        "the input should reach the aggregate as more than one partial"
    );
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_prefix_rollup_count_schema(batch, &output_schema);
    assert_eq!(prefix_rollup_count_rows(batch), expected_prefix_rollup_count_rows());

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn per_set_grouping_sets_empty_input_identity_row_cases() {
    run_named_cases(&[
        (
            "aggregate_execution::grouping_sets::cube_count_over_empty_input_keeps_grand_total_row",
            cube_count_over_empty_input_keeps_grand_total_row,
        ),
        (
            "aggregate_execution::grouping_sets::rollup_count_and_avg_over_empty_input_keeps_grand_total_row",
            rollup_count_and_avg_over_empty_input_keeps_grand_total_row,
        ),
        (
            "aggregate_execution::grouping_sets::cube_float32_sum_over_empty_input_keeps_grand_total_row",
            cube_float32_sum_over_empty_input_keeps_grand_total_row,
        ),
        (
            "aggregate_execution::grouping_sets::rollup_decimal_avg_over_empty_input_keeps_grand_total_row",
            rollup_decimal_avg_over_empty_input_keeps_grand_total_row,
        ),
        (
            "aggregate_execution::grouping_sets::bounded_rollup_count_and_avg_over_runtime_filtered_empty_input_keeps_grand_total_row",
            bounded_rollup_count_and_avg_over_runtime_filtered_empty_input_keeps_grand_total_row,
        ),
    ]);
}

/// The four CUBE(a, b) masks, which the prefix-rollup shortcut does not accept.
fn cube_ab_masks() -> Vec<Vec<bool>> {
    vec![vec![false, false], vec![false, true], vec![true, false], vec![true, true]]
}

fn cube_count_over_empty_input_keeps_grand_total_row() {
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int32, true),
        Field::new("b", DataType::Int32, true),
        Field::new("__grouping_id", DataType::UInt32, false),
        Field::new("c", DataType::Int64, false),
    ]));
    let batch = execute_empty_input_grouping_sets(
        vec![Expr::column("a"), Expr::column("b")],
        vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("y"), "c")],
        cube_ab_masks(),
        &output_schema,
    );
    assert_empty_input_grand_total_row(&batch, &output_schema, 3, &["c"]);
}

fn rollup_count_and_avg_over_empty_input_keeps_grand_total_row() {
    // AVG keeps ROLLUP(a) off the prefix-rollup shortcut.
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int32, true),
        Field::new("__grouping_id", DataType::UInt32, false),
        Field::new("c", DataType::Int64, false),
        Field::new("m", DataType::Float64, true),
    ]));
    let batch = execute_empty_input_grouping_sets(
        vec![Expr::column("a")],
        vec![
            AggregateExpr::new(AggregateFunction::Count, Expr::column("y"), "c"),
            AggregateExpr::new(AggregateFunction::Avg, Expr::column("y"), "m"),
        ],
        vec![vec![false], vec![true]],
        &output_schema,
    );
    assert_empty_input_grand_total_row(&batch, &output_schema, 1, &["c"]);
}

fn cube_float32_sum_over_empty_input_keeps_grand_total_row() {
    // SUM of a Float32 column has a Float64 output, as in DataFusion.
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int32, true),
        Field::new("b", DataType::Int32, true),
        Field::new("__grouping_id", DataType::UInt32, false),
        Field::new("s", DataType::Float64, true),
    ]));
    let batch = execute_empty_input_grouping_sets(
        vec![Expr::column("a"), Expr::column("b")],
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("f"), "s")],
        cube_ab_masks(),
        &output_schema,
    );
    assert_empty_input_grand_total_row(&batch, &output_schema, 3, &[]);
}

fn rollup_decimal_avg_over_empty_input_keeps_grand_total_row() {
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int32, true),
        Field::new("__grouping_id", DataType::UInt32, false),
        Field::new("m", DataType::Decimal128(11, 6), true),
    ]));
    let batch = execute_empty_input_grouping_sets(
        vec![Expr::column("a")],
        vec![AggregateExpr::new(AggregateFunction::Avg, Expr::column("d"), "m")],
        vec![vec![false], vec![true]],
        &output_schema,
    );
    assert_empty_input_grand_total_row(&batch, &output_schema, 1, &[]);
}

fn bounded_rollup_count_and_avg_over_runtime_filtered_empty_input_keeps_grand_total_row() {
    let dir = unique_temp_dir("query-engine-bounded-filtered-empty-rollup-count-avg");
    let path = dir.join("bench.parquet");
    let context = backend_context();
    let input_schema = write_filtered_empty_rollup_input_parquet(&path);
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, true),
        Field::new("__grouping_id", DataType::UInt32, false),
        Field::new("c", DataType::Int64, false),
        Field::new("m", DataType::Float64, true),
    ]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "payload".to_string()],
    );
    let filter = Expr::binary(Expr::column("key"), BinaryOp::Gt, Expr::literal(LiteralValue::Int32(100_000)));

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let filtered = builder.add_filter(source, filter).unwrap();
    let aggregate = builder
        .add_aggregate(
            filtered,
            AggregateSpec::new(
                vec![Expr::column("key")],
                vec![
                    AggregateExpr::new(AggregateFunction::Count, Expr::column("payload"), "c"),
                    AggregateExpr::new(AggregateFunction::Avg, Expr::column("payload"), "m"),
                ],
            )
            .with_grouping_sets(GroupingSetsSpec::new(vec![vec![false], vec![true]], "__grouping_id")),
            Arc::clone(&output_schema),
        )
        .unwrap();
    let plan = builder.finish_with_host_sink(aggregate).unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_empty_input_grand_total_row(batch, &output_schema, 1, &["c"]);

    fs::remove_dir_all(dir).unwrap();
}

/// Runs a grouping-sets aggregate over an empty batch of `a`, `b` and `y` (Int32), `f`
/// (Float32) and `d` (Decimal128(7, 2)).
fn execute_empty_input_grouping_sets(
    group_exprs: Vec<Expr>,
    aggr_exprs: Vec<AggregateExpr>,
    masks: Vec<Vec<bool>>,
    output_schema: &Arc<Schema>,
) -> RecordBatch {
    let context = backend_context();
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int32, true),
        Field::new("b", DataType::Int32, true),
        Field::new("y", DataType::Int32, true),
        Field::new("f", DataType::Float32, true),
        Field::new("d", DataType::Decimal128(7, 2), true),
    ]));
    let empty_batch = RecordBatch::new_empty(Arc::clone(&input_schema));

    let mut builder = QueryPlanBuilder::new();
    let source = builder
        .add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(&input_schema), vec![empty_batch])));
    let aggregate = builder
        .add_aggregate(
            source,
            AggregateSpec::new(group_exprs, aggr_exprs)
                .with_grouping_sets(GroupingSetsSpec::new(masks, "__grouping_id")),
            Arc::clone(output_schema),
        )
        .unwrap();
    let plan = builder.finish_with_host_sink(aggregate).unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    batch.clone()
}

/// SQL keeps only the grand-total set over empty input: NULL keys, COUNT 0 for each column in
/// `count_columns`, and NULL for every other aggregate.
fn assert_empty_input_grand_total_row(
    batch: &RecordBatch,
    output_schema: &Arc<Schema>,
    grouping_id: u32,
    count_columns: &[&str],
) {
    let actual_types = batch.schema().fields().iter().map(|field| field.data_type().clone()).collect::<Vec<_>>();
    let expected_types = output_schema.fields().iter().map(|field| field.data_type().clone()).collect::<Vec<_>>();
    assert_eq!(actual_types, expected_types);
    assert_eq!(batch.num_rows(), 1);
    for (index, field) in output_schema.fields().iter().enumerate() {
        let column = batch.column(index);
        if field.name() == "__grouping_id" {
            assert_eq!(column.as_any().downcast_ref::<UInt32Array>().unwrap().value(0), grouping_id);
        } else if count_columns.contains(&field.name().as_str()) {
            assert_eq!(int64_array(batch, index).value(0), 0, "{} must count no rows", field.name());
        } else {
            assert!(column.is_null(0), "{} must be NULL", field.name());
        }
    }
}

#[test]
fn bounded_compiled_join_rollup_avg_final_merge_concatenates_typed_null_partials() {
    let dir = unique_temp_dir("query-engine-bounded-compiled-rollup-avg-typed-null-partials");
    let inventory_path = dir.join("inventory.parquet");
    let item_path = dir.join("item.parquet");
    // One row group per read gives the partitioned join several probe pieces,
    // and each piece reaches the aggregate as one frame.
    // The three-row item build is large under the policy's build-row threshold.
    let context = backend_context_with_memory_control_and_planning_policy(
        MemoryControlPolicy {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 1,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        },
        planning_policy_with_large_build_rows(3),
    );
    let inventory_schema = write_inventory_rollup_parquet(&inventory_path);
    let item_schema = write_item_rollup_parquet(&item_path);
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("product_name", DataType::Utf8, true),
        Field::new("brand", DataType::Utf8, true),
        Field::new("class", DataType::Utf8, true),
        Field::new("category", DataType::Utf8, true),
        Field::new("quantity_on_hand", DataType::Int32, false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("product_name", DataType::Utf8, true),
        Field::new("brand", DataType::Utf8, true),
        Field::new("class", DataType::Utf8, true),
        Field::new("category", DataType::Utf8, true),
        Field::new("__grouping_id", DataType::UInt32, false),
        Field::new("avg_quantity_on_hand", DataType::Float64, true),
    ]));

    let inventory_source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&inventory_path).with_file_size_bytes(fs::metadata(&inventory_path).unwrap().len())],
        inventory_schema,
        vec!["item_sk".to_string(), "quantity_on_hand".to_string()],
    )
    .with_column_roles(vec!["item_sk".to_string()], vec!["quantity_on_hand".to_string()]);
    let item_source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&item_path).with_file_size_bytes(fs::metadata(&item_path).unwrap().len())],
        item_schema,
        vec![
            "item_sk".to_string(),
            "product_name".to_string(),
            "brand".to_string(),
            "class".to_string(),
            "category".to_string(),
        ],
    );

    let mut builder = QueryPlanBuilder::new();
    let inventory = builder.add_source(SourceNode::LocalParquet(inventory_source));
    let item = builder.add_source(SourceNode::LocalParquet(item_source));
    let join = builder
        .add_join(
            inventory,
            item,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(
                    ColumnRef::input(0, "item_sk", "inventory_item_sk"),
                    ColumnRef::input(1, "item_sk", "item_item_sk"),
                )],
            )
            .with_output_projection(vec![
                JoinOutputColumn::right(ColumnRef::input(1, "product_name", "product_name"), "product_name"),
                JoinOutputColumn::right(ColumnRef::input(1, "brand", "brand"), "brand"),
                JoinOutputColumn::right(ColumnRef::input(1, "class", "class"), "class"),
                JoinOutputColumn::right(ColumnRef::input(1, "category", "category"), "category"),
                JoinOutputColumn::left(ColumnRef::input(0, "quantity_on_hand", "quantity_on_hand"), "quantity_on_hand"),
            ])
            .with_build_side(JoinSide::Right),
            Arc::clone(&join_schema),
        )
        .unwrap();
    let aggregate = builder
        .add_aggregate(
            join,
            AggregateSpec::new(
                vec![
                    Expr::column("product_name"),
                    Expr::column("brand"),
                    Expr::column("class"),
                    Expr::column("category"),
                ],
                vec![AggregateExpr::new(
                    AggregateFunction::Avg,
                    Expr::column("quantity_on_hand"),
                    "avg_quantity_on_hand",
                )],
            )
            .with_grouping_sets(GroupingSetsSpec::new(
                vec![
                    vec![false, false, false, false],
                    vec![false, false, false, true],
                    vec![false, false, true, true],
                    vec![false, true, true, true],
                    vec![true, true, true, true],
                ],
                "__grouping_id",
            )),
            Arc::clone(&output_schema),
        )
        .unwrap();
    let sort = builder
        .add_sort(
            aggregate,
            SortSpec::new(vec![
                SortExpr::new(Expr::column("__grouping_id"), false, false),
                SortExpr::new(Expr::column("product_name"), false, true),
                SortExpr::new(Expr::column("brand"), false, true),
                SortExpr::new(Expr::column("class"), false, true),
                SortExpr::new(Expr::column("category"), false, true),
            ]),
        )
        .unwrap();
    let plan = builder.finish_with_host_sink(sort).unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.compiled_pipeline_attempts, 1);
    assert!(
        metrics
            .operators
            .iter()
            .filter_map(|operator| operator.join.as_ref())
            .any(|join| join.execution_strategy == "bounded_partitioned_hash_equi_join"),
        "q22-style ROLLUP AVG regression should exercise the partitioned join stream path"
    );
    assert!(metrics.source_chunks_committed >= 2);
    assert!(
        metrics.partial_aggregate_count >= 2,
        "join-fed ROLLUP AVG should aggregate more than one streamed partial"
    );
    assert!(metrics.final_merge_attempts >= 1);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 13);
    assert_eq!(batch.schema().field(0).data_type(), &DataType::Utf8);
    assert_eq!(batch.schema().field(1).data_type(), &DataType::Utf8);
    assert_eq!(batch.schema().field(2).data_type(), &DataType::Utf8);
    assert_eq!(batch.schema().field(3).data_type(), &DataType::Utf8);
    assert_eq!(batch.schema().field(4).data_type(), &DataType::UInt32);
    assert_eq!(batch.schema().field(5).data_type(), &DataType::Float64);
    assert!(string_array(batch, 0).null_count() > 0);
    assert!(string_array(batch, 1).null_count() > 0);
    assert!(string_array(batch, 2).null_count() > 0);
    assert!(string_array(batch, 3).null_count() > 0);
    assert_rollup_avg_rows_eq(
        rollup_avg_rows(batch),
        vec![
            (
                Some("p1".to_string()),
                Some("BrandA".to_string()),
                Some("Class1".to_string()),
                Some("CatA".to_string()),
                0,
                12.0,
            ),
            (
                Some("p2".to_string()),
                Some("BrandA".to_string()),
                Some("Class1".to_string()),
                Some("CatA".to_string()),
                0,
                6.0,
            ),
            (
                Some("p3".to_string()),
                Some("BrandB".to_string()),
                Some("Class2".to_string()),
                Some("CatB".to_string()),
                0,
                24.0,
            ),
            (Some("p1".to_string()), Some("BrandA".to_string()), Some("Class1".to_string()), None, 1, 12.0),
            (Some("p2".to_string()), Some("BrandA".to_string()), Some("Class1".to_string()), None, 1, 6.0),
            (Some("p3".to_string()), Some("BrandB".to_string()), Some("Class2".to_string()), None, 1, 24.0),
            (Some("p1".to_string()), Some("BrandA".to_string()), None, None, 3, 12.0),
            (Some("p2".to_string()), Some("BrandA".to_string()), None, None, 3, 6.0),
            (Some("p3".to_string()), Some("BrandB".to_string()), None, None, 3, 24.0),
            (Some("p1".to_string()), None, None, None, 7, 12.0),
            (Some("p2".to_string()), None, None, None, 7, 6.0),
            (Some("p3".to_string()), None, None, None, 7, 24.0),
            (None, None, None, None, 15, 114.0 / 7.0),
        ],
        "q22-style ROLLUP AVG rows should preserve typed NULL grouping keys and merged AVG state",
    );

    fs::remove_dir_all(dir).unwrap();
}

fn prefix_rollup_count_input_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int32, true),
        Field::new("b", DataType::Int32, true),
        Field::new("y", DataType::Int32, true),
    ]))
}

fn prefix_rollup_count_input_batch(schema: &Arc<Schema>) -> RecordBatch {
    RecordBatch::try_new(
        Arc::clone(schema),
        vec![
            Arc::new(Int32Array::from(vec![Some(1), Some(1), Some(2), None, Some(2)])) as _,
            Arc::new(Int32Array::from(vec![Some(10), Some(20), Some(10), Some(10), None])) as _,
            Arc::new(Int32Array::from(vec![Some(5), None, Some(7), Some(9), Some(11)])) as _,
        ],
    )
    .unwrap()
}

/// DataFusion output types of `a, b, COUNT(a), COUNT(y), SUM(y), MIN(y), MAX(y)`.
fn prefix_rollup_count_output_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int32, true),
        Field::new("b", DataType::Int32, true),
        Field::new("__grouping_id", DataType::UInt32, false),
        Field::new("c_a", DataType::Int64, false),
        Field::new("c_y", DataType::Int64, false),
        Field::new("s", DataType::Int64, true),
        Field::new("mn", DataType::Int32, true),
        Field::new("mx", DataType::Int32, true),
    ]))
}

/// `SELECT a, b, COUNT(a), COUNT(y), SUM(y), MIN(y), MAX(y) ... GROUP BY ROLLUP(a, b)`, ordered
/// by grouping id, `a` and `b` with NULLs last.
fn prefix_rollup_count_plan(source: SourceNode, output_schema: &Arc<Schema>) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(source);
    let aggregate = builder
        .add_aggregate(
            source,
            AggregateSpec::new(
                vec![Expr::column("a"), Expr::column("b")],
                vec![
                    AggregateExpr::new(AggregateFunction::Count, Expr::column("a"), "c_a"),
                    AggregateExpr::new(AggregateFunction::Count, Expr::column("y"), "c_y"),
                    AggregateExpr::new(AggregateFunction::Sum, Expr::column("y"), "s"),
                    AggregateExpr::new(AggregateFunction::Min, Expr::column("y"), "mn"),
                    AggregateExpr::new(AggregateFunction::Max, Expr::column("y"), "mx"),
                ],
            )
            .with_grouping_sets(GroupingSetsSpec::new(
                vec![vec![false, false], vec![false, true], vec![true, true]],
                "__grouping_id",
            )),
            Arc::clone(output_schema),
        )
        .unwrap();
    let sort = builder
        .add_sort(
            aggregate,
            SortSpec::new(vec![
                SortExpr::new(Expr::column("__grouping_id"), false, false),
                SortExpr::new(Expr::column("a"), false, false),
                SortExpr::new(Expr::column("b"), false, false),
            ]),
        )
        .unwrap();
    builder.finish_with_host_sink(sort).unwrap()
}

fn assert_prefix_rollup_count_schema(batch: &RecordBatch, output_schema: &Arc<Schema>) {
    let actual = batch.schema().fields().iter().map(|field| field.data_type().clone()).collect::<Vec<_>>();
    let expected = output_schema.fields().iter().map(|field| field.data_type().clone()).collect::<Vec<_>>();
    assert_eq!(actual, expected);
}

/// `(a, b, grouping id, COUNT(a), COUNT(y), SUM(y), MIN(y), MAX(y))`.
type PrefixRollupCountRow = (Option<i32>, Option<i32>, u32, i64, i64, Option<i64>, Option<i32>, Option<i32>);

fn prefix_rollup_count_rows(batch: &RecordBatch) -> Vec<PrefixRollupCountRow> {
    let grouping_ids = batch.column(2).as_any().downcast_ref::<UInt32Array>().unwrap();
    let sums = int64_array(batch, 5);
    (0..batch.num_rows())
        .map(|row| {
            let int32_value = |index: usize| {
                let array = int32_array(batch, index);
                (!array.is_null(row)).then(|| array.value(row))
            };
            (
                int32_value(0),
                int32_value(1),
                grouping_ids.value(row),
                int64_array(batch, 3).value(row),
                int64_array(batch, 4).value(row),
                (!sums.is_null(row)).then(|| sums.value(row)),
                int32_value(6),
                int32_value(7),
            )
        })
        .collect()
}

/// SQL ROLLUP result for `prefix_rollup_count_input_batch`.
fn expected_prefix_rollup_count_rows() -> Vec<PrefixRollupCountRow> {
    vec![
        (Some(1), Some(10), 0, 1, 1, Some(5), Some(5), Some(5)),
        (Some(1), Some(20), 0, 1, 0, None, None, None),
        (Some(2), Some(10), 0, 1, 1, Some(7), Some(7), Some(7)),
        (Some(2), None, 0, 1, 1, Some(11), Some(11), Some(11)),
        (None, Some(10), 0, 0, 1, Some(9), Some(9), Some(9)),
        (Some(1), None, 1, 2, 1, Some(5), Some(5), Some(5)),
        (Some(2), None, 1, 2, 2, Some(18), Some(7), Some(11)),
        (None, None, 1, 0, 1, Some(9), Some(9), Some(9)),
        (None, None, 3, 4, 4, Some(32), Some(5), Some(11)),
    ]
}

fn rollup_avg_rows(batch: &RecordBatch) -> Vec<RollupAvgRow> {
    let product_names = string_array(batch, 0);
    let brands = string_array(batch, 1);
    let classes = string_array(batch, 2);
    let categories = string_array(batch, 3);
    let grouping_ids = batch.column(4).as_any().downcast_ref::<UInt32Array>().unwrap();
    let averages = float64_array(batch, 5);
    (0..batch.num_rows())
        .map(|row| {
            (
                string_value(product_names, row),
                string_value(brands, row),
                string_value(classes, row),
                string_value(categories, row),
                grouping_ids.value(row),
                averages.value(row),
            )
        })
        .collect()
}

fn assert_rollup_avg_rows_eq(actual: Vec<RollupAvgRow>, expected: Vec<RollupAvgRow>, message: &str) {
    assert_eq!(actual.len(), expected.len(), "{message}: row count mismatch");
    for (row, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            (&actual.0, &actual.1, &actual.2, &actual.3, actual.4),
            (&expected.0, &expected.1, &expected.2, &expected.3, expected.4),
            "{message}: row {row} grouping columns mismatch"
        );
        assert!(
            (actual.5 - expected.5).abs() <= 1e-9,
            "{message}: row {row} avg mismatch, actual={}, expected={}",
            actual.5,
            expected.5
        );
    }
}

fn string_value(array: &StringArray, row: usize) -> Option<String> {
    (!array.is_null(row)).then(|| array.value(row).to_string())
}

fn write_filtered_empty_rollup_input_parquet(path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from((0..32).map(|i| i % 4).collect::<Vec<_>>())),
            Arc::new(Int64Array::from((0..32).map(|i| i as i64).collect::<Vec<_>>())),
        ],
    )
    .unwrap();
    write_single_batch_parquet(path, Arc::clone(&schema), batch);
    schema
}

fn write_inventory_rollup_parquet(path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("item_sk", DataType::Int32, false),
        Field::new("quantity_on_hand", DataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 1, 1, 2, 3, 3, 3])),
            Arc::new(Int32Array::from(vec![10, 12, 14, 6, 20, 24, 28])),
        ],
    )
    .unwrap();
    write_record_batch_with_row_group_rows(path, Arc::clone(&schema), batch, 4);
    schema
}

fn write_item_rollup_parquet(path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("item_sk", DataType::Int32, false),
        Field::new("product_name", DataType::Utf8, true),
        Field::new("brand", DataType::Utf8, true),
        Field::new("class", DataType::Utf8, true),
        Field::new("category", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec![Some("p1"), Some("p2"), Some("p3")])),
            Arc::new(StringArray::from(vec![Some("BrandA"), Some("BrandA"), Some("BrandB")])),
            Arc::new(StringArray::from(vec![Some("Class1"), Some("Class1"), Some("Class2")])),
            Arc::new(StringArray::from(vec![Some("CatA"), Some("CatA"), Some("CatB")])),
        ],
    )
    .unwrap();
    write_single_batch_parquet(path, Arc::clone(&schema), batch);
    schema
}

fn write_single_batch_parquet(path: &Path, schema: Arc<Schema>, batch: RecordBatch) {
    let file = File::create(path).unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(3)).build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

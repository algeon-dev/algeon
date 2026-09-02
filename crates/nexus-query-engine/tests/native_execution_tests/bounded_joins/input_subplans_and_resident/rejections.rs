use std::{path::Path, sync::Arc};

use super::*;

#[test]
fn bounded_native_query_rejects_unsupported_join_under_aggregate_before_source_read() {
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("amount", DataType::Int64, false),
    ]));
    let right_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("amount", DataType::Int64, false),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("sum_amount", DataType::Int64, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_path(
        "/tmp/bounded-aggregate-join-left-does-not-need-to-exist.parquet",
        Arc::clone(&left_schema),
        vec!["id".into(), "amount".into()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_path(
        "/tmp/bounded-aggregate-join-right-does-not-need-to-exist.parquet",
        Arc::clone(&right_schema),
        vec!["id".into()],
    )));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::LeftSemi,
            vec![JoinKey::new(
                nexus_query_engine::expr::ColumnRef::input(0, "id", "left_id"),
                nexus_query_engine::expr::ColumnRef::input(1, "id", "right_id"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "id", "id"), "id"),
            JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "amount", "amount"), "amount"),
        ])
        .with_build_side(JoinSide::Left),
        join_schema,
    );
    let aggregate = builder.add_aggregate(
        join,
        AggregateSpec::new(
            vec![Expr::column("id")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("amount"), "sum_amount")],
        ),
        aggregate_schema,
    );
    let plan = builder.finish_with_host_sink(aggregate);
    assert!(!plan.capability_analysis().admits_root_join());
    assert!(!plan.capability_analysis().admits_single_join_post_root_pipeline());

    let context = backend_context_with_grant(1024 * 1024);

    let error = match execute_query_plan(&context, &plan) {
        Ok(_) => panic!("bounded aggregate join should reject before source reads"),
        Err(error) => error,
    };

    assert_bounded_join_rejection_contract(&error);
}

#[test]
fn bounded_native_query_rejects_grouped_aggregate_join_build_input_without_bounded_envelope() {
    let runtime = backend_runtime();
    let dir = unique_temp_dir("nexus-query-engine-bounded-aggregate-join-input");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("amount", DataType::Int64, false),
    ]));
    let right_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("sum_amount", DataType::Int64, true),
    ]));
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("sum_amount", DataType::Int64, true),
    ]));
    write_aggregate_join_input_parquets(&runtime, &left_path, &right_path, Arc::clone(&left_schema));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&left_path).with_file_size_bytes(1024)],
        Arc::clone(&left_schema),
        vec!["id".into(), "amount".into()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&right_path).with_file_size_bytes(1024)],
        Arc::clone(&right_schema),
        vec!["id".into()],
    )));
    let aggregate = builder.add_aggregate(
        left,
        AggregateSpec::new(
            vec![Expr::column("id")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("amount"), "sum_amount")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let join = builder.add_join(
        aggregate,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                nexus_query_engine::expr::ColumnRef::input(0, "id", "left_id"),
                nexus_query_engine::expr::ColumnRef::input(1, "id", "right_id"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "id", "id"), "id"),
            JoinOutputColumn::left(
                nexus_query_engine::expr::ColumnRef::input(0, "sum_amount", "sum_amount"),
                "sum_amount",
            ),
        ])
        .with_build_side(JoinSide::Left),
        join_schema,
    );
    let plan = builder.finish_with_host_sink(join);
    let context = bounded_join_execution_context(&runtime);

    let error = match execute_query_plan(&context, &plan) {
        Ok(_) => panic!("grouped aggregate join-build input needs explicit bounded envelope admission"),
        Err(error) => error,
    };
    assert_bounded_join_rejection_contract(&error);
    assert_eq!(error.fact_value("bounded_join_rejection_boundary"), Some("admission_capability"));
    assert_eq!(
        error.fact_value("bounded_join_rejection_reason"),
        Some("source_leaf_pipeline_requires_source_filter_projection")
    );

    fs::remove_dir_all(dir).unwrap();
}

fn write_aggregate_join_input_parquets(
    runtime: &cudf::Runtime,
    left_path: &Path,
    right_path: &Path,
    left_schema: Arc<Schema>,
) {
    let left_batch = RecordBatch::try_new(
        left_schema,
        vec![Arc::new(Int32Array::from(vec![1, 1, 2, 3])), Arc::new(Int64Array::from(vec![10, 5, 20, 30]))],
    )
    .unwrap();
    DataFrame::from_record_batch(runtime, &left_batch).unwrap().write_parquet(left_path).unwrap();

    let right_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let right_batch = RecordBatch::try_new(right_schema, vec![Arc::new(Int32Array::from(vec![1, 2, 4]))]).unwrap();
    DataFrame::from_record_batch(runtime, &right_batch).unwrap().write_parquet(right_path).unwrap();
}

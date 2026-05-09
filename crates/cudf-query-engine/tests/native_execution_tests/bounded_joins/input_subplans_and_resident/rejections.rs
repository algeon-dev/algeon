use super::*;

#[test]
fn bounded_native_query_rejects_unsupported_join_under_aggregate_before_source_read() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
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
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/bounded-aggregate-join-left-does-not-need-to-exist.parquet",
        Arc::clone(&left_schema),
        vec!["id".into(), "amount".into()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
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
                cudf_query_engine::ColumnRef::input(0, "id", "left_id"),
                cudf_query_engine::ColumnRef::input(1, "id", "right_id"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "id", "id"), "id"),
            JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "amount", "amount"), "amount"),
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
    assert!(!plan.has_bounded_executable_root_join());
    assert!(!plan.has_bounded_executable_join_pipeline());

    let context = ExecutionContext::builder()
        .with_shared_runtime(runtime)
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 1024 * 1024,
            max_source_chunk_bytes: 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();

    let error = match execute_query_plan(&context, &plan) {
        Ok(_) => panic!("bounded aggregate join should reject before source reads"),
        Err(error) => error,
    };

    assert_bounded_join_rejection_contract(&error);
}

#[test]
fn bounded_native_query_rejects_aggregate_join_input_before_source_read() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
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

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/bounded-aggregate-join-input-left-does-not-need-to-exist.parquet",
        Arc::clone(&left_schema),
        vec!["id".into(), "amount".into()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/bounded-aggregate-join-input-right-does-not-need-to-exist.parquet",
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
                cudf_query_engine::ColumnRef::input(0, "id", "left_id"),
                cudf_query_engine::ColumnRef::input(1, "id", "right_id"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "id", "id"), "id"),
            JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "sum_amount", "sum_amount"), "sum_amount"),
        ])
        .with_build_side(JoinSide::Right),
        join_schema,
    );
    let plan = builder.finish_with_host_sink(join);
    assert!(!plan.has_bounded_executable_root_join());
    assert!(!plan.has_bounded_executable_join_pipeline());

    let context = ExecutionContext::builder()
        .with_shared_runtime(runtime)
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 1024 * 1024,
            max_source_chunk_bytes: 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();

    let error = match execute_query_plan(&context, &plan) {
        Ok(_) => panic!("bounded join with aggregate input should reject before source reads"),
        Err(error) => error,
    };

    assert_bounded_join_rejection_contract(&error);
}

use super::*;

#[test]
fn unbounded_semijoin_reduction_matches_original_execution_on_small_fixture() {
    let dir = unique_temp_dir("nexus-query-engine-unbounded-semijoin-reduction-parity");
    let a_path = dir.join("a.parquet");
    let b_path = dir.join("b.parquet");
    let keyset_path = dir.join("keyset.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());

    let a_schema = Arc::new(Schema::new(vec![
        Field::new("a_key", DataType::Int32, false),
        Field::new("a_payload", DataType::Int32, false),
    ]));
    let b_schema = Arc::new(Schema::new(vec![
        Field::new("b_key", DataType::Int32, false),
        Field::new("b_payload", DataType::Int32, false),
    ]));
    let keyset_schema = Arc::new(Schema::new(vec![Field::new("b_key", DataType::Int32, false)]));

    write_i32_parquet(
        &runtime,
        &a_path,
        Arc::clone(&a_schema),
        vec![("a_key", vec![1, 2, 3]), ("a_payload", vec![10, 20, 30])],
    );
    write_i32_parquet(
        &runtime,
        &b_path,
        Arc::clone(&b_schema),
        vec![("b_key", vec![1, 2, 3, 4]), ("b_payload", vec![100, 200, 300, 400])],
    );
    write_i32_parquet(&runtime, &keyset_path, Arc::clone(&keyset_schema), vec![("b_key", vec![2, 2, 4])]);

    let plan = semijoin_reduction_execution_plan(&a_path, &a_schema, &b_path, &b_schema, &keyset_path, &keyset_schema);
    let optimizer_result = nexus_query_engine::optimizer::optimize_query_plan(plan.clone(), &Default::default());
    assert_eq!(optimizer_result.change, nexus_query_engine::optimizer::NativeOptimizerChange::Changed);
    assert_eq!(optimizer_result.diagnostics.semijoin_reduction.applied_count, 1);

    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime)).unwrap();
    assert_eq!(context.memory_control_policy(), MemoryControlPolicy::Unbounded);
    let original = execute_query_plan(&context, &plan).unwrap();
    let optimized = execute_query_plan(&context, &optimizer_result.plan).unwrap();

    let QueryOutput::HostArrow(original_batch) = original.output() else {
        panic!("original semijoin plan should materialize host output");
    };
    let QueryOutput::HostArrow(optimized_batch) = optimized.output() else {
        panic!("optimized semijoin plan should materialize host output");
    };
    assert_eq!(original_batch.schema(), optimized_batch.schema());
    assert_eq!(original_batch.num_columns(), optimized_batch.num_columns());
    assert_eq!(original_batch.num_rows(), optimized_batch.num_rows());
    for column_index in 0..original_batch.num_columns() {
        assert_eq!(
            int32_array(original_batch, column_index).values().as_ref(),
            int32_array(optimized_batch, column_index).values().as_ref()
        );
    }
    assert_eq!(int32_array(optimized_batch, 0).values().as_ref(), &[2]);
    assert_eq!(int32_array(optimized_batch, 1).values().as_ref(), &[20]);
    assert_eq!(int32_array(optimized_batch, 2).values().as_ref(), &[2]);
    assert_eq!(int32_array(optimized_batch, 3).values().as_ref(), &[200]);

    fs::remove_dir_all(dir).unwrap();
}

fn semijoin_reduction_execution_plan(
    a_path: &Path,
    a_schema: &Arc<Schema>,
    b_path: &Path,
    b_schema: &Arc<Schema>,
    keyset_path: &Path,
    keyset_schema: &Arc<Schema>,
) -> nexus_query_engine::plan::QueryPlan {
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("a_key", DataType::Int32, false),
        Field::new("a_payload", DataType::Int32, false),
        Field::new("b_key", DataType::Int32, false),
        Field::new("b_payload", DataType::Int32, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let a = builder.add_source(SourceNode::LocalParquet(join_left_source(a_path, a_schema, &["a_key", "a_payload"])));
    let b = builder.add_source(SourceNode::LocalParquet(join_right_source(b_path, b_schema, &["b_key", "b_payload"])));
    let keyset =
        builder.add_source(SourceNode::LocalParquet(join_right_source(keyset_path, keyset_schema, &["b_key"])));
    let keyset = builder.add_aggregate(
        keyset,
        AggregateSpec::new(vec![Expr::column("b_key")], Vec::new()),
        Arc::clone(keyset_schema),
    );
    let joined = builder.add_join(
        a,
        b,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                nexus_query_engine::expr::ColumnRef::input(0, "a_key", "a_key"),
                nexus_query_engine::expr::ColumnRef::input(1, "b_key", "b_key"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "a_key", "a_key"), "a_key"),
            JoinOutputColumn::left(
                nexus_query_engine::expr::ColumnRef::input(0, "a_payload", "a_payload"),
                "a_payload",
            ),
            JoinOutputColumn::right(nexus_query_engine::expr::ColumnRef::input(1, "b_key", "b_key"), "b_key"),
            JoinOutputColumn::right(
                nexus_query_engine::expr::ColumnRef::input(1, "b_payload", "b_payload"),
                "b_payload",
            ),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let semi = builder.add_join(
        joined,
        keyset,
        JoinSpec::new(
            JoinType::LeftSemi,
            vec![JoinKey::new(
                nexus_query_engine::expr::ColumnRef::input(0, "b_key", "b_key"),
                nexus_query_engine::expr::ColumnRef::input(1, "b_key", "b_key"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "a_key", "a_key"), "a_key"),
            JoinOutputColumn::left(
                nexus_query_engine::expr::ColumnRef::input(0, "a_payload", "a_payload"),
                "a_payload",
            ),
            JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "b_key", "b_key"), "b_key"),
            JoinOutputColumn::left(
                nexus_query_engine::expr::ColumnRef::input(0, "b_payload", "b_payload"),
                "b_payload",
            ),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let sort =
        builder.add_sort(semi, SortSpec::new(vec![SortExpr::new(Expr::column("a_key"), false, false)]), output_schema);
    builder.finish_with_host_sink(sort)
}

fn write_i32_parquet(runtime: &cudf::Runtime, path: &Path, schema: Arc<Schema>, columns: Vec<(&str, Vec<i32>)>) {
    let batch = RecordBatch::try_new(
        schema,
        columns.into_iter().map(|(_name, values)| Arc::new(Int32Array::from(values)) as Arc<dyn Array>).collect(),
    )
    .unwrap();
    DataFrame::from_record_batch(runtime, &batch).unwrap().write_parquet(path).unwrap();
}

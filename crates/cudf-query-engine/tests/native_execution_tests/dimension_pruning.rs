use super::*;

#[test]
fn dimension_derived_row_group_pruning_uses_filtered_dimension_key_range() {
    let dir = unique_temp_dir("cudf-query-engine-dimension-pruning");
    let dim_path = dir.join("dim.parquet");
    let fact_path = dir.join("fact.parquet");

    let dim_schema = Arc::new(Schema::new(vec![
        Field::new("dim_key", DataType::Int32, false),
        Field::new("day", DataType::Date32, false),
    ]));
    let dim_batch = RecordBatch::try_new(
        Arc::clone(&dim_schema),
        vec![Arc::new(Int32Array::from(vec![10, 30, 50])), Arc::new(Date32Array::from(vec![10, 30, 30]))],
    )
    .unwrap();
    write_record_batch_with_row_group_rows(&dim_path, Arc::clone(&dim_schema), dim_batch, 1);

    let fact_schema = Arc::new(Schema::new(vec![
        Field::new("fact_key", DataType::Int32, false),
        Field::new("payload", DataType::Int32, false),
    ]));
    let fact_batch = RecordBatch::try_new(
        Arc::clone(&fact_schema),
        vec![Arc::new(Int32Array::from(vec![10, 20, 30, 40])), Arc::new(Int32Array::from(vec![100, 200, 300, 400]))],
    )
    .unwrap();
    write_record_batch_with_row_group_rows(&fact_path, Arc::clone(&fact_schema), fact_batch, 1);

    let plan = dimension_pruning_plan(&dim_path, &dim_schema, &fact_path, &fact_schema);
    let optimizer_result = cudf_query_engine::optimize_query_plan(plan, &Default::default());

    assert_eq!(optimizer_result.change, cudf_query_engine::NativeOptimizerChange::Changed);
    assert_eq!(optimizer_result.diagnostics.dimension_derived_row_group_pruning.applied_count, 1);
    let candidate = optimizer_result
        .diagnostics
        .dimension_derived_row_group_pruning
        .candidates
        .iter()
        .find(|candidate| candidate.applied)
        .expect("dimension-derived pruning should apply");
    assert_eq!(candidate.derived_min_key, Some(30));
    assert_eq!(candidate.derived_max_key, Some(50));
    assert_eq!(candidate.selected_row_groups_before, Some(4));
    assert_eq!(candidate.selected_row_groups_after, Some(2));

    let fact_source = optimizer_result
        .plan
        .nodes()
        .iter()
        .find_map(|node| match &node.kind {
            cudf_query_engine::PlanNode::Source(source) => match source.as_ref() {
                SourceNode::LocalParquet(source) if source.files.first().is_some_and(|file| file.path == fact_path) => {
                    Some(source)
                },
                _ => None,
            },
            _ => None,
        })
        .expect("optimized plan should retain fact source");
    assert_eq!(fact_source.files[0].row_groups.as_deref(), Some(&[2, 3][..]));

    fs::remove_dir_all(dir).unwrap();
}

fn dimension_pruning_plan(
    dim_path: &Path,
    dim_schema: &Arc<Schema>,
    fact_path: &Path,
    fact_schema: &Arc<Schema>,
) -> cudf_query_engine::QueryPlan {
    let mut dim_source =
        LocalParquetSource::new(dim_path, Arc::clone(dim_schema), vec!["dim_key".to_string(), "day".to_string()]);
    dim_source = dim_source.with_reader_predicate(Expr::binary(
        Expr::binary(Expr::column("day"), BinaryOp::GtEq, Expr::literal(LiteralValue::Date32(15))),
        BinaryOp::And,
        Expr::binary(Expr::column("day"), BinaryOp::LtEq, Expr::literal(LiteralValue::Date32(35))),
    ));
    let fact_source = LocalParquetSource::new(
        fact_path,
        Arc::clone(fact_schema),
        vec!["fact_key".to_string(), "payload".to_string()],
    );
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("dim_key", DataType::Int32, false),
        Field::new("payload", DataType::Int32, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let dim = builder.add_source(SourceNode::LocalParquet(dim_source));
    let dim = builder.add_projection(
        dim,
        vec![ProjectionOutput::new(Expr::column("dim_key"), Field::new("dim_key", DataType::Int32, false))],
    );
    let fact = builder.add_source(SourceNode::LocalParquet(fact_source));
    let fact = builder.add_projection(
        fact,
        vec![
            ProjectionOutput::new(Expr::column("fact_key"), Field::new("fact_key", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("payload"), Field::new("payload", DataType::Int32, false)),
        ],
    );
    let join = builder.add_join(
        dim,
        fact,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                cudf_query_engine::ColumnRef::input(0, "dim_key", "dim_key"),
                cudf_query_engine::ColumnRef::input(1, "fact_key", "fact_key"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "dim_key", "dim_key"), "dim_key"),
            JoinOutputColumn::right(cudf_query_engine::ColumnRef::input(1, "payload", "payload"), "payload"),
        ]),
        Arc::clone(&output_schema),
    );
    builder.finish_with_host_sink(join)
}

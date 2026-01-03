use super::*;

pub(crate) fn mixed_keyed_residual_join_grouped_aggregate_input_plan(
    fact_path: &Path,
    grouped_fact_path: &Path,
    dim_path: &Path,
    fact_schema: &Arc<Schema>,
    grouped_fact_schema: &Arc<Schema>,
    dim_schema: &Arc<Schema>,
) -> cudf_query_engine::QueryPlan {
    let joined_fact_schema = Arc::new(Schema::new(vec![
        Field::new("entity_id", DataType::Int32, false),
        Field::new("measure", DataType::Float64, false),
        Field::new("quantity", DataType::Float64, false),
    ]));
    let joined_fact_residual_schema = Arc::new(Schema::new(vec![
        Field::new("entity_id", DataType::Int32, false),
        Field::new("measure", DataType::Float64, false),
        Field::new("quantity", DataType::Float64, false),
        Field::new("__cudf_join_residual_left_0", DataType::Float64, false),
    ]));
    let grouped_schema = Arc::new(Schema::new(vec![
        Field::new("entity_id", DataType::Int32, false),
        Field::new("avg_quantity", DataType::Float64, true),
    ]));
    let root_output_schema = Arc::new(Schema::new(vec![Field::new("measure", DataType::Float64, false)]));
    let aggregate_input_schema = Arc::new(Schema::new(vec![Field::new("aggregate_arg", DataType::Float64, false)]));
    let aggregate_schema = Arc::new(Schema::new(vec![Field::new("sum_measure", DataType::Float64, true)]));
    let output_schema = Arc::new(Schema::new(vec![Field::new("result", DataType::Float64, true)]));

    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(SourceNode::LocalParquet(mixed_aggregate_fact_source(
        fact_path,
        fact_schema,
        &["entity_id", "quantity", "measure"],
    )));
    let dim = builder.add_source(SourceNode::LocalParquet(mixed_aggregate_dim_source(dim_path, dim_schema)));
    let fact_dim = builder.add_join(
        fact,
        dim,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                cudf_query_engine::ColumnRef::input(0, "entity_id", "entity_id"),
                cudf_query_engine::ColumnRef::input(1, "entity_id", "entity_id"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "entity_id", "entity_id"), "entity_id"),
            JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "measure", "measure"), "measure"),
            JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "quantity", "quantity"), "quantity"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&joined_fact_schema),
    );
    let joined_projection = builder.add_projection(
        fact_dim,
        projection_outputs(
            vec![Expr::column("entity_id"), Expr::column("measure"), Expr::column("quantity")],
            Arc::clone(&joined_fact_schema),
        ),
    );
    let joined_projection = builder.add_projection(
        joined_projection,
        projection_outputs(
            vec![
                Expr::column("entity_id"),
                Expr::column("measure"),
                Expr::column("quantity"),
                Expr::cast(Expr::column("quantity"), DataType::Float64).alias("__cudf_join_residual_left_0"),
            ],
            Arc::clone(&joined_fact_residual_schema),
        ),
    );

    let grouped_fact = builder.add_source(SourceNode::LocalParquet(mixed_aggregate_grouped_fact_source(
        grouped_fact_path,
        grouped_fact_schema,
        &["entity_id", "quantity"],
    )));
    let grouped = builder.add_aggregate(
        grouped_fact,
        AggregateSpec::new(
            vec![Expr::column("entity_id")],
            vec![AggregateExpr::new(AggregateFunction::Avg, Expr::column("quantity"), "avg_quantity")],
        ),
        Arc::clone(&grouped_schema),
    );
    let grouped = builder.add_projection(
        grouped,
        projection_outputs(vec![Expr::column("entity_id"), Expr::column("avg_quantity")], Arc::clone(&grouped_schema)),
    );

    let root = builder.add_join(
        joined_projection,
        grouped,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                cudf_query_engine::ColumnRef::input(0, "entity_id", "entity_id"),
                cudf_query_engine::ColumnRef::input(1, "entity_id", "entity_id"),
            )],
        )
        .with_residual_predicate(Expr::binary(
            Expr::input_column(0, "quantity", "quantity"),
            BinaryOp::Lt,
            Expr::input_column(1, "avg_quantity", "avg_quantity"),
        ))
        .with_output_projection(vec![JoinOutputColumn::left(
            cudf_query_engine::ColumnRef::input(0, "measure", "measure"),
            "measure",
        )])
        .with_build_side(JoinSide::Right),
        Arc::clone(&root_output_schema),
    );
    let aggregate_input = builder.add_projection(
        root,
        projection_outputs(vec![Expr::column("measure").alias("aggregate_arg")], Arc::clone(&aggregate_input_schema)),
    );
    let aggregate = builder.add_aggregate(
        aggregate_input,
        AggregateSpec::new(
            Vec::new(),
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("aggregate_arg"), "sum_measure")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let output = builder.add_projection(
        aggregate,
        projection_outputs(vec![Expr::column("sum_measure").alias("result")], output_schema),
    );
    builder.finish_with_host_sink(output)
}

fn mixed_aggregate_fact_source(path: &Path, schema: &Arc<Schema>, columns: &[&str]) -> LocalParquetSource {
    LocalParquetSource::from_files(
        vec![LocalParquetFile::new(path).with_file_size_bytes(fs::metadata(path).unwrap().len())],
        Arc::clone(schema),
        columns.iter().map(|column| (*column).to_string()).collect(),
    )
}

fn mixed_aggregate_grouped_fact_source(path: &Path, schema: &Arc<Schema>, columns: &[&str]) -> LocalParquetSource {
    LocalParquetSource::from_files(
        vec![LocalParquetFile::new(path)],
        Arc::clone(schema),
        columns.iter().map(|column| (*column).to_string()).collect(),
    )
}

fn mixed_aggregate_dim_source(path: &Path, schema: &Arc<Schema>) -> LocalParquetSource {
    LocalParquetSource::from_files(
        vec![LocalParquetFile::new(path).with_file_size_bytes(fs::metadata(path).unwrap().len())],
        Arc::clone(schema),
        vec!["entity_id".to_string()],
    )
}

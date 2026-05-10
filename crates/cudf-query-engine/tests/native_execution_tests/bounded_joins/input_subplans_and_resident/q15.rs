use super::*;

#[test]
fn bounded_native_query_executes_keyed_hash_join_resident_subplan_pipeline() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let dir = unique_temp_dir("cudf-query-engine-bounded-q15-resident-subplan");
    let supplier_path = dir.join("supplier.parquet");
    let lineitem_path = dir.join("lineitem.parquet");
    let supplier_schema = write_q15_supplier_parquet(&runtime, &supplier_path);
    let lineitem_schema = write_q15_lineitem_parquet(&runtime, &lineitem_path);
    let plan =
        q15_keyed_hash_join_resident_subplan_plan(&supplier_path, &lineitem_path, &supplier_schema, &lineitem_schema);

    assert!(!plan.has_bounded_executable_root_join());
    assert!(!plan.has_bounded_executable_join_pipeline());
    assert!(!plan.has_bounded_executable_keyless_conditional_join_input_subplan_pipeline());
    assert!(!plan.has_bounded_executable_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline());
    assert!(!plan.has_bounded_executable_multi_join_pipeline());
    assert!(!plan.has_bounded_executable_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline());
    assert!(plan.has_bounded_executable_keyed_hash_join_resident_subplan_pipeline());
    assert!(plan.bounded_join_unsupported_shape_facts(ExecutionBoundaryPolicy::default()).is_none());

    let context = bounded_join_execution_context(&runtime);
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.phase_c_memory_control_policy.as_str(), "bounded");
    assert_eq!(metrics.boundary.host_to_device_uploads, 0);
    assert_eq!(metrics.boundary.device_to_host_materializations, 1);
    assert_eq!(metrics.reservation_denied_count, 0);
    assert_eq!(metrics.hash_join_build_count, 2);
    assert_eq!(metrics.operators.iter().filter(|operator| operator.join.is_some()).count(), 2);
    assert!(metrics.reservation_events.iter().any(|event| event.memory_boundary == "final_sort"));
    assert!(metrics.final_sort_time.as_nanos() > 0);
    assert!(metrics.reservation_events.iter().any(|event| event.memory_boundary == "host_materialization"));

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 2);
    assert_eq!(batch.num_columns(), 3);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2]);
    assert_eq!(string_array(batch, 1).value(0), "Supplier#1");
    assert_eq!(string_array(batch, 1).value(1), "Supplier#2");
    assert_eq!(int64_array(batch, 2).values().as_ref(), &[120, 120]);

    fs::remove_dir_all(dir).unwrap();
}

fn write_q15_supplier_parquet(runtime: &cudf::Runtime, path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("s_suppkey", DataType::Int32, false),
        Field::new("s_name", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec!["Supplier#1", "Supplier#2", "Supplier#3"])),
        ],
    )
    .unwrap();
    DataFrame::from_record_batch(runtime, &batch).unwrap().write_parquet(path).unwrap();
    schema
}

fn write_q15_lineitem_parquet(runtime: &cudf::Runtime, path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("l_suppkey", DataType::Int32, false),
        Field::new("l_shipdate", DataType::Int32, false),
        Field::new("l_extendedprice", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 1, 2, 3])),
            Arc::new(Int32Array::from(vec![1, 2, 3, 4])),
            Arc::new(Int64Array::from(vec![50, 70, 120, 10])),
        ],
    )
    .unwrap();
    DataFrame::from_record_batch(runtime, &batch).unwrap().write_parquet(path).unwrap();
    schema
}

fn q15_keyed_hash_join_resident_subplan_plan(
    supplier_path: &Path,
    lineitem_path: &Path,
    supplier_schema: &Arc<Schema>,
    lineitem_schema: &Arc<Schema>,
) -> cudf_query_engine::QueryPlan {
    let revenue_projection_schema = Arc::new(Schema::new(vec![
        Field::new("supplier_no", DataType::Int32, false),
        Field::new("revenue", DataType::Int64, false),
    ]));
    let revenue_schema = Arc::new(Schema::new(vec![
        Field::new("supplier_no", DataType::Int32, false),
        Field::new("total_revenue", DataType::Int64, true),
    ]));
    let scalar_revenue_input_schema = Arc::new(Schema::new(vec![Field::new("total_revenue", DataType::Int64, true)]));
    let supplier_revenue_schema = Arc::new(Schema::new(vec![
        Field::new("s_suppkey", DataType::Int32, false),
        Field::new("s_name", DataType::Utf8, false),
        Field::new("total_revenue", DataType::Int64, true),
    ]));
    let max_revenue_schema = Arc::new(Schema::new(vec![Field::new("max_revenue", DataType::Int64, true)]));

    let mut builder = QueryPlanBuilder::new();
    let supplier = builder.add_source(SourceNode::LocalParquet(join_left_source(
        supplier_path,
        supplier_schema,
        &["s_suppkey", "s_name"],
    )));
    let revenue = build_q15_revenue_branch(
        &mut builder,
        lineitem_path,
        lineitem_schema,
        Arc::clone(&revenue_projection_schema),
        Arc::clone(&revenue_schema),
    );
    let supplier_revenue = builder.add_join(
        supplier,
        revenue,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                cudf_query_engine::ColumnRef::input(0, "s_suppkey", "s_suppkey"),
                cudf_query_engine::ColumnRef::input(1, "supplier_no", "supplier_no"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "s_suppkey", "s_suppkey"), "s_suppkey"),
            JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "s_name", "s_name"), "s_name"),
            JoinOutputColumn::right(
                cudf_query_engine::ColumnRef::input(1, "total_revenue", "total_revenue"),
                "total_revenue",
            ),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&supplier_revenue_schema),
    );
    let supplier_revenue = builder.add_projection(
        supplier_revenue,
        projection_outputs(
            vec![Expr::column("s_suppkey"), Expr::column("s_name"), Expr::column("total_revenue")],
            Arc::clone(&supplier_revenue_schema),
        ),
    );

    let revenue_for_max = build_q15_revenue_aggregate(
        &mut builder,
        lineitem_path,
        lineitem_schema,
        revenue_projection_schema,
        Arc::clone(&revenue_schema),
    );
    let revenue_for_max = builder.add_projection(
        revenue_for_max,
        projection_outputs(vec![Expr::column("total_revenue")], scalar_revenue_input_schema),
    );
    let max_revenue = builder.add_aggregate(
        revenue_for_max,
        AggregateSpec::new(
            Vec::new(),
            vec![AggregateExpr::new(AggregateFunction::Max, Expr::column("total_revenue"), "max_revenue")],
        ),
        Arc::clone(&max_revenue_schema),
    );
    let root = builder.add_join(
        supplier_revenue,
        max_revenue,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                cudf_query_engine::ColumnRef::input(0, "total_revenue", "total_revenue"),
                cudf_query_engine::ColumnRef::input(1, "max_revenue", "max_revenue"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "s_suppkey", "s_suppkey"), "s_suppkey"),
            JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "s_name", "s_name"), "s_name"),
            JoinOutputColumn::left(
                cudf_query_engine::ColumnRef::input(0, "total_revenue", "total_revenue"),
                "total_revenue",
            ),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&supplier_revenue_schema),
    );
    let output = builder.add_sort(
        root,
        SortSpec::new(vec![SortExpr::new(Expr::column("s_suppkey"), false, false)]),
        supplier_revenue_schema,
    );
    builder.finish_with_host_sink(output)
}

fn build_q15_revenue_branch(
    builder: &mut QueryPlanBuilder,
    lineitem_path: &Path,
    lineitem_schema: &Arc<Schema>,
    revenue_projection_schema: Arc<Schema>,
    revenue_schema: Arc<Schema>,
) -> cudf_query_engine::NodeId {
    let aggregate = build_q15_revenue_aggregate(
        builder,
        lineitem_path,
        lineitem_schema,
        revenue_projection_schema,
        Arc::clone(&revenue_schema),
    );
    builder.add_projection(
        aggregate,
        projection_outputs(vec![Expr::column("supplier_no"), Expr::column("total_revenue")], revenue_schema),
    )
}

fn build_q15_revenue_aggregate(
    builder: &mut QueryPlanBuilder,
    lineitem_path: &Path,
    lineitem_schema: &Arc<Schema>,
    revenue_projection_schema: Arc<Schema>,
    revenue_schema: Arc<Schema>,
) -> cudf_query_engine::NodeId {
    let lineitem = builder.add_source(SourceNode::LocalParquet(join_left_source(
        lineitem_path,
        lineitem_schema,
        &["l_suppkey", "l_shipdate", "l_extendedprice"],
    )));
    let filtered = builder.add_filter(
        lineitem,
        Expr::binary(Expr::column("l_shipdate"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int32(0))),
        Arc::clone(lineitem_schema),
    );
    let projected = builder.add_projection(
        filtered,
        projection_outputs(
            vec![Expr::column("l_suppkey").alias("supplier_no"), Expr::column("l_extendedprice").alias("revenue")],
            revenue_projection_schema,
        ),
    );
    builder.add_aggregate(
        projected,
        AggregateSpec::new(
            vec![Expr::column("supplier_no")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("revenue"), "total_revenue")],
        ),
        revenue_schema,
    )
}

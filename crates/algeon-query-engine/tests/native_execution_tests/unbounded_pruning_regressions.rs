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
    let filtered_schema = Arc::clone(&schema);
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("median_m00", DataType::Float64, true),
    ]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let filtered = builder.add_filter(
        source,
        Expr::binary(Expr::column("payload"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int32(10))),
        filtered_schema,
    );
    let aggregate = builder.add_aggregate(
        filtered,
        AggregateSpec::new(
            vec![Expr::column("key")],
            vec![AggregateExpr::new(AggregateFunction::Median, Expr::column("m00"), "median_m00")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let sorted = builder.add_sort(
        aggregate,
        SortSpec::new(vec![
            SortExpr::new(Expr::column("median_m00"), true, false),
            SortExpr::new(Expr::column("key"), false, false),
        ]),
        aggregate_schema,
    );
    let plan = builder.finish_with_host_sink(sorted);

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
        let aggregate = builder.add_aggregate(
            source,
            AggregateSpec::new(
                Vec::new(),
                vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("key"), "count_key")],
            ),
            count_schema,
        );
        let result = execute_query_plan(&context, &builder.finish_with_host_sink(aggregate)).unwrap();
        let QueryOutput::HostArrow(batch) = result.output() else { panic!("COUNT must materialize to Arrow") };
        assert_eq!(batch.num_rows(), 1);
        assert_eq!(int64_array(batch, 0).value(0), 0);

        let probe = LocalParquetSource::count_star_probe(vec![file], Arc::clone(&schema), vec!["key".to_string()])
            .with_column_roles(Vec::new(), vec!["key".to_string()]);
        let count_star_schema = Arc::new(Schema::new(vec![Field::new("count_rows", DataType::Int64, true)]));
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(probe));
        let aggregate = builder.add_aggregate(
            source,
            AggregateSpec::new(
                Vec::new(),
                vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column(ZERO_COLUMN_COUNT_COLUMN), "count_rows")],
            ),
            count_star_schema,
        );
        let result = execute_query_plan(&context, &builder.finish_with_host_sink(aggregate)).unwrap();
        let QueryOutput::HostArrow(batch) = result.output() else { panic!("COUNT(*) must materialize to Arrow") };
        assert_eq!(batch.num_rows(), 1);
        assert_eq!(int64_array(batch, 0).value(0), 0);
    }

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn unbounded_tpch_q3_multi_join_aggregate_topk_preserves_pruning_regression() {
    let dir = unique_temp_dir("query-engine-unbounded-tpch-q3");
    let customer_path = dir.join("customer.parquet");
    let orders_path = dir.join("orders.parquet");
    let lineitem_path = dir.join("lineitem.parquet");
    let customer_schema = Arc::new(Schema::new(vec![
        Field::new("c_custkey", DataType::Int32, false),
        Field::new("c_mktsegment", DataType::Utf8, false),
    ]));
    let orders_schema = Arc::new(Schema::new(vec![
        Field::new("o_orderkey", DataType::Int32, false),
        Field::new("o_custkey", DataType::Int32, false),
        Field::new("o_orderdate", DataType::Int32, false),
        Field::new("o_shippriority", DataType::Int32, false),
    ]));
    let lineitem_schema = Arc::new(Schema::new(vec![
        Field::new("l_orderkey", DataType::Int32, false),
        Field::new("l_extendedprice", DataType::Int64, false),
    ]));
    write_test_parquet(
        &customer_path,
        Arc::clone(&customer_schema),
        vec![Arc::new(Int32Array::from(vec![1, 2])), Arc::new(StringArray::from(vec!["BUILDING", "AUTOMOBILE"]))],
    );
    write_test_parquet(
        &orders_path,
        Arc::clone(&orders_schema),
        vec![
            Arc::new(Int32Array::from(vec![10, 11])),
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(Int32Array::from(vec![0, 0])),
        ],
    );
    write_test_parquet(
        &lineitem_path,
        Arc::clone(&lineitem_schema),
        vec![Arc::new(Int32Array::from(vec![10, 10, 11])), Arc::new(Int64Array::from(vec![100, 50, 200]))],
    );

    let customer_orders_schema = Arc::new(Schema::new(vec![
        Field::new("o_orderkey", DataType::Int32, false),
        Field::new("o_orderdate", DataType::Int32, false),
        Field::new("o_shippriority", DataType::Int32, false),
    ]));
    let joined_schema = Arc::new(Schema::new(vec![
        Field::new("o_orderkey", DataType::Int32, false),
        Field::new("o_orderdate", DataType::Int32, false),
        Field::new("o_shippriority", DataType::Int32, false),
        Field::new("l_extendedprice", DataType::Int64, false),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("o_orderkey", DataType::Int32, false),
        Field::new("o_orderdate", DataType::Int32, false),
        Field::new("o_shippriority", DataType::Int32, false),
        Field::new("revenue", DataType::Int64, true),
    ]));
    let mut builder = QueryPlanBuilder::new();
    let customer = builder.add_source(SourceNode::LocalParquet(join_left_source(
        &customer_path,
        &customer_schema,
        &["c_custkey", "c_mktsegment"],
    )));
    let customer = builder.add_filter(
        customer,
        Expr::binary(
            Expr::column("c_mktsegment"),
            BinaryOp::Eq,
            Expr::literal(LiteralValue::Utf8("BUILDING".to_string())),
        ),
        customer_schema,
    );
    let orders = builder.add_source(SourceNode::LocalParquet(join_right_source(
        &orders_path,
        &orders_schema,
        &["o_orderkey", "o_custkey", "o_orderdate", "o_shippriority"],
    )));
    let customer_orders = builder.add_join(
        customer,
        orders,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                query_engine::expr::ColumnRef::input(0, "c_custkey", "c_custkey"),
                query_engine::expr::ColumnRef::input(1, "o_custkey", "o_custkey"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::right(query_engine::expr::ColumnRef::input(1, "o_orderkey", "o_orderkey"), "o_orderkey"),
            JoinOutputColumn::right(
                query_engine::expr::ColumnRef::input(1, "o_orderdate", "o_orderdate"),
                "o_orderdate",
            ),
            JoinOutputColumn::right(
                query_engine::expr::ColumnRef::input(1, "o_shippriority", "o_shippriority"),
                "o_shippriority",
            ),
        ])
        .with_build_side(JoinSide::Left),
        customer_orders_schema,
    );
    let lineitem = builder.add_source(SourceNode::LocalParquet(join_right_source(
        &lineitem_path,
        &lineitem_schema,
        &["l_orderkey", "l_extendedprice"],
    )));
    let joined = builder.add_join(
        customer_orders,
        lineitem,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                query_engine::expr::ColumnRef::input(0, "o_orderkey", "o_orderkey"),
                query_engine::expr::ColumnRef::input(1, "l_orderkey", "l_orderkey"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(query_engine::expr::ColumnRef::input(0, "o_orderkey", "o_orderkey"), "o_orderkey"),
            JoinOutputColumn::left(
                query_engine::expr::ColumnRef::input(0, "o_orderdate", "o_orderdate"),
                "o_orderdate",
            ),
            JoinOutputColumn::left(
                query_engine::expr::ColumnRef::input(0, "o_shippriority", "o_shippriority"),
                "o_shippriority",
            ),
            JoinOutputColumn::right(
                query_engine::expr::ColumnRef::input(1, "l_extendedprice", "l_extendedprice"),
                "l_extendedprice",
            ),
        ])
        .with_build_side(JoinSide::Left),
        joined_schema,
    );
    let aggregate = builder.add_aggregate(
        joined,
        AggregateSpec::new(
            vec![Expr::column("o_orderkey"), Expr::column("o_orderdate"), Expr::column("o_shippriority")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("l_extendedprice"), "revenue")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let sorted = builder.add_sort(
        aggregate,
        SortSpec::new(vec![
            SortExpr::new(Expr::column("revenue"), true, false),
            SortExpr::new(Expr::column("o_orderdate"), false, false),
        ]),
        Arc::clone(&aggregate_schema),
    );
    let limited = builder.add_limit(sorted, LimitSpec::fetch(10), aggregate_schema);

    let result = execute_query_plan(&admitted_execution_context(), &builder.finish_with_host_sink(limited)).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else { panic!("Q3 must materialize to Arrow") };
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(int32_array(batch, 0).value(0), 10);
    assert_eq!(int64_array(batch, 3).value(0), 150);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn unbounded_tpch_q19_join_residual_in_list_preserves_pruning_regression() {
    let dir = unique_temp_dir("query-engine-unbounded-tpch-q19");
    let lineitem_path = dir.join("lineitem.parquet");
    let part_path = dir.join("part.parquet");
    let lineitem_schema = Arc::new(Schema::new(vec![
        Field::new("l_partkey", DataType::Int32, false),
        Field::new("l_quantity", DataType::Float64, false),
        Field::new("l_extendedprice", DataType::Int64, false),
        Field::new("l_shipmode", DataType::Utf8, false),
        Field::new("l_shipinstruct", DataType::Utf8, false),
    ]));
    let part_schema = Arc::new(Schema::new(vec![
        Field::new("p_partkey", DataType::Int32, false),
        Field::new("p_brand", DataType::Utf8, false),
        Field::new("p_container", DataType::Utf8, false),
        Field::new("p_size", DataType::Int32, false),
    ]));
    write_test_parquet(
        &lineitem_path,
        Arc::clone(&lineitem_schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 1, 2, 3])),
            Arc::new(Float64Array::from(vec![5.0, 20.0, 15.0, 5.0])),
            Arc::new(Int64Array::from(vec![100, 400, 200, 800])),
            Arc::new(StringArray::from(vec!["AIR", "AIR", "AIR REG", "AIR"])),
            Arc::new(StringArray::from(vec![
                "DELIVER IN PERSON",
                "DELIVER IN PERSON",
                "DELIVER IN PERSON",
                "DELIVER IN PERSON",
            ])),
        ],
    );
    write_test_parquet(
        &part_path,
        Arc::clone(&part_schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec!["Brand#12", "Brand#23", "Brand#99"])),
            Arc::new(StringArray::from(vec!["SM BOX", "MED BAG", "LG BOX"])),
            Arc::new(Int32Array::from(vec![3, 8, 3])),
        ],
    );
    let joined_schema = Arc::new(Schema::new(vec![Field::new("l_extendedprice", DataType::Int64, false)]));
    let aggregate_schema = Arc::new(Schema::new(vec![Field::new("revenue", DataType::Int64, true)]));
    let mut builder = QueryPlanBuilder::new();
    let lineitem = builder.add_source(SourceNode::LocalParquet(join_left_source(
        &lineitem_path,
        &lineitem_schema,
        &["l_partkey", "l_quantity", "l_extendedprice", "l_shipmode", "l_shipinstruct"],
    )));
    let part = builder.add_source(SourceNode::LocalParquet(join_right_source(
        &part_path,
        &part_schema,
        &["p_partkey", "p_brand", "p_container", "p_size"],
    )));
    let joined = builder.add_join(
        lineitem,
        part,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                query_engine::expr::ColumnRef::input(0, "l_partkey", "l_partkey"),
                query_engine::expr::ColumnRef::input(1, "p_partkey", "p_partkey"),
            )],
        )
        .with_residual_predicate(q19_residual_predicate())
        .with_output_projection(vec![JoinOutputColumn::left(
            query_engine::expr::ColumnRef::input(0, "l_extendedprice", "l_extendedprice"),
            "l_extendedprice",
        )])
        .with_build_side(JoinSide::Right),
        joined_schema,
    );
    let aggregate = builder.add_aggregate(
        joined,
        AggregateSpec::new(
            Vec::new(),
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("l_extendedprice"), "revenue")],
        ),
        aggregate_schema,
    );

    let result = execute_query_plan(&admitted_execution_context(), &builder.finish_with_host_sink(aggregate)).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else { panic!("Q19 must materialize to Arrow") };
    assert_eq!(int64_array(batch, 0).value(0), 300);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn unbounded_tpch_q15_revenue_subplan_root_join_preserves_pruning_regression() {
    let dir = unique_temp_dir("query-engine-unbounded-tpch-q15");
    let supplier_path = dir.join("supplier.parquet");
    let lineitem_path = dir.join("lineitem.parquet");
    let supplier_schema = Arc::new(Schema::new(vec![
        Field::new("s_suppkey", DataType::Int32, false),
        Field::new("s_name", DataType::Utf8, false),
    ]));
    let lineitem_schema = Arc::new(Schema::new(vec![
        Field::new("l_suppkey", DataType::Int32, false),
        Field::new("l_extendedprice", DataType::Int64, false),
    ]));
    write_test_parquet(
        &supplier_path,
        Arc::clone(&supplier_schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec!["Supplier#1", "Supplier#2", "Supplier#3"])),
        ],
    );
    write_test_parquet(
        &lineitem_path,
        Arc::clone(&lineitem_schema),
        vec![Arc::new(Int32Array::from(vec![1, 1, 2, 3])), Arc::new(Int64Array::from(vec![50, 70, 120, 10]))],
    );

    let revenue_projection_schema = Arc::new(Schema::new(vec![
        Field::new("supplier_no", DataType::Int32, false),
        Field::new("revenue", DataType::Int64, false),
    ]));
    let revenue_schema = Arc::new(Schema::new(vec![
        Field::new("supplier_no", DataType::Int32, false),
        Field::new("total_revenue", DataType::Int64, true),
    ]));
    let supplier_revenue_schema = Arc::new(Schema::new(vec![
        Field::new("s_suppkey", DataType::Int32, false),
        Field::new("s_name", DataType::Utf8, false),
        Field::new("total_revenue", DataType::Int64, true),
    ]));
    let max_revenue_schema = Arc::new(Schema::new(vec![Field::new("max_revenue", DataType::Int64, true)]));
    let mut builder = QueryPlanBuilder::new();
    let lineitem = builder.add_source(SourceNode::LocalParquet(join_left_source(
        &lineitem_path,
        &lineitem_schema,
        &["l_suppkey", "l_extendedprice"],
    )));
    let projected = builder.add_projection(
        lineitem,
        projection_outputs(
            vec![Expr::column("l_suppkey").alias("supplier_no"), Expr::column("l_extendedprice").alias("revenue")],
            revenue_projection_schema,
        ),
    );
    let revenue = builder.add_aggregate(
        projected,
        AggregateSpec::new(
            vec![Expr::column("supplier_no")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("revenue"), "total_revenue")],
        ),
        Arc::clone(&revenue_schema),
    );
    let supplier = builder.add_source(SourceNode::LocalParquet(join_left_source(
        &supplier_path,
        &supplier_schema,
        &["s_suppkey", "s_name"],
    )));
    let revenue_for_supplier = builder.add_projection(
        revenue,
        projection_outputs(
            vec![Expr::column("supplier_no"), Expr::column("total_revenue")],
            Arc::clone(&revenue_schema),
        ),
    );
    let supplier_revenue = builder.add_join(
        supplier,
        revenue_for_supplier,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                query_engine::expr::ColumnRef::input(0, "s_suppkey", "s_suppkey"),
                query_engine::expr::ColumnRef::input(1, "supplier_no", "supplier_no"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(query_engine::expr::ColumnRef::input(0, "s_suppkey", "s_suppkey"), "s_suppkey"),
            JoinOutputColumn::left(query_engine::expr::ColumnRef::input(0, "s_name", "s_name"), "s_name"),
            JoinOutputColumn::right(
                query_engine::expr::ColumnRef::input(1, "total_revenue", "total_revenue"),
                "total_revenue",
            ),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&supplier_revenue_schema),
    );
    let revenue_for_max = builder.add_projection(
        revenue,
        projection_outputs(
            vec![Expr::column("total_revenue")],
            Arc::new(Schema::new(vec![Field::new("total_revenue", DataType::Int64, true)])),
        ),
    );
    let max_revenue = builder.add_aggregate(
        revenue_for_max,
        AggregateSpec::new(
            Vec::new(),
            vec![AggregateExpr::new(AggregateFunction::Max, Expr::column("total_revenue"), "max_revenue")],
        ),
        max_revenue_schema,
    );
    let maximum_suppliers = builder.add_join(
        supplier_revenue,
        max_revenue,
        JoinSpec::new(JoinType::Inner, Vec::new())
            .with_residual_predicate(Expr::binary(
                Expr::input_column(0, "total_revenue", "total_revenue"),
                BinaryOp::Eq,
                Expr::input_column(1, "max_revenue", "max_revenue"),
            ))
            .with_output_projection(vec![
                JoinOutputColumn::left(query_engine::expr::ColumnRef::input(0, "s_suppkey", "s_suppkey"), "s_suppkey"),
                JoinOutputColumn::left(query_engine::expr::ColumnRef::input(0, "s_name", "s_name"), "s_name"),
                JoinOutputColumn::left(
                    query_engine::expr::ColumnRef::input(0, "total_revenue", "total_revenue"),
                    "total_revenue",
                ),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&supplier_revenue_schema),
    );
    let sorted = builder.add_sort(
        maximum_suppliers,
        SortSpec::new(vec![SortExpr::new(Expr::column("s_suppkey"), false, false)]),
        supplier_revenue_schema,
    );

    let plan = builder.finish_with_host_sink(sorted);
    assert_eq!(
        plan.capability_analysis(query_engine::rapids_backend::rapids_analysis_context())
            .root_full_sort_join_aggregate_admission,
        query_engine::capability::CapabilityDecision::Satisfied,
        "Q15 shared grouped-aggregate keyless root must satisfy root full-sort admission",
    );
    let result = execute_query_plan(&admitted_execution_context(), &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else { panic!("Q15 must materialize to Arrow") };
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2]);
    assert_eq!(int64_array(batch, 2).values().as_ref(), &[120, 120]);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn unbounded_tpch_q16_distinct_anti_join_preserves_pruning_regression() {
    let fixture_runtime = backend_runtime();
    let dir = unique_temp_dir("query-engine-unbounded-tpch-q16");
    let part_path = dir.join("part.parquet");
    let partsupp_path = dir.join("partsupp.parquet");
    let complaints_path = dir.join("complaints.parquet");
    let part_schema = write_part_distinct_count_part_parquet(&fixture_runtime, &part_path);
    let partsupp_schema = write_part_distinct_count_partsupp_parquet(&fixture_runtime, &partsupp_path);
    let complaints_schema = write_part_distinct_count_supplier_parquet(&fixture_runtime, &complaints_path);
    drop(fixture_runtime);

    let anti_schema = Arc::new(Schema::new(vec![
        Field::new("ps_partkey", DataType::Int32, false),
        Field::new("ps_suppkey", DataType::Int32, false),
    ]));
    let joined_schema = Arc::new(Schema::new(vec![
        Field::new("p_brand", DataType::Utf8, false),
        Field::new("p_type", DataType::Utf8, false),
        Field::new("p_size", DataType::Int32, false),
        Field::new("ps_suppkey", DataType::Int32, false),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("p_brand", DataType::Utf8, false),
        Field::new("p_type", DataType::Utf8, false),
        Field::new("p_size", DataType::Int32, false),
        Field::new("supplier_cnt", DataType::Int64, false),
    ]));
    let mut builder = QueryPlanBuilder::new();
    let partsupp = builder.add_source(SourceNode::LocalParquet(join_left_source(
        &partsupp_path,
        &partsupp_schema,
        &["ps_partkey", "ps_suppkey"],
    )));
    let complaints = builder.add_source(SourceNode::LocalParquet(join_right_source(
        &complaints_path,
        &complaints_schema,
        &["s_suppkey"],
    )));
    let allowed = builder.add_join(
        partsupp,
        complaints,
        JoinSpec::new(
            JoinType::LeftAnti,
            vec![JoinKey::new(
                query_engine::expr::ColumnRef::input(0, "ps_suppkey", "ps_suppkey"),
                query_engine::expr::ColumnRef::input(1, "s_suppkey", "s_suppkey"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(query_engine::expr::ColumnRef::input(0, "ps_partkey", "ps_partkey"), "ps_partkey"),
            JoinOutputColumn::left(query_engine::expr::ColumnRef::input(0, "ps_suppkey", "ps_suppkey"), "ps_suppkey"),
        ])
        .with_build_side(JoinSide::Right),
        anti_schema,
    );
    let part = builder.add_source(SourceNode::LocalParquet(join_right_source(
        &part_path,
        &part_schema,
        &["p_partkey", "p_brand", "p_type", "p_size"],
    )));
    let joined = builder.add_join(
        allowed,
        part,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                query_engine::expr::ColumnRef::input(0, "ps_partkey", "ps_partkey"),
                query_engine::expr::ColumnRef::input(1, "p_partkey", "p_partkey"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::right(query_engine::expr::ColumnRef::input(1, "p_brand", "p_brand"), "p_brand"),
            JoinOutputColumn::right(query_engine::expr::ColumnRef::input(1, "p_type", "p_type"), "p_type"),
            JoinOutputColumn::right(query_engine::expr::ColumnRef::input(1, "p_size", "p_size"), "p_size"),
            JoinOutputColumn::left(query_engine::expr::ColumnRef::input(0, "ps_suppkey", "ps_suppkey"), "ps_suppkey"),
        ])
        .with_build_side(JoinSide::Right),
        joined_schema,
    );
    let aggregate = builder.add_aggregate(
        joined,
        AggregateSpec::new(
            vec![Expr::column("p_brand"), Expr::column("p_type"), Expr::column("p_size")],
            vec![AggregateExpr::new(AggregateFunction::NUnique, Expr::column("ps_suppkey"), "supplier_cnt")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let sorted = builder.add_sort(
        aggregate,
        SortSpec::new(vec![
            SortExpr::new(Expr::column("supplier_cnt"), true, false),
            SortExpr::new(Expr::column("p_brand"), false, false),
        ]),
        aggregate_schema,
    );

    let result = execute_query_plan(&admitted_execution_context(), &builder.finish_with_host_sink(sorted)).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else { panic!("Q16 must materialize to Arrow") };
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(string_array(batch, 0).value(0), "Brand#11");
    assert_eq!(int64_array(batch, 3).value(0), 1);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn unbounded_tpch_q11_nested_having_subquery_preserves_pruning_regression() {
    let fixture_runtime = backend_runtime();
    let dir = unique_temp_dir("query-engine-unbounded-tpch-q11");
    let grouped_fact_path = dir.join("grouped-partsupp.parquet");
    let scalar_fact_path = dir.join("scalar-partsupp.parquet");
    let supplier_path = dir.join("supplier.parquet");
    let grouped_fact_schema = write_supplier_cost_partsupp_parquet(
        &fixture_runtime,
        &grouped_fact_path,
        &[1, 1, 2, 3, 4],
        &[10, 20, 10, 30, 99],
        &[10, 5, 25, 3, 100],
    );
    let scalar_fact_schema =
        write_supplier_cost_partsupp_parquet(&fixture_runtime, &scalar_fact_path, &[8, 9], &[10, 20], &[7, 5]);
    let supplier_schema = write_supplier_cost_supplier_parquet(&fixture_runtime, &supplier_path);
    drop(fixture_runtime);
    let plan = keyless_conditional_aggregate_over_join_subplan_inputs_plan(
        &grouped_fact_path,
        &scalar_fact_path,
        &supplier_path,
        &grouped_fact_schema,
        &scalar_fact_schema,
        &supplier_schema,
    );

    let result = execute_query_plan(&admitted_execution_context(), &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else { panic!("Q11 must materialize to Arrow") };
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[2, 1]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[25, 15]);
    fs::remove_dir_all(dir).unwrap();
}

fn write_test_parquet(path: &Path, schema: Arc<Schema>, columns: Vec<Arc<dyn Array>>) {
    let batch = RecordBatch::try_new(Arc::clone(&schema), columns).unwrap();
    write_record_batch_with_row_group_rows(path, schema, batch, 2);
}

fn q19_residual_predicate() -> Expr {
    let branch = |brand: &str, containers: &[&str], quantity_min: f64, quantity_max: f64, size_max: i32| {
        and_all(vec![
            Expr::binary(
                Expr::input_column(1, "p_brand", "p_brand"),
                BinaryOp::Eq,
                Expr::literal(LiteralValue::Utf8(brand.to_string())),
            ),
            Expr::in_list(
                Expr::input_column(1, "p_container", "p_container"),
                containers.iter().map(|value| Expr::literal(LiteralValue::Utf8((*value).to_string()))).collect(),
            ),
            Expr::binary(
                Expr::input_column(0, "l_quantity", "l_quantity"),
                BinaryOp::GtEq,
                Expr::literal(LiteralValue::Float64(quantity_min)),
            ),
            Expr::binary(
                Expr::input_column(0, "l_quantity", "l_quantity"),
                BinaryOp::LtEq,
                Expr::literal(LiteralValue::Float64(quantity_max)),
            ),
            Expr::binary(
                Expr::input_column(1, "p_size", "p_size"),
                BinaryOp::LtEq,
                Expr::literal(LiteralValue::Int32(size_max)),
            ),
            Expr::in_list(
                Expr::input_column(0, "l_shipmode", "l_shipmode"),
                vec![
                    Expr::literal(LiteralValue::Utf8("AIR".to_string())),
                    Expr::literal(LiteralValue::Utf8("AIR REG".to_string())),
                ],
            ),
            Expr::binary(
                Expr::input_column(0, "l_shipinstruct", "l_shipinstruct"),
                BinaryOp::Eq,
                Expr::literal(LiteralValue::Utf8("DELIVER IN PERSON".to_string())),
            ),
        ])
    };
    Expr::binary(
        branch("Brand#12", &["SM CASE", "SM BOX", "SM PACK", "SM PKG"], 1.0, 11.0, 5),
        BinaryOp::Or,
        branch("Brand#23", &["MED BAG", "MED BOX", "MED PKG", "MED PACK"], 10.0, 20.0, 10),
    )
}

fn and_all(mut predicates: Vec<Expr>) -> Expr {
    let first = predicates.remove(0);
    predicates.into_iter().fold(first, |left, right| Expr::binary(left, BinaryOp::And, right))
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

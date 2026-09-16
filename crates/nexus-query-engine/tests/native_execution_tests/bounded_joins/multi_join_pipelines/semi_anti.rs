use std::sync::Arc;

use super::*;

#[test]
fn bounded_native_query_executes_mixed_left_semi_left_anti_multi_join_pipeline_with_ordered_reservations() {
    let runtime = backend_runtime();
    let dir = unique_temp_dir("nexus-query-engine-bounded-mixed-semi-anti-multi-join");
    let left_path = dir.join("lineitem.parquet");
    let semi_path = dir.join("semi_orders.parquet");
    let anti_path = dir.join("anti_orders.parquet");
    let left_schema = write_join_left_parquet(&runtime, &left_path);
    let semi_schema = write_join_right_parquet(&runtime, &semi_path);
    let anti_schema = write_join_right_keys_parquet(&runtime, &anti_path, &[3, 5]);
    let left_only_join_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let lineitem =
        builder.add_source(SourceNode::LocalParquet(join_left_source(&left_path, &left_schema, &["order_id", "qty"])));
    let semi_orders =
        builder.add_source(SourceNode::LocalParquet(join_right_source(&semi_path, &semi_schema, &["order_id"])));
    let anti_orders =
        builder.add_source(SourceNode::LocalParquet(join_right_source(&anti_path, &anti_schema, &["order_id"])));
    let semi_join = builder.add_join(
        lineitem,
        semi_orders,
        JoinSpec::new(JoinType::LeftSemi, vec![join_order_id_key()])
            .with_residual_predicate(qty_gt_right_order_id_times_ten())
            .with_output_projection(vec![
                JoinOutputColumn::left(
                    nexus_query_engine::expr::ColumnRef::input(0, "order_id", "order_id"),
                    "order_id",
                ),
                JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "qty", "qty"), "qty"),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&left_only_join_schema),
    );
    let anti_join = builder.add_join(
        semi_join,
        anti_orders,
        JoinSpec::new(JoinType::LeftAnti, vec![join_order_id_key()])
            .with_residual_predicate(qty_gt_right_order_id_times_ten())
            .with_output_projection(vec![
                JoinOutputColumn::left(
                    nexus_query_engine::expr::ColumnRef::input(0, "order_id", "order_id"),
                    "order_id",
                ),
                JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "qty", "qty"), "qty"),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&left_only_join_schema),
    );
    let plan = builder.finish_with_host_sink(anti_join);
    assert!(!plan.capability_analysis().admits_root_join());
    assert!(nexus_query_engine::admission::bounded_join_unsupported_shape_facts(&plan).is_none());

    let context = bounded_join_execution_context(&runtime);
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.memory_control_policy.as_str(), "bounded");
    assert_eq!(metrics.parquet_files, 3);
    assert_eq!(metrics.hash_join_build_count, 0);
    assert_eq!(metrics.hash_join_hash_table_bytes, 0);
    assert!(metrics.mixed_join_size_data_bytes > 0);
    assert_eq!(metrics.operators.iter().filter(|operator| operator.join.is_some()).count(), 2);
    let join_metrics = metrics.operators.iter().filter_map(|operator| operator.join.as_ref()).collect::<Vec<_>>();
    assert_eq!(join_metrics[0].execution_strategy, "bounded_mixed_keyed_residual_join");
    assert_eq!(join_metrics[0].join_type, "left_semi");
    assert_eq!(join_metrics[0].build_side, "right");
    assert_eq!(join_metrics[0].probe_side, "left");
    assert_eq!(join_metrics[0].key_count, 1);
    assert!(join_metrics[0].residual_predicate);
    assert_eq!(join_metrics[0].hash_table_bytes, 0);
    assert_eq!(join_metrics[0].output_rows, 1);
    assert!(join_metrics[0].mixed_size_data_bytes > 0);
    assert!(join_metrics[0].gather_map_bytes > 0);
    assert_eq!(join_metrics[1].execution_strategy, "bounded_mixed_keyed_residual_join");
    assert_eq!(join_metrics[1].join_type, "left_anti");
    assert_eq!(join_metrics[1].build_side, "right");
    assert_eq!(join_metrics[1].probe_side, "left");
    assert_eq!(join_metrics[1].key_count, 1);
    assert!(join_metrics[1].residual_predicate);
    assert_eq!(join_metrics[1].hash_table_bytes, 0);
    assert_eq!(join_metrics[1].output_rows, 1);
    assert!(join_metrics[1].mixed_size_data_bytes > 0);
    assert!(join_metrics[1].gather_map_bytes > 0);
    assert_eq!(
        join_metrics[0].mixed_size_data_bytes + join_metrics[1].mixed_size_data_bytes,
        metrics.mixed_join_size_data_bytes
    );

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(batch.num_columns(), 2);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[2]);
    assert_eq!(int32_array(batch, 1).values().as_ref(), &[25]);

    fs::remove_dir_all(dir).unwrap();
}

// TPC-H Q21 shape: one filtered, projected lineitem leaf feeds both the inner
// probe against supplier and the build side of the NOT EXISTS anti join.
const SHARED_LEAF_ROWS: i32 = 2_000_000;
const SHARED_LEAF_ROW_GROUP_ROWS: usize = 62_500;
const SHARED_LEAF_SUPPLIERS: i32 = 1_000;
const SHARED_LEAF_MATCHED_SUPPLIERS: i32 = 900;
const SHARED_LEAF_NATIONS: i32 = 7;
// Between the peak of streaming the leaf into each join consumer and the peak
// of materializing all four source columns once for both; see the plan doc.
const SHARED_LEAF_GRANT_BYTES: u64 = 40 * 1024 * 1024;

struct SharedLeafDetail {
    order_id: Option<i32>,
    supp: Option<i32>,
    late: bool,
}

// Four rows per order with distinct suppliers, one in three late; null order
// keys never match the anti build and null suppliers never match supplier.
fn shared_leaf_detail(row: i32) -> SharedLeafDetail {
    SharedLeafDetail {
        order_id: (row % 100_003 != 0).then_some(row / 4),
        supp: (row % 50_000 != 0).then_some(row % SHARED_LEAF_SUPPLIERS),
        late: row % 3 == 0,
    }
}

fn write_shared_leaf_detail_parquet(path: &Path, rows: i32) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, true),
        Field::new("supp", DataType::Int32, true),
        Field::new("commit", DataType::Int32, false),
        Field::new("receipt", DataType::Int32, false),
    ]));
    let details = (0..rows).map(shared_leaf_detail).collect::<Vec<_>>();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(details.iter().map(|detail| detail.order_id).collect::<Vec<_>>())),
            Arc::new(Int32Array::from(details.iter().map(|detail| detail.supp).collect::<Vec<_>>())),
            Arc::new(Int32Array::from((0..rows).collect::<Vec<_>>())),
            Arc::new(Int32Array::from(
                details
                    .iter()
                    .enumerate()
                    .map(|(row, detail)| row as i32 + if detail.late { 1 } else { -1 })
                    .collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap();
    write_record_batch_with_row_group_rows(path, Arc::clone(&schema), batch, SHARED_LEAF_ROW_GROUP_ROWS);
    schema
}

fn write_shared_leaf_supplier_parquet(path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("supp_id", DataType::Int32, false),
        Field::new("nation", DataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from((0..SHARED_LEAF_MATCHED_SUPPLIERS).collect::<Vec<_>>())),
            Arc::new(Int32Array::from(
                (0..SHARED_LEAF_MATCHED_SUPPLIERS).map(|supp| supp % SHARED_LEAF_NATIONS).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap();
    write_record_batch_with_row_group_rows(path, Arc::clone(&schema), batch, SHARED_LEAF_ROW_GROUP_ROWS);
    schema
}

fn expected_shared_leaf_late_counts(rows: i32) -> Vec<(i32, i64)> {
    let late = (0..rows).map(shared_leaf_detail).filter(|detail| detail.late).collect::<Vec<_>>();
    let mut other_suppliers = std::collections::HashMap::<i32, Vec<i32>>::new();
    for detail in &late {
        if let (Some(order_id), Some(supp)) = (detail.order_id, detail.supp) {
            other_suppliers.entry(order_id).or_default().push(supp);
        }
    }
    let mut counts = vec![0i64; SHARED_LEAF_NATIONS as usize];
    for detail in &late {
        let Some(supp) = detail.supp.filter(|supp| *supp < SHARED_LEAF_MATCHED_SUPPLIERS) else {
            continue;
        };
        let excluded = detail
            .order_id
            .and_then(|order_id| other_suppliers.get(&order_id))
            .is_some_and(|suppliers| suppliers.iter().any(|other| *other != supp));
        if !excluded {
            counts[(supp % SHARED_LEAF_NATIONS) as usize] += 1;
        }
    }
    counts.into_iter().enumerate().map(|(nation, count)| (nation as i32, count)).collect()
}

fn shared_leaf_inner_probe_anti_build_plan(
    detail_path: &Path,
    detail_schema: &Arc<Schema>,
    supplier_path: &Path,
    supplier_schema: &Arc<Schema>,
) -> QueryPlan {
    let keys_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, true),
        Field::new("supp", DataType::Int32, true),
    ]));
    let inner_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, true),
        Field::new("supp", DataType::Int32, true),
        Field::new("nation", DataType::Int32, true),
    ]));
    let anti_schema = Arc::new(Schema::new(vec![Field::new("nation", DataType::Int32, true)]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("nation", DataType::Int32, true),
        Field::new("late_count", DataType::Int64, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let detail = builder.add_source(SourceNode::LocalParquet(join_left_source(
        detail_path,
        detail_schema,
        &["order_id", "supp", "commit", "receipt"],
    )));
    let late = builder.add_filter(
        detail,
        Expr::binary(Expr::column("receipt"), BinaryOp::Gt, Expr::column("commit")),
        Arc::clone(detail_schema),
    );
    let keys = builder.add_projection(
        late,
        projection_outputs(vec![Expr::column("order_id"), Expr::column("supp")], Arc::clone(&keys_schema)),
    );
    let supplier = builder.add_source(SourceNode::LocalParquet(join_right_source(
        supplier_path,
        supplier_schema,
        &["supp_id", "nation"],
    )));
    let inner = builder.add_join(
        supplier,
        keys,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                nexus_query_engine::expr::ColumnRef::input(0, "supp_id", "supp_id"),
                nexus_query_engine::expr::ColumnRef::input(1, "supp", "supp"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::right(nexus_query_engine::expr::ColumnRef::input(1, "order_id", "order_id"), "order_id"),
            JoinOutputColumn::right(nexus_query_engine::expr::ColumnRef::input(1, "supp", "supp"), "supp"),
            JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "nation", "nation"), "nation"),
        ])
        .with_build_side(JoinSide::Left),
        inner_schema,
    );
    let anti = builder.add_join(
        inner,
        keys,
        JoinSpec::new(
            JoinType::LeftAnti,
            vec![JoinKey::new(
                nexus_query_engine::expr::ColumnRef::input(0, "order_id", "order_id"),
                nexus_query_engine::expr::ColumnRef::input(1, "order_id", "order_id"),
            )],
        )
        .with_residual_predicate(Expr::binary(
            Expr::input_column(1, "supp", "supp"),
            BinaryOp::NotEq,
            Expr::input_column(0, "supp", "supp"),
        ))
        .with_output_projection(vec![JoinOutputColumn::left(
            nexus_query_engine::expr::ColumnRef::input(0, "nation", "nation"),
            "nation",
        )])
        .with_build_side(JoinSide::Right),
        anti_schema,
    );
    let aggregate = builder.add_aggregate(
        anti,
        AggregateSpec::new(
            vec![Expr::column("nation")],
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("nation"), "late_count")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let sort = builder.add_sort(
        aggregate,
        SortSpec::new(vec![SortExpr::new(Expr::column("nation"), false, false)]),
        aggregate_schema,
    );
    builder.finish_with_host_sink(sort)
}

#[test]
fn bounded_native_query_streams_a_shared_staged_leaf_into_each_join_consumer() {
    let dir = unique_temp_dir("nexus-query-engine-shared-staged-leaf-join-consumers");
    let detail_path = dir.join("detail.parquet");
    let supplier_path = dir.join("supplier.parquet");
    let detail_schema = write_shared_leaf_detail_parquet(&detail_path, SHARED_LEAF_ROWS);
    let supplier_schema = write_shared_leaf_supplier_parquet(&supplier_path);
    let plan = shared_leaf_inner_probe_anti_build_plan(&detail_path, &detail_schema, &supplier_path, &supplier_schema);
    assert!(nexus_query_engine::admission::bounded_join_unsupported_shape_facts(&plan).is_none());

    let context = backend_context_with_grant(SHARED_LEAF_GRANT_BYTES);
    let (result, metrics, peak_bytes) = execute_admitted_with_peak(&context, &plan);
    let output = result.unwrap_or_else(|error| panic!("streamed shared leaf should fit the grant: {error:?}"));
    let QueryOutput::HostArrow(batch) = &output else {
        panic!("grouped count should materialize host output");
    };
    let expected = expected_shared_leaf_late_counts(SHARED_LEAF_ROWS);
    let nations = int32_array(batch, 0).iter().map(|nation| nation.unwrap()).collect::<Vec<_>>();
    let counts = int64_array(batch, 1).iter().map(|count| count.unwrap()).collect::<Vec<_>>();
    assert_eq!(nations.into_iter().zip(counts).collect::<Vec<_>>(), expected);
    assert!(expected.iter().all(|(_, count)| *count > 0));
    assert!(peak_bytes < SHARED_LEAF_GRANT_BYTES, "peak={peak_bytes}");
    let detail_chunks = (SHARED_LEAF_ROWS as usize).div_ceil(SHARED_LEAF_ROW_GROUP_ROWS) as u64;
    assert!(
        metrics.source_chunks_committed >= 2 * detail_chunks,
        "both join consumers re-stream the leaf instead of sharing one frame: {}",
        metrics.source_chunks_committed
    );
    assert_eq!(metrics.source_chunks_retried, 0);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_fails_typed_when_a_shared_leaf_consumer_cannot_shrink_below_the_cap() {
    let dir = unique_temp_dir("nexus-query-engine-shared-staged-leaf-typed-cap-failure");
    let detail_path = dir.join("detail.parquet");
    let supplier_path = dir.join("supplier.parquet");
    let detail_schema = write_shared_leaf_detail_parquet(&detail_path, SHARED_LEAF_ROWS);
    let supplier_schema = write_shared_leaf_supplier_parquet(&supplier_path);
    let plan = shared_leaf_inner_probe_anti_build_plan(&detail_path, &detail_schema, &supplier_path, &supplier_schema);

    // Single leaf chunks fit, but no split makes the anti build side or the
    // probe output fit, so the cap rejection surfaces with its typed cause.
    let context = backend_context_with_grant(3 * 1024 * 1024_u64);
    let admitted = context.admit_plan(&plan).unwrap();
    let outcome = execute_query_plan_direct(
        admitted.context.context(),
        QueryExecutionRequest::new(&plan).with_gpu_inputs(&GpuResidentInputMap::new()),
    );
    let (result, metrics, _observation) = outcome.into_parts();
    let error = result.err().expect("the shared leaf consumers cannot fit under the cap");
    assert_eq!(error.code(), QueryErrorCode::QueryMemoryLimitExceeded, "{error:?}");
    assert_eq!(error.fact_value("failure_source"), Some("RESERVATION_LIMIT"));
    assert!(metrics.retry_resource_exhausted_count >= 1, "cap rejections before the floor are recorded");
    assert_attempt_domain_released(admitted.context.context());

    admitted.finish();
    drop(admitted);
    wait_for_ledger_release(context.policy);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_releases_a_shared_leaf_stream_when_cancelled_mid_stream() {
    let dir = unique_temp_dir("nexus-query-engine-shared-staged-leaf-cancel");
    let detail_path = dir.join("detail.parquet");
    let supplier_path = dir.join("supplier.parquet");
    let rows = 4 * SHARED_LEAF_ROWS;
    let detail_schema = write_shared_leaf_detail_parquet(&detail_path, rows);
    let supplier_schema = write_shared_leaf_supplier_parquet(&supplier_path);
    let plan = shared_leaf_inner_probe_anti_build_plan(&detail_path, &detail_schema, &supplier_path, &supplier_schema);
    // Four times the rows need a proportionally larger cap; this test only
    // exercises cancellation cleanup, not the memory bound.
    let grant_bytes = 4 * SHARED_LEAF_GRANT_BYTES;
    let context = backend_context_with_grant(grant_bytes);
    // Warm the reader and kernels so the deadline below elapses while leaf
    // chunks are streaming rather than during first-use initialization.
    let warm = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = warm.output() else {
        panic!("grouped count should materialize host output");
    };
    assert_eq!(batch.num_rows(), SHARED_LEAF_NATIONS as usize);

    let cancellation = nexus_query_engine::runtime::ExecutionCancellation::with_deadline(Some(
        std::time::Instant::now() + std::time::Duration::from_millis(40),
    ));
    let evidence = capability_evidence_for_fixture_plan(&plan).unwrap();
    let handle = suite_backend_with_policy(context.policy)
        .session_capability()
        .admit_query_with_request(
            QueryAdmissionRequest::new()
                .with_query_plan_capability_evidence(evidence)
                .with_query_cap_bytes(grant_bytes),
        )
        .unwrap();
    let admitted = handle.acquire_context_with_cancellation(cancellation).unwrap();
    let outcome = execute_query_plan_direct(
        admitted.context(),
        QueryExecutionRequest::new(&plan).with_gpu_inputs(&GpuResidentInputMap::new()),
    );
    let (result, metrics, _observation) = outcome.into_parts();
    let error = result.err().expect("the deadline elapses while the leaf streams");
    assert_eq!(error.code(), QueryErrorCode::Cancelled, "{error:?}");
    assert_eq!(error.fact_value("reason"), Some("deadline_expired"));
    let detail_chunks = (rows as usize).div_ceil(SHARED_LEAF_ROW_GROUP_ROWS) as u64;
    assert!(metrics.source_chunks_committed < 2 * detail_chunks, "cancellation must land mid-stream: {metrics:?}");
    assert_attempt_domain_released(admitted.context());

    drop(admitted);
    handle.finish();
    drop(handle);
    wait_for_ledger_release(context.policy);
    fs::remove_dir_all(dir).unwrap();
}

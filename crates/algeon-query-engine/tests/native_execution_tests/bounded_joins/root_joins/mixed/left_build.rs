use std::sync::Arc;

use query_planning::expr::ColumnRef;

use super::*;

fn left_match_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, true),
        Field::new("qty", DataType::Int32, true),
        Field::new("payload", DataType::Utf8, true),
    ]))
}

fn left_match_plan(
    left_path: &Path,
    right_path: &Path,
    join_type: JoinType,
    null_equality: JoinNullEquality,
) -> QueryPlan {
    let schema = left_match_schema();
    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(join_left_source(
        left_path,
        &schema,
        &["order_id", "qty", "payload"],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(join_right_source(
        right_path,
        &schema,
        &["order_id", "qty", "payload"],
    )));
    let predicate = Expr::binary(
        Expr::binary(Expr::input_column(0, "qty", "left_qty"), BinaryOp::Gt, Expr::input_column(1, "qty", "right_qty")),
        BinaryOp::And,
        Expr::binary(
            Expr::input_column(0, "payload", "left_payload"),
            BinaryOp::Eq,
            Expr::input_column(1, "payload", "right_payload"),
        ),
    );
    let join = builder
        .add_join(
            left,
            right,
            JoinSpec::new(join_type, vec![join_order_id_key()])
                .with_residual_predicate(predicate)
                .with_null_equality(null_equality)
                .with_build_side(JoinSide::Left)
                .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty")]),
            Arc::new(Schema::new(vec![Field::new("qty", DataType::Int32, true)])),
        )
        .unwrap();
    builder.finish_with_host_sink(join).unwrap()
}

fn write_left_match_batch(
    path: &Path,
    keys: Vec<Option<i32>>,
    quantities: Vec<Option<i32>>,
    payloads: Vec<Option<String>>,
    rows_per_group: usize,
) {
    let schema = left_match_schema();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(keys)),
            Arc::new(Int32Array::from(quantities)),
            Arc::new(StringArray::from(payloads)),
        ],
    )
    .unwrap();
    write_record_batch_with_row_group_rows(path, schema, batch, rows_per_group);
}

#[test]
fn bounded_native_mixed_left_build_preserves_identity_and_finalizes_after_all_probe_batches() {
    let dir = unique_temp_dir("query-engine-mixed-left-build-batches");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    let empty_path = dir.join("empty.parquet");
    let payload = "a".repeat(8192);
    write_left_match_batch(
        &left_path,
        vec![Some(1), Some(1), Some(2), Some(3), None, Some(4), Some(5)],
        vec![Some(10), Some(10), Some(20), Some(30), Some(40), None, Some(50)],
        vec![
            Some(payload.clone()),
            Some(payload.clone()),
            Some(payload.clone()),
            Some(payload.clone()),
            Some(payload.clone()),
            Some(payload.clone()),
            None,
        ],
        8,
    );
    write_left_match_batch(&empty_path, vec![], vec![], vec![], 1);
    let rows = 64 * 128;
    let mut keys = vec![Some(99); rows];
    let mut quantities = vec![Some(5); rows];
    let mut payloads = vec![Some(payload); rows];
    // Every match arrives in the final row group, after a probe volume larger than the grant.
    keys[rows - 7..].copy_from_slice(&[Some(1), Some(1), Some(2), Some(3), None, Some(4), Some(5)]);
    quantities[rows - 4] = None;
    payloads[rows - 5] = Some("b".repeat(8192));
    payloads[rows - 1] = None;
    write_left_match_batch(&right_path, keys, quantities, payloads, 128);

    const CAP: u64 = 16 * 1024 * 1024;
    for null_equality in [JoinNullEquality::NullsDoNotMatch, JoinNullEquality::NullsMatch] {
        for join_type in [JoinType::LeftSemi, JoinType::LEFT_ANTI] {
            for (left, right) in [(&left_path, &right_path), (&left_path, &empty_path), (&empty_path, &right_path)] {
                let plan = left_match_plan(left, right, join_type.clone(), null_equality);
                let context = backend_context_with_grant(CAP);
                let admitted = context.admit_plan(&plan).unwrap();
                let admitted_plan = admit_query_plan(
                    &plan,
                    admitted.statistical_aggregate_execution_mode(),
                    admitted.planning_policy(),
                )
                .unwrap();
                let outcome = execute_query_plan_direct(
                    admitted.context.context(),
                    QueryExecutionRequest::admitted(&admitted_plan).with_gpu_inputs(&GpuResidentInputMap::new()),
                );
                let (result, metrics, _) = outcome.into_parts();
                let metrics = metrics.snapshot();
                let output = result.unwrap_or_else(|error| panic!("{join_type:?}/{null_equality:?}: {error:?}"));
                let QueryOutput::HostArrow(batch) = &output else {
                    panic!("expected host output");
                };
                let mut observed = int32_array(batch, 0).iter().collect::<Vec<_>>();
                observed.sort();
                let mut expected = if left == &empty_path {
                    vec![]
                } else if right == &empty_path {
                    if join_type == JoinType::LeftSemi {
                        vec![]
                    } else {
                        vec![Some(10), Some(10), Some(20), Some(30), Some(40), None, Some(50)]
                    }
                } else if join_type == JoinType::LeftSemi {
                    if null_equality == JoinNullEquality::NullsMatch {
                        vec![Some(10), Some(10), Some(40)]
                    } else {
                        vec![Some(10), Some(10)]
                    }
                } else if null_equality == JoinNullEquality::NullsMatch {
                    vec![Some(20), Some(30), None, Some(50)]
                } else {
                    vec![Some(20), Some(30), Some(40), None, Some(50)]
                };
                expected.sort();
                assert_eq!(observed, expected, "{join_type:?}/{null_equality:?}");
                if left != &empty_path && right != &empty_path {
                    let join = metrics.operators.iter().find_map(|operator| operator.join.as_ref()).unwrap();
                    assert_eq!(join.execution_strategy, "bounded_mixed_left_match_join");
                    assert!(join.probe_bytes > CAP, "the streamed probe must exceed the grant");
                    assert_eq!(join.probe_rows, rows as u64);
                    assert!(metrics.source_chunks_committed > 2);
                }
                let usage = admitted.attempt_allocation_domain_usage().unwrap().unwrap();
                assert_eq!(usage.current_bytes, 0, "completed join retained GPU state: {usage:?}");
                assert_eq!(usage.current_allocations, 0);
                admitted.finish();
            }
        }
    }
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_mixed_left_build_releases_state_after_late_probe_cap_failure() {
    let dir = unique_temp_dir("query-engine-mixed-left-build-probe-failure");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    let payload = "a".repeat(8192);
    write_left_match_batch(&left_path, vec![Some(1)], vec![Some(10)], vec![Some(payload.clone())], 1);
    write_left_match_batch(
        &right_path,
        vec![Some(99), Some(99), Some(1)],
        vec![Some(5); 3],
        vec![Some(payload.clone()), Some(payload), Some("a".repeat(20 * 1024 * 1024))],
        1,
    );
    let plan = left_match_plan(&left_path, &right_path, JoinType::LEFT_ANTI, JoinNullEquality::NullsDoNotMatch);
    let context = backend_context_with_grant(8 * 1024 * 1024);
    let admitted = context.admit_plan(&plan).unwrap();
    let admitted_plan =
        admit_query_plan(&plan, admitted.statistical_aggregate_execution_mode(), admitted.planning_policy()).unwrap();
    let outcome = execute_query_plan_direct(
        admitted.context.context(),
        QueryExecutionRequest::admitted(&admitted_plan).with_gpu_inputs(&GpuResidentInputMap::new()),
    );
    let (result, metrics, _) = outcome.into_parts();
    let metrics = metrics.snapshot();
    let error = result.err().expect("the indivisible final probe row must exceed the cap");
    assert_eq!(error.code(), QueryErrorCode::QueryMemoryLimitExceeded, "{error:?}");
    assert_eq!(error.fact_value("build_side"), Some("left"));
    assert!(metrics.source_chunks_committed >= 3, "earlier probe batches must have been consumed");
    let usage = admitted.attempt_allocation_domain_usage().unwrap().unwrap();
    assert_eq!(usage.current_bytes, 0, "failed probe retained its left match state: {usage:?}");
    assert_eq!(usage.current_allocations, 0);
    admitted.finish();
    fs::remove_dir_all(dir).unwrap();
}

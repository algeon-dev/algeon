use super::*;

fn partitioned_probe_retry_plan(runtime: &cudf::Runtime, dir: &Path, rows_per_chunk: usize) -> QueryPlan {
    let probe_path = dir.join("probe.parquet");
    let build_path = dir.join("build.parquet");
    let probe_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&probe_schema),
        vec![
            Arc::new(Int32Array::from(vec![2; rows_per_chunk * 2])),
            Arc::new(Int32Array::from_iter_values(0..i32::try_from(rows_per_chunk * 2).unwrap())),
        ],
    )
    .unwrap();
    write_record_batch_with_row_group_rows(&probe_path, Arc::clone(&probe_schema), batch, rows_per_chunk);
    let build_schema = write_join_right_parquet(runtime, &build_path);
    let mut builder = QueryPlanBuilder::new();
    let probe = builder.add_source(SourceNode::LocalParquet(join_left_source(
        &probe_path,
        &probe_schema,
        &["order_id", "qty"],
    )));
    let build =
        builder.add_source(SourceNode::LocalParquet(join_right_source(&build_path, &build_schema, &["order_id"])));
    let join = builder
        .add_join(
            probe,
            build,
            JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
                .with_build_side(JoinSide::Right)
                .with_output_projection(vec![
                    JoinOutputColumn::left(ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                    JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
                ]),
            Arc::clone(&probe_schema),
        )
        .unwrap();
    let sorted = builder.add_sort(join, SortSpec::new(vec![SortExpr::new(Expr::column("qty"), true, false)])).unwrap();
    let limited = builder.add_limit(sorted, LimitSpec::fetch(2)).unwrap();
    builder.finish_with_host_sink(limited).unwrap()
}

fn partitioned_probe_context(cap: u64, min_units: usize) -> BackendQueryContext {
    backend_context_with_memory_control_and_planning_policy(
        partitioned_probe_retry_policy(cap, min_units),
        PlanningPolicy {
            keyed_hash_join: KeyedHashJoinStrategyOptions::default()
                .with_small_reusable_hash_build_row_threshold(1)
                .with_large_keyed_hash_build_row_threshold(1),
            ..PlanningPolicy::default()
        },
    )
}

fn partitioned_probe_retry_policy(cap: u64, min_units: usize) -> MemoryControlPolicy {
    MemoryControlPolicy {
        device_budget_bytes: cap,
        max_source_chunk_bytes: 64 * 1024 * 1024,
        max_row_groups_per_chunk: 1,
        max_retry_attempts: 12,
        min_retry_chunk_row_groups: min_units,
    }
}

#[test]
fn bounded_native_partitioned_probe_recovers_hash_partition_cap_rejections() {
    let dir = unique_temp_dir("algeon-partitioned-probe-retry");
    let fixture_context = partitioned_probe_context(128 * 1024 * 1024, 1);
    let plan = partitioned_probe_retry_plan(&fixture_context.runtime().unwrap(), &dir, 1 << 20);
    for (cap_mb, should_retry) in [(128, false), (28, true)] {
        let context = partitioned_probe_context(cap_mb * 1024 * 1024, 1);
        let admitted = context.admit_plan(&plan).unwrap();
        let admitted_plan =
            admit_query_plan(&plan, admitted.statistical_aggregate_execution_mode(), admitted.planning_policy())
                .unwrap();
        let outcome = execute_query_plan_direct(&admitted, QueryExecutionRequest::admitted(&admitted_plan));
        let (result, metrics, _) = outcome.into_parts();
        let metrics = metrics.snapshot();
        let output = result.unwrap();
        let batch = output.as_host_arrow().unwrap();
        assert_eq!(int32_array(batch, 1).values().as_ref(), &[(1 << 21) - 1, (1 << 21) - 2]);
        assert_eq!(metrics.source_chunks_retried, 0, "the source reader must fit without shrinking");
        assert_eq!(metrics.source_chunks_committed, 3);
        let join = metrics.operators.iter().find_map(|operator| operator.join.as_ref()).unwrap();
        assert_eq!(join.probe_rows, 1 << 21);
        assert_eq!(join.output_rows, 1 << 21, "retry must neither duplicate nor omit probe rows");
        let decision = metrics
            .join_strategy_decisions
            .iter()
            .find(|decision| decision.facts.contains_key("probe_partition_batch_count"))
            .unwrap();
        let partitions: usize = decision.facts["partition_count"].parse().unwrap();
        let probe_batches: usize = decision.facts["probe_partition_batch_count"].parse().unwrap();
        if should_retry {
            assert!(metrics.retry_resource_exhausted_count > 0);
            assert!(probe_batches > partitions, "hash-partition input chunks must be subdivided");
        } else {
            assert_eq!(metrics.retry_resource_exhausted_count, 0);
            assert_eq!(probe_batches, partitions);
        }
        let usage = admitted.attempt_allocation_domain_usage().unwrap().unwrap();
        assert_eq!(usage.current_bytes, 0);
        assert_eq!(usage.current_allocations, 0);
        admitted.finish();
    }
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_partitioned_probe_reports_indivisible_cap_failure_and_cleans_up() {
    let dir = unique_temp_dir("algeon-partitioned-probe-minimum");
    let context = partitioned_probe_context(28 * 1024 * 1024, 2048);
    let plan = partitioned_probe_retry_plan(&context.runtime().unwrap(), &dir, 1 << 20);
    let admitted = context.admit_plan(&plan).unwrap();
    let admitted_plan =
        admit_query_plan(&plan, admitted.statistical_aggregate_execution_mode(), admitted.planning_policy()).unwrap();
    let outcome = execute_query_plan_direct(&admitted, QueryExecutionRequest::admitted(&admitted_plan));
    let (result, metrics, _) = outcome.into_parts();
    let metrics = metrics.snapshot();
    let error = result.err().expect("the configured probe row floor prevents subdivision");
    assert_eq!(error.code(), QueryErrorCode::QueryMemoryLimitExceeded, "{error:?}");
    assert_eq!(error.operation(), "query_engine.bounded.join.hash_partition", "{error:?}");
    assert_eq!(error.fact_value("retry_action"), Some("probe_partition_min_rows_reached"), "{error:?}");
    assert_eq!(error.wire_fact("resume_boundary").as_deref(), Some("partitioned_join_probe"));
    assert_eq!(metrics.source_chunks_retried, 0);
    assert_eq!(metrics.retry_resource_exhausted_count, 1);
    let usage = admitted.attempt_allocation_domain_usage().unwrap().unwrap();
    assert_eq!(usage.current_bytes, 0, "{usage:?}");
    assert_eq!(usage.current_allocations, 0, "{usage:?}");
    admitted.finish();
    fs::remove_dir_all(dir).unwrap();
}

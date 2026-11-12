use query_runtime::metrics::JoinOperatorMetrics;

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
        // Each build partition's table is built at the first probe piece and
        // serves every later piece, however many pieces the retries produce.
        assert_eq!(metrics.hash_join_build_count, partitions as u64, "{metrics:#?}");
        assert_eq!(join.partition_hash_table_build_count, partitions as u64);
        assert_eq!(join.hash_table_bytes, metrics.hash_join_hash_table_bytes);
        assert!(join.hash_table_bytes > 0);
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

const RETAINED_SHARE_BUILD_ROWS: usize = 2_000_000;
const RETAINED_SHARE_PROBE_ROWS_PER_CHUNK: usize = 50_000;
const RETAINED_SHARE_PROBE_CHUNKS: usize = 4;

/// How the partitioned join receives its probe pieces and hands on its output.
#[derive(Clone, Copy, Debug)]
enum RetainedShareShape {
    /// The build is read in one chunk, and the join reads the probe leaf chunk by chunk.
    Leaf,
    /// A left-semi join that keeps every probe row hands the join one resident frame per probe
    /// chunk, and the coordinator spreads those chunks over execution lanes.
    LaneChunks,
    /// A filter over the probe leaf makes the probe a lazy frame stream, which a grouped aggregate
    /// pulls one piece at a time.
    LazyIntoAggregate,
}

// A narrow build with many distinct keys makes its partition hash tables large
// next to the rest of the working set, so a grant can be sized around them. The
// probe streams in one-row-group chunks whose keys each match one build row.
fn retained_share_plan(dir: &Path, shape: RetainedShareShape) -> QueryPlan {
    let build_path = dir.join("build.parquet");
    let probe_path = dir.join("probe.parquet");
    let build_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
    ]));
    let build_keys = 0..i32::try_from(RETAINED_SHARE_BUILD_ROWS).unwrap();
    let build = RecordBatch::try_new(
        Arc::clone(&build_schema),
        vec![
            Arc::new(Int32Array::from_iter_values(build_keys.clone())),
            Arc::new(Int64Array::from_iter_values(build_keys.map(|key| i64::from(key) * 3))),
        ],
    )
    .unwrap();
    // Only a build read in one chunk lets the join read the probe leaf itself. A build read in
    // several chunks is partitioned as it streams in, and the probe then reaches the join as a
    // frame stream.
    let build_row_group_rows = match shape {
        RetainedShareShape::Leaf => RETAINED_SHARE_BUILD_ROWS,
        RetainedShareShape::LaneChunks | RetainedShareShape::LazyIntoAggregate => RETAINED_SHARE_BUILD_ROWS / 4,
    };
    write_record_batch_with_row_group_rows(&build_path, Arc::clone(&build_schema), build, build_row_group_rows);
    let probe_schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("w", DataType::Int64, false)]));
    let probe_rows = RETAINED_SHARE_PROBE_ROWS_PER_CHUNK * RETAINED_SHARE_PROBE_CHUNKS;
    let probe = RecordBatch::try_new(
        Arc::clone(&probe_schema),
        vec![
            Arc::new(Int32Array::from_iter_values((0..probe_rows).map(retained_share_probe_key))),
            Arc::new(Int64Array::from_iter_values((0..probe_rows).map(|row| row as i64))),
        ],
    )
    .unwrap();
    write_record_batch_with_row_group_rows(
        &probe_path,
        Arc::clone(&probe_schema),
        probe,
        RETAINED_SHARE_PROBE_ROWS_PER_CHUNK,
    );

    let mut builder = QueryPlanBuilder::new();
    let probe_leaf =
        builder.add_source(SourceNode::LocalParquet(join_left_source(&probe_path, &probe_schema, &["k", "w"])));
    let probe = match shape {
        RetainedShareShape::Leaf => probe_leaf,
        RetainedShareShape::LaneChunks => {
            let every_w_schema = Arc::new(Schema::new(vec![Field::new("w", DataType::Int64, false)]));
            let every_w = RecordBatch::try_new(
                Arc::clone(&every_w_schema),
                vec![Arc::new(Int64Array::from_iter_values((0..probe_rows).map(|row| row as i64))) as ArrayRef],
            )
            .unwrap();
            let every_w =
                builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(every_w_schema, vec![every_w])));
            builder
                .add_join(
                    probe_leaf,
                    every_w,
                    JoinSpec::new(
                        JoinType::LeftSemi,
                        vec![JoinKey::new(ColumnRef::input(0, "w", "w"), ColumnRef::input(1, "w", "w"))],
                    )
                    .with_output_projection(vec![
                        JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
                        JoinOutputColumn::left(ColumnRef::input(0, "w", "w"), "w"),
                    ])
                    .with_build_side(JoinSide::Right),
                    Arc::clone(&probe_schema),
                )
                .unwrap()
        },
        RetainedShareShape::LazyIntoAggregate => builder
            .add_filter(
                probe_leaf,
                Expr::binary(Expr::column("w"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int64(0))),
            )
            .unwrap(),
    };
    let build =
        builder.add_source(SourceNode::LocalParquet(join_right_source(&build_path, &build_schema, &["k", "payload"])));
    let joined_schema = Arc::new(Schema::new(vec![
        Field::new("w", DataType::Int64, false),
        Field::new("payload", DataType::Int64, false),
    ]));
    let join = builder
        .add_join(
            probe,
            build,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(ColumnRef::input(0, "k", "left_k"), ColumnRef::input(1, "k", "right_k"))],
            )
            .with_build_side(JoinSide::Right)
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "w", "w"), "w"),
                JoinOutputColumn::right(ColumnRef::input(1, "payload", "payload"), "payload"),
            ]),
            Arc::clone(&joined_schema),
        )
        .unwrap();
    let output = if matches!(shape, RetainedShareShape::LazyIntoAggregate) {
        // Every `w` is distinct, so each group holds exactly one joined row.
        builder
            .add_aggregate(
                join,
                AggregateSpec::new(
                    vec![Expr::column("w")],
                    vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("payload"), "payload")],
                ),
                joined_schema,
            )
            .unwrap()
    } else {
        join
    };
    builder.finish_with_host_sink(output).unwrap()
}

// 7919 is prime and shares no factor with the build row count, so distinct
// probe rows take distinct build keys.
fn retained_share_probe_key(row: usize) -> i32 {
    i32::try_from(row * 7919 % RETAINED_SHARE_BUILD_ROWS).unwrap()
}

/// Runs the plan under `grant` and checks its output against the fixture: every
/// probe row with the payload of its build row. Returns the metrics, the
/// partitioned join's metric row, the partition count and the number of probe
/// pieces.
fn run_retained_share_query(plan: &QueryPlan, grant: u64) -> (QueryMetrics, JoinOperatorMetrics, usize, usize) {
    // A 1 MiB read target keeps each 600 KiB probe row group its own piece.
    let context = backend_context_with_memory_control_and_planning_policy(
        MemoryControlPolicy { max_source_chunk_bytes: 1024 * 1024, ..partitioned_probe_retry_policy(grant, 1) },
        PlanningPolicy {
            keyed_hash_join: KeyedHashJoinStrategyOptions::default()
                .with_small_reusable_hash_build_row_threshold(1)
                .with_large_keyed_hash_build_row_threshold(1),
            ..PlanningPolicy::default()
        },
    );
    let (result, metrics, _) = execute_admitted_with_peak(&context, plan);
    let output = result.unwrap();
    let batch = output.as_host_arrow().unwrap();
    let probe_rows = RETAINED_SHARE_PROBE_ROWS_PER_CHUNK * RETAINED_SHARE_PROBE_CHUNKS;
    assert_eq!(batch.num_rows(), probe_rows);
    let mut joined = (0..batch.num_rows())
        .map(|row| (int64_array(batch, 0).value(row), int64_array(batch, 1).value(row)))
        .collect::<Vec<_>>();
    joined.sort_unstable();
    let expected =
        (0..probe_rows).map(|row| (row as i64, i64::from(retained_share_probe_key(row)) * 3)).collect::<Vec<_>>();
    assert_eq!(joined, expected);

    let decision = metrics
        .join_strategy_decisions
        .iter()
        .find(|decision| decision.facts.contains_key("probe_partition_batch_count"))
        .unwrap_or_else(|| panic!("the streamed probe reports its partitioned pieces: {metrics:#?}"));
    let partitions: usize = decision.facts["partition_count"].parse().unwrap();
    let probe_batches: usize = decision.facts["probe_partition_batch_count"].parse().unwrap();
    let join = metrics
        .operators
        .iter()
        .filter_map(|operator| operator.join.as_ref())
        .find(|join| join.execution_strategy == "bounded_partitioned_hash_equi_join")
        .unwrap_or_else(|| panic!("the partitioned join records one metric row: {metrics:#?}"))
        .clone();
    (metrics, join, partitions, probe_batches / partitions)
}

fn assert_unkept_partition_builds_once_per_batch(shape: RetainedShareShape) {
    let dir = unique_temp_dir("algeon-partitioned-probe-retained-share");
    let plan = retained_share_plan(&dir, shape);

    // A quarter of a 1 GiB grant holds every partition table.
    let (roomy, roomy_join, partitions, pieces) = run_retained_share_query(&plan, 1024 * 1024 * 1024);
    assert_eq!(pieces, RETAINED_SHARE_PROBE_CHUNKS, "{shape:?}: {roomy:#?}");
    // The partitioned join's own row; the lane shape's semi join builds a table of its own.
    assert_eq!(roomy_join.partition_hash_table_build_count, partitions as u64, "{shape:?}: {roomy:#?}");
    let kept_table_bytes = roomy_join.hash_table_bytes;
    assert!(kept_table_bytes > 0);

    // With a grant of four times the tables' bytes the share equals those
    // bytes. Each table is charged with its key columns as well, so the first
    // partition's table fits and the second's no longer does. The second
    // partition's rows of every probe piece are held, all within one
    // partitioned-join target, and joined as one batch with one table built
    // after the last piece.
    let (squeezed, squeezed_join, squeezed_partitions, squeezed_pieces) =
        run_retained_share_query(&plan, 4 * kept_table_bytes);
    assert_eq!((squeezed_partitions, squeezed_pieces), (partitions, pieces), "{shape:?}");
    assert_eq!(partitions, 2, "the share arithmetic above assumes two partitions");
    assert_eq!(squeezed_join.partition_hash_table_build_count, 2, "{shape:?}: {squeezed:#?}");
    assert!(squeezed_join.hash_table_bytes < kept_table_bytes, "{shape:?}: {squeezed:#?}");
    if matches!(shape, RetainedShareShape::LaneChunks) {
        let parallel = squeezed
            .join_strategy_decisions
            .iter()
            .find(|decision| decision.facts.get("parallel_probe").map(String::as_str) == Some("partitioned"))
            .unwrap_or_else(|| panic!("the probe chunks reach the lanes: {squeezed:#?}"));
        assert_eq!(parallel.facts["parallel_probe_chunks"], pieces.to_string(), "{squeezed:#?}");
    }

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_partitioned_leaf_probe_builds_an_unkept_partition_once_per_deferred_batch() {
    assert_unkept_partition_builds_once_per_batch(RetainedShareShape::Leaf);
}

#[test]
fn bounded_native_partitioned_lane_probe_builds_an_unkept_partition_once_per_deferred_batch() {
    assert_unkept_partition_builds_once_per_batch(RetainedShareShape::LaneChunks);
}

#[test]
fn bounded_native_partitioned_lazy_probe_builds_an_unkept_partition_once_per_deferred_batch() {
    assert_unkept_partition_builds_once_per_batch(RetainedShareShape::LazyIntoAggregate);
}

// Partition-wise consumption of a large join build side: the build input
// streams in chunk by chunk and is hash-partitioned as it arrives, so neither
// a whole-leaf concat nor an upstream join-output concat is ever resident.

use query_engine::capability::KeyedHashJoinStrategyOptions;

const PARTITIONED_BUILD_PAYLOAD_BYTES: usize = 8192;

// Every row group is roughly `rows_per_row_group * 8 KiB` decoded, so a few
// row groups already dominate a query cap of tens of MiB.
fn write_partitioned_build_payload_parquet(
    path: &Path,
    row_groups: usize,
    rows_per_row_group: usize,
    distinct_payloads: bool,
    order_id_for_row_group: impl Fn(usize) -> i32,
) -> Arc<Schema> {
    let total_rows = row_groups * rows_per_row_group;
    let schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
        Field::new("payload", DataType::Utf8, false),
    ]));
    let mut order_ids = Vec::with_capacity(total_rows);
    let mut quantities = Vec::with_capacity(total_rows);
    let mut payloads = Vec::with_capacity(total_rows);
    for row_group in 0..row_groups {
        let order_id = order_id_for_row_group(row_group);
        for row in 0..rows_per_row_group {
            order_ids.push(order_id);
            quantities.push(i32::try_from(row_group * rows_per_row_group + row).unwrap());
            // Distinct payloads defeat dictionary encoding so the encoded size
            // estimate tracks the decoded size; repeated payloads keep the
            // estimate far below the decoded size.
            let stamp = if distinct_payloads { format!("{row_group:04}-{row:06}-") } else { "x".repeat(12) };
            payloads.push(stamp.repeat(PARTITIONED_BUILD_PAYLOAD_BYTES / 12));
        }
    }
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(order_ids)),
            Arc::new(Int32Array::from(quantities)),
            Arc::new(StringArray::from(payloads)),
        ],
    )
    .unwrap();
    write_record_batch_with_row_group_rows(path, Arc::clone(&schema), batch, rows_per_row_group);
    schema
}

fn partitioned_build_join_output_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
        Field::new("payload", DataType::Utf8, false),
        Field::new("category", DataType::Utf8, false),
    ]))
}

fn partitioned_build_join_projection() -> Vec<JoinOutputColumn> {
    vec![
        JoinOutputColumn::left(ColumnRef::input(0, "order_id", "order_id"), "order_id"),
        JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
        JoinOutputColumn::left(ColumnRef::input(0, "payload", "payload"), "payload"),
        JoinOutputColumn::right(ColumnRef::input(1, "category", "category"), "category"),
    ]
}

fn expected_qty_total(batch: &RecordBatch, matched_order_ids: &[i32], rows_per_row_group: usize) -> (usize, i64) {
    let order_ids = int32_array(batch, 0);
    let quantities = int32_array(batch, 1);
    let mut rows = 0;
    let mut total = 0i64;
    for row in 0..batch.num_rows() {
        assert!(matched_order_ids.contains(&order_ids.value(row)), "unexpected order_id {}", order_ids.value(row));
        rows += 1;
        total += i64::from(quantities.value(row));
    }
    let expected_rows = matched_order_ids.len() * rows_per_row_group;
    let expected_total = matched_order_ids
        .iter()
        .map(|order_id| {
            let row_group = usize::try_from(order_id - 1).unwrap();
            (0..rows_per_row_group).map(|row| (row_group * rows_per_row_group + row) as i64).sum::<i64>()
        })
        .sum::<i64>();
    assert_eq!(rows, expected_rows);
    assert_eq!(total, expected_total);
    (rows, total)
}

fn partitioned_build_decision<'a>(metrics: &'a QueryMetrics, node_id: &str) -> &'a JoinStrategyDecisionMetrics {
    metrics
        .join_strategy_decisions
        .iter()
        .find(|decision| {
            decision.node_id == node_id
                && decision.facts.get("build_consumption").map(String::as_str) == Some("partitioned_input_stream")
        })
        .unwrap_or_else(|| panic!("join {node_id} should commit to partition-wise build consumption: {metrics:#?}"))
}

// One 2 MiB row group per chunk under the suite's default 1 MiB chunk policy.
fn build_recovery_policy(device_budget_bytes: u64, max_row_groups_per_chunk: usize) -> MemoryControlPolicy {
    MemoryControlPolicy::Bounded {
        device_budget_bytes,
        max_source_chunk_bytes: 64 * 1024 * 1024,
        max_row_groups_per_chunk,
        max_retry_attempts: 8,
        min_retry_chunk_row_groups: 1,
    }
}

#[test]
fn bounded_native_query_partitions_streamed_leaf_build_without_whole_leaf_concat() {
    let runtime = backend_runtime();
    let dir = unique_temp_dir("query-engine-bounded-partitioned-leaf-build");
    let build_path = dir.join("build.parquet");
    let probe_path = dir.join("probe.parquet");
    let row_groups = 8;
    let rows_per_row_group = 256;
    let build_schema =
        write_partitioned_build_payload_parquet(&build_path, row_groups, rows_per_row_group, true, |row_group| {
            i32::try_from(row_group + 1).unwrap()
        });
    let probe_schema = write_join_right_keys_parquet(&runtime, &probe_path, &[2, 3]);

    // Footer row counts far above the large-build threshold settle the
    // partitioned strategy before the first chunk is read.
    let mut build_source = join_left_source(&build_path, &build_schema, &["order_id", "qty", "payload"]);
    build_source.files[0].row_group_rows = Some(vec![25_000_000; row_groups]);

    let mut builder = QueryPlanBuilder::new();
    let build = builder.add_source(SourceNode::LocalParquet(build_source));
    let probe = builder.add_source(SourceNode::LocalParquet(join_right_source(
        &probe_path,
        &probe_schema,
        &["order_id", "category"],
    )));
    let join = builder.add_join(
        build,
        probe,
        JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
            .with_output_projection(partitioned_build_join_projection())
            .with_build_side(JoinSide::Left),
        partitioned_build_join_output_schema(),
    );
    let plan = builder.finish_with_host_sink(join);
    // The 16 MiB build side streams in eight 2 MiB chunks. Materializing the
    // whole leaf first needs the raw chunks, their concat, and a partitioned
    // copy at once, which this cap cannot hold; partition-wise consumption
    // keeps one chunk, its partitioned copy, and the accumulated partitions.
    let context = backend_context_with_grant(30 * 1024 * 1024);

    let (result, metrics, peak_bytes) = execute_admitted_with_peak(&context, &plan);
    let output = result.unwrap();
    let metrics = &metrics;
    let decision = partitioned_build_decision(metrics, "n2");
    assert_eq!(decision.facts.get("build_chunks_before_commitment").map(String::as_str), Some("0"));
    let partition_count = decision.facts["partition_count"].parse::<usize>().unwrap();
    assert!(partition_count >= 2, "{decision:?}");
    let join_metrics =
        metrics.operators.iter().find_map(|operator| operator.join.as_ref()).expect("join metrics should be recorded");
    assert_eq!(join_metrics.execution_strategy, "bounded_partitioned_hash_equi_join");
    assert_eq!(join_metrics.build_rows, (row_groups * rows_per_row_group) as u64);
    // Whole-leaf materialization holds the build side twice (chunks plus their
    // concat, then the concat plus its partitioned copy); partition-wise
    // consumption never holds a second whole copy.
    assert!(
        peak_bytes < join_metrics.build_bytes * 7 / 4,
        "query domain peak {peak_bytes} must stay below a second whole build copy of {} bytes",
        join_metrics.build_bytes
    );
    assert_eq!(metrics.hash_join_build_count, 0);
    assert!(metrics.source_chunks_committed >= row_groups as u64, "{metrics:#?}");
    assert!(metrics.join_strategy_decisions.iter().any(|decision| {
        decision.node_id == "n2"
            && decision.facts.get("build_input_chunks").map(String::as_str) == Some(&row_groups.to_string())
    }));

    let QueryOutput::HostArrow(batch) = &output else {
        panic!("host sink should materialize one RecordBatch");
    };
    expected_qty_total(batch, &[2, 3], rows_per_row_group);
    let categories = string_array(batch, 3);
    assert!((0..batch.num_rows()).all(|row| categories.value(row).starts_with("category_")));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_commits_to_build_partitions_once_observed_rows_outgrow_a_misestimated_build() {
    let runtime = backend_runtime();
    let dir = unique_temp_dir("query-engine-bounded-partitioned-build-misestimated");
    let build_path = dir.join("build.parquet");
    let probe_path = dir.join("probe.parquet");
    let row_groups = 8;
    let rows_per_row_group = 256;
    let build_schema =
        write_partitioned_build_payload_parquet(&build_path, row_groups, rows_per_row_group, true, |row_group| {
            i32::try_from(row_group + 1).unwrap()
        });
    let probe_schema = write_join_right_keys_parquet(&runtime, &probe_path, &[2, 3]);

    // The footer claims ten rows per row group, far below the large-build
    // threshold, so nothing settles before the chunks are observed; the
    // second 256-row chunk is what crosses the threshold.
    let mut build_source = join_left_source(&build_path, &build_schema, &["order_id", "qty", "payload"]);
    build_source.files[0].row_group_rows = Some(vec![10; row_groups]);

    let mut builder = QueryPlanBuilder::new();
    let build = builder.add_source(SourceNode::LocalParquet(build_source));
    let probe = builder.add_source(SourceNode::LocalParquet(join_right_source(
        &probe_path,
        &probe_schema,
        &["order_id", "category"],
    )));
    let join = builder.add_join(
        build,
        probe,
        JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
            .with_output_projection(partitioned_build_join_projection())
            .with_build_side(JoinSide::Left),
        partitioned_build_join_output_schema(),
    );
    let plan = builder.finish_with_host_sink(join).with_keyed_hash_join_strategy_options(
        KeyedHashJoinStrategyOptions::default()
            .with_small_reusable_hash_build_row_threshold(100)
            .with_large_keyed_hash_build_row_threshold(300),
    );
    let context = backend_context_with_grant(30 * 1024 * 1024);

    let (result, metrics, peak_bytes) = execute_admitted_with_peak(&context, &plan);
    let output = result.unwrap();
    let metrics = &metrics;
    let decision = partitioned_build_decision(metrics, "n2");
    assert_eq!(decision.facts.get("build_chunks_before_commitment").map(String::as_str), Some("2"));
    assert_eq!(decision.facts.get("observed_build_rows").map(String::as_str), Some("512"));
    assert_eq!(decision.facts.get("estimated_build_rows").map(String::as_str), Some("80"));
    let join_metrics =
        metrics.operators.iter().find_map(|operator| operator.join.as_ref()).expect("join metrics should be recorded");
    assert_eq!(join_metrics.execution_strategy, "bounded_partitioned_hash_equi_join");
    assert_eq!(join_metrics.build_rows, (row_groups * rows_per_row_group) as u64);
    assert!(
        peak_bytes < join_metrics.build_bytes * 7 / 4,
        "query domain peak {peak_bytes} must stay below a second whole build copy of {} bytes",
        join_metrics.build_bytes
    );

    let QueryOutput::HostArrow(batch) = &output else {
        panic!("host sink should materialize one RecordBatch");
    };
    expected_qty_total(batch, &[2, 3], rows_per_row_group);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_partitions_upstream_join_output_stream_into_next_build() {
    let runtime = backend_runtime();
    let dir = unique_temp_dir("query-engine-bounded-partitioned-join-fed-build");
    let fact_path = dir.join("fact.parquet");
    let first_dim_path = dir.join("first_dim.parquet");
    let second_dim_path = dir.join("second_dim.parquet");
    let row_groups = 8;
    let rows_per_row_group = 256;
    let fact_schema =
        write_partitioned_build_payload_parquet(&fact_path, row_groups, rows_per_row_group, true, |row_group| {
            i32::try_from(row_group + 1).unwrap()
        });
    let first_dim_schema = write_join_right_keys_parquet(&runtime, &first_dim_path, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let second_dim_schema = write_join_right_keys_parquet(&runtime, &second_dim_path, &[2, 3, 4]);
    let inner_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
        Field::new("payload", DataType::Utf8, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(SourceNode::LocalParquet(join_left_source(
        &fact_path,
        &fact_schema,
        &["order_id", "qty", "payload"],
    )));
    let first_dim = builder.add_source(SourceNode::LocalParquet(join_right_source(
        &first_dim_path,
        &first_dim_schema,
        &["order_id"],
    )));
    let inner = builder.add_join(
        fact,
        first_dim,
        JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
                JoinOutputColumn::left(ColumnRef::input(0, "payload", "payload"), "payload"),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&inner_schema),
    );
    let second_dim = builder.add_source(SourceNode::LocalParquet(join_right_source(
        &second_dim_path,
        &second_dim_schema,
        &["order_id", "category"],
    )));
    let outer = builder.add_join(
        inner,
        second_dim,
        JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
            .with_output_projection(partitioned_build_join_projection())
            .with_build_side(JoinSide::Left),
        partitioned_build_join_output_schema(),
    );
    // The inner join output has no plan estimate; the lazy inner stream
    // carries its probe leaf's encoded size, which settles the outer build
    // as large before its first chunk arrives.
    let plan = builder.finish_with_host_sink(outer).with_keyed_hash_join_strategy_options(
        KeyedHashJoinStrategyOptions::default()
            .with_small_reusable_hash_build_bytes_threshold(1024 * 1024)
            .with_large_keyed_hash_build_bytes_threshold(3 * 1024 * 1024),
    );
    // Eagerly collecting the 16 MiB inner output, concatenating it for the
    // outer build, and partitioning that frame needs three copies at once;
    // the lazy inner output feeds the outer partitions one chunk at a time.
    let context = backend_context_with_grant(30 * 1024 * 1024);

    let (result, metrics, peak_bytes) = execute_admitted_with_peak(&context, &plan);
    let output = result.unwrap();
    let metrics = &metrics;
    let decision = partitioned_build_decision(metrics, "n4");
    assert_eq!(decision.facts.get("build_chunks_before_commitment").map(String::as_str), Some("0"));
    let joins = metrics.operators.iter().filter_map(|operator| operator.join.as_ref()).collect::<Vec<_>>();
    assert!(joins.iter().any(|join| join.execution_strategy.starts_with("bounded_reusable_hash_join")), "{joins:?}");
    let outer_metrics = joins
        .iter()
        .find(|join| join.execution_strategy == "bounded_partitioned_hash_equi_join")
        .expect("outer join should run partition-wise");
    assert_eq!(outer_metrics.build_rows, (row_groups * rows_per_row_group) as u64);
    // Collecting the inner output eagerly and concatenating it for the outer
    // build holds that output twice; streaming it into partitions does not.
    assert!(
        peak_bytes < outer_metrics.build_bytes * 7 / 4,
        "query domain peak {peak_bytes} must stay below a second whole copy of the {} byte inner output",
        outer_metrics.build_bytes
    );
    // The inner join yields its output in gather-map slices, so the outer
    // build consumes several frames rather than one concatenated input.
    assert!(
        metrics.join_strategy_decisions.iter().any(|decision| {
            decision.node_id == "n4"
                && decision
                    .facts
                    .get("build_input_chunks")
                    .and_then(|chunks| chunks.parse::<usize>().ok())
                    .is_some_and(|chunks| chunks > 2)
        }),
        "{:?}",
        metrics.join_strategy_decisions
    );

    let QueryOutput::HostArrow(batch) = &output else {
        panic!("host sink should materialize one RecordBatch");
    };
    expected_qty_total(batch, &[2, 3, 4], rows_per_row_group);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_reports_typed_failure_when_a_skewed_build_partition_cannot_shrink() {
    let runtime = backend_runtime();
    let dir = unique_temp_dir("query-engine-bounded-partitioned-build-skew");
    let build_path = dir.join("build.parquet");
    let probe_path = dir.join("probe.parquet");
    let row_groups = 8;
    let rows_per_row_group = 256;
    // Every build row shares one key, so no partition count can split it.
    let build_schema =
        write_partitioned_build_payload_parquet(&build_path, row_groups, rows_per_row_group, true, |_| 1);
    let probe_schema = write_join_right_keys_parquet(&runtime, &probe_path, &[1, 2]);
    let mut build_source = join_left_source(&build_path, &build_schema, &["order_id", "qty", "payload"]);
    build_source.files[0].row_group_rows = Some(vec![25_000_000; row_groups]);

    let mut builder = QueryPlanBuilder::new();
    let build = builder.add_source(SourceNode::LocalParquet(build_source));
    let probe = builder.add_source(SourceNode::LocalParquet(join_right_source(
        &probe_path,
        &probe_schema,
        &["order_id", "category"],
    )));
    let join = builder.add_join(
        build,
        probe,
        JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
            .with_output_projection(partitioned_build_join_projection())
            .with_build_side(JoinSide::Left),
        partitioned_build_join_output_schema(),
    );
    let plan = builder.finish_with_host_sink(join);
    // The eight 2 MiB chunks accumulate into one 16 MiB partition; the cap
    // holds the accumulation but not the partition concat on top of it.
    let context = backend_context_with_grant(26 * 1024 * 1024);
    let admitted = context.admit_plan(&plan).unwrap();

    let outcome = execute_query_plan_direct(
        admitted.context.context(),
        QueryExecutionRequest::new(&plan).with_gpu_inputs(&GpuResidentInputMap::new()),
    );
    let (result, metrics, _observation) = outcome.into_parts();
    let error = result.err().expect("an indivisible build partition must fail with the cap cause");
    assert_eq!(error.code(), QueryErrorCode::QueryMemoryLimitExceeded, "{error:?}");
    assert_eq!(error.operation(), "query_engine.bounded.join.build_partition_concat", "{error:?}");
    assert_eq!(error.fact_value("build_side"), Some("left"));
    assert_eq!(error.fact_value("partition_rows"), Some((row_groups * rows_per_row_group).to_string().as_str()));
    assert_eq!(error.fact_value("slice_count"), Some(row_groups.to_string().as_str()));
    assert_eq!(error.fact_value("build_chunks"), Some(row_groups.to_string().as_str()));
    assert!(error.fact_value("partition").is_some());
    assert!(error.fact_value("partition_count").is_some());
    assert_eq!(error.fact_value("plan_node_id"), Some("n2"));
    assert_eq!(error.fact_value("pipeline_task_kind"), Some("join_build"));
    assert!(partitioned_build_decision(&metrics, "n2").facts.contains_key("partition_count"));
    let usage = admitted
        .attempt_allocation_domain_usage()
        .expect("allocation-domain usage samples")
        .expect("admitted context owns one allocation domain");
    assert_eq!(usage.current_bytes, 0, "failed build partitions must be released: {usage:?}");
    assert_eq!(usage.current_allocations, 0, "no build partition may outlive the failed build: {usage:?}");

    admitted.finish();
    drop(admitted);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_releases_build_partitions_when_cancelled_during_build_recovery() {
    let row_groups = 32;
    let row_groups_per_chunk = 8;
    let (policy, _) =
        BackendTestPolicy::from_memory_control_policy(build_recovery_policy(14 * 1024 * 1024, row_groups_per_chunk));
    let runtime = backend_runtime_with_policy(policy);
    let dir = unique_temp_dir("query-engine-bounded-partitioned-build-cancel");
    let build_path = dir.join("build.parquet");
    let probe_path = dir.join("probe.parquet");
    let rows_per_row_group = 64;
    let build_schema =
        write_partitioned_build_payload_parquet(&build_path, row_groups, rows_per_row_group, true, |row_group| {
            i32::try_from(row_group % 4 + 1).unwrap()
        });
    let probe_schema = write_join_right_keys_parquet(&runtime, &probe_path, &[1, 2]);
    let mut build_source = join_left_source(&build_path, &build_schema, &["order_id", "qty", "payload"]);
    build_source.files[0].row_group_rows = Some(vec![25_000_000; row_groups]);

    let mut builder = QueryPlanBuilder::new();
    let build = builder.add_source(SourceNode::LocalParquet(build_source));
    let probe = builder.add_source(SourceNode::LocalParquet(join_right_source(
        &probe_path,
        &probe_schema,
        &["order_id", "category"],
    )));
    let join = builder.add_join(
        build,
        probe,
        JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
            .with_output_projection(partitioned_build_join_projection())
            .with_build_side(JoinSide::Left),
        partitioned_build_join_output_schema(),
    );
    let plan = builder.finish_with_host_sink(join);
    // Warm the reader and kernels so the deadline below elapses inside the
    // build chunk recovery tree rather than during first-use initialization.
    let warm =
        backend_context_with_memory_control_policy(build_recovery_policy(128 * 1024 * 1024, row_groups_per_chunk));
    let warm_result = execute_query_plan(&warm, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = warm_result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 2 * 8 * rows_per_row_group);

    // Every 8-row-group chunk is 4 MiB decoded. The first chunk, its
    // partitioned copy, and its slices fit; each later chunk must halve at
    // least once beside the accumulated partitions, with a backoff between
    // halvings, so the deadline elapses inside the recovery tree while build
    // partitions are resident.
    let cancellation = ExecutionCancellation::with_deadline(Some(Instant::now() + Duration::from_millis(150)));
    let context =
        backend_context_with_memory_control_policy(build_recovery_policy(14 * 1024 * 1024, row_groups_per_chunk));
    let evidence = capability_evidence_for_fixture_plan(&plan).unwrap();
    let handle = suite_backend_with_policy(context.policy)
        .session_capability()
        .admit_query_with_request(
            QueryAdmissionRequest::new()
                .with_query_plan_capability_evidence(evidence)
                .with_query_cap_bytes(14 * 1024 * 1024),
        )
        .unwrap();
    let admitted = handle.acquire_context_with_cancellation(cancellation).unwrap();
    let outcome = execute_query_plan_direct(
        admitted.context(),
        QueryExecutionRequest::new(&plan).with_gpu_inputs(&GpuResidentInputMap::new()),
    );
    let (result, metrics, _observation) = outcome.into_parts();
    let error = result.err().expect("the deadline elapses before the build recovery completes");
    assert_eq!(error.code(), QueryErrorCode::Cancelled, "{error:?}");
    assert_eq!(error.fact_value("reason"), Some("deadline_expired"));
    assert!(
        metrics.source_chunks_retried >= 1,
        "cancellation must land inside the build recovery tree: {error:?} {metrics:#?}"
    );
    assert!(
        metrics.source_chunks_committed >= 1,
        "at least one build chunk must be partitioned before cancellation: {metrics:#?}"
    );
    assert!(partitioned_build_decision(&metrics, "n2").facts.contains_key("partition_count"));
    let usage = admitted
        .context()
        .attempt_allocation_domain_usage()
        .expect("allocation-domain usage samples")
        .expect("admitted context owns one allocation domain");
    assert_eq!(usage.current_bytes, 0, "cancelled build partitions must be released: {usage:?}");
    assert_eq!(usage.current_allocations, 0, "no build partition may outlive the cancelled build: {usage:?}");

    drop(admitted);
    handle.finish();
    drop(handle);
    fs::remove_dir_all(dir).unwrap();
}

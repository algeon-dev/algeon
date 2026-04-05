use super::*;

#[test]
fn bounded_native_query_executes_mixed_keyed_residual_join_with_grouped_aggregate_input() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let dir = unique_temp_dir("nexus-query-engine-bounded-mixed-aggregate-input");
    let fact_path = dir.join("fact.parquet");
    let grouped_fact_path = dir.join("grouped_fact.parquet");
    let dim_path = dir.join("dim.parquet");
    let fact_schema = write_mixed_aggregate_input_fact_parquet(&runtime, &fact_path);
    let grouped_fact_schema = write_mixed_aggregate_input_fact_row_grouped_parquet(&grouped_fact_path);
    let dim_schema = write_mixed_aggregate_input_dim_parquet(&runtime, &dim_path);
    let plan = mixed_keyed_residual_join_grouped_aggregate_input_plan(
        &fact_path,
        &grouped_fact_path,
        &dim_path,
        &fact_schema,
        &grouped_fact_schema,
        &dim_schema,
    );
    assert!(!plan.capability_analysis().admits_root_join());
    assert!(!plan.capability_analysis().admits_single_join_post_root_pipeline());
    assert!(!plan.capability_analysis().admits_keyless_conditional_join_input_subplan_pipeline());
    assert!(!plan.capability_analysis().admits_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline());
    assert!(!plan.capability_analysis().admits_closed_multi_join_subgraph());
    assert!(plan.capability_analysis().admits_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline());
    assert!(plan.bounded_join_unsupported_shape_facts(ExecutionBoundaryPolicy::default()).is_none());

    let context = mixed_grouped_aggregate_execution_context(&runtime, 8 * 1024 * 1024 * 1024);
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.memory_control_policy.as_str(), "bounded");
    assert_eq!(metrics.boundary.host_to_device_uploads, 0);
    assert_eq!(metrics.boundary.device_to_host_materializations, 1);
    assert_eq!(metrics.reservation_denied_count, 0);
    assert_eq!(metrics.compiled_pipeline_attempts, 1);
    assert!(metrics.compiled_pipeline_tasks_executed > 0);
    assert_eq!(metrics.compiled_pipeline_host_arrow_sinks, 1);
    assert_eq!(metrics.hash_join_build_count, 1);
    assert_eq!(metrics.operators.iter().filter(|operator| operator.join.is_some()).count(), 2);
    let join_metrics = metrics.operators.iter().filter_map(|operator| operator.join.as_ref()).collect::<Vec<_>>();
    assert_eq!(join_metrics[0].execution_strategy, "bounded_reusable_hash_join_single_pass_probe");
    assert_eq!(join_metrics[0].join_type, "inner");
    assert_eq!(join_metrics[1].execution_strategy, "bounded_mixed_keyed_residual_join");
    assert_eq!(join_metrics[1].join_type, "inner");
    assert!(join_metrics[1].residual_predicate);
    assert_eq!(join_metrics[1].output_rows, 2);
    assert!(metrics.reservation_events.iter().all(|event| event.memory_boundary != "join_pipeline_sort"));
    assert!(metrics.reservation_events.iter().all(|event| event.reservation_kind != ReservationKind::FinalSort));
    assert_granted_reservation_trace(
        metrics,
        &[
            (ReservationKind::SourceRead, "join_source_read"),
            (ReservationKind::SourceRead, "join_source_read"),
            (ReservationKind::BuildSide, "join_build_side"),
            (ReservationKind::HashTable, "join_hash_table"),
            (ReservationKind::ProbeSide, "join_probe_side"),
            (ReservationKind::GatherMap, "join_gather_map"),
            (ReservationKind::JoinOutput, "join_output"),
            (ReservationKind::OperatorOutput, "compiled_projection"),
            (ReservationKind::OperatorOutput, "compiled_projection_output"),
            (ReservationKind::OperatorOutput, "compiled_projection"),
            (ReservationKind::OperatorOutput, "compiled_projection_output"),
            (ReservationKind::OperatorOutput, "compiled_projection_output"),
            (ReservationKind::SourceRead, "join_source_read"),
            (ReservationKind::PartialState, "compiled_aggregate_partial_state_work"),
            (ReservationKind::PartialState, "compiled_aggregate_partial_state"),
            (ReservationKind::SourceRead, "join_source_read"),
            (ReservationKind::PartialState, "compiled_aggregate_partial_state_work"),
            (ReservationKind::PartialState, "compiled_aggregate_partial_state"),
            (ReservationKind::FinalMerge, "final_aggregate_hash_partition"),
            (ReservationKind::FinalMerge, "final_aggregate_partition"),
            (ReservationKind::FinalMerge, "final_aggregate_partition"),
            (ReservationKind::FinalMerge, "final_aggregate_hash_partition"),
            (ReservationKind::FinalMerge, "final_aggregate_partition"),
            (ReservationKind::FinalMerge, "final_aggregate_partition"),
            (ReservationKind::FinalMerge, "compiled_partitioned_final_aggregate_merge"),
            (ReservationKind::FinalMerge, "compiled_partitioned_final_aggregate_merge_output"),
            (ReservationKind::FinalMerge, "compiled_partitioned_final_aggregate_merge"),
            (ReservationKind::FinalMerge, "compiled_partitioned_final_aggregate_merge_output"),
            (ReservationKind::FinalMerge, "partitioned_final_aggregate_output_concat"),
            (ReservationKind::FinalMerge, "partitioned_final_aggregate_output_concat"),
            (ReservationKind::FinalMerge, "compiled_partitioned_final_aggregate_merge_output"),
            (ReservationKind::FinalMerge, "compiled_aggregate_final_projection"),
            (ReservationKind::OperatorOutput, "compiled_aggregate_output"),
            (ReservationKind::OperatorOutput, "compiled_projection"),
            (ReservationKind::OperatorOutput, "compiled_projection_output"),
            (ReservationKind::BuildSide, "join_build_side"),
            (ReservationKind::ProbeSide, "join_probe_side"),
            (ReservationKind::MixedSizeData, "join_mixed_size_data"),
            (ReservationKind::GatherMap, "join_gather_map"),
            (ReservationKind::JoinOutput, "join_output"),
            (ReservationKind::OperatorOutput, "compiled_projection"),
            (ReservationKind::OperatorOutput, "compiled_projection_output"),
            (ReservationKind::OperatorOutput, "compiled_aggregate"),
            (ReservationKind::OperatorOutput, "compiled_aggregate_output"),
            (ReservationKind::OperatorOutput, "compiled_projection"),
            (ReservationKind::OperatorOutput, "compiled_projection_output"),
            (ReservationKind::Materialization, "host_materialization"),
        ],
    );
    assert_budgeted_success_event_with_contract(
        metrics,
        "nexus_query_engine.aggregate.execute_partial",
        2,
        "compiled_aggregate_partial_state_work",
        Some("libcudf_groupby_aggregate_temporary_workspace_preflight_unavailable"),
    );

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(batch.num_columns(), 1);
    assert_eq!(float64_array(batch, 0).values().as_ref(), &[40.0]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_denies_mixed_keyed_residual_grouped_aggregate_source_before_root_join() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let dir = unique_temp_dir("nexus-query-engine-bounded-mixed-aggregate-input-source-denial");
    let fact_path = dir.join("fact.parquet");
    let grouped_fact_path = dir.join("grouped_fact.parquet");
    let dim_path = dir.join("dim.parquet");
    let fact_schema = write_mixed_aggregate_input_fact_parquet(&runtime, &fact_path);
    let grouped_fact_schema = write_mixed_aggregate_input_fact_row_grouped_parquet(&grouped_fact_path);
    let dim_schema = write_mixed_aggregate_input_dim_parquet(&runtime, &dim_path);
    let plan = mixed_keyed_residual_join_grouped_aggregate_input_plan(
        &fact_path,
        &grouped_fact_path,
        &dim_path,
        &fact_schema,
        &grouped_fact_schema,
        &dim_schema,
    );

    let baseline =
        execute_query_plan(&mixed_grouped_aggregate_execution_context(&runtime, 8 * 1024 * 1024 * 1024), &plan)
            .unwrap();
    let retained_join_subtree_bytes =
        retained_bytes_before_grouped_aggregate_source_read(baseline.metrics()).saturating_add(8192);
    let constrained_context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: retained_join_subtree_bytes,
            max_source_chunk_bytes: 16 * 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();
    let error = match execute_query_plan(&constrained_context, &plan) {
        Ok(_) => panic!("constrained mixed keyed residual grouped aggregate input should fail source reservation"),
        Err(error) => error,
    };

    assert_eq!(error.code(), QueryErrorCode::MemoryReservationFailed);
    assert_eq!(error.fact_value("bounded_join_execution_family"), None);
    assert_eq!(error.fact_value("memory_boundary"), Some("join_source_read"));
    assert_eq!(error.fact_value("reservation_kind"), Some("source_read"));
    assert_eq!(error.fact_value("stage"), Some("join_source_read"));
    assert_eq!(error.fact_value("source_read_contract_quality"), Some("exact_lower_level_contract"));
    assert_eq!(error.fact_value("attempt"), Some("1"));
    assert_eq!(error.fact_value("max_attempts"), Some("1"));
    assert!(error.fact_value("compiled_task_id").is_some());
    assert!(error.fact_value("plan_node_id").is_some());

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_enforces_mixed_keyed_residual_grouped_aggregate_hard_budget_before_root_join() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let dir = unique_temp_dir("nexus-query-engine-bounded-mixed-aggregate-input-hard-budget");
    let fact_path = dir.join("fact.parquet");
    let grouped_fact_path = dir.join("grouped_fact.parquet");
    let dim_path = dir.join("dim.parquet");
    let fact_schema = write_mixed_aggregate_input_fact_parquet(&runtime, &fact_path);
    let grouped_fact_schema = write_mixed_aggregate_input_fact_row_grouped_parquet(&grouped_fact_path);
    let dim_schema = write_mixed_aggregate_input_dim_parquet(&runtime, &dim_path);
    let plan = mixed_keyed_residual_join_grouped_aggregate_input_plan(
        &fact_path,
        &grouped_fact_path,
        &dim_path,
        &fact_schema,
        &grouped_fact_schema,
        &dim_schema,
    );

    let baseline =
        execute_query_plan(&mixed_grouped_aggregate_execution_context(&runtime, 8 * 1024 * 1024 * 1024), &plan)
            .unwrap();
    let (retained_join_subtree_bytes, source_read_bytes, chunk_operator_bytes, retained_partial_state_bytes) =
        grouped_aggregate_chunk_budget_window(baseline.metrics());
    let operator_token_budget = chunk_operator_bytes.saturating_sub(1);
    let constrained_budget = retained_join_subtree_bytes
        .saturating_add(source_read_bytes)
        .saturating_add(retained_partial_state_bytes)
        .saturating_add(operator_token_budget);
    let constrained_context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: constrained_budget,
            max_source_chunk_bytes: 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();
    let error = match execute_query_plan(&constrained_context, &plan) {
        Ok(_) => panic!("constrained mixed keyed residual grouped aggregate input should fail aggregate hard budget"),
        Err(error) => error,
    };
    assert_eq!(error.fact_value("bounded_join_execution_family"), None);
    match error.code() {
        QueryErrorCode::CudfResourceExhausted => {
            assert_eq!(error.fact_value("budget_boundary"), Some("compiled_aggregate"));
            let budget_bytes = error
                .fact_value("budget_bytes")
                .and_then(|value| value.parse::<u64>().ok())
                .expect("aggregate hard-budget failure should report budget bytes");
            assert!(budget_bytes >= operator_token_budget);
            assert_eq!(error.fact_value("budget_enforcement"), Some("stream_bound_reservation"));
            assert_eq!(error.fact_value("memory_boundary"), None);
            assert_eq!(error.fact_value("stage"), None);
            assert_eq!(error.fact_value("attempt"), None);
            assert_eq!(error.fact_value("max_attempts"), None);
        },
        QueryErrorCode::MemoryReservationFailed => {
            assert_eq!(error.fact_value("budget_boundary"), None);
            assert_eq!(error.fact_value("budget_bytes"), Some(constrained_budget.to_string().as_str()));
            assert_eq!(error.fact_value("budget_enforcement"), None);
            assert_eq!(error.fact_value("reservation_kind"), Some("partial_state"));
            assert_eq!(error.fact_value("memory_boundary"), Some("compiled_aggregate_partial_state_work"));
            assert_eq!(error.fact_value("stage"), Some("compiled_aggregate"));
            assert_eq!(error.fact_value("attempt"), None);
            assert_eq!(error.fact_value("max_attempts"), None);
        },
        other => panic!("unexpected grouped aggregate hard-budget error code: {other:?}"),
    }
    assert!(error.fact_value("compiled_task_id").is_some());
    assert!(error.fact_value("plan_node_id").is_some());

    fs::remove_dir_all(dir).unwrap();
}

fn mixed_grouped_aggregate_execution_context(
    runtime: &Arc<cudf::Runtime>,
    device_budget_bytes: u64,
) -> ExecutionContext {
    ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes,
            max_source_chunk_bytes: 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap()
}

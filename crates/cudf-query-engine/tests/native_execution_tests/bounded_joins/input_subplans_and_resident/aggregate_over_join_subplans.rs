use super::*;

#[test]
fn bounded_native_query_executes_keyless_conditional_join_with_aggregate_over_join_subplan_inputs() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let dir = unique_temp_dir("cudf-query-engine-bounded-aggregate-over-join-subplan-inputs");
    let grouped_fact_path = dir.join("grouped_partsupp.parquet");
    let scalar_fact_path = dir.join("scalar_partsupp.parquet");
    let supplier_path = dir.join("supplier.parquet");
    let grouped_fact_schema = write_supplier_cost_partsupp_parquet(
        &runtime,
        &grouped_fact_path,
        &[1, 1, 2, 3, 4],
        &[10, 20, 10, 30, 99],
        &[10, 5, 25, 3, 100],
    );
    let scalar_fact_schema =
        write_supplier_cost_partsupp_parquet(&runtime, &scalar_fact_path, &[8, 9], &[10, 20], &[7, 5]);
    let supplier_schema = write_supplier_cost_supplier_parquet(&runtime, &supplier_path);
    let plan = keyless_conditional_aggregate_over_join_subplan_inputs_plan(
        &grouped_fact_path,
        &scalar_fact_path,
        &supplier_path,
        &grouped_fact_schema,
        &scalar_fact_schema,
        &supplier_schema,
    );
    assert!(!plan.capability_analysis().admits_root_join());
    assert!(!plan.capability_analysis().admits_single_join_post_root_pipeline());
    assert!(!plan.capability_analysis().admits_keyless_conditional_join_input_subplan_pipeline());
    assert!(plan.capability_analysis().admits_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline());
    assert!(!plan.capability_analysis().admits_closed_multi_join_subgraph());
    assert!(plan.bounded_join_unsupported_shape_facts(ExecutionBoundaryPolicy::default()).is_none());

    let context = bounded_join_execution_context(&runtime);
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.phase_c_memory_control_policy.as_str(), "bounded");
    assert_eq!(metrics.boundary.host_to_device_uploads, 0);
    assert_eq!(metrics.boundary.device_to_host_materializations, 1);
    assert_eq!(metrics.reservation_denied_count, 0);
    assert_eq!(metrics.compiled_pipeline_attempts, 1);
    assert!(metrics.compiled_pipeline_tasks_executed > 0);
    assert_eq!(metrics.compiled_pipeline_host_arrow_sinks, 1);
    assert_eq!(metrics.hash_join_build_count, 2);
    assert_eq!(metrics.operators.iter().filter(|operator| operator.join.is_some()).count(), 3);
    let join_metrics = metrics.operators.iter().filter_map(|operator| operator.join.as_ref()).collect::<Vec<_>>();
    assert_eq!(join_metrics[0].execution_strategy, "bounded_reusable_hash_join");
    assert_eq!(join_metrics[0].join_type, "inner");
    assert_eq!(join_metrics[1].execution_strategy, "bounded_reusable_hash_join");
    assert_eq!(join_metrics[1].join_type, "inner");
    assert_eq!(join_metrics[2].execution_strategy, "bounded_keyless_conditional_join");
    assert_eq!(join_metrics[2].join_type, "inner");
    assert_eq!(join_metrics[2].key_count, 0);
    assert!(join_metrics[2].residual_predicate);
    assert_eq!(join_metrics[2].output_rows, 2);
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
            (ReservationKind::OperatorOutput, "compiled_aggregate"),
            (ReservationKind::OperatorOutput, "compiled_aggregate_output"),
            (ReservationKind::OperatorOutput, "compiled_projection"),
            (ReservationKind::OperatorOutput, "compiled_projection_output"),
            (ReservationKind::SourceRead, "join_source_read"),
            (ReservationKind::BuildSide, "join_build_side"),
            (ReservationKind::HashTable, "join_hash_table"),
            (ReservationKind::ProbeSide, "join_probe_side"),
            (ReservationKind::GatherMap, "join_gather_map"),
            (ReservationKind::JoinOutput, "join_output"),
            (ReservationKind::OperatorOutput, "compiled_aggregate"),
            (ReservationKind::OperatorOutput, "compiled_aggregate_output"),
            (ReservationKind::OperatorOutput, "compiled_projection"),
            (ReservationKind::OperatorOutput, "compiled_projection_output"),
            (ReservationKind::BuildSide, "join_build_side"),
            (ReservationKind::ProbeSide, "join_probe_side"),
            (ReservationKind::GatherMap, "join_gather_map"),
            (ReservationKind::JoinOutput, "join_output"),
            (ReservationKind::OperatorOutput, "compiled_projection"),
            (ReservationKind::OperatorOutput, "compiled_projection_output"),
            (ReservationKind::FinalSort, "final_sort"),
            (ReservationKind::FinalSort, "final_sort"),
            (ReservationKind::Materialization, "host_materialization"),
        ],
    );

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 2);
    assert_eq!(batch.num_columns(), 2);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[2, 1]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[25, 15]);

    let constrained_context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 1,
            max_source_chunk_bytes: 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();
    let error = match execute_query_plan(&constrained_context, &plan) {
        Ok(_) => panic!("constrained aggregate-over-join subplan inputs pipeline should fail reservation"),
        Err(error) => error,
    };
    assert_eq!(error.code(), QueryErrorCode::MemoryReservationFailed);
    assert_eq!(error.fact_value("bounded_join_execution_family"), None);
    assert_eq!(error.fact_value("memory_boundary"), Some("join_source_read"));
    assert_eq!(error.fact_value("reservation_kind"), Some("source_read"));
    assert!(error.fact_value("requested_bytes").is_some());
    assert_eq!(error.fact_value("granted_bytes"), Some("1"));
    assert_eq!(error.fact_value("budget_bytes"), Some("1"));
    assert!(error.fact_value("node_id").is_some());
    assert_eq!(error.fact_value("stage"), Some("join_source_read"));
    assert_eq!(error.fact_value("attempt"), Some("1"));
    assert_eq!(error.fact_value("max_attempts"), Some("1"));
    assert!(error.fact_value("compiled_task_id").is_some());
    assert!(error.fact_value("plan_node_id").is_some());

    fs::remove_dir_all(dir).unwrap();
}

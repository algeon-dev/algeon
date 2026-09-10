use std::sync::Arc;

use super::*;

#[test]
fn bounded_native_query_executes_keyless_conditional_join_with_resident_input_subplans() {
    let runtime = backend_runtime();
    let dir = unique_temp_dir("nexus-query-engine-bounded-keyless-input-subplans");
    let lineitem_path = dir.join("lineitem.parquet");
    let anti_path = dir.join("anti_orders.parquet");
    let threshold_path = dir.join("threshold.parquet");
    let lineitem_schema = write_join_left_parquet(&runtime, &lineitem_path);
    let anti_schema = write_join_right_keys_parquet(&runtime, &anti_path, &[1]);
    let threshold_schema = write_join_right_keys_parquet(&runtime, &threshold_path, &[15]);
    let left_subplan_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
    ]));
    let threshold_aggregate_schema = Arc::new(Schema::new(vec![Field::new("threshold", DataType::Int32, true)]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
        Field::new("threshold", DataType::Int32, true),
    ]));

    let plan = keyless_conditional_join_input_subplan_plan(
        &lineitem_path,
        &anti_path,
        &threshold_path,
        &lineitem_schema,
        &anti_schema,
        &threshold_schema,
        Arc::clone(&left_subplan_schema),
        Arc::clone(&threshold_aggregate_schema),
        Arc::clone(&output_schema),
        false,
    );
    assert!(!plan.capability_analysis().admits_root_join());
    assert!(!plan.capability_analysis().admits_single_join_post_root_pipeline());
    assert!(plan.capability_analysis().admits_keyless_conditional_join_input_subplan_pipeline());
    assert!(!plan.capability_analysis().admits_closed_multi_join_subgraph());

    let context = bounded_join_execution_context(&runtime);
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.memory_control_policy.as_str(), "bounded");
    assert_eq!(metrics.boundary.host_to_device_uploads, 0);
    assert_eq!(metrics.boundary.device_to_host_materializations, 1);
    assert_eq!(metrics.compiled_pipeline_attempts, 1);
    assert!(metrics.compiled_pipeline_tasks_executed > 0);
    assert_eq!(metrics.compiled_pipeline_host_arrow_sinks, 1);
    assert_eq!(metrics.filtered_join_build_count, 1);
    assert_eq!(metrics.operators.iter().filter(|operator| operator.join.is_some()).count(), 2);
    let join_metrics = metrics.operators.iter().filter_map(|operator| operator.join.as_ref()).collect::<Vec<_>>();
    assert_eq!(join_metrics[0].execution_strategy, "bounded_reusable_filtered_join");
    assert_eq!(join_metrics[0].join_type, "left_anti");
    assert_eq!(join_metrics[0].output_rows, 3);
    assert_eq!(join_metrics[1].execution_strategy, "bounded_keyless_conditional_join");
    assert_eq!(join_metrics[1].join_type, "inner");
    assert_eq!(join_metrics[1].key_count, 0);
    assert!(join_metrics[1].residual_predicate);
    assert_eq!(join_metrics[1].output_rows, 3);
    assert!(join_metrics[1].gather_map_bytes > 0);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(batch.num_columns(), 3);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[2, 2, 3]);
    assert_eq!(int32_array(batch, 1).values().as_ref(), &[20, 25, 30]);
    assert_eq!(int32_array(batch, 2).values().as_ref(), &[15, 15, 15]);

    let gpu_plan = keyless_conditional_join_input_subplan_plan(
        &lineitem_path,
        &anti_path,
        &threshold_path,
        &lineitem_schema,
        &anti_schema,
        &threshold_schema,
        left_subplan_schema,
        threshold_aggregate_schema,
        Arc::clone(&output_schema),
        true,
    );
    assert!(gpu_plan.capability_analysis().admits_keyless_conditional_join_input_subplan_pipeline());
    let gpu_result = execute_query_plan(&context, &gpu_plan).expect("GPU sink compiled keyless subplan should execute");
    let handle = gpu_result.output().as_gpu_resident().expect("GPU sink should keep output resident").clone();
    let mut resident_inputs = GpuResidentInputMap::new();
    resident_inputs.insert("compiled_output".to_owned(), handle);
    let mut decode_builder = QueryPlanBuilder::new();
    let decode_source = decode_builder
        .add_source(SourceNode::GpuResident(GpuResidentSource::new("compiled_output", Arc::clone(&output_schema))));
    let decode_projection = decode_builder.add_projection(
        decode_source,
        projection_outputs(
            vec![
                Expr::binary(Expr::column("order_id"), BinaryOp::Add, Expr::literal(LiteralValue::Int32(0))),
                Expr::binary(Expr::column("qty"), BinaryOp::Add, Expr::literal(LiteralValue::Int32(0))),
                Expr::binary(Expr::column("threshold"), BinaryOp::Add, Expr::literal(LiteralValue::Int32(0))),
            ],
            Arc::clone(&output_schema),
        ),
    );
    let decode_plan = decode_builder.finish_with_host_sink(decode_projection);
    let decode_result = execute_query_plan_with_gpu_inputs(&backend_context(), &decode_plan, &resident_inputs)
        .expect("a separately admitted consumer should decode the frozen resident output");
    assert_eq!(decode_result.metrics().boundary.device_to_device_copies, 0);
    let QueryOutput::HostArrow(gpu_batch) = decode_result.output() else {
        panic!("resident consumer should materialize one RecordBatch");
    };
    assert_eq!(gpu_batch.num_rows(), 3);
    assert_eq!(gpu_batch.num_columns(), 3);
    assert_eq!(int32_array(gpu_batch, 0).values().as_ref(), &[2, 2, 3]);
    assert_eq!(int32_array(gpu_batch, 1).values().as_ref(), &[20, 25, 30]);
    assert_eq!(int32_array(gpu_batch, 2).values().as_ref(), &[15, 15, 15]);
    assert_eq!(gpu_result.metrics().compiled_pipeline_attempts, 1);
    assert_eq!(gpu_result.metrics().compiled_pipeline_host_arrow_sinks, 0);
    drop(gpu_result);
    let constrained_context = backend_context_with_grant(1);
    let error = match execute_query_plan(&constrained_context, &plan) {
        Ok(_) => panic!("constrained keyless input-subplan pipeline should exceed its query cap"),
        Err(error) => error,
    };
    assert_eq!(error.code(), QueryErrorCode::QueryMemoryLimitExceeded, "{error:?}");
    assert_eq!(error.fact_value("bounded_join_execution_family"), None);
    assert_eq!(error.fact_value("budget_enforcement"), Some("query_domain_member"));

    fs::remove_dir_all(dir).unwrap();
}
